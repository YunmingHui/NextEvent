import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import AntiSymmetricConv, TransformerConv
from torch_geometric.nn.models.tgn import TimeEncoder as PyGTimeEncoder


class CTANMemoryBank(nn.Module):

    def __init__(self, num_nodes: int, memory_dim: int, sampler_size: int, init_time: float):
        super().__init__()
        self.num_nodes, self.memory_dim, self.sampler_size = num_nodes, memory_dim, sampler_size
        self.register_buffer('init_time', torch.tensor(float(init_time)))
        self.register_buffer('node_memories', torch.zeros(num_nodes, memory_dim))
        self.register_buffer('node_last_updated_times', torch.full((num_nodes,), float(init_time)))
        # each node's sampler_size most recent neighbors, oldest first; empty slots (edge id -1) are at the front
        self.register_buffer('neighbor_ids', torch.zeros(num_nodes, sampler_size, dtype=torch.long))
        self.register_buffer('neighbor_edge_ids', torch.full((num_nodes, sampler_size), -1, dtype=torch.long))
        self.register_buffer('neighbor_times', torch.zeros(num_nodes, sampler_size))

    def __init_memory_bank__(self):
        self.node_memories.zero_()
        self.node_last_updated_times.fill_(float(self.init_time))
        self.neighbor_ids.zero_()
        self.neighbor_edge_ids.fill_(-1)
        self.neighbor_times.zero_()

    def backup_memory_bank(self):
        return tuple(b.clone() for b in (self.node_memories, self.node_last_updated_times, self.neighbor_ids,
                                         self.neighbor_edge_ids, self.neighbor_times))

    def reload_memory_bank(self, backup_memory_bank: tuple):
        for buf, saved in zip((self.node_memories, self.node_last_updated_times, self.neighbor_ids, self.neighbor_edge_ids,
                               self.neighbor_times), backup_memory_bank):
            buf.copy_(saved)

    def detach_memory_bank(self):
        # memories are stored detached already (as in the original SimpleMemory)
        self.node_memories.detach_()

    def update(self, src: torch.Tensor, dst: torch.Tensor, times: torch.Tensor, edge_ids: torch.Tensor,
               src_emb: torch.Tensor, dst_emb: torch.Tensor):
        """write one batch of (time-ordered) edges: new memories (LastAggregator) and new recent neighbors"""
        nodes = torch.cat([src, dst])
        t2 = torch.cat([times, times])
        emb = torch.cat([src_emb, dst_emb]).detach()
        # each node's latest occurrence in the batch (ties in time: later position wins, dst half after src half)
        order = torch.argsort(t2, stable=True)
        last_pos = torch.full((self.num_nodes,), -1, dtype=torch.long, device=nodes.device).scatter_reduce(
            0, nodes[order], torch.arange(len(order), device=nodes.device), reduce='amax')
        touched = torch.nonzero(last_pos >= 0).squeeze(1)
        chosen = order[last_pos[touched]]
        self.node_memories[touched] = emb[chosen]
        self.node_last_updated_times[touched] = t2[chosen]

        # recent neighbors: both directions of every edge, appended in batch order, keep the last sampler_size per node
        nbr_of = torch.cat([dst, src])
        e2 = torch.cat([edge_ids, edge_ids])
        pos = torch.cat([torch.arange(len(src), device=src.device)] * 2)
        key = nodes * (len(src) + 1) + pos  # sort by node, then by position in the batch (= time order)
        perm = torch.argsort(key)
        n_sorted = nodes[perm]
        uniq, counts = torch.unique_consecutive(n_sorted, return_counts=True)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(len(n_sorted), device=src.device) - torch.repeat_interleave(starts, counts)
        row = torch.repeat_interleave(torch.arange(len(uniq), device=src.device), counts)
        max_new = int(counts.max())
        new_ids = torch.zeros(len(uniq), max_new, dtype=torch.long, device=src.device)
        new_eids = torch.full((len(uniq), max_new), -1, dtype=torch.long, device=src.device)
        new_times = torch.zeros(len(uniq), max_new, device=src.device)
        new_ids[row, rank] = nbr_of[perm]
        new_eids[row, rank] = e2[perm]
        new_times[row, rank] = t2[perm].float()
        all_ids = torch.cat([self.neighbor_ids[uniq], new_ids], dim=1)
        all_eids = torch.cat([self.neighbor_edge_ids[uniq], new_eids], dim=1)
        all_times = torch.cat([self.neighbor_times[uniq], new_times], dim=1)
        # move empty slots to the front, keeping the chronological order of the filled ones, then keep the last K
        keep = torch.argsort((all_eids >= 0).int(), dim=1, stable=True)[:, -self.sampler_size:]
        self.neighbor_ids[uniq] = all_ids.gather(1, keep)
        self.neighbor_edge_ids[uniq] = all_eids.gather(1, keep)
        self.neighbor_times[uniq] = all_times.gather(1, keep)


class CTAN(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, memory_dim: int = None, time_dim: int = 16,
                 num_iters: int = 1, epsilon: float = 0.1, gamma: float = 0.1, sampler_size: int = 5, act: str = 'tanh',
                 final_act: str = None, mean_delta_t: float = 0., std_delta_t: float = 1., init_time: float = 0., device: str = 'cpu'):
        """
        :param node_raw_features: ndarray, shape (num_nodes, node_feat_dim) (only its size is used, see module docstring)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim), the edge features used as messages
        :param memory_dim: int, memory / embedding dimension (default: node_feat_dim)
        :param time_dim: int, dimension of the time encoding of the edge attributes
        :param num_iters: int, AntiSymmetricConv iterations (= hops of the recent neighborhood used)
        :param epsilon: float, AntiSymmetricConv step size
        :param gamma: float, AntiSymmetricConv diffusion strength
        :param sampler_size: int, number of most recent neighbors kept per node
        :param mean_delta_t: float, std_delta_t: float, normalization of |neighbor's last update - edge time|
        :param init_time: float, initial last update time of every node
        """
        super().__init__()
        self.device = device
        self.num_nodes = node_raw_features.shape[0]
        self.memory_dim = memory_dim or node_raw_features.shape[1]
        self.num_iters = num_iters
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.register_buffer('mean_delta_t', torch.tensor(float(mean_delta_t)))
        self.register_buffer('std_delta_t', torch.tensor(float(std_delta_t)))

        self.memory_bank = CTANMemoryBank(self.num_nodes, self.memory_dim, sampler_size, init_time)
        # CTANEmbedding
        self.enc_x = nn.Linear(self.memory_dim, self.memory_dim)
        phi = TransformerConv(self.memory_dim, self.memory_dim, edge_dim=self.edge_raw_features.shape[1] + time_dim, root_weight=False)
        self.aconv = AntiSymmetricConv(self.memory_dim, phi, num_iters, epsilon, gamma, act=act)
        self.time_enc = PyGTimeEncoder(time_dim)
        self.final_act = getattr(torch, final_act) if final_act is not None else None

    def set_neighbor_sampler(self, neighbor_sampler):
        # CTAN keeps its own recent-neighbor lists in the memory bank and does not use DyGLib's neighbor sampler
        pass

    def _neighborhood(self, n_id: torch.Tensor):
        """PyG LastNeighborLoader called num_iters times: returns the expanded node set and the edges into the nodes of the
        last-but-one expansion, as (edge_index over local ids [neighbor -> node], edge ids, edge times)"""
        mb = self.memory_bank
        for _ in range(self.num_iters):
            nbrs, eids, ts = mb.neighbor_ids[n_id], mb.neighbor_edge_ids[n_id], mb.neighbor_times[n_id]
            nodes = n_id.view(-1, 1).expand_as(nbrs)
            mask = eids >= 0
            nbrs, nodes, eids, ts = nbrs[mask], nodes[mask], eids[mask], ts[mask]
            n_id = torch.cat([n_id, nbrs]).unique()
        assoc = torch.empty(self.num_nodes, dtype=torch.long, device=n_id.device)
        assoc[n_id] = torch.arange(len(n_id), device=n_id.device)
        return n_id, assoc, torch.stack([assoc[nbrs], assoc[nodes]]), eids, ts

    def compute_node_embeddings(self, node_ids: torch.Tensor):
        """embeddings of node_ids (LongTensor on device) from the current state"""
        n_id, assoc, edge_index, eids, ts = self._neighborhood(node_ids.unique())
        mb = self.memory_bank
        z = mb.node_memories[n_id]
        rel_t = (mb.node_last_updated_times[n_id][edge_index[0]] - ts).abs()
        rel_t = ((rel_t - self.mean_delta_t) / self.std_delta_t).to(z.dtype)
        edge_attr = torch.cat([self.edge_raw_features[eids], self.time_enc(rel_t)], dim=-1)
        z = self.aconv(self.enc_x(z), edge_index, edge_attr=edge_attr)
        if self.final_act is not None:
            z = self.final_act(z)
        return z[assoc[node_ids]]

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
                                                 edge_ids: np.ndarray = None, edges_are_positive: bool = True, num_neighbors: int = None):
        """
        :param edges_are_positive: bool, whether to write this batch (as real edges) into the state after embedding it
        :param num_neighbors: unused (CTAN's neighborhood size is sampler_size), kept for DyGLib call compatibility
        """
        src = torch.from_numpy(np.asarray(src_node_ids)).long().to(self.device)
        dst = torch.from_numpy(np.asarray(dst_node_ids)).long().to(self.device)
        z = self.compute_node_embeddings(torch.cat([src, dst]))
        src_emb, dst_emb = z[:len(src)], z[len(src):]
        if edges_are_positive:
            assert edge_ids is not None
            self.memory_bank.update(src, dst, torch.from_numpy(np.asarray(node_interact_times)).float().to(self.device),
                                    torch.from_numpy(np.asarray(edge_ids)).long().to(self.device), src_emb, dst_emb)
        return src_emb, dst_emb
