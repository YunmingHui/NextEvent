import math
import os
import random
import torch
import torch.nn as nn
import numpy as np

from utils.DataLoader import Data

# which NeighborSampler.get_historical_neighbors to use for the 'recent' and 'uniform' strategies: 'loop' (the original per-node
# Python loop) or 'vectorized' (the whole batch at once, same neighbors and the same random draws). 'time_interval_aware' always
# runs the loop
NEIGHBOR_SAMPLER_IMPL = os.environ.get('DYG_NEIGHBOR_SAMPLER_IMPL', 'vectorized')


def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_gpu(*data, device: str):
    """
    convert data from cpu to gpu, accelerate the running speed
    :param data: can be any type, including Tensor, Module, ...
    :param device: str
    """
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    if len(res) > 1:
        res = tuple(res)
    else:
        res = res[0]
    return res


def get_parameter_sizes(model: nn.Module):
    """
    get parameter size of trainable parameters in model
    :param model: nn.Module
    :return:
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def create_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.0):
    """
    create optimizer
    :param model: nn.Module
    :param optimizer_name: str, optimizer name
    :param learning_rate: float, learning rate
    :param weight_decay: float, weight decay
    :return:
    """
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Wrong value for optimizer {optimizer_name}!")

    return optimizer


def log_normal_mixture_log_prob(x: torch.Tensor, log_weights: torch.Tensor, means: torch.Tensor, log_stds: torch.Tensor):
    """
    log-density of a K-component log-normal mixture at x (x must be > 0), i.e. log(sum_k pi_k * LogNormal(x; mu_k, sigma_k))
    :param x: Tensor, shape (num_samples, ), must be strictly positive
    :param log_weights: Tensor, shape (num_samples, num_mixtures), log(pi_k), already normalized (sums to 1 over the mixture dim)
    :param means: Tensor, shape (num_samples, num_mixtures), mu_k in log-seconds space
    :param log_stds: Tensor, shape (num_samples, num_mixtures), log(sigma_k)
    :return: Tensor, shape (num_samples, )
    """
    log_x = torch.log(x).unsqueeze(dim=-1)
    sigma = log_stds.exp()
    # standard log-normal log-density, per component
    component_log_prob = -log_x - log_stds - 0.5 * math.log(2 * math.pi) - 0.5 * ((log_x - means) / sigma) ** 2
    return torch.logsumexp(log_weights + component_log_prob, dim=-1)


def log_normal_mixture_log_survival(x: torch.Tensor, log_weights: torch.Tensor, means: torch.Tensor, log_stds: torch.Tensor):
    """
    log survival probability log P(T > x) of a K-component log-normal mixture at x (x must be > 0),
    used for the likelihood contribution of right-censored samples (we only know T > x, not its exact value)
    :param x: Tensor, shape (num_samples, ), must be strictly positive
    :param log_weights: Tensor, shape (num_samples, num_mixtures)
    :param means: Tensor, shape (num_samples, num_mixtures)
    :param log_stds: Tensor, shape (num_samples, num_mixtures)
    :return: Tensor, shape (num_samples, )
    """
    log_x = torch.log(x).unsqueeze(dim=-1)
    sigma = log_stds.exp()
    # log P(T_k > x) = log Phi(-(log(x) - mu_k) / sigma_k), via the numerically stable log of the normal CDF
    component_log_survival = torch.special.log_ndtr(-(log_x - means) / sigma)
    return torch.logsumexp(log_weights + component_log_survival, dim=-1)


def get_node_activity_tpp_loss(log_weights: torch.Tensor, means: torch.Tensor, log_stds: torch.Tensor,
                               targets: torch.Tensor, is_censored: torch.Tensor):
    """
    true negative log-likelihood loss for the node activity task under a LogNormMix (log-normal mixture)
    temporal point process formulation: uncensored samples contribute their exact log-density, censored
    samples contribute the log-survival probability at their known lower bound - both are just two cases
    of the same likelihood, averaged with EQUAL WEIGHT PER SAMPLE (not per group) so this is the actual
    MLE objective - grouping into two subgroup means and adding them (as an earlier version of this
    function did, with a "censored_weight" knob defaulting to 1.0) silently gives the censored group the
    same total weight as the uncensored group regardless of how many samples are in each, which on this
    task's typical censored ratio (a few percent) over-weights every censored sample's gradient by roughly
    one to two orders of magnitude relative to true MLE
    :param log_weights: Tensor, shape (num_samples, num_mixtures)
    :param means: Tensor, shape (num_samples, num_mixtures)
    :param log_stds: Tensor, shape (num_samples, num_mixtures)
    :param targets: Tensor, shape (num_samples, ), true gap (uncensored) or known lower bound (censored), raw seconds
    :param is_censored: Tensor of bool, shape (num_samples, )
    :return:
        loss (Tensor scalar, the true per-sample-averaged NLL), uncensored_nll (Tensor scalar, mean NLL within
        the uncensored subgroup, for logging only), censored_nll (same, censored subgroup, for logging only)
    """
    # avoid log(0) for a same-timestamp (zero-gap) observation
    safe_targets = targets.clamp(min=1.0)
    log_prob = log_normal_mixture_log_prob(safe_targets, log_weights, means, log_stds)
    log_survival = log_normal_mixture_log_survival(safe_targets, log_weights, means, log_stds)
    nll_per_sample = torch.where(is_censored, -log_survival, -log_prob)

    loss = nll_per_sample.mean()

    uncensored_mask = ~is_censored
    uncensored_nll = nll_per_sample[uncensored_mask].mean() if uncensored_mask.any() else torch.zeros((), device=targets.device)
    censored_nll = nll_per_sample[is_censored].mean() if is_censored.any() else torch.zeros((), device=targets.device)

    return loss, uncensored_nll, censored_nll


def log_normal_mixture_median(log_weights: torch.Tensor, means: torch.Tensor, log_stds: torch.Tensor, num_iters: int = 60):
    """
    median of a K-component log-normal mixture, found by bisection on log(x) (the mixture survival
    function is monotonically decreasing in x, so bisecting for survival(x) == 0.5 is well-posed; there
    is no closed form for a mixture's median. This is robust to a single component having a large sigma -
    one heavy-tailed component barely moves the median (it only shifts probability mass far out in the
    tail), unlike the mixture's mean which such a component can dominate via exp(mu + sigma^2/2)
    :param log_weights: Tensor, shape (num_samples, num_mixtures)
    :param means: Tensor, shape (num_samples, num_mixtures)
    :param log_stds: Tensor, shape (num_samples, num_mixtures)
    :param num_iters: int, number of bisection iterations (60 is far more than enough for float32 precision)
    :return: Tensor, shape (num_samples, )
    """
    log_half = math.log(0.5)
    lo = torch.full_like(means[..., 0], -15.0)
    hi = torch.full_like(means[..., 0], 45.0)
    for _ in range(num_iters):
        mid = (lo + hi) / 2
        log_surv = log_normal_mixture_log_survival(torch.exp(mid), log_weights, means, log_stds)
        # survival(mid) > 0.5 means the median lies further right (mid underestimates it)
        still_high = log_surv > log_half
        lo = torch.where(still_high, mid, lo)
        hi = torch.where(still_high, hi, mid)
    return torch.exp((lo + hi) / 2)


def count_node_history(node_occurrence_times: dict, query_node_ids: np.ndarray, query_times: np.ndarray):
    """
    for each (node, time) query, how many of that node's OWN occurrences happened strictly before it -
    the same quantity that determines both the RNN context's sequence length and how many historical
    neighbors the GNN backbone has to attend over. Used to stratify evaluation by history depth, since a
    handful of near-zero-history ("cold start") nodes are a real, reportable phenomenon in this task
    rather than noise to average away (see build_node_history_sequences for how the dict is built).
    :param node_occurrence_times: dict[int, np.ndarray], from build_node_history_sequences
    :param query_node_ids: ndarray, shape (batch_size, )
    :param query_times: ndarray, shape (batch_size, )
    :return: ndarray of int, shape (batch_size, )
    """
    return np.array([int(np.searchsorted(node_occurrence_times[int(n)], t, side='left'))
                     for n, t in zip(query_node_ids, query_times)])


def build_rnn_context(encoder: nn.Module, node_gap_sequences: dict, node_occurrence_times: dict,
                      query_node_ids: np.ndarray, query_times: np.ndarray, device: str, max_history: int = None):
    """
    builds the LogNormMix-style RNN context vector for a batch of (node, time) queries, using
    NodeHistoryRNNEncoder over each queried node's own occurrence history (see build_node_history_sequences).
    For every query, the GRU reads (from its initial state) the node's gaps up to this occurrence - only the last
    max_history of them if set, which bounds memory on nodes with very long histories (reddit) - and its last
    output is the context. A node's first-ever occurrence (no gap yet) gets the learned initial context.
    With max_history >= a node's number of past gaps this is exactly the GRU over that node's full history.
    :param encoder: NodeHistoryRNNEncoder
    :param node_gap_sequences: dict[int, np.ndarray], from build_node_history_sequences
    :param node_occurrence_times: dict[int, np.ndarray], from build_node_history_sequences
    :param query_node_ids: ndarray, shape (batch_size, )
    :param query_times: ndarray, shape (batch_size, )
    :param device: str
    :param max_history: int, the most recent gaps the GRU reads per query; None reads the full history
    :return: Tensor, shape (batch_size, encoder.hidden_dim)
    """
    # number of earlier gaps of each query = position of this occurrence within the node's own chronological sequence
    # (the gap ending at this occurrence is the last one; occurrences at the same timestamp are not counted)
    positions = np.array([int(np.searchsorted(node_occurrence_times[int(node_id)], t, side='left'))
                          for node_id, t in zip(query_node_ids, query_times)], dtype=np.int64)
    lengths = positions if max_history is None else np.minimum(positions, max_history)
    contexts = encoder.initial_context.unsqueeze(dim=0).expand(len(query_node_ids), -1)

    rows = np.flatnonzero(lengths > 0)
    if len(rows) == 0:
        # every queried node is on its very first-ever occurrence: no gap history exists for any of them
        return contexts

    padded = np.zeros((len(rows), int(lengths[rows].max()), 1), dtype=np.float32)
    for k, row in enumerate(rows):
        position, length = positions[row], lengths[row]
        padded[k, :length, 0] = node_gap_sequences[int(query_node_ids[row])][position - length:position]

    # Tensor, shape (num_rows, max_length, hidden_dim); each sequence is left-aligned, so its last real step is length - 1
    outputs = encoder(torch.from_numpy(padded).to(device))
    contexts = contexts.clone()
    contexts[torch.from_numpy(rows).to(device)] = outputs[torch.arange(len(rows), device=device),
                                                          torch.from_numpy(lengths[rows] - 1).to(device)]
    return contexts


class NeighborSampler:

    def __init__(self, adj_list: list, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed: int = None):
        """
        Neighbor sampler.
        :param adj_list: list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
        :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware'
        :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
        a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
        :param seed: int, random seed
        """
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed

        # list of each node's neighbor ids, edge ids and interaction times, which are sorted by interaction times
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        if self.sample_neighbor_strategy == 'time_interval_aware':
            self.nodes_neighbor_sampled_probabilities = []
            self.time_scaling_factor = time_scaling_factor

        # the list at the first position in adj_list is empty, hence, sorted() will return an empty list for the first position
        # its corresponding value in self.nodes_neighbor_ids, self.nodes_edge_ids, self.nodes_neighbor_times will also be empty with length 0
        for node_idx, per_node_neighbors in enumerate(adj_list):
            # per_node_neighbors is a list of tuples (neighbor_id, edge_id, timestamp)
            # sort the list based on timestamps, sorted() function is stable
            # Note that sort the list based on edge id is also correct, as the original data file ensures the interactions are chronological
            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_per_node_neighbors]))
            self.nodes_edge_ids.append(np.array([x[1] for x in sorted_per_node_neighbors]))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_per_node_neighbors]))

            # additional for time interval aware sampling strategy (proposed in CAWN paper)
            if self.sample_neighbor_strategy == 'time_interval_aware':
                self.nodes_neighbor_sampled_probabilities.append(self.compute_sampled_probabilities(np.array([x[2] for x in sorted_per_node_neighbors])))

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

        # the same neighbors in CSR form for the vectorized get_historical_neighbors: node n's neighbors are positions
        # nodes_offsets[n]:nodes_offsets[n + 1] of the flat arrays, sorted by time; flat_keys orders them by (node, time rank)
        # so one searchsorted finds, for every queried node, how many of its neighbors are before the query time
        self.neighbor_sampler_impl = NEIGHBOR_SAMPLER_IMPL
        num_neighbors_per_node = np.array([len(times) for times in self.nodes_neighbor_times], dtype=np.int64)
        self.nodes_offsets = np.concatenate([[0], np.cumsum(num_neighbors_per_node)])
        non_empty = [n for n in range(len(self.nodes_neighbor_times)) if num_neighbors_per_node[n] > 0]
        if non_empty:
            self.flat_neighbor_ids = np.concatenate([self.nodes_neighbor_ids[n] for n in non_empty]).astype(np.longlong)
            self.flat_edge_ids = np.concatenate([self.nodes_edge_ids[n] for n in non_empty]).astype(np.longlong)
            self.flat_neighbor_times = np.concatenate([self.nodes_neighbor_times[n] for n in non_empty])
        else:
            self.flat_neighbor_ids, self.flat_edge_ids = np.zeros(0, dtype=np.longlong), np.zeros(0, dtype=np.longlong)
            self.flat_neighbor_times = np.zeros(0)
        self.unique_neighbor_times = np.unique(self.flat_neighbor_times)
        self.time_rank_base = len(self.unique_neighbor_times) + 1
        self.flat_keys = np.repeat(np.arange(len(num_neighbors_per_node), dtype=np.int64), num_neighbors_per_node) * self.time_rank_base \
            + np.searchsorted(self.unique_neighbor_times, self.flat_neighbor_times)

    def query_times(self, node_interact_times: np.ndarray):
        """
        the times the neighbors are looked up at (subclasses may restrict them)
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        return node_interact_times

    def count_neighbors_before(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        vectorized find_neighbors_before: for each node, the number of its neighbors with time strictly before its query time
        (the i of find_neighbors_before) and the position of its first neighbor in the flat arrays
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, ), already passed through query_times
        :return: two ndarrays, shape (batch_size, )
        """
        # neighbor time < t  <=>  rank of the neighbor time < number of distinct neighbor times < t
        query_ranks = np.searchsorted(self.unique_neighbor_times, node_interact_times, side='left')
        positions = np.searchsorted(self.flat_keys, node_ids.astype(np.int64) * self.time_rank_base + query_ranks, side='left')
        starts = self.nodes_offsets[node_ids]
        return positions - starts, starts

    def get_historical_neighbors_vectorized(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        get_historical_neighbors for the whole batch at once ('recent' and 'uniform' strategies), with the same output: for
        'uniform' the rows draw from the random state in the same order and amounts as the loop does
        """
        node_ids = np.asarray(node_ids).astype(np.int64)
        counts, starts = self.count_neighbors_before(node_ids, self.query_times(np.asarray(node_interact_times)))
        nodes_neighbor_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        nodes_edge_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        nodes_neighbor_times = np.zeros((len(node_ids), num_neighbors)).astype(np.float32)

        if self.sample_neighbor_strategy == 'recent':
            # the most recent min(count, num_neighbors) neighbors, at the back positions
            offsets = counts[:, np.newaxis] - num_neighbors + np.arange(num_neighbors)
            valid = offsets >= 0
            flat_positions = (starts[:, np.newaxis] + offsets)[valid]
            nodes_neighbor_ids[valid] = self.flat_neighbor_ids[flat_positions]
            nodes_edge_ids[valid] = self.flat_edge_ids[flat_positions]
            nodes_neighbor_times[valid] = self.flat_neighbor_times[flat_positions]
        elif self.sample_neighbor_strategy == 'uniform':
            rows = np.flatnonzero(counts > 0)
            if len(rows) > 0:
                # choice(a=count, size=num_neighbors) is randint(0, count, size=num_neighbors); drawing all rows in one call
                # takes the same numbers from the random state in the same (row-major) order
                random_state = self.random_state if self.seed is not None else np.random.mtrand._rand
                sampled_indices = random_state.randint(0, counts[rows][:, np.newaxis], size=(len(rows), num_neighbors))
                flat_positions = starts[rows][:, np.newaxis] + sampled_indices
                sampled_ids = self.flat_neighbor_ids[flat_positions]
                sampled_edge_ids = self.flat_edge_ids[flat_positions]
                sampled_times = self.flat_neighbor_times[flat_positions].astype(np.float32)
                # resort by (float32) time, as the loop does
                sorted_positions = sampled_times.argsort(axis=1)
                nodes_neighbor_ids[rows] = np.take_along_axis(sampled_ids, sorted_positions, axis=1)
                nodes_edge_ids[rows] = np.take_along_axis(sampled_edge_ids, sorted_positions, axis=1)
                nodes_neighbor_times[rows] = np.take_along_axis(sampled_times, sorted_positions, axis=1)
        else:
            raise ValueError(f'Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!')
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        """
        compute the sampled probabilities of historical neighbors based on their interaction times
        :param node_neighbor_times: ndarray, shape (num_historical_neighbors, )
        :return:
        """
        if len(node_neighbor_times) == 0:
            return np.array([])
        # compute the time delta with regard to the last time in node_neighbor_times
        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        # compute the normalized sampled probabilities of historical neighbors
        exp_node_neighbor_times = np.exp(self.time_scaling_factor * node_neighbor_times)
        sampled_probabilities = exp_node_neighbor_times / np.cumsum(exp_node_neighbor_times)
        # note that the first few values in exp_node_neighbor_times may be all zero, which make the corresponding values in sampled_probabilities
        # become nan (divided by zero), so we replace the nan by a very large negative number -1e10 to denote the sampled probabilities
        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10
        return sampled_probabilities

    def find_neighbors_before(self, node_id: int, interact_time: float, return_sampled_probabilities: bool = False):
        """
        extracts all the interactions happening before interact_time (less than interact_time) for node_id in the overall interaction graph
        the returned interactions are sorted by time.
        :param node_id: int, node id
        :param interact_time: float, interaction time
        :param return_sampled_probabilities: boolean, whether return the sampled probabilities of neighbors
        :return: neighbors, edge_ids, timestamps and sampled_probabilities (if return_sampled_probabilities is True) with shape (historical_nodes_num, )
        """
        # return index i, which satisfies list[i - 1] < v <= list[i]
        # return 0 for the first position in self.nodes_neighbor_times since the value at the first position is empty
        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)

        if return_sampled_probabilities:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i], \
                   self.nodes_neighbor_sampled_probabilities[node_id][:i]
        else:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i], None

    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids with interactions before the corresponding time in node_interact_times
        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_neighbors > 0, 'Number of sampled neighbors for each node should be greater than 0!'
        if self.neighbor_sampler_impl == 'vectorized' and self.sample_neighbor_strategy in ['recent', 'uniform']:
            return self.get_historical_neighbors_vectorized(node_ids=node_ids, node_interact_times=node_interact_times, num_neighbors=num_neighbors)
        # All interactions described in the following three matrices are sorted in each row by time
        # each entry in position (i,j) represents the id of the j-th dst node of src node node_ids[i] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the id of the edge with src node node_ids[i] and dst node nodes_neighbor_ids[i][j] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_edge_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the interaction time between src node node_ids[i] and dst node nodes_neighbor_ids[i][j], before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_times = np.zeros((len(node_ids), num_neighbors)).astype(np.float32)

        # extracts all neighbors ids, edge ids and interaction times of nodes in node_ids, which happened before the corresponding time in node_interact_times
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, node_neighbor_sampled_probabilities = \
                self.find_neighbors_before(node_id=node_id, interact_time=node_interact_time, return_sampled_probabilities=self.sample_neighbor_strategy == 'time_interval_aware')

            if len(node_neighbor_ids) > 0:
                if self.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                    # when self.sample_neighbor_strategy == 'uniform', we shuffle the data before sampling with node_neighbor_sampled_probabilities as None
                    # when self.sample_neighbor_strategy == 'time_interval_aware', we sample neighbors based on node_neighbor_sampled_probabilities
                    # for time_interval_aware sampling strategy, we additionally use softmax to make the sum of sampled probabilities be 1
                    if node_neighbor_sampled_probabilities is not None:
                        # for extreme case that node_neighbor_sampled_probabilities only contains -1e10, which will make the denominator of softmax be zero,
                        # torch.softmax() function can tackle this case
                        node_neighbor_sampled_probabilities = torch.softmax(torch.from_numpy(node_neighbor_sampled_probabilities).float(), dim=0).numpy()
                    if self.seed is None:
                        sampled_indices = np.random.choice(a=len(node_neighbor_ids), size=num_neighbors, p=node_neighbor_sampled_probabilities)
                    else:
                        sampled_indices = self.random_state.choice(a=len(node_neighbor_ids), size=num_neighbors, p=node_neighbor_sampled_probabilities)

                    nodes_neighbor_ids[idx, :] = node_neighbor_ids[sampled_indices]
                    nodes_edge_ids[idx, :] = node_edge_ids[sampled_indices]
                    nodes_neighbor_times[idx, :] = node_neighbor_times[sampled_indices]

                    # resort based on timestamps, return the ids in sorted increasing order, note this maybe unstable when multiple edges happen at the same time
                    # (we still do this though this is unnecessary for TGAT or CAWN to guarantee the order of nodes,
                    # since TGAT computes in an order-agnostic manner with relative time encoding, and CAWN computes for each walk while the sampled nodes are in different walks)
                    sorted_position = nodes_neighbor_times[idx, :].argsort()
                    nodes_neighbor_ids[idx, :] = nodes_neighbor_ids[idx, :][sorted_position]
                    nodes_edge_ids[idx, :] = nodes_edge_ids[idx, :][sorted_position]
                    nodes_neighbor_times[idx, :] = nodes_neighbor_times[idx, :][sorted_position]
                elif self.sample_neighbor_strategy == 'recent':
                    # Take most recent interactions with number num_neighbors
                    node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                    node_edge_ids = node_edge_ids[-num_neighbors:]
                    node_neighbor_times = node_neighbor_times[-num_neighbors:]

                    # put the neighbors' information at the back positions
                    nodes_neighbor_ids[idx, num_neighbors - len(node_neighbor_ids):] = node_neighbor_ids
                    nodes_edge_ids[idx, num_neighbors - len(node_edge_ids):] = node_edge_ids
                    nodes_neighbor_times[idx, num_neighbors - len(node_neighbor_times):] = node_neighbor_times
                else:
                    raise ValueError(f'Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!')

        # three ndarrays, with shape (batch_size, num_neighbors)
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_multi_hop_neighbors(self, num_hops: int, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids within num_hops hops
        :param num_hops: int, number of sampled hops
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_hops > 0, 'Number of sampled hops should be greater than 0!'

        # get the temporal neighbors at the first hop
        # nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times -> ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=node_ids,
                                                                                                 node_interact_times=node_interact_times,
                                                                                                 num_neighbors=num_neighbors)
        # three lists to store the neighbor ids, edge ids and interaction timestamp information
        nodes_neighbor_ids_list = [nodes_neighbor_ids]
        nodes_edge_ids_list = [nodes_edge_ids]
        nodes_neighbor_times_list = [nodes_neighbor_times]
        for hop in range(1, num_hops):
            # get information of neighbors sampled at the current hop
            # three ndarrays, with shape (batch_size * num_neighbors ** hop, num_neighbors)
            nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=nodes_neighbor_ids_list[-1].flatten(),
                                                                                                     node_interact_times=nodes_neighbor_times_list[-1].flatten(),
                                                                                                     num_neighbors=num_neighbors)
            # three ndarrays with shape (batch_size, num_neighbors ** (hop + 1))
            nodes_neighbor_ids = nodes_neighbor_ids.reshape(len(node_ids), -1)
            nodes_edge_ids = nodes_edge_ids.reshape(len(node_ids), -1)
            nodes_neighbor_times = nodes_neighbor_times.reshape(len(node_ids), -1)

            nodes_neighbor_ids_list.append(nodes_neighbor_ids)
            nodes_edge_ids_list.append(nodes_edge_ids)
            nodes_neighbor_times_list.append(nodes_neighbor_times)

        # tuple, each element in the tuple is a list of num_hops ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        get historical neighbors of nodes in node_ids at the first hop with max_num_neighbors as the maximal number of neighbors (make the computation feasible)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :return:
        """
        # three lists to store the first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list = [], [], []
        # get the temporal neighbors at the first hop
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, _ = self.find_neighbors_before(node_id=node_id,
                                                                                                  interact_time=node_interact_time,
                                                                                                  return_sampled_probabilities=False)
            nodes_neighbor_ids_list.append(node_neighbor_ids)
            nodes_edge_ids_list.append(node_edge_ids)
            nodes_neighbor_times_list.append(node_neighbor_times)

        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_sampler(data: Data, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed: int = None):
    """
    get neighbor sampler
    :param data: Data
    :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware''
    :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
    a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
    :param seed: int, random seed
    :return:
    """
    max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())
    # the adjacency vector stores edges for each node (source or destination), undirected
    # adj_list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
    # the list at the first position in adj_list is empty
    adj_list = [[] for _ in range(max_node_id + 1)]
    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(data.src_node_ids, data.dst_node_ids, data.edge_ids, data.node_interact_times):
        adj_list[src_node_id].append((dst_node_id, edge_id, node_interact_time))
        adj_list[dst_node_id].append((src_node_id, edge_id, node_interact_time))

    return NeighborSampler(adj_list=adj_list, sample_neighbor_strategy=sample_neighbor_strategy, time_scaling_factor=time_scaling_factor, seed=seed)


class NegativeEdgeSampler(object):

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, interact_times: np.ndarray = None, last_observed_time: float = None,
                 negative_sample_strategy: str = 'random', seed: int = None):
        """
        Negative Edge Sampler, which supports three strategies: "random", "historical", "inductive".
        :param src_node_ids: ndarray, (num_src_nodes, ), source node ids, num_src_nodes == num_dst_nodes
        :param dst_node_ids: ndarray, (num_dst_nodes, ), destination node ids
        :param interact_times: ndarray, (num_src_nodes, ), interaction timestamps
        :param last_observed_time: float, time of the last observation (for inductive negative sampling strategy)
        :param negative_sample_strategy: str, negative sampling strategy, can be "random", "historical", "inductive"
        :param seed: int, random seed
        """
        self.seed = seed
        self.negative_sample_strategy = negative_sample_strategy
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.interact_times = interact_times
        self.unique_src_node_ids = np.unique(src_node_ids)
        self.unique_dst_node_ids = np.unique(dst_node_ids)
        self.unique_interact_times = np.unique(interact_times)
        self.earliest_time = min(self.unique_interact_times)
        self.last_observed_time = last_observed_time

        if self.negative_sample_strategy != 'random':
            # all the possible edges that connect source nodes in self.unique_src_node_ids with destination nodes in self.unique_dst_node_ids
            self.possible_edges = set((src_node_id, dst_node_id) for src_node_id in self.unique_src_node_ids for dst_node_id in self.unique_dst_node_ids)

        if self.negative_sample_strategy == 'inductive':
            # set of observed edges
            self.observed_edges = self.get_unique_edges_between_start_end_time(self.earliest_time, self.last_observed_time)

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def get_unique_edges_between_start_end_time(self, start_time: float, end_time: float):
        """
        get unique edges happened between start and end time
        :param start_time: float, start timestamp
        :param end_time: float, end timestamp
        :return: a set of edges, where each edge is a tuple of (src_node_id, dst_node_id)
        """
        selected_time_interval = np.logical_and(self.interact_times >= start_time, self.interact_times <= end_time)
        # return the unique select source and destination nodes in the selected time interval
        return set((src_node_id, dst_node_id) for src_node_id, dst_node_id in zip(self.src_node_ids[selected_time_interval], self.dst_node_ids[selected_time_interval]))

    def sample(self, size: int, batch_src_node_ids: np.ndarray = None, batch_dst_node_ids: np.ndarray = None,
               current_batch_start_time: float = 0.0, current_batch_end_time: float = 0.0):
        """
        sample negative edges, support random, historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        if self.negative_sample_strategy == 'random':
            negative_src_node_ids, negative_dst_node_ids = self.random_sample(size=size)
        elif self.negative_sample_strategy == 'historical':
            negative_src_node_ids, negative_dst_node_ids = self.historical_sample(size=size, batch_src_node_ids=batch_src_node_ids,
                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                  current_batch_start_time=current_batch_start_time,
                                                                                  current_batch_end_time=current_batch_end_time)
        elif self.negative_sample_strategy == 'inductive':
            negative_src_node_ids, negative_dst_node_ids = self.inductive_sample(size=size, batch_src_node_ids=batch_src_node_ids,
                                                                                 batch_dst_node_ids=batch_dst_node_ids,
                                                                                 current_batch_start_time=current_batch_start_time,
                                                                                 current_batch_end_time=current_batch_end_time)
        else:
            raise ValueError(f'Not implemented error for negative_sample_strategy {self.negative_sample_strategy}!')
        return negative_src_node_ids, negative_dst_node_ids

    def random_sample(self, size: int):
        """
        random sampling strategy, which is used by previous works
        :param size: int, number of sampled negative edges
        :return:
        """
        if self.seed is None:
            random_sample_edge_src_node_indices = np.random.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = np.random.randint(0, len(self.unique_dst_node_ids), size)
        else:
            random_sample_edge_src_node_indices = self.random_state.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = self.random_state.randint(0, len(self.unique_dst_node_ids), size)
        return self.unique_src_node_ids[random_sample_edge_src_node_indices], self.unique_dst_node_ids[random_sample_edge_dst_node_indices]

    def random_sample_with_collision_check(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray):
        """
        random sampling strategy with collision check, which guarantees that the sampled edges do not appear in the current batch,
        used for historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :return:
        """
        assert batch_src_node_ids is not None and batch_dst_node_ids is not None
        batch_edges = set((batch_src_node_id, batch_dst_node_id) for batch_src_node_id, batch_dst_node_id in zip(batch_src_node_ids, batch_dst_node_ids))
        possible_random_edges = list(self.possible_edges - batch_edges)
        assert len(possible_random_edges) > 0
        # if replace is True, then a value in the list can be selected multiple times, otherwise, a value can be selected only once at most
        random_edge_indices = self.random_state.choice(len(possible_random_edges), size=size, replace=len(possible_random_edges) < size)
        return np.array([possible_random_edges[random_edge_idx][0] for random_edge_idx in random_edge_indices]), \
               np.array([possible_random_edges[random_edge_idx][1] for random_edge_idx in random_edge_indices])

    def historical_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                          current_batch_start_time: float, current_batch_end_time: float):
        """
        historical sampling strategy, first randomly samples among historical edges that are not in the current batch,
        if number of historical edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time, end_time=current_batch_start_time)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time, end_time=current_batch_end_time)
        # get source and destination node ids of unique historical edges
        unique_historical_edges = historical_edges - current_batch_edges
        unique_historical_edges_src_node_ids = np.array([edge[0] for edge in unique_historical_edges])
        unique_historical_edges_dst_node_ids = np.array([edge[1] for edge in unique_historical_edges])

        # if sample size is larger than number of unique historical edges, then fill in remaining edges with randomly sampled edges with collision check
        if size > len(unique_historical_edges):
            num_random_sample_edges = size - len(unique_historical_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(size=num_random_sample_edges,
                                                                                                             batch_src_node_ids=batch_src_node_ids,
                                                                                                             batch_dst_node_ids=batch_dst_node_ids)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_historical_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_historical_edges_dst_node_ids])
        else:
            historical_sample_edge_node_indices = self.random_state.choice(len(unique_historical_edges), size=size, replace=False)
            negative_src_node_ids = unique_historical_edges_src_node_ids[historical_sample_edge_node_indices]
            negative_dst_node_ids = unique_historical_edges_dst_node_ids[historical_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def inductive_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                         current_batch_start_time: float, current_batch_end_time: float):
        """
        inductive sampling strategy, first randomly samples among inductive edges that are not in self.observed_edges and the current batch,
        if number of inductive edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time, end_time=current_batch_start_time)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time, end_time=current_batch_end_time)
        # get source and destination node ids of historical edges but 1) not in self.observed_edges; 2) not in the current batch
        unique_inductive_edges = historical_edges - self.observed_edges - current_batch_edges
        unique_inductive_edges_src_node_ids = np.array([edge[0] for edge in unique_inductive_edges])
        unique_inductive_edges_dst_node_ids = np.array([edge[1] for edge in unique_inductive_edges])

        # if sample size is larger than number of unique inductive edges, then fill in remaining edges with randomly sampled edges
        if size > len(unique_inductive_edges):
            num_random_sample_edges = size - len(unique_inductive_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(size=num_random_sample_edges,
                                                                                                             batch_src_node_ids=batch_src_node_ids,
                                                                                                             batch_dst_node_ids=batch_dst_node_ids)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_inductive_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_inductive_edges_dst_node_ids])
        else:
            inductive_sample_edge_node_indices = self.random_state.choice(len(unique_inductive_edges), size=size, replace=False)
            negative_src_node_ids = unique_inductive_edges_src_node_ids[inductive_sample_edge_node_indices]
            negative_dst_node_ids = unique_inductive_edges_dst_node_ids[inductive_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)
