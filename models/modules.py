import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# which NeighborCooccurrenceEncoder.count_nodes_appearances (DyGFormer) to use: 'loop' (the original per-pair
# Python loop) or 'vectorized' (count_nodes_appearances_vectorized, identical counts)
COOCCURRENCE_IMPL = os.environ.get('DYG_COOCCURRENCE_IMPL', 'vectorized')


def count_nodes_appearances_vectorized(src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray,
                                       device: str, max_elements: int = 2 ** 28):
    """
    batched version of NeighborCooccurrenceEncoder.count_nodes_appearances: for every position of every sequence, how often its
    node appears in the sequence of the source node and in that of the destination node of the same pair (padding node 0 -> 0)
    :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
    :param dst_padded_nodes_neighbor_ids: ndarray, shape (batch_size, dst_max_seq_length)
    :param device: str, device
    :param max_elements: int, largest (rows, seq_length, seq_length) comparison tensor built at once
    :return: two Tensors, shape (batch_size, src_max_seq_length, 2) and (batch_size, dst_max_seq_length, 2), float
    """
    src = torch.from_numpy(src_padded_nodes_neighbor_ids).to(device)
    dst = torch.from_numpy(dst_padded_nodes_neighbor_ids).to(device)
    max_seq_length = max(src.shape[1], dst.shape[1])
    rows = max(1, max_elements // (max_seq_length * max_seq_length))
    src_appearances, dst_appearances = [], []
    for start in range(0, src.shape[0], rows):
        s, d = src[start:start + rows], dst[start:start + rows]
        # (rows, seq_length, 2): [appearances in the source sequence, appearances in the destination sequence]
        src_appearances.append(torch.stack([(s.unsqueeze(2) == s.unsqueeze(1)).sum(dim=2),
                                            (s.unsqueeze(2) == d.unsqueeze(1)).sum(dim=2)], dim=2))
        dst_appearances.append(torch.stack([(d.unsqueeze(2) == s.unsqueeze(1)).sum(dim=2),
                                            (d.unsqueeze(2) == d.unsqueeze(1)).sum(dim=2)], dim=2))
    src_appearances = torch.cat(src_appearances, dim=0).float()
    dst_appearances = torch.cat(dst_appearances, dim=0).float()
    src_appearances[src == 0] = 0.0
    dst_appearances[dst == 0] = 0.0
    return src_appearances, dst_appearances


class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True, beta: float = 9):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        :param beta: float, frequencies are initialized as 1 / 10 ** linspace(0, beta, time_dim) (default 9, as in DyGLib)
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, beta, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output


class MergeLayer(nn.Module):

    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        """
        merge and project the inputs
        :param input_1: Tensor, shape (*, input_dim1)
        :param input_2: Tensor, shape (*, input_dim2)
        :return:
        """
        # Tensor, shape (*, input_dim1 + input_dim2)
        x = torch.cat([input_1, input_2], dim=1)
        # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        """
        Multi-Layer Perceptron Classifier.
        :param input_dim: int, dimension of input
        :param dropout: float, dropout rate
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 80)
        self.fc2 = nn.Linear(80, 10)
        self.fc3 = nn.Linear(10, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        multi-layer perceptron classifier forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        # Tensor, shape (*, 80)
        x = self.dropout(self.act(self.fc1(x)))
        # Tensor, shape (*, 10)
        x = self.dropout(self.act(self.fc2(x)))
        # Tensor, shape (*, 1)
        return self.fc3(x)


class NodeHistoryRNNEncoder(nn.Module):

    def __init__(self, hidden_dim: int = 172, num_layers: int = 1):
        """
        Standard LogNormMix-style context encoder: a GRU that processes a node's OWN past sequence of
        inter-event gaps (no graph structure at all - a node is just an independent time series here),
        producing a context vector for predicting its next gap. This is the original context source from
        "Intensity-Free Learning of Temporal Point Processes" (Shchur et al., 2020), kept as an alternative
        (or complement) to the GNN embedding as input to LogNormMixHead.
        :param hidden_dim: int, GRU hidden size (and therefore the context vector's dimension)
        :param num_layers: int, number of GRU layers
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        # learned context for a node's very first-ever occurrence, where no gap history exists yet
        self.initial_context = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, padded_gap_sequences: torch.Tensor):
        """
        :param padded_gap_sequences: Tensor, shape (num_nodes, max_len, 1), log1p(gap) sequences, zero-padded
        at the end; padding past a node's real length is never read by the caller, so its content is irrelevant
        :return: Tensor, shape (num_nodes, max_len, hidden_dim); output[n, k] is the context available for
        predicting the gap that follows the (k+1)-th realized gap in node n's history (0-indexed k)
        """
        outputs, _ = self.gru(padded_gap_sequences)
        return outputs


class LogNormMixHead(nn.Module):

    def __init__(self, input_dim: int, num_mixtures: int = 16, dropout: float = 0.1,
                 target_log_mean: float = 0.0, target_log_std: float = 1.0, hidden_dims: tuple = (80, 10)):
        """
        Mixture-density head that turns a node embedding into the parameters of a K-component
        log-normal mixture over the (raw, seconds) gap until the next interaction, following the
        LogNormMix parameterization from "Intensity-Free Learning of Temporal Point Processes"
        (Shchur et al., 2020).

        The MLP trunk predicts in a STANDARDIZED space (z-scored log(gap), using target_log_mean/std
        computed once from the training set's observed gaps), not directly in raw log-seconds - this
        keeps its raw outputs near O(1) instead of needing to output values like 30+ to represent a
        multi-year gap, which is what let a handful of near-zero-history nodes push mu to nonsensical
        values with nothing to hold it back. forward() applies the inverse affine transform internally,
        so every caller still receives (and every downstream loss/metric still consumes) means/log_stds
        in the original log(gap) space (the space the LogNormMix/TPP literature defines the mixture over,
        and the space log_normal_mixture_log_prob/log_survival actually evaluate) - only the *internal*
        parameterization changed.
        :param input_dim: int, dimension of input
        :param num_mixtures: int, number of log-normal mixture components
        :param dropout: float, dropout rate
        :param target_log_mean: float, mean of log(observed gap) over the training set
        :param target_log_std: float, std of log(observed gap) over the training set (must be > 0)
        :param hidden_dims: tuple of int, width of each hidden layer. The default (80, 10) matches
        MLPClassifier's trunk, which this head's shape was originally copied from - but MLPClassifier
        was designed to end in a single scalar (10 -> 1), so narrowing all the way down to 10 units
        before expanding back out to 3*num_mixtures outputs is a real bottleneck here, unlike there.
        A wider trunk (e.g. (128, 64)) gives the head more capacity to reconstruct that many numbers.
        """
        super().__init__()
        self.num_mixtures = num_mixtures
        # layers are named fc1, fc2, ..., fcN (via setattr, so each is properly registered as a submodule)
        # rather than held in an nn.ModuleList, so that the default hidden_dims=(80, 10) produces EXACTLY
        # the same state_dict keys (fc1/fc2/fc3) as the original hardcoded version - existing checkpoints
        # (trained before this got parameterized) still load with strict=True. self._layers is a plain
        # Python list of the same layer objects, only for ordered iteration in forward(); since each layer
        # was already registered once via setattr, this does not add any extra state_dict entries.
        dims = [input_dim] + list(hidden_dims) + [3 * num_mixtures]
        self._layers = []
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            setattr(self, f'fc{i + 1}', layer)
            self._layers.append(layer)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # registered as buffers so they move with .to(device) and are saved/loaded with the checkpoint
        self.register_buffer('target_log_mean', torch.tensor(float(target_log_mean)))
        self.register_buffer('target_log_std', torch.tensor(float(target_log_std)))

    def forward(self, x: torch.Tensor):
        """
        mixture-density head forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
            log_weights: Tensor, shape (*, num_mixtures), log-softmax mixture weights (log pi_k)
            means: Tensor, shape (*, num_mixtures), mu_k of each log-normal component, in the ORIGINAL
            log(gap in seconds) space (already de-standardized)
            log_stds: Tensor, shape (*, num_mixtures), log(sigma_k) of each component, same space
        """
        for layer in self._layers[:-1]:
            x = self.dropout(self.act(layer(x)))
        # Tensor, shape (*, 3 * num_mixtures)
        out = self._layers[-1](x)
        raw_weights, means_z, log_stds_z = out.chunk(3, dim=-1)
        log_weights = torch.log_softmax(raw_weights, dim=-1)
        # clamp in the STANDARDIZED space, relative to the training data's own scale, rather than as
        # absolute numbers guessed with no reference point: a component's mean stays within +-6 std of the
        # training distribution, and its own spread stays within roughly [0.018x, 7.4x] of that distribution's
        # overall spread - both are generous safety nets, not meant to bind for a typical, well-supported node
        means_z = means_z.clamp(min=-6.0, max=6.0)
        log_stds_z = log_stds_z.clamp(min=-4.0, max=2.0)
        # de-standardize back to the original log(gap) space
        means = means_z * self.target_log_std + self.target_log_mean
        log_stds = log_stds_z + torch.log(self.target_log_std)
        return log_weights, means, log_stds


class MultiHeadAttention(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int,
                 num_heads: int = 2, dropout: float = 0.1):
        """
        Multi-head Attention module.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiHeadAttention, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        self.head_dim = self.query_dim // num_heads

        self.query_projection = nn.Linear(self.query_dim, num_heads * self.head_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)

        self.scaling_factor = self.head_dim ** -0.5

        self.layer_norm = nn.LayerNorm(self.query_dim)

        self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor, neighbor_node_features: torch.Tensor,
                neighbor_node_time_features: torch.Tensor, neighbor_node_edge_features: torch.Tensor, neighbor_masks: np.ndarray):
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        :return:
        """
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        # shape (batch_size, 1, num_heads, self.head_dim)
        query = self.query_projection(query).reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features], dim=2)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        key = self.key_projection(key).reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        value = self.value_projection(value).reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        value = value.permute(0, 2, 1, 3)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention = torch.einsum('bhld,bhnd->bhln', query, key)
        attention = attention * self.scaling_factor

        # Tensor, shape (batch_size, 1, num_neighbors)
        attention_mask = torch.from_numpy(neighbor_masks).to(node_features.device).unsqueeze(dim=1)
        attention_mask = attention_mask == 0
        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        attention_mask = torch.stack([attention_mask for _ in range(self.num_heads)], dim=1)

        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        # note that if a node has no valid neighbor (whose neighbor_masks are all zero), directly set the masks to -np.inf will make the
        # attention scores after softmax be nan. Therefore, we choose a very large negative number (-1e10 following TGAT) instead of -np.inf to tackle this case
        attention = attention.masked_fill(attention_mask, -1e10)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention_scores = self.dropout(torch.softmax(attention, dim=-1))

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        attention_output = torch.einsum('bhln,bhnd->bhld', attention_scores, value)

        # Tensor, shape (batch_size, 1, num_heads * self.head_dim), where num_heads * self.head_dim is equal to node_feat_dim + time_feat_dim
        attention_output = attention_output.permute(0, 2, 1, 3).flatten(start_dim=2)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.dropout(self.residual_fc(attention_output))

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.layer_norm(output + residual)

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        output = output.squeeze(dim=1)
        # Tensor, shape (batch_size, num_heads, num_neighbors)
        attention_scores = attention_scores.squeeze(dim=2)

        return output, attention_scores


class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs_query: torch.Tensor, inputs_key: torch.Tensor = None, inputs_value: torch.Tensor = None,
                neighbor_masks: np.ndarray = None):
        """
        encode the inputs by Transformer encoder
        :param inputs_query: Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        :param inputs_key: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param inputs_value: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param neighbor_masks: ndarray, shape (batch_size, source_seq_length), used to create mask of neighbors for nodes in the batch
        :return:
        """
        if inputs_key is None or inputs_value is None:
            assert inputs_key is None and inputs_value is None
            inputs_key = inputs_value = inputs_query
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # transposed_inputs_query, Tensor, shape (target_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_key, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_value, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        transposed_inputs_query, transposed_inputs_key, transposed_inputs_value = inputs_query.transpose(0, 1), inputs_key.transpose(0, 1), inputs_value.transpose(0, 1)

        if neighbor_masks is not None:
            # Tensor, shape (batch_size, source_seq_length)
            neighbor_masks = torch.from_numpy(neighbor_masks).to(inputs_query.device) == 0

        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs_query, key=transposed_inputs_key,
                                                  value=transposed_inputs_value, key_padding_mask=neighbor_masks)[0].transpose(0, 1)
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[0](inputs_query + self.dropout(hidden_states))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        return outputs
