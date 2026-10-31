from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


def get_link_prediction_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)

    # the setting of seed follows previous works
    random.seed(2020)

    # union to get node set
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    # compute nodes which appear at test time
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
    # sample nodes which we keep as new nodes (to test inductiveness), so then we have to remove all their edges from training
    new_test_node_set = set(random.sample(sorted(test_node_set), int(0.1 * num_total_unique_node_ids)))

    # mask for each source and destination to denote whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0
    # new nodes that are not in the training set
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    # new edges with new nodes in the val and test set (for inductive evaluation)
    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test data
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                             node_interact_times=node_interact_times[new_node_val_mask],
                             edge_ids=edge_ids[new_node_val_mask], labels=labels[new_node_val_mask])

    new_node_test_data = Data(src_node_ids=src_node_ids[new_node_test_mask], dst_node_ids=dst_node_ids[new_node_test_mask],
                              node_interact_times=node_interact_times[new_node_test_mask],
                              edge_ids=edge_ids[new_node_test_mask], labels=labels[new_node_test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data


class NodeActivityData(Data):

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray,
                 src_target: np.ndarray, src_is_censored: np.ndarray, dst_target: np.ndarray, dst_is_censored: np.ndarray, window_end_time: float):
        """
        Data object for the node activity (time-to-next-interaction) task, extending Data with per-edge regression targets for both endpoints.
        :param src_target: ndarray, for the source node of each edge, the observed gap to its next interaction (uncensored) or the known lower bound (censored)
        :param src_is_censored: ndarray of bool, whether src_target is a censored lower bound rather than the true gap
        :param dst_target: ndarray, same as src_target but for the destination node of each edge
        :param dst_is_censored: ndarray of bool, same as src_is_censored but for the destination node of each edge
        :param window_end_time: float, time T at which this split's observation window ends (used to derive the censoring lower bounds)
        """
        super().__init__(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
        self.src_target = src_target
        self.src_is_censored = src_is_censored
        self.dst_target = dst_target
        self.dst_is_censored = dst_is_censored
        self.window_end_time = window_end_time


def get_node_activity_data(dataset_name: str, val_ratio: float, test_ratio: float, when_target: str = 'any'):
    """
    generate data for the node activity (time-to-next-interaction) task
    for every occurrence of a node (as source or destination of an edge), the sample target is either
    the true gap to the node's next interaction (if it happens before the split's own window end T), or
    the known lower bound T - t on that gap (if not, i.e. the sample is right-censored)
    with when_target == 'src' (directed edges): only the source occurrences are samples, and the target is the gap to the
    node's next interaction AS SOURCE (the next one it initiates); dst_target is then NaN (and dst_is_censored False)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :param when_target: str, 'any' (next interaction in either role) or 'src' (next interaction as source)
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data (Data / NodeActivityData objects)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    num_edges = len(src_node_ids)
    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)

    # each edge's own split determines the window end T used for computing its censoring lower bound:
    # train samples only look forward as far as val_time, val samples as far as test_time, test samples as far as the last observed timestamp
    window_end_time_per_edge = np.where(node_interact_times <= val_time, val_time,
                               np.where(node_interact_times <= test_time, test_time, node_interact_times.max()))

    assert when_target in ['any', 'src'], f'Wrong value for when_target {when_target}!'
    # build one occurrence row per (node, edge, endpoint): every edge contributes one occurrence for its source and (unless
    # only source occurrences count, when_target == 'src') one for its destination
    num_endpoints = 1 if when_target == 'src' else 2
    occurrence_node_ids = np.concatenate([src_node_ids, dst_node_ids][:num_endpoints])
    occurrence_times = np.concatenate([node_interact_times, node_interact_times][:num_endpoints])
    occurrence_window_end = np.concatenate([window_end_time_per_edge, window_end_time_per_edge][:num_endpoints])
    occurrence_edge_row = np.concatenate([np.arange(num_edges), np.arange(num_edges)][:num_endpoints])
    # endpoint == 0 marks a source occurrence, endpoint == 1 marks a destination occurrence
    occurrence_endpoint = np.concatenate([np.zeros(num_edges, dtype=np.int8), np.ones(num_edges, dtype=np.int8)][:num_endpoints])

    occurrence_df = pd.DataFrame({'node': occurrence_node_ids, 't': occurrence_times, 'window_end': occurrence_window_end,
                                  'edge_row': occurrence_edge_row, 'endpoint': occurrence_endpoint})

    # unique (node, t) pairs, sorted by node then time, so that a groupby shift(-1) yields each node's next strictly-later distinct interaction time
    unique_node_time = occurrence_df[['node', 't']].drop_duplicates().sort_values(['node', 't'], kind='mergesort').reset_index(drop=True)
    unique_node_time['next_t'] = unique_node_time.groupby('node')['t'].shift(-1)

    # broadcast next_t back onto every occurrence row that shares the same (node, t) (there can be several, e.g. multiple edges of the same node at the same timestamp)
    occurrence_df = occurrence_df.merge(unique_node_time, on=['node', 't'], how='left')

    is_censored = occurrence_df['next_t'].isna().values | (occurrence_df['next_t'].values > occurrence_df['window_end'].values)
    true_gap = occurrence_df['next_t'].values - occurrence_df['t'].values
    lower_bound_gap = occurrence_df['window_end'].values - occurrence_df['t'].values
    occurrence_df['target'] = np.where(is_censored, lower_bound_gap, true_gap)
    occurrence_df['is_censored'] = is_censored

    src_occurrences = occurrence_df[occurrence_df['endpoint'] == 0].sort_values('edge_row')
    dst_occurrences = occurrence_df[occurrence_df['endpoint'] == 1].sort_values('edge_row')

    src_target_all = src_occurrences['target'].values
    src_is_censored_all = src_occurrences['is_censored'].values
    if when_target == 'src':
        # destination occurrences are not samples
        dst_target_all, dst_is_censored_all = np.full(num_edges, np.nan), np.zeros(num_edges, dtype=bool)
    else:
        dst_target_all = dst_occurrences['target'].values
        dst_is_censored_all = dst_occurrences['is_censored'].values

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    def build_split_data(mask: np.ndarray, split_window_end_time: float):
        return NodeActivityData(src_node_ids=src_node_ids[mask], dst_node_ids=dst_node_ids[mask],
                                node_interact_times=node_interact_times[mask], edge_ids=edge_ids[mask], labels=labels[mask],
                                src_target=src_target_all[mask], src_is_censored=src_is_censored_all[mask],
                                dst_target=dst_target_all[mask], dst_is_censored=dst_is_censored_all[mask],
                                window_end_time=split_window_end_time)

    train_data = build_split_data(train_mask, val_time)
    val_data = build_split_data(val_mask, test_time)
    test_data = build_split_data(test_mask, node_interact_times.max())

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    for split_name, split_data in [('training', train_data), ('validation', val_data), ('test', test_data)]:
        print("The {} dataset has {} interactions, involving {} different nodes, censored ratio {:.4f}".format(
            split_name, split_data.num_interactions, split_data.num_unique_nodes,
            (split_data.src_is_censored.sum() + split_data.dst_is_censored.sum()) / (num_endpoints * max(split_data.num_interactions, 1))))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data


def truncate_train_window(train_data: NodeActivityData, train_window_days: float):
    """
    restricts a node activity train split to only its most recent `train_window_days` days, i.e. drops
    training occurrences earlier than (window_end_time - train_window_days), used to study how much
    training history the model actually needs; window_end_time (the causal boundary used for censoring)
    is left unchanged, only which occurrences are available as training samples shrinks
    :param train_data: NodeActivityData, the full train split returned by get_node_activity_data
    :param train_window_days: float, number of days immediately before window_end_time to keep
    :return: NodeActivityData, the truncated train split
    """
    window_start_time = train_data.window_end_time - train_window_days * 86400.0
    mask = train_data.node_interact_times >= window_start_time

    return NodeActivityData(src_node_ids=train_data.src_node_ids[mask], dst_node_ids=train_data.dst_node_ids[mask],
                            node_interact_times=train_data.node_interact_times[mask], edge_ids=train_data.edge_ids[mask],
                            labels=train_data.labels[mask],
                            src_target=train_data.src_target[mask], src_is_censored=train_data.src_is_censored[mask],
                            dst_target=train_data.dst_target[mask], dst_is_censored=train_data.dst_is_censored[mask],
                            window_end_time=train_data.window_end_time)


def build_node_history_sequences(full_data: Data, when_target: str = 'any'):
    """
    for every node, its own chronological sequence of occurrence times (across the whole dataset, both as
    source and destination) and the log1p(gap) between consecutive occurrences - the input a standard
    LogNormMix-style RNN context encoder (NodeHistoryRNNEncoder) needs, since that encoder looks only at a
    node's own history and never at the graph's neighbor structure
    :param full_data: Data, the full dataset (as returned by get_node_activity_data)
    :param when_target: str, 'any' (occurrences in either role) or 'src' (only the interactions the node initiates,
    the process predicted with --when_target src; nodes that never initiate one are absent)
    :return:
        node_occurrence_times: dict[int, np.ndarray], node_id -> sorted array of its own occurrence times
        node_gap_sequences: dict[int, np.ndarray of float32], node_id -> log1p(gap) between consecutive
        occurrences (length = len(node_occurrence_times[node_id]) - 1)
    """
    num_endpoints = 1 if when_target == 'src' else 2
    nodes = np.concatenate([full_data.src_node_ids, full_data.dst_node_ids][:num_endpoints])
    times = np.concatenate([full_data.node_interact_times, full_data.node_interact_times][:num_endpoints])
    order = np.lexsort((times, nodes))
    nodes_sorted, times_sorted = nodes[order], times[order]

    unique_nodes, start_idx = np.unique(nodes_sorted, return_index=True)
    start_idx = list(start_idx) + [len(nodes_sorted)]

    node_occurrence_times, node_gap_sequences = {}, {}
    for k, node_id in enumerate(unique_nodes):
        seg = times_sorted[start_idx[k]:start_idx[k + 1]]
        node_occurrence_times[int(node_id)] = seg
        node_gap_sequences[int(node_id)] = np.log1p(np.diff(seg)).astype(np.float32)

    return node_occurrence_times, node_gap_sequences
