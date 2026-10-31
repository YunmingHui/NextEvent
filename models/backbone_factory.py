import argparse
import numpy as np

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.CTAN import CTAN
from utils.DataLoader import Data
from utils.utils import NeighborSampler

MEMORY_BASED_MODELS = ['TGN', 'CTAN']
# backbones that only embed (src, dst) pairs jointly (neighbor co-occurrence features)
PAIR_BASED_MODELS = ['DyGFormer']


def build_backbone(args: argparse.Namespace, node_raw_features: np.ndarray, edge_raw_features: np.ndarray,
                   neighbor_sampler: NeighborSampler, train_data: Data):
    """
    creates the dynamic backbone named by args.model_name from its hyperparameters in args; shared by train_link_prediction.py
    and train_node_activity_tpp.py
    :param args: argparse.Namespace (or any object with the same attributes)
    :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
    :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
    :param neighbor_sampler: NeighborSampler
    :param train_data: Data, the training split (its time statistics normalize the memory-based models' time deltas)
    :return: nn.Module
    """
    if args.model_name == 'TGAT':
        return TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,
                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
    elif args.model_name == 'TGN':
        # mean and standard deviation of source and destination node time shifts in the training data (required by MemoryModel)
        src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
            compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
        return MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,
                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
    elif args.model_name == 'GraphMixer':
        return GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,
                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
    elif args.model_name == 'DyGFormer':
        return DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=neighbor_sampler,
                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
    elif args.model_name == 'CTAN':
        # normalization of |neighbor's last update - edge time|, from the training split as for TGN's time shifts
        src_node_mean_time_shift, src_node_std_time_shift, _, _ = \
            compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
        return CTAN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, time_dim=args.ctan_time_dim,
                    num_iters=args.ctan_num_iters, epsilon=args.ctan_epsilon, gamma=args.ctan_gamma,
                    sampler_size=args.ctan_sampler_size, mean_delta_t=src_node_mean_time_shift,
                    std_delta_t=src_node_std_time_shift, init_time=float(train_data.node_interact_times.min()), device=args.device)
    raise ValueError(f"Wrong value for model_name {args.model_name}!")
