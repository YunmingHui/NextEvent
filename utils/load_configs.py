import argparse
import sys
import torch


def get_link_prediction_args():
    """
    get the args for the link prediction task
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia',
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci', 'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts',
                                 'bitcoinotc', 'bitcoinalpha'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--model_name', type=str, default='DyGFormer', help='name of the model',
                        choices=['TGAT', 'TGN', 'GraphMixer', 'DyGFormer', 'CTAN'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--seed', type=int, default=None, help='if set, run only this seed (overrides --num_runs)')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--ctan_num_iters', type=int, default=3, help='CTAN: AntiSymmetricConv iterations (= hops of the recent neighborhood)')
    parser.add_argument('--ctan_epsilon', type=float, default=0.1, help='CTAN: AntiSymmetricConv step size')
    parser.add_argument('--ctan_gamma', type=float, default=0.1, help='CTAN: AntiSymmetricConv diffusion strength')
    parser.add_argument('--ctan_sampler_size', type=int, default=5, help='CTAN: number of most recent neighbors kept per node')
    parser.add_argument('--ctan_time_dim', type=int, default=16, help='CTAN: dimension of the edge time encoding')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        # non-zero exit status, so that a calling shell script with set -e stops
        sys.exit(2)

    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)

    return args


def load_link_prediction_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the link prediction task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name == 'TGAT':
        args.num_neighbors = 20
        args.num_layers = 2
        if args.dataset_name in ['enron', 'CanParl', 'UNvote']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        if args.dataset_name in ['reddit', 'CanParl', 'UNtrade']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'TGN':
        args.num_neighbors = 10
        args.num_layers = 1
        if args.dataset_name in ['mooc', 'UNtrade']:
            args.dropout = 0.2
        elif args.dataset_name in ['lastfm', 'CanParl']:
            args.dropout = 0.3
        elif args.dataset_name in ['enron', 'SocialEvo']:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
        if args.dataset_name in ['CanParl', 'UNvote']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'GraphMixer':
        args.num_layers = 2
        if args.dataset_name in ['wikipedia']:
            args.num_neighbors = 30
        elif args.dataset_name in ['reddit', 'lastfm']:
            args.num_neighbors = 10
        else:
            args.num_neighbors = 20
        if args.dataset_name in ['wikipedia', 'reddit', 'enron']:
            args.dropout = 0.5
        elif args.dataset_name in ['mooc', 'uci', 'USLegis']:
            args.dropout = 0.4
        elif args.dataset_name in ['lastfm', 'UNvote']:
            args.dropout = 0.0
        elif args.dataset_name in ['SocialEvo']:
            args.dropout = 0.3
        elif args.dataset_name in ['Flights', 'CanParl']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        if args.dataset_name in ['CanParl', 'UNtrade', 'UNvote']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'DyGFormer':
        args.num_layers = 2
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        elif args.dataset_name in ['mooc', 'enron', 'Flights', 'USLegis', 'UNtrade']:
            args.max_input_sequence_length = 256
            args.patch_size = 8
        elif args.dataset_name in ['lastfm']:
            args.max_input_sequence_length = 512
            args.patch_size = 16
        elif args.dataset_name in ['CanParl']:
            args.max_input_sequence_length = 2048
            args.patch_size = 64
        elif args.dataset_name in ['UNvote']:
            args.max_input_sequence_length = 128
            args.patch_size = 4
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ['reddit', 'UNvote']:
            args.dropout = 0.2
        elif args.dataset_name in ['enron', 'USLegis', 'UNtrade', 'Contacts']:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
    elif args.model_name == 'CTAN':
        # no per-dataset best configuration is published for these datasets; keep the command-line / default values
        pass
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")


def get_node_activity_args():
    """
    get the args for the node activity (time-to-next-interaction) task
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the node activity (time-to-next-interaction) task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia',
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci', 'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts',
                                 'bitcoinotc', 'bitcoinalpha'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--model_name', type=str, default='DyGFormer', help='name of the model, trained from scratch jointly with the regression head; '
                        '"LogNormMix" is the graph-free baseline (no backbone, only with --context_source rnn)',
                        choices=['TGAT', 'TGN', 'GraphMixer', 'DyGFormer', 'CTAN', 'LogNormMix'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--seed', type=int, default=None, help='if set, run only this seed (overrides --num_runs)')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--train_window_days', type=float, default=None,
                        help='if set, only keep training occurrences within this many days of the train window end (val_time), '
                        'to study how much training history the model needs; default (unset) uses the full training window')
    parser.add_argument('--num_mixtures', type=int, default=16,
                        help='number of log-normal mixture components for the LogNormMix (TPP) formulation of the task '
                        '(only used by train_node_activity_tpp.py). Default 16 is the default of the LogNormMix class in the '
                        'original implementation (shchur/ifl-tpp); the paper reports the model is robust to K (its Table 3). '
                        'A value other than the default adds a "_k<K>" suffix to the checkpoint name')
    parser.add_argument('--context_source', type=str, default='embedding', choices=['embedding', 'rnn', 'both'],
                        help='what to feed the LogNormMixHead (only used by train_node_activity_tpp.py): "embedding" is the '
                        'GNN backbone embedding (graph-aware, no RNN), "rnn" is the standard LogNormMix context - a GRU over '
                        'just the node\'s own past inter-event gaps, no graph at all - and "both" concatenates the two')
    parser.add_argument('--when_target', type=str, default='src', choices=['src', 'any'],
                        help='what the when model predicts (only used by train_node_activity_tpp.py): "src" (edges are directed) - '
                        'at every interaction a node initiates, the time to the next one it initiates (only source occurrences '
                        'are samples; in a bipartite graph these are the source side only); "any" - at every occurrence of a node '
                        'in either role, the time to its next occurrence in either role (the earlier setting). "src" adds a '
                        '"_srcwhen" suffix to the checkpoint name, so the earlier checkpoints keep theirs')
    parser.add_argument('--output_root', type=str, default='.',
                        help='where saved_models/, saved_results/ and logs/ go (only used by train_node_activity_tpp.py); the '
                        'default "." is the repository root. Hyperparameter searches set their own folder, so a run whose name '
                        'does not encode every searched value (e.g. the learning rate) never overwrites the main results')
    parser.add_argument('--max_history', type=int, default=512,
                        help='the most recent inter-event gaps of a node the RNN context reads per query (context_source rnn / both); '
                        'the full history of very active nodes does not fit in GPU memory (reddit). A value other than the default '
                        'adds a "_hist<N>" suffix to the checkpoint name')
    parser.add_argument('--head_hidden_dims', type=str, default='128,64',
                        help='comma-separated hidden layer widths for the LogNormMixHead MLP trunk (only used by '
                        'train_node_activity_tpp.py). The earlier "80,10" (copied from MLPClassifier\'s trunk) narrows down to '
                        '10 units before expanding back out to 3*num_mixtures outputs, a bottleneck here unlike in '
                        'MLPClassifier\'s single-scalar output. A value other than the default adds a "_head<dims>" suffix to the '
                        'checkpoint name, and a checkpoint is only loadable with the value it was trained with.')
    parser.add_argument('--dygformer_partner', type=str, default='pad', choices=['pad', 'edge'],
                        help='what the pair-based backbone (DyGFormer) pairs each node with when computing its embedding '
                        '(only used by train_node_activity_tpp.py): "pad" pairs every node with padding node 0, '
                        'so its embedding (including the neighbor co-occurrence features) depends only on its own history - '
                        'like every other backbone here; "edge" pairs src with dst of the current edge (the original behavior, '
                        'which leaks "is this a repeat partner" into the when embedding)')
    parser.add_argument('--ctan_num_iters', type=int, default=3, help='CTAN: AntiSymmetricConv iterations (= hops of the recent neighborhood)')
    parser.add_argument('--ctan_epsilon', type=float, default=0.1, help='CTAN: AntiSymmetricConv step size')
    parser.add_argument('--ctan_gamma', type=float, default=0.1, help='CTAN: AntiSymmetricConv diffusion strength')
    parser.add_argument('--ctan_sampler_size', type=int, default=5, help='CTAN: number of most recent neighbors kept per node')
    parser.add_argument('--ctan_time_dim', type=int, default=16, help='CTAN: dimension of the edge time encoding')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        # non-zero exit status, so that a calling shell script with set -e stops
        sys.exit(2)

    if args.model_name == 'LogNormMix' and args.context_source != 'rnn':
        parser.error('--model_name LogNormMix has no graph backbone, use it with --context_source rnn')
    if args.max_history < 1:
        parser.error('--max_history must be at least 1')
    if args.load_best_configs:
        load_node_activity_best_configs(args=args)

    return args


def load_node_activity_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the node activity task
    the backbone is trained from scratch jointly with the regression head, with the link prediction task's
    per-model/per-dataset configurations
    :param args: argparse.Namespace
    :return:
    """
    if args.model_name == 'LogNormMix':
        # no backbone: the GRU and the head keep their defaults
        return
    load_link_prediction_best_configs(args=args)
