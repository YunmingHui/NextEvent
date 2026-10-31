import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn

from models.modules import LogNormMixHead, NodeHistoryRNNEncoder
from models.backbone_factory import build_backbone, MEMORY_BASED_MODELS
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer, get_node_activity_tpp_loss, build_rnn_context
from utils.utils import get_neighbor_sampler
from utils.metrics import get_node_activity_tpp_metrics
from utils.DataLoader import get_idx_data_loader, get_node_activity_data, truncate_train_window, build_node_history_sequences
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_node_activity_args


# the defaults of --num_mixtures / --head_hidden_dims (utils/load_configs.py); only other values appear in a checkpoint name
DEFAULT_NUM_MIXTURES = 16
DEFAULT_HEAD_HIDDEN_DIMS = '128,64'
DEFAULT_MAX_HISTORY = 512


def node_activity_save_name(args):
    """checkpoint / result name of a when model. Non-default settings add a
    suffix, so a run with them never collides with (or silently overwrites) the default run"""
    window_suffix = f'_trainwindow{args.train_window_days}d' if args.train_window_days is not None else ''
    mixtures_suffix = '' if args.num_mixtures == DEFAULT_NUM_MIXTURES else f'_k{args.num_mixtures}'
    head_suffix = '' if args.head_hidden_dims == DEFAULT_HEAD_HIDDEN_DIMS else f'_head{args.head_hidden_dims.replace(",", "-")}'
    # only the RNN context reads the history window
    max_history = getattr(args, 'max_history', DEFAULT_MAX_HISTORY)
    history_suffix = f'_hist{max_history}' if args.context_source in ('rnn', 'both') and max_history != DEFAULT_MAX_HISTORY else ''
    # checkpoints from before --when_target predict the next interaction in either role and have no suffix
    target_suffix = '_srcwhen' if getattr(args, 'when_target', 'any') == 'src' else ''
    # DyGFormer's original checkpoints (from before --dygformer_partner) were paired with the current edge's
    # other end and have no suffix, so the padding-paired variant gets its own name and never overwrites them
    partner_suffix = '_dstpad' if args.model_name == 'DyGFormer' and args.dygformer_partner == 'pad' else ''
    return f'node_activity_tpp_{args.context_source}_{args.model_name}_seed{args.seed}{window_suffix}{mixtures_suffix}{head_suffix}' \
           f'{history_suffix}{target_suffix}{partner_suffix}'


def compute_context(model: nn.ModuleDict, model_name: str, context_source: str,
                    src_ids: np.ndarray, dst_ids: np.ndarray, times: np.ndarray, edge_ids: np.ndarray,
                    node_gap_sequences: dict, node_occurrence_times: dict,
                    num_neighbors: int, time_gap: int, device: str, dygformer_partner: str = 'pad',
                    max_history: int = DEFAULT_MAX_HISTORY, when_target: str = 'any'):
    """
    builds the src/dst context vectors fed into the LogNormMixHead, from whichever source(s) --context_source
    selects; shared by both the training loop and evaluation so the two never drift apart
    :param dygformer_partner: str, 'pad' or 'edge', see --dygformer_partner. Must match how the checkpoint was
    trained: checkpoints without the "_dstpad" suffix were trained with 'edge'
    :param max_history: int, the most recent gaps the RNN context reads per query (see --max_history)
    :param when_target: str, see --when_target; with 'src' only the source nodes are queries and dst_context is None
    (a memory-based backbone still embeds the whole edge, so its memory sees the destination too)
    :return: (src_context, dst_context), each Tensor of shape (batch_size, context_dim)
    """
    src_only = when_target == 'src'
    src_parts, dst_parts = [], []

    if context_source in ('embedding', 'both'):
        backbone = model['backbone']
        if model_name == 'TGAT':
            se, de = backbone.compute_src_dst_node_temporal_embeddings(src_node_ids=src_ids, dst_node_ids=dst_ids,
                                                                       node_interact_times=times, num_neighbors=num_neighbors)
        elif model_name in ['TGN', 'CTAN']:
            se, de = backbone.compute_src_dst_node_temporal_embeddings(src_node_ids=src_ids, dst_node_ids=dst_ids,
                                                                       node_interact_times=times, edge_ids=edge_ids,
                                                                       edges_are_positive=True, num_neighbors=num_neighbors)
        elif model_name == 'GraphMixer':
            se, de = backbone.compute_src_dst_node_temporal_embeddings(src_node_ids=src_ids, dst_node_ids=dst_ids,
                                                                       node_interact_times=times, num_neighbors=num_neighbors, time_gap=time_gap)
        elif model_name == 'DyGFormer' and dygformer_partner == 'pad':
            # DyGFormer only embeds (src, dst) pairs; pair every node with padding node 0 (no neighbors, zero
            # features) instead of its current partner, so the co-occurrence channel
            # only sees the node's own history and the partner side is a constant token - one call covers both src and dst nodes
            node_ids = src_ids if src_only else np.concatenate([src_ids, dst_ids])
            embeds, _ = backbone.compute_src_dst_node_temporal_embeddings(src_node_ids=node_ids, dst_node_ids=np.zeros_like(node_ids),
                                                                         node_interact_times=times if src_only else np.concatenate([times, times]))
            se, de = embeds[:len(src_ids)], embeds[len(src_ids):]
        elif model_name == 'DyGFormer':
            se, de = backbone.compute_src_dst_node_temporal_embeddings(src_node_ids=src_ids, dst_node_ids=dst_ids, node_interact_times=times)
        else:
            raise ValueError(f"Wrong value for model_name {model_name}!")
        src_parts.append(se)
        dst_parts.append(de)

    if context_source in ('rnn', 'both'):
        rnn_encoder = model['rnn_encoder']
        src_parts.append(build_rnn_context(rnn_encoder, node_gap_sequences, node_occurrence_times, src_ids, times, device=device,
                                           max_history=max_history))
        if not src_only:
            dst_parts.append(build_rnn_context(rnn_encoder, node_gap_sequences, node_occurrence_times, dst_ids, times, device=device,
                                               max_history=max_history))

    if src_only:
        return torch.cat(src_parts, dim=-1), None
    return torch.cat(src_parts, dim=-1), torch.cat(dst_parts, dim=-1)


def stack_queries(src_context: torch.Tensor, dst_context: torch.Tensor, data, indices: np.ndarray):
    """
    the head's input and the targets of a batch of edges: the src queries, then (unless dst_context is None, --when_target src)
    the dst queries
    :return: context Tensor, targets ndarray, is_censored ndarray
    """
    if dst_context is None:
        return src_context, data.src_target[indices], data.src_is_censored[indices]
    return (torch.cat([src_context, dst_context], dim=0), np.concatenate([data.src_target[indices], data.dst_target[indices]]),
            np.concatenate([data.src_is_censored[indices], data.dst_is_censored[indices]]))


def evaluate_split(model: nn.ModuleDict, model_name: str, context_source: str, evaluate_data, idx_loader,
                   node_gap_sequences: dict, node_occurrence_times: dict, num_neighbors: int, time_gap: int,
                   device: str, dygformer_partner: str = 'pad', max_history: int = DEFAULT_MAX_HISTORY, when_target: str = 'any'):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        all_log_weights, all_means, all_log_stds, all_targets, all_is_censored = [], [], [], [], []
        loader_tqdm = tqdm(idx_loader, ncols=120)
        for batch_idx, indices in enumerate(loader_tqdm):
            indices = indices.numpy()
            src_ids, dst_ids, times, edge_ids = (evaluate_data.src_node_ids[indices], evaluate_data.dst_node_ids[indices],
                                                  evaluate_data.node_interact_times[indices], evaluate_data.edge_ids[indices])

            src_context, dst_context = compute_context(model, model_name, context_source, src_ids, dst_ids, times, edge_ids,
                                                        node_gap_sequences, node_occurrence_times, num_neighbors, time_gap, device,
                                                        dygformer_partner, max_history, when_target)
            batch_context, batch_targets, batch_is_censored = stack_queries(src_context, dst_context, evaluate_data, indices)

            log_weights, means, log_stds = model['head'](x=batch_context)
            targets = torch.from_numpy(batch_targets).float().to(device)
            is_censored = torch.from_numpy(batch_is_censored).bool().to(device)

            loss, _, _ = get_node_activity_tpp_loss(log_weights=log_weights, means=means, log_stds=log_stds,
                                                     targets=targets, is_censored=is_censored)
            total_loss += loss.item()
            all_log_weights.append(log_weights.detach().cpu())
            all_means.append(means.detach().cpu())
            all_log_stds.append(log_stds.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_is_censored.append(is_censored.detach().cpu())
            loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        total_loss /= (batch_idx + 1)
        all_log_weights, all_means, all_log_stds = torch.cat(all_log_weights), torch.cat(all_means), torch.cat(all_log_stds)
        all_targets, all_is_censored = torch.cat(all_targets), torch.cat(all_is_censored)
        metrics = get_node_activity_tpp_metrics(log_weights=all_log_weights, means=all_means, log_stds=all_log_stds,
                                                targets=all_targets, is_censored=all_is_censored)
    return total_loss, metrics


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    args = get_node_activity_args()

    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data = \
        get_node_activity_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio, when_target=args.when_target)

    if args.train_window_days is not None:
        train_data = truncate_train_window(train_data, args.train_window_days)
        print(f'truncated training window to the most recent {args.train_window_days} days: '
              f'{train_data.num_interactions} interactions remain')

    # a node's own occurrence history, for the RNN context source - built once from the full (train+val+test)
    # timeline, same causal convention as the graph neighbor sampler (a later query only ever reads an earlier
    # position in its own node's sequence, never a future one)
    # (--when_target src: the node's own initiated interactions only, the process being predicted)
    node_occurrence_times, node_gap_sequences = build_node_history_sequences(full_data, when_target=args.when_target)

    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # standardization stats for LogNormMixHead: mean/std of log(gap) over the training set's actually
    # OBSERVED gaps (uncensored only - a censored lower bound is not a realized sample of the target
    # distribution, so it's excluded here, same reasoning as everywhere else censored values are kept
    # separate from true values in this task). log(gap), not log1p(gap): the LogNormMix/TPP literature
    # defines the mixture directly over log(gap) (and get_node_activity_tpp_loss's log_normal_mixture_*
    # functions do too, via torch.log(x)), so standardizing on log1p would silently parameterize a
    # shifted log-normal instead of the standard one, and make the NLL not comparable to other work.
    # This dataset has no zero-length gaps (smallest observed gap is 1 second), so log() has no domain
    # issue here, but the clamp mirrors the model side's clamp(min=1.0) as a protection regardless.
    # (--when_target src: the source queries only, the dst targets are not samples)
    train_observed_log_gaps = np.concatenate([
        np.log(np.clip(train_data.src_target[~train_data.src_is_censored], 1.0, None)),
    ] + ([] if args.when_target == 'src' else [np.log(np.clip(train_data.dst_target[~train_data.dst_is_censored], 1.0, None))]))
    target_log_mean = float(train_observed_log_gaps.mean())
    target_log_std = float(train_observed_log_gaps.std())
    print(f'LogNormMixHead standardization stats (from observed train gaps): mean={target_log_mean:.3f}  std={target_log_std:.3f}')

    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, test_metric_all_runs = [], []

    # --seed runs that single seed only (one cluster job per seed); otherwise seeds 0 .. num_runs - 1
    for run in ([args.seed] if args.seed is not None else range(args.num_runs)):

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = node_activity_save_name(args)

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        log_folder = os.path.join(args.output_root, 'logs', args.model_name, args.dataset_name, args.save_model_name)
        os.makedirs(log_folder, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_folder, f'{str(time.time())}.log'))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")
        logger.info(f'configuration is {args}')

        components = {}

        if args.context_source in ('embedding', 'both'):
            dynamic_backbone = build_backbone(args, node_raw_features, edge_raw_features, full_neighbor_sampler, train_data)
            components['backbone'] = dynamic_backbone

        if args.context_source in ('rnn', 'both'):
            components['rnn_encoder'] = NodeHistoryRNNEncoder(hidden_dim=node_raw_features.shape[1])

        context_dim = node_raw_features.shape[1] * (2 if args.context_source == 'both' else 1)
        head_hidden_dims = tuple(int(d) for d in args.head_hidden_dims.split(','))
        components['head'] = LogNormMixHead(input_dim=context_dim, num_mixtures=args.num_mixtures, dropout=args.dropout,
                                            target_log_mean=target_log_mean, target_log_std=target_log_std,
                                            hidden_dims=head_hidden_dims)

        model = nn.ModuleDict(components)
        # memory-based backbones carry a memory bank that evolves with every batch it embeds (compute_context passes
        # edges_are_positive=True), so it is reset each epoch, detached each batch, and checkpointed together with
        # the weights, following train_link_prediction.py
        memory_based = args.model_name in MEMORY_BASED_MODELS and 'backbone' in model
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, context_source: {args.context_source}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        model = convert_to_gpu(model, device=args.device)

        save_model_folder = os.path.join(args.output_root, 'saved_models', args.model_name, args.dataset_name, args.save_model_name) + '/'
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)
        # the hyperparameters this checkpoint was trained with, so evaluation can rebuild it exactly (models/backbone_factory.py)
        with open(os.path.join(save_model_folder, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger,
                                       model_name=args.model_name if memory_based else None)

        for epoch in range(args.num_epochs):

            model.train()
            if 'backbone' in model:
                model['backbone'].set_neighbor_sampler(full_neighbor_sampler)
            if memory_based:
                model['backbone'].memory_bank.__init_memory_bank__()

            train_total_loss, train_uncensored_nll_total, train_censored_nll_total = 0.0, 0.0, 0.0
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                src_ids, dst_ids, times, edge_ids = (train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices],
                                                      train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices])

                src_context, dst_context = compute_context(model, args.model_name, args.context_source, src_ids, dst_ids, times, edge_ids,
                                                            node_gap_sequences, node_occurrence_times, args.num_neighbors, args.time_gap, args.device,
                                                            args.dygformer_partner, args.max_history, args.when_target)
                batch_context, batch_targets, batch_is_censored = stack_queries(src_context, dst_context, train_data, train_data_indices)

                log_weights, means, log_stds = model['head'](x=batch_context)
                targets = torch.from_numpy(batch_targets).float().to(args.device)
                is_censored = torch.from_numpy(batch_is_censored).bool().to(args.device)

                loss, uncensored_nll, censored_nll = get_node_activity_tpp_loss(log_weights=log_weights, means=means, log_stds=log_stds,
                                                                                 targets=targets, is_censored=is_censored)

                train_total_loss += loss.item()
                train_uncensored_nll_total += uncensored_nll.item()
                train_censored_nll_total += censored_nll.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if memory_based:
                    # so the next batch does not back propagate through all earlier batches' memory updates
                    model['backbone'].memory_bank.detach_memory_bank()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}, '
                                                            f'uncensored nll: {uncensored_nll.item()}, censored nll: {censored_nll.item()}')

            train_total_loss /= (batch_idx + 1)
            train_uncensored_nll_total /= (batch_idx + 1)
            train_censored_nll_total /= (batch_idx + 1)

            val_total_loss, val_metrics = evaluate_split(model, args.model_name, args.context_source, val_data, val_idx_data_loader,
                                                          node_gap_sequences, node_occurrence_times, args.num_neighbors, args.time_gap,
                                                          args.device, args.dygformer_partner, args.max_history, args.when_target)

            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {train_total_loss:.4f} '
                        f'(uncensored nll: {train_uncensored_nll_total:.4f}, censored nll: {train_censored_nll_total:.4f})')
            logger.info(f'validate loss: {val_total_loss:.4f}')
            for metric_name in val_metrics.keys():
                logger.info(f'validate {metric_name}, {val_metrics[metric_name]}')

            if (epoch + 1) % args.test_interval_epochs == 0:
                if memory_based:
                    # memory now holds train + val; the checkpoint saved below must not also hold the test split
                    val_backup_memory_bank = model['backbone'].memory_bank.backup_memory_bank()
                test_total_loss, test_metrics = evaluate_split(model, args.model_name, args.context_source, test_data, test_idx_data_loader,
                                                                node_gap_sequences, node_occurrence_times, args.num_neighbors, args.time_gap,
                                                                args.device, args.dygformer_partner, args.max_history, args.when_target)
                logger.info(f'test loss: {test_total_loss:.4f}')
                for metric_name in test_metrics.keys():
                    logger.info(f'test {metric_name}, {test_metrics[metric_name]}')
                if memory_based:
                    model['backbone'].memory_bank.reload_memory_bank(val_backup_memory_bank)

            early_stop = early_stopping.step([('loss', val_total_loss, False)], model)
            if early_stopping.counter == 0:
                # this epoch's model was just saved as the best one so far
                best_val_total_loss, best_val_metrics = val_total_loss, val_metrics
            if early_stop:
                break

        early_stopping.load_checkpoint(model)

        logger.info(f'get final performance on dataset {args.dataset_name}...')

        if memory_based:
            # the best checkpoint's memory has already consumed the val split, so val can't be re-run causally: report
            # the val metrics recorded at the epoch that checkpoint was saved (the same model), and test continues from it
            val_total_loss, val_metrics = best_val_total_loss, best_val_metrics
        else:
            val_total_loss, val_metrics = evaluate_split(model, args.model_name, args.context_source, val_data, val_idx_data_loader,
                                                          node_gap_sequences, node_occurrence_times, args.num_neighbors, args.time_gap,
                                                          args.device, args.dygformer_partner, args.max_history, args.when_target)
        test_total_loss, test_metrics = evaluate_split(model, args.model_name, args.context_source, test_data, test_idx_data_loader,
                                                        node_gap_sequences, node_occurrence_times, args.num_neighbors, args.time_gap,
                                                        args.device, args.dygformer_partner, args.max_history, args.when_target)

        val_metric_dict, test_metric_dict = {}, {}
        logger.info(f'validate loss: {val_total_loss:.4f}')
        for metric_name in val_metrics.keys():
            logger.info(f'validate {metric_name}, {val_metrics[metric_name]}')
            val_metric_dict[metric_name] = val_metrics[metric_name]

        logger.info(f'test loss: {test_total_loss:.4f}')
        for metric_name in test_metrics.keys():
            logger.info(f'test {metric_name}, {test_metrics[metric_name]}')
            test_metric_dict[metric_name] = test_metrics[metric_name]

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        result_json = json.dumps({"validate metrics": val_metric_dict, "test metrics": test_metric_dict}, indent=4)
        save_result_folder = os.path.join(args.output_root, 'saved_results', args.model_name, args.dataset_name)
        os.makedirs(save_result_folder, exist_ok=True)
        with open(os.path.join(save_result_folder, f"{args.save_model_name}.json"), 'w') as file:
            file.write(result_json)

    logger.info(f'metrics over {args.num_runs} runs:')
    for metric_name in val_metric_all_runs[0].keys():
        values = [r[metric_name] for r in val_metric_all_runs]
        logger.info(f'validate {metric_name}, {values}')
        logger.info(f'average validate {metric_name}, {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}')
    for metric_name in test_metric_all_runs[0].keys():
        values = [r[metric_name] for r in test_metric_all_runs]
        logger.info(f'test {metric_name}, {values}')
        logger.info(f'average test {metric_name}, {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}')

    sys.exit()
