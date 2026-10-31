import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_node_activity_tpp_metrics(log_weights: torch.Tensor, means: torch.Tensor, log_stds: torch.Tensor,
                                  targets: torch.Tensor, is_censored: torch.Tensor):
    """
    get metrics for the LogNormMix (temporal point process) formulation of the node activity task.
    The primary metric is negative log-likelihood (NLL), which - unlike the point-estimate task - needs
    no separate treatment for censored vs. uncensored samples (both are just two cases of the same
    likelihood), but we still report them split for comparability with the point-estimate task, plus a
    point-estimate MAE/RMSE (via the mixture's closed-form mean) and a calibration check for censored
    samples (their predicted survival probability at the known lower bound - should not be low, since we
    know for a fact the true value exceeds that bound)
    :param log_weights: Tensor, shape (num_samples, num_mixtures)
    :param means: Tensor, shape (num_samples, num_mixtures)
    :param log_stds: Tensor, shape (num_samples, num_mixtures)
    :param targets: Tensor, shape (num_samples, ), true gap (uncensored) or known lower bound (censored), raw seconds
    :param is_censored: Tensor of bool, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    from utils.utils import log_normal_mixture_log_prob, log_normal_mixture_log_survival, log_normal_mixture_median

    log_weights, means, log_stds = log_weights.cpu().detach(), means.cpu().detach(), log_stds.cpu().detach()
    targets = targets.cpu().detach()
    is_censored = is_censored.cpu().detach().bool()
    safe_targets = targets.clamp(min=1.0)

    num_uncensored = int((~is_censored).sum().item())
    num_censored = int(is_censored.sum().item())
    num_total = num_uncensored + num_censored

    metrics = {
        'num_uncensored': num_uncensored,
        'num_censored': num_censored,
        'censored_ratio': num_censored / num_total if num_total > 0 else float('nan'),
    }

    # point-estimate summary of the predicted distribution, via its MEDIAN (found by bisection) rather
    # than its closed-form mean - the mean of a log-normal mixture is not robust, a single component
    # with a large sigma can dominate exp(mu + sigma^2/2) even though it carries little density near the
    # observed data; the median has no such pathology and is the fair like-for-like point estimate to
    # compare against the last-gap baseline
    if num_total > 0:
        point_estimate_all = log_normal_mixture_median(log_weights, means, log_stds)

    if num_uncensored > 0:
        u = ~is_censored
        uncensored_log_prob = log_normal_mixture_log_prob(safe_targets[u], log_weights[u], means[u], log_stds[u])
        metrics['uncensored_nll'] = (-uncensored_log_prob).mean().item()

        point_estimate = point_estimate_all[u]
        metrics['uncensored_mae'] = (point_estimate - targets[u]).abs().mean().item()
        metrics['uncensored_rmse'] = torch.sqrt(((point_estimate - targets[u]) ** 2).mean()).item()
        metrics['uncensored_log_mae'] = (torch.log(point_estimate.clamp(min=1.0)) - torch.log(targets[u].clamp(min=1.0))).abs().mean().item()
    else:
        metrics['uncensored_nll'] = metrics['uncensored_mae'] = metrics['uncensored_rmse'] = metrics['uncensored_log_mae'] = float('nan')

    if num_censored > 0:
        c = is_censored
        censored_log_survival = log_normal_mixture_log_survival(safe_targets[c], log_weights[c], means[c], log_stds[c])
        metrics['censored_nll'] = (-censored_log_survival).mean().item()
        # calibration (TPP-only, not comparable to censored_accuracy below): mean predicted P(T > known
        # lower bound) - a well-calibrated model should not assign low probability to something we know is
        # true. This is a probability the model itself reports, not a hit-rate, so it is NOT the same
        # quantity as censored_accuracy and the two should never be compared head-to-head.
        metrics['censored_survival_prob'] = censored_log_survival.exp().mean().item()
        # true like-for-like analogue of the point-estimate task's censored_accuracy: does the MEDIAN point
        # estimate itself clear the known lower bound? same binary hit-rate definition, comparable to
        # last-gap and the point-estimate models
        metrics['censored_accuracy'] = (point_estimate_all[c] >= targets[c]).float().mean().item()
    else:
        metrics['censored_nll'] = metrics['censored_survival_prob'] = metrics['censored_accuracy'] = float('nan')

    if num_total > 0:
        all_log_prob = torch.where(is_censored,
                                   log_normal_mixture_log_survival(safe_targets, log_weights, means, log_stds),
                                   log_normal_mixture_log_prob(safe_targets, log_weights, means, log_stds))
        metrics['overall_nll'] = (-all_log_prob).mean().item()
    else:
        metrics['overall_nll'] = float('nan')

    return metrics
