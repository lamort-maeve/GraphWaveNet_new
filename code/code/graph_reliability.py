"""Past-only structural variation. q values are not causal probabilities."""
import numpy as np
import torch


def reliability_table(bank, history=3, gamma=2., control='real', seed=42):
    """Compute once on CPU, O(W*K*L*N*N), never in a training batch.

    Lag zero and sentinel are excluded. No available comparison => weight 1.
    Shuffled control draws only from already available variation scores;
    random control draws exp(-gamma*U[0,1]), independent of target values.
    """
    if history < 1 or gamma < 0 or not np.isfinite(gamma):
        raise ValueError('history >= 1 and finite gamma >= 0 required')
    if control not in ('real', 'random', 'shuffled', 'uniform'):
        raise ValueError(control)
    values = np.asarray(bank[:, 1:], dtype=np.float64)
    variation = np.zeros(len(bank), dtype=np.float64)
    for i in range(2, len(bank)):
        previous = values[max(1, i-history):i]
        axes = tuple(range(1, previous.ndim))
        numerator = np.abs(previous-values[i]).sum(axis=axes)
        denominator = np.abs(previous).sum(axis=axes) + np.abs(values[i]).sum()
        variation[i] = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                                 where=denominator > 0).mean()
    used = variation.copy()
    rng = np.random.default_rng(seed)
    if control == 'random':
        used[2:] = rng.uniform(size=max(0, len(bank)-2))
    elif control == 'shuffled':
        for i in range(2, len(bank)):
            used[i] = variation[rng.integers(2, i+1)]
    elif control == 'uniform':
        used.fill(0)
    return variation.astype(np.float32), np.exp(-gamma*used).astype(np.float32)


def weighted_consistency(prediction, pseudo, labeled, weights):
    """Original-unit MAE, normalized by eligible weight mass times H*N."""
    eligible = (~labeled).to(prediction.device).reshape(-1)
    weight = weights.to(prediction).reshape(-1) * eligible
    error = (prediction-pseudo.detach()).abs().mean(dim=(1, 2))
    return (error*weight).sum() / weight.sum().clamp_min(torch.finfo(prediction.dtype).tiny)


def lag_reliability_table(bank, history=3, gamma=2., floor=.5, control='real', seed=42):
    """Lag-balanced signed structural anomaly score, calibrated ONLY to past graphs.

    Distance is relative L1 against available predecessor graphs, per lag.
    Ordinary evolution is not automatically declared unreliable: positive
    excess above the preceding distances' median is scaled by past MAD.
    Empty lags are excluded. No future graph, teacher prediction or label enters.
    This is a structural heuristic, NOT an estimated probability of correctness.
    """
    if history < 2 or not np.isfinite(gamma) or gamma < 0 or not 0 <= floor <= 1:
        raise ValueError('history>=2, gamma>=0 and floor in [0,1] required')
    if control not in ('real', 'random', 'shuffled', 'uniform'):
        raise ValueError(control)
    values = np.asarray(bank[:, 1:], dtype=np.float64)
    windows, lags = values.shape[:2]
    distance = np.zeros((windows, lags), np.float64)
    active = np.zeros((windows, lags), bool)
    excess = np.zeros_like(distance)
    score = np.zeros(windows, np.float64)
    for i in range(2, windows):
        previous = values[max(1, i-history):i]
        numerator = np.abs(previous-values[i]).sum(axis=(-1, -2))
        denominator = np.abs(previous).sum(axis=(-1, -2)) + np.abs(values[i]).sum(axis=(-1, -2))
        valid = denominator > 0
        ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=valid)
        active[i] = valid.any(0)
        distance[i] = ratio.sum(0)/np.maximum(valid.sum(0), 1)
        for lag in range(lags):
            start = max(2, i-history)
            past = distance[start:i, lag][active[start:i, lag]]
            if active[i, lag] and len(past) >= 2:
                center = np.median(past)
                scale = max(.05, 1.4826*np.median(np.abs(past-center)))
                excess[i, lag] = np.clip((distance[i, lag]-center)/scale, 0, 5)
        score[i] = excess[i].sum()/max(1, active[i].sum())
    used = score.copy()
    rng = np.random.default_rng(seed)
    if control == 'shuffled':
        for i in range(2, windows):
            used[i] = score[rng.integers(2, i+1)]
    elif control == 'random':
        used[2:] = rng.uniform(0, 5, max(0, windows-2))
    elif control == 'uniform':
        used.fill(0)
    weights = floor+(1-floor)*np.exp(-gamma*used)
    diagnostics = dict(lag_distance=distance.astype(np.float32),
                       lag_active=active, lag_excess=excess.astype(np.float32),
                       structural_score=score.astype(np.float32), applied_score=used.astype(np.float32))
    return weights.astype(np.float32), diagnostics


def absolute_weighted_consistency(prediction, pseudo, labeled, weights):
    """Keep absolute confidence strength; weights=1 recovers ordinary MT MAE."""
    eligible = (~labeled).to(prediction.device).expand_as(prediction)
    error = (prediction-pseudo.detach()).abs()
    weight = weights.to(prediction).reshape(-1, 1, 1)
    return torch.where(eligible, error*weight, torch.zeros_like(error)).sum()/eligible.sum().clamp_min(1)
