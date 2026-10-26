"""Opt-in v3 modules. All graph and hidden-state views are past-only."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from causal_fusion import HistoricalLagConv


class ContextualHistoricalConv(HistoricalLagConv):
    """Node/time/lag-specific history weights, shrunk towards uniform.

    A pointwise query avoids mixing future hidden timestamps. Split signs before
    aggregation. Start exactly at uniform weights; bounded logits and a uniform
    mixture prevent early single-window collapse. No graph-bank mutation.
    """
    def __init__(self, channels, lag_count, kernel, uniform_mix=.5):
        super().__init__(channels, lag_count, kernel, 'learned', 'masked', 'valid', 'uniform')
        if not 0 <= uniform_mix <= 1:
            raise ValueError('uniform_mix must be in [0,1]')
        self.uniform_mix = uniform_mix
        self.query = nn.Conv2d(channels, lag_count * kernel, 1)
        nn.init.zeros_(self.query.weight)
        nn.init.zeros_(self.query.bias)

    def forward(self, history, bank, ids, output_length, signed_bank=None):
        batch, _, nodes, _ = history.shape
        active = ids[:, -output_length:]
        prior = (active[:, :, None] - torch.arange(self.kernel-1, -1, -1, device=ids.device)).clamp_min(0)
        valid = (prior > 0).permute(0, 2, 1)[:, :, None, :]
        uniform = valid.to(history.dtype) / valid.sum(1, keepdim=True).clamp_min(1)
        query = self.query(history[..., -output_length:]).reshape(batch, -1, self.kernel, nodes, output_length)
        count = min(bank.shape[1], history.shape[-1])-1
        lag_weights = self.lag_logits[:count].softmax(0)
        positive = torch.zeros_like(history[..., -output_length:])
        negative = torch.zeros_like(positive)
        for lag in range(1, count+1):
            source = F.pad(history[..., :-lag], (lag, 0))[..., -output_length:]
            logits = 2 * torch.tanh(query[:, lag] + self.history_logits[lag][None, :, None, None])
            learned = logits.masked_fill(~valid, -1e4).softmax(1) * valid
            learned = learned / learned.sum(1, keepdim=True).clamp_min(1e-12)
            weights = self.uniform_mix*uniform + (1-self.uniform_mix)*learned
            for k in range(self.kernel):
                index = prior[:, :, k]
                if signed_bank is None:
                    edge = bank[:, lag][index]
                    pos, neg = edge.clamp_min(0), (-edge).clamp_min(0)
                else:
                    pos, neg = signed_bank[0][:, lag][index], signed_bank[1][:, lag][index]
                w = lag_weights[lag-1] * weights[:, k][:, None]
                positive = positive + w * torch.einsum('bcst,btsd->bcdt', source, pos)
                negative = negative + w * torch.einsum('bcst,btsd->bcdt', source, neg)
        return self.gate.sigmoid() * self.project(torch.cat([positive, -negative], 1))


class SelectiveCausalInjection(nn.Module):
    """Bound message RMS per node/time and open a contextual residual gate.

    RMS is over channels only (never time). The cap does not amplify weak graph
    messages; zero messages remain exactly zero. Gate starts at 0.2.
    """
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Conv2d(2*channels, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(.2/.8))

    def forward(self, temporal, message):
        reference = temporal.square().mean(1, keepdim=True).add(1e-6).sqrt()
        magnitude = message.square().mean(1, keepdim=True).add(1e-6).sqrt()
        message = message * (reference/magnitude).clamp(max=1)
        gate = self.gate(torch.cat([temporal/reference, message/reference], 1)).sigmoid()
        return gate * message


def calibrated_consistency(teacher, inputs, ids, prediction, labeled, mean, std,
                           structural_weight, noise=.02, beta=2., floor=.5, return_stats=False,
                           noise_generator=None):
    """Teacher-only per-horizon/node uncertainty; no withheld target access.

    Graph sensitivity compares the current available graph with its predecessor;
    graph zero/first graph are not replaced with a spurious unavailable view.
    Structural drift amplifies measured graph sensitivity, never discounts a
    sample solely because drift is high. Normalize by eligible count so uniform
    low confidence actually lowers the loss. Floor bounds discount strength.
    """
    if not 0 <= floor <= 1 or not math.isfinite(beta) or beta < 0 or noise < 0:
        raise ValueError('Invalid confidence parameters')
    with torch.no_grad():
        clean = teacher(inputs, ids)
        perturbation = (torch.randn_like(inputs) if noise_generator is None else
                        torch.randn(inputs.shape, dtype=inputs.dtype, device=inputs.device, generator=noise_generator))
        noisy = teacher(inputs + perturbation*noise, ids)
        previous_ids = torch.where(ids > 1, ids-1, ids)
        previous = teacher(inputs, previous_ids)
        drift = (1-structural_weight).reshape(-1, 1, 1)
        uncertainty = (clean-noisy).abs() + (1+drift)*(clean-previous).abs()
        confidence = floor + (1-floor)*torch.exp(-beta*uncertainty)
        pseudo = clean*std + mean
    eligible = (~labeled).to(prediction.device).expand_as(prediction)
    error = (prediction-pseudo).abs()
    loss = torch.where(eligible, confidence*error, torch.zeros_like(error)).sum()/eligible.sum().clamp_min(1)
    if return_stats:
        return loss, torch.stack([(confidence*eligible).sum(),
                                  (uncertainty*eligible).sum(), eligible.sum()]).detach()
    return loss
