"""Past-only graph-history fusion inspired by v2, isolated from reference controls."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def split_signed_bank(bank):
    """Split fixed graph signs once per model forward, shared by all layers.

    Both arrays have the original bank layout. Gathering from them produces
    exactly the same values/shapes as splitting AFTER each layer's gather.
    No floating-point sums or learnable computations are reordered. Keeping
    this cache local to forward avoids stale data after load_state_dict/.to().
    """
    return bank.clamp_min(0), (-bank).clamp_min(0)


class HistoricalLagConv(nn.Module):
    """Learn convex weights over current/past snapshots, independently per lag.

    Split signs BEFORE mixing. Graph zero is the unavailable sentinel. The
    unique-ID gather avoids constructing B*T*K*N*N history tensors.
    """
    def __init__(self, channels, lag_count, kernel=3, fusion='learned',
                 history_padding='zero', lag_normalization='all', history_init='recent', uniform_mix=.5,
                 lag_readout='aligned', history_weight_sharing='per_lag'):
        super().__init__()
        if fusion not in ('learned', 'uniform', 'shrunk') or history_padding not in ('zero', 'masked'):
            raise ValueError('Invalid historical fusion policy')
        self.fusion, self.history_padding = fusion, history_padding
        if not 0 <= uniform_mix <= 1:
            raise ValueError('uniform_mix must be in [0,1]')
        self.uniform_mix = uniform_mix
        if lag_normalization not in ('all','valid') or kernel < 1 or lag_count < 2:
            raise ValueError('Invalid lag normalization or historical dimensions')
        self.lag_normalization = lag_normalization
        self.kernel = kernel
        if lag_readout not in ('aligned', 'synchronous'):
            raise ValueError('Invalid lag readout')
        if history_weight_sharing not in ('per_lag', 'shared'):
            raise ValueError('Invalid history weight sharing')
        self.lag_readout = lag_readout
        self.history_weight_sharing = history_weight_sharing
        self.history_logits = nn.Parameter(torch.zeros(lag_count if history_weight_sharing == 'per_lag' else 1, kernel))
        self.history_logits.requires_grad_(fusion in ('learned', 'shrunk'))
        if history_init not in ('recent', 'uniform'):
            raise ValueError(history_init)
        if kernel > 1 and history_init == 'recent':
            with torch.no_grad():
                self.history_logits[:, -1] = math.log(4 * (kernel - 1))
        self.lag_logits = nn.Parameter(torch.zeros(max(1, lag_count - 1)))
        self.project = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.gate = nn.Parameter(torch.tensor(-1.0))

    def forward(self, history, bank, ids, output_length, signed_bank=None, prepared_history=None):
        active = ids[:, -output_length:]
        if prepared_history is None:
            unique, inverse = torch.unique(active, sorted=True, return_inverse=True)
            offsets = torch.arange(self.kernel - 1, -1, -1, device=ids.device)
            prior = (unique[:, None] - offsets).clamp_min(0)
        elif self.fusion != 'uniform':
            raise ValueError('Only fixed uniform fusion can use a premerged cache')
        positive = torch.zeros_like(history[..., -output_length:])
        negative = torch.zeros_like(positive)
        lag_count = min(bank.shape[1], history.shape[-1]) - 1
        lag_weights = (self.lag_logits[:lag_count] if self.lag_normalization == 'valid'
                       else self.lag_logits).softmax(0)
        for lag in range(1, min(bank.shape[1], history.shape[-1])):
            # Keep graph layers, signs, history weights and lag mixture unchanged.
            # The synchronous control only removes the feature time shift.
            source = (F.pad(history[..., :-lag], (lag, 0))[..., -output_length:]
                      if self.lag_readout == 'aligned' else history[..., -output_length:])
            if prepared_history is not None:
                pos = prepared_history[0][:, lag][active]
                neg = prepared_history[1][:, lag][active]
                positive = positive + lag_weights[lag-1] * torch.einsum('bcst,btsd->bcdt', source, pos)
                negative = negative + lag_weights[lag-1] * torch.einsum('bcst,btsd->bcdt', source, neg)
                continue
            if signed_bank is None:
                edges = bank[:, lag][prior]  # unique,K,source,target
                positive_edges = edges.clamp_min(0)
                negative_edges = (-edges).clamp_min(0)
            else:
                positive_edges = signed_bank[0][:, lag][prior]
                negative_edges = signed_bank[1][:, lag][prior]
            row = lag if self.history_weight_sharing == 'per_lag' else 0
            logits = (self.history_logits[row] if self.fusion in ('learned', 'shrunk')
                      else torch.zeros_like(self.history_logits[row]))
            if self.fusion == 'shrunk':
                logits = 2*logits.tanh()
            weights = logits.softmax(0)[None, :].expand(len(unique), -1)
            if self.history_padding == 'masked':
                weights = weights * (prior > 0)
                weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
            if self.fusion == 'shrunk':
                base = torch.ones_like(weights)/self.kernel
                if self.history_padding == 'masked':
                    base = base*(prior > 0)
                    base = base/base.sum(1, keepdim=True).clamp_min(1e-12)
                weights = self.uniform_mix*base + (1-self.uniform_mix)*weights
            weights = weights[:, :, None, None]
            pos = (positive_edges * weights).sum(1)[inverse]
            neg = (negative_edges * weights).sum(1)[inverse]
            positive = positive + lag_weights[lag-1] * torch.einsum('bcst,btsd->bcdt', source, pos)
            negative = negative + lag_weights[lag-1] * torch.einsum('bcst,btsd->bcdt', source, neg)
        return self.gate.sigmoid() * self.project(torch.cat([positive, -negative], 1))


class StaticSpatialConv(nn.Module):
    """A compressed static adjacency has spatial edges, not recoverable delays."""
    def __init__(self, channels):
        super().__init__()
        self.project = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.gate = nn.Parameter(torch.tensor(-1.0))

    def forward(self, history, matrix, available, output_length):
        source = history[..., -output_length:]
        pos = torch.einsum('bcst,sd->bcdt', source, matrix.clamp_min(0))
        neg = torch.einsum('bcst,sd->bcdt', source, (-matrix).clamp_min(0))
        message = self.gate.sigmoid() * self.project(torch.cat([pos, -neg], 1))
        return message * available[:, None, None, -output_length:]
