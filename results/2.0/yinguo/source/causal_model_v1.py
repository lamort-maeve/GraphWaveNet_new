"""Graph WaveNet with a time-indexed, signed, lag-resolved PCMCI operator."""
import torch
from torch import nn
from torch.nn import functional as F


def lag_message(x, adjacency, lag):
    """x[B,C,S,T], A[B,T,S,D]; messages travel source(t-lag) -> target(t)."""
    if lag >= x.shape[-1]:
        return torch.zeros_like(x)
    shifted = F.pad(x[..., :-lag], (lag, 0)) if lag else x
    return torch.einsum('bcst,btsd->bcdt', shifted, adjacency)


class CausalGraphConv(nn.Module):
    def __init__(self, channels, lag_count, causal, adaptive, dropout):
        super().__init__()
        self.causal, self.adaptive, self.dropout = causal, adaptive, dropout
        self.lag_logits = nn.Parameter(torch.zeros(lag_count)) if causal else None
        self.project = nn.Conv2d(channels * (1 + int(causal) + 2*int(adaptive)), channels, 1)

    def forward(self, x, bank, graph_ids, adaptive_adj):
        parts = [x]
        if self.causal:
            message = torch.zeros_like(x)
            weights = self.lag_logits.softmax(0)
            for lag in range(bank.shape[1]):
                # Gather one lag at a time, avoiding a B*T*L*N*N allocation.
                a = bank[:, lag][graph_ids]
                message = message + weights[lag] * lag_message(x, a, lag)
            parts.append(message)
        if self.adaptive:
            first = torch.einsum('bcst,sd->bcdt', x, adaptive_adj)
            second = torch.einsum('bcst,sd->bcdt', first, adaptive_adj)
            parts.extend([first, second])
        return F.dropout(self.project(torch.cat(parts, dim=1)), self.dropout, self.training)


class DynamicGraphWaveNet(nn.Module):
    def __init__(self, num_variables, horizon, flow_indices, bank, mode='causal',
                 hidden=32, blocks=4, layers=2, dropout=0.3):
        super().__init__()
        if mode not in ('causal', 'adaptive', 'combined'):
            raise ValueError(mode)
        self.mode = mode
        causal, adaptive = mode != 'adaptive', mode != 'causal'
        # Stored once, moved with the model, included in checkpoints for reproducibility.
        self.register_buffer('graph_bank', torch.as_tensor(bank, dtype=torch.float32))
        self.register_buffer('flow_indices', torch.as_tensor(flow_indices, dtype=torch.long))
        if adaptive:
            self.nodevec1 = nn.Parameter(torch.randn(num_variables, 10) * 0.1)
            self.nodevec2 = nn.Parameter(torch.randn(10, num_variables) * 0.1)
        self.start = nn.Conv2d(1, hidden, 1)
        self.filters, self.gates, self.graphs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.skips, self.norms = nn.ModuleList(), nn.ModuleList()
        self.dilations = [2**layer for _ in range(blocks) for layer in range(layers)]
        self.receptive_field = 1 + sum(self.dilations)
        for layer_index, dilation in enumerate(self.dilations):
            self.filters.append(nn.Conv2d(hidden, hidden, (1, 2), dilation=(1, dilation)))
            self.gates.append(nn.Conv2d(hidden, hidden, (1, 2), dilation=(1, dilation)))
            self.graphs.append(CausalGraphConv(hidden, bank.shape[1], causal, adaptive, dropout))
            self.skips.append(nn.Conv2d(hidden, hidden*8, 1))
            # Channel-only normalization at each (node,time), no future-time statistics.
            self.norms.append(nn.LayerNorm(hidden) if layer_index < len(self.dilations)-1 else nn.Identity())
        self.end1 = nn.Conv2d(hidden*8, hidden*16, 1)
        self.end2 = nn.Conv2d(hidden*16, horizon, 1)

    def forward(self, x, graph_ids):
        if graph_ids.shape != (x.shape[0], x.shape[-1]):
            raise ValueError('Graph IDs must align with every input time step')
        x = self.start(x)
        adaptive = None
        if self.mode != 'causal':
            adaptive = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)
        skip = None
        for dilation, filt, gate, graph, proj, norm in zip(
                self.dilations, self.filters, self.gates, self.graphs, self.skips, self.norms):
            residual = x
            padded = F.pad(x, (dilation, 0))
            x = torch.tanh(filt(padded)) * torch.sigmoid(gate(padded))
            x = graph(x, self.graph_bank, graph_ids, adaptive)
            # Skip AFTER graph convolution: even the last graph layer affects forecasts.
            contribution = proj(x)
            skip = contribution if skip is None else skip + contribution
            x = norm((x + residual).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.end2(F.relu(self.end1(F.relu(skip[..., -1:]))))
        return out[..., 0].index_select(2, self.flow_indices)  # B,H,20
