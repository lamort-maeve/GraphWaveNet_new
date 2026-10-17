"""Forecast architecture only. Semi-supervised training lives in mean_teacher.py."""
import torch
from torch import nn
from torch.nn import functional as F

from graphwavenet import ReferenceGraphWaveNet

class LaggedPCMCIConv(nn.Module):
    """A[t, lag, source, target] acts on hidden[source, t-lag].

    Snapshot selection is as-of EACH hidden timestamp, never at the forecast
    origin for the entire sequence. Lag zero is excluded (no oriented
    contemporaneous-edge evidence in this archive). No averaging of windows.
    """
    def __init__(self, channels, lag_count):
        super().__init__()
        self.lag_logits = nn.Parameter(torch.zeros(max(1, lag_count-1)))
        self.project = nn.Conv2d(2*channels, channels, 1, bias=False)
        self.gate = nn.Parameter(torch.tensor(-1.0))

    def messages(self, history, bank, ids, output_length):
        length = history.shape[-1]
        positive = torch.zeros_like(history[..., -output_length:])
        negative = torch.zeros_like(positive)
        weights = self.lag_logits.softmax(0)
        # One bank slice per lag avoids a B*T*L*N*N materialization.
        for lag in range(1, min(bank.shape[1], length)):
            source = F.pad(history[..., :-lag], (lag, 0))[..., -output_length:]
            edges = bank[:, lag][ids[:, -output_length:]]  # B,T,S,D
            positive = positive + weights[lag-1]*torch.einsum('bcst,btsd->bcdt', source, edges.clamp_min(0))
            negative = negative + weights[lag-1]*torch.einsum('bcst,btsd->bcdt', source, (-edges).clamp_min(0))
        return torch.cat([positive, -negative], dim=1)

    def forward(self, history, bank, ids, output_length):
        return torch.sigmoid(self.gate)*self.project(self.messages(history, bank, ids, output_length))

class PaperGraphWaveNet(nn.Module):
    """Unmodified adaptive operator + timestamp-aligned internal PCMCI layers.

    All modes train end-to-end with the same schedule. Adaptive forward calls
    the reference implementation directly and has no PCMCI state at all.
    """
    def __init__(self, num_variables, horizon, flow_indices, bank, mode='adaptive',
                 hidden=32, blocks=4, layers=2, dropout=0.3, static_weight=0.):
        super().__init__()
        if mode not in ('adaptive', 'causal', 'combined'):
            raise ValueError(mode)
        if not 0 <= static_weight <= 1 or (mode == 'adaptive' and static_weight):
            raise ValueError('Static mixing requires a causal branch and weight in [0, 1]')
        self.static_weight = static_weight
        self.mode = mode
        self.register_buffer('flow_indices', torch.as_tensor(flow_indices, dtype=torch.long))
        self.backbone = ReferenceGraphWaveNet(torch.device('cpu'), num_variables,
            dropout=dropout, supports=None, gcn_bool=mode != 'causal', addaptadj=mode != 'causal',
            in_dim=1, out_dim=horizon, residual_channels=hidden, dilation_channels=hidden,
            skip_channels=hidden*8, end_channels=hidden*16, blocks=blocks, layers=layers)
        self.receptive_field = self.backbone.receptive_field
        if mode != 'adaptive':
            self.register_buffer('graph_bank', torch.as_tensor(bank, dtype=torch.float32))
            if static_weight:
                # Index zero remains graph-free. Never average future windows.
                self.register_buffer('static_graph_bank', self.graph_bank[:2].clone())
            self.causal_layers = nn.ModuleList([LaggedPCMCIConv(hidden, bank.shape[1])
                                                for _ in range(blocks*layers)])

    def forward(self, inputs, graph_ids=None):
        b = self.backbone
        if self.mode == 'adaptive':
            return b(inputs)[..., -1].index_select(2, self.flow_indices)
        if graph_ids is None or graph_ids.shape != (inputs.shape[0], inputs.shape[-1]):
            raise ValueError('PCMCI graph IDs must align with input timestamps')
        padding = max(0, self.receptive_field-inputs.shape[-1])
        x = b.start_conv(F.pad(inputs, (padding, 0)))
        ids = F.pad(graph_ids, (padding, 0))
        # Padded time positions must not become artificial lagged observations
        # through the bias of start_conv.
        if padding:
            x = torch.cat([torch.zeros_like(x[..., :padding]), x[..., padding:]], -1)
        supports = None
        if self.mode == 'combined':
            supports = [F.softmax(F.relu(b.nodevec1 @ b.nodevec2), dim=1)]
        skip = None
        for i, causal in enumerate(self.causal_layers):
            history = x
            temporal = torch.tanh(b.filter_convs[i](history))*torch.sigmoid(b.gate_convs[i](history))
            length = temporal.shape[-1]
            message = causal(history, self.graph_bank, ids, length)
            if self.static_weight:
                fixed = causal(history, self.static_graph_bank, (ids > 0).long(), length)
                # Shared parameters and a fixed mixing coefficient keep capacity
                # identical to the dynamic-only control.
                message = (1-self.static_weight)*message + self.static_weight*fixed
            # Causal information reaches both the hidden state and the SAME
            # layer's skip path, so even a one-layer model uses the graph.
            s = b.skip_convs[i](temporal + message)
            skip = s if skip is None else skip[..., -length:] + s
            spatial = b.gconv[i](temporal, supports) if supports is not None else b.residual_convs[i](temporal)
            x = b.bn[i](spatial + message + history[..., -length:])
            ids = ids[..., -length:]
        output = b.end_conv_2(F.relu(b.end_conv_1(F.relu(skip))))
        return output[..., -1].index_select(2, self.flow_indices)
