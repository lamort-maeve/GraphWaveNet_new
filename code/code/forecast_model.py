"""Forecast architecture only. Semi-supervised training lives in mean_teacher.py."""
import torch
from torch import nn
from torch.nn import functional as F

from graphwavenet import ReferenceGraphWaveNet
from causal_revision import ContextualHistoricalConv, SelectiveCausalInjection
from causal_fusion import HistoricalLagConv, StaticSpatialConv, split_signed_bank
from execution import FixedGraphCache

class LaggedPCMCIConv(nn.Module):
    """A[t, lag, source, target] acts on hidden[source, t-lag].

    Snapshot selection is as-of EACH hidden timestamp, never at the forecast
    origin for the entire sequence. Lag zero is excluded (no oriented
    contemporaneous-edge evidence in this archive). No averaging of windows.
    """
    def __init__(self, channels, lag_count, lag_normalization='all', lag_readout='aligned'):
        super().__init__()
        if lag_readout not in ('aligned', 'synchronous'):
            raise ValueError('Invalid lag readout')
        self.lag_readout = lag_readout
        self.lag_normalization = lag_normalization
        self.lag_logits = nn.Parameter(torch.zeros(max(1, lag_count-1)))
        self.project = nn.Conv2d(2*channels, channels, 1, bias=False)
        self.gate = nn.Parameter(torch.tensor(-1.0))

    def messages(self, history, bank, ids, output_length, signed_bank=None):
        length = history.shape[-1]
        positive = torch.zeros_like(history[..., -output_length:])
        negative = torch.zeros_like(positive)
        count = min(bank.shape[1], length) - 1
        weights = (self.lag_logits[:count] if self.lag_normalization == 'valid'
                   else self.lag_logits).softmax(0)
        # One bank slice per lag avoids a B*T*L*N*N materialization.
        for lag in range(1, min(bank.shape[1], length)):
            source = (F.pad(history[..., :-lag], (lag, 0))[..., -output_length:]
                      if self.lag_readout == 'aligned' else history[..., -output_length:])
            active = ids[:, -output_length:]
            if signed_bank is None:
                edges = bank[:, lag][active]  # B,T,S,D
                positive_edges, negative_edges = edges.clamp_min(0), (-edges).clamp_min(0)
            else:
                positive_edges = signed_bank[0][:, lag][active]
                negative_edges = signed_bank[1][:, lag][active]
            positive = positive + weights[lag-1]*torch.einsum('bcst,btsd->bcdt', source, positive_edges)
            negative = negative + weights[lag-1]*torch.einsum('bcst,btsd->bcdt', source, negative_edges)
        return torch.cat([positive, -negative], dim=1)

    def forward(self, history, bank, ids, output_length, signed_bank=None):
        return torch.sigmoid(self.gate)*self.project(self.messages(history, bank, ids, output_length, signed_bank))

class PaperGraphWaveNet(nn.Module):
    """Reference adaptive baseline + optional timestamp-aligned PCMCI revision.

    All modes train end-to-end with the same schedule. Adaptive forward calls
    the reference implementation directly and has no PCMCI state at all.
    """
    def __init__(self, num_variables, horizon, flow_indices, bank, mode='adaptive',
                 hidden=32, blocks=4, layers=2, dropout=0.3, static_weight=0.,
                 static_matrix=None, static_min_graph_id=1, graph_kernel=1,
                 custom_supports=None, custom_adaptive=False, causal_execution='reference',
                 history_fusion='learned', history_padding='zero', lag_normalization='all', history_init='recent', causal_revision='legacy', adaptive_scale=1., uniform_mix=.5,
                 static_fusion='convex', lag_readout='aligned', history_weight_sharing='per_lag',
                 dynamic_message_scale=1.):
        super().__init__()
        if mode not in ('adaptive', 'causal', 'combined', 'custom'):
            raise ValueError(mode)
        if not 0 <= static_weight <= 1 or (mode in ('adaptive', 'custom') and static_weight):
            raise ValueError('Static mixing requires a causal branch and weight in [0, 1]')
        if causal_revision not in ("legacy", "selective") or not 0 <= adaptive_scale <= 1:
            raise ValueError("Invalid revision/adaptive scale")
        if history_fusion == "contextual" and graph_kernel < 2:
            raise ValueError("Contextual history requires K >= 2")
        if history_fusion == 'contextual' and (history_padding, lag_normalization, history_init) != ('masked', 'valid', 'uniform'):
            raise ValueError('Contextual history requires masked/valid/uniform policies')
        if history_fusion == 'shrunk' and (graph_kernel < 2 or (history_padding, lag_normalization, history_init) != ('masked', 'valid', 'uniform')):
            raise ValueError('Shrunk history requires K>=2 and masked/valid/uniform policies')
        if adaptive_scale != 1 and mode != 'combined':
            raise ValueError('Adaptive scaling is only defined for combined mode')
        if lag_readout not in ('aligned', 'synchronous') or history_weight_sharing not in ('per_lag', 'shared'):
            raise ValueError('Invalid static/lag ablation configuration')
        if not 0 <= dynamic_message_scale <= 1:
            raise ValueError('dynamic_message_scale must be in [0,1]')
        if (lag_readout != 'aligned' or history_weight_sharing != 'per_lag' or dynamic_message_scale != 1) and mode not in ('combined', 'causal'):
            raise ValueError('Static/lag ablations require a causal branch')
        if history_fusion == 'contextual' and (lag_readout != 'aligned' or history_weight_sharing != 'per_lag'):
            raise ValueError('Static/lag readout controls are defined for uniform/learned/shrunk fusion')
        if history_weight_sharing == 'shared' and (graph_kernel < 2 or history_fusion not in ('learned', 'shrunk')):
            raise ValueError('Shared history weights require K>=2 and learned/shrunk fusion')
        self.dynamic_message_scale = dynamic_message_scale
        self.causal_revision, self.adaptive_scale = causal_revision, adaptive_scale
        self.static_weight = static_weight
        if static_fusion not in ('convex', 'gated_residual'):
            raise ValueError('Invalid static fusion')
        if static_fusion == 'gated_residual' and (static_matrix is None or not static_weight):
            raise ValueError('Gated residual requires an available external long-term prior and nonzero static weight')
        self.static_fusion = static_fusion
        self.mode = mode
        if causal_execution not in ('reference', 'signed_cache', 'persistent_cache'):
            raise ValueError(causal_execution)
        self.causal_execution = causal_execution
        self.execution_cache = FixedGraphCache()
        self.graph_kernel, self.history_padding, self.history_fusion = graph_kernel, history_padding, history_fusion
        self.static_min_graph_id = static_min_graph_id
        if graph_kernel < 1:
            raise ValueError('graph_kernel must be positive')
        if mode == 'custom' and not custom_supports:
            raise ValueError('Custom reference mode requires adjacency supports')
        supports = ([torch.as_tensor(a, dtype=torch.float32) for a in custom_supports]
                    if mode == 'custom' else None)
        self.register_buffer('flow_indices', torch.as_tensor(flow_indices, dtype=torch.long))
        self.backbone = ReferenceGraphWaveNet(torch.device('cpu'), num_variables,
            dropout=dropout, supports=supports, gcn_bool=mode != 'causal',
            addaptadj=custom_adaptive if mode == 'custom' else mode != 'causal',
            in_dim=1, out_dim=horizon, residual_channels=hidden, dilation_channels=hidden,
            skip_channels=hidden*8, end_channels=hidden*16, blocks=blocks, layers=layers)
        self.receptive_field = self.backbone.receptive_field
        if mode == 'custom':
            # Original reference uses a plain list. Register the same tensors in
            # the wrapper so .to(), teacher copies and checkpoints preserve them.
            for i, support in enumerate(supports):
                self.register_buffer(f'custom_support_{i}', support)
            self.custom_support_count = len(supports)
        if mode not in ('adaptive', 'custom'):
            self.register_buffer('graph_bank', torch.as_tensor(bank, dtype=torch.float32))
            if static_matrix is not None:
                self.register_buffer('static_matrix', torch.as_tensor(static_matrix, dtype=torch.float32))
                if static_fusion == 'gated_residual':
                    # Preserve dynamic-branch initialization in the matched no-prior control.
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(torch.initial_seed()+7919)
                        self.static_layers = nn.ModuleList([StaticSpatialConv(hidden) for _ in range(blocks*layers)])
                    self.static_residual_logits = nn.Parameter(torch.zeros(blocks*layers))
                else:
                    self.static_layers = nn.ModuleList([StaticSpatialConv(hidden) for _ in range(blocks*layers)])
            elif static_weight:
                # Index zero remains graph-free. Never average future windows.
                self.register_buffer('static_graph_bank', self.graph_bank[:2].clone())
            self.causal_layers = nn.ModuleList([(ContextualHistoricalConv(hidden, bank.shape[1], graph_kernel, uniform_mix)
                                                if history_fusion == "contextual" else HistoricalLagConv(hidden, bank.shape[1], graph_kernel,
                                                history_fusion, history_padding, lag_normalization, history_init, uniform_mix,
                                                lag_readout, history_weight_sharing)
                                                if graph_kernel > 1 else LaggedPCMCIConv(hidden, bank.shape[1], lag_normalization, lag_readout))
                                                for _ in range(blocks*layers)])
            if causal_revision == "selective":
                self.injections = nn.ModuleList([SelectiveCausalInjection(hidden) for _ in range(blocks*layers)])

    def forward(self, inputs, graph_ids=None):
        b = self.backbone
        if self.mode == 'custom':
            b.supports = [getattr(self, f'custom_support_{i}') for i in range(self.custom_support_count)]
        if self.mode in ('adaptive', 'custom'):
            return b(inputs)[..., -1].index_select(2, self.flow_indices)
        if graph_ids is None or graph_ids.shape != (inputs.shape[0], inputs.shape[-1]):
            raise ValueError('PCMCI graph IDs must align with input timestamps')
        signed_bank, signed_static, prepared_history = None, None, None
        if self.causal_execution == 'persistent_cache':
            signed_bank, prepared_history = self.execution_cache.get(self.graph_bank,
                self.graph_kernel, self.history_padding, self.history_fusion == 'uniform' and self.graph_kernel > 1)
        if self.causal_execution == 'signed_cache':
            signed_bank = split_signed_bank(self.graph_bank)
            if self.static_weight and hasattr(self, 'static_graph_bank'):
                signed_static = split_signed_bank(self.static_graph_bank)
        padding = max(0, self.receptive_field-inputs.shape[-1])
        x = b.start_conv(F.pad(inputs, (padding, 0)))
        ids = F.pad(graph_ids, (padding, 0))
        # Padded time positions must not become artificial lagged observations
        # through the bias of start_conv.
        if padding:
            x = torch.cat([torch.zeros_like(x[..., :padding]), x[..., padding:]], -1)
        supports = None
        if self.mode == 'combined':
            supports = [self.adaptive_scale * F.softmax(F.relu(b.nodevec1 @ b.nodevec2), dim=1)]
        skip = None
        for i, causal in enumerate(self.causal_layers):
            history = x
            temporal = torch.tanh(b.filter_convs[i](history))*torch.sigmoid(b.gate_convs[i](history))
            length = temporal.shape[-1]
            if prepared_history is not None:
                message = causal(history, self.graph_bank, ids, length, signed_bank, prepared_history)
            else:
                message = causal(history, self.graph_bank, ids, length, signed_bank)
            if self.dynamic_message_scale != 1:
                # Disable dynamic messages without changing the graph archive,
                # graph IDs, static availability, or Graph-MT reliability inputs.
                message = message * self.dynamic_message_scale
            if self.static_weight:
                if hasattr(self, 'static_matrix'):
                    fixed = self.static_layers[i](history, self.static_matrix,
                                                  ids >= self.static_min_graph_id, length)
                else:
                    fixed = causal(history, self.static_graph_bank, (ids > 0).long(), length, signed_static)
                # The legacy first-graph control shares parameters. An external
                # spatial matrix uses its own projection; report that capacity.
                available = ids >= self.static_min_graph_id if hasattr(self, 'static_matrix') else ids > 0
                effective = self.static_weight * available[:, None, None, -length:]
                if self.static_fusion == 'gated_residual':
                    message = message + effective*self.static_residual_logits[i].tanh()*fixed
                else:
                    message = (1-effective)*message + effective*fixed
            # Causal information reaches both the hidden state and the SAME
            # layer's skip path, so even a one-layer model uses the graph.
            if self.causal_revision == "selective":
                message = self.injections[i](temporal, message)
            s = b.skip_convs[i](temporal + message)
            skip = s if skip is None else skip[..., -length:] + s
            spatial = b.gconv[i](temporal, supports) if supports is not None else b.residual_convs[i](temporal)
            x = b.bn[i](spatial + message + history[..., -length:])
            ids = ids[..., -length:]
        output = b.end_conv_2(F.relu(b.end_conv_1(F.relu(skip))))
        return output[..., -1].index_select(2, self.flow_indices)
