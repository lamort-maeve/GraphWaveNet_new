"""V2: causal graphs stay a time tensor; a causal 1D convolution mixes their history."""
import torch
from torch import nn
from torch.nn import functional as F
from causal_model_v1 import lag_message, DynamicGraphWaveNet as V1GraphWaveNet, ReferenceGraphWaveNet


def temporal_change(x):
    """Causal first difference. The first step uses a zero past, never a future step."""
    return x - F.pad(x[..., :-1], (1, 0))


class MultiScaleTemporal(nn.Module):
    """Node-local causal filters with state-dependent selection of time scales."""
    def __init__(self, channels):
        super().__init__()
        self.dilations = (1, 2, 4, 8)
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels, (1, 3), dilation=(1, d), groups=channels)
            for d in self.dilations])
        self.select = nn.Conv2d(2 * channels, len(self.dilations), 1)
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate = nn.Conv2d(2 * channels, channels, 1)
        nn.init.zeros_(self.select.weight)
        nn.init.zeros_(self.select.bias)
        nn.init.normal_(self.project.weight, std=0.001)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.)

    def forward(self, x):
        state = torch.cat([x, temporal_change(x)], dim=1)
        weights = self.select(state).softmax(dim=1)
        mixed = sum(weights[:, i:i+1] * torch.tanh(branch(F.pad(x, (2*d, 0))))
                    for i, (d, branch) in enumerate(zip(self.dilations, self.branches)))
        return x + torch.sigmoid(self.gate(state)) * self.project(mixed)


class GatedLagConv(nn.Module):
    """Per-lag/sign transforms; no new edges outside the PCMCI graph."""
    def __init__(self, channels, lag_count, causal, adaptive, dropout, num_nodes):
        super().__init__()
        self.causal, self.adaptive, self.dropout = causal, adaptive, dropout
        self.local = nn.Conv2d(channels, channels, 1)
        if causal:
            self.lag_project = nn.Conv2d(2*lag_count*channels, channels, 1, bias=False)
            self.causal_gate = nn.Conv2d(3*channels, channels, 1)
            nn.init.constant_(self.causal_gate.bias, -1.)
            nn.init.zeros_(self.causal_gate.weight[:, 2*channels:])
            self.causal_node_bias = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))
        if adaptive:
            self.adaptive_project = nn.Conv2d(2*channels, channels, 1, bias=False)
            self.adaptive_gate = nn.Conv2d(3*channels, channels, 1)
            nn.init.constant_(self.adaptive_gate.bias, -1.)
            nn.init.zeros_(self.adaptive_gate.weight[:, 2*channels:])
            self.adaptive_node_bias = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))

    def causal_message(self, x, bank, graph_ids, prepared=None):
        messages = []
        for lag in range(bank.shape[1]):
            a = bank[:, lag][graph_ids]
            messages.append(lag_message(x, a.clamp_min(0), lag))
            messages.append(-lag_message(x, (-a).clamp_min(0), lag))
        return self.lag_project(torch.cat(messages, dim=1))

    def _gated(self, gate, node_bias, state, message):
        if node_bias.shape[2] not in (1, state.shape[2]):
            raise ValueError('Graph gate nodes do not match the hidden state')
        opened = torch.sigmoid(gate(torch.cat([state, message, temporal_change(state)], 1)) + node_bias)
        return opened * message

    def forward(self, x, bank, graph_ids, adaptive_adj, prepared=None):
        out = self.local(x)
        if self.causal:
            message = self.causal_message(x, bank, graph_ids, prepared)
            out = out + self._gated(self.causal_gate, self.causal_node_bias, x, message)
        if self.adaptive:
            first = torch.einsum('bcst,sd->bcdt', x, adaptive_adj)
            second = torch.einsum('bcst,sd->bcdt', first, adaptive_adj)
            message = self.adaptive_project(torch.cat([first, second], 1))
            out = out + self._gated(self.adaptive_gate, self.adaptive_node_bias, x, message)
        return F.dropout(out, self.dropout, self.training)


class TemporalTensorConv(GatedLagConv):
    """Stack the latest causal graphs and apply a 1D convolution along that time axis.

    The convolution only sees the graph active now and older graphs. An edge that is
    zero throughout that history stays zero, so the layer cannot invent a PCMCI edge.
    """
    def __init__(self, channels, lag_count, causal, adaptive, dropout, kernel_size=3, num_nodes=1,
                 history_fusion='signed_convex'):
        super().__init__(channels, lag_count, causal, adaptive, dropout, num_nodes)
        if history_fusion not in ('signed_convex', 'signed_conv'):
            raise ValueError(history_fusion)
        self.history_fusion = history_fusion
        if kernel_size < 1:
            raise ValueError(kernel_size)
        self.causal, self.adaptive, self.dropout = causal, adaptive, dropout
        self.kernel_size = kernel_size
        if causal:
            self.graph_conv = nn.Conv1d(lag_count, lag_count, kernel_size, groups=lag_count, bias=False)
            nn.init.zeros_(self.graph_conv.weight)
            with torch.no_grad():
                # Convex mode starts with 80% on the current snapshot (K > 1).
                self.graph_conv.weight[:, 0, -1] = (
                    torch.log(torch.tensor(4. * (kernel_size - 1))).item()
                    if history_fusion == 'signed_convex' and kernel_size > 1 else 1.)

    def evolved_adjacency(self, bank_lag, graph_ids, lag):
        width = self.kernel_size
        # Many samples/time steps share a graph. Convolve each active graph once,
        # then gather it back; avoid retaining B*T*K*N*N convolution inputs.
        unique_ids, inverse = torch.unique(graph_ids, sorted=True, return_inverse=True)
        history = [(unique_ids - age).clamp_min(0) for age in range(width - 1, -1, -1)]
        stacked = torch.stack([bank_lag[idx] for idx in history], dim=-1)
        flat = stacked.reshape(-1, 1, width)
        kernel = self.graph_conv.weight[lag, 0].view(1, 1, width)
        if self.history_fusion == 'signed_convex':
            kernel = kernel.softmax(dim=-1)
        evolved = F.conv1d(flat, kernel).reshape(stacked.shape[:-1])
        return evolved[inverse.reshape(-1)].reshape(*graph_ids.shape, *bank_lag.shape[-2:])

    def causal_message(self, x, bank, graph_ids, prepared=None):
        messages = []
        prepared = self.prepare_history(graph_ids) if prepared is None else prepared
        for lag in range(bank.shape[1]):
            positive, negative = self.signed_adjacency(bank[:, lag], graph_ids, lag, prepared)
            messages.append(lag_message(x, positive, lag))
            messages.append(-lag_message(x, negative, lag))
        return self.lag_project(torch.cat(messages, dim=1))

    def prepare_history(self, graph_ids, graph_lookup=None):
        unique, inverse = (torch.unique(graph_ids, sorted=True, return_inverse=True)
                           if graph_lookup is None else graph_lookup)
        ages = torch.arange(self.kernel_size - 1, -1, -1, device=graph_ids.device)
        # This dictionary lives for ONE forward only. Cache constant history,
        # never trainable mixtures or feature-dependent attention.
        return (unique[:, None] - ages).clamp_min(0), inverse.reshape(-1), {}

    def signed_adjacency(self, bank_lag, graph_ids, lag, prepared):
        indices, inverse, cache = prepared
        # Gather ONLY the few windows needed by this batch before splitting signs.
        # Previously every layer/lag clamped the entire year's graph bank twice.
        if lag not in cache:
            history = bank_lag[indices]
            cache[lag] = (history, history.clamp_min(0), (-history).clamp_min(0))
        history, positive_history, negative_history = cache[lag]
        kernel = self.graph_conv.weight[lag, 0]
        if self.history_fusion == 'signed_convex':
            kernel = kernel.softmax(-1)
            positive = torch.einsum('uksd,k->usd', positive_history, kernel)
            negative = torch.einsum('uksd,k->usd', negative_history, kernel)
        else:
            adjacency = torch.einsum('uksd,k->usd', history, kernel)
            positive, negative = adjacency.clamp_min(0), (-adjacency).clamp_min(0)
        shape = (*graph_ids.shape, *bank_lag.shape[-2:])
        return positive[inverse].reshape(shape), negative[inverse].reshape(shape)

class PCMCIRefinedConv(TemporalTensorConv):
    """State-conditioned reweighting only inside the historical PCMCI support."""
    def __init__(self, channels, lag_count, dropout, num_nodes, kernel_size=3,
                 history_fusion='signed_convex', adaptive=False):
        super().__init__(channels, lag_count, True, adaptive, dropout, kernel_size,
                         num_nodes, history_fusion)
        attention_channels = min(channels, 8)
        self.query = nn.Conv2d(channels, attention_channels, 1, bias=False)
        self.key = nn.Conv2d(channels, attention_channels, 1, bias=False)
        self.lag_select = nn.Conv2d(channels, lag_count, 1)
        self.lag_node_bias = nn.Parameter(torch.zeros(1, lag_count, num_nodes, 1))
        nn.init.zeros_(self.lag_select.weight)
        nn.init.zeros_(self.lag_select.bias)

    def causal_message(self, x, bank, graph_ids, prepared=None):
        query = self.query(x)
        key = self.key(x)
        # Independent bounded gates, not a softmax that forces distinct lags
        # to compete even when several delays carry useful information.
        lag_weights = 2. * torch.sigmoid(self.lag_select(x) + self.lag_node_bias)
        messages = []
        prepared = self.prepare_history(graph_ids) if prepared is None else prepared
        for lag in range(bank.shape[1]):
            positive, negative = self.signed_adjacency(bank[:, lag], graph_ids, lag, prepared)
            past_key = (torch.zeros_like(key) if lag >= key.shape[-1] else
                        F.pad(key[..., :-lag], (lag, 0)) if lag else key)
            score = torch.einsum('bcst,bcdt->btsd', past_key, query) / query.shape[1] ** 0.5
            # Multiplication preserves zeros/signs. No free all-to-all adjacency.
            edge_weight = 2. * torch.sigmoid(score)
            selected = lag_weights[:, lag:lag+1]
            messages.append(selected * lag_message(x, positive * edge_weight, lag))
            messages.append(-selected * lag_message(x, negative * edge_weight, lag))
        return self.lag_project(torch.cat(messages, dim=1))


class CausalReadout(nn.Module):
    """One graph hop from the shallow state directly to the forecast correction."""
    def __init__(self, channels, lag_count, horizon, num_nodes, kernel_size, history_fusion):
        super().__init__()
        self.graph = PCMCIRefinedConv(channels, lag_count, 0., num_nodes, kernel_size, history_fusion)
        # This branch carries graph messages only: no local path or constant bias.
        del self.graph.local
        self.output = nn.Conv2d(channels, horizon, 1, bias=False)
        nn.init.normal_(self.output.weight, std=0.001)

    def forward(self, state, bank, graph_ids, prepared=None):
        message = self.graph.causal_message(state, bank, graph_ids, prepared)
        message = self.graph._gated(self.graph.causal_gate, self.graph.causal_node_bias, state, message)
        return self.output(message)[..., -1]


class DynamicGraphWaveNet(V1GraphWaveNet):
    def __init__(self, num_variables, horizon, flow_indices, bank, mode='causal',
                 hidden=32, blocks=4, layers=2, dropout=0.3,
                 context=25, graph_operator='temporal_tensor', persistence=True, variable_embedding=True,
                 graph_kernel=3, multiscale=True, graph_history_fusion='signed_convex',
                 pcmci_refinement=True, causal_readout=True):
        super().__init__(num_variables, horizon, flow_indices, bank, mode, hidden, blocks, layers, dropout)
        if graph_operator not in ('temporal_tensor', 'gated_lag', 'v1'):
            raise ValueError(graph_operator)
        self.context, self.persistence = context, persistence
        self.graph_operator = graph_operator
        if graph_operator == 'temporal_tensor':
            self.graphs = nn.ModuleList([
                TemporalTensorConv(hidden, bank.shape[1], mode != 'adaptive', mode != 'causal', dropout,
                                   kernel_size=graph_kernel, num_nodes=num_variables,
                                   history_fusion=graph_history_fusion)
                for _ in self.dilations])
            self.graph_kernel = self.graphs[0].kernel_size
        elif graph_operator == 'gated_lag':
            self.graphs = nn.ModuleList([
                GatedLagConv(hidden, bank.shape[1], mode != 'adaptive', mode != 'causal', dropout, num_variables)
                for _ in self.dilations])
        self._cover_context(hidden)
        self.multiscale = MultiScaleTemporal(hidden) if multiscale else nn.Identity()
        if multiscale:
            self.receptive_field += 16
        self.variable_embedding = nn.Parameter(torch.zeros(1, hidden, num_variables, 1)) if variable_embedding else None
        if variable_embedding:
            nn.init.normal_(self.variable_embedding, std=0.02)
        self.time_mix = nn.Conv2d(self.end1.in_channels, 1, 1, bias=False)
        nn.init.zeros_(self.time_mix.weight)
        self.recency = nn.Parameter(torch.tensor(4.0))
        self.temporal_readout = nn.Linear(context, horizon, bias=False)
        nn.init.zeros_(self.temporal_readout.weight)
        self.road_gain = nn.Parameter(torch.ones(int(self.flow_indices.numel())))
        nn.init.normal_(self.end2.weight, std=0.001)
        nn.init.zeros_(self.end2.bias)
        # Construct extras after the existing backbone and isolate RNG use.
        # Enabling these modules does not change adaptive baseline initialization.
        self.causal_readout = None
        if mode != 'adaptive' and graph_operator == 'temporal_tensor':
            with torch.random.fork_rng(devices=[]):
                if pcmci_refinement:
                    refined = nn.ModuleList()
                    for old in self.graphs:
                        layer = PCMCIRefinedConv(hidden, bank.shape[1], dropout, num_variables,
                                                graph_kernel, graph_history_fusion, adaptive=mode == 'combined')
                        # Preserve local/graph projections and optional adaptive
                        # branch exactly; attach only PCMCI-specific selectors.
                        layer.load_state_dict(old.state_dict(), strict=False)
                        refined.append(layer)
                    self.graphs = refined
                if causal_readout:
                    self.causal_readout = CausalReadout(hidden, bank.shape[1], horizon,
                                                       num_variables, graph_kernel, graph_history_fusion)

    def _cover_context(self, hidden):
        """Widen the causal temporal kernel until its receptive field covers the context."""
        span = int(sum(self.dilations))
        kernel = 2
        while 1 + (kernel - 1) * span < self.context:
            kernel += 1
        self.temporal_kernel = kernel
        self.receptive_field = 1 + (kernel - 1) * span
        if kernel == 2:
            return
        filters, gates = nn.ModuleList(), nn.ModuleList()
        for dilation in self.dilations:
            filters.append(nn.Conv2d(hidden, hidden, (1, kernel), dilation=(1, dilation)))
            gates.append(nn.Conv2d(hidden, hidden, (1, kernel), dilation=(1, dilation)))
        self.filters, self.gates = filters, gates

    def _pool_skip(self, skip):
        length = skip.shape[-1]
        position = torch.linspace(0, 1, length, device=skip.device, dtype=skip.dtype)
        score = self.time_mix(skip) + self.recency * position.view(1, 1, 1, length)
        weight = torch.softmax(score, dim=-1)
        return (skip * weight).sum(dim=-1, keepdim=True)

    def forward(self, inputs, graph_ids, graph_lookup=None):
        if graph_ids.shape != (inputs.shape[0], inputs.shape[-1]):
            raise ValueError('Graph IDs must align with every input time step')
        x = self.start(inputs)
        if self.variable_embedding is not None:
            x = x + self.variable_embedding
        x = self.multiscale(x)
        shallow = x
        adaptive = None
        if self.mode != 'causal':
            adaptive = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)
        skip = None
        prepared = (self.graphs[0].prepare_history(graph_ids, graph_lookup)
                    if self.mode != 'adaptive' and self.graph_operator == 'temporal_tensor' else None)
        for dilation, filt, gate, graph, proj, norm in zip(
                self.dilations, self.filters, self.gates, self.graphs, self.skips, self.norms):
            residual = x
            padded = F.pad(x, (dilation * (self.temporal_kernel - 1), 0))
            filtered = filt(padded)
            if filtered.shape[-1] != residual.shape[-1]:
                raise RuntimeError('Temporal convolution changed the sequence length')
            x = torch.tanh(filtered) * torch.sigmoid(gate(padded))
            if isinstance(graph, GatedLagConv):
                x = graph(x, self.graph_bank, graph_ids, adaptive, prepared)
            else:
                x = graph(x, self.graph_bank, graph_ids, adaptive)
            contribution = proj(x)
            skip = contribution if skip is None else skip + contribution
            x = norm((x+residual).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.end2(F.relu(self.end1(F.relu(self._pool_skip(skip)))))[..., 0]
        if self.causal_readout is not None:
            out = out + self.causal_readout(shallow, self.graph_bank, graph_ids, prepared)
        out = out.index_select(2, self.flow_indices)
        flow = inputs[:, 0].index_select(1, self.flow_indices)
        history = flow[..., -self.context:]
        if history.shape[-1] < self.context:
            history = F.pad(history, (self.context-history.shape[-1], 0))
        reference = flow[..., -1:]
        correction = (out + self.temporal_readout(history-reference).transpose(1, 2)) * self.road_gain.view(1, 1, -1)
        if self.persistence:
            correction = correction + reference.transpose(1, 2)
        return correction


class ForecastPCMCI(nn.Module):
    """Predict target(t+h) using PCMCI source(t+h-lag), never future observations.

    Graph snapshots are restricted to those available at forecast origin t.
    The temporal kernel operates on graph-window history, separately by sign.
    """
    def __init__(self, lag_count, horizon_offsets, output_nodes, hidden=16, kernel=3):
        super().__init__()
        self.offsets = tuple(int(h) for h in horizon_offsets)
        self.lag_count, self.kernel = lag_count, kernel
        self.register_buffer('output_nodes', torch.as_tensor(output_nodes, dtype=torch.long))
        self.history_logits = nn.Parameter(torch.zeros(lag_count, kernel))
        if kernel > 1:
            with torch.no_grad():
                self.history_logits[:, -1] = torch.log(torch.tensor(4. * (kernel-1)))
        self.transforms = nn.ModuleList([
            nn.Sequential(nn.Linear(4*lag_count, hidden, bias=False), nn.Tanh(),
                          nn.Linear(hidden, 1, bias=False)) for _ in self.offsets])
        self.gates = nn.ModuleList([nn.Linear(3, 1) for _ in self.offsets])
        for transform, gate in zip(self.transforms, self.gates):
            nn.init.zeros_(transform[-1].weight)
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -1.)

    def messages(self, inputs, bank, graph_ids):
        # Only the origin graph and its past are used; lag 0 / lag<h cannot
        # identify an observed parent of the forecast and are not imputed.
        ids = graph_ids[:, -1]
        ages = torch.arange(self.kernel-1, -1, -1, device=ids.device)
        indices = (ids[:, None] - ages).clamp_min(0)
        history = bank[indices].index_select(-1, self.output_nodes)  # B,K,L,S,D
        weights = self.history_logits.softmax(-1)
        pos = torch.einsum('bklsd,lk->blsd', history.clamp_min(0), weights)
        neg = torch.einsum('bklsd,lk->blsd', (-history).clamp_min(0), weights)
        source = inputs[:, 0]
        results = []
        for offset in self.offsets:
            channels = []
            for lag in range(self.lag_count):
                index = source.shape[-1]-1+offset-lag
                if lag < offset or index < 0:
                    zero = source.new_zeros(source.shape[0], self.output_nodes.numel())
                    channels.extend([zero, zero, zero, zero])
                    continue
                level = source[..., index]
                delta = level-source[..., index-1] if index > 0 else torch.zeros_like(level)
                channels.extend([torch.einsum('bs,bsd->bd', level, pos[:, lag]),
                                 -torch.einsum('bs,bsd->bd', level, neg[:, lag]),
                                 torch.einsum('bs,bsd->bd', delta, pos[:, lag]),
                                 -torch.einsum('bs,bsd->bd', delta, neg[:, lag])])
            results.append(torch.stack(channels, -1))
        return results

    def forward(self, inputs, bank, graph_ids, baseline):
        messages = self.messages(inputs, bank, graph_ids)
        flow = inputs[:, 0].index_select(1, self.output_nodes)
        last = flow[..., -1]
        change = last-flow[..., -2] if flow.shape[-1] > 1 else torch.zeros_like(last)
        residuals = []
        for i, (message, transform, gate) in enumerate(zip(messages, self.transforms, self.gates)):
            context = torch.stack([last, change, baseline[:, i].detach()], -1)
            residuals.append(transform(message).squeeze(-1) * torch.sigmoid(gate(context).squeeze(-1)))
        return torch.stack(residuals, 1)


class ReferencePCMCIModel(nn.Module):
    """Original adaptive GraphWaveNet, with PCMCI confined to an external branch."""
    def __init__(self, num_variables, horizon, flow_indices, bank, mode='adaptive',
                 hidden=32, blocks=4, layers=2, dropout=0.3, horizon_offsets=None,
                 graph_kernel=3):
        super().__init__()
        if mode not in ('adaptive', 'causal', 'combined'):
            raise ValueError(mode)
        self.mode, self.frozen_backbone = mode, False
        self.register_buffer('flow_indices', torch.as_tensor(flow_indices, dtype=torch.long))
        # No changes to the reference adaptive adjacency, graph convolution,
        # temporal stack, BatchNorm, skip connections or output head.
        self.backbone = ReferenceGraphWaveNet(torch.device('cpu'), num_variables,
            dropout=dropout, supports=None, gcn_bool=mode != 'causal', addaptadj=mode != 'causal',
            in_dim=1, out_dim=horizon, residual_channels=hidden, dilation_channels=hidden,
            skip_channels=hidden*8, end_channels=hidden*16, blocks=blocks, layers=layers)
        self.receptive_field = self.backbone.receptive_field
        self.pcmci = None
        if mode != 'adaptive':
            self.register_buffer('graph_bank', torch.as_tensor(bank, dtype=torch.float32))
            offsets = list(range(1, horizon+1)) if horizon_offsets is None else list(horizon_offsets)
            if len(offsets) != horizon or min(offsets) < 1:
                raise ValueError('Invalid forecast offsets')
            self.pcmci = ForecastPCMCI(bank.shape[1], offsets, flow_indices, hidden=min(hidden, 16), kernel=graph_kernel)

    def freeze_backbone(self):
        self.frozen_backbone = True
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_backbone:
            self.backbone.eval()  # Keep BatchNorm statistics and dropout fixed too.
        return self

    def forward(self, inputs, graph_ids=None):
        baseline = self.backbone(inputs)[..., -1].index_select(2, self.flow_indices)
        if self.pcmci is None:
            return baseline
        if graph_ids is None or graph_ids.shape != (inputs.shape[0], inputs.shape[-1]):
            raise ValueError('PCMCI graph IDs must align with input times')
        return baseline + self.pcmci(inputs, self.graph_bank, graph_ids, baseline)


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
                 hidden=32, blocks=4, layers=2, dropout=0.3):
        super().__init__()
        if mode not in ('adaptive', 'causal', 'combined'):
            raise ValueError(mode)
        self.mode = mode
        self.register_buffer('flow_indices', torch.as_tensor(flow_indices, dtype=torch.long))
        self.backbone = ReferenceGraphWaveNet(torch.device('cpu'), num_variables,
            dropout=dropout, supports=None, gcn_bool=mode != 'causal', addaptadj=mode != 'causal',
            in_dim=1, out_dim=horizon, residual_channels=hidden, dilation_channels=hidden,
            skip_channels=hidden*8, end_channels=hidden*16, blocks=blocks, layers=layers)
        self.receptive_field = self.backbone.receptive_field
        if mode != 'adaptive':
            self.register_buffer('graph_bank', torch.as_tensor(bank, dtype=torch.float32))
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
            # Causal information reaches both the hidden state and the SAME
            # layer's skip path, so even a one-layer model uses the graph.
            s = b.skip_convs[i](temporal + message)
            skip = s if skip is None else skip[..., -length:] + s
            spatial = b.gconv[i](temporal, supports) if supports is not None else b.residual_convs[i](temporal)
            x = b.bn[i](spatial + message + history[..., -length:])
            ids = ids[..., -length:]
        output = b.end_conv_2(F.relu(b.end_conv_1(F.relu(skip))))
        return output[..., -1].index_select(2, self.flow_indices)
