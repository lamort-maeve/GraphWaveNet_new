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


# Reference GraphWaveNet: verbatim computation, only class identifiers renamed.
# Source SHA256: f4a1692171791746cf103f29e63c74d637b49945f4a4cf082b037d7e42550f86
class ReferenceNConv(nn.Module):
    def __init__(self):
        super(ReferenceNConv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class ReferenceLinear(nn.Module):
    def __init__(self,c_in,c_out):
        super(ReferenceLinear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class ReferenceGCN(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(ReferenceGCN,self).__init__()
        self.ReferenceNConv = ReferenceNConv()
        c_in = (order*support_len+1)*c_in
        self.mlp = ReferenceLinear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.ReferenceNConv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.ReferenceNConv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class ReferenceGraphWaveNet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(ReferenceGraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1




        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(ReferenceGCN(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x





