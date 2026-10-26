"""Laine & Aila (ICLR 2017) temporal ensembling, regression adaptation.

Targets are bias-corrected prediction EMAs indexed by forecast origin, not
batch position. Only completed training epochs update the target bank.
Normalized MSE applies only to unlabeled origins with an existing target.
Banks stay on the training device; checkpointing transfers them once per epoch.
"""
import torch


class TemporalEnsembling:
    active = True

    def __init__(self, size, decay, device):
        self.size, self.decay, self.device = size, decay, device
        self.ensemble = self.count = self.pending = self.seen = None

    def start_epoch(self, epoch, model):
        if self.pending is not None:
            self.pending.zero_()
            self.seen.zero_()

    @staticmethod
    def perturb(x, noise):
        return x + torch.randn_like(x)*noise

    def loss(self, output, clean_x, graph_ids, origin, labeled):
        if self.ensemble is None:
            shape = (self.size, *output.shape[1:])
            self.ensemble = output.new_zeros(shape)
            self.pending = output.new_zeros(shape)
            self.count = torch.zeros(self.size, device=self.device, dtype=torch.long)
            self.seen = torch.zeros_like(self.count)
        ids = origin.to(self.device, non_blocking=True)
        count = self.count[ids]
        correction = (1-self.decay**count).clamp_min(1e-12)
        target = self.ensemble[ids]/correction.view(-1, 1, 1)
        mask = ((~labeled.to(self.device, non_blocking=True)) & (count > 0).view(-1, 1, 1)).expand_as(output)
        loss = torch.where(mask, (output-target).square(), torch.zeros_like(output)).sum()/mask.sum().clamp_min(1)
        with torch.no_grad():
            self.pending.index_add_(0, ids, output.detach())
            self.seen.index_add_(0, ids, torch.ones_like(ids))
        return loss

    @torch.no_grad()
    def end_epoch(self):
        if self.ensemble is None:
            return
        seen = self.seen > 0
        prediction = self.pending/self.seen.clamp_min(1).view(-1, 1, 1)
        self.ensemble[seen] = self.decay*self.ensemble[seen]+(1-self.decay)*prediction[seen]
        self.count[seen] += 1

    def state_dict(self):
        return {'ensemble': self.ensemble, 'count': self.count}

    def load_state_dict(self, state, model=None):
        if state['ensemble'] is not None:
            self.ensemble = state['ensemble'].to(self.device)
            self.count = state['count'].to(self.device)
            if self.ensemble.shape[0] != self.size:
                raise ValueError('Temporal ensemble origin count changed')
            self.pending = torch.zeros_like(self.ensemble)
            self.seen = torch.zeros_like(self.count)
