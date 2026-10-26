"""Periodic hard-snapshot pseudo-label self-training for regression.

Supervised pretraining precedes pseudo-label use. A frozen eval-mode snapshot
predicts continuous targets on clean histories; the student sees perturbations.
All unlabeled origins are used, with no classification confidence heuristic.
This is a basic self-training adaptation, not a reproduction of FixMatch or
Curriculum Labeling. Snapshot targets never receive gradients or EMA updates.
"""
import torch
from mean_teacher import make_teacher, is_constant_buffer


class SelfTraining:
    def __init__(self, pretrain, refresh):
        self.pretrain, self.refresh = pretrain, refresh
        self.snapshot = None

    @property
    def active(self):
        return self.snapshot is not None

    def start_epoch(self, epoch, model):
        if epoch > self.pretrain and (epoch-self.pretrain-1) % self.refresh == 0:
            # Reuse the graph-sharing copy utility, not Mean Teacher training.
            self.snapshot = make_teacher(model)

    @staticmethod
    def perturb(x, noise):
        return x + torch.randn_like(x)*noise

    def loss(self, output, clean_x, graph_ids, origin, labeled):
        if not self.active:
            return output.new_zeros(())
        with torch.no_grad():
            target = self.snapshot(clean_x, graph_ids)
        mask = (~labeled.to(output.device, non_blocking=True)).expand_as(output)
        return torch.where(mask, (output-target).square(), torch.zeros_like(output)).sum()/mask.sum().clamp_min(1)

    def end_epoch(self):
        pass

    def state_dict(self):
        # Constant graph buffers are already in the student checkpoint.
        return {'snapshot': None if not self.active else {
            k: v for k, v in self.snapshot.state_dict().items() if not is_constant_buffer(k)}}

    def load_state_dict(self, state, model):
        if state['snapshot'] is not None:
            self.snapshot = make_teacher(model)
            complete = self.snapshot.state_dict()
            complete.update(state['snapshot'])
            self.snapshot.load_state_dict(complete, strict=True)
