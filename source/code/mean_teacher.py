"""Optional training strategy, independent of the forecasting architecture.

No teacher is constructed when SSL is disabled. Validation/test use the student.
The forecast model never imports this module.
"""
import copy
import torch
from metrics import masked_mae

def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model

def is_constant_buffer(name):
    """One definition for teacher sharing, EMA exclusion and checkpoint caching."""
    return name in ('graph_bank', 'static_graph_bank', 'flow_indices')

def make_teacher(model):
    original = unwrap(model)
    memo = {id(v): v for name, v in original.named_buffers() if is_constant_buffer(name)}
    return copy.deepcopy(original, memo).requires_grad_(False).eval()

@torch.no_grad()
def update_teacher(teacher, student, decay):
    for target, source in zip(teacher.parameters(), unwrap(student).parameters()):
        target.lerp_(source, 1.-decay)
    # Copy BN statistics; EMA applies to learned parameters, not archive data.
    for name, source in unwrap(student).named_buffers():
        if not is_constant_buffer(name):
            teacher.get_buffer(name).copy_(source)

def perturb_inputs(inputs, noise):
    """Student-only perturbation, in normalized input units."""
    return inputs + torch.randn_like(inputs)*noise


def consistency_loss(teacher, inputs, graph_ids, prediction, labeled, mean, std):
    """Use teacher predictions, never withheld true targets."""
    with torch.no_grad():
        pseudo = teacher(inputs, graph_ids)*std + mean
    unlabelled = (~labeled).to(prediction.device, non_blocking=True).expand_as(prediction)
    return masked_mae(prediction, pseudo, unlabelled)
