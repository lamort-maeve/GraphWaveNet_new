"""Pi-model consistency, adapted to continuous forecasting.

Laine & Aila, Temporal Ensembling for Semi-Supervised Learning (ICLR 2017),
https://arxiv.org/abs/1610.02242 . Two stochastic student passes, no EMA teacher.
Unlike the classification formulation, compare standardized continuous outputs;
apply consistency only to withheld forecast origins to match this label protocol.
Both passes receive gradients. No classification confidence threshold is used.
"""
import torch


def perturb(inputs, noise):
    return inputs + torch.randn_like(inputs)*noise


def consistency_loss(first, second, labeled):
    """MSE in normalized target units; no true target values are accepted."""
    mask = (~labeled).to(first.device, non_blocking=True).expand_as(first)
    return torch.where(mask, (first-second).square(), torch.zeros_like(first)).sum() / mask.sum().clamp_min(1)
