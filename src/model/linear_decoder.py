"""Bias-free linear decoder with unit-norm columns."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnitNormLinearDecoder(nn.Module):
    """Linear map whose effective weight columns always have unit norm."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight_raw, a=5.0**0.5)

    @property
    def weight(self) -> torch.Tensor:
        return F.normalize(self.weight_raw, p=2.0, dim=0, eps=1.0e-12)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, bias=None)


def build_linear_decoder(in_features: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(UnitNormLinearDecoder(in_features, out_features))
