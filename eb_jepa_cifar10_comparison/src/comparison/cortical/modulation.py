"""FiLM modulation copied from ``Predictor/modules/modulation.py``.

The equations and layer configuration are unchanged.
"""

import torch
import torch.nn as nn


class FiLMModulation(nn.Module):
    def __init__(self, dim_h: int, dim_c: int) -> None:
        super().__init__()
        self.dim_h = dim_h
        self.dim_c = dim_c
        self.gamma = nn.Linear(dim_c, dim_h)
        self.beta = nn.Linear(dim_c, dim_h)
        self.ln = nn.LayerNorm(dim_h, elementwise_affine=False)

    def forward(self, H: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(C)
        beta = self.beta(C)
        return gamma * self.ln(H) + beta
