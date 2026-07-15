"""Recursive integration copied from ``Predictor/modules/integration.py``.

The leaf, child aggregation and output projection equations are unchanged.
"""

import torch
import torch.nn as nn


class LeafProjection(nn.Module):
    def __init__(self, dim_hidden: int, dim_U: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_hidden, dim_U)

    def forward(self, B_v: torch.Tensor) -> torch.Tensor:
        return self.proj(B_v)


class RecursiveIntegration(nn.Module):
    def __init__(self, dim_hidden: int, dim_U: int, n: int) -> None:
        super().__init__()
        self.agg_v = nn.Linear(n * dim_U, dim_U)
        self.g_v = nn.Linear(dim_hidden + dim_U, dim_U)

    def forward(
        self,
        B_v: torch.Tensor,
        U_children: list[torch.Tensor],
    ) -> torch.Tensor:
        U_stacked = torch.cat(U_children, dim=-1)
        S_v = self.agg_v(U_stacked)
        combined = torch.cat([B_v, S_v], dim=-1)
        return self.g_v(combined)


class LatentOutputProjection(nn.Module):
    def __init__(self, dim_U: int, dim_target: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_U, dim_target)

    def forward(self, U_root: torch.Tensor) -> torch.Tensor:
        return self.proj(U_root)
