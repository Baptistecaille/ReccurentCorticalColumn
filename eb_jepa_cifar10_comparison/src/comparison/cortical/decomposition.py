"""Tree decomposition copied from ``Predictor/modules/decomposition.py``.

The independent projection per child is unchanged.
"""

import torch
import torch.nn as nn


class ChildDecomposition(nn.Module):
    def __init__(self, dim_parent: int, dim_child: int, n: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(dim_parent, dim_child) for _ in range(n)]
        )

    def forward(self, B_v: torch.Tensor) -> list[torch.Tensor]:
        return [projection(B_v) for projection in self.projections]


class FeedbackProjection(nn.Module):
    def __init__(self, dim_parent: int, dim_feedback: int, n: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(dim_parent, dim_feedback) for _ in range(n)]
        )

    def forward(self, B_v: torch.Tensor) -> list[torch.Tensor]:
        return [projection(B_v) for projection in self.projections]
