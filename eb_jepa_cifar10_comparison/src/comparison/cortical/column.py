"""Cortical column modules copied from ``Predictor/modules/column.py``.

Local adaptation: corrected ``torch.torch.Tensor`` to ``torch.Tensor``. The
attention, modulation and residual equations are unchanged.
"""

import torch
import torch.nn as nn

from .modulation import FiLMModulation


def _lateral_attention(
    W_q: nn.Module,
    W_k: nn.Module,
    W_v: nn.Module,
    H_self: torch.Tensor,
    H_others: list[torch.Tensor],
) -> torch.Tensor:
    if not H_others:
        return torch.zeros_like(H_self)
    q = W_q(H_self)
    keys = torch.stack([W_k(hidden) for hidden in H_others], dim=1)
    values = torch.stack([W_v(hidden) for hidden in H_others], dim=1)
    attention = torch.softmax(
        torch.einsum("bd,bnd->bn", q, keys) / (q.shape[-1] ** 0.5),
        dim=-1,
    )
    return torch.einsum("bn,bnd->bd", attention, values)


class ColumnStep(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 2) -> None:
        super().__init__()
        hidden_dim = dim * hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.mlp(hidden)


class CorticalColumn(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_feedback: int,
        is_root: bool,
        dim_k: int | None = None,
    ) -> None:
        super().__init__()
        self.is_root = is_root
        self.E_v = nn.Linear(dim_in, dim_hidden)
        self.step = ColumnStep(dim_hidden)

        if not is_root:
            self.mod_lat = FiLMModulation(dim_h=dim_hidden, dim_c=dim_hidden)
            self.mod_fb = FiLMModulation(dim_h=dim_hidden, dim_c=dim_feedback)
            dim_k = dim_k or (dim_hidden // 2 if dim_hidden >= 2 else 1)
            self.W_q = nn.Linear(dim_hidden, dim_k, bias=False)
            self.W_k = nn.Linear(dim_hidden, dim_k, bias=False)
            self.W_v = nn.Linear(dim_hidden, dim_hidden, bias=False)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.E_v(inputs)

    def lateral_attention(
        self,
        H_self: torch.Tensor,
        H_others: list[torch.Tensor],
    ) -> torch.Tensor:
        return _lateral_attention(self.W_q, self.W_k, self.W_v, H_self, H_others)

    def forward(
        self,
        H_v: torch.Tensor,
        A_v: torch.Tensor | None,
        R_v: torch.Tensor | None,
    ) -> torch.Tensor:
        term = H_v
        if A_v is not None:
            term = term + self.mod_lat(H_v, A_v)
        if R_v is not None:
            term = term + self.mod_fb(H_v, R_v)
        return self.step(term)
