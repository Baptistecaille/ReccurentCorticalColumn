import torch
import torch.nn as nn

class LeafProjection(nn.Module):
    def __init__(self, dim_hidden: int, dim_U: int):
        super().__init__()
        self.proj = nn.Linear(dim_hidden, dim_U)

    def forward(self, B_v: torch.Tensor) -> torch.Tensor:
        """B_v: (batch, dim_hidden) -> U_v: (batch, dim_U)"""
        return self.proj(B_v)


class RecursiveIntegration(nn.Module):
    def __init__(self, dim_hidden: int, dim_U: int, n: int):
        """
        Agg_v : concat des n porteurs (n * dim_U) -> projection vers dim_U
        G_v   : concat(B_v, S_v) -> projection vers dim_U
        """
        super().__init__()
        self.agg_v = nn.Linear(n * dim_U, dim_U)
        self.g_v = nn.Linear(dim_hidden + dim_U, dim_U)

    def forward(self, B_v: torch.Tensor, U_children: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            B_v: (batch, dim_hidden)
            U_children: liste de n tenseurs (batch, dim_U)
        Returns:
            U_v: (batch, dim_U)
        """
        U_stacked = torch.cat(U_children, dim=-1)
        S_v = self.agg_v(U_stacked)
        
        combined = torch.cat([B_v, S_v], dim=-1)
        U_v = self.g_v(combined)
        return U_v


class LatentOutputProjection(nn.Module):
    def __init__(self, dim_U: int, dim_target: int):
        super().__init__()
        self.proj = nn.Linear(dim_U, dim_target)

    def forward(self, U_root: torch.Tensor) -> torch.Tensor:
        """U_root: (batch, dim_U) -> z_hat_t: (batch, dim_target)"""
        return self.proj(U_root)
