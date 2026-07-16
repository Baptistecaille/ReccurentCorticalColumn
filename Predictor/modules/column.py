import torch
import torch.nn as nn
from .modulation import FiLMModulation


def _lateral_attention(W_q, W_k, W_v, H_self: torch.Tensor, H_others: list[torch.Tensor]) -> torch.Tensor:
    if not H_others:
        return torch.zeros_like(H_self)
    q = W_q(H_self)
    K = torch.stack([W_k(h) for h in H_others], dim=1)
    V = torch.stack([W_v(h) for h in H_others], dim=1)
    attn = torch.softmax(
        torch.einsum("bd,bnd->bn", q, K) / (q.shape[-1] ** 0.5), dim=-1
    )
    return torch.einsum("bn,bnd->bd", attn, V)

class ColumnStep(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 2):
        
        super().__init__()
        hidden_dim = dim * hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        

    def forward(self, h: torch.torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, dim)
        Returns:
            (batch, dim), même shape (résiduel)
        """
        return h + self.mlp(h)
    
    
class CorticalColumn(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_feedback: int,
        is_root: bool,
        dim_k: int | None = None,
    ):
        """
        is_root: si True, cette colonne n'a pas de sœurs ni de parent : A_v et
                 R_v sont toujours None (pas de modulation)[cite: 104].
        dim_k: dimension des clés/requêtes de l'attention latérale (eq. 5).
        """
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

    def encode(self, X_v: torch.Tensor) -> torch.Tensor:
        """H_v = E_v(X_v), eq. (3) [cite: 77]. Appelé une seule fois par colonne."""
        return self.E_v(X_v)

    def lateral_attention(self, H_self: torch.Tensor, H_others: list[torch.Tensor]) -> torch.Tensor:
        """
        Attention latérale eq. (5), avec LES PROJECTIONS DE CETTE COLONNE
        appliquées à ses sœurs (W_q/W_k/W_v indexées par la colonne
        destinataire, pas partagées entre sœurs).
        """
        return _lateral_attention(self.W_q, self.W_k, self.W_v, H_self, H_others)

    def forward(
        self,
        H_v: torch.Tensor,
        A_v: torch.Tensor | None,
        R_v: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Args:
            H_v: (batch, dim_hidden)        — sortie de encode(X_v), eq. (3)
            A_v: (batch, dim_hidden)|None   — contexte latéral, eq. (5) [cite: 62, 86]
            R_v: (batch, dim_feedback)|None — feedback du parent, eq. (6) [cite: 62, 93]
        Returns:
            B_v: (batch, dim_hidden) [cite: 62]
        """
        term = H_v
        if A_v is not None:
            term = term + self.mod_lat(H_v, A_v)
        if R_v is not None:
            term = term + self.mod_fb(H_v, R_v)

        return self.step(term)  # eq. (8) [cite: 101, 103]