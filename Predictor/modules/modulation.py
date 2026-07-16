import torch
import torch.nn as nn

class FiLMModulation(nn.Module):

    def __init__(self, dim_h: int, dim_c: int):
        """
        dim_h: dimension de H (dimension à moduler)
        dim_c: dimension du contexte C (peut différer de dim_h)
        """
        super().__init__()

        self.dim_h = dim_h
        self.dim_c = dim_c
        self.gamma = nn.Linear(dim_c, dim_h)
        self.beta = nn.Linear(dim_c, dim_h)
        self.ln = nn.LayerNorm(dim_h, elementwise_affine=False)

    def forward(self, H: torch.Tensor, C: torch.Tensor):
        
        gamma = self.gamma(C)
        beta = self.beta(C)

        return gamma * self.ln(H) + beta