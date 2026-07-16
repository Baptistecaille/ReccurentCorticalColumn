import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass

class HingeStdLoss(nn.Module):
    """ Compute the std loss """

    def __init__(self, target: float = 1.0, epsilon: float = 1e-4):
        super().__init__()
        self.target = target
        self.epsilon = epsilon

    def forward(self, x:torch.Tensor):

        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + self.epsilon)
        std_loss = torch.mean(F.relu(self.target - std))
        return std_loss


class CovarianceLoss(torch.nn.Module):
    def __init__(self):
        """
        Penalizes off-diagonal elements of the covariance matrix to encourage
        feature decorrelation.

        Normalizes by D * (D - 1) where D is feature dimensionality.
        """
        super().__init__()

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        """
        batch_size = x.shape[0]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)  # [D, D]
        # Calculate off-diagonal loss
        cov_loss = self.off_diagonal(cov).pow(2).mean() 

        return cov_loss
    
class VICRegLoss(nn.Module):
    def __init__(
        self,
        invariance_coeff: float = 1.0,
        std_coeff: float = 1.0,
        cov_coeff: float = 80.0,
        variance_target: float = 1.0,
        variance_epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        self.invariance_coeff = invariance_coeff
        self.variance_target = variance_target
        self.variance_epsilon = variance_epsilon
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.std_loss_fn = HingeStdLoss(
            target=variance_target,
            epsilon=variance_epsilon,
        )
        self.cov_loss_fn = CovarianceLoss()

    def components(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the total VICReg loss and its unweighted components.

        Args:
            z1: [B, D] - First projection tensor
            z2: [B, D] - Second projection tensor

        Returns:
            dict with keys: loss, invariance_loss, var_loss, cov_loss
        """
        if z1.shape != z2.shape:
            raise ValueError(
                "z1 and z2 must have the same shape, "
                f"got {z1.shape} and {z2.shape}"
            )

        # Invariance loss (similarity)
        sim_loss = F.mse_loss(z1, z2)

        # Variance loss (applied symmetrically to both views and averaged)
        var_loss = 0.5 * (self.std_loss_fn(z1) + self.std_loss_fn(z2))

        # Covariance loss (applied symmetrically to both views and averaged)
        cov_loss = 0.5 * (self.cov_loss_fn(z1) + self.cov_loss_fn(z2))

        total_loss = (
            self.invariance_coeff * sim_loss
            + self.std_coeff * var_loss
            + self.cov_coeff * cov_loss
        )

        return {
            "loss": total_loss,
            "invariance_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Return the differentiable scalar VICReg objective."""
        return self.components(z1, z2)["loss"]


@dataclass(frozen=True)
class CollapseDiagnostics:
    mean_std: float
    min_std: float
    collapsed_fraction: float


def collapse_diagnostics(representations: torch.Tensor, threshold: float = 0.1) -> CollapseDiagnostics:

    """ Compute collapse diagnostics for a batch of representations."""

    assert threshold > 0, "Threshold must be positive"

    if representations.ndim != 2:
        raise ValueError("representations doit avoir la forme (B, D)")

    B, D = representations.shape

    if B < 2:
        raise ValueError("Le batch doit contenir au moins deux représentations")
    
    values_std = []


    with torch.no_grad():
        for i in range(D):
            values_std.append(representations[:, i].std(dim=0))
        
        values_std = torch.stack(values_std)
        collapsed_fraction = (values_std < threshold).float().mean().item()
    
    return CollapseDiagnostics(mean_std=values_std.mean().item(), min_std=values_std.min().item(), collapsed_fraction=collapsed_fraction)
