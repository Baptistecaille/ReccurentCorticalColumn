import torch
import torch.nn as nn
import torch.nn.functional as F

class JEPAPredictorLoss(nn.Module):
    def __init__(self, distance: str = "mse"):
        super().__init__()
        self.distance = distance.lower()
        if self.distance not in ["mse", "cosine", "smooth_l1"]:
            raise ValueError(f"Distance type '{distance}' non supportée.")

    def forward(self, z_hat: torch.Tensor, z_target: torch.Tensor):
        """
        z_target doit déjà être détaché (sg) par l'appelant — cette classe ne
        fait pas le detach elle-même, pour rester explicite sur qui est
        responsable du stop-gradient.
        Returns: scalaire (perte moyenne sur le batch)
        """
        if self.distance == "mse":
            return F.mse_loss(z_hat, z_target)
        elif self.distance == "smooth_l1":
            return F.smooth_l1_loss(z_hat, z_target)
        elif self.distance == "cosine":
            cos_sim = F.cosine_similarity(z_hat, z_target, dim=-1)
            return (1.0 - cos_sim).mean()


def test_no_gradient_into_target():
    z_hat = torch.randn(4, 32, requires_grad=True)
    z_target = torch.randn(4, 32, requires_grad=True)
    loss_fn = JEPAPredictorLoss("mse")
    loss = loss_fn(z_hat, z_target.detach())  # detach fait par l'appelant
    loss.backward()
    assert z_hat.grad is not None
    assert z_target.grad is None


if __name__ == "__main__":
    test_no_gradient_into_target()