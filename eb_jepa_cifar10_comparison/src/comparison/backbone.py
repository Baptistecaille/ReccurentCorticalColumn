"""Locally available CIFAR-10 encoder backbones.

Includes a ResNet-18 adapted from facebookresearch/eb_jepa and a lightweight MLP.
"""

import torchvision
from torch import Tensor
from torch import nn


class ResNet18(nn.Module):
    """ResNet-18 backbone implementation."""

    features_dim: int = 512

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()  # Remove final classification layer
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=2, bias=False
        )
        self.backbone.maxpool = nn.Identity()

    def forward(self, images: Tensor) -> Tensor:
        return self.backbone(images)


class MLPEncoder(nn.Module):
    """Lightweight encoder for normalized 32x32 RGB images."""

    features_dim: int = 512

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, self.features_dim),
            nn.LayerNorm(self.features_dim),
            nn.GELU(),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.layers(images)
