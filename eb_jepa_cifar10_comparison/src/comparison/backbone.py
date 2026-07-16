"""ResNet-18 backbone adapted from facebookresearch/eb_jepa.

Source: examples/image_jepa/main.py:61-75
The architecture is unchanged from the CIFAR-10 reference.
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
