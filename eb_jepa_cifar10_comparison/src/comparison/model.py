from torch import Tensor, nn

from .backbone import ResNet18
from .heads import build_head
from omegaconf import DictConfig
from dataclasses import dataclass


class ImageSSL(nn.Module):
    def __init__(self, backbone: ResNet18, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        # 1. Calcule les features.
        # 2. Calcule les projections depuis ces mêmes features.
        features = self.backbone(images)
        projections = self.head(features)
        return features, projections
    
def build_model(cfg: DictConfig) -> ImageSSL:
    
    # Backbone
    backbone = ResNet18()

    # Head
    head = build_head(cfg)

    return ImageSSL(backbone=backbone, head=head)

@dataclass(frozen=True)
class ParameterCounts:
    backbone: int
    head: int
    total: int


def count_parameters(model: ImageSSL) -> ParameterCounts:
    

    backbone_total = 0
    head_total = 0

    for parameter in model.backbone.parameters():
        if parameter.requires_grad == True:
            backbone_total += parameter.numel()

    for parameter in model.head.parameters():
        if parameter.requires_grad == True:
            head_total += parameter.numel()
    
    return ParameterCounts(backbone=backbone_total, head=head_total, total=head_total+backbone_total)
