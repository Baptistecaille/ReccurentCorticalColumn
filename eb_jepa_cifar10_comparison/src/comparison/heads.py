"""Representation heads used by the CIFAR-10 comparison.

``MLPProjector`` is extracted from ``examples/image_jepa/main.py:78-102`` in
facebookresearch/eb_jepa. It is converted from an inline
``nn.Sequential`` to a named module without changing its layer order.
``CorticalHead`` and ``build_head`` are local comparison adapters.
"""

from omegaconf import DictConfig
from torch import Tensor, nn

from .cortical import FixedTreePredictor


class MLPProjector(nn.Module):
    """Official EB-JEPA image projector exposed as a named module."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


class CorticalHead(nn.Module):
    """ Check the input and output dimensions of the cortical head. """

    def __init__(
        self,
        predictor: FixedTreePredictor,
        input_dim: int = 512,
        output_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                "CorticalHead expected features with shape "
                f"(batch, {self.input_dim}), received {tuple(features.shape)}"
            )

        projections = self.predictor(features)
        if projections.ndim != 2 or projections.shape[-1] != self.output_dim:
            raise ValueError(
                "FixedTreePredictor returned shape "
                f"{tuple(projections.shape)}; expected (batch, {self.output_dim})"
            )
        return projections


def build_head(cfg: DictConfig) -> nn.Module:
    """Build a fresh baseline or cortical representation head."""
    head_type = str(cfg.model.head_type).lower()

    if head_type == "baseline":
        return MLPProjector(
            input_dim=cfg.model.feature_dim,
            hidden_dim=cfg.model.get("mlp_hidden_dim", cfg.model.output_dim),
            output_dim=cfg.model.output_dim,
        )

    if head_type == "cortical":
        predictor = FixedTreePredictor(
            n=cfg.cortical.n,
            L_max=cfg.cortical.L_max,
            dim_in=cfg.cortical.dim_in,
            dim_hidden=cfg.cortical.dim_hidden,
            dim_feedback=cfg.cortical.dim_feedback,
            dim_U=cfg.cortical.dim_U,
            dim_target=cfg.cortical.dim_target,
            disable_lateral=cfg.cortical.disable_lateral,
            disable_feedback=cfg.cortical.disable_feedback,
        )
        return CorticalHead(
            predictor=predictor,
            input_dim=cfg.model.feature_dim,
            output_dim=cfg.model.output_dim,
        )

    raise ValueError(
        f"Unknown model.head_type {head_type!r}; expected 'baseline' or 'cortical'"
    )
