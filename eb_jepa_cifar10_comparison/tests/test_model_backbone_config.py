from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

import comparison.backbone as backbone_module
from comparison.checkpoint import load_checkpoint
from comparison.config import load_config
from comparison.model import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"
CORTICAL_CONFIG = PROJECT_ROOT / "configs" / "cortical.yaml"


@pytest.mark.parametrize(
    ("name", "expected_class_name"),
    [("mlp", "MLPEncoder"), ("MLP", "MLPEncoder"), ("resnet18", "ResNet18")],
)
def test_build_backbone_selects_supported_encoder(name, expected_class_name):
    factory = getattr(backbone_module, "build_backbone", None)
    assert factory is not None, "build_backbone has not been implemented"

    assert type(factory(name)).__name__ == expected_class_name


def test_build_backbone_rejects_unknown_encoder():
    factory = getattr(backbone_module, "build_backbone", None)
    assert factory is not None, "build_backbone has not been implemented"

    with pytest.raises(ValueError, match="mlp.*resnet18"):
        factory("transformer")


def test_config_and_model_accept_mlp_override():
    cfg = load_config(BASELINE_CONFIG, ["model.backbone=mlp"])

    model = build_model(cfg)

    assert type(model.backbone).__name__ == "MLPEncoder"


def test_config_rejects_unknown_backbone():
    with pytest.raises(ValueError, match="mlp.*resnet18"):
        load_config(BASELINE_CONFIG, ["model.backbone=transformer"])


@pytest.mark.parametrize("config_path", [BASELINE_CONFIG, CORTICAL_CONFIG])
def test_default_config_builds_mlp_with_unchanged_dimensions(config_path):
    cfg = load_config(config_path)
    model = build_model(cfg).eval()

    with torch.no_grad():
        features, projections = model(torch.randn(2, 3, 32, 32))

    assert cfg.model.backbone == "mlp"
    assert features.shape == (2, 512)
    assert projections.shape == (2, 2048)


def test_resnet_checkpoint_is_rejected_by_mlp_configuration(tmp_path):
    resnet_cfg = load_config(BASELINE_CONFIG, ["model.backbone=resnet18"])
    mlp_cfg = load_config(BASELINE_CONFIG, ["model.backbone=mlp"])
    checkpoint_path = tmp_path / "resnet.pt"
    torch.save(
        {
            "format_version": 2,
            "epoch": 0,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "best_validation": 1.0,
            "config": OmegaConf.to_container(resnet_cfg, resolve=True),
            "head_type": "baseline",
            "seed": 1,
            "rng_state": {},
            "train_generator_state": torch.Generator().get_state(),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="model.backbone"):
        load_checkpoint(
            checkpoint_path,
            model=build_model(mlp_cfg),
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            cfg=mlp_cfg,
            device=torch.device("cpu"),
            seed=1,
            train_generator=torch.Generator(),
        )
