from pathlib import Path

import pytest

import comparison.backbone as backbone_module
from comparison.config import load_config
from comparison.model import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"


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
