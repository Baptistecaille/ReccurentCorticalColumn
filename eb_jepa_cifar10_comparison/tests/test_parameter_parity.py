from pathlib import Path

from omegaconf import OmegaConf

from comparison.config import load_config
from comparison.model import build_model, count_parameters


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs"

# baseline.yaml carries working-tree edits that fail validate_config without
# these; head parameter counts are unaffected by optimization values.
PROTOCOL_OVERRIDES = [
    "model.backbone=resnet18",
    "optimization.learning_rate=0.3",
    "optimization.warmup_start_lr=0.00003",
]
PARITY_TOLERANCE = 0.01


def _head_parameters(config_name: str) -> int:
    cfg = load_config(CONFIG_DIR / config_name, PROTOCOL_OVERRIDES)
    return count_parameters(build_model(cfg)).head


def test_matched_head_matches_baseline_within_tolerance() -> None:
    baseline = _head_parameters("baseline.yaml")
    matched = _head_parameters("cortical_matched.yaml")
    relative_gap = abs(matched - baseline) / baseline
    assert relative_gap <= PARITY_TOLERANCE, (baseline, matched, relative_gap)


def test_matched_config_changes_only_two_cortical_dimensions() -> None:
    original = load_config(CONFIG_DIR / "cortical.yaml", PROTOCOL_OVERRIDES)
    matched = load_config(CONFIG_DIR / "cortical_matched.yaml", PROTOCOL_OVERRIDES)

    assert int(matched.cortical.dim_U) == 512
    assert int(matched.cortical.dim_feedback) == 168
    assert str(matched.model.head_type).lower() == "cortical"

    original_dict = OmegaConf.to_container(original, resolve=True)
    matched_dict = OmegaConf.to_container(matched, resolve=True)
    for cfg in (original_dict, matched_dict):
        cfg["cortical"]["dim_U"] = None
        cfg["cortical"]["dim_feedback"] = None
    assert original_dict == matched_dict
