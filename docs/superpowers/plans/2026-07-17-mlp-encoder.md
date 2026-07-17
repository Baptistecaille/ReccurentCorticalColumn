# Lightweight MLP Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the default Baseline and Cortical CIFAR-10 experiments use a lightweight `Flatten -> Linear -> LayerNorm -> GELU` encoder while retaining ResNet-18 as an explicitly selectable compatible option.

**Architecture:** Add `MLPEncoder` and a strict `build_backbone` factory to the focused backbone module. Keep `ImageSSL` generic over any `nn.Module`, route `cfg.model.backbone` through the factory, widen protocol validation to exactly `mlp` and `resnet18`, and switch both default YAML files to `mlp` without changing VICReg's 2,048-dimensional output.

**Tech Stack:** Python 3.12, PyTorch, torchvision, OmegaConf, pytest, uv

## Global Constraints

- `MLPEncoder` is exactly `Flatten -> Linear(3072, 512) -> LayerNorm(512) -> GELU`.
- Encoder inputs are normalized CIFAR-10 tensors shaped `(B, 3, 32, 32)` and features remain shaped `(B, 512)`.
- `model.backbone` accepts exactly `mlp` or `resnet18`, case-insensitively.
- Both shipped experiment configurations default to `mlp` and continue to use the same encoder.
- `model.feature_dim` and `cortical.dim_in` remain 512.
- `model.output_dim` and `cortical.dim_target` remain 2,048; reducing VICReg dimensionality is outside this change.
- ResNet-18 remains constructible and its existing checkpoints remain usable with ResNet-18 configurations.
- Checkpoint architecture mismatches fail strictly through the existing `model.backbone` compatibility check.
- Preserve all unrelated working-tree changes. In particular, stage only the `model.backbone` YAML hunks and do not stage pre-existing epoch changes.

---

### Task 1: Implement the lightweight encoder

**Files:**
- Modify: `eb_jepa_cifar10_comparison/src/comparison/backbone.py`
- Create: `eb_jepa_cifar10_comparison/tests/test_backbone.py`

**Interfaces:**
- Consumes: CIFAR-10 tensors shaped `(B, 3, 32, 32)`.
- Produces: `class MLPEncoder(nn.Module)` with `features_dim: int = 512`, a zero-argument constructor, and `forward(images: Tensor) -> Tensor` returning `(B, 512)`.

- [ ] **Step 1: Write a failing behavior test without causing a collection error**

Create `tests/test_backbone.py`:

```python
import torch

import comparison.backbone as backbone_module


def test_mlp_encoder_maps_cifar_images_to_512_features():
    encoder_class = getattr(backbone_module, "MLPEncoder", None)
    assert encoder_class is not None, "MLPEncoder has not been implemented"

    encoder = encoder_class()
    features = encoder(torch.randn(4, 3, 32, 32))

    assert features.shape == (4, 512)


def test_mlp_encoder_has_expected_trainable_parameter_count():
    encoder_class = getattr(backbone_module, "MLPEncoder", None)
    assert encoder_class is not None, "MLPEncoder has not been implemented"

    encoder = encoder_class()
    trainable = sum(
        parameter.numel()
        for parameter in encoder.parameters()
        if parameter.requires_grad
    )

    assert trainable == 1_574_400
```

- [ ] **Step 2: Run the new tests and verify the expected red state**

Run:

```bash
cd eb_jepa_cifar10_comparison
uv run pytest tests/test_backbone.py -v
```

Expected: two assertion failures containing `MLPEncoder has not been implemented`.

- [ ] **Step 3: Add the minimal encoder implementation**

In `src/comparison/backbone.py`, preserve `ResNet18` and add:

```python
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
```

Update the module docstring so it describes both locally available encoders instead of describing the file as only a copied ResNet implementation.

- [ ] **Step 4: Run the focused tests and verify green**

Run:

```bash
uv run pytest tests/test_backbone.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit Task 1**

```bash
git add eb_jepa_cifar10_comparison/src/comparison/backbone.py \
  eb_jepa_cifar10_comparison/tests/test_backbone.py
git commit -m "feat: add lightweight MLP encoder"
```

---

### Task 2: Select and validate the configured backbone

**Files:**
- Modify: `eb_jepa_cifar10_comparison/src/comparison/backbone.py`
- Modify: `eb_jepa_cifar10_comparison/src/comparison/model.py`
- Modify: `eb_jepa_cifar10_comparison/src/comparison/config.py`
- Create: `eb_jepa_cifar10_comparison/tests/test_model_backbone_config.py`

**Interfaces:**
- Consumes: `build_backbone(name: object) -> nn.Module` and `cfg.model.backbone`.
- Produces: strict case-insensitive selection of `MLPEncoder` or `ResNet18`; `ImageSSL` accepts any `nn.Module` backbone; `validate_config` accepts the same two names.

- [ ] **Step 1: Write failing selection and validation tests**

Create `tests/test_model_backbone_config.py`:

```python
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
```

- [ ] **Step 2: Run the focused file and verify red**

Run:

```bash
uv run pytest tests/test_model_backbone_config.py -v
```

Expected: selection tests fail because `build_backbone` is missing, and the MLP configuration test fails because validation still only accepts `resnet18`.

- [ ] **Step 3: Implement the strict backbone factory**

Append to `src/comparison/backbone.py`:

```python
def build_backbone(name: object) -> nn.Module:
    """Build a fresh encoder selected by its configuration name."""
    normalized_name = str(name).lower()
    if normalized_name == "mlp":
        return MLPEncoder()
    if normalized_name == "resnet18":
        return ResNet18()
    raise ValueError(
        f"model.backbone must be 'mlp' or 'resnet18', received {name!r}"
    )
```

- [ ] **Step 4: Make the model container backbone-agnostic**

In `src/comparison/model.py`, replace the `ResNet18` import with:

```python
from .backbone import build_backbone
```

Change the constructor annotation and factory body to:

```python
class ImageSSL(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone(images)
        projections = self.head(features)
        return features, projections


def build_model(cfg: DictConfig) -> ImageSSL:
    backbone = build_backbone(cfg.model.backbone)
    head = build_head(cfg)
    return ImageSSL(backbone=backbone, head=head)
```

Leave `ParameterCounts` and `count_parameters` behavior unchanged.

- [ ] **Step 5: Widen protocol validation to the same exact names**

In `src/comparison/config.py`, replace the ResNet-only requirement with:

```python
    backbone = str(cfg.model.backbone).lower()
    _require(
        backbone in {"mlp", "resnet18"},
        "model.backbone must be 'mlp' or 'resnet18', "
        f"received {cfg.model.backbone!r}",
    )
```

- [ ] **Step 6: Run focused and adjacent tests**

Run:

```bash
uv run pytest tests/test_backbone.py tests/test_model_backbone_config.py tests/test_device.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add eb_jepa_cifar10_comparison/src/comparison/backbone.py \
  eb_jepa_cifar10_comparison/src/comparison/model.py \
  eb_jepa_cifar10_comparison/src/comparison/config.py \
  eb_jepa_cifar10_comparison/tests/test_model_backbone_config.py
git commit -m "feat: select encoder from configuration"
```

---

### Task 3: Switch defaults and enforce dimensional and checkpoint contracts

**Files:**
- Modify: `eb_jepa_cifar10_comparison/configs/baseline.yaml`
- Modify: `eb_jepa_cifar10_comparison/configs/cortical.yaml`
- Modify: `eb_jepa_cifar10_comparison/tests/test_model_backbone_config.py`

**Interfaces:**
- Consumes: the shipped Baseline and Cortical YAML configurations, `build_model`, and `load_checkpoint`.
- Produces: default MLP models with `(B, 512)` features and `(B, 2048)` projections, while rejecting a ResNet checkpoint under an MLP configuration before loading weights.

- [ ] **Step 1: Add failing default-configuration and end-to-end shape tests**

Append to `tests/test_model_backbone_config.py`:

```python
import torch
from omegaconf import OmegaConf
from unittest.mock import MagicMock

from comparison.checkpoint import load_checkpoint


CORTICAL_CONFIG = PROJECT_ROOT / "configs" / "cortical.yaml"


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
```

- [ ] **Step 2: Run only the two default-config cases and verify red**

Run:

```bash
uv run pytest tests/test_model_backbone_config.py::test_default_config_builds_mlp_with_unchanged_dimensions -v
```

Expected: two failures because both shipped configurations still say `resnet18`.

- [ ] **Step 3: Switch only the backbone setting in each default YAML**

In both `configs/baseline.yaml` and `configs/cortical.yaml`, change:

```yaml
  backbone: resnet18
```

to:

```yaml
  backbone: mlp
```

Do not alter `feature_dim`, `output_dim`, `dim_in`, or `dim_target`. Preserve the pre-existing local epoch edits already present in these files.

- [ ] **Step 4: Run all encoder/config contract tests**

Run:

```bash
uv run pytest tests/test_backbone.py tests/test_model_backbone_config.py -v
```

Expected: all tests pass, including strict ResNet-to-MLP checkpoint rejection.

- [ ] **Step 5: Run the complete project test suite**

Run:

```bash
uv run pytest -v
```

Expected: all tests pass without new warnings or errors.

- [ ] **Step 6: Perform a small CPU timing smoke comparison**

Run from `eb_jepa_cifar10_comparison`:

```bash
uv run python -c '
import time
import torch
from comparison.backbone import MLPEncoder, ResNet18

x = torch.randn(32, 3, 32, 32)
for name, model in (("mlp", MLPEncoder()), ("resnet18", ResNet18())):
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(x)
        started = time.perf_counter()
        for _ in range(10):
            model(x)
    print(name, (time.perf_counter() - started) / 10)
'
```

Expected: both encoders execute and the MLP has lower local mean forward latency. Record the observed numbers in the handoff only; do not turn a machine-specific ratio into a test.

- [ ] **Step 7: Stage only the intended YAML hunks and test change**

Because both YAML files already contain unrelated unstaged epoch changes, use interactive staging:

```bash
git add eb_jepa_cifar10_comparison/tests/test_model_backbone_config.py
git add -p eb_jepa_cifar10_comparison/configs/baseline.yaml
git add -p eb_jepa_cifar10_comparison/configs/cortical.yaml
git diff --cached --check
git diff --cached
```

Expected cached diff: the added tests and only `backbone: resnet18` to `backbone: mlp` in each YAML. The existing `epochs` hunks remain unstaged.

- [ ] **Step 8: Commit Task 3**

```bash
git commit -m "feat: use MLP encoder in default experiments"
```

---

### Task 4: Final verification and review

**Files:**
- Verify only; no planned production changes.

**Interfaces:**
- Consumes: all deliverables from Tasks 1-3.
- Produces: evidence that the implementation matches the design without absorbing unrelated workspace changes.

- [ ] **Step 1: Inspect the branch diff and repository state**

```bash
git status --short
git diff --check
git diff codex/mps-support...HEAD -- \
  eb_jepa_cifar10_comparison/src/comparison/backbone.py \
  eb_jepa_cifar10_comparison/src/comparison/model.py \
  eb_jepa_cifar10_comparison/src/comparison/config.py \
  eb_jepa_cifar10_comparison/configs/baseline.yaml \
  eb_jepa_cifar10_comparison/configs/cortical.yaml \
  eb_jepa_cifar10_comparison/tests/test_backbone.py \
  eb_jepa_cifar10_comparison/tests/test_model_backbone_config.py
```

Expected: only the designed encoder, configuration, tests, and the already-approved design/plan commits belong to this branch; unrelated local changes remain uncommitted.

- [ ] **Step 2: Re-run the complete suite as fresh completion evidence**

```bash
cd eb_jepa_cifar10_comparison
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Request code review**

Invoke `superpowers:requesting-code-review` and review the branch against `codex/mps-support`, with special attention to:

- preservation of the exact MLP layer order and dimensions;
- strict accepted backbone names and error messages;
- default equality between Baseline and Cortical encoders;
- strict checkpoint mismatch behavior;
- exclusion of unrelated working-tree changes.

- [ ] **Step 4: Apply only confirmed review fixes and verify again**

For each actionable issue, use the relevant TDD red-green cycle, then run:

```bash
uv run pytest -v
git diff --check
```

Expected: all tests pass and no whitespace errors remain.
