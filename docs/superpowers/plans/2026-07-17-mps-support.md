# PyTorch MPS Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Apple Silicon MPS support to training and evaluation while preserving the CUDA-only A100 benchmark protocol.

**Architecture:** Centralize device selection and device-dependent runtime decisions in `comparison.checkpoint`. Pass the selected `torch.device` into data-loader construction, and make Train and Evaluate consume the same helpers so MPS never enables `bfloat16` autocast or pinned memory.

**Tech Stack:** Python 3.12, PyTorch 2.6, OmegaConf, pytest, CIFAR-10 workflow.

## Global Constraints

- `setup_device("auto")` must select CUDA, then MPS, then CPU.
- Explicit `"mps"` selection must fail clearly when MPS is unavailable.
- MPS must disable configured `bfloat16` autocast and `pin_memory`.
- Existing CPU and CUDA behavior must remain unchanged.
- Benchmarking must remain restricted to NVIDIA A100, CUDA, bfloat16, and batch size 256.
- Checkpoint format and RNG-state schema must remain unchanged.
- No YAML or notebook device parameter is added.
- Existing unrelated working-tree changes must not be staged or committed.

---

### Task 1: Device selection and capability helpers

**Files:**
- Create: `eb_jepa_cifar10_comparison/tests/test_device.py`
- Modify: `eb_jepa_cifar10_comparison/src/comparison/checkpoint.py:102-138`

**Interfaces:**
- Consumes: `torch.cuda.is_available()`, `torch.backends.mps.is_available()`, `torch.device`.
- Produces: `setup_device(device_name: str = "auto") -> torch.device` with MPS support.
- Produces: `should_use_bfloat16(precision: object, device: torch.device) -> bool`.
- Produces: `should_pin_memory(requested: object, device: torch.device) -> bool`.

- [ ] **Step 1: Write failing device-selection tests**

Create `tests/test_device.py` with hardware-independent monkeypatches:

```python
import pytest
import torch

from comparison.checkpoint import setup_device


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_setup_device_auto_priority(
    monkeypatch,
    cuda_available,
    mps_available,
    expected,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(
        torch.backends.mps,
        "is_available",
        lambda: mps_available,
    )

    assert setup_device().type == expected


def test_setup_device_accepts_available_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert setup_device("mps") == torch.device("mps")


def test_setup_device_rejects_unavailable_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS.*explicitly requested"):
        setup_device("mps")
```

- [ ] **Step 2: Run the selection tests and confirm RED**

Run:

```bash
cd eb_jepa_cifar10_comparison
uv run pytest tests/test_device.py -v
```

Expected: the MPS cases fail because `setup_device` currently accepts only CPU and CUDA and `auto` falls back directly to CPU.

- [ ] **Step 3: Implement minimal MPS selection**

Update `setup_device` in `checkpoint.py`:

```python
def _mps_is_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def setup_device(device_name: str = "auto") -> torch.device:
    """Select CUDA, MPS, or CPU and reject unavailable explicit devices."""
    normalized_name = device_name.lower()

    if normalized_name == "auto":
        if torch.cuda.is_available():
            normalized_name = "cuda"
        elif _mps_is_available():
            normalized_name = "mps"
        else:
            normalized_name = "cpu"

    try:
        device = torch.device(normalized_name)
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            f"Unknown device {device_name!r}; expected 'auto', 'cpu', "
            "'mps', 'cuda' or 'cuda:<index>'"
        ) from error

    if device.type not in {"cpu", "mps", "cuda"}:
        raise ValueError(
            f"Unsupported device type {device.type!r}; "
            "expected CPU, MPS or CUDA"
        )

    if device.type == "mps" and not _mps_is_available():
        raise RuntimeError(
            "MPS was explicitly requested, but no MPS device is available"
        )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was explicitly requested, but no CUDA device is available"
            )
        if device.index is not None:
            device_count = torch.cuda.device_count()
            if device.index >= device_count:
                raise RuntimeError(
                    f"CUDA device {device.index} was requested, "
                    f"but only {device_count} CUDA device(s) are available"
                )

    return device
```

- [ ] **Step 4: Run the selection tests and confirm GREEN**

Run: `uv run pytest tests/test_device.py -v`

Expected: 5 parameterized/explicit MPS cases pass.

- [ ] **Step 5: Write failing capability tests**

Append to `tests/test_device.py`:

```python
from comparison.checkpoint import should_pin_memory, should_use_bfloat16


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [("cuda", True), ("cpu", True), ("mps", False)],
)
def test_bfloat16_is_disabled_only_on_mps(device_name, expected):
    assert should_use_bfloat16(
        "bfloat16",
        torch.device(device_name),
    ) is expected


def test_non_bfloat16_precision_stays_disabled():
    assert not should_use_bfloat16("float32", torch.device("cuda"))


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [("cuda", True), ("cpu", True), ("mps", False)],
)
def test_pin_memory_is_disabled_only_on_mps(device_name, expected):
    assert should_pin_memory(True, torch.device(device_name)) is expected


def test_disabled_pin_memory_stays_disabled():
    assert not should_pin_memory(False, torch.device("cuda"))
```

- [ ] **Step 6: Run the capability tests and confirm RED**

Run: `uv run pytest tests/test_device.py -v`

Expected: collection fails because both capability helpers are missing.

- [ ] **Step 7: Implement capability helpers**

Add to `checkpoint.py` immediately after `setup_device`:

```python
def should_use_bfloat16(
    precision: object,
    device: torch.device,
) -> bool:
    """Return whether configured bfloat16 autocast is safe on the device."""
    return str(precision).lower() == "bfloat16" and device.type != "mps"


def should_pin_memory(requested: object, device: torch.device) -> bool:
    """Honor pin_memory except on MPS, where PyTorch cannot use it."""
    return bool(requested) and device.type != "mps"
```

- [ ] **Step 8: Run all Task 1 tests and confirm GREEN**

Run: `uv run pytest tests/test_device.py -v`

Expected: all selection and capability tests pass with no warnings.

- [ ] **Step 9: Commit Task 1 only**

```bash
git add eb_jepa_cifar10_comparison/tests/test_device.py \
  eb_jepa_cifar10_comparison/src/comparison/checkpoint.py
git commit -m "feat: add MPS device selection"
```

---

### Task 2: Wire MPS-safe options into Train and Evaluate

**Files:**
- Modify: `eb_jepa_cifar10_comparison/tests/test_device.py`
- Modify: `eb_jepa_cifar10_comparison/src/comparison/data.py:130-190`
- Modify: `eb_jepa_cifar10_comparison/src/comparison/train.py:14,171-220`
- Modify: `eb_jepa_cifar10_comparison/src/comparison/evaluate.py:13,125-190`

**Interfaces:**
- Consumes: `should_use_bfloat16(precision, device) -> bool` from Task 1.
- Consumes: `should_pin_memory(requested, device) -> bool` from Task 1.
- Changes: `make_dataloaders(cfg, train_transform, eval_transform, seed, device) -> DataLoaders`.
- Produces: Train and Evaluate paths that use standard precision and unpinned memory on MPS.

- [ ] **Step 1: Write a failing DataLoader integration test**

Append to `tests/test_device.py`:

```python
from omegaconf import OmegaConf
from torch.utils.data import Dataset

import comparison.data as data_module


class _SizedDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.zeros(3, 4, 4), 0


def test_make_dataloaders_disables_pin_memory_on_mps(monkeypatch):
    def fake_cifar10(*, train, **kwargs):
        del kwargs
        return _SizedDataset(6 if train else 3)

    monkeypatch.setattr(data_module, "CIFAR10", fake_cifar10)
    cfg = OmegaConf.create(
        {
            "data": {
                "root": "unused",
                "download": False,
                "train_size": 4,
                "validation_size": 2,
                "test_size": 3,
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": True,
                "split_seed": 42,
            }
        }
    )

    loaders = data_module.make_dataloaders(
        cfg=cfg,
        train_transform=lambda image: image,
        eval_transform=lambda image: image,
        seed=1,
        device=torch.device("mps"),
    )

    assert not loaders.train.pin_memory
    assert not loaders.validation.pin_memory
    assert not loaders.test.pin_memory
```

- [ ] **Step 2: Run the integration test and confirm RED**

Run:

```bash
uv run pytest \
  tests/test_device.py::test_make_dataloaders_disables_pin_memory_on_mps -v
```

Expected: FAIL because `make_dataloaders` does not accept `device`.

- [ ] **Step 3: Pass the device through DataLoader construction**

In `data.py`, import the helper, add the parameter, and resolve the option:

```python
from .checkpoint import seed_worker, should_pin_memory


def make_dataloaders(
    cfg: DictConfig,
    train_transform: Callable,
    eval_transform: Callable,
    seed: int,
    device: torch.device,
) -> DataLoaders:
    # Existing dataset and split construction remains unchanged.
    loader_options = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": should_pin_memory(
            cfg.data.get("pin_memory", True),
            device,
        ),
        "worker_init_fn": seed_worker,
    }
```

In `train.py`, pass the already selected device:

```python
loaders = make_dataloaders(
    cfg=cfg,
    train_transform=train_transforms,
    eval_transform=validation_transforms,
    seed=seed,
    device=device,
)
```

- [ ] **Step 4: Run the DataLoader test and confirm GREEN**

Run:

```bash
uv run pytest \
  tests/test_device.py::test_make_dataloaders_disables_pin_memory_on_mps -v
```

Expected: PASS.

- [ ] **Step 5: Wire precision and evaluation pinned memory**

Update imports in `train.py`:

```python
from .checkpoint import (
    load_checkpoint,
    save_checkpoint,
    setup_device,
    setup_seed,
    should_use_bfloat16,
)
```

Replace Train's precision decision:

```python
use_bf16 = should_use_bfloat16(cfg.optimization.precision, device)
```

Update imports in `evaluate.py`:

```python
from .checkpoint import (
    setup_device,
    should_pin_memory,
    should_use_bfloat16,
)
```

Replace Evaluate's precision and loader options:

```python
use_bf16 = should_use_bfloat16(cfg.optimization.precision, device)

loader = DataLoader(
    test_dataset,
    batch_size=cfg.data.batch_size,
    num_workers=cfg.data.num_workers,
    pin_memory=should_pin_memory(
        cfg.data.get("pin_memory", True),
        device,
    ),
    shuffle=False,
    drop_last=False,
)
```

- [ ] **Step 6: Run targeted device and progress tests**

Run:

```bash
uv run pytest tests/test_device.py tests/test_train_progress.py -v
```

Expected: all tests pass; the progress test confirms the existing training loop still accepts an explicit `use_bf16` decision.

- [ ] **Step 7: Commit Task 2 only**

```bash
git add eb_jepa_cifar10_comparison/tests/test_device.py \
  eb_jepa_cifar10_comparison/src/comparison/data.py \
  eb_jepa_cifar10_comparison/src/comparison/train.py \
  eb_jepa_cifar10_comparison/src/comparison/evaluate.py
git commit -m "feat: use MPS-safe training options"
```

---

### Task 3: Document MPS workflow and verify the branch

**Files:**
- Modify: `eb_jepa_cifar10_comparison/README.md`

**Interfaces:**
- Consumes: final automatic selection and runtime behavior from Tasks 1-2.
- Produces: user-facing instructions that distinguish portable Train/Evaluate from CUDA-only Benchmark.

- [ ] **Step 1: Add the MPS runtime section**

Insert after the notebook startup instructions in `README.md`:

```markdown
### Accélération matérielle

Train et Evaluate sélectionnent automatiquement le premier périphérique
disponible dans l'ordre CUDA → MPS → CPU. Sur un Mac Apple Silicon avec un
PyTorch compatible, MPS est donc utilisé lorsqu'aucun GPU CUDA n'est présent.

Sur MPS, le workflow désactive automatiquement l'autocast `bfloat16` et
`pin_memory`; l'entraînement et l'évaluation utilisent sinon les mêmes
configurations et checkpoints. Benchmark reste volontairement limité à une
NVIDIA A100 avec CUDA, `bfloat16` et un batch de 256 conformément au protocole.
```

- [ ] **Step 2: Run formatting and placeholder checks**

Run:

```bash
git diff --check
rg -n "T[B]D|TO[D]O|FIX[M]E|X[X]X" \
  docs/superpowers/specs/2026-07-17-mps-support-design.md \
  docs/superpowers/plans/2026-07-17-mps-support.md \
  eb_jepa_cifar10_comparison/README.md
```

Expected: `git diff --check` exits 0; `rg` finds no placeholders introduced by this feature.

- [ ] **Step 3: Run the complete project test suite**

Run:

```bash
cd eb_jepa_cifar10_comparison
uv run pytest -v
```

Expected: all collected tests pass with zero failures.

- [ ] **Step 4: Run a real MPS smoke check when available**

Run:

```bash
uv run python -c "from comparison.checkpoint import setup_device, should_pin_memory, should_use_bfloat16; d=setup_device(); print(d, should_use_bfloat16('bfloat16', d), should_pin_memory(True, d))"
```

Expected on Apple Silicon with MPS: `mps False False`. On CUDA or CPU, the selected device is printed with `True True`; the hardware-independent tests still prove the MPS branch.

- [ ] **Step 5: Review only feature-owned diffs**

Run:

```bash
git diff -- \
  eb_jepa_cifar10_comparison/src/comparison/checkpoint.py \
  eb_jepa_cifar10_comparison/src/comparison/data.py \
  eb_jepa_cifar10_comparison/src/comparison/train.py \
  eb_jepa_cifar10_comparison/src/comparison/evaluate.py \
  eb_jepa_cifar10_comparison/tests/test_device.py \
  eb_jepa_cifar10_comparison/README.md
```

Expected: only MPS selection, capability wiring, tests, and documentation appear; benchmark code and unrelated local changes are absent.

- [ ] **Step 6: Commit documentation**

```bash
git add eb_jepa_cifar10_comparison/README.md
git commit -m "docs: describe MPS workflow"
```

- [ ] **Step 7: Verify final branch state**

Run:

```bash
git log --oneline -4
git status --short --branch
```

Expected: the branch contains the design, implementation-plan, MPS selection, runtime wiring, and documentation commits. Pre-existing unrelated changes may remain unstaged and must not appear in any feature commit.
