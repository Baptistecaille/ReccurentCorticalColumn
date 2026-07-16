# Training Progress Bars Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display nested epoch and training-batch progress bars with running loss metrics during CIFAR-10 training in Colab and terminals.

**Architecture:** Keep progress rendering inside `comparison.train`, immediately around the two loops whose work it represents. Use `tqdm.auto.tqdm` so the same implementation selects the notebook or terminal renderer, and replace `tqdm` with a recording double in focused unit tests so no real progress output or model-scale computation is required.

**Tech Stack:** Python 3.12, PyTorch 2.6, `tqdm>=4.66`, `unittest`, `unittest.mock`, OmegaConf.

## Global Constraints

- Do not change the scientific protocol, numerical results, checkpoint behavior, or public training API.
- Keep all configuration keys, function signatures, return values, optimizer steps, scheduler steps, validation logic, and checkpoint rules unchanged.
- Use `tqdm.auto.tqdm` for both progress bars.
- Label the persistent outer bar `Epochs` and the transient inner bar `Training`.
- Set `leave=False` on the inner bar.
- Report running mean training loss on the inner bar.
- Report completed `train_loss` and `val_loss` on the outer bar.
- Preserve all unrelated working-tree changes.

---

## File Structure

- Create `eb_jepa_cifar10_comparison/tests/test_train_progress.py`: focused recording-double tests for batch and epoch progress behavior.
- Modify `eb_jepa_cifar10_comparison/src/comparison/train.py`: import `tqdm.auto.tqdm`, wrap the existing loops, and update loss postfixes.

No new runtime dependency is required because `tqdm>=4.66` is already declared in `eb_jepa_cifar10_comparison/pyproject.toml`.

### Task 1: Add nested training progress behavior

**Files:**
- Create: `eb_jepa_cifar10_comparison/tests/test_train_progress.py`
- Modify: `eb_jepa_cifar10_comparison/src/comparison/train.py:1-13`
- Modify: `eb_jepa_cifar10_comparison/src/comparison/train.py:37-84`
- Modify: `eb_jepa_cifar10_comparison/src/comparison/train.py:243-307`

**Interfaces:**
- Consumes: `train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, use_bf16) -> EpochMetrics` and `run(cfg, seed, output_dir, resume_from=None) -> RunResult` without signature changes.
- Produces: one `tqdm` call with `desc="Training"`, `leave=False`, and `unit="batch"` per training epoch; one `tqdm` call with `desc="Epochs"` and `unit="epoch"` per run; loss postfix dictionaries on both bars.

- [ ] **Step 1: Create a recording progress double and a failing batch-progress test**

Create `eb_jepa_cifar10_comparison/tests/test_train_progress.py` with:

```python
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf
import torch
from torch import nn

from comparison import train as train_module
from comparison.losses import CollapseDiagnostics


class RecordingProgress:
    def __init__(self, iterable, options):
        self.iterable = iterable
        self.options = options
        self.postfixes = []

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **values):
        self.postfixes.append(values)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 1)

    def forward(self, values):
        projection = self.projection(values)
        return values, projection


class TrainProgressTest(unittest.TestCase):
    def test_train_epoch_reports_batches_and_running_loss(self):
        bars = []

        def record_progress(iterable, **options):
            bar = RecordingProgress(iterable, options)
            bars.append(bar)
            return bar

        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = MagicMock()
        loader = [
            (torch.tensor([[1.0], [2.0]]), torch.tensor([[1.5], [2.5]])),
            (torch.tensor([[3.0]]), torch.tensor([[3.5]])),
        ]

        with patch.object(train_module, "tqdm", side_effect=record_progress):
            metrics = train_module.train_epoch(
                model=model,
                train_loader=loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=nn.MSELoss(),
                device=torch.device("cpu"),
                use_bf16=False,
            )

        self.assertEqual(metrics.num_batches, 2)
        self.assertEqual(len(bars), 1)
        self.assertEqual(
            bars[0].options,
            {"desc": "Training", "leave": False, "unit": "batch"},
        )
        self.assertEqual(len(bars[0].postfixes), 2)
        self.assertEqual(set(bars[0].postfixes[-1]), {"loss"})
        self.assertEqual(
            bars[0].postfixes[-1]["loss"],
            f"{metrics.mean_loss:.4f}",
        )
        self.assertEqual(scheduler.step.call_count, 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the batch-progress test and verify the expected red state**

Run from `eb_jepa_cifar10_comparison`:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -p 'test_train_progress.py' -v
```

Expected: `ERROR` because `comparison.train` has no attribute `tqdm`. This proves the test detects the missing progress feature.

- [ ] **Step 3: Add the minimal batch progress implementation**

In `src/comparison/train.py`, add:

```python
from tqdm.auto import tqdm
```

Replace `for batch in train_loader:` with:

```python
batch_progress = tqdm(
    train_loader,
    desc="Training",
    leave=False,
    unit="batch",
)

for batch in batch_progress:
```

Immediately after incrementing `num_batches`, add:

```python
batch_progress.set_postfix(
    loss=f"{total_loss / total_exemples:.4f}",
)
```

- [ ] **Step 4: Run the focused test and verify the batch bar is green**

Run:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -p 'test_train_progress.py' -v
```

Expected: `Ran 1 test` and `OK`.

- [ ] **Step 5: Add a failing outer epoch-progress test**

Insert this method inside `TrainProgressTest`, before the `if __name__` footer:

```python
    def test_run_reports_epochs_and_completed_losses(self):
        bars = []

        def record_progress(iterable, **options):
            bar = RecordingProgress(iterable, options)
            bars.append(bar)
            return bar

        cfg = OmegaConf.create(
            {
                "data": {"crop_scale": [0.2, 1.0]},
                "loss": {
                    "invariance_coeff": 1.0,
                    "std_coeff": 1.0,
                    "cov_coeff": 80.0,
                    "variance_target": 1.0,
                    "variance_epsilon": 0.0001,
                },
                "optimization": {
                    "epochs": 2,
                    "learning_rate": 0.3,
                    "warmup_epochs": 0,
                    "warmup_start_lr": 0.00003,
                    "min_lr": 0.0,
                    "weight_decay": 0.0001,
                    "momentum": 0.9,
                    "precision": "bfloat16",
                },
                "experiment": {"checkpoint_every": 50},
            }
        )
        loaders = SimpleNamespace(
            train=[object()],
            validation=[object()],
            train_generator=torch.Generator(),
        )
        training = [
            train_module.EpochMetrics(mean_loss=1.25, num_batches=1),
            train_module.EpochMetrics(mean_loss=0.75, num_batches=1),
        ]
        validation = [
            train_module.ValidationMetrics(
                mean_loss=1.0,
                num_batches=1,
                collapse_diagnostics=CollapseDiagnostics(1.0, 1.0, 0.0),
            ),
            train_module.ValidationMetrics(
                mean_loss=0.5,
                num_batches=1,
                collapse_diagnostics=CollapseDiagnostics(1.0, 1.0, 0.0),
            ),
        ]

        with tempfile.TemporaryDirectory() as directory, patch.multiple(
            train_module,
            tqdm=MagicMock(side_effect=record_progress),
            setup_device=MagicMock(return_value=torch.device("cpu")),
            setup_seed=MagicMock(),
            get_train_transforms=MagicMock(return_value=object()),
            make_dataloaders=MagicMock(return_value=loaders),
            build_model=MagicMock(return_value=TinyModel()),
            VICRegLoss=MagicMock(return_value=MagicMock()),
            LARS=MagicMock(return_value=MagicMock()),
            WarmupCosineScheduler=MagicMock(return_value=MagicMock()),
            train_epoch=MagicMock(side_effect=training),
            evaluate_validation=MagicMock(side_effect=validation),
            save_checkpoint=MagicMock(),
        ):
            result = train_module.run(
                cfg=cfg,
                seed=1,
                output_dir=Path(directory) / "run",
            )

        self.assertEqual(result.final_epoch, 1)
        self.assertEqual(len(bars), 1)
        self.assertEqual(
            bars[0].options,
            {"desc": "Epochs", "unit": "epoch"},
        )
        self.assertEqual(
            bars[0].postfixes,
            [
                {"train_loss": "1.2500", "val_loss": "1.0000"},
                {"train_loss": "0.7500", "val_loss": "0.5000"},
            ],
        )
```

- [ ] **Step 6: Run both focused tests and verify only the outer-bar test fails**

Run:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -p 'test_train_progress.py' -v
```

Expected: the batch-progress test passes; `test_run_reports_epochs_and_completed_losses` fails because no `Epochs` bar is recorded.

- [ ] **Step 7: Add the minimal epoch progress implementation**

Replace `for epoch in range(start_epoch, cfg.optimization.epochs):` with:

```python
epoch_progress = tqdm(
    range(start_epoch, cfg.optimization.epochs),
    desc="Epochs",
    unit="epoch",
)

for epoch in epoch_progress:
```

Immediately after `final_epoch = epoch`, add:

```python
epoch_progress.set_postfix(
    train_loss=f"{train_metrics.mean_loss:.4f}",
    val_loss=f"{validation_metrics.mean_loss:.4f}",
)
```

- [ ] **Step 8: Run both focused tests and verify the nested progress behavior is green**

Run:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -p 'test_train_progress.py' -v
```

Expected: `Ran 2 tests` and `OK`.

- [ ] **Step 9: Run the complete subproject test suite**

Run:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

Expected: all discovered tests pass with `OK`. If a pre-existing test fails because of unrelated working-tree changes, record the exact failing test and error without modifying those unrelated files.

- [ ] **Step 10: Inspect the final diff for scope and whitespace errors**

Run from the repository root:

```bash
git diff --check -- eb_jepa_cifar10_comparison/src/comparison/train.py eb_jepa_cifar10_comparison/tests/test_train_progress.py
git diff -- eb_jepa_cifar10_comparison/src/comparison/train.py eb_jepa_cifar10_comparison/tests/test_train_progress.py
```

Expected: `git diff --check` exits successfully with no output; the content diff contains only the `tqdm` import, two progress wrappers, postfix updates, and focused tests.

- [ ] **Step 11: Commit only the progress implementation and its tests**

```bash
git add eb_jepa_cifar10_comparison/src/comparison/train.py eb_jepa_cifar10_comparison/tests/test_train_progress.py
git commit -m "feat: show training progress"
```

Expected: one commit containing exactly the implementation and focused test file; unrelated notebook, lockfile, data, and workspace changes remain unstaged.
