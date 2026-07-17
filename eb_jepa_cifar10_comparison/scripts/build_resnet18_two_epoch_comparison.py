"""Reproducible builder for the ResNet-18 two-epoch comparison notebook.

The notebook source lives here as Python strings so the generated ``.ipynb`` is
reviewable in diffs and regenerated deterministically. Run this module to emit
``scripts/resnet18_two_epoch_comparison.ipynb`` next to it.
"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


DEFAULT_OUTPUT = Path(__file__).with_name("resnet18_two_epoch_comparison.ipynb")


def markdown(cell_id: str, source: str):
    cell = new_markdown_cell(source.strip() + "\n")
    cell["id"] = cell_id
    return cell


def code(cell_id: str, source: str):
    cell = new_code_cell(source.strip() + "\n")
    cell["id"] = cell_id
    return cell


GOAL_MARKDOWN = """
# ResNet-18 Two-Epoch Comparison

Train and fully compare the **EB-JEPA MLP projector** (`baseline` head) and the
**`FixedTreePredictor`** (`cortical` head) with an **identical ResNet-18 backbone**
over exactly **two CIFAR-10 epochs**.

Both arms share the same training seed, batch size, VICReg loss, optimizer, and
protocol overrides; only the projection head differs. The notebook runs locally
(CUDA / Apple MPS / CPU) and on Google Colab without any hard-coded paths.

> **Interpretation limit:** two epochs is a fast, portable comparison, not a
> converged result. Active-device timing here is a portable benchmark, not the
> repository's strict A100 protocol.
"""


PORTABLE_SETUP = """
# Portable environment discovery: locate the project, expose ``src`` on the path,
# and select the best available compute device without hard-coded paths.
import importlib.util
import sys
from pathlib import Path

import torch


def _running_in_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def _find_project_root() -> Path:
    marker = "eb_jepa_cifar10_comparison"
    # Notebook execution has no ``__file__``; start from the working directory.
    candidates = [Path.cwd(), *Path.cwd().parents]
    if _running_in_colab():
        candidates = [Path("/content"), *Path("/content").glob("**/"), *candidates]
    for candidate in candidates:
        if candidate.name == marker and (candidate / "src" / "comparison").is_dir():
            return candidate
        nested = candidate / marker
        if (nested / "src" / "comparison").is_dir():
            return nested
    raise RuntimeError(
        f"Could not locate {marker!r}. Run this notebook from inside the "
        "cloned repository (locally or under /content on Colab)."
    )


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


IN_COLAB = _running_in_colab()
PROJECT_ROOT = _find_project_root()
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

DEVICE = _select_device()

print(f"Colab: {IN_COLAB}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Device: {DEVICE}")
"""


SHARED_IMPORTS = """
# Shared imports: the notebook delegates model, training, evaluation, checkpoint,
# and FLOP behavior to the existing ``comparison`` package.
import hashlib
import json
import math
import platform
from dataclasses import asdict
from time import perf_counter

import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from comparison.benchmark import measure_flops
from comparison.config import load_config
from comparison.evaluate import evaluate_test
from comparison.model import build_model, count_parameters
from comparison.train import run
"""


EXPERIMENT_PARAMETERS = """
# Fixed comparison protocol. These constants are asserted by the structural
# test suite and must not drift.
EPOCHS = 2
TRAINING_SEED = 1
PAIR_SEEDS = (11, 22, 33, 44, 55)
BATCH_SIZE = 256
NUM_WORKERS = 2

ARM_LABELS = ("baseline", "predictor")

# Notebook-local data and output roots kept apart from the strict A100 workflow.
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "resnet18_two_epoch"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Portable active-device timing repetitions (not the strict 100/500 protocol).
BENCHMARK_WARMUPS = 5
BENCHMARK_MEASUREMENTS = 20

print(f"Epochs: {EPOCHS} | seed: {TRAINING_SEED} | pair seeds: {PAIR_SEEDS}")
print(f"Data root: {DATA_ROOT}")
print(f"Output root: {OUTPUT_ROOT}")
"""


CONFIG_MARKDOWN = """
## Fair configurations

Both arms load their published config and apply **one shared override list** so
the only difference is the projection head. The ResNet-18 backbone, two epochs,
zero warm-up, batch size, and VICReg loss are identical.
"""


BUILD_CONFIGURATIONS = """
# One shared override list guarantees a fair comparison: identical backbone,
# schedule, data, and loss for both arms. ``load_config`` validates every value.
SHARED_OVERRIDES = [
    f"data.root={DATA_ROOT}",
    f"data.batch_size={BATCH_SIZE}",
    f"data.num_workers={NUM_WORKERS}",
    "model.backbone=resnet18",
    f"optimization.epochs={EPOCHS}",
    "optimization.warmup_epochs=0",
    "optimization.learning_rate=0.3",
    "optimization.warmup_start_lr=0.00003",
]
CONFIG_PATHS = {
    "baseline": PROJECT_ROOT / "configs" / "baseline.yaml",
    "predictor": PROJECT_ROOT / "configs" / "cortical.yaml",
}
configs = {
    label: load_config(path, SHARED_OVERRIDES)
    for label, path in CONFIG_PATHS.items()
}
for label, cfg in configs.items():
    print(f"{label}: head={cfg.model.head_type} backbone={cfg.model.backbone}")
"""


PROTOCOL_CHECKS = """
# Guardrails: confirm both resolved configs share the ResNet-18 backbone and the
# fixed two-epoch, zero-warm-up protocol, and differ only by the head type.
EXPECTED_HEADS = {"baseline": "baseline", "predictor": "cortical"}

for label, cfg in configs.items():
    assert str(cfg.model.backbone).lower() == "resnet18", label
    assert int(cfg.optimization.epochs) == EPOCHS, label
    assert int(cfg.optimization.warmup_epochs) == 0, label
    assert int(cfg.data.batch_size) == BATCH_SIZE, label
    assert str(cfg.model.head_type).lower() == EXPECTED_HEADS[label], label

# The two arms must agree on every shared data/loss/optimization value.
baseline_cfg, predictor_cfg = configs["baseline"], configs["predictor"]
for path in (
    "data.batch_size",
    "data.crop_scale",
    "loss.cov_coeff",
    "loss.invariance_coeff",
    "loss.std_coeff",
    "optimization.epochs",
    "optimization.warmup_epochs",
    "optimization.learning_rate",
    "optimization.warmup_start_lr",
    "optimization.weight_decay",
):
    left = OmegaConf.select(baseline_cfg, path)
    right = OmegaConf.select(predictor_cfg, path)
    assert left == right, f"Mismatch on {path}: {left} != {right}"

protocol_table = pd.DataFrame(
    [
        {
            "arm": label,
            "head_type": str(cfg.model.head_type),
            "backbone": str(cfg.model.backbone),
            "epochs": int(cfg.optimization.epochs),
            "warmup_epochs": int(cfg.optimization.warmup_epochs),
            "batch_size": int(cfg.data.batch_size),
            "learning_rate": float(cfg.optimization.learning_rate),
        }
        for label, cfg in configs.items()
    ]
)
protocol_table
"""


TRAIN_MODELS = """
# Sequential training: each arm trains from the same seed into its own output
# directory. Active-device wall-clock time is recorded as a portable benchmark.
training_results = {}
training_seconds = {}

for label in ARM_LABELS:
    output_dir = OUTPUT_ROOT / label / str(TRAINING_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training {label} for {EPOCHS} epochs on {DEVICE} ...")
    start = perf_counter()
    result = run(configs[label], seed=TRAINING_SEED, output_dir=output_dir)
    training_seconds[label] = perf_counter() - start
    training_results[label] = result

    assert result.final_epoch == EPOCHS - 1, (label, result.final_epoch)
    assert result.best_checkpoint.is_file(), result.best_checkpoint
    print(
        f"  {label}: {training_seconds[label]:.1f}s | "
        f"best val {result.best_validation:.4f} @ epoch {result.best_epoch}"
    )
"""


EVALUATE_CHECKPOINTS = """
# Strict evaluation of each best checkpoint on the same five deterministic view
# pairs, so the two arms are scored on identical inputs.
evaluations = {}

for label in ARM_LABELS:
    checkpoint = training_results[label].best_checkpoint
    evaluation = evaluate_test(
        configs[label],
        checkpoint,
        pair_seeds=PAIR_SEEDS,
    )
    seen = {pair.pair_seed for pair in evaluation.pairs}
    assert seen == set(PAIR_SEEDS), (label, seen)
    assert len(evaluation.pairs) == 5, label
    for pair in evaluation.pairs:
        assert math.isfinite(pair.total), (label, pair.pair_seed)
    evaluations[label] = evaluation
    print(
        f"{label}: mean total {evaluation.mean_total:.4f} "
        f"(std {evaluation.std_total:.4f})"
    )
"""


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        markdown("goal", GOAL_MARKDOWN),
        code("portable-setup", PORTABLE_SETUP),
        code("shared-imports", SHARED_IMPORTS),
        code("experiment-parameters", EXPERIMENT_PARAMETERS),
        markdown("configurations-heading", CONFIG_MARKDOWN),
        code("build-configurations", BUILD_CONFIGURATIONS),
        code("protocol-checks", PROTOCOL_CHECKS),
        code("train-models", TRAIN_MODELS),
        code("evaluate-checkpoints", EVALUATE_CHECKPOINTS),
    ]
    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )


def write_notebook(output_path: Path = DEFAULT_OUTPUT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), output_path)
    return output_path


if __name__ == "__main__":
    print(write_notebook())
