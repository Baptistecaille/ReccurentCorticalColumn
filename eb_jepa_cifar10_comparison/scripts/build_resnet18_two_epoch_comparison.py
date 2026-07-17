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


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        markdown("goal", GOAL_MARKDOWN),
        code("portable-setup", PORTABLE_SETUP),
        code("shared-imports", SHARED_IMPORTS),
        code("experiment-parameters", EXPERIMENT_PARAMETERS),
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
