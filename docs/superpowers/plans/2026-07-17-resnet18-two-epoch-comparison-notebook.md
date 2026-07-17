# ResNet-18 Two-Epoch Comparison Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portable local/Colab notebook that trains and fully compares the EB-JEPA MLP projector and `FixedTreePredictor`, each with the same ResNet-18 backbone and exactly two CIFAR-10 epochs.

**Architecture:** A small `nbformat` builder owns the notebook source so the generated `.ipynb` is reproducible and reviewable without hand-editing JSON. The notebook delegates model, training, evaluation, checkpoint, and FLOP behavior to the existing `comparison` package; notebook-local helpers handle environment discovery, cross-device timing, tabulation, plots, and exports. A focused structural test suite validates the generated artifact and compiles every Python cell without running the expensive experiment.

**Tech Stack:** Python 3.12, Jupyter/nbformat, PyTorch 2.6, torchvision 0.21, OmegaConf, fvcore, pandas, matplotlib, pytest.

## Global Constraints

- Create `eb_jepa_cifar10_comparison/scripts/resnet18_two_epoch_comparison.ipynb` as a separate artifact.
- Do not modify `scripts/workflow.ipynb`, `configs/baseline.yaml`, or `configs/cortical.yaml`.
- Both arms use CIFAR-10, training seed `1`, batch size `256`, identical ResNet-18 backbones, VICReg, and exactly two epochs.
- Override `optimization.warmup_epochs=0` for both arms.
- Evaluate both checkpoints with pair seeds `(11, 22, 33, 44, 55)`.
- Support Google Colab, CUDA, MPS, and CPU without hard-coded `/Users/...` paths.
- Treat active-device timing as a portable benchmark, not the repository's strict A100 protocol.
- Export flat CSV tables and a schema-version-1 JSON artifact with environment, configurations, checkpoint hashes, raw evaluations, compute metrics, and derived differences.
- Do not execute the two complete training runs during repository validation.

---

### Task 1: Lock the notebook contract and portable setup

**Files:**
- Modify: `eb_jepa_cifar10_comparison/pyproject.toml`
- Modify: `eb_jepa_cifar10_comparison/uv.lock`
- Create: `eb_jepa_cifar10_comparison/tests/test_resnet18_two_epoch_notebook.py`
- Create: `eb_jepa_cifar10_comparison/scripts/build_resnet18_two_epoch_comparison.py`
- Create: `eb_jepa_cifar10_comparison/scripts/resnet18_two_epoch_comparison.ipynb`

**Interfaces:**
- Consumes: `nbformat.v4.new_notebook`, `new_markdown_cell`, and `new_code_cell`.
- Produces: `build_notebook() -> nbformat.NotebookNode` and `write_notebook(output_path: Path) -> Path`.

- [ ] **Step 1: Declare notebook presentation dependencies**

Add the two direct reader-facing dependencies to the existing `notebook`
optional dependency group:

```toml
[project.optional-dependencies]
notebook = [
    "jupyterlab",
    "ipykernel",
    "matplotlib>=3.9",
    "pandas>=2.2",
]
```

Run `uv lock` from `eb_jepa_cifar10_comparison` and preserve unrelated
existing lockfile changes.

- [ ] **Step 2: Write failing structural tests**

Create tests that load the artifact and assert stable cell IDs, portable setup, and fixed protocol parameters:

```python
import ast
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_DIR / "scripts" / "resnet18_two_epoch_comparison.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_by_id() -> dict[str, str]:
    return {
        cell["id"]: "".join(cell.get("source", []))
        for cell in _load_notebook()["cells"]
        if cell["cell_type"] == "code"
    }


def test_notebook_has_portable_setup_and_fixed_protocol() -> None:
    code = _code_by_id()
    setup = code["portable-setup"]
    parameters = code["experiment-parameters"]
    assert "google.colab" in setup
    assert "mps" in setup
    assert "/Users/" not in "\n".join(code.values())
    assert "EPOCHS = 2" in parameters
    assert "TRAINING_SEED = 1" in parameters
    assert "PAIR_SEEDS = (11, 22, 33, 44, 55)" in parameters
    assert "BATCH_SIZE = 256" in parameters


def test_every_python_cell_compiles() -> None:
    for cell_id, source in _code_by_id().items():
        ast.parse(source, filename=f"{NOTEBOOK_PATH.name}:{cell_id}")
```

- [ ] **Step 3: Run the tests and verify the contract is missing**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python -m pytest tests/test_resnet18_two_epoch_notebook.py -q
```

Expected: FAIL because `scripts/resnet18_two_epoch_comparison.ipynb` does not exist.

- [ ] **Step 4: Implement the notebook builder and initial cells**

Create a builder with this public shape:

```python
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
```

`PORTABLE_SETUP` must detect Colab with `importlib.util.find_spec("google.colab")`, search upward for `eb_jepa_cifar10_comparison`, add `src` to `sys.path`, and select CUDA → MPS → CPU. `EXPERIMENT_PARAMETERS` must define the exact constants asserted above, notebook-specific data/output roots, and portable timing repetitions.

- [ ] **Step 5: Generate the artifact and run structural tests**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python scripts/build_resnet18_two_epoch_comparison.py
.venv/bin/python -m pytest tests/test_resnet18_two_epoch_notebook.py -q
```

Expected: PASS for portable setup, fixed parameters, and code-cell syntax.

- [ ] **Step 6: Commit the contract and portable scaffold**

```bash
git add eb_jepa_cifar10_comparison/pyproject.toml eb_jepa_cifar10_comparison/uv.lock eb_jepa_cifar10_comparison/tests/test_resnet18_two_epoch_notebook.py eb_jepa_cifar10_comparison/scripts/build_resnet18_two_epoch_comparison.py eb_jepa_cifar10_comparison/scripts/resnet18_two_epoch_comparison.ipynb
git commit -m "feat: scaffold portable ResNet-18 comparison notebook"
```

### Task 2: Add fair configuration, training, and evaluation flows

**Files:**
- Modify: `eb_jepa_cifar10_comparison/tests/test_resnet18_two_epoch_notebook.py`
- Modify: `eb_jepa_cifar10_comparison/scripts/build_resnet18_two_epoch_comparison.py`
- Regenerate: `eb_jepa_cifar10_comparison/scripts/resnet18_two_epoch_comparison.ipynb`

**Interfaces:**
- Consumes: `comparison.config.load_config`, `comparison.train.run`, `comparison.evaluate.evaluate_test`, and notebook constants from Task 1.
- Produces: `configs: dict[str, DictConfig]`, `training_results: dict[str, RunResult]`, `training_seconds: dict[str, float]`, and `evaluations: dict[str, TestEvaluation]`.

- [ ] **Step 1: Add failing configuration and workflow tests**

Add these assertions:

```python
def test_notebook_builds_both_resnet18_arms_with_shared_overrides() -> None:
    code = _code_by_id()
    config_source = code["build-configurations"]
    assert '"model.backbone=resnet18"' in config_source
    assert 'f"optimization.epochs={EPOCHS}"' in config_source
    assert '"optimization.warmup_epochs=0"' in config_source
    assert '"optimization.learning_rate=0.3"' in config_source
    assert '"baseline": PROJECT_ROOT / "configs" / "baseline.yaml"' in config_source
    assert '"predictor": PROJECT_ROOT / "configs" / "cortical.yaml"' in config_source


def test_notebook_trains_and_evaluates_both_arms() -> None:
    code = _code_by_id()
    training = code["train-models"]
    evaluation = code["evaluate-checkpoints"]
    assert "for label in ARM_LABELS" in training
    assert "run(" in training
    assert "perf_counter" in training
    assert "evaluate_test(" in evaluation
    assert "pair_seeds=PAIR_SEEDS" in evaluation
```

- [ ] **Step 2: Run the focused tests to verify failure**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python -m pytest tests/test_resnet18_two_epoch_notebook.py -q
```

Expected: FAIL because the configuration, training, and evaluation cells do not exist.

- [ ] **Step 3: Implement configuration and protocol-check cells**

Add cells with IDs `build-configurations` and `protocol-checks`. The configuration cell must use one shared override list:

```python
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
```

The protocol cell must assert `model.backbone == "resnet18"`, epochs `== 2`, warm-up `== 0`, matching data/loss/optimization values, and `head_type` values `baseline` and `cortical`. Display a two-row resolved-protocol table.

- [ ] **Step 4: Implement sequential training and evaluation cells**

Add `train-models` using `perf_counter()` around `run(...)`, separate `OUTPUT_ROOT / label / str(TRAINING_SEED)` directories, and explicit checks that `final_epoch == EPOCHS - 1` and `best_checkpoint.is_file()`. Add `evaluate-checkpoints` using the same `PAIR_SEEDS` for both arms and require exactly five distinct pair results with finite components.

- [ ] **Step 5: Regenerate and run focused tests**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python scripts/build_resnet18_two_epoch_comparison.py
.venv/bin/python -m pytest tests/test_resnet18_two_epoch_notebook.py -q
```

Expected: all notebook tests PASS without training models.

- [ ] **Step 6: Commit the experiment flow**

```bash
git add eb_jepa_cifar10_comparison/tests/test_resnet18_two_epoch_notebook.py eb_jepa_cifar10_comparison/scripts/build_resnet18_two_epoch_comparison.py eb_jepa_cifar10_comparison/scripts/resnet18_two_epoch_comparison.ipynb
git commit -m "feat: add two-epoch ResNet-18 experiment flow"
```

### Task 3: Add portable profiling, complete results, and exports

**Files:**
- Modify: `eb_jepa_cifar10_comparison/tests/test_resnet18_two_epoch_notebook.py`
- Modify: `eb_jepa_cifar10_comparison/scripts/build_resnet18_two_epoch_comparison.py`
- Regenerate: `eb_jepa_cifar10_comparison/scripts/resnet18_two_epoch_comparison.ipynb`

**Interfaces:**
- Consumes: `comparison.model.build_model`, `comparison.model.count_parameters`, `comparison.benchmark.measure_flops`, completed checkpoints, and evaluation objects.
- Produces: `quality_table: pandas.DataFrame`, `compute_table: pandas.DataFrame`, `comparison_table: pandas.DataFrame`, `comparison_results.csv`, `compute_results.csv`, and `comparison_results.json`.

- [ ] **Step 1: Add failing profile, presentation, and export tests**

```python
def test_notebook_profiles_active_device_and_exports_results() -> None:
    code = _code_by_id()
    profile = code["portable-profiler"] + code["profile-models"]
    export = code["export-results"]
    assert "torch.cuda.synchronize" in profile
    assert "torch.mps.synchronize" in profile
    assert "measure_flops" in profile
    assert "count_parameters" in profile
    assert "comparison_results.csv" in export
    assert "compute_results.csv" in export
    assert "comparison_results.json" in export
    assert '"schema_version": 1' in export


def test_notebook_contains_tables_plots_takeaways_and_final_checks() -> None:
    code = _code_by_id()
    assert "quality_table" in code["build-result-tables"]
    assert "compute_table" in code["build-result-tables"]
    assert "plt.subplots" in code["plot-results"]
    assert "percentage" in code["derive-takeaways"]
    assert "math.isfinite" in code["final-checks"]
```

- [ ] **Step 2: Run tests to verify missing result cells**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python -m pytest tests/test_resnet18_two_epoch_notebook.py -q
```

Expected: FAIL because profiling, result, plot, export, and final-check cells are absent.

- [ ] **Step 3: Implement cross-device synchronization and timing helpers**

Add a `portable-profiler` cell defining:

```python
def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_portable_latency(call, batch_size: int) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(BENCHMARK_WARMUPS):
            call()
        synchronize(DEVICE)
        start = perf_counter()
        for _ in range(BENCHMARK_MEASUREMENTS):
            call()
        synchronize(DEVICE)
    latency_ms = (
        (perf_counter() - start) * 1_000.0 / BENCHMARK_MEASUREMENTS
    )
    return latency_ms, batch_size / (latency_ms / 1_000.0)
```

The profiler must use identical random inputs and feature tensors for both arms, load the best checkpoint strictly, call `count_parameters`, and catch FLOP-analysis exceptions as explicit caveat strings with `math.nan` values.

- [ ] **Step 4: Implement result tables, plots, and takeaways**

Build quality rows from `training_results`, `training_seconds`, and `evaluations`; build head/full compute rows from the profile. Add absolute and percentage predictor-versus-baseline changes with a zero-denominator guard. Display tables with units in column names. Plot VICReg components and normalized compute ratios. Generate concise text from the observed values and state that two epochs do not imply convergence.

- [ ] **Step 5: Implement JSON/CSV export and final checks**

Write `comparison_results.csv`, `compute_results.csv`, and `comparison_results.json` below `OUTPUT_ROOT / "reports"`. JSON serialization must include resolved OmegaConf containers, `sha256` checkpoint hashes, `asdict()` evaluation values, device/dtype/timing protocol, compute caveats, and derived differences. The `final-checks` cell must assert both checkpoints exist, both runs ended at epoch index `1`, five distinct pair seeds exist per arm, required non-FLOP values are finite, and all three exports exist.

- [ ] **Step 6: Regenerate and run the notebook test suite**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python scripts/build_resnet18_two_epoch_comparison.py
.venv/bin/python -m pytest tests/test_resnet18_two_epoch_notebook.py -q
```

Expected: all notebook tests PASS.

- [ ] **Step 7: Commit profiling and reporting**

```bash
git add eb_jepa_cifar10_comparison/tests/test_resnet18_two_epoch_notebook.py eb_jepa_cifar10_comparison/scripts/build_resnet18_two_epoch_comparison.py eb_jepa_cifar10_comparison/scripts/resnet18_two_epoch_comparison.ipynb
git commit -m "feat: report full portable predictor comparison"
```

### Task 4: Document and verify the runnable handoff

**Files:**
- Modify: `eb_jepa_cifar10_comparison/README.md`
- Verify: `eb_jepa_cifar10_comparison/scripts/resnet18_two_epoch_comparison.ipynb`
- Verify: `eb_jepa_cifar10_comparison/tests/test_resnet18_two_epoch_notebook.py`

**Interfaces:**
- Consumes: generated notebook and existing `uv` environment.
- Produces: documented local and Colab launch instructions plus final verification evidence.

- [ ] **Step 1: Add failing README assertions**

```python
def test_readme_documents_local_and_colab_execution() -> None:
    readme = (PROJECT_DIR / "README.md").read_text(encoding="utf-8")
    assert "resnet18_two_epoch_comparison.ipynb" in readme
    assert "uv run jupyter lab" in readme
    assert "Google Colab" in readme
    assert "two epochs" in readme.lower()
```

- [ ] **Step 2: Run the README test and verify failure**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python -m pytest tests/test_resnet18_two_epoch_notebook.py::test_readme_documents_local_and_colab_execution -q
```

Expected: FAIL because the new notebook is not documented.

- [ ] **Step 3: Document local and Colab usage**

Add a README subsection that states the two-epoch interpretation limit and gives:

```bash
uv sync --extra notebook
uv run jupyter lab scripts/resnet18_two_epoch_comparison.ipynb
```

For Colab, instruct the reader to clone or upload the repository under `/content`, open the same notebook, install the project dependencies in the runtime, and restart the runtime only if Colab requests it.

- [ ] **Step 4: Run final focused and regression tests**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python scripts/build_resnet18_two_epoch_comparison.py
.venv/bin/python -m pytest tests/test_resnet18_two_epoch_notebook.py tests/test_model_backbone_config.py tests/test_backbone.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 5: Validate notebook structure with nbformat**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python -c 'import nbformat; p="scripts/resnet18_two_epoch_comparison.ipynb"; nb=nbformat.read(p, as_version=4); nbformat.validate(nb); print(len(nb.cells))'
```

Expected: prints a positive cell count and exits 0.

- [ ] **Step 6: Confirm protected files were not changed by this work**

Run:

```bash
git diff --name-only HEAD -- eb_jepa_cifar10_comparison/scripts/workflow.ipynb eb_jepa_cifar10_comparison/configs/baseline.yaml eb_jepa_cifar10_comparison/configs/cortical.yaml
```

Expected: no new diff attributable to this implementation; preserve any pre-existing user changes.

- [ ] **Step 7: Commit documentation and final verification changes**

```bash
git add eb_jepa_cifar10_comparison/README.md eb_jepa_cifar10_comparison/tests/test_resnet18_two_epoch_notebook.py
git commit -m "docs: explain two-epoch comparison workflow"
```

The full experiment remains intentionally unexecuted during implementation. To execute it after setup, run:

```bash
cd eb_jepa_cifar10_comparison
uv run jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=-1 scripts/resnet18_two_epoch_comparison.ipynb
```
