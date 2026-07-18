# Parameter-Matched Predictor Arm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third `predictor_matched` arm whose head parameter count matches the baseline MLP projector within 1%, so the observed quality gap can be split into a capacity component and an architecture component.

**Architecture:** A new `configs/cortical_matched.yaml` carries the two changed cortical dimensions, which keeps `SHARED_OVERRIDES` strictly identical across all three arms. The notebook builder generalises from two hard-coded arms to an `ARM_LABELS` tuple of three, gains a fail-fast parameter-parity cell before training, and switches `comparison_table` from a baseline/predictor pair to one row per (arm, metric).

**Tech Stack:** Python 3.12, nbformat, PyTorch 2.6, OmegaConf, pandas, matplotlib, pytest.

## Global Constraints

- All notebook edits go to `scripts/build_resnet18_two_epoch_comparison.py`; the `.ipynb` is regenerated, never hand-edited.
- Do not modify `configs/baseline.yaml`, `configs/cortical.yaml`, or `scripts/workflow.ipynb`.
- The matched arm differs from `predictor` on exactly two values: `cortical.dim_U` 256 → 512 and `cortical.dim_feedback` 128 → 168.
- Target head parameter counts: baseline `9,451,520`; predictor `6,508,288`; matched `9,447,920` (gap 0.038%).
- `PARITY_TOLERANCE` is `0.01` (1%).
- Tests that load `configs/baseline.yaml` MUST pass `optimization.learning_rate=0.3` and `optimization.warmup_start_lr=0.00003` as overrides; the file carries uncommitted working-tree edits that otherwise fail `validate_config`.
- Do not execute the training runs during repository validation.
- Run all commands from `eb_jepa_cifar10_comparison/` using `.venv/bin/python`.

---

### Task 1: Add the parameter-matched configuration

**Files:**
- Create: `eb_jepa_cifar10_comparison/configs/cortical_matched.yaml`
- Create: `eb_jepa_cifar10_comparison/tests/test_parameter_parity.py`

**Interfaces:**
- Consumes: `comparison.config.load_config`, `comparison.model.build_model`, `comparison.model.count_parameters`.
- Produces: `configs/cortical_matched.yaml`, consumed by Task 2's `CONFIG_PATHS`.

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_parameter_parity.py`:

```python
from pathlib import Path

from comparison.config import load_config
from comparison.model import build_model, count_parameters


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs"

# baseline.yaml carries working-tree edits that fail validate_config without
# these; head parameter counts are unaffected by optimization values.
PROTOCOL_OVERRIDES = [
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

    for key in ("n", "L_max", "dim_in", "dim_hidden", "dim_target"):
        assert original.cortical[key] == matched.cortical[key], key
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python -m pytest tests/test_parameter_parity.py -q
```

Expected: FAIL with `FileNotFoundError: Config file not found: .../configs/cortical_matched.yaml`.

- [ ] **Step 3: Create the matched configuration**

Create `configs/cortical_matched.yaml`. It is `cortical.yaml` with `dim_U: 512`
and `dim_feedback: 168`; every other value is identical.

```yaml
# Parameter-matched variant of cortical.yaml.
#
# Differs from cortical.yaml on exactly two values so the FixedTreePredictor
# head matches the baseline MLP projector's 9,451,520 parameters:
#   cortical.dim_U:        256 -> 512   (primary capacity step)
#   cortical.dim_feedback: 128 -> 168   (fine adjustment, ~4,600 params/unit)
# Resulting head size: 9,447,920 parameters (0.038% below baseline).
# The tree topology (n, L_max, dim_hidden) is unchanged; only its width grows.

data:
  dataset: cifar10
  root: ./data
  download: true
  train_size: 45000
  validation_size: 5000
  test_size: 10000
  batch_size: 256
  num_workers: 4
  pin_memory: true
  split_seed: 42
  crop_scale: [0.2, 1.0]

model:
  head_type: cortical
  backbone: mlp
  feature_dim: 512
  mlp_hidden_dim: 2048
  output_dim: 2048

cortical:
  n: 2
  L_max: 2
  dim_in: 512
  dim_hidden: 256
  dim_feedback: 168
  dim_U: 512
  dim_target: 2048
  disable_lateral: false
  disable_feedback: false

loss:
  type: vicreg
  invariance_coeff: 1.0
  std_coeff: 1.0
  cov_coeff: 80.0
  variance_target: 1.0
  variance_epsilon: 0.0001
  collapse_threshold: 0.1

optimization:
  optimizer: lars
  epochs: 50
  learning_rate: 0.3
  warmup_epochs: 10
  warmup_start_lr: 0.00003
  min_lr: 0.0
  weight_decay: 0.0001
  momentum: 0.9
  precision: bfloat16

experiment:
  seeds: [1, 1000, 10000]
  checkpoint_every: 50

benchmark:
  required_gpu: NVIDIA A100
  batch_size: 256
  warmups: 100
  measurements: 500
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python -m pytest tests/test_parameter_parity.py -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add eb_jepa_cifar10_comparison/configs/cortical_matched.yaml eb_jepa_cifar10_comparison/tests/test_parameter_parity.py
git commit -m "feat: add parameter-matched cortical configuration"
```

### Task 2: Register the third arm in the notebook

**Files:**
- Modify: `eb_jepa_cifar10_comparison/tests/test_resnet18_two_epoch_notebook.py`
- Modify: `eb_jepa_cifar10_comparison/scripts/build_resnet18_two_epoch_comparison.py`
- Regenerate: `eb_jepa_cifar10_comparison/scripts/resnet18_two_epoch_comparison.ipynb`

**Interfaces:**
- Consumes: `configs/cortical_matched.yaml` from Task 1.
- Produces: `ARM_LABELS = ("baseline", "predictor", "predictor_matched")` and a three-entry `configs` dict, both consumed by Tasks 3 and 4.

- [ ] **Step 1: Add the failing third-arm tests**

Append to `tests/test_resnet18_two_epoch_notebook.py`:

```python
def test_notebook_declares_three_arms_including_matched_predictor() -> None:
    code = _code_by_id()
    parameters = code["experiment-parameters"]
    config_source = code["build-configurations"]
    assert 'ARM_LABELS = ("baseline", "predictor", "predictor_matched")' in parameters
    assert (
        '"predictor_matched": PROJECT_ROOT / "configs" / "cortical_matched.yaml"'
        in config_source
    )


def test_notebook_compares_every_arm_against_the_baseline() -> None:
    checks = _code_by_id()["protocol-checks"]
    assert '"predictor_matched": "cortical"' in checks
    assert "for label in ARM_LABELS" in checks
    assert 'if label == "baseline"' in checks
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python -m pytest tests/test_resnet18_two_epoch_notebook.py -q
```

Expected: FAIL — `ARM_LABELS = ("baseline", "predictor")` is still a two-tuple.

- [ ] **Step 3: Extend `ARM_LABELS`**

In `scripts/build_resnet18_two_epoch_comparison.py`, inside the
`EXPERIMENT_PARAMETERS` string, replace:

```python
ARM_LABELS = ("baseline", "predictor")
```

with:

```python
ARM_LABELS = ("baseline", "predictor", "predictor_matched")
```

- [ ] **Step 4: Add the third config path**

In the `BUILD_CONFIGURATIONS` string, replace the `CONFIG_PATHS` dict with:

```python
CONFIG_PATHS = {
    "baseline": PROJECT_ROOT / "configs" / "baseline.yaml",
    "predictor": PROJECT_ROOT / "configs" / "cortical.yaml",
    "predictor_matched": PROJECT_ROOT / "configs" / "cortical_matched.yaml",
}
```

`SHARED_OVERRIDES` and the `configs` comprehension below it are unchanged — the
same override list applies to all three arms.

- [ ] **Step 5: Generalise the protocol checks to N arms**

In the `PROTOCOL_CHECKS` string, replace everything from the `EXPECTED_HEADS = {`
line through the final `assert left == right, f"Mismatch on {path}: ..."` line —
that is, the old `EXPECTED_HEADS` dict, the `for label, cfg in configs.items():`
validation loop, and the `baseline_cfg, predictor_cfg = ...` agreement block —
with the following. Leave the `protocol_table = pd.DataFrame(` block that
follows untouched; it already iterates `configs.items()` and generalises to
three arms on its own.

```python
EXPECTED_HEADS = {
    "baseline": "baseline",
    "predictor": "cortical",
    "predictor_matched": "cortical",
}

SHARED_PROTOCOL_PATHS = (
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
)

for label in ARM_LABELS:
    cfg = configs[label]
    assert str(cfg.model.backbone).lower() == "resnet18", label
    assert int(cfg.optimization.epochs) == EPOCHS, label
    assert int(cfg.optimization.warmup_epochs) == 0, label
    assert int(cfg.data.batch_size) == BATCH_SIZE, label
    assert str(cfg.model.head_type).lower() == EXPECTED_HEADS[label], label

    # Every arm must agree with the baseline on the shared protocol values.
    # Intentional differences live under cortical.*, which is not compared here.
    if label == "baseline":
        continue
    for path in SHARED_PROTOCOL_PATHS:
        left = OmegaConf.select(configs["baseline"], path)
        right = OmegaConf.select(cfg, path)
        assert left == right, f"{label} mismatch on {path}: {left} != {right}"
```

Delete the now-superseded `for label, cfg in configs.items():` loop and the
`baseline_cfg, predictor_cfg = ...` block that preceded it.

- [ ] **Step 6: Regenerate and run the tests**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python scripts/build_resnet18_two_epoch_comparison.py
.venv/bin/python -m pytest tests/test_resnet18_two_epoch_notebook.py -q
```

Expected: all notebook tests PASS.

- [ ] **Step 7: Commit**

```bash
git add eb_jepa_cifar10_comparison/tests/test_resnet18_two_epoch_notebook.py eb_jepa_cifar10_comparison/scripts/build_resnet18_two_epoch_comparison.py eb_jepa_cifar10_comparison/scripts/resnet18_two_epoch_comparison.ipynb
git commit -m "feat: register parameter-matched arm in comparison notebook"
```

### Task 3: Add the fail-fast parameter-parity cell

**Files:**
- Modify: `eb_jepa_cifar10_comparison/tests/test_resnet18_two_epoch_notebook.py`
- Modify: `eb_jepa_cifar10_comparison/scripts/build_resnet18_two_epoch_comparison.py`
- Regenerate: `eb_jepa_cifar10_comparison/scripts/resnet18_two_epoch_comparison.ipynb`

**Interfaces:**
- Consumes: `configs` and `ARM_LABELS` from Task 2; `comparison.model.build_model` and `comparison.model.count_parameters`, already imported by the `shared-imports` cell.
- Produces: `head_parameters: dict[str, int]` and `parity_table: pandas.DataFrame`.

- [ ] **Step 1: Add the failing parity-cell test**

Append to `tests/test_resnet18_two_epoch_notebook.py`:

```python
def test_notebook_checks_parameter_parity_before_training() -> None:
    notebook = _load_notebook()
    ids = [cell["id"] for cell in notebook["cells"]]
    assert ids.index("parameter-parity") < ids.index("train-models")

    parity = _code_by_id()["parameter-parity"]
    assert "PARITY_TOLERANCE = 0.01" in parity
    assert "count_parameters" in parity
    assert "build_model" in parity
    assert "assert relative_gap <= PARITY_TOLERANCE" in parity
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python -m pytest tests/test_resnet18_two_epoch_notebook.py::test_notebook_checks_parameter_parity_before_training -q
```

Expected: FAIL with `KeyError: 'parameter-parity'`.

- [ ] **Step 3: Define the parity cell source**

In `scripts/build_resnet18_two_epoch_comparison.py`, add this string immediately
after the `PROTOCOL_CHECKS` definition:

```python
PARAMETER_PARITY = """
# Fail fast: verify head-parameter parity BEFORE spending GPU time on training.
# Building the models is instantaneous and trains nothing.
PARITY_TOLERANCE = 0.01  # 1%

head_parameters = {}
backbone_parameters = {}
for label in ARM_LABELS:
    counts = count_parameters(build_model(configs[label]))
    head_parameters[label] = counts.head
    backbone_parameters[label] = counts.backbone

# All arms share the identical ResNet-18 backbone.
assert len(set(backbone_parameters.values())) == 1, backbone_parameters

relative_gap = (
    abs(head_parameters["predictor_matched"] - head_parameters["baseline"])
    / head_parameters["baseline"]
)
assert relative_gap <= PARITY_TOLERANCE, (head_parameters, relative_gap)

parity_table = pd.DataFrame(
    [
        {
            "arm": label,
            "head_parameters": head_parameters[label],
            "backbone_parameters": backbone_parameters[label],
            "total_parameters": head_parameters[label] + backbone_parameters[label],
            "head_vs_baseline_%": (
                (head_parameters[label] - head_parameters["baseline"])
                / head_parameters["baseline"]
                * 100.0
            ),
        }
        for label in ARM_LABELS
    ]
)
print(f"Matched-arm gap vs baseline: {relative_gap * 100:.3f}%")
parity_table
"""
```

- [ ] **Step 4: Insert the cell into the notebook**

In `build_notebook()`, insert the parity cell between `protocol-checks` and
`train-models`:

```python
        code("protocol-checks", PROTOCOL_CHECKS),
        code("parameter-parity", PARAMETER_PARITY),
        code("train-models", TRAIN_MODELS),
```

- [ ] **Step 5: Regenerate and run the tests**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python scripts/build_resnet18_two_epoch_comparison.py
.venv/bin/python -m pytest tests/test_resnet18_two_epoch_notebook.py -q
```

Expected: all notebook tests PASS.

- [ ] **Step 6: Commit**

```bash
git add eb_jepa_cifar10_comparison/tests/test_resnet18_two_epoch_notebook.py eb_jepa_cifar10_comparison/scripts/build_resnet18_two_epoch_comparison.py eb_jepa_cifar10_comparison/scripts/resnet18_two_epoch_comparison.ipynb
git commit -m "feat: guard head-parameter parity before training"
```

### Task 4: Make results, plots, and takeaways N-arm aware

**Files:**
- Modify: `eb_jepa_cifar10_comparison/tests/test_resnet18_two_epoch_notebook.py`
- Modify: `eb_jepa_cifar10_comparison/scripts/build_resnet18_two_epoch_comparison.py`
- Regenerate: `eb_jepa_cifar10_comparison/scripts/resnet18_two_epoch_comparison.ipynb`

**Interfaces:**
- Consumes: `evaluations`, `training_seconds`, `profiles`, `means_by_arm`, and `ARM_LABELS`.
- Produces: `comparison_table` with columns `arm`, `metric`, `baseline`, `arm_value`, `absolute_change`, `percentage_change`; plus `arm_metric_values(label) -> dict[str, float]`, `BASELINE_METRICS`, and `COMPARISON_ARMS`, all consumed by `plot-results` and `derive-takeaways`.

- [ ] **Step 1: Add the failing row-orientation tests**

Append to `tests/test_resnet18_two_epoch_notebook.py`:

```python
def test_comparison_table_is_row_oriented_per_arm() -> None:
    code = _code_by_id()
    tables = code["build-result-tables"]
    assert "def arm_metric_values(label: str)" in tables
    assert "COMPARISON_ARMS" in tables
    assert '"arm": label' in tables
    assert '"arm_value": value' in tables
    assert "COMPARISON_METRICS" not in tables


def test_plots_and_takeaways_consume_the_arm_column() -> None:
    code = _code_by_id()
    assert "COMPARISON_ARMS" in code["plot-results"]
    assert "row.arm" in code["derive-takeaways"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python -m pytest tests/test_resnet18_two_epoch_notebook.py -q
```

Expected: FAIL — `COMPARISON_METRICS` is still present and `arm_metric_values` is undefined.

- [ ] **Step 3: Replace the comparison block**

In the `BUILD_RESULT_TABLES` string, replace the whole block from
`COMPARISON_METRICS = {` through the closing `)` of the `comparison_table`
assignment with:

```python
COMPARISON_ARMS = tuple(label for label in ARM_LABELS if label != "baseline")


def arm_metric_values(label: str) -> dict[str, float]:
    return {
        "test_total_mean": evaluations[label].mean_total,
        "train_time_s": training_seconds[label],
        "head_parameters": profiles[label]["params_head"],
        "full_flops_per_sample": profiles[label]["full_flops_per_sample"],
        "full_latency_ms": profiles[label]["full_latency_ms"],
    }


BASELINE_METRICS = arm_metric_values("baseline")
comparison_table = pd.DataFrame(
    [
        {
            "arm": label,
            "metric": metric,
            "baseline": BASELINE_METRICS[metric],
            "arm_value": value,
            "absolute_change": value - BASELINE_METRICS[metric],
            "percentage_change": percentage_change(value, BASELINE_METRICS[metric]),
        }
        for label in COMPARISON_ARMS
        for metric, value in arm_metric_values(label).items()
    ]
)
```

Leave `percentage_change`, `quality_table`, `compute_table`, and the trailing
`display(...)` lines unchanged.

- [ ] **Step 4: Generalise both plots**

In the `PLOT_RESULTS` string, replace `bar_width = 0.35` with:

```python
bar_width = 0.8 / len(ARM_LABELS)
```

and replace the entire ratio-chart block (from `ratio_metrics = [` to
`axes[1].tick_params(...)`) with:

```python
ratio_metrics = ["head_parameters", "full_flops_per_sample", "full_latency_ms"]
ratio_width = 0.8 / len(COMPARISON_ARMS)
ratio_positions = range(len(ratio_metrics))
for offset, label in enumerate(COMPARISON_ARMS):
    values = arm_metric_values(label)
    ratios = [
        values[metric] / BASELINE_METRICS[metric]
        if BASELINE_METRICS[metric] and math.isfinite(BASELINE_METRICS[metric])
        else math.nan
        for metric in ratio_metrics
    ]
    axes[1].bar(
        [p + offset * ratio_width for p in ratio_positions],
        ratios,
        width=ratio_width,
        label=label,
    )
axes[1].set_xticks(
    [p + ratio_width * (len(COMPARISON_ARMS) - 1) / 2 for p in ratio_positions]
)
axes[1].set_xticklabels(ratio_metrics)
axes[1].axhline(1.0, color="grey", linestyle="--", linewidth=1)
axes[1].set_ylabel("arm / baseline")
axes[1].set_title("Normalised compute ratios")
axes[1].legend()
axes[1].tick_params(axis="x", rotation=20)
```

- [ ] **Step 5: Update the takeaways to name each arm**

In the `DERIVE_TAKEAWAYS` string, replace the two `lines.append(...)` calls
inside the `for row in comparison_table.itertuples(index=False):` loop with:

```python
    if math.isfinite(percentage):
        lines.append(
            f"- [{row.arm}] {row.metric}: baseline {row.baseline:.4g} -> "
            f"{row.arm_value:.4g} ({percentage:+.1f}% percentage change)"
        )
    else:
        lines.append(
            f"- [{row.arm}] {row.metric}: baseline {row.baseline:.4g} -> "
            f"{row.arm_value:.4g} (percentage change undefined)"
        )
```

- [ ] **Step 6: Regenerate and run the full notebook suite**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python scripts/build_resnet18_two_epoch_comparison.py
.venv/bin/python -m pytest tests/test_resnet18_two_epoch_notebook.py tests/test_parameter_parity.py -q
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add eb_jepa_cifar10_comparison/tests/test_resnet18_two_epoch_notebook.py eb_jepa_cifar10_comparison/scripts/build_resnet18_two_epoch_comparison.py eb_jepa_cifar10_comparison/scripts/resnet18_two_epoch_comparison.ipynb
git commit -m "feat: report every predictor arm against the baseline"
```

### Task 5: Document the third arm and verify the handoff

**Files:**
- Modify: `eb_jepa_cifar10_comparison/README.md`
- Modify: `eb_jepa_cifar10_comparison/tests/test_resnet18_two_epoch_notebook.py`
- Verify: `eb_jepa_cifar10_comparison/scripts/resnet18_two_epoch_comparison.ipynb`

**Interfaces:**
- Consumes: the regenerated notebook and `configs/cortical_matched.yaml`.
- Produces: README documentation of the three arms and final verification evidence.

- [ ] **Step 1: Add the failing README test**

Append to `tests/test_resnet18_two_epoch_notebook.py`:

```python
def test_readme_documents_the_parameter_matched_arm() -> None:
    readme = (PROJECT_DIR / "README.md").read_text(encoding="utf-8")
    assert "cortical_matched.yaml" in readme
    assert "predictor_matched" in readme
    assert "9 447 920" in readme or "9,447,920" in readme
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python -m pytest tests/test_resnet18_two_epoch_notebook.py::test_readme_documents_the_parameter_matched_arm -q
```

Expected: FAIL — the README does not mention the matched arm.

- [ ] **Step 3: Document the three arms**

In `README.md`, inside the `## Comparaison ResNet-18 sur deux epochs` section,
insert this subsection immediately before `### Exécution locale`:

```markdown
### Les trois bras

| Bras | Config | Tête | Paramètres tête |
|---|---|---|---|
| `baseline` | `configs/baseline.yaml` | Projecteur MLP EB-JEPA | 9 451 520 |
| `predictor` | `configs/cortical.yaml` | `FixedTreePredictor` | 6 508 288 (−31,1 %) |
| `predictor_matched` | `configs/cortical_matched.yaml` | `FixedTreePredictor` élargi | 9 447 920 (−0,038 %) |

`cortical_matched.yaml` ne diffère de `cortical.yaml` que par `cortical.dim_U`
(256 → 512) et `cortical.dim_feedback` (128 → 168) : la topologie de l’arbre est
inchangée, seule sa largeur augmente. Comparer `predictor` et
`predictor_matched` à la même baseline sépare l’effet de capacité de l’effet
d’architecture.

La cellule `parameter-parity` vérifie l’appariement à 1 % près **avant** tout
entraînement : si une dimension est modifiée et casse l’appariement, le notebook
échoue en quelques secondes plutôt qu’après trois entraînements.
```

- [ ] **Step 4: Run the full test suite for this feature**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python scripts/build_resnet18_two_epoch_comparison.py
.venv/bin/python -m pytest tests/test_resnet18_two_epoch_notebook.py tests/test_parameter_parity.py tests/test_backbone.py -q
```

Expected: all selected tests PASS.

Note: `tests/test_model_backbone_config.py` and `tests/test_workflow_notebook.py`
fail for pre-existing reasons unrelated to this work (uncommitted `baseline.yaml`
edits and a committed `workflow.ipynb`/test mismatch). Do not attempt to fix
them here; both touch protected files.

- [ ] **Step 5: Validate the notebook structure**

Run:

```bash
cd eb_jepa_cifar10_comparison
.venv/bin/python -c 'import nbformat; p="scripts/resnet18_two_epoch_comparison.ipynb"; nb=nbformat.read(p, as_version=4); nbformat.validate(nb); print(len(nb.cells))'
```

Expected: prints `20` and exits 0.

- [ ] **Step 6: Confirm protected files were not changed**

Run:

```bash
git diff --name-only HEAD -- eb_jepa_cifar10_comparison/scripts/workflow.ipynb eb_jepa_cifar10_comparison/configs/baseline.yaml eb_jepa_cifar10_comparison/configs/cortical.yaml
```

Expected: only the pre-existing user modifications appear; nothing attributable
to this implementation.

- [ ] **Step 7: Commit**

```bash
git add eb_jepa_cifar10_comparison/README.md eb_jepa_cifar10_comparison/tests/test_resnet18_two_epoch_notebook.py
git commit -m "docs: document the parameter-matched predictor arm"
```

---

To run the three-arm experiment after implementation:

```bash
cd eb_jepa_cifar10_comparison
uv run jupyter lab scripts/resnet18_two_epoch_comparison.ipynb
```

Expected runtime is roughly 10 minutes on a Colab GPU.
