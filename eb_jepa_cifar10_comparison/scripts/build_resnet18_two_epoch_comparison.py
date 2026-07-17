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


PROFILING_MARKDOWN = """
## Portable compute profiling

Parameter counts and per-sample FLOPs are exact; latency and throughput are
measured on the **active device** as a portable benchmark. Both arms are profiled
with identical random image and feature tensors so head cost is isolated fairly.
"""


PORTABLE_PROFILER = """
# Cross-device timing helpers. Synchronisation is a no-op on CPU and applies the
# correct barrier on CUDA or Apple MPS before and after the timed region.
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
    latency_ms = (perf_counter() - start) * 1_000.0 / BENCHMARK_MEASUREMENTS
    return latency_ms, batch_size / (latency_ms / 1_000.0)


def safe_flops_per_sample(module, example) -> tuple[float, str | None]:
    try:
        return measure_flops(module, example), None
    except Exception as error:  # fvcore may not trace every op on every device
        return math.nan, f"FLOP analysis unavailable: {type(error).__name__}: {error}"
"""


PROFILE_MODELS = """
# Profile both arms on identical inputs. Weights come strictly from the best
# checkpoint; parameters and FLOPs are exact, timing is device-local.
FEATURE_DIM = int(configs["baseline"].model.feature_dim)
torch.manual_seed(0)
profile_images = torch.randn(BATCH_SIZE, 3, 32, 32, device=DEVICE)
profile_features = torch.randn(BATCH_SIZE, FEATURE_DIM, device=DEVICE)

profiles = {}
compute_caveats = {}

for label in ARM_LABELS:
    cfg = configs[label]
    model = build_model(cfg).to(DEVICE)
    checkpoint = torch.load(
        training_results[label].best_checkpoint,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    params = count_parameters(model)
    head_flops, head_caveat = safe_flops_per_sample(model.head, profile_features)
    full_flops, full_caveat = safe_flops_per_sample(model, profile_images)
    head_latency, head_throughput = measure_portable_latency(
        lambda m=model: m.head(profile_features), BATCH_SIZE
    )
    full_latency, full_throughput = measure_portable_latency(
        lambda m=model: m(profile_images), BATCH_SIZE
    )

    profiles[label] = {
        "params_backbone": params.backbone,
        "params_head": params.head,
        "params_total": params.total,
        "head_flops_per_sample": head_flops,
        "full_flops_per_sample": full_flops,
        "head_latency_ms": head_latency,
        "full_latency_ms": full_latency,
        "head_throughput": head_throughput,
        "full_throughput": full_throughput,
    }
    caveats = [note for note in (head_caveat, full_caveat) if note]
    if caveats:
        compute_caveats[label] = caveats
    print(
        f"{label}: head params {params.head:,} | "
        f"full latency {full_latency:.2f} ms"
    )
"""


RESULTS_MARKDOWN = """
## Results, differences, and plots

Quality and compute tables carry units in their column names. The comparison
table reports predictor-versus-baseline absolute and percentage changes with a
zero-denominator guard.
"""


BUILD_RESULT_TABLES = """
# Assemble the reader-facing tables from training, evaluation, and profile data.
def component_means(label: str) -> dict[str, float]:
    pairs = evaluations[label].pairs
    return {
        "invariance": sum(p.invariance for p in pairs) / len(pairs),
        "variance": sum(p.variance for p in pairs) / len(pairs),
        "covariance": sum(p.covariance for p in pairs) / len(pairs),
    }


means_by_arm = {label: component_means(label) for label in ARM_LABELS}

quality_table = pd.DataFrame(
    [
        {
            "arm": label,
            "head_type": str(configs[label].model.head_type),
            "train_time (s)": training_seconds[label],
            "best_val_loss": training_results[label].best_validation,
            "test_total_mean": evaluations[label].mean_total,
            "test_total_std": evaluations[label].std_total,
            "test_invariance_mean": means_by_arm[label]["invariance"],
            "test_variance_mean": means_by_arm[label]["variance"],
            "test_covariance_mean": means_by_arm[label]["covariance"],
        }
        for label in ARM_LABELS
    ]
)

compute_rows = []
for label in ARM_LABELS:
    profile = profiles[label]
    for component, params_key, flops_key, latency_key, throughput_key in (
        ("head", "params_head", "head_flops_per_sample", "head_latency_ms", "head_throughput"),
        ("full", "params_total", "full_flops_per_sample", "full_latency_ms", "full_throughput"),
    ):
        compute_rows.append(
            {
                "arm": label,
                "component": component,
                "parameters": profile[params_key],
                "flops_per_sample": profile[flops_key],
                "latency (ms)": profile[latency_key],
                "throughput (samples/s)": profile[throughput_key],
            }
        )
compute_table = pd.DataFrame(compute_rows)


def percentage_change(new: float, old: float) -> float:
    if old == 0 or not math.isfinite(old) or not math.isfinite(new):
        return math.nan
    return (new - old) / abs(old) * 100.0


COMPARISON_METRICS = {
    "test_total_mean": (evaluations["baseline"].mean_total, evaluations["predictor"].mean_total),
    "train_time_s": (training_seconds["baseline"], training_seconds["predictor"]),
    "head_parameters": (profiles["baseline"]["params_head"], profiles["predictor"]["params_head"]),
    "full_flops_per_sample": (profiles["baseline"]["full_flops_per_sample"], profiles["predictor"]["full_flops_per_sample"]),
    "full_latency_ms": (profiles["baseline"]["full_latency_ms"], profiles["predictor"]["full_latency_ms"]),
}
comparison_table = pd.DataFrame(
    [
        {
            "metric": metric,
            "baseline": base,
            "predictor": pred,
            "absolute_change": pred - base,
            "percentage_change": percentage_change(pred, base),
        }
        for metric, (base, pred) in COMPARISON_METRICS.items()
    ]
)

display(quality_table)
display(compute_table)
comparison_table
"""


PLOT_RESULTS = """
# Visualise VICReg components per arm and predictor/baseline compute ratios.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

components = ["invariance", "variance", "covariance"]
bar_width = 0.35
positions = range(len(components))
for offset, label in enumerate(ARM_LABELS):
    values = [means_by_arm[label][component] for component in components]
    axes[0].bar(
        [p + offset * bar_width for p in positions],
        values,
        width=bar_width,
        label=label,
    )
axes[0].set_xticks([p + bar_width / 2 for p in positions])
axes[0].set_xticklabels(components)
axes[0].set_ylabel("mean component value")
axes[0].set_title("VICReg components (test)")
axes[0].legend()

ratio_metrics = ["head_parameters", "full_flops_per_sample", "full_latency_ms"]
ratios = []
for metric in ratio_metrics:
    base, pred = COMPARISON_METRICS[metric]
    ratios.append(pred / base if base not in (0, None) and math.isfinite(base) else math.nan)
axes[1].bar(ratio_metrics, ratios, color="#4c72b0")
axes[1].axhline(1.0, color="grey", linestyle="--", linewidth=1)
axes[1].set_ylabel("predictor / baseline")
axes[1].set_title("Normalised compute ratios")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.show()
"""


DERIVE_TAKEAWAYS = """
# Plain-language takeaways derived from the observed values. Each statement is a
# percentage change relative to the baseline; positive means the predictor is
# larger or slower on that metric.
lines = []
for row in comparison_table.itertuples(index=False):
    percentage = row.percentage_change
    if math.isfinite(percentage):
        lines.append(
            f"- {row.metric}: baseline {row.baseline:.4g} -> predictor "
            f"{row.predictor:.4g} ({percentage:+.1f}% percentage change)"
        )
    else:
        lines.append(
            f"- {row.metric}: baseline {row.baseline:.4g} -> predictor "
            f"{row.predictor:.4g} (percentage change undefined)"
        )

if compute_caveats:
    lines.append(f"- FLOP caveats: {compute_caveats}")

lines.append(
    "- Interpretation limit: two epochs is a fast portable comparison and does "
    "NOT imply either arm has converged."
)
print("\\n".join(lines))
"""


EXPORT_MARKDOWN = """
## Export artifacts

Flat CSV tables plus a schema-version-1 JSON artifact capturing the environment,
resolved configurations, checkpoint hashes, raw evaluations, compute metrics,
caveats, and derived differences.
"""


EXPORT_RESULTS = """
# Persist portable, self-describing result artifacts under runs/.../reports.
REPORTS_DIR = OUTPUT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


quality_table.to_csv(REPORTS_DIR / "comparison_results.csv", index=False)
compute_table.to_csv(REPORTS_DIR / "compute_results.csv", index=False)

artifact = {
    "schema_version": 1,
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(DEVICE),
        "in_colab": IN_COLAB,
    },
    "protocol": {
        "epochs": EPOCHS,
        "training_seed": TRAINING_SEED,
        "pair_seeds": list(PAIR_SEEDS),
        "batch_size": BATCH_SIZE,
        "benchmark_warmups": BENCHMARK_WARMUPS,
        "benchmark_measurements": BENCHMARK_MEASUREMENTS,
        "timing_dtype": str(profile_images.dtype),
    },
    "configurations": {
        label: OmegaConf.to_container(cfg, resolve=True)
        for label, cfg in configs.items()
    },
    "checkpoints": {
        label: {
            "path": str(training_results[label].best_checkpoint),
            "sha256": sha256_of(training_results[label].best_checkpoint),
            "best_epoch": training_results[label].best_epoch,
            "final_epoch": training_results[label].final_epoch,
            "best_validation": training_results[label].best_validation,
        }
        for label in ARM_LABELS
    },
    "training_seconds": training_seconds,
    "evaluations": {
        label: {
            "mean_total": evaluations[label].mean_total,
            "std_total": evaluations[label].std_total,
            "checkpoint_sha256": evaluations[label].checkpoint_sha256,
            "pairs": [asdict(pair) for pair in evaluations[label].pairs],
        }
        for label in ARM_LABELS
    },
    "compute": profiles,
    "compute_caveats": compute_caveats,
    "differences": comparison_table.to_dict(orient="records"),
}

report_json_path = REPORTS_DIR / "comparison_results.json"
report_json_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
print(f"Wrote {REPORTS_DIR / 'comparison_results.csv'}")
print(f"Wrote {REPORTS_DIR / 'compute_results.csv'}")
print(f"Wrote {report_json_path}")
"""


FINAL_CHECKS = """
# Final invariants over the produced artifacts.
for label in ARM_LABELS:
    result = training_results[label]
    assert result.best_checkpoint.is_file(), label
    assert result.final_epoch == 1, (label, result.final_epoch)
    pair_seeds = {pair.pair_seed for pair in evaluations[label].pairs}
    assert pair_seeds == set(PAIR_SEEDS) and len(pair_seeds) == 5, label
    assert math.isfinite(evaluations[label].mean_total), label
    profile = profiles[label]
    for key in (
        "params_backbone",
        "params_head",
        "params_total",
        "head_latency_ms",
        "full_latency_ms",
        "head_throughput",
        "full_throughput",
    ):
        assert math.isfinite(profile[key]), (label, key)

for name in ("comparison_results.csv", "compute_results.csv", "comparison_results.json"):
    assert (REPORTS_DIR / name).is_file(), name

print("All final checks passed.")
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
        markdown("profiling-heading", PROFILING_MARKDOWN),
        code("portable-profiler", PORTABLE_PROFILER),
        code("profile-models", PROFILE_MODELS),
        markdown("results-heading", RESULTS_MARKDOWN),
        code("build-result-tables", BUILD_RESULT_TABLES),
        code("plot-results", PLOT_RESULTS),
        code("derive-takeaways", DERIVE_TAKEAWAYS),
        markdown("export-heading", EXPORT_MARKDOWN),
        code("export-results", EXPORT_RESULTS),
        code("final-checks", FINAL_CHECKS),
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
