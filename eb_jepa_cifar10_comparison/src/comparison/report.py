"""Aggregate and report the fixed baseline-versus-cortical experiment."""

import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .benchmark import BenchmarkResult, ComputeMetrics
from .evaluate import TestEvaluation


REQUIRED_TRAINING_SEEDS = frozenset({1, 1_000, 10_000})
SUPPORTED_HEAD_TYPES = frozenset({"baseline", "cortical"})


@dataclass(frozen=True)
class SeedResult:
    head_type: str
    training_seed: int
    evaluation: TestEvaluation
    benchmark: BenchmarkResult


@dataclass(frozen=True)
class MeanComputeMetrics:
    parameters: float
    flops_per_sample: float
    latency_ms: float
    throughput_samples_s: float
    peak_memory_bytes: float


@dataclass(frozen=True)
class AggregateResult:
    head_type: str
    runs: tuple[SeedResult, SeedResult, SeedResult]
    mean_test_score: float
    std_test_score: float
    minimum_mean_std: float
    maximum_collapsed_fraction: float
    head_compute: MeanComputeMetrics
    full_model_compute: MeanComputeMetrics


@dataclass(frozen=True)
class ComparisonDecision:
    quality_ok: bool
    collapse_ok: bool
    compute_ok: bool
    success: bool
    relative_score_gap: float
    compute_reduction: float


def _require_finite(name: str, value: float, *, positive: bool = False) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, received {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive, received {value!r}")
    if not positive and value < 0:
        raise ValueError(f"{name} must be non-negative, received {value!r}")


def _validate_compute(prefix: str, metrics: ComputeMetrics) -> None:
    _require_finite(f"{prefix}.parameters", float(metrics.parameters))
    _require_finite(
        f"{prefix}.flops_per_sample",
        metrics.flops_per_sample,
        positive=True,
    )
    _require_finite(f"{prefix}.latency_ms", metrics.latency_ms, positive=True)
    _require_finite(
        f"{prefix}.throughput_samples_s",
        metrics.throughput_samples_s,
        positive=True,
    )
    _require_finite(
        f"{prefix}.peak_memory_bytes",
        float(metrics.peak_memory_bytes),
    )


def _mean_compute(metrics: Sequence[ComputeMetrics]) -> MeanComputeMetrics:
    return MeanComputeMetrics(
        parameters=statistics.mean(item.parameters for item in metrics),
        flops_per_sample=statistics.mean(
            item.flops_per_sample for item in metrics
        ),
        latency_ms=statistics.mean(item.latency_ms for item in metrics),
        throughput_samples_s=statistics.mean(
            item.throughput_samples_s for item in metrics
        ),
        peak_memory_bytes=statistics.mean(
            item.peak_memory_bytes for item in metrics
        ),
    )


def aggregate_results(results: Sequence[SeedResult]) -> AggregateResult:
    """Aggregate exactly the three protocol runs for one architecture."""
    runs = tuple(results)
    if len(runs) != 3:
        raise ValueError(f"Exactly three runs are required, received {len(runs)}")

    seeds = [run.training_seed for run in runs]
    if set(seeds) != REQUIRED_TRAINING_SEEDS or len(set(seeds)) != len(seeds):
        raise ValueError(
            "Training seeds must be exactly {1, 1000, 10000} without "
            f"duplicates, received {seeds}"
        )

    head_types = {run.head_type.strip().lower() for run in runs}
    if len(head_types) != 1:
        raise ValueError(f"All runs must use one head type, received {head_types}")
    head_type = head_types.pop()
    if head_type not in SUPPORTED_HEAD_TYPES:
        raise ValueError(f"Unsupported head type: {head_type!r}")

    protocols = {
        (run.benchmark.gpu_name, run.benchmark.dtype, run.benchmark.batch_size)
        for run in runs
    }
    if len(protocols) != 1:
        raise ValueError(
            "All runs must share GPU, dtype, and benchmark batch size"
        )

    pair_results = []
    for run in runs:
        if not run.evaluation.pairs:
            raise ValueError("Every evaluation must contain at least one pair")
        _require_finite("evaluation.mean_total", run.evaluation.mean_total)
        _require_finite("evaluation.std_total", run.evaluation.std_total)
        _validate_compute("benchmark.head", run.benchmark.head)
        _validate_compute("benchmark.full_model", run.benchmark.full_model)
        for pair in run.evaluation.pairs:
            _require_finite("pair.mean_std", pair.mean_std)
            _require_finite(
                "pair.collapsed_fraction",
                pair.collapsed_fraction,
            )
            if pair.collapsed_fraction > 1:
                raise ValueError("pair.collapsed_fraction must not exceed 1")
            pair_results.append(pair)

    sorted_runs = tuple(sorted(runs, key=lambda run: run.training_seed))
    return AggregateResult(
        head_type=head_type,
        runs=sorted_runs,
        mean_test_score=statistics.mean(
            run.evaluation.mean_total for run in sorted_runs
        ),
        std_test_score=statistics.stdev(
            run.evaluation.mean_total for run in sorted_runs
        ),
        minimum_mean_std=min(pair.mean_std for pair in pair_results),
        maximum_collapsed_fraction=max(
            pair.collapsed_fraction for pair in pair_results
        ),
        head_compute=_mean_compute(
            [run.benchmark.head for run in sorted_runs]
        ),
        full_model_compute=_mean_compute(
            [run.benchmark.full_model for run in sorted_runs]
        ),
    )


def _aggregate_protocol(result: AggregateResult) -> tuple[str, str, int]:
    benchmark = result.runs[0].benchmark
    return benchmark.gpu_name, benchmark.dtype, benchmark.batch_size


def decide_comparison(
    baseline: AggregateResult,
    cortical: AggregateResult,
    score_tolerance: float = 0.02,
) -> ComparisonDecision:
    """Apply the fixed quality, collapse, and compute success gates."""
    if baseline.head_type != "baseline" or cortical.head_type != "cortical":
        raise ValueError(
            "Expected baseline then cortical aggregates, received "
            f"{baseline.head_type!r} then {cortical.head_type!r}"
        )
    if not math.isfinite(score_tolerance) or score_tolerance < 0:
        raise ValueError("score_tolerance must be finite and non-negative")
    if baseline.mean_test_score <= 0 or not math.isfinite(
        baseline.mean_test_score
    ):
        raise ValueError("The baseline mean test score must be finite and positive")
    baseline_head_flops = baseline.head_compute.flops_per_sample
    if baseline_head_flops <= 0 or not math.isfinite(baseline_head_flops):
        raise ValueError("The baseline head FLOPs must be finite and positive")
    if _aggregate_protocol(baseline) != _aggregate_protocol(cortical):
        raise ValueError(
            "Baseline and cortical benchmarks must use the same GPU, dtype, "
            "and batch size"
        )

    relative_score_gap = (
        cortical.mean_test_score - baseline.mean_test_score
    ) / baseline.mean_test_score
    compute_reduction = (
        1.0 - cortical.head_compute.flops_per_sample / baseline_head_flops
    )
    if not math.isfinite(relative_score_gap) or not math.isfinite(
        compute_reduction
    ):
        raise ValueError("Comparison ratios must be finite")

    quality_ok = relative_score_gap <= score_tolerance
    collapse_ok = (
        cortical.minimum_mean_std >= 0.1
        and cortical.maximum_collapsed_fraction <= 0.1
    )
    full_system_improved = any((
        cortical.full_model_compute.flops_per_sample
        < baseline.full_model_compute.flops_per_sample,
        cortical.full_model_compute.latency_ms
        < baseline.full_model_compute.latency_ms,
        cortical.full_model_compute.throughput_samples_s
        > baseline.full_model_compute.throughput_samples_s,
        cortical.full_model_compute.peak_memory_bytes
        < baseline.full_model_compute.peak_memory_bytes,
    ))
    compute_ok = (
        cortical.head_compute.flops_per_sample < baseline_head_flops
        and full_system_improved
    )

    return ComparisonDecision(
        quality_ok=quality_ok,
        collapse_ok=collapse_ok,
        compute_ok=compute_ok,
        success=quality_ok and collapse_ok and compute_ok,
        relative_score_gap=relative_score_gap,
        compute_reduction=compute_reduction,
    )


def _aggregate_payload(result: AggregateResult) -> dict[str, object]:
    payload = asdict(result)
    payload.pop("runs")
    gpu_name, dtype, batch_size = _aggregate_protocol(result)
    payload["benchmark_protocol"] = {
        "gpu_name": gpu_name,
        "dtype": dtype,
        "batch_size": batch_size,
    }
    return payload


def _report_payload(
    baseline: AggregateResult,
    cortical: AggregateResult,
    decision: ComparisonDecision,
    score_tolerance: float,
) -> dict[str, object]:
    if baseline.head_type != "baseline" or cortical.head_type != "cortical":
        raise ValueError("Reports require baseline then cortical aggregates")
    return {
        "runs": [
            asdict(run)
            for aggregate in (baseline, cortical)
            for run in aggregate.runs
        ],
        "aggregates": {
            "baseline": _aggregate_payload(baseline),
            "cortical": _aggregate_payload(cortical),
        },
        "score_tolerance": score_tolerance,
        "decision": asdict(decision),
    }


def _number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def _render_markdown(payload: dict[str, object]) -> str:
    runs = payload["runs"]
    aggregates = payload["aggregates"]
    score_tolerance = payload["score_tolerance"]
    decision = payload["decision"]
    assert isinstance(runs, list)
    assert isinstance(aggregates, dict)
    assert isinstance(decision, dict)

    lines = [
        "# EB-JEPA CIFAR-10 — baseline versus cortical",
        "",
        "## Raw runs",
        "",
        (
            "| Head | Seed | VICReg total | Min mean std | Max collapsed "
            "fraction | Head FLOPs/sample | Full FLOPs/sample | Full latency "
            "(ms) | Full throughput (samples/s) | Full peak memory (bytes) | "
            "GPU | Checkpoint SHA-256 |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
        ),
    ]
    for run in runs:
        evaluation = run["evaluation"]
        benchmark = run["benchmark"]
        pairs = evaluation["pairs"]
        minimum_mean_std = min(pair["mean_std"] for pair in pairs)
        maximum_collapsed = max(
            pair["collapsed_fraction"] for pair in pairs
        )
        lines.append(
            "| "
            + " | ".join((
                str(run["head_type"]).strip().lower(),
                str(run["training_seed"]),
                _number(evaluation["mean_total"]),
                _number(minimum_mean_std),
                _number(maximum_collapsed),
                _number(benchmark["head"]["flops_per_sample"]),
                _number(benchmark["full_model"]["flops_per_sample"]),
                _number(benchmark["full_model"]["latency_ms"]),
                _number(benchmark["full_model"]["throughput_samples_s"]),
                _number(benchmark["full_model"]["peak_memory_bytes"]),
                str(benchmark["gpu_name"]),
                str(evaluation["checkpoint_sha256"]),
            ))
            + " |"
        )

    lines.extend((
        "",
        "## Aggregates across training seeds",
        "",
        (
            "| Head | VICReg mean | VICReg sample std | Minimum mean std | "
            "Maximum collapsed fraction | Mean head FLOPs/sample | Mean full "
            "FLOPs/sample | Mean full latency (ms) | Mean full throughput "
            "(samples/s) | Mean full peak memory (bytes) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for head_type in ("baseline", "cortical"):
        aggregate = aggregates[head_type]
        lines.append(
            "| "
            + " | ".join((
                head_type,
                _number(aggregate["mean_test_score"]),
                _number(aggregate["std_test_score"]),
                _number(aggregate["minimum_mean_std"]),
                _number(aggregate["maximum_collapsed_fraction"]),
                _number(aggregate["head_compute"]["flops_per_sample"]),
                _number(aggregate["full_model_compute"]["flops_per_sample"]),
                _number(aggregate["full_model_compute"]["latency_ms"]),
                _number(
                    aggregate["full_model_compute"]["throughput_samples_s"]
                ),
                _number(
                    aggregate["full_model_compute"]["peak_memory_bytes"]
                ),
            ))
            + " |"
        )

    conclusion = "SUCCESS" if decision["success"] else "FAILURE"
    lines.extend((
        "",
        "## Decision",
        "",
        (
            "- Quality gate (relative VICReg gap ≤ "
            f"{_number(score_tolerance * 100)}%): {decision['quality_ok']}"
        ),
        (
            "- Collapse gate (minimum mean std ≥ 0.1 and maximum collapsed "
            f"fraction ≤ 0.1): {decision['collapse_ok']}"
        ),
        (
            "- Compute gate (lower head FLOPs plus a full-model improvement): "
            f"{decision['compute_ok']}"
        ),
        (
            "- Relative VICReg score gap: "
            f"{_number(decision['relative_score_gap'] * 100)}%"
        ),
        (
            "- Head FLOPs reduction: "
            f"{_number(decision['compute_reduction'] * 100)}%"
        ),
        "",
        f"**Conclusion: {conclusion}.**",
        "",
    ))
    return "\n".join(lines)


def write_report(
    markdown_path: str | Path,
    json_path: str | Path,
    baseline: AggregateResult,
    cortical: AggregateResult,
    decision: ComparisonDecision,
    score_tolerance: float = 0.02,
) -> None:
    """Write coherent, auditable JSON and Markdown comparison reports."""
    if not math.isfinite(score_tolerance) or score_tolerance < 0:
        raise ValueError("score_tolerance must be finite and non-negative")
    markdown_path = Path(markdown_path)
    json_path = Path(json_path)
    if markdown_path.resolve() == json_path.resolve():
        raise ValueError("Markdown and JSON report paths must be distinct")

    payload = _report_payload(
        baseline,
        cortical,
        decision,
        score_tolerance,
    )
    json_text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    markdown_text = _render_markdown(payload)

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_text, encoding="utf-8")
