"""File-oriented workflows shared by evaluation, benchmarking, and reports."""

from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

from omegaconf import DictConfig, OmegaConf
import torch

from .benchmark import BenchmarkResult, ComputeMetrics
from .evaluate import PairEvaluation, TestEvaluation


_SCHEMA_VERSION = 1
_HEAD_TYPES = {"baseline", "cortical"}
_METADATA_KEYS = {"head_type", "training_seed", "checkpoint_sha256"}
_ENVELOPE_KEYS = {
    "schema_version",
    *_METADATA_KEYS,
    "evaluation",
    "benchmark",
}
_PAIR_KEYS = {field.name for field in fields(PairEvaluation)}
_EVALUATION_KEYS = {field.name for field in fields(TestEvaluation)}
_COMPUTE_KEYS = {field.name for field in fields(ComputeMetrics)}
_BENCHMARK_KEYS = {field.name for field in fields(BenchmarkResult)}


def _require_exact_keys(
    payload: object,
    expected: set[str],
    description: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"{description} must be a dictionary")

    supplied = set(payload)
    if supplied != expected:
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        details = []
        if missing:
            details.append("missing keys: " + ", ".join(missing))
        if unknown:
            details.append("unknown keys: " + ", ".join(unknown))
        raise ValueError(f"Invalid {description}: " + "; ".join(details))
    return payload


def _validate_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{description} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{description} must be hexadecimal") from error
    return value.lower()


def _validate_seed(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{description} must be an integer and not a boolean")
    if not 0 <= value < 2**32:
        raise ValueError(
            f"{description} must be between 0 and {2**32 - 1}, "
            f"received {value}"
        )
    return value


def _validate_integer(
    value: object,
    description: str,
    *,
    positive: bool,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{description} must be an integer and not a boolean")
    lower_bound = 1 if positive else 0
    if value < lower_bound:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{description} must be {qualifier}, received {value}")
    return value


def _validate_float(
    value: object,
    description: str,
    *,
    positive: bool = False,
) -> float:
    if not isinstance(value, float):
        raise TypeError(f"{description} must be a float")
    if not math.isfinite(value):
        raise ValueError(f"{description} must be finite, received {value}")
    lower_bound = 0.0
    if value < lower_bound or (positive and value == lower_bound):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{description} must be {qualifier}, received {value}")
    return value


def _validate_nonempty_string(value: object, description: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{description} must be a string")
    if not value.strip():
        raise ValueError(f"{description} must not be empty")
    return value


def _validate_metadata(metadata: object) -> dict[str, object]:
    values = _require_exact_keys(metadata, _METADATA_KEYS, "artifact metadata")

    head_type = values["head_type"]
    if not isinstance(head_type, str) or head_type.lower() not in _HEAD_TYPES:
        raise ValueError(
            "head_type must be 'baseline' or 'cortical', "
            f"received {head_type!r}"
        )

    training_seed = _validate_seed(values["training_seed"], "training_seed")

    return {
        "head_type": head_type.lower(),
        "training_seed": training_seed,
        "checkpoint_sha256": _validate_sha256(
            values["checkpoint_sha256"], "checkpoint_sha256"
        ),
    }


def _validate_evaluation(
    evaluation: object,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    values = _require_exact_keys(
        evaluation, _EVALUATION_KEYS, "evaluation payload"
    )
    pairs = values["pairs"]
    if not isinstance(pairs, (list, tuple)):
        raise TypeError("evaluation pairs must be a list or tuple")
    if not pairs:
        raise ValueError("evaluation pairs must not be empty")
    validated_pairs = []
    for pair_index, pair in enumerate(pairs):
        pair_values = _require_exact_keys(
            pair, _PAIR_KEYS, f"pair evaluation {pair_index}"
        )
        validated_pair = dict(pair_values)
        validated_pair["pair_seed"] = _validate_seed(
            pair_values["pair_seed"], f"pair evaluation {pair_index} pair_seed"
        )
        for metric in (
            "total",
            "invariance",
            "variance",
            "covariance",
            "mean_std",
            "min_std",
            "collapsed_fraction",
        ):
            validated_pair[metric] = _validate_float(
                pair_values[metric],
                f"pair evaluation {pair_index} {metric}",
            )
        collapsed_fraction = validated_pair["collapsed_fraction"]
        if collapsed_fraction > 1.0:
            raise ValueError(
                "pair evaluation collapsed_fraction must be between 0 and 1, "
                f"received {collapsed_fraction}"
            )
        validated_pairs.append(validated_pair)

    mean_total = _validate_float(values["mean_total"], "evaluation mean_total")
    std_total = _validate_float(values["std_total"], "evaluation std_total")

    evaluation_digest = _validate_sha256(
        values["checkpoint_sha256"],
        "evaluation checkpoint_sha256",
    )
    if evaluation_digest != checkpoint_sha256:
        raise ValueError(
            "Evaluation checkpoint hash does not match artifact metadata"
        )
    validated = dict(values)
    validated["pairs"] = validated_pairs
    validated["mean_total"] = mean_total
    validated["std_total"] = std_total
    validated["checkpoint_sha256"] = evaluation_digest
    return validated


def _validate_benchmark(benchmark: object) -> dict[str, Any]:
    values = _require_exact_keys(
        benchmark, _BENCHMARK_KEYS, "benchmark payload"
    )
    validated = dict(values)
    for section in ("head", "full_model"):
        metrics = _require_exact_keys(
            values[section], _COMPUTE_KEYS, f"{section} compute metrics"
        )
        validated_metrics = dict(metrics)
        validated_metrics["parameters"] = _validate_integer(
            metrics["parameters"],
            f"{section} parameters",
            positive=True,
        )
        for metric in (
            "flops_per_sample",
            "latency_ms",
            "throughput_samples_s",
        ):
            validated_metrics[metric] = _validate_float(
                metrics[metric],
                f"{section} {metric}",
                positive=True,
            )
        validated_metrics["peak_memory_bytes"] = _validate_integer(
            metrics["peak_memory_bytes"],
            f"{section} peak_memory_bytes",
            positive=False,
        )
        validated[section] = validated_metrics

    validated["gpu_name"] = _validate_nonempty_string(
        values["gpu_name"], "benchmark gpu_name"
    )
    validated["dtype"] = _validate_nonempty_string(
        values["dtype"], "benchmark dtype"
    )
    validated["batch_size"] = _validate_integer(
        values["batch_size"], "benchmark batch_size", positive=True
    )
    return validated


def _validate_envelope(payload: object) -> dict[str, Any]:
    values = _require_exact_keys(payload, _ENVELOPE_KEYS, "result artifact")
    schema_version = values["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != _SCHEMA_VERSION
    ):
        raise ValueError(
            f"schema_version must be {_SCHEMA_VERSION}, "
            f"received {schema_version!r}"
        )

    metadata = _validate_metadata(
        {key: values[key] for key in _METADATA_KEYS}
    )
    if values["head_type"] != metadata["head_type"]:
        raise ValueError("Artifact head_type must use its normalized value")
    if values["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
        raise ValueError(
            "Artifact checkpoint_sha256 must use its normalized value"
        )

    validated = dict(values)
    evaluation = values["evaluation"]
    benchmark = values["benchmark"]
    if evaluation is not None:
        validated["evaluation"] = _validate_evaluation(
            evaluation, metadata["checkpoint_sha256"]
        )
    if benchmark is not None:
        validated["benchmark"] = _validate_benchmark(benchmark)
    return validated


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not permitted: {value}")


def _read_result_artifact(path: str | Path) -> dict[str, Any]:
    """Read and strictly validate one shared result artifact."""
    artifact_path = Path(path)
    with artifact_path.open("r", encoding="utf-8") as artifact_file:
        payload = json.load(artifact_file, parse_constant=_reject_json_constant)
    return _validate_envelope(payload)


def _write_result_artifact(
    path: str | Path,
    metadata: dict[str, object],
    *,
    evaluation: dict[str, object] | None = None,
    benchmark: dict[str, object] | None = None,
) -> None:
    """Atomically write one section while preserving its matching complement."""
    if (evaluation is None) == (benchmark is None):
        raise ValueError("Exactly one of evaluation or benchmark must be provided")

    normalized_metadata = _validate_metadata(metadata)
    checkpoint_sha256 = str(normalized_metadata["checkpoint_sha256"])
    if evaluation is not None:
        evaluation = _validate_evaluation(evaluation, checkpoint_sha256)
    if benchmark is not None:
        benchmark = _validate_benchmark(benchmark)

    output_path = Path(path)
    existing_evaluation = None
    existing_benchmark = None
    if output_path.exists():
        existing = _read_result_artifact(output_path)
        existing_metadata = {key: existing[key] for key in _METADATA_KEYS}
        if existing_metadata != normalized_metadata:
            raise ValueError(
                "Existing result artifact metadata does not match the checkpoint"
            )
        existing_evaluation = existing["evaluation"]
        existing_benchmark = existing["benchmark"]

    envelope = {
        "schema_version": _SCHEMA_VERSION,
        **normalized_metadata,
        "evaluation": (
            evaluation if evaluation is not None else existing_evaluation
        ),
        "benchmark": benchmark if benchmark is not None else existing_benchmark,
    }
    _validate_envelope(envelope)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                envelope,
                temporary_file,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            temporary_file.write("\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _checkpoint_metadata(
    cfg: DictConfig,
    checkpoint_path: str | Path,
) -> tuple[dict[str, object], dict[str, Any]]:
    """Load a checkpoint on CPU and return normalized artifact metadata."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary")

    required = {"head_type", "seed", "model_state_dict"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            "Checkpoint is missing keys: " + ", ".join(sorted(missing))
        )

    saved_head_type = checkpoint["head_type"]
    if not isinstance(saved_head_type, str):
        raise TypeError("Checkpoint head_type must be a string")
    saved_head_type = saved_head_type.lower()
    configured_head_type = OmegaConf.select(cfg, "model.head_type")
    if configured_head_type is None:
        raise ValueError("Configuration is missing model.head_type")
    configured_head_type = str(configured_head_type).lower()
    if configured_head_type not in _HEAD_TYPES:
        raise ValueError(
            "Configured model.head_type must be 'baseline' or 'cortical'"
        )
    if saved_head_type != configured_head_type:
        raise ValueError(
            "Checkpoint head mismatch: "
            f"checkpoint uses {saved_head_type!r}, "
            f"configuration uses {configured_head_type!r}"
        )

    seed = _validate_seed(checkpoint["seed"], "Checkpoint seed")
    model_state_dict = checkpoint["model_state_dict"]
    if not isinstance(model_state_dict, dict):
        raise TypeError("Checkpoint model_state_dict must be a dictionary")

    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)

    metadata = _validate_metadata(
        {
            "head_type": saved_head_type,
            "training_seed": seed,
            "checkpoint_sha256": digest.hexdigest(),
        }
    )
    return metadata, model_state_dict
