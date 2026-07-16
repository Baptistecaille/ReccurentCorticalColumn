from collections.abc import Callable
from dataclasses import dataclass

from fvcore.nn import FlopCountAnalysis
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor

from .model import ImageSSL


@dataclass(frozen=True)
class ComputeMetrics:
    parameters: int
    flops_per_sample: float
    latency_ms: float
    throughput_samples_s: float
    peak_memory_bytes: int


@dataclass(frozen=True)
class BenchmarkResult:
    head: ComputeMetrics
    full_model: ComputeMetrics
    gpu_name: str
    dtype: str
    batch_size: int


def measure_flops(module: nn.Module, example_input: Tensor) -> float:
    """Estimate FLOPs per sample for one forward pass of ``module``."""
    if example_input.ndim == 0 or example_input.shape[0] == 0:
        raise ValueError("example_input must contain at least one sample")

    module.eval()
    total_flops = FlopCountAnalysis(module, (example_input,)).total()
    return float(total_flops / example_input.shape[0])


def measure_latency(
    call: Callable[[], Tensor],
    batch_size: int,
    warmups: int = 100,
    measurements: int = 500,
) -> tuple[float, float]:
    """Measure mean CUDA latency and the corresponding sample throughput."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if measurements <= 0:
        raise ValueError("measurements must be positive")

    with torch.no_grad():
        for _ in range(warmups):
            call()

        torch.cuda.synchronize()

        event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for _ in range(measurements):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            event_pairs.append((start, end))

        torch.cuda.synchronize()

    durations_ms = [start.elapsed_time(end) for start, end in event_pairs]
    latency_ms = sum(durations_ms) / len(durations_ms)
    if latency_ms <= 0:
        raise RuntimeError(f"Measured non-positive CUDA latency: {latency_ms}")

    throughput_samples_s = batch_size / (latency_ms / 1_000.0)
    return float(latency_ms), float(throughput_samples_s)


def measure_peak_memory(call: Callable[[], Tensor]) -> int:
    """Return the absolute peak CUDA allocation, in bytes, during ``call``."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        call()

    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def benchmark_model(
    model: ImageSSL,
    cfg: DictConfig,
    device: torch.device,
) -> BenchmarkResult:
    """Benchmark the head and complete model under the fixed A100 protocol."""
    if device.type != "cuda":
        raise RuntimeError(
            f"The benchmark requires a CUDA device, received {device}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    gpu_name = torch.cuda.get_device_name(device)
    required_gpu = str(cfg.benchmark.required_gpu)
    if required_gpu.lower() not in gpu_name.lower():
        raise RuntimeError(
            f"The benchmark requires {required_gpu!r}, found {gpu_name!r}"
        )

    batch_size = int(cfg.benchmark.batch_size)
    warmups = int(cfg.benchmark.warmups)
    measurements = int(cfg.benchmark.measurements)
    if batch_size != 256:
        raise ValueError(
            f"benchmark.batch_size must be 256, received {batch_size}"
        )
    if warmups != 100 or measurements != 500:
        raise ValueError(
            "The benchmark protocol requires exactly 100 warmups and 500 "
            f"measurements, received {warmups} and {measurements}"
        )
    if str(cfg.optimization.precision).lower() != "bfloat16":
        raise ValueError("The benchmark protocol requires bfloat16 precision")

    dtype = torch.bfloat16
    model = model.eval().to(device=device, dtype=dtype)

    with torch.cuda.device(device):
        images = torch.randn(
            batch_size,
            3,
            32,
            32,
            device=device,
            dtype=dtype,
        )

        with torch.no_grad():
            features = model.backbone(images)

        def head_call() -> Tensor:
            return model.head(features)

        head_latency_ms, head_throughput = measure_latency(
            head_call,
            batch_size=batch_size,
            warmups=warmups,
            measurements=measurements,
        )
        head_metrics = ComputeMetrics(
            parameters=sum(
                parameter.numel() for parameter in model.head.parameters()
            ),
            flops_per_sample=measure_flops(model.head, features),
            latency_ms=head_latency_ms,
            throughput_samples_s=head_throughput,
            peak_memory_bytes=measure_peak_memory(head_call),
        )

        del head_call, features

        def full_model_call() -> Tensor:
            _, projections = model(images)
            return projections

        full_latency_ms, full_throughput = measure_latency(
            full_model_call,
            batch_size=batch_size,
            warmups=warmups,
            measurements=measurements,
        )
        full_model_metrics = ComputeMetrics(
            parameters=sum(
                parameter.numel() for parameter in model.parameters()
            ),
            flops_per_sample=measure_flops(model, images),
            latency_ms=full_latency_ms,
            throughput_samples_s=full_throughput,
            peak_memory_bytes=measure_peak_memory(full_model_call),
        )

    return BenchmarkResult(
        head=head_metrics,
        full_model=full_model_metrics,
        gpu_name=gpu_name,
        dtype="bfloat16",
        batch_size=batch_size,
    )
