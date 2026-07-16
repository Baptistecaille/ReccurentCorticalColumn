"""Configuration loading and protocol validation.

The dot-notation override mechanism is adapted from
``facebookresearch/eb_jepa/eb_jepa/training_utils.py:241-269`` at commit
966e61e9285b3a876f49b9774e9720d9a99a7925. Local validation enforces the
CIFAR-10 comparison protocol shared by the baseline and cortical heads.
"""

from math import isclose
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_keys(cfg: DictConfig, paths: tuple[str, ...]) -> None:
    missing = [path for path in paths if OmegaConf.select(cfg, path) is None]
    if missing:
        raise ValueError(
            "Missing required configuration keys: " + ", ".join(missing)
        )


def _require_close(path: str, value: float, expected: float) -> None:
    _require(
        isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12),
        f"{path} must be {expected}, received {value}",
    )


def validate_config(cfg: DictConfig) -> None:
    """Validate architecture and scientific-protocol invariants."""
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"cfg must be a DictConfig, received {type(cfg).__name__}")

    supplied_head_type = OmegaConf.select(cfg, "model.head_type")
    if supplied_head_type is not None:
        _require(
            str(supplied_head_type).lower() in {"baseline", "cortical"},
            "model.head_type must be 'baseline' or 'cortical', "
            f"received {supplied_head_type!r}",
        )

    _require_keys(
        cfg,
        (
            "data.dataset",
            "data.root",
            "data.download",
            "data.train_size",
            "data.validation_size",
            "data.test_size",
            "data.batch_size",
            "data.num_workers",
            "data.split_seed",
            "data.crop_scale",
            "model.head_type",
            "model.backbone",
            "model.feature_dim",
            "model.mlp_hidden_dim",
            "model.output_dim",
            "cortical.n",
            "cortical.L_max",
            "cortical.dim_in",
            "cortical.dim_hidden",
            "cortical.dim_feedback",
            "cortical.dim_U",
            "cortical.dim_target",
            "cortical.disable_lateral",
            "cortical.disable_feedback",
            "loss.type",
            "loss.invariance_coeff",
            "loss.std_coeff",
            "loss.cov_coeff",
            "loss.variance_target",
            "loss.variance_epsilon",
            "loss.collapse_threshold",
            "optimization.optimizer",
            "optimization.epochs",
            "optimization.learning_rate",
            "optimization.warmup_epochs",
            "optimization.warmup_start_lr",
            "optimization.min_lr",
            "optimization.weight_decay",
            "optimization.momentum",
            "optimization.precision",
            "experiment.seeds",
            "experiment.checkpoint_every",
            "benchmark.required_gpu",
            "benchmark.batch_size",
            "benchmark.warmups",
            "benchmark.measurements",
        ),
    )

    _require(
        str(cfg.data.dataset).lower() == "cifar10",
        f"data.dataset must be 'cifar10', received {cfg.data.dataset!r}",
    )
    _require(
        int(cfg.data.train_size) == 45_000,
        f"data.train_size must be 45000, received {cfg.data.train_size}",
    )
    _require(
        int(cfg.data.validation_size) == 5_000,
        "data.validation_size must be 5000, "
        f"received {cfg.data.validation_size}",
    )
    _require(
        int(cfg.data.test_size) == 10_000,
        f"data.test_size must be 10000, received {cfg.data.test_size}",
    )
    _require(
        int(cfg.data.batch_size) == 256,
        f"data.batch_size must be 256, received {cfg.data.batch_size}",
    )
    _require(
        int(cfg.data.num_workers) >= 0,
        f"data.num_workers must be non-negative, received {cfg.data.num_workers}",
    )
    crop_scale = list(cfg.data.crop_scale)
    _require(
        len(crop_scale) == 2
        and 0.0 < float(crop_scale[0]) <= float(crop_scale[1]) <= 1.0,
        "data.crop_scale must contain two ordered values in (0, 1], "
        f"received {crop_scale}",
    )

    head_type = str(cfg.model.head_type).lower()
    _require(
        head_type in {"baseline", "cortical"},
        "model.head_type must be 'baseline' or 'cortical', "
        f"received {cfg.model.head_type!r}",
    )
    _require(
        str(cfg.model.backbone).lower() == "resnet18",
        f"model.backbone must be 'resnet18', received {cfg.model.backbone!r}",
    )
    _require(
        int(cfg.model.feature_dim) == 512,
        f"model.feature_dim must be 512, received {cfg.model.feature_dim}",
    )
    _require(
        int(cfg.model.mlp_hidden_dim) == 2048,
        "model.mlp_hidden_dim must be 2048, "
        f"received {cfg.model.mlp_hidden_dim}",
    )
    _require(
        int(cfg.model.output_dim) == 2048,
        f"model.output_dim must be 2048, received {cfg.model.output_dim}",
    )

    _require(
        int(cfg.cortical.n) >= 2,
        f"cortical.n must be at least 2, received {cfg.cortical.n}",
    )
    _require(
        int(cfg.cortical.L_max) >= 0,
        f"cortical.L_max must be non-negative, received {cfg.cortical.L_max}",
    )
    for path in (
        "cortical.dim_hidden",
        "cortical.dim_feedback",
        "cortical.dim_U",
    ):
        value = OmegaConf.select(cfg, path)
        _require(int(value) > 0, f"{path} must be positive, received {value}")
    _require(
        int(cfg.cortical.dim_in) == int(cfg.model.feature_dim),
        "cortical.dim_in must equal model.feature_dim; "
        f"received {cfg.cortical.dim_in} and {cfg.model.feature_dim}",
    )
    _require(
        int(cfg.cortical.dim_target) == int(cfg.model.output_dim),
        "cortical.dim_target must equal model.output_dim; "
        f"received {cfg.cortical.dim_target} and {cfg.model.output_dim}",
    )
    _require(
        isinstance(cfg.cortical.disable_lateral, bool)
        and isinstance(cfg.cortical.disable_feedback, bool),
        "cortical.disable_lateral and cortical.disable_feedback must be booleans",
    )

    _require(
        str(cfg.loss.type).lower() == "vicreg",
        f"loss.type must be 'vicreg', received {cfg.loss.type!r}",
    )
    _require_close("loss.invariance_coeff", cfg.loss.invariance_coeff, 1.0)
    _require_close("loss.std_coeff", cfg.loss.std_coeff, 1.0)
    _require_close("loss.cov_coeff", cfg.loss.cov_coeff, 80.0)
    _require_close("loss.variance_target", cfg.loss.variance_target, 1.0)
    _require_close("loss.variance_epsilon", cfg.loss.variance_epsilon, 1e-4)
    _require(
        float(cfg.loss.collapse_threshold) > 0.0,
        "loss.collapse_threshold must be positive, "
        f"received {cfg.loss.collapse_threshold}",
    )

    _require(
        str(cfg.optimization.optimizer).lower() == "lars",
        "optimization.optimizer must be 'lars', "
        f"received {cfg.optimization.optimizer!r}",
    )
    epochs = int(cfg.optimization.epochs)
    warmup_epochs = int(cfg.optimization.warmup_epochs)

    _require(
        epochs > 0,
        f"optimization.epochs must be positive, received {epochs}",
    )
    _require_close(
        "optimization.learning_rate", cfg.optimization.learning_rate, 0.3
    )
    _require(
        warmup_epochs >= 0,
        "optimization.warmup_epochs must be non-negative, "
        f"received {warmup_epochs}",
    )
    _require_close(
        "optimization.warmup_start_lr", cfg.optimization.warmup_start_lr, 3e-5
    )
    _require_close("optimization.min_lr", cfg.optimization.min_lr, 0.0)
    _require_close(
        "optimization.weight_decay", cfg.optimization.weight_decay, 1e-4
    )
    _require_close("optimization.momentum", cfg.optimization.momentum, 0.9)
    _require(
        str(cfg.optimization.precision).lower() == "bfloat16",
        "optimization.precision must be 'bfloat16', "
        f"received {cfg.optimization.precision!r}",
    )

    seeds = [int(seed) for seed in cfg.experiment.seeds]
    _require(
        seeds == [1, 1000, 10000],
        f"experiment.seeds must be [1, 1000, 10000], received {seeds}",
    )
    _require(
        int(cfg.experiment.checkpoint_every) > 0,
        "experiment.checkpoint_every must be positive, "
        f"received {cfg.experiment.checkpoint_every}",
    )

    _require(
        "A100" in str(cfg.benchmark.required_gpu).upper(),
        "benchmark.required_gpu must identify an A100, "
        f"received {cfg.benchmark.required_gpu!r}",
    )
    _require(
        int(cfg.benchmark.batch_size) == 256,
        "benchmark.batch_size must be 256, "
        f"received {cfg.benchmark.batch_size}",
    )
    _require(
        int(cfg.benchmark.warmups) == 100,
        f"benchmark.warmups must be 100, received {cfg.benchmark.warmups}",
    )
    _require(
        int(cfg.benchmark.measurements) == 500,
        "benchmark.measurements must be 500, "
        f"received {cfg.benchmark.measurements}",
    )


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load YAML, apply dot-list overrides, validate, and freeze its structure."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    OmegaConf.set_struct(cfg, True)
    if overrides:
        malformed = [override for override in overrides if "=" not in override]
        if malformed:
            raise ValueError(
                "Overrides must use key=value notation: " + ", ".join(malformed)
            )
        try:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        except OmegaConfBaseException as error:
            raise ValueError(f"Invalid configuration override: {error}") from error

    validate_config(cfg)
    OmegaConf.set_struct(cfg, True)
    return cfg
