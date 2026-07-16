from pathlib import Path
import random
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from .model import ImageSSL
from .optim import WarmupCosineScheduler
import numpy as np
from dataclasses import dataclass



# Helpers
def capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def validate_rng_state(state: object) -> dict[str, object]:
    """Validate a serialized RNG state without changing global RNGs."""
    if not isinstance(state, dict):
        raise TypeError("Invalid RNG state: expected a dictionary")

    required_keys = {"python", "numpy", "torch", "cuda"}
    missing_keys = required_keys.difference(state)
    if missing_keys:
        raise ValueError(
            "RNG state is missing keys: "
            + ", ".join(sorted(missing_keys))
        )

    # Validate Python and NumPy states on isolated generators.
    random.Random().setstate(state["python"])
    np.random.RandomState().set_state(state["numpy"])

    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        raise TypeError("Invalid PyTorch RNG state")
    torch.Generator().set_state(torch_state.cpu())

    cuda_state = state["cuda"]
    if cuda_state is not None:
        if not isinstance(cuda_state, (list, tuple)):
            raise TypeError("Invalid CUDA RNG state")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The checkpoint contains CUDA RNG states, "
                "but CUDA is unavailable"
            )
        if len(cuda_state) != torch.cuda.device_count():
            raise ValueError(
                "CUDA RNG device-count mismatch: "
                f"checkpoint has {len(cuda_state)}, current runtime has "
                f"{torch.cuda.device_count()}"
            )

        for device_index, rng_state in enumerate(cuda_state):
            if not isinstance(rng_state, torch.Tensor):
                raise TypeError(
                    f"Invalid CUDA RNG state for device {device_index}"
                )
            torch.Generator(device=f"cuda:{device_index}").set_state(
                rng_state.cpu()
            )

    return state


def restore_rng_state(state: dict[str, object]) -> None:
    state = validate_rng_state(state)

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        raise TypeError("Invalid PyTorch RNG state")
    torch.set_rng_state(torch_state.cpu())

    cuda_state = state["cuda"]

    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The checkpoint contains CUDA RNG states, "
                "but CUDA is unavailable"
            )

        torch.cuda.set_rng_state_all(
            [rng_state.cpu() for rng_state in cuda_state]
        )


def setup_device(device_name: str = "auto") -> torch.device:
    """Select CPU or CUDA without silently hiding unavailable CUDA."""

    normalized_name = device_name.lower()

    if normalized_name == "auto":
        normalized_name = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        device = torch.device(normalized_name)
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            f"Unknown device {device_name!r}; expected 'auto', 'cpu', "
            "'cuda' or 'cuda:<index>'"
        ) from error

    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"Unsupported device type {device.type!r}; expected CPU or CUDA"
        )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was explicitly requested, but no CUDA device is available"
            )

        if device.index is not None:
            device_count = torch.cuda.device_count()

            if device.index >= device_count:
                raise RuntimeError(
                    f"CUDA device {device.index} was requested, "
                    f"but only {device_count} CUDA device(s) are available"
                )

    return device


def setup_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            f"seed must be an integer, received {type(seed).__name__}"
        )

    if not 0 <= seed < 2**32:
        raise ValueError(
            f"seed must be between 0 and {2**32 - 1}, received {seed}"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside a DataLoader worker."""

    del worker_id

    worker_seed = torch.initial_seed() % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)



def save_checkpoint(
    path: str | Path,
    *,
    model: ImageSSL,
    optimizer: Optimizer,
    scheduler: WarmupCosineScheduler,
    epoch: int,
    best_validation: float,
    best_epoch: int,
    cfg: DictConfig,
    seed: int,
    train_generator: torch.Generator,
) -> None:
    """Save a complete training checkpoint atomically."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = checkpoint_path.with_name(
        f".{checkpoint_path.name}.tmp"
    )

    resolved_config = OmegaConf.to_container(
        cfg,
        resolve=True,
        enum_to_str=True,
    )

    checkpoint = {
    "format_version": 3,
    "epoch": int(epoch),
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_validation": float(best_validation),
    "best_epoch": int(best_epoch),
    "config": resolved_config,
    "head_type": str(cfg.model.head_type),
    "seed": int(seed),
    "rng_state": capture_rng_state(),
    "train_generator_state": train_generator.get_state(),
    }

    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


@dataclass(frozen=True)
class ResumeState:
    start_epoch: int
    best_validation: float
    best_epoch: int


def load_checkpoint(
    path: str | Path,
    model: ImageSSL,
    optimizer: Optimizer,
    scheduler: WarmupCosineScheduler,
    cfg: DictConfig,
    device: torch.device,
    seed: int,
    train_generator: torch.Generator,
) -> ResumeState:
    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Checkpoint must contain a dictionary"
        )

    required_keys = {
    "format_version",
    "epoch",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "best_validation",
    "config",
    "head_type",
    "seed",
    "rng_state",
    "train_generator_state",
}

    missing_keys = required_keys.difference(checkpoint)

    if missing_keys:
        raise ValueError(
            "Checkpoint is missing keys: "
            + ", ".join(sorted(missing_keys))
        )

    format_version = checkpoint["format_version"]
    if format_version not in {2, 3}:
        raise ValueError(
            "Unsupported checkpoint format version: "
            f"{format_version}"
        )

    if format_version >= 3 and "best_epoch" not in checkpoint:
        raise ValueError("Checkpoint is missing key: best_epoch")

    saved_head_type = str(checkpoint["head_type"]).lower()
    current_head_type = str(cfg.model.head_type).lower()

    if saved_head_type != current_head_type:
        raise ValueError(
            "Checkpoint head mismatch: "
            f"checkpoint uses {saved_head_type!r}, "
            f"current configuration uses {current_head_type!r}"
        )

    saved_config = OmegaConf.create(checkpoint["config"])

    critical_config_paths = (
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
    )

    for config_path in critical_config_paths:
        saved_value = OmegaConf.select(
            saved_config,
            config_path,
        )
        current_value = OmegaConf.select(
            cfg,
            config_path,
        )

        if saved_value != current_value:
            raise ValueError(
                f"Checkpoint configuration mismatch for "
                f"{config_path}: checkpoint has "
                f"{saved_value!r}, current configuration has "
                f"{current_value!r}"
            )

    saved_model_state = checkpoint["model_state_dict"]
    current_model_state = model.state_dict()

    saved_parameter_names = set(saved_model_state)
    current_parameter_names = set(current_model_state)

    missing_parameters = (
        current_parameter_names - saved_parameter_names
    )
    unexpected_parameters = (
        saved_parameter_names - current_parameter_names
    )

    if missing_parameters:
        raise ValueError(
            "Checkpoint model state is missing parameters: "
            + ", ".join(sorted(missing_parameters))
        )

    if unexpected_parameters:
        raise ValueError(
            "Checkpoint model state contains unexpected parameters: "
            + ", ".join(sorted(unexpected_parameters))
        )

    for parameter_name, current_value in current_model_state.items():
        saved_value = saved_model_state[parameter_name]

        if saved_value.shape != current_value.shape:
            raise ValueError(
                f"Parameter shape mismatch for {parameter_name}: "
                f"checkpoint has {tuple(saved_value.shape)}, "
                f"current model expects {tuple(current_value.shape)}"
            )

    saved_optimizer_state = checkpoint["optimizer_state_dict"]
    current_optimizer_state = optimizer.state_dict()

    saved_groups = saved_optimizer_state.get("param_groups")
    current_groups = current_optimizer_state.get("param_groups")

    if not isinstance(saved_groups, list):
        raise ValueError(
            "Checkpoint optimizer state has no valid param_groups"
        )

    if len(saved_groups) != len(current_groups):
        raise ValueError(
            "Optimizer parameter-group mismatch: "
            f"checkpoint has {len(saved_groups)}, "
            f"current optimizer has {len(current_groups)}"
        )

    for group_index, (saved_group, current_group) in enumerate(
        zip(saved_groups, current_groups)
    ):
        if len(saved_group["params"]) != len(current_group["params"]):
            raise ValueError(
                f"Optimizer group {group_index} contains "
                f"{len(saved_group['params'])} parameters in the "
                f"checkpoint but {len(current_group['params'])} "
                "in the current optimizer"
            )

    saved_scheduler_state = checkpoint[
        "scheduler_state_dict"
    ]
    current_scheduler_state = scheduler.state_dict()

    scheduler_config_keys = (
        "warmup_steps",
        "total_steps",
        "start_lr",
        "base_lr",
        "final_lr",
    )

    completed_epoch = int(checkpoint["epoch"])

    if completed_epoch < 0:
        raise ValueError(
            f"Checkpoint epoch must be non-negative: {completed_epoch}"
        )
    
    saved_seed = int(checkpoint["seed"])

    if saved_seed != seed:
        raise ValueError(
            "Checkpoint seed mismatch: "
            f"checkpoint uses {saved_seed}, current run uses {seed}"
        )

    for key in scheduler_config_keys:
        if key not in saved_scheduler_state:
            raise ValueError(
                f"Scheduler checkpoint is missing {key!r}"
            )

        if saved_scheduler_state[key] != current_scheduler_state[key]:
            raise ValueError(
                f"Scheduler mismatch for {key}: "
                f"checkpoint has {saved_scheduler_state[key]!r}, "
                f"current scheduler expects "
                f"{current_scheduler_state[key]!r}"
            )

    generator_state = checkpoint["train_generator_state"]
    if not isinstance(generator_state, torch.Tensor):
        raise TypeError("Invalid DataLoader generator state")

    # Validate on an isolated generator before mutating training objects.
    torch.Generator().set_state(generator_state.cpu())
    rng_state = validate_rng_state(checkpoint["rng_state"])

    # Toutes les validations sont terminées.
    # Les objets peuvent maintenant être modifiés.
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"],
    )
    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"],
    )

    train_generator.set_state(generator_state.cpu())
    restore_rng_state(rng_state)

    return ResumeState(
        start_epoch=completed_epoch + 1,
        best_validation=float(checkpoint["best_validation"]),
        best_epoch=int(checkpoint.get("best_epoch", -1)),
    )
        
