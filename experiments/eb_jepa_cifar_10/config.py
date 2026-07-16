import torch
from dataclasses import dataclass, asdict, is_dataclass
from typing import Literal
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    root: str
    train_size: int
    validation_size: int
    test_size: int
    batch_size: int
    num_workers: int
    split_seed: int
    download: bool

    def __post_init__(self):
        if self.train_size < 0:
            raise ValueError("train_size must be non-negative")
        if self.validation_size < 0:
            raise ValueError("validation_size must be non-negative")
        if self.test_size < 0:
            raise ValueError("test_size must be non-negative")
        if self.batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")

        if self.train_size + self.validation_size != 50_000:
            raise ValueError("train_size + validation_size must equal 50,000")
        if self.test_size != 10_000:
            raise ValueError("test_size must equal 10,000")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2 for VICReg covariance computation")
        

@dataclass(frozen=True)
class ModelConfig:
    head_type: Literal["baseline", "cortical"]
    feature_dim: int
    output_dim: int
    mlp_hidden_dim: int

    def __post_init__(self):

        if self.feature_dim != 512:
            raise ValueError("feature_dim must be equal to 512")
        if self.output_dim != 2048:
            raise ValueError("output_dim must be equal to 2048")
        if self.feature_dim <= 0 or self.output_dim <= 0 or self.mlp_hidden_dim <= 0:
            raise ValueError("dimensions must be strictly positive")


@dataclass(frozen=True)
class CorticalConfig:
    n: int
    L_max: int
    dim_in: int
    dim_hidden: int
    dim_feedback: int
    dim_U: int
    dim_target: int
    disable_lateral: bool
    disable_feedback: bool


    def __post_init__(self):
            if self.n < 2:
                raise ValueError("n must be >= 2")
            if self.L_max < 0:
                raise ValueError("L_max must be >= 0")
            if self.dim_in != 512:
                raise ValueError("dim_in must be exactly 512")
            if self.dim_target != 2048:
                raise ValueError("dim_target must be exactly 2048")
            
            if self.dim_hidden <= 0:
                raise ValueError("dim_hidden must be positive (> 0)")
            if self.dim_feedback <= 0:
                raise ValueError("dim_feedback must be positive (> 0)")
            if self.dim_U <= 0:
                raise ValueError("dim_U must be positive (> 0)")
            

@dataclass(frozen=True)
class LossConfig:
    invariance_coeff: float
    std_coeff: float
    cov_coeff: float
    variance_target: float
    epsilon: float

    def __post_init__(self):
        if self.invariance_coeff < 0:
            raise ValueError("invariance_coeff must be non-negative")
        if self.std_coeff < 0:
            raise ValueError("std_coeff must be non-negative")
        if self.cov_coeff < 0:
            raise ValueError("cov_coeff must be non-negative")
        if self.variance_target < 0:
            raise ValueError("variance_target must be non-negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be strictly positive (> 0)")


@dataclass(frozen=True)
class OptimizationConfig:
    epochs: int
    learning_rate: float
    warmup_epochs: int
    warmup_start_lr: float
    min_lr: float
    weight_decay: float
    momentum: float
    precision: Literal["bfloat16", "float16", "float32"]

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be strictly positive (> 0)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be strictly positive (> 0)")
        if self.min_lr < 0:
            raise ValueError("min_lr must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        
        if not (0 <= self.warmup_epochs < self.epochs):
            raise ValueError("warmup_epochs must satisfy 0 <= warmup_epochs < epochs")
        if not (0 <= self.momentum < 1):
            raise ValueError("momentum must satisfy 0 <= momentum < 1")
        if self.warmup_start_lr > self.learning_rate:
            raise ValueError("warmup_start_lr must be <= learning_rate")
        if self.warmup_start_lr < 0:
            raise ValueError("warmup_start_lr must be non-negative")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    output_dir: str
    checkpoint_every: int
    data: DataConfig
    model: ModelConfig
    cortical: CorticalConfig
    loss: LossConfig
    optimization: OptimizationConfig


import yaml
from pathlib import Path
from typing import Any, Dict

def load_config(path: str | Path) -> ExperimentConfig:
    """Charge un YAML, construit les dataclasses et valide la configuration."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
    with open(yaml_path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {yaml_path}: {str(e)}") from e
            
    if data is None:
        raise ValueError(f"Configuration file {yaml_path} is empty")

    def instantiate(cls: Any, data_dict: Dict[str, Any], path_prefix: str = "") -> Any:
        if not isinstance(data_dict, dict):
            raise ValueError(f"Invalid format at '{path_prefix}': expected a dictionary, got {type(data_dict).__name__}")
        
        annotations = getattr(cls, "__annotations__", {})
        for field_name in annotations:
            if field_name not in data_dict:
                field_path = f"{path_prefix}.{field_name}" if path_prefix else field_name
                raise KeyError(f"Missing required field: '{field_path}'")
                
        for key in data_dict:
            if key not in annotations:
                field_path = f"{path_prefix}.{key}" if path_prefix else key
                raise ValueError(f"Unknown configuration key: '{field_path}'")
                
        resolved_kwargs = {}
        for field_name, field_type in annotations.items():
            value = data_dict[field_name]
            field_path = f"{path_prefix}.{field_name}" if path_prefix else field_name
            
            if hasattr(field_type, "__dataclass_fields__"):
                resolved_kwargs[field_name] = instantiate(field_type, value, field_path)
            else:
                resolved_kwargs[field_name] = value

        try:
            return cls(**resolved_kwargs)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value at '{path_prefix}': {str(e)}") from e

    return instantiate(ExperimentConfig, data)


def config_to_dict(config: ExperimentConfig) -> dict[str, object]:
    """Produit un dictionnaire sérialisable en YAML/JSON, sans dataclass."""
    def serialize(obj: Any) -> Any:
        if is_dataclass(obj):
            return {k: serialize(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [serialize(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    return serialize(config)