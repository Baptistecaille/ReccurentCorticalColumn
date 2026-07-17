import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from .augmentation import get_train_transforms
from .checkpoint import (
    setup_device,
    should_pin_memory,
    should_use_bfloat16,
)
from .data import DeterministicPairDataset
from .losses import VICRegLoss, collapse_diagnostics
from .model import ImageSSL, build_model

@dataclass(frozen=True)
class PairEvaluation:
    pair_seed: int
    total: float
    invariance: float
    variance: float
    covariance: float
    mean_std: float
    min_std: float
    collapsed_fraction: float


def evaluate_pair(
    model: ImageSSL,
    loader: DataLoader,
    loss_fn: VICRegLoss,
    pair_seed: int,
    device: torch.device,
    use_bf16: bool,
) -> PairEvaluation:
    """Evaluate a model on a paired-view dataset."""
    model.eval()
    total_loss = 0.0
    total_invariance = 0.0
    total_variance = 0.0
    total_covariance = 0.0
    total_examples = 0
    all_projections: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) != 2:
                raise ValueError(
                    "The paired-view loader must return exactly two views "
                    "and no label"
                )
            x1, x2 = batch
            x1 = x1.to(device)
            x2 = x2.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                _, z1 = model(x1)
                _, z2 = model(x2)
                values = loss_fn.components(z1, z2)

            if any(not torch.isfinite(value) for value in values.values()):
                raise FloatingPointError("Non-finite VICReg component during test")

            batch_size = x1.shape[0]
            total_loss += values["loss"].item() * batch_size
            total_invariance += values["invariance_loss"].item() * batch_size
            total_variance += values["var_loss"].item() * batch_size
            total_covariance += values["cov_loss"].item() * batch_size
            total_examples += batch_size
            all_projections.extend(
                (z1.detach().float().cpu(), z2.detach().float().cpu())
            )

    if total_examples == 0:
        raise ValueError("The DataLoader is empty")

    diagnostics = collapse_diagnostics(torch.cat(all_projections, dim=0))
    return PairEvaluation(
        pair_seed=pair_seed,
        total=total_loss / total_examples,
        invariance=total_invariance / total_examples,
        variance=total_variance / total_examples,
        covariance=total_covariance / total_examples,
        mean_std=diagnostics.mean_std,
        min_std=diagnostics.min_std,
        collapsed_fraction=diagnostics.collapsed_fraction,
    )


@dataclass(frozen=True)
class TestEvaluation:
    pairs: tuple[PairEvaluation, ...]
    mean_total: float
    std_total: float
    checkpoint_sha256: str


def evaluate_test(
    cfg: DictConfig,
    checkpoint_path: str | Path,
    pair_seeds: tuple[int, int, int, int, int],
) -> TestEvaluation:
    """Évalue strictement un checkpoint sur cinq paires de vues déterministes."""

    if len(pair_seeds) != 5:
        raise ValueError(
            f"Exactly five pair seeds are required, received {len(pair_seeds)}"
        )

    if len(set(pair_seeds)) != 5:
        raise ValueError(
            f"pair_seeds must be pairwise distinct, received {pair_seeds}"
        )

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = setup_device()

    # Load inference weights only: no optimizer, scheduler, or RNG restoration.
    raw_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(raw_checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary")

    # Vérification des clés requises.
    required_keys = {"head_type", "model_state_dict"}
    missing_keys = required_keys.difference(raw_checkpoint)
    if missing_keys:
        raise ValueError(
            "Checkpoint is missing keys: " + ", ".join(sorted(missing_keys))
        )

    saved_head_type = str(raw_checkpoint["head_type"]).lower()
    current_head_type = str(cfg.model.head_type).lower()

    if saved_head_type != current_head_type:
        raise ValueError(
            "Checkpoint head mismatch: "
            f"checkpoint uses {saved_head_type!r}, "
            f"configuration uses {current_head_type!r}"
        )

    # Construction du modèle et chargement des poids.
    model = build_model(cfg).to(device)
    model.load_state_dict(
        raw_checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    loss_fn = VICRegLoss(
        invariance_coeff=cfg.loss.invariance_coeff,
        std_coeff=cfg.loss.std_coeff,
        cov_coeff=cfg.loss.cov_coeff,
        variance_target=cfg.loss.variance_target,
        variance_epsilon=cfg.loss.variance_epsilon,
    )

    use_bf16 = should_use_bfloat16(cfg.optimization.precision, device)

    eval_transform = get_train_transforms(
        crop_scale=tuple(cfg.data.crop_scale)
    )

    raw_test = CIFAR10(
        root=cfg.data.root,
        train=False,
        download=cfg.data.download,
    )
    pairs: list[PairEvaluation] = []

    # Évaluation du modèle sur chaque paire de vues
    for pair_seed in pair_seeds:
        test_dataset = DeterministicPairDataset(
            dataset=raw_test,
            transform=eval_transform,
            pair_seed=pair_seed,
        )

        loader = DataLoader(
            test_dataset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=should_pin_memory(
                cfg.data.get("pin_memory", True),
                device,
            ),
            shuffle=False,
            drop_last=False,
        )
        pairs.append(
            evaluate_pair(
                model=model,
                loader=loader,
                loss_fn=loss_fn,
                pair_seed=pair_seed,
                device=device,
                use_bf16=use_bf16,
            )
        )

    totals = [pair.total for pair in pairs]

    # statistics.stdev utilise l'écart-type échantillonnal, ddof=1.
    mean_total = statistics.mean(totals)
    std_total = statistics.stdev(totals)

    digest = hashlib.sha256()

    with checkpoint_path.open("rb") as checkpoint_file:
        for chunk in iter(
            lambda: checkpoint_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return TestEvaluation(
        pairs=tuple(pairs),
        mean_total=mean_total,
        std_total=std_total,
        checkpoint_sha256=digest.hexdigest(),
    )


def write_evaluation_json(
    path: str | Path,
    evaluation: TestEvaluation,
    cfg: DictConfig,
    training_seed: int,
) -> None:
    """Write an auditable evaluation payload as stable UTF-8 JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = asdict(evaluation)
    output_data.update(
        {
            "training_seed": int(training_seed),
            "head_type": str(cfg.model.head_type),
            "config": OmegaConf.to_container(
                cfg,
                resolve=True,
                enum_to_str=True,
            ),
        }
    )

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            output_data,
            output_file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        output_file.write("\n")
