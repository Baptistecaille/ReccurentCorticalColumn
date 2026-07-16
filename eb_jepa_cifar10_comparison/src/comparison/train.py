import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from pathlib import Path
from omegaconf import DictConfig
from tqdm.auto import tqdm

from .augmentation import get_train_transforms
from .data import make_dataloaders
from .losses import VICRegLoss, collapse_diagnostics, CollapseDiagnostics
from .model import ImageSSL, build_model
from .optim import LARS, WarmupCosineScheduler
from .checkpoint import save_checkpoint, load_checkpoint, setup_device, setup_seed



@dataclass(frozen=True)
class EpochMetrics:
    mean_loss: float
    num_batches: int

@dataclass(frozen=True)
class ValidationMetrics:
    mean_loss: float
    num_batches: int
    collapse_diagnostics: CollapseDiagnostics


@dataclass(frozen=True)
class RunResult:
    best_checkpoint: Path
    best_validation: float
    best_epoch: int
    final_epoch: int


def train_epoch(
    model: ImageSSL,
    train_loader: DataLoader,
    optimizer: LARS,
    scheduler: WarmupCosineScheduler,
    loss_fn: VICRegLoss,
    device: torch.device,
    use_bf16: bool,
    progress_description: str = "Training",
) -> EpochMetrics:
    
    """ Train the model for one epoch """

    # Set the model to training mode
    model.train()

    total_loss = 0.0
    num_batches = 0
    total_exemples = 0

    progress = tqdm(
        train_loader,
        desc=progress_description,
        unit="batch",
        dynamic_ncols=True,
    )

    for batch in progress:

        view1, view2 = batch
        view1, view2 = view1.to(device), view2.to(device)

        # Gradient reinitialization
        optimizer.zero_grad()

        # Forward pass 
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            _, proj1 = model(view1)
            _, proj2 = model(view2)

            # Compute VICReg loss
            loss = loss_fn(proj1, proj2)
        
        # Error if loss is None or not finite
        if loss is None:
            raise ValueError("Loss is None. Check the loss function and model outputs.")
        elif not torch.isfinite(loss):
            raise FloatingPointError("Loss is not finite. Check the loss function and model outputs.")
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = view1.size(0)
        total_loss += loss.item() * batch_size
        total_exemples += batch_size
        num_batches += 1
        progress.set_postfix(loss=f"{total_loss / total_exemples:.4f}")

    # Check if the loader is empty
    if num_batches == 0:
        raise ValueError("The DataLoader is empty. Ensure that the dataset is not empty and that the DataLoader is correctly configured.")
    
    return EpochMetrics(mean_loss=total_loss / total_exemples, num_batches=num_batches)



def evaluate_validation(model: ImageSSL, loader: DataLoader, loss_fn: VICRegLoss, device: torch.device, use_bf16: bool) -> ValidationMetrics:

    # Set the model to evaluation mode
    model.eval()

    # Initialize metrics
    total_loss = 0.0
    num_batches = 0
    total_exemples = 0
    all_projections = []


    # Iterate over the validation data
    with torch.no_grad():
        for batch in loader:
            view1, view2 = batch
            view1, view2 = view1.to(device), view2.to(device)

            # Forward pass
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                _, proj1 = model(view1)
                _, proj2 = model(view2)

                

                # Compute VICReg loss
                loss = loss_fn(proj1, proj2)
                
                # Add projections for collapse diagnostics
                all_projections.append(proj1.detach().float().cpu())
                all_projections.append(proj2.detach().float().cpu())

            # Error if loss is None or not finite
            if loss is None:
                raise ValueError("Loss is None. Check the loss function and model outputs.")
            elif not torch.isfinite(loss):
                raise FloatingPointError("Loss is not finite. Check the loss function and model outputs.")

            batch_size = view1.size(0)
            total_loss += loss.item() * batch_size
            total_exemples += batch_size
            num_batches += 1
        
        representations = torch.cat(all_projections, dim=0)
        diagnostics = collapse_diagnostics(representations)

    return ValidationMetrics(mean_loss=total_loss / total_exemples, num_batches=num_batches, collapse_diagnostics=diagnostics)


def run(cfg: DictConfig, seed: int, output_dir: str | Path, resume_from: str | Path | None = None) -> RunResult:

    """Run training and validation for one configuration and one seed."""

    # ------------------------------------------------------------------
    # 1. Préparer le chemin de sortie
    # ------------------------------------------------------------------
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    

    # ------------------------------------------------------------------
    # 2. Configurer l’environnement reproductible
    # ------------------------------------------------------------------
    device = setup_device() # GPU -> CPU 

    setup_seed(seed) # set seed for reproducibility based on the device 

    # ------------------------------------------------------------------
    # 3. Construire les données
    # ------------------------------------------------------------------
    train_transforms = get_train_transforms(
        crop_scale=tuple(cfg.data.crop_scale),
    )
    validation_transforms = get_train_transforms(
        crop_scale=tuple(cfg.data.crop_scale),
    )

    loaders = make_dataloaders(
        cfg=cfg,
        train_transform=train_transforms,
        eval_transform=validation_transforms,
        seed=seed,
    )
    train_loader = loaders.train
    validation_loader = loaders.validation

    # ------------------------------------------------------------------
    # 4. Construire le modèle et la loss
    # ------------------------------------------------------------------
    model = build_model(cfg=cfg).to(device)

    loss_fn = VICRegLoss(
        invariance_coeff=cfg.loss.invariance_coeff,
        std_coeff=cfg.loss.std_coeff,
        cov_coeff=cfg.loss.cov_coeff,
        variance_target=cfg.loss.variance_target,
        variance_epsilon=cfg.loss.variance_epsilon,
    )

    # ------------------------------------------------------------------
    # 5. Construire l’optimiseur
    # ------------------------------------------------------------------
    optimizer = LARS(
        model.parameters(),
        lr=cfg.optimization.learning_rate,
        weight_decay=cfg.optimization.weight_decay,
        momentum=cfg.optimization.momentum,
    )

    # ------------------------------------------------------------------
    # 6. Construire le scheduler
    # ------------------------------------------------------------------
    steps_per_epoch = len(train_loader)
    total_steps = cfg.optimization.epochs * steps_per_epoch
    warmup_steps = cfg.optimization.warmup_epochs * steps_per_epoch

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        start_lr=cfg.optimization.warmup_start_lr,
        base_lr=cfg.optimization.learning_rate,
        final_lr=cfg.optimization.min_lr,
    )
    use_bf16 = str(cfg.optimization.precision).lower() == "bfloat16"

    # ------------------------------------------------------------------
    # 7. Initialiser l’état du run
    # ------------------------------------------------------------------
    start_epoch = 0
    best_validation = float('inf')
    best_epoch = -1
    best_checkpoint = output_dir / "best.pt"

    # ------------------------------------------------------------------
    # 8. Reprendre un checkpoint si demandé
    # ------------------------------------------------------------------
    if resume_from is not None:
        resume_state = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            device=device,
            seed=seed,
            train_generator=loaders.train_generator,
        )

        start_epoch = resume_state.start_epoch
        best_validation = resume_state.best_validation
        best_epoch = resume_state.best_epoch

    # ------------------------------------------------------------------
    # 9. Boucle principale
    # ------------------------------------------------------------------
    final_epoch = start_epoch - 1

    for epoch in range(start_epoch, cfg.optimization.epochs):
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            use_bf16=use_bf16,
            progress_description=f"Epoch {epoch + 1}/{cfg.optimization.epochs}",
        )

        validation_metrics = evaluate_validation(
            model=model,
            loader=validation_loader,
            loss_fn=loss_fn,
            device=device,
            use_bf16=use_bf16,
        )

        final_epoch = epoch

        # --------------------------------------------------------------
        # 9a. Vérifier si cette validation est la meilleure
        # --------------------------------------------------------------
        if validation_metrics.mean_loss < best_validation:
            best_validation = validation_metrics.mean_loss
            best_epoch = epoch

            save_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_validation=best_validation,
                best_epoch=best_epoch,
                cfg=cfg,
                seed=seed,
                train_generator=loaders.train_generator,
            )

        # --------------------------------------------------------------
        # 9b. Checkpoint périodique
        # --------------------------------------------------------------
        if (epoch + 1) % cfg.experiment.checkpoint_every == 0:
            periodic_checkpoint = output_dir / f"periodic_checkpoint_epoch_{epoch + 1}.pt"

            save_checkpoint(
                periodic_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_validation=best_validation,
                best_epoch=best_epoch,
                cfg=cfg,
                seed=seed,
                train_generator=loaders.train_generator,
            )

    # ------------------------------------------------------------------
    # 10. Construire le résultat final
    # ------------------------------------------------------------------
    return RunResult(
        best_checkpoint=best_checkpoint,
        best_validation=best_validation,
        best_epoch=best_epoch,
        final_epoch=final_epoch,
    )
