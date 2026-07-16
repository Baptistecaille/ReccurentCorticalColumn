"""CIFAR-10 datasets, deterministic splits, and paired-view loaders.

``PairedViewDataset`` is adapted from ``ImageDataset`` in
``facebookresearch/eb_jepa/examples/image_jepa/dataset.py:99-113``
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import random

import numpy as np
from omegaconf import DictConfig
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10

from .checkpoint import seed_worker


def _extract_image(sample: object) -> object:
    """Discard a dataset label while accepting image-only datasets too."""
    if isinstance(sample, tuple):
        return sample[0]
    return sample


class PairedViewDataset(Dataset):
    """Return two independent augmentations of the same source image."""

    def __init__(self, dataset: Dataset, transform: Callable) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image = _extract_image(self.dataset[index])
        return self.transform(image), self.transform(image)


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    validation: list[int]


def build_split_indices(
    dataset_size: int,
    validation_size: int,
    split_seed: int,
) -> SplitIndices:
    """Create a deterministic, exhaustive train/validation partition."""
    if dataset_size < 2:
        raise ValueError(f"dataset_size must be at least 2, received {dataset_size}")
    if not 1 <= validation_size < dataset_size:
        raise ValueError(
            "validation_size must be between 1 and dataset_size - 1; "
            f"received validation_size={validation_size}, dataset_size={dataset_size}"
        )

    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    validation = indices[:validation_size]
    train = indices[validation_size:]
    return SplitIndices(train=train, validation=validation)


@contextmanager
def _temporary_random_seed(seed: int) -> Iterator[None]:
    """Temporarily seed CPU RNGs and restore them even when a transform fails."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


class DeterministicPairDataset(Dataset):
    """Return a stable pair of augmented views for each source image."""

    def __init__(
        self,
        dataset: Dataset,
        transform: Callable,
        pair_seed: int,
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.pair_seed = pair_seed

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image = _extract_image(self.dataset[index])
        seed_base = (self.pair_seed * 1_000_003 + index * 2) % (2**63 - 1)

        with _temporary_random_seed(seed_base):
            view1 = self.transform(image)
        with _temporary_random_seed(seed_base + 1):
            view2 = self.transform(image)
        return view1, view2


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader

    @property
    def train_generator(self) -> torch.Generator:
        generator = self.train.generator
        if generator is None:
            raise RuntimeError("The train DataLoader has no generator")
        return generator


def make_dataloaders(
    cfg: DictConfig,
    train_transform: Callable,
    eval_transform: Callable,
    seed: int,
) -> DataLoaders:
    """Build paired-view CIFAR-10 loaders without exposing labels."""
    raw_train = CIFAR10(
        root=cfg.data.root,
        train=True,
        download=cfg.data.download,
    )
    raw_test = CIFAR10(
        root=cfg.data.root,
        train=False,
        download=cfg.data.download,
    )

    expected_train_total = cfg.data.train_size + cfg.data.validation_size
    if len(raw_train) != expected_train_total:
        raise ValueError(
            f"CIFAR-10 train contains {len(raw_train)} images, but the configured "
            f"train/validation sizes sum to {expected_train_total}"
        )
    if len(raw_test) != cfg.data.test_size:
        raise ValueError(
            f"CIFAR-10 test contains {len(raw_test)} images, but "
            f"data.test_size={cfg.data.test_size}"
        )

    split = build_split_indices(
        dataset_size=len(raw_train),
        validation_size=cfg.data.validation_size,
        split_seed=cfg.data.split_seed,
    )
    train_subset = Subset(raw_train, split.train)
    validation_subset = Subset(raw_train, split.validation)

    train_dataset = PairedViewDataset(train_subset, train_transform)
    validation_dataset = DeterministicPairDataset(
        validation_subset,
        eval_transform,
        pair_seed=cfg.data.split_seed,
    )
    test_dataset = DeterministicPairDataset(
        raw_test,
        eval_transform,
        pair_seed=cfg.data.split_seed,
    )

    loader_options = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": bool(cfg.data.get("pin_memory", True)),
        "worker_init_fn": seed_worker,
    }

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=train_generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    return DataLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
    )
