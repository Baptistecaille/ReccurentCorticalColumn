import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from comparison.checkpoint import setup_device
import comparison.data as data_module


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_setup_device_auto_priority(
    monkeypatch,
    cuda_available,
    mps_available,
    expected,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(
        torch.backends.mps,
        "is_available",
        lambda: mps_available,
    )

    assert setup_device().type == expected


def test_setup_device_accepts_available_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert setup_device("mps") == torch.device("mps")


def test_setup_device_rejects_unavailable_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS.*explicitly requested"):
        setup_device("mps")


from comparison.checkpoint import should_pin_memory, should_use_bfloat16


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [("cuda", True), ("cpu", True), ("mps", False)],
)
def test_bfloat16_is_disabled_only_on_mps(device_name, expected):
    assert should_use_bfloat16(
        "bfloat16",
        torch.device(device_name),
    ) is expected


def test_non_bfloat16_precision_stays_disabled():
    assert not should_use_bfloat16("float32", torch.device("cuda"))


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [("cuda", True), ("cpu", True), ("mps", False)],
)
def test_pin_memory_is_disabled_only_on_mps(device_name, expected):
    assert should_pin_memory(True, torch.device(device_name)) is expected


def test_disabled_pin_memory_stays_disabled():
    assert not should_pin_memory(False, torch.device("cuda"))


class _SizedDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.zeros(3, 4, 4), 0


def test_make_dataloaders_disables_pin_memory_on_mps(monkeypatch):
    def fake_cifar10(*, train, **kwargs):
        del kwargs
        return _SizedDataset(6 if train else 3)

    monkeypatch.setattr(data_module, "CIFAR10", fake_cifar10)
    cfg = OmegaConf.create(
        {
            "data": {
                "root": "unused",
                "download": False,
                "train_size": 4,
                "validation_size": 2,
                "test_size": 3,
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": True,
                "split_seed": 42,
            }
        }
    )

    loaders = data_module.make_dataloaders(
        cfg=cfg,
        train_transform=lambda image: image,
        eval_transform=lambda image: image,
        seed=1,
        device=torch.device("mps"),
    )

    assert not loaders.train.pin_memory
    assert not loaders.validation.pin_memory
    assert not loaders.test.pin_memory
