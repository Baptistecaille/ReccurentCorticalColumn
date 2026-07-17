from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from comparison.checkpoint import setup_device
import comparison.data as data_module
import comparison.evaluate as evaluate_module
import comparison.train as train_module


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


def test_train_run_disables_bfloat16_for_mocked_mps(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "data": {"crop_scale": [0.2, 1.0]},
            "loss": {
                "invariance_coeff": 1.0,
                "std_coeff": 1.0,
                "cov_coeff": 1.0,
                "variance_target": 1.0,
                "variance_epsilon": 0.0001,
            },
            "optimization": {
                "epochs": 1,
                "learning_rate": 0.1,
                "warmup_epochs": 0,
                "warmup_start_lr": 0.01,
                "min_lr": 0.0,
                "weight_decay": 0.0,
                "momentum": 0.0,
                "precision": "bfloat16",
            },
            "experiment": {"checkpoint_every": 1},
        }
    )
    model = MagicMock()
    model.to.return_value = model
    model.parameters.return_value = [torch.nn.Parameter(torch.ones(1))]
    loaders = SimpleNamespace(
        train=[object()],
        validation=[object()],
        train_generator=torch.Generator(),
    )
    train_epoch = MagicMock(
        return_value=train_module.EpochMetrics(mean_loss=1.0, num_batches=1)
    )

    monkeypatch.setattr(train_module, "setup_device", lambda: torch.device("mps"))
    monkeypatch.setattr(train_module, "get_train_transforms", lambda **_: object())
    monkeypatch.setattr(train_module, "make_dataloaders", lambda **_: loaders)
    monkeypatch.setattr(train_module, "build_model", lambda **_: model)
    monkeypatch.setattr(train_module, "train_epoch", train_epoch)
    monkeypatch.setattr(
        train_module,
        "evaluate_validation",
        lambda **_: train_module.ValidationMetrics(
            mean_loss=1.0,
            num_batches=1,
            collapse_diagnostics=MagicMock(),
        ),
    )
    monkeypatch.setattr(train_module, "save_checkpoint", lambda *_, **__: None)

    train_module.run(cfg=cfg, seed=1, output_dir=tmp_path)

    assert train_epoch.call_args.kwargs["use_bf16"] is False


def test_evaluate_test_disables_mps_bfloat16_and_pin_memory(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "data": {
                "root": "unused",
                "download": False,
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": True,
                "crop_scale": [0.2, 1.0],
            },
            "model": {"head_type": "baseline"},
            "loss": {
                "invariance_coeff": 1.0,
                "std_coeff": 1.0,
                "cov_coeff": 1.0,
                "variance_target": 1.0,
                "variance_epsilon": 0.0001,
            },
            "optimization": {"precision": "bfloat16"},
        }
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    model = MagicMock()
    model.to.return_value = model
    captured_calls = []

    def capture_evaluation(*, loader, pair_seed, use_bf16, **_):
        captured_calls.append((use_bf16, loader.pin_memory))
        return evaluate_module.PairEvaluation(
            pair_seed=pair_seed,
            total=1.0,
            invariance=1.0,
            variance=1.0,
            covariance=1.0,
            mean_std=1.0,
            min_std=1.0,
            collapsed_fraction=0.0,
        )

    monkeypatch.setattr(
        evaluate_module,
        "setup_device",
        lambda: torch.device("mps"),
    )
    monkeypatch.setattr(
        evaluate_module.torch,
        "load",
        lambda *_args, **_kwargs: {
            "head_type": "baseline",
            "model_state_dict": {},
        },
    )
    monkeypatch.setattr(evaluate_module, "build_model", lambda _: model)
    monkeypatch.setattr(evaluate_module, "get_train_transforms", lambda **_: lambda x: x)
    monkeypatch.setattr(evaluate_module, "CIFAR10", lambda **_: _SizedDataset(1))
    monkeypatch.setattr(evaluate_module, "evaluate_pair", capture_evaluation)

    evaluate_module.evaluate_test(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        pair_seeds=(1, 2, 3, 4, 5),
    )

    assert captured_calls == [(False, False)] * 5
