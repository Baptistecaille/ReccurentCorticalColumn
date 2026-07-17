import pytest
import torch

from comparison.checkpoint import setup_device


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
