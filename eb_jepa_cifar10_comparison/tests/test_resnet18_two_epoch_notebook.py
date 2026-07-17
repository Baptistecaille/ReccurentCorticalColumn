import ast
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_DIR / "scripts" / "resnet18_two_epoch_comparison.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_by_id() -> dict[str, str]:
    return {
        cell["id"]: "".join(cell.get("source", []))
        for cell in _load_notebook()["cells"]
        if cell["cell_type"] == "code"
    }


def test_notebook_has_portable_setup_and_fixed_protocol() -> None:
    code = _code_by_id()
    setup = code["portable-setup"]
    parameters = code["experiment-parameters"]
    assert "google.colab" in setup
    assert "mps" in setup
    assert "/Users/" not in "\n".join(code.values())
    assert "EPOCHS = 2" in parameters
    assert "TRAINING_SEED = 1" in parameters
    assert "PAIR_SEEDS = (11, 22, 33, 44, 55)" in parameters
    assert "BATCH_SIZE = 256" in parameters


def test_every_python_cell_compiles() -> None:
    for cell_id, source in _code_by_id().items():
        ast.parse(source, filename=f"{NOTEBOOK_PATH.name}:{cell_id}")


def test_notebook_builds_both_resnet18_arms_with_shared_overrides() -> None:
    code = _code_by_id()
    config_source = code["build-configurations"]
    assert '"model.backbone=resnet18"' in config_source
    assert 'f"optimization.epochs={EPOCHS}"' in config_source
    assert '"optimization.warmup_epochs=0"' in config_source
    assert '"optimization.learning_rate=0.3"' in config_source
    assert '"baseline": PROJECT_ROOT / "configs" / "baseline.yaml"' in config_source
    assert '"predictor": PROJECT_ROOT / "configs" / "cortical.yaml"' in config_source


def test_notebook_trains_and_evaluates_both_arms() -> None:
    code = _code_by_id()
    training = code["train-models"]
    evaluation = code["evaluate-checkpoints"]
    assert "for label in ARM_LABELS" in training
    assert "run(" in training
    assert "perf_counter" in training
    assert "evaluate_test(" in evaluation
    assert "pair_seeds=PAIR_SEEDS" in evaluation
