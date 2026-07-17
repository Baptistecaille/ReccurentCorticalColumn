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
