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


def test_notebook_profiles_active_device_and_exports_results() -> None:
    code = _code_by_id()
    profile = code["portable-profiler"] + code["profile-models"]
    export = code["export-results"]
    assert "torch.cuda.synchronize" in profile
    assert "torch.mps.synchronize" in profile
    assert "measure_flops" in profile
    assert "count_parameters" in profile
    assert "comparison_results.csv" in export
    assert "compute_results.csv" in export
    assert "comparison_results.json" in export
    assert '"schema_version": 1' in export


def test_notebook_contains_tables_plots_takeaways_and_final_checks() -> None:
    code = _code_by_id()
    assert "quality_table" in code["build-result-tables"]
    assert "compute_table" in code["build-result-tables"]
    assert "plt.subplots" in code["plot-results"]
    assert "percentage" in code["derive-takeaways"]
    assert "math.isfinite" in code["final-checks"]


def test_readme_documents_local_and_colab_execution() -> None:
    readme = (PROJECT_DIR / "README.md").read_text(encoding="utf-8")
    assert "resnet18_two_epoch_comparison.ipynb" in readme
    assert "uv run jupyter lab" in readme
    assert "Google Colab" in readme
    assert "two epochs" in readme.lower()


def test_notebook_declares_three_arms_including_matched_predictor() -> None:
    code = _code_by_id()
    parameters = code["experiment-parameters"]
    config_source = code["build-configurations"]
    assert 'ARM_LABELS = ("baseline", "predictor", "predictor_matched")' in parameters
    assert (
        '"predictor_matched": PROJECT_ROOT / "configs" / "cortical_matched.yaml"'
        in config_source
    )


def test_notebook_compares_every_arm_against_the_baseline() -> None:
    checks = _code_by_id()["protocol-checks"]
    assert '"predictor_matched": "cortical"' in checks
    assert "for label in ARM_LABELS" in checks
    assert 'if label == "baseline"' in checks


def test_notebook_checks_parameter_parity_before_training() -> None:
    notebook = _load_notebook()
    ids = [cell["id"] for cell in notebook["cells"]]
    assert ids.index("parameter-parity") < ids.index("train-models")

    parity = _code_by_id()["parameter-parity"]
    assert "PARITY_TOLERANCE = 0.01" in parity
    assert "count_parameters" in parity
    assert "build_model" in parity
    assert "assert relative_gap <= PARITY_TOLERANCE" in parity
