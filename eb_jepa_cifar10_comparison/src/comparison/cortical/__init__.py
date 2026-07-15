"""Public API for the fixed cortical tree implementation."""

from importlib import import_module
from typing import Any

__all__ = ["FixedTreePredictor"]


def __getattr__(name: str) -> Any:
    """Load the predictor lazily until its implementation phase is complete."""
    if name != "FixedTreePredictor":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(".fixed_tree_predictor", package=__name__)
    predictor = module.FixedTreePredictor
    globals()[name] = predictor
    return predictor
