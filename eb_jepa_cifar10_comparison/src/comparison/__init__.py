"""Public API for the CIFAR-10 predictor comparison package.

The implementation modules are introduced in later roadmap phases. Imports are
resolved lazily so the phase-0 package remains importable while those modules
are still being written.
"""

from importlib import import_module
from typing import Any

_PUBLIC_OBJECTS = {
    "ImageSSL": (".heads", "ImageSSL"),
    "build_head": (".heads", "build_head"),
    "build_model": (".heads", "build_model"),
    "VICRegLoss": (".losses", "VICRegLoss"),
    "load_config": (".config", "load_config"),
}

__all__ = list(_PUBLIC_OBJECTS)


def __getattr__(name: str) -> Any:
    """Load a public object only when it is first requested."""
    try:
        module_name, object_name = _PUBLIC_OBJECTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

    module = import_module(module_name, package=__name__)
    value = getattr(module, object_name)
    globals()[name] = value
    return value
