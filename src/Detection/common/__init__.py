"""Detection utilities (lazy submodules — avoids importing ``pycocotools`` until you use ``eval``)."""
from __future__ import annotations

import importlib
from typing import Any

_SUBMODULES = (
    "anchors",
    "boxes",
    "data",
    "eval",
    "hungarian",
    "infer",
    "matcher",
    "selective_search",
    "targets",
    "train",
)

__all__ = list(_SUBMODULES)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
