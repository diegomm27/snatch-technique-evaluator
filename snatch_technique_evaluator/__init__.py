"""Tools for side-view snatch video analysis."""

from importlib import import_module
from typing import Any

__all__ = [
    "AnalyzerConfig",
    "SnatchAnalysisSession",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        analysis = import_module(".analysis", __name__)
        return getattr(analysis, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
