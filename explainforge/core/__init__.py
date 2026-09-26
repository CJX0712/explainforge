"""Core primitives: seed / types / errors / config / interfaces."""
from __future__ import annotations

from .seed import set_all, rng, get_seed
from .errors import (
    ExplainForgeError,
    ConfigError,
    DataError,
    ModelError,
    ExplainError,
    MetricError,
    BackendUnavailableError,
)
from .types import Attributions, Dataset, BenchmarkRow
from .config import Config
from .interfaces import ModelProtocol, ExplainerProtocol

__all__ = [
    "set_all", "rng", "get_seed",
    "ExplainForgeError", "ConfigError", "DataError", "ModelError",
    "ExplainError", "MetricError", "BackendUnavailableError",
    "Attributions", "Dataset", "BenchmarkRow", "Config",
    "ModelProtocol", "ExplainerProtocol",
]
