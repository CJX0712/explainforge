"""Protocol interfaces — the contract every module depends on.

Implementations may be swapped (numpy vs sklearn vs shap) as long as they honour
these protocols, keeping the call graph acyclic: cli -> pipeline -> {data,
explain, eval} -> core.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import Attributions


@runtime_checkable
class ModelProtocol(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class ExplainerProtocol(Protocol):
    name: str
    backend: str  # "numpy" | "sklearn" | "shap"

    def fit(self, model: ModelProtocol, X_background: np.ndarray) -> "ExplainerProtocol": ...
    def explain(self, X: np.ndarray, model: ModelProtocol | None = None,
                class_index: int = 0) -> Attributions | list[Attributions]: ...
