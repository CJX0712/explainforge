"""Shared dataclasses passed between modules (no business logic here)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np


@dataclass
class Attributions:
    """Per-instance feature attribution.

    ``values`` is a signed (n_features,) array. Convention: larger absolute
    value = more important, consistent across every explainer so cross-method
    faithfulness comparison is fair.
    """

    values: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    method: str = "unknown"
    background: Optional[np.ndarray] = None
    base_value: float = 0.0

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)

    def ranked_features(self, k: Optional[int] = None) -> List[int]:
        order = np.argsort(-np.abs(self.values))
        return list(order[:k]) if k is not None else list(order)

    def top_k_mask(self, k: int) -> np.ndarray:
        mask = np.zeros(len(self.values), dtype=bool)
        for idx in self.ranked_features(k):
            mask[idx] = True
        return mask


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    signal_mask: Optional[np.ndarray] = None  # boolean: which features carry signal
    name: str = "dataset"

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=int)


@dataclass
class BenchmarkRow:
    method: str
    backend: str
    comprehension_auc: float
    comprehension_at_ksignal: float
    sufficiency: float
    insufficiency: float
    vs_uniform: float
    vs_gradient: float
