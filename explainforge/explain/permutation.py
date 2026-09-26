"""scikit-learn permutation importance (Breiman, 2001).

A standard, widely-used global attribution method. Used both as a strong
reference baseline and as a cross-check on the local explainers. Requires a
labelled background set (fit with ``y_background``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.errors import ExplainError
from ..core.types import Attributions


class PermutationExplainer:
    name = "permutation"
    backend = "sklearn"

    def __init__(self, n_repeats: int = 10, seed: int = 42):
        self.n_repeats = int(n_repeats)
        self.seed = int(seed)
        self.Xbg: Optional[np.ndarray] = None
        self.ybg: Optional[np.ndarray] = None
        self.model = None

    def fit(self, model, X_background: np.ndarray, y_background=None):
        self.model = model
        self.Xbg = np.asarray(X_background, dtype=np.float64)
        self.ybg = None if y_background is None else np.asarray(y_background, dtype=int)
        return self

    def explain(self, X, model=None, class_index=0, n_repeats: Optional[int] = None):
        from sklearn.inspection import permutation_importance

        model = model if model is not None else self.model
        if model is None:
            raise ExplainError("explain requires a model (fit or pass one)")
        if self.ybg is None or self.Xbg is None:
            raise ExplainError("fit() requires X_background and y_background")
        n_rep = n_repeats or self.n_repeats
        pi = permutation_importance(
            model, self.Xbg, self.ybg,
            n_repeats=n_rep, random_state=self.seed, scoring="accuracy",
        )
        imp = np.asarray(pi.importances_mean, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 1
        N = 1 if single else X.shape[0]
        out = [
            Attributions(
                values=imp.copy(),
                feature_names=[f"f{i}" for i in range(len(imp))],
                method=self.name,
                background=None,
            )
            for _ in range(N)
        ]
        return out[0] if single else out
