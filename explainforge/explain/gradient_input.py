"""Finite-difference gradient*input saliency (classic baseline).

Model-agnostic: perturbs each feature by ``eps`` and measures the change in the
target class probability. Attribution = gradient * input. This is the classic
local sensitivity method; on correlated features it spreads credit across the
group, which is exactly the regime where KernelSHAP is expected to localise
better — making it a fair, honest strong baseline rather than a strawman.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.errors import ExplainError
from ..core.types import Attributions


class GradientInputExplainer:
    name = "gradient_input"
    backend = "numpy"

    def __init__(self, eps: float = 1e-3):
        self.eps = float(eps)
        self.model = None

    def fit(self, model, X_background: np.ndarray):
        self.model = model
        return self

    def explain(self, X, model=None, class_index=0):
        model = model if model is not None else self.model
        if model is None:
            raise ExplainError("explain requires a model (fit or pass one)")
        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X[None, :]
        N, M = X.shape
        ci = np.full(N, int(class_index), dtype=int) if isinstance(
            class_index, (int, np.integer)) else np.asarray(class_index, dtype=int)

        P0 = model.predict_proba(X)
        C = P0.shape[1]
        f0 = P0[np.arange(N), ci]  # (N,)
        grad = np.zeros((N, M), dtype=np.float64)
        for j in range(M):
            Xp = X.copy()
            Xp[:, j] += self.eps
            Pp = model.predict_proba(Xp)
            fp = Pp[np.arange(N), ci]
            grad[:, j] = (fp - f0) / self.eps
        attr = grad * X  # gradient * input

        out = [
            Attributions(
                values=attr[n].copy(),
                feature_names=[f"f{i}" for i in range(M)],
                method=self.name,
                background=None,
            )
            for n in range(N)
        ]
        return out[0] if single else out
