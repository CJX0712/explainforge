"""Pure-numpy LIME (Ribeiro et al., 2016).

Samples binary perturbations of each instance, weights them by proximity to the
original point, and fits a weighted ridge regression whose coefficients are the
local importances. Offline default, deterministic given ``self.seed``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.errors import ExplainError
from ..core.types import Attributions


class LimeExplainer:
    name = "lime"
    backend = "numpy"

    def __init__(self, nsamples: int = 1024, kernel_width: Optional[float] = None,
                 ridge: float = 1e-3, seed: int = 42):
        self.nsamples = int(nsamples)
        self.kernel_width = kernel_width
        self.ridge = ridge
        self.seed = int(seed)
        self.background: Optional[np.ndarray] = None
        self.model = None

    def fit(self, model, X_background: np.ndarray):
        self.model = model
        self.background = np.asarray(X_background, dtype=np.float64)
        return self

    def explain(self, X, model=None, class_index=0):
        model = model if model is not None else self.model
        if model is None:
            raise ExplainError("explain requires a model (fit or pass one)")
        if self.background is None:
            raise ExplainError("call fit() with a background set first")
        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X[None, :]
        N, M = X.shape
        bg = np.asarray(self.background, dtype=np.float64)
        ci = np.full(N, int(class_index), dtype=int) if isinstance(
            class_index, (int, np.integer)) else np.asarray(class_index, dtype=int)
        sigma = self.kernel_width if self.kernel_width is not None else (np.sqrt(M) * 0.75)

        gen = np.random.default_rng(self.seed)
        masks = gen.integers(0, 2, size=(self.nsamples, M)).astype(np.float64)
        masks[0, :] = 1.0
        pres = masks
        Phi = np.hstack([np.ones((self.nsamples, 1)), pres])  # (S, M+1)
        I = np.eye(M + 1)
        I[0, 0] = 0.0  # do not regularise the intercept

        out = []
        for n in range(N):
            x = X[n]
            Z = pres * x[None, :] + (1.0 - pres) * bg[None, :]
            P = model.predict_proba(Z)
            C = P.shape[1]
            fz = P[np.arange(self.nsamples), ci[n]] if N > 1 else P[:, ci[0]]
            dist = np.sqrt(((Z - x) ** 2).sum(axis=1))
            w = np.exp(-(dist ** 2) / (sigma ** 2))
            sw = np.sqrt(np.clip(w, 1e-12, None))
            A = Phi * sw[:, None]
            AtA = A.T @ A + self.ridge * I
            B = A.T @ (sw[:, None] * fz[:, None])
            coef = np.linalg.solve(AtA, B).ravel()
            out.append(Attributions(
                values=coef[1:].copy(),
                feature_names=[f"f{i}" for i in range(M)],
                method=self.name,
                background=bg,
                base_value=float(coef[0]),
            ))
        return out[0] if single else out
