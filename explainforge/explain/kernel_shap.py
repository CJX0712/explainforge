"""Pure-numpy KernelSHAP (Lundberg & Lee, 2017).

Implements the sampling-based Shapley value estimator with the exact Shapley
kernel weighting and a weighted least-squares fit. No torch / no sklearn
dependency — this is the offline-default SOTA backend. Deterministic given
``self.seed``.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..core.errors import ExplainError
from ..core.types import Attributions


class KernelShapExplainer:
    name = "kernel_shap"
    backend = "numpy"

    def __init__(self, nsamples: int = 1024, seed: int = 42):
        self.nsamples = int(nsamples)
        self.seed = int(seed)
        self.background: Optional[np.ndarray] = None
        self.model = None

    def fit(self, model, X_background: np.ndarray):
        self.model = model
        self.background = np.asarray(X_background, dtype=np.float64)
        return self

    @staticmethod
    def _shapley_kernel(M: int, sz: int) -> float:
        if sz == 0 or sz == M:
            return 1e4  # anchor empty / full coalitions
        c = math.comb(M, sz)
        return (M - 1) / (c * sz * (M - sz))

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

        gen = np.random.default_rng(self.seed)
        masks = gen.integers(0, 2, size=(self.nsamples, M)).astype(np.float64)
        masks[0, :] = 1.0
        if self.nsamples > 1:
            masks[1, :] = 0.0
        weights = np.array([self._shapley_kernel(M, int(m.sum())) for m in masks])

        pres = masks  # (S, M) presence indicators
        Z = pres[None, :, :] * X[:, None, :] + (1.0 - pres)[None, :, :] * bg[None, None, :]
        Z = Z.reshape(N * self.nsamples, M)
        P = model.predict_proba(Z)  # (N*S, C)
        C = P.shape[1]
        P = P.reshape(N, self.nsamples, C)
        fz = P[np.arange(N), :, ci]  # (N, S)

        Phi = np.hstack([np.ones((self.nsamples, 1)), pres])  # (S, M+1)
        sw = np.sqrt(weights)
        A = Phi * sw[:, None]
        AtA = A.T @ A
        B = A.T @ (sw[:, None] * fz.T)  # (M+1, N)
        coef = np.linalg.lstsq(AtA, B, rcond=None)[0]  # (M+1, N)
        phi = coef[1:, :].T  # (N, M)
        base = coef[0, :]    # (N,)

        out = [
            Attributions(
                values=phi[n].copy(),
                feature_names=[f"f{i}" for i in range(M)],
                method=self.name,
                background=bg,
                base_value=float(base[n]),
            )
            for n in range(N)
        ]
        return out[0] if single else out
