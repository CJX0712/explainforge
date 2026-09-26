"""Optional ``shap`` reference backend.

Used only to validate that the pure-numpy KernelSHAP matches the reference
implementation within tolerance. Never on the determinism-critical path: if
``shap`` is unavailable the system still runs fully offline.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.errors import BackendUnavailableError
from ..core.types import Attributions

try:  # pragma: no cover - import guard
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    SHAP_AVAILABLE = False


class ShapRefExplainer:
    name = "shap_ref"
    backend = "shap"

    def __init__(self, nsamples: int = 256, seed: int = 42):
        self.nsamples = int(nsamples)
        self.seed = int(seed)
        self.bg: Optional[np.ndarray] = None
        self.model = None

    @staticmethod
    def available() -> bool:
        return SHAP_AVAILABLE

    def fit(self, model, X_background: np.ndarray):
        if not SHAP_AVAILABLE:
            raise BackendUnavailableError("shap is not installed")
        self.model = model
        # Align background with the pure-numpy backend (single mean point) so the
        # comparison is apples-to-apples; shap's reference is the background mean.
        self.bg = np.asarray(X_background, dtype=np.float64).mean(axis=0, keepdims=True)
        return self

    def explain(self, X, model=None, class_index=0):
        if not SHAP_AVAILABLE:
            raise BackendUnavailableError("shap is not installed")
        model = model if model is not None else self.model
        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X[None, :]
        N, M = X.shape
        ci = int(class_index) if isinstance(class_index, (int, np.integer)) else int(class_index[0])
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, ci], self.bg)
        sv = np.asarray(explainer.shap_values(X, nsamples=self.nsamples))
        # KernelExplainer returns (N, M) for a single class target
        if sv.ndim == 3:
            sv = sv[:, :, ci]
        out = [
            Attributions(
                values=sv[n].copy(),
                feature_names=[f"f{i}" for i in range(M)],
                method=self.name,
                background=self.bg.mean(axis=0),
                base_value=0.0,
            )
            for n in range(N)
        ]
        return out[0] if single else out
