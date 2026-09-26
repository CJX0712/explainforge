"""Configuration object with ENV_XXX_ overrides and schema validation.

All tunables live here so the pipeline is reproducible from a single source.
Environment overrides use the ``EXPLAINFORGE_`` prefix (e.g. ``EXPLAINFORGE_SEED``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    n_signal: int = 4           # truly informative features
    n_redundant: int = 4        # correlated copies of signal (SHAP marginalization edge)
    n_noise: int = 6            # pure noise features
    n_samples: int = 600
    n_test: int = 60
    nsamples: int = 1024        # KernelSHAP coalition samples
    lime_samples: int = 1024
    class_sep: float = 1.2
    model: str = "rf"           # logreg | rf
    ks: tuple = (1, 2, 3, 4, 5, 6, 8, 10, 12)

    @classmethod
    def from_env(cls, prefix: str = "EXPLAINFORGE_") -> "Config":
        cfg = cls()
        mapping = {
            "SEED": "seed", "N_SIGNAL": "n_signal", "N_REDUNDANT": "n_redundant",
            "N_NOISE": "n_noise", "N_SAMPLES": "n_samples", "N_TEST": "n_test",
            "NSAMPLES": "nsamples", "LIME_SAMPLES": "lime_samples",
            "CLASS_SEP": "class_sep", "MODEL": "model",
        }
        for env_key, attr in mapping.items():
            val = os.environ.get(prefix + env_key)
            if val is None:
                continue
            cur = getattr(cfg, attr)
            if isinstance(cur, bool):
                setattr(cfg, attr, val.lower() in ("1", "true", "yes"))
            elif isinstance(cur, int):
                setattr(cfg, attr, int(val))
            elif isinstance(cur, float):
                setattr(cfg, attr, float(val))
            else:
                setattr(cfg, attr, val)
        return cfg

    def validate(self) -> "Config":
        if self.n_signal < 1:
            raise ValueError("n_signal must be >= 1")
        if self.n_samples < 50:
            raise ValueError("n_samples too small")
        if self.n_test < 10:
            raise ValueError("n_test too small")
        if self.nsamples < 64:
            raise ValueError("nsamples too small")
        if self.model not in ("logreg", "rf"):
            raise ValueError("model must be logreg|rf")
        if self.n_signal + self.n_redundant + self.n_noise < 2:
            raise ValueError("need at least 2 features")
        return self

    @property
    def n_features(self) -> int:
        return self.n_signal + self.n_redundant + self.n_noise
