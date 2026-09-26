"""Faithfulness (ERASER-style comprehension) for feature attributions.

Semantics, unified across every explainer so cross-method comparison is fair:
  * larger absolute attribution -> more important feature
  * "remove top-k" = SHUFFLE those k feature columns across instances. This is a
    distribution-preserving, model-agnostic removal: the feature's marginal
    statistics are intact but its relationship to the label is destroyed, so any
    black-box (linear OR tree) loses exactly the signal those columns carried.
    (Mean-imputation would leave an informative constant and is unfair to trees.)
  * global selection: features are ranked by mean |attribution| over instances,
    the top-k columns are perturbed for ALL instances (standard comprehension setup).
  * comprehension = normalised confidence drop when the top-k features are removed.
    Lives in [0, 1]; higher = more faithful. A faithful explanation makes the
    model's confidence collapse fast as we remove its most-attributed features.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz
_PERM_SEED = 1234  # fixed -> deterministic comprehension regardless of cfg


def _predicted_class(model, X):
    P = model.predict_proba(np.asarray(X, dtype=np.float64))
    return P.argmax(axis=1), P


def _confidence_of(P, pred):
    return P[np.arange(P.shape[0]), pred]


def _global_topk_cols(attributions: np.ndarray, k: int) -> np.ndarray:
    imp = np.abs(np.asarray(attributions, dtype=np.float64)).mean(axis=0)
    return np.argsort(-imp)[: min(k, len(imp))]


def _permute_columns(X: np.ndarray, cols, rng) -> np.ndarray:
    Xk = X.copy()
    for j in cols:
        Xk[:, j] = rng.permutation(X[:, j])
    return Xk


def faithfulness_metrics(
    model,
    X: np.ndarray,
    attributions: np.ndarray,
    background: Optional[np.ndarray] = None,
    ks: Sequence[int] = (1, 2, 3, 4, 5, 6, 8, 10, 12),
    pred: Optional[np.ndarray] = None,
):
    X = np.asarray(X, dtype=np.float64)
    N, M = X.shape
    attr = np.asarray(attributions, dtype=np.float64)
    if pred is None:
        pred, P0 = _predicted_class(model, X)
    else:
        P0 = model.predict_proba(X)
    c0 = _confidence_of(P0, pred)  # (N,)

    rng = np.random.default_rng(_PERM_SEED)
    Xblank = _permute_columns(X, np.arange(M), rng)  # destroy every relationship
    pblank = _confidence_of(model.predict_proba(Xblank), pred)
    max_drop = np.clip(c0 - pblank, 1e-9, None)

    comp_per_k = []
    for k in ks:
        cols = _global_topk_cols(attr, k)
        Xk = _permute_columns(X, cols, rng)
        pk = _confidence_of(model.predict_proba(Xk), pred)
        drop = c0 - pk
        comp = np.clip(drop / max_drop, 0.0, 1.0)
        comp_per_k.append(float(comp.mean()))

    auc = float(_trapz(np.asarray(comp_per_k), np.asarray(list(ks))) / max(ks))
    return {"comprehension_auc": auc, "comprehension_per_k": comp_per_k, "ks": list(ks)}


def sufficiency_at_k(model, X, attributions, background=None, k: int = 1,
                     pred: Optional[np.ndarray] = None) -> float:
    """Keep only top-k features (perturb the rest); higher retained confidence = more sufficient."""
    X = np.asarray(X, dtype=np.float64)
    if pred is None:
        pred, P0 = _predicted_class(model, X)
    else:
        P0 = model.predict_proba(X)
    c0 = _confidence_of(P0, pred)
    rng = np.random.default_rng(_PERM_SEED)
    all_cols = np.arange(X.shape[1])
    top = _global_topk_cols(attributions, k)
    rest = np.array([j for j in all_cols if j not in set(top.tolist())])
    Xk = _permute_columns(X, rest, rng)  # perturb everything except the top-k
    pk = _confidence_of(model.predict_proba(Xk), pred)
    return float(np.clip(pk / np.clip(c0, 1e-9, None), 0.0, 1.0).mean())


def insufficiency_at_k(model, X, attributions, background=None, k: int = 1,
                      pred: Optional[np.ndarray] = None) -> float:
    """Remove top-k features (perturb them); higher confidence loss = more insufficient (faithful)."""
    X = np.asarray(X, dtype=np.float64)
    if pred is None:
        pred, P0 = _predicted_class(model, X)
    else:
        P0 = model.predict_proba(X)
    c0 = _confidence_of(P0, pred)
    rng = np.random.default_rng(_PERM_SEED)
    top = _global_topk_cols(attributions, k)
    Xk = _permute_columns(X, top, rng)
    pk = _confidence_of(model.predict_proba(Xk), pred)
    loss = (c0 - pk) / np.clip(c0, 1e-9, None)
    return float(np.clip(loss, 0.0, 1.0).mean())


def naive_uniform_attributions(M: int, N: int, seed: int = 0) -> np.ndarray:
    """Naive control: a FIXED random ranking (no signal information).

    Top-k therefore perturbs random columns, yielding low comprehension. This is
    the honest weak baseline we must clearly beat; it is not a strawman because
    the ranking is fixed and reproducible.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    values = np.empty(M, dtype=np.float64)
    values[perm] = np.arange(M, dtype=np.float64)
    return np.tile(values, (N, 1))


def oracle_signal_attributions(signal_mask: np.ndarray, M: int, N: int) -> np.ndarray:
    """Oracle: rank truly-informative features first (upper bound on comprehension)."""
    values = np.where(np.asarray(signal_mask, dtype=bool), 1.0, 0.0).astype(np.float64)
    return np.tile(values, (N, 1))
