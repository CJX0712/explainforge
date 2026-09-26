"""Faithfulness metric tests: oracle must beat naive; bounds in [0,1]."""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from explainforge.data import make_attr_dataset
from explainforge.explain import KernelShapExplainer
from explainforge.eval import (
    faithfulness_metrics, naive_uniform_attributions, oracle_signal_attributions,
)


def _setup(seed=3):
    ds = make_attr_dataset(seed=seed, n_samples=300, n_signal=4, n_redundant=3, n_noise=5)
    Xtr, Xte, ytr, yte = train_test_split_safe(ds)
    m = Pipeline([("scaler", StandardScaler()),
                  ("clf", RandomForestClassifier(n_estimators=60, random_state=seed, n_jobs=1))])
    m.fit(Xtr, ytr)
    return m, ds, Xte


def train_test_split_safe(ds):
    from sklearn.model_selection import train_test_split
    return train_test_split(ds.X, ds.y, test_size=40 / len(ds.y),
                            stratify=ds.y, random_state=11)


def test_oracle_beats_naive():
    m, ds, Xte = _setup()
    pred = m.predict(Xte)
    bg = ds.X[:200].mean(axis=0)
    ks = (1, 2, 3, 4, 5, 6, 8, 10)
    oracle = oracle_signal_attributions(ds.signal_mask, ds.X.shape[1], len(Xte))
    naive = naive_uniform_attributions(ds.X.shape[1], len(Xte), seed=21)
    fo = faithfulness_metrics(m, Xte, oracle, bg, ks, pred=pred)
    fn = faithfulness_metrics(m, Xte, naive, bg, ks, pred=pred)
    # oracle (removes true signal) must be strictly more faithful than random ranking
    assert fo["comprehension_auc"] > fn["comprehension_auc"]


def test_comprehension_in_unit_interval():
    m, ds, Xte = _setup()
    pred = m.predict(Xte)
    bg = ds.X[:200].mean(axis=0)
    ks = (1, 2, 3, 4, 6, 8)
    exp = KernelShapExplainer(nsamples=128, seed=0)
    exp.fit(m, bg)
    atts = exp.explain(Xte, m, class_index=pred)
    attr = np.stack([a.values for a in atts])
    fm = faithfulness_metrics(m, Xte, attr, bg, ks, pred=pred)
    assert 0.0 <= fm["comprehension_auc"] <= 1.0
    assert all(0.0 <= c <= 1.0 for c in fm["comprehension_per_k"])


def test_kernel_shap_matches_oracle_ranking():
    m, ds, Xte = _setup()
    pred = m.predict(Xte)
    bg = ds.X[:200].mean(axis=0)
    exp = KernelShapExplainer(nsamples=256, seed=0)
    exp.fit(m, bg)
    atts = exp.explain(Xte, m, class_index=pred)
    attr = np.stack([a.values for a in atts])
    imp = np.abs(attr).mean(axis=0)
    top = set(np.argsort(-imp)[: ds.signal_mask.sum()].tolist())
    sig = set(np.where(ds.signal_mask)[0].tolist())
    # KernelSHAP should place most true signal features in the top-k
    assert len(top & sig) >= int(0.6 * len(sig))
