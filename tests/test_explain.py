"""Explainer tests: interface contract, shapes, graceful degradation."""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from explainforge.data import make_attr_dataset
from explainforge.explain import (
    KernelShapExplainer, LimeExplainer, PermutationExplainer,
    GradientInputExplainer, build, available, SHAP_AVAILABLE,
)


def _model(seed=0):
    ds = make_attr_dataset(seed=seed, n_samples=200, n_signal=3, n_redundant=2, n_noise=3)
    m = Pipeline([("scaler", StandardScaler()),
                  ("clf", RandomForestClassifier(n_estimators=40, random_state=seed, n_jobs=1))])
    m.fit(ds.X, ds.y)
    return m, ds


def test_kernel_shap_shape_and_values():
    m, ds = _model()
    exp = KernelShapExplainer(nsamples=128, seed=0)
    exp.fit(m, ds.X.mean(axis=0))
    pred = m.predict(ds.X[:10])
    atts = exp.explain(ds.X[:10], m, class_index=pred)
    assert len(atts) == 10
    arr = np.stack([a.values for a in atts])
    assert arr.shape == (10, ds.X.shape[1])
    assert np.all(np.isfinite(arr))


def test_lime_shape():
    m, ds = _model()
    exp = LimeExplainer(nsamples=128, seed=0)
    exp.fit(m, ds.X.mean(axis=0))
    pred = m.predict(ds.X[:10])
    atts = exp.explain(ds.X[:10], m, class_index=pred)
    assert np.stack([a.values for a in atts]).shape == (10, ds.X.shape[1])


def test_gradient_input_shape():
    m, ds = _model()
    exp = GradientInputExplainer()
    exp.fit(m, ds.X.mean(axis=0))
    pred = m.predict(ds.X[:10])
    atts = exp.explain(ds.X[:10], m, class_index=pred)
    assert np.stack([a.values for a in atts]).shape == (10, ds.X.shape[1])


def test_permutation_requires_labels():
    m, ds = _model()
    exp = PermutationExplainer(n_repeats=5, seed=0)
    exp.fit(m, ds.X, ds.y)
    atts = exp.explain(ds.X[:10], m)
    assert np.stack([a.values for a in atts]).shape[1] == ds.X.shape[1]


def test_registry_build_and_available():
    assert build("kernel_shap", nsamples=64).name == "kernel_shap"
    assert available("kernel_shap") is True
    assert available("shap_ref") is SHAP_AVAILABLE
    # offline backends always available
    for name in ("kernel_shap", "lime", "permutation", "gradient_input"):
        assert available(name)


def test_permutation_without_labels_raises():
    m, ds = _model()
    exp = PermutationExplainer()
    exp.fit(m, ds.X)  # no y
    try:
        exp.explain(ds.X[:5], m)
        raised = False
    except Exception:
        raised = True
    assert raised
