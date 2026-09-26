"""Pipeline + determinism tests (small config for speed)."""
from __future__ import annotations

import json

from explainforge.core import Config
from explainforge.pipeline import ExplainPipeline


def _small_cfg():
    return Config(seed=42, n_signal=3, n_redundant=3, n_noise=4,
                  n_samples=200, n_test=24, nsamples=128, lime_samples=128)


def test_run_produces_benchmark_rows():
    pipe = ExplainPipeline(_small_cfg())
    rep = pipe.run()
    assert "benchmark" in rep
    methods = {r["method"] for r in rep["benchmark"]}
    assert {"kernel_shap", "lime", "permutation", "gradient_input"} <= methods
    assert rep["verdict"]["grade"] in ("S", "A", "B", "C")


def test_determinism_core_metrics():
    cfg = _small_cfg()
    a = ExplainPipeline(cfg).run()
    b = ExplainPipeline(cfg).run()
    ra = next(r for r in a["benchmark"] if r["method"] == "kernel_shap")
    rb = next(r for r in b["benchmark"] if r["method"] == "kernel_shap")
    assert abs(ra["comprehension_auc"] - rb["comprehension_auc"]) < 1e-9
    assert abs(ra["comprehension_at_ksignal"] - rb["comprehension_at_ksignal"]) < 1e-9


def test_offline_backends_present_in_report():
    # offline = the four non-shap explainers; shap is optional extra validation only
    rep = ExplainPipeline(_small_cfg()).run()
    present = {r["method"] for r in rep["benchmark"]}
    for name in ("kernel_shap", "lime", "permutation", "gradient_input"):
        assert name in present


def test_report_is_json_serializable():
    rep = ExplainPipeline(_small_cfg()).run()
    s = json.dumps(rep, ensure_ascii=False)
    assert "verdict" in json.loads(s)
