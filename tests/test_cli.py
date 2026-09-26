"""CLI smoke test (offline path, no shap required)."""
from __future__ import annotations

from explainforge.cli import main


def test_cli_runs_and_writes_json(tmp_path):
    out = tmp_path / "benchmark.json"
    rep = main(["--seed", "1", "--model", "rf",
                "--n-signal", "3", "--n-redundant", "3", "--n-noise", "4",
                "--nsamples", "128", "--out", str(out), "--quiet"])
    assert out.exists()
    assert "benchmark" in rep
    assert rep["verdict"]["grade"] in ("S", "A", "B", "C")


def test_cli_logreg_runs(tmp_path):
    out = tmp_path / "b2.json"
    rep = main(["--seed", "2", "--model", "logreg",
                "--n-signal", "3", "--n-redundant", "2", "--n-noise", "3",
                "--nsamples", "128", "--out", str(out), "--quiet"])
    assert out.exists()
    assert any(r["method"] == "kernel_shap" for r in rep["benchmark"])
