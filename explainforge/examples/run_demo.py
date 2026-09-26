"""End-to-end demo: runs the full ExplainForge benchmark and writes benchmark.json.

Run from the project root:
    python explainforge/examples/run_demo.py
The result is a deterministic, reproducible benchmark.json (re-run yields identical
core metrics under the same seed).
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # project root containing the `explainforge` package
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from explainforge.core import Config  # noqa: E402
from explainforge.pipeline import ExplainPipeline  # noqa: E402


def main() -> dict:
    cfg = Config()
    pipe = ExplainPipeline(cfg)
    report = pipe.run()
    out_path = os.path.join(ROOT, "benchmark.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"[ExplainForge] wrote {out_path}")
    print(f"[ExplainForge] grade = {report['verdict']['grade']} | "
          f"kernel_shap comp_auc = {report['verdict']['kernel_shap_comprehension_auc']:.4f}")
    return report


if __name__ == "__main__":
    main()
