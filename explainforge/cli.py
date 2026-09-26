"""ExplainForge command-line entry point.

Usage:
    python -m explainforge.cli --seed 42 --model logreg --out benchmark.json
    python -m explainforge.cli --help
"""
from __future__ import annotations

import argparse
import json
import sys


def _print_table(report: dict) -> None:
    rows = report.get("benchmark", [])
    cols = [
        ("method", 16), ("backend", 10), ("comp_auc", 10),
        ("comp@ksig", 11), ("vs_uniform", 12), ("vs_grad", 11),
    ]
    header = "".join(f"{h:<{w}}" for h, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['method']:<16}{r['backend']:<10}"
            f"{r['comprehension_auc']:.4f}    "
            f"{r['comprehension_at_ksignal']:.4f}    "
            f"{r['vs_uniform']:+.4f}    "
            f"{r['vs_gradient']:+.4f}"
        )
    print()
    v = report["verdict"]
    print(f"Grade: {v['grade']}  vs_uniform={v['vs_uniform']:+.4f}  vs_gradient={v['vs_gradient']:+.4f}")
    print(f"shap_validation: {report['shap_validation']}")


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(
        prog="explainforge",
        description="ExplainForge: model-agnostic feature attribution (XAI).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    p.add_argument("--n-signal", type=int, default=4)
    p.add_argument("--n-redundant", type=int, default=4)
    p.add_argument("--n-noise", type=int, default=6)
    p.add_argument("--nsamples", type=int, default=1024)
    p.add_argument("--out", default="benchmark.json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    from .core import Config
    from .pipeline import ExplainPipeline

    cfg = Config(
        seed=args.seed, model=args.model, n_signal=args.n_signal,
        n_redundant=args.n_redundant, n_noise=args.n_noise, nsamples=args.nsamples,
    )
    pipe = ExplainPipeline(cfg)
    report = pipe.run()
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    if not args.quiet:
        _print_table(report)
    return report


if __name__ == "__main__":
    main()
