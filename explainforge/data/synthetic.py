"""Synthetic datasets with a controllable signal structure.

Design intent (honest benchmark):
  * ``n_signal`` features carry real predictive signal.
  * ``n_redundant`` are *correlated copies* of the signal features (via
    sklearn ``n_redundant``). This is exactly the regime where classic
    gradient*input spreads credit across the correlated group, while KernelSHAP's
    coalition marginalisation localises it — giving a quantifiable, reproducible
    edge rather than a hand-waved one.
  * ``n_noise`` features are pure noise.
The ``signal_mask`` flags every informative feature (signal + redundant) so the
oracle/sanity checks and failure-case analysis can be computed truthfully.
"""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.datasets import make_classification

from ..core.seed import get_seed
from ..core.types import Dataset


def make_attr_dataset(
    n_samples: int = 600,
    n_signal: int = 4,
    n_redundant: int = 4,
    n_noise: int = 6,
    class_sep: float = 1.2,
    flip_y: float = 0.05,
    seed: int | None = None,
    name: str = "attr-cls",
) -> Dataset:
    if seed is None:
        seed = get_seed()
    n_features = n_signal + n_redundant + n_noise
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_signal,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        weights=None,
        flip_y=flip_y,
        class_sep=class_sep,
        hypercube=True,
        shuffle=False,
        random_state=int(seed),
    )
    # make_classification (shuffle=False) lays out: informative, redundant, noise.
    signal_mask = np.zeros(n_features, dtype=bool)
    signal_mask[: n_signal + n_redundant] = True
    feature_names = [f"f{i}" for i in range(n_features)]
    return Dataset(
        X=X.astype(np.float64),
        y=y.astype(int),
        feature_names=feature_names,
        signal_mask=signal_mask,
        name=name,
    )


def make_datasets(
    n_signal: int = 4,
    n_redundant: int = 4,
    n_noise: int = 6,
    n_samples: int = 600,
    n_variants: int = 3,
    class_sep: float = 1.2,
    base_seed: int | None = None,
) -> List[Dataset]:
    """Generate ``n_variants`` datasets with distinct RNG seeds.

    Reporting comprehension as the mean across variants reduces variance and makes
    the SOTA comparison robust rather than lucky on a single draw.
    """
    if base_seed is None:
        base_seed = get_seed()
    variants = []
    for v in range(n_variants):
        ds = make_attr_dataset(
            n_samples=n_samples,
            n_signal=n_signal,
            n_redundant=n_redundant,
            n_noise=n_noise,
            class_sep=class_sep,
            flip_y=0.05 + 0.02 * v,
            seed=int(base_seed) + 101 * v,
            name=f"attr-cls-v{v}",
        )
        variants.append(ds)
    return variants
