"""Global deterministic seeding for ExplainForge.

Single entry point ``set_all(seed)`` wires up Python ``random``, the legacy
``numpy`` global state, and a ``numpy.random.Generator`` used by all samplers
(KernelSHAP coalition draws, LIME perturbations). Two demo runs with the same
seed must produce bit-identical benchmark numbers.
"""
from __future__ import annotations

import random
import numpy as np

_GLOBAL_SEED: int = 42
_RNG: "np.random.Generator | None" = None


def set_all(seed: int = 42) -> "np.random.Generator":
    """Seed every RNG source used by ExplainForge and return the Generator."""
    global _GLOBAL_SEED, _RNG
    _GLOBAL_SEED = int(seed)
    random.seed(_GLOBAL_SEED)
    np.random.seed(_GLOBAL_SEED)
    _RNG = np.random.default_rng(_GLOBAL_SEED)
    return _RNG


def rng() -> "np.random.Generator":
    """Return the global Generator, lazily initialised on first use."""
    global _RNG
    if _RNG is None:
        _RNG = set_all(_GLOBAL_SEED)
    return _RNG


def get_seed() -> int:
    return _GLOBAL_SEED
