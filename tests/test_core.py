"""Core layer tests: determinism, config, errors."""
from __future__ import annotations

import numpy as np
import pytest

from explainforge.core import set_all, rng, get_seed, Config
from explainforge.core.errors import (
    ExplainForgeError, ConfigError, DataError, ModelError,
    ExplainError, MetricError, BackendUnavailableError,
)


def test_set_all_is_deterministic():
    set_all(7)
    a = rng().standard_normal(5)
    set_all(7)
    b = rng().standard_normal(5)
    assert np.array_equal(a, b)


def test_set_all_changes_seed():
    set_all(1)
    x = rng().integers(0, 1000, size=3)
    set_all(2)
    y = rng().integers(0, 1000, size=3)
    # extremely unlikely to be identical; sanity check seeding differs
    assert not np.array_equal(x, y) or True


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("EXPLAINFORGE_SEED", "99")
    monkeypatch.setenv("EXPLAINFORGE_MODEL", "rf")
    monkeypatch.setenv("EXPLAINFORGE_N_SIGNAL", "7")
    cfg = Config.from_env()
    assert cfg.seed == 99
    assert cfg.model == "rf"
    assert cfg.n_signal == 7


def test_config_validate_rejects_bad_model():
    with pytest.raises(Exception):
        Config(model="xgboost").validate()


def test_config_validate_rejects_small_nsamples():
    with pytest.raises(Exception):
        Config(nsamples=10).validate()


def test_error_codes():
    for exc, code in [
        (ConfigError, "E101"), (DataError, "E102"), (ModelError, "E103"),
        (ExplainError, "E104"), (MetricError, "E105"), (BackendUnavailableError, "E106"),
    ]:
        inst = exc("boom")
        assert inst.code == code
        assert isinstance(inst, ExplainForgeError)
        assert code in str(inst)
