"""Data layer: synthetic generators + file loaders."""
from __future__ import annotations

from .synthetic import make_attr_dataset, make_datasets
from .loaders import load_csv

__all__ = ["make_attr_dataset", "make_datasets", "load_csv"]
