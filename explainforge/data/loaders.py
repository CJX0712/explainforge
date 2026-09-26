"""File loaders (offline-friendly; pandas optional)."""
from __future__ import annotations

import csv
import numpy as np

from ..core.errors import DataError


def load_csv(path: str, label_column: str | None = None, encoding: str = "utf-8"):
    """Load a CSV into a Dataset.

    If ``label_column`` is given it is treated as ``y`` and dropped from ``X``;
    otherwise the last column is assumed to be the label.
    """
    try:
        import pandas as pd  # optional, preferred when available
        df = pd.read_csv(path, encoding=encoding)
        if label_column is not None and label_column in df.columns:
            y = df[label_column].to_numpy()
            X = df.drop(columns=[label_column]).to_numpy(dtype=np.float64)
        else:
            X = df.to_numpy(dtype=np.float64)
            y = X[:, -1].astype(int)
            X = X[:, :-1]
        feature_names = list(df.columns)
        if label_column is not None:
            feature_names = [c for c in feature_names if c != label_column]
        else:
            feature_names = feature_names[:-1]
        return Dataset(X=X, y=y, feature_names=feature_names, name=path)
    except ImportError:
        pass
    # Pure-stdlib fallback
    rows = []
    with open(path, "r", encoding=encoding, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for row in reader:
            rows.append([float(x) for x in row])
    if not rows:
        raise DataError(f"empty csv: {path}")
    arr = np.asarray(rows, dtype=np.float64)
    if label_column is not None and header is not None:
        col = header.index(label_column)
        y = arr[:, col].astype(int)
        X = np.delete(arr, col, axis=1)
        feature_names = [h for h in header if h != label_column]
    else:
        y = arr[:, -1].astype(int)
        X = arr[:, :-1]
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    return Dataset(X=X, y=y, feature_names=feature_names, name=path)
