"""Evaluation utilities for Random Forest models."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_random_forest(model, features: pd.DataFrame, labels: pd.Series) -> Mapping[str, float]:
    """Compute regression metrics with NaN-safe handling."""

    y_true = np.asarray(labels, dtype=float)
    y_pred = np.asarray(model.predict(features), dtype=float)  # type: ignore[attr-defined]

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan"), "n": 0}

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n": int(mask.sum()),
    }


__all__ = ["evaluate_random_forest"]
