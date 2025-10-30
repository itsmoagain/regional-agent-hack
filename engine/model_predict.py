"""Model signal helpers for the insight engine."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:  # scikit-learn and joblib may not be present in minimal environments.
    from joblib import dump, load
    from sklearn.ensemble import RandomForestRegressor
except Exception:  # pragma: no cover - degrade gracefully when unavailable
    dump = load = None  # type: ignore[assignment]
    RandomForestRegressor = None  # type: ignore[assignment]


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def predict_outcomes(region: str, df: pd.DataFrame) -> np.ndarray:
    """Return a 0-1 *model_signal* for every row in ``df``."""

    if df.empty:
        return np.zeros(0, dtype=float)

    features = _prepare_features(df)
    if features.empty:
        return np.zeros(len(df), dtype=float)

    model = _load_model(region)
    if model is None:
        model = _train_lightweight_model(region, features, df)

    try:
        raw_pred = np.asarray(model.predict(features))  # type: ignore[call-arg]
    except Exception:
        raw_pred = features.mean(axis=1).to_numpy()

    return _normalise_signal(raw_pred)


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"]).copy()
    for col in list(numeric.columns):
        if col.lower() in {"model_signal"}:
            numeric.drop(columns=col, inplace=True)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    numeric = numeric.fillna(numeric.mean())
    return numeric


def _load_model(region: str):
    if load is None:
        return None

    direct_path = MODELS_DIR / f"{region}_rf.pkl"
    if direct_path.exists():
        try:
            return load(direct_path)
        except Exception:
            pass

    nested_dir = MODELS_DIR / region
    if nested_dir.is_dir():
        candidates = sorted(nested_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in candidates:
            try:
                return load(path)
            except Exception:
                continue
    return None


def _train_lightweight_model(region: str, features: pd.DataFrame, df: pd.DataFrame):
    if RandomForestRegressor is None:
        return _FallbackModel()

    target_name = _choose_target(df)
    if target_name is None or target_name not in df.columns:
        return _FallbackModel()

    target = df[target_name].replace([np.inf, -np.inf], np.nan).fillna(df[target_name].mean())
    X = features.drop(columns=[c for c in features.columns if c == target_name], errors="ignore")

    if X.empty:
        return _FallbackModel()

    model = RandomForestRegressor(n_estimators=80, random_state=42)
    try:
        model.fit(X, target)
    except Exception:
        return _FallbackModel()

    if dump is not None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MODELS_DIR / f"{region}_rf.pkl"
        try:
            dump(model, out_path)
        except Exception:
            pass
    return model


def _choose_target(df: pd.DataFrame) -> str | None:
    candidates = [
        "model_target",
        "yield_anomaly",
        "ndvi_anomaly",
        "ndvi_zscore",
        "spi",
        "spi_30",
        "soil_surface_moisture",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    numeric_cols = df.select_dtypes(include=["number"]).columns
    return numeric_cols[0] if len(numeric_cols) else None


def _normalise_signal(pred: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(pred), dtype=float)
    if arr.size == 0:
        return arr

    mask = np.isfinite(arr)
    if not mask.any():
        return np.zeros_like(arr)

    finite = arr[mask]
    p_low, p_high = np.percentile(finite, [5, 95]) if finite.size > 1 else (finite[0], finite[0])
    if np.isclose(p_low, p_high):
        scaled = (finite - finite.mean())
        scaled = 0.5 + 0.1 * scaled
    else:
        scaled = (finite - p_low) / (p_high - p_low)
    scaled = np.clip(scaled, 0.0, 1.0)
    result = np.zeros_like(arr)
    result[mask] = scaled
    return result


class _FallbackModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        series = X.mean(axis=1)
        series = series.replace([np.inf, -np.inf], np.nan).fillna(series.mean())
        return series.to_numpy()


__all__ = ["predict_outcomes"]
diff --git a/engine/utils/__init__.py b/engine/utils/__init__.py
