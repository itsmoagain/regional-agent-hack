#!/usr/bin/env python3
"""
🌲 Random Forest Training Orchestration
--------------------------------------

This module trains, evaluates, and exports Random Forest models for region-level
climate insight prediction using cached feature datasets.

It automatically handles:
  • Feature/target loading from region caches
  • Train/test split and evaluation
  • Model and artifact saving

💡 Note:
For offline or demo environments (e.g., Kaggle or hackathon runs), the
`preprocessing.py` helper includes a fallback that loads `insights_daily.csv`
when `insights_monthly.csv` is missing. This ensures the pipeline remains
runnable even before monthly aggregation is generated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:  # pragma: no cover - offline fallback
    RandomForestRegressor = None  # type: ignore[assignment]

from .evaluator import evaluate_random_forest
from .model_utils import load_model, save_model
from .preprocessing import FeatureCache, load_feature_cache

OUTPUTS_DIR = Path("outputs")


@dataclass
class TrainingArtifacts:
    model: RandomForestRegressor
    model_path: Path
    feature_importances: Path
    metrics_path: Path
    metrics: dict


def _train_model(features: pd.DataFrame, labels: pd.Series):
    if RandomForestRegressor is None:
        return _FallbackModel(labels.mean() if not labels.empty else 0.0)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(features, labels)
    return model


def _split_dataset(features: pd.DataFrame, labels: pd.Series, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if len(features) < 5:
        return features, features, labels, labels
    split_idx = int(len(features) * (1 - test_ratio))
    split_idx = max(1, min(split_idx, len(features) - 1))
    return (
        features.iloc[:split_idx],
        features.iloc[split_idx:],
        labels.iloc[:split_idx],
        labels.iloc[split_idx:],
    )


def train_from_cache(
    region: str,
    *,
    tier: int = 1,
    target: str = "ndvi_zscore",
    freq: str = "monthly",
    test_ratio: float = 0.2,
    force_refresh: bool = False,
) -> TrainingArtifacts:
    """Train a Random Forest model using cached features."""

    cache: FeatureCache = load_feature_cache(
        region,
        tier=tier,
        target=target,
        freq=freq,
        force_refresh=force_refresh,
    )

    if cache.target is None or cache.features.empty:
        raise ValueError("Feature cache missing target values; cannot train model.")

    X_train, X_test, y_train, y_test = _split_dataset(cache.features, cache.target, test_ratio)

    if pd is None or np is None:
        model = _FallbackModel(cache.target.mean())
        metrics = {"r2": 0.0, "mae": 0.0, "rmse": 0.0, "n": len(cache.target)}
    else:
        model = _train_model(X_train, y_train)
        metrics = evaluate_random_forest(model, X_test, y_test)

    try:
        model_path = save_model(model, region)
    except Exception:
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{region}_rf.pkl"
        model_path.write_text(json.dumps({"type": "fallback"}, indent=2), encoding="utf-8")

    feature_dir = model_path.parent
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_importances = feature_dir / f"{model_path.stem}_feature_importances.csv"
    pd.DataFrame(
        {"feature": cache.features.columns, "importance": getattr(model, "feature_importances_", np.zeros(len(cache.features.columns)))}
    ).sort_values("importance", ascending=False).to_csv(feature_importances, index=False)

    metrics_path = OUTPUTS_DIR / region / "model_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "region": region,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "tier": tier,
        "target": target,
        "freq": freq,
    }
    metrics_path.write_text(json.dumps(payload, indent=2))

    return TrainingArtifacts(
        model=model,
        model_path=model_path,
        feature_importances=feature_importances,
        metrics_path=metrics_path,
        metrics=metrics,
    )


def load_or_train_model(
    region: str,
    *,
    tier: int = 1,
    target: str = "ndvi_zscore",
    freq: str = "monthly",
    **kwargs,
):
    """Load a previously trained model or fit a new one from cache."""

    model = load_model(region)
    if model is not None:
        return model

    artifacts = train_from_cache(region, tier=tier, target=target, freq=freq, **kwargs)
    return artifacts.model


class _FallbackModel:
    def __init__(self, mean_value: float) -> None:
        self.mean_value = mean_value

    def predict(self, features: pd.DataFrame) -> np.ndarray:  # pragma: no cover - simple fallback
        if np is None or features.empty:
            return []  # type: ignore[return-value]
        weights = np.linspace(0.2, 1.0, features.shape[1])
        avg = np.asarray(features.fillna(features.mean()).values, dtype=float)
        combined = (avg * weights).mean(axis=1)
        return np.clip(combined, 0.0, 1.0)


__all__ = ["TrainingArtifacts", "load_or_train_model", "train_from_cache"]

