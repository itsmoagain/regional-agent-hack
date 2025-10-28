#!/usr/bin/env python3
"""
Train Random Forest models for a given region, tier, and data frequency.

Usage:
  python scripts/train_region_model.py --region austin_farmland --tier 2 --freq daily --target ndvi_zscore

Inputs:
  - data/<region>/insights_<freq>.csv   (daily or monthly)
  - Optional: phenology.csv, practice_logs.csv, rags.csv, context_layers/
Outputs (created under models/<region>/):
  - tier{tier}_{freq}_model.pkl
  - tier{tier}_{freq}_feature_importances.csv
  - tier{tier}_{freq}_metrics.json
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from agent.features.builder import build_features

# ------------------------------------------------------------
# Helper: compute evaluation metrics
# ------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """NaN-safe metrics computation."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"r2": np.nan, "mae": np.nan, "rmse": np.nan, "n": 0}

    y_true, y_pred = y_true[mask], y_pred[mask]
    return {
        "r2": round(r2_score(y_true, y_pred), 3),
        "mae": round(mean_absolute_error(y_true, y_pred), 3),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "n": int(mask.sum()),
    }


# ------------------------------------------------------------
# Main training function
# ------------------------------------------------------------
def train_region_model(region: str, tier: int = 1, target: str = "ndvi_zscore", freq: str = "monthly"):
    region_path = Path("data") / region
    model_dir = Path("models") / region
    model_dir.mkdir(parents=True, exist_ok=True)

    # Determine which insights file to use
    insight_file = region_path / f"insights_{freq}.csv"
    if not insight_file.exists():
        fallback = region_path / "insights_monthly.csv"
        if fallback.exists():
            insight_file = fallback
            freq = "monthly"
            print(f"⚠️  No insights_{freq}.csv found; using monthly fallback.")
        else:
            raise FileNotFoundError(f"No insight file found for {region} ({freq})")

    print(f"🏗  Building Tier {tier} features for {region} ({freq} data)...")
    X, y = build_features(region, tier, insight_file=insight_file, target=target)
    if y is None or X.empty:
        raise ValueError("No valid target or features found — cannot train model.")

    print(f"📦 Feature matrix: {X.shape[0]} rows × {X.shape[1]} columns")

    # Temporal 80/20 split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"🧩 Split: {len(X_train)} train / {len(X_test)} test")

    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    # Save artifacts
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    prefix = f"tier{tier}_{freq}"
    model_file = model_dir / f"{prefix}_model.pkl"
    feat_file = model_dir / f"{prefix}_feature_importances.csv"
    metrics_file = model_dir / f"{prefix}_metrics.json"

    joblib.dump(model, model_file)

    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    importances.to_csv(feat_file, index=False)

    with open(metrics_file, "w") as f:
        json.dump({
            "region": region,
            "tier": tier,
            "freq": freq,
            "timestamp": timestamp,
            "records": len(X),
            "metrics": metrics
        }, f, indent=2)

    print(f"✅ Model trained and saved for {region} (Tier {tier}, {freq})")
    print(f"📈 Metrics: R²={metrics['r2']}  MAE={metrics['mae']}  RMSE={metrics['rmse']}  (n={metrics['n']})")
    print(f"🧠 Model → {model_file}")
    print(f"📊 Feature importances → {feat_file}")
    print(f"📘 Metrics → {metrics_file}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train a region-specific Random Forest model.")
    p.add_argument("--region", required=True, help="Region name, e.g. hungary_farmland or austin_farmland")
    p.add_argument("--tier", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--target", default="ndvi_zscore", help="Target variable to predict")
    p.add_argument("--freq", choices=["daily", "monthly"], default="monthly", help="Data frequency to use")
    args = p.parse_args()

    train_region_model(args.region, args.tier, args.target, args.freq)
