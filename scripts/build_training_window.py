#!/usr/bin/env python3
"""
build_training_window.py — create training-ready dataset for a region

Usage:
    python scripts/build_training_window.py --region austin_farmland

Generates:
    data/<region>/current/training_window.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime

from _shared import get_region_current_dir

def build_training_window(region: str, base_dir: Path | None = None):
    base_path = base_dir or get_region_current_dir(region)
    merged = base_path / "daily_merged.csv"
    out_path = base_path / "training_window.csv"

    if not merged.exists():
        raise FileNotFoundError(f"Missing merged data file: {merged}")

    print(f"📦 Loading merged dataset → {merged}")
    df = pd.read_csv(merged, parse_dates=["date"])

    # --- Basic sanity ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    print(f"🧩 Found {len(numeric_cols)} numeric columns: {numeric_cols[:8]}...")

    # --- Sort chronologically ---
    df = df.sort_values("date").reset_index(drop=True)

    # --- Define predictors (X) ---
    predictors = [
        c for c in df.columns
        if any(k in c for k in [
            "precip", "t2m_mean", "t2m_max", "t2m_min",
            "soil_surface_moisture", "soil_rootzone_moisture"
        ])
    ]

    # --- Define target (y): NDVI next-period value or anomaly ---
    if "ndvi" in df.columns:
        df["ndvi_next"] = df["ndvi"].shift(-7)   # 7-day forward
        df["ndvi_delta"] = df["ndvi_next"] - df["ndvi"]
        df["ndvi_anomaly"] = (df["ndvi"] - df["ndvi"].rolling(30, min_periods=5).mean()) / (
            df["ndvi"].rolling(30, min_periods=5).std() + 1e-6
        )
        target_col = "ndvi_delta"
    else:
        raise KeyError("No NDVI column found in merged dataset — cannot build target.")

    # --- Drop early NaNs ---
    df = df.dropna(subset=[target_col])

    # --- Feature lags (to capture temporal memory) ---
    for col in predictors:
        df[f"{col}_lag7"] = df[col].shift(7)
        df[f"{col}_lag14"] = df[col].shift(14)

    # --- Final feature selection ---
    features = [c for c in df.columns if any(k in c for k in predictors)] + [
        f"{col}_lag7" for col in predictors
    ] + [f"{col}_lag14" for col in predictors]

    df = df.dropna(subset=features + [target_col]).reset_index(drop=True)

    # --- Save training window ---
    df_out = df[["date"] + features + [target_col, "ndvi", "ndvi_anomaly"]]
    df_out.to_csv(out_path, index=False)
    print(f"✅ Training window saved → {out_path} ({len(df_out)} rows, {len(df_out.columns)} cols)")

    # --- Preview ---
    print(df_out.head(5).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Build training window for region.")
    parser.add_argument("--region", required=True, help="Region name, e.g. austin_farmland")
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Override output directory (defaults to data/<region>/current)",
    )
    args = parser.parse_args()
    build_training_window(args.region, base_dir=args.base_dir)

if __name__ == "__main__":
    main()
