#!/usr/bin/env python3
"""
Compute rolling z-score anomalies for climate and vegetation variables.

Usage:
  python scripts/compute_anomalies.py --region austin_farmland --window 30

This script:
  • Loads merged daily dataset (data/<region>/daily_merged.csv)
  • Computes rolling mean and std for each numeric variable
  • Calculates z-score anomalies ((x - mean) / std)
  • Saves enriched output → data/<region>/daily_anomalies.csv
  • Updates metadata.json
  • Optionally renders NDVI anomaly preview chart
"""

import argparse
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Helper: Compute rolling z-scores
# ------------------------------------------------------------
def compute_rolling_anomalies(df, window=30, min_periods=10):
    df = df.copy()
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in "if" and c not in ["year", "month", "day"]]

    print(f"🧮 Found {len(numeric_cols)} numeric columns for anomaly computation.")
    print(f"Using {window}-day rolling window...")

    anomalies = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        roll_mean = df[col].rolling(window=window, min_periods=min_periods).mean()
        roll_std = df[col].rolling(window=window, min_periods=min_periods).std()
        anomalies[f"{col}_anomaly"] = (df[col] - roll_mean) / roll_std

    return anomalies


# ------------------------------------------------------------
# Helper: Update metadata.json
# ------------------------------------------------------------
def update_metadata(region_dir, summary_dict):
    meta_path = region_dir / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except json.JSONDecodeError:
            meta = {}
    else:
        meta = {}

    meta["anomaly_computation"] = summary_dict
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"🧾 Metadata updated → {meta_path}")


# ------------------------------------------------------------
# Helper: Visualization
# ------------------------------------------------------------
def plot_ndvi_anomalies(df, region_name):
    if "ndvi_anomaly" not in df.columns:
        print("⚠️ No NDVI anomaly column found — skipping visualization.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["ndvi_anomaly"], color="green", linewidth=1.2)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title(f"NDVI Anomalies – {region_name}")
    plt.xlabel("Date")
    plt.ylabel("Z-Score")
    plt.tight_layout()
    out_path = Path(f"data/{region_name}/ndvi_anomaly_preview.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📊 NDVI anomaly plot saved → {out_path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute rolling anomalies for a region.")
    parser.add_argument("--region", required=True, help="Region name (e.g. austin_farmland)")
    parser.add_argument("--window", type=int, default=30, help="Rolling window size (days)")
    args = parser.parse_args()

    region_dir = Path("data") / args.region
    input_path = region_dir / "daily_merged.csv"
    output_path = region_dir / "daily_anomalies.csv"

    if not input_path.exists():
        print(f"❌ Missing dataset: {input_path}")
        sys.exit(1)

    print(f"📂 Loading data → {input_path}")
    df = pd.read_csv(input_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    anomalies = compute_rolling_anomalies(df, window=args.window)
    df_out = pd.concat([df, anomalies], axis=1)
    df_out.to_csv(output_path, index=False)
    print(f"✅ Saved enriched dataset → {output_path} ({len(df_out)} rows)")

    # Summarize
    summary = {
        "window_days": args.window,
        "columns_processed": len(anomalies.columns),
        "date_range": [str(df['date'].min().date()), str(df['date'].max().date())],
    }
    update_metadata(region_dir, summary)

    # Visualization
    try:
        plot_ndvi_anomalies(df_out, args.region)
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")

    print("🎉 Anomaly computation complete.")


if __name__ == "__main__":
    main()
