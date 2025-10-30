from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np

from src.regional_agent.config import get_region_current_dir
from engine.model_predict import predict_outcomes

OUTPUT_DIR = Path("outputs")


def distill_region(region: str) -> Path:
    """
    Aggregate daily anomalies into a monthly summary for *region*,
    attach Random Forest predictions, and save to outputs/<region>/distilled_summary.csv
    """
    source = get_region_current_dir(region) / "daily_anomalies.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing daily anomalies for {region}: {source}")

    df = pd.read_csv(source)
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in anomaly data.")

    # Ensure month column exists
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

    # Aggregate numeric columns monthly
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    agg = df.groupby("month")[numeric_cols].mean().reset_index()

    # Attach model predictions if available
    try:
        preds = predict_outcomes(region, agg)
        agg = pd.concat([agg, preds], axis=1)
    except Exception as e:
        print(f"⚠️ Model prediction skipped for {region}: {e}")

    out_dir = OUTPUT_DIR / region
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "distilled_summary.csv"
    agg.to_csv(out_path, index=False)
    print(f"✅ Saved distilled summary: {out_path}")
    return out_path


def write_insight_feed(region: str) -> Path:
    """
    Generate a plain-language insight feed from the distilled summary.
    Saves to outputs/<region>/insight_feed.csv
    """
    distilled_path = OUTPUT_DIR / region / "distilled_summary.csv"
    if not distilled_path.exists():
        raise FileNotFoundError(f"Distilled summary missing for {region}")

    df = pd.read_csv(distilled_path)

    # Create simple example insight text
    if "ndvi_anomaly" in df.columns and "spi" in df.columns:
        df["insight_text"] = (
            "Month " + df["month"].astype(str)
            + ": NDVI anomaly " + df["ndvi_anomaly"].round(2).astype(str)
            + ", rainfall SPI " + df["spi"].round(2).astype(str)
        )
    else:
        df["insight_text"] = "No anomaly metrics available."

    out_path = OUTPUT_DIR / region / "insight_feed.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved insight feed: {out_path}")
    return out_path
