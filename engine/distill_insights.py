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
    Generate readable climate insights from the distilled summary.
    """
    distilled_path = OUTPUT_DIR / region / "distilled_summary.csv"
    if not distilled_path.exists():
        raise FileNotFoundError(f"Distilled summary missing for {region}")

    df = pd.read_csv(distilled_path)

    insights = []
    for _, row in df.iterrows():
        spi = row.get("spi", np.nan)
        ndvi = row.get("ndvi_anomaly", np.nan)
        temp = row.get("temp_mean", np.nan)
        crop = row.get("crop_type", "crop")
        region_name = row.get("region_name", region.replace("_", " ").title())
        month = row.get("month", "unknown")

        # Simple narrative logic
        parts = [f"{region_name} ({crop}) — {month}:"]
        if not np.isnan(spi):
            if spi < -1:
                parts.append("dry conditions (below-normal rainfall)")
            elif spi > 1:
                parts.append("wet conditions (above-normal rainfall)")
            else:
                parts.append("near-normal rainfall")

        if not np.isnan(ndvi):
            if ndvi < -0.1:
                parts.append("vegetation stress observed")
            elif ndvi > 0.1:
                parts.append("healthy vegetation growth")
            else:
                parts.append("stable vegetation levels")

        if not np.isnan(temp):
            if temp > 28:
                parts.append("high temperatures may increase evapotranspiration.")
            elif temp < 15:
                parts.append("cooler-than-average temperatures noted.")
            else:
                parts.append("temperatures within typical range.")

        insights.append(" ".join(parts))

    df["insight_text"] = insights
    out_path = OUTPUT_DIR / region / "insight_feed.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved insight feed: {out_path}")
    return out_path

