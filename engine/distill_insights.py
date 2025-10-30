#!/usr/bin/env python3
"""Aggregate daily anomaly signals into monthly insight features."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping

import pandas as pd
import numpy as np

from regional_agent.config import get_region_current_dir, get_region_data_root
from .model_predict import predict_outcomes

OUTPUT_DIR = Path("outputs")


# -------------------------------------------------------
# 🗣️ Plain-Language Insight Generator
# -------------------------------------------------------

def generate_plain_language(row):
    """
    Translate numeric anomaly and model signals into interpretable text.
    Used in the hackathon demo to show human-readable insights.
    """

    msgs = []
    precip = row.get("precip_anom", np.nan)
    ndvi = row.get("ndvi_anomaly", np.nan)
    spi = row.get("spi", np.nan)
    soil = row.get("soil_surface_moisture", np.nan)
    rf_signal = row.get("model_signal", np.nan)

    # 🌧️ Rainfall anomalies
    if pd.notna(spi) and spi < -1.0:
        msgs.append("⚠️ Prolonged rainfall deficit and negative SPI indicate early-season drought stress.")
    elif pd.notna(precip) and precip < -1:
        msgs.append("🌤️ Rainfall below normal — reduced soil recharge expected.")
    elif pd.notna(precip) and precip > 1:
        msgs.append("🌦️ Above-average rainfall — possible waterlogging or field access delays.")
    else:
        msgs.append("🌤️ Rainfall near seasonal norms.")

    # 🌿 Vegetation response
    if pd.notna(ndvi) and ndvi < -1:
        msgs.append("🪴 NDVI drop suggests vegetation stress or delayed growth.")
    elif pd.notna(ndvi) and ndvi > 1:
        msgs.append("🌱 Vegetation vigor above average — canopy growth strong.")
    else:
        msgs.append("🌾 Vegetation conditions appear typical for this period.")

    # 🪱 Soil moisture
    if pd.notna(soil) and soil < -1:
        msgs.append("🧱 Soil moisture deficit may limit root uptake.")
    elif pd.notna(soil) and soil > 1:
        msgs.append("💧 Soil moisture above average — irrigation needs reduced.")

    # 🤖 Model signal
    if pd.notna(rf_signal) and rf_signal > 0.8:
        msgs.append("🤖 Model predicts high probability of crop stress.")
    elif pd.notna(rf_signal) and 0.4 < rf_signal <= 0.8:
        msgs.append("🤖 Model indicates moderate variability in crop response.")
    else:
        msgs.append("🤖 Model indicates stable growth conditions.")

    return " ".join(msgs).strip()


def attach_insight_text(df: pd.DataFrame) -> pd.DataFrame:
    """Attach plain-language text summaries for each monthly record."""
    if "insight_text" not in df.columns:
        df["insight_text"] = df.apply(generate_plain_language, axis=1)
    return df


# -------------------------------------------------------
# 🌎 Region Distillation
# -------------------------------------------------------

def distill_region(region: str) -> str:
    """Distil daily anomalies for *region* into a monthly summary table."""

    source = _resolve_input_path(region)
    df = pd.read_csv(source)
    df = _prepare_dataframe(df)

    if "month" not in df.columns:
        raise ValueError("Distillation requires a 'month' column derived from dates.")

    metric_map = _select_metric_columns(df)
    agg_map: Dict[str, str] = {col: "mean" for col in metric_map.values() if col}
    rename_map = {v: k for k, v in metric_map.items() if v}
    if not agg_map:
        numeric_cols = [col for col in df.select_dtypes(include=["number"]).columns if col != "month"]
        agg_map = {col: "mean" for col in numeric_cols}
        rename_map = {}
    if not agg_map:
        raise ValueError("No numeric columns available for distillation.")

    grouped = df.groupby("month", as_index=False).agg(agg_map).rename(columns=rename_map).sort_values("month")

    # Run model predictions
    grouped["model_signal"] = predict_outcomes(region, grouped)

    # Attach readable insights
    grouped = attach_insight_text(grouped)

    # Save outputs
    output_dir = OUTPUT_DIR / region
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "distilled_summary.csv"
    grouped.to_csv(output_path, index=False)
    return str(output_path)


# -------------------------------------------------------
# 🔍 Helpers
# -------------------------------------------------------

def _resolve_input_path(region: str) -> Path:
    candidates = [
        get_region_current_dir(region) / "daily_anomalies.csv",
        get_region_current_dir(region) / "anomalies.csv",
        get_region_data_root(region) / "anomalies.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No anomaly dataset found for '{region}'. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_column = None
    for candidate in ("date", "timestamp", "day"):
        if candidate in df.columns:
            date_column = candidate
            break
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])
        df["month"] = df[date_column].dt.to_period("M").astype(str)
    elif "month" in df.columns:
        df["month"] = df["month"].astype(str)
    return df


def _select_metric_columns(df: pd.DataFrame) -> Mapping[str, str | None]:
    lookup = {col.lower(): col for col in df.columns}

    def pick(*candidates: str, contains: Iterable[str] | None = None) -> str | None:
        contains = contains or []
        for candidate in candidates:
            if candidate in lookup:
                return lookup[candidate]
        for column in df.columns:
            name = column.lower()
            if all(token in name for token in contains):
                return column
        return None

    return {
        "spi": pick("spi", "spi_30", contains=("spi",)),
        "precip_anom": pick("precip_anom", "precipitation_anomaly", contains=("precip", "anom")),
        "ndvi_anomaly": pick("ndvi_anomaly", "ndvi_zscore", contains=("ndvi", "anom")),
        "soil_surface_moisture": pick(
            "soil_surface_moisture",
            "soil_moisture",
            contains=("soil", "moist"),
        ),
        "temp_mean": pick(
            "temp_mean",
            "t2m_mean",
            "temperature_mean",
            contains=("temp", "mean"),
        ),
    }


__all__ = ["distill_region"]

