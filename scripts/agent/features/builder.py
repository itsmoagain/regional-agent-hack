#!/usr/bin/env python3
"""
Feature builder for Regional Climate Agent models.

Inputs (auto-detected if present under data/<region>/current/):
  - insight_monthly.csv         ← harmonized base
  - phenology.csv               ← phenology stage data (optional)
  - practices_logs.csv          ← management logs (optional)
  - rags.csv                    ← spatial/soil cluster metadata (optional)
  - context_layers/*.csv        ← topography / soil / remote sensing (optional)

Outputs:
  - feature DataFrame (X)
  - target series (y)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from _shared import get_region_current_dir


# ------------------------------------------------------------
# Tier 1 — core climate / vegetation / soil features
# ------------------------------------------------------------
def add_tier1_features(df: pd.DataFrame):
    df = df.copy()

    # Month/season encodings
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Rolling climate features
    for col in [c for c in df.columns if "precip" in c.lower()]:
        df[f"{col}_rolling30"] = df[col].rolling(3, min_periods=1).sum()
        df[f"{col}_rolling90"] = df[col].rolling(9, min_periods=1).sum()

    for col in [c for c in df.columns if "t2m" in c.lower()]:
        df[f"{col}_rolling30"] = df[col].rolling(3, min_periods=1).mean()
        df[f"{col}_rolling90"] = df[col].rolling(9, min_periods=1).mean()

    # NDVI lags
    if "ndvi_anomaly" in df.columns:
        df["ndvi_anomaly_lag1"] = df["ndvi_anomaly"].shift(1)
        df["ndvi_anomaly_lag2"] = df["ndvi_anomaly"].shift(2)

    # Soil moisture lags
    soil_cols = [c for c in df.columns if "soil" in c.lower()]
    for col in soil_cols:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)

    df.dropna(inplace=True)
    return df


# ------------------------------------------------------------
# Tier 2 — phenology & practice log features
# ------------------------------------------------------------
def add_phenology_features(df: pd.DataFrame, region_path: Path):
    phen_file = region_path / "phenology.csv"
    if not phen_file.exists():
        return df

    phen = pd.read_csv(phen_file)
    df["month"] = df["date"].dt.month
    df = df.merge(phen, how="left", on="month")  # simple merge; can refine later

    # One-hot encode crop phase
    if "phase" in df.columns:
        df = pd.get_dummies(df, columns=["phase"], prefix="phase")
    return df


def add_practice_features(df: pd.DataFrame, region_path: Path):
    logs_file = region_path / "practices_logs.csv"
    if not logs_file.exists():
        return df

    logs = pd.read_csv(logs_file, parse_dates=["date"])
    # Example: irrigation flag and fertilizer intensity in 60-day window
    for idx, row in df.iterrows():
        window_start = row["date"] - pd.Timedelta(days=60)
        window_logs = logs[(logs["date"] >= window_start) & (logs["date"] <= row["date"])]

        df.loc[idx, "irrigation_flag_60d"] = int(
            (window_logs["action"] == "irrigation").any()
        )
        df.loc[idx, "fertilizer_events_60d"] = int(
            (window_logs["action"] == "fertilization").sum()
        )
    return df


# ------------------------------------------------------------
# Tier 3 — contextual / RAG / environmental features
# ------------------------------------------------------------
def add_context_features(df: pd.DataFrame, region_path: Path):
    # Merge RAG metadata if available
    rag_file = region_path / "rags.csv"
    if rag_file.exists():
        rags = pd.read_csv(rag_file)
        df = df.merge(rags, how="left", on="rag_id", validate="m:1")

    # Merge any files under context_layers/
    context_dir = region_path / "context_layers"
    if context_dir.exists():
        for file in context_dir.glob("*.csv"):
            ctx = pd.read_csv(file)
            df = df.merge(ctx, how="left", on="rag_id", suffixes=("", f"_{file.stem}"))

    # One-hot encode soil_type etc. if present
    cat_cols = [c for c in df.columns if c in ["soil_type", "elevation_band"]]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)
    return df


# ------------------------------------------------------------
# Main builder function
# ------------------------------------------------------------
def build_features(region: str, tier: int = 1, insight_file: Path | str | None = None,
                   target: str | None = None):
    """Assemble feature matrix/target vector for a region.

    Parameters
    ----------
    region: str
        Region key matching ``data/<region>``.
    tier: int
        Feature tier depth (1=core, 2=+phenology/practices, 3=+context).
    insight_file: Optional path
        Explicit insight CSV override. Defaults to ``insights_monthly.csv``.
    target: Optional[str]
        Name of the target column. Falls back to ``ndvi_anomaly`` or
        ``ndvi_zscore`` if present.
    """

    region_path = get_region_current_dir(region)
    if insight_file is None:
        insight_file = region_path / "insights_monthly.csv"
        if not insight_file.exists():
            # Backwards compatibility with older naming convention
            legacy = region_path / "insight_monthly.csv"
            if legacy.exists():
                insight_file = legacy
            else:
                raise FileNotFoundError(
                    f"No insights file found for region {region} (looked for {insight_file})"
                )
    insight_file = Path(insight_file)

    df = pd.read_csv(insight_file, parse_dates=["date"])
    df = add_tier1_features(df)

    if tier >= 2:
        df = add_phenology_features(df, region_path)
        df = add_practice_features(df, region_path)

    if tier >= 3:
        df = add_context_features(df, region_path)

    # Define target (example: NDVI anomaly)
    candidate_targets = [t for t in [target, "ndvi_anomaly", "ndvi_zscore"] if t]
    target_col = next((t for t in candidate_targets if t in df.columns), None)

    if target_col:
        y = df[target_col].shift(-1).dropna()
        X = df.iloc[:-1].drop(columns=["date", target_col])
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
    else:
        X, y = df.drop(columns=["date"]).reset_index(drop=True), None

    return X, y
