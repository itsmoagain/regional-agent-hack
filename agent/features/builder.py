"""Feature builder for Random Forest training and inference."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from scripts import _shared as scripts_shared  # pragma: no cover - optional path helper
except ImportError:  # pragma: no cover - package usage
    scripts_shared = None

if scripts_shared is not None:
    get_region_current_dir = scripts_shared.get_region_current_dir
else:
    from src.regional_agent.config import get_region_current_dir  # type: ignore  # pragma: no cover


# ------------------------------------------------------------
# Tier 1 — core climate / vegetation / soil features
# ------------------------------------------------------------
def add_tier1_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])

    if "date" in df.columns:
        df["month"] = df["date"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    for col in [c for c in df.columns if "precip" in c.lower()]:
        df[f"{col}_rolling30"] = df[col].rolling(3, min_periods=1).sum()
        df[f"{col}_rolling90"] = df[col].rolling(9, min_periods=1).sum()

    for col in [c for c in df.columns if "t2m" in c.lower() or "temp" in c.lower()]:
        df[f"{col}_rolling30"] = df[col].rolling(3, min_periods=1).mean()
        df[f"{col}_rolling90"] = df[col].rolling(9, min_periods=1).mean()

    if "ndvi_anomaly" in df.columns:
        df["ndvi_anomaly_lag1"] = df["ndvi_anomaly"].shift(1)
        df["ndvi_anomaly_lag2"] = df["ndvi_anomaly"].shift(2)

    soil_cols = [c for c in df.columns if "soil" in c.lower()]
    for col in soil_cols:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)

    df.dropna(inplace=True)
    return df


# ------------------------------------------------------------
# Tier 2 — phenology & practice log features
# ------------------------------------------------------------
def add_phenology_features(df: pd.DataFrame, region_path: Path) -> pd.DataFrame:
    phen_file = region_path / "phenology.csv"
    if not phen_file.exists():
        return df

    phen = pd.read_csv(phen_file)
    if "date" in df.columns and "month" not in df.columns:
        df["month"] = df["date"].dt.month
    df = df.merge(phen, how="left", on="month")

    if "phase" in df.columns:
        df = pd.get_dummies(df, columns=["phase"], prefix="phase")
    return df


def add_practice_features(df: pd.DataFrame, region_path: Path) -> pd.DataFrame:
    logs_file = region_path / "practices_logs.csv"
    if not logs_file.exists():
        return df

    logs = pd.read_csv(logs_file, parse_dates=["date"])
    if "date" not in df.columns:
        return df

    df = df.copy()
    for idx, row in df.iterrows():
        window_start = row["date"] - pd.Timedelta(days=60)
        window_logs = logs[(logs["date"] >= window_start) & (logs["date"] <= row["date"])]
        df.loc[idx, "irrigation_flag_60d"] = int((window_logs["action"] == "irrigation").any())
        df.loc[idx, "fertilizer_events_60d"] = int((window_logs["action"] == "fertilization").sum())
    return df


# ------------------------------------------------------------
# Tier 3 — contextual / RAG / environmental features
# ------------------------------------------------------------
def add_context_features(df: pd.DataFrame, region_path: Path) -> pd.DataFrame:
    df = df.copy()

    rag_file = region_path / "rags.csv"
    if rag_file.exists():
        rags = pd.read_csv(rag_file)
        if "rag_id" in df.columns and "rag_id" in rags.columns:
            df = df.merge(rags, how="left", on="rag_id", validate="m:1")

    context_dir = region_path / "context_layers"
    if context_dir.exists():
        for file in context_dir.glob("*.csv"):
            ctx = pd.read_csv(file)
            join_cols = [c for c in ("rag_id", "grid_id", "zone") if c in df.columns and c in ctx.columns]
            if join_cols:
                df = df.merge(ctx, how="left", on=join_cols, suffixes=("", f"_{file.stem}"))

    cat_cols = [c for c in df.columns if c in {"soil_type", "elevation_band", "phase"}]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)
    return df


# ------------------------------------------------------------
# Main builder function
# ------------------------------------------------------------
def build_features(
    region: str,
    tier: int = 1,
    insight_file: Path | str | None = None,
    target: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series | None]:
    """Assemble feature matrix/target vector for a region."""

    region_path = get_region_current_dir(region)
    if insight_file is None:
        candidates = [
            region_path / "insights_monthly.csv",
            region_path / "insight_monthly.csv",
            region_path / "distilled_summary.csv",
        ]
        for cand in candidates:
            if cand.exists():
                insight_file = cand
                break
        if insight_file is None:
            raise FileNotFoundError(
                f"No insights file found for region {region}. Checked: "
                + ", ".join(str(c) for c in candidates)
            )

    insight_path = Path(insight_file)
    df = pd.read_csv(insight_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = add_tier1_features(df)

    if tier >= 2:
        df = add_phenology_features(df, region_path)
        df = add_practice_features(df, region_path)

    if tier >= 3:
        df = add_context_features(df, region_path)

    candidate_targets = [t for t in [target, "ndvi_anomaly", "ndvi_zscore"] if t]
    target_col = next((t for t in candidate_targets if t in df.columns), None)

    if target_col:
        y = df[target_col].shift(-1).dropna()
        X = df.iloc[:-1].drop(columns=["date", target_col], errors="ignore")
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        return X, y

    X = df.drop(columns=["date"], errors="ignore").reset_index(drop=True)
    return X, None


__all__ = ["build_features"]
