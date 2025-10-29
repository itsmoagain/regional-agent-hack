#!/usr/bin/env python3
"""
Compute climate anomalies, short-term deltas, and derived insights for each region.

Reads:
  data/<region>/daily_merged.csv and/or monthly_merged.csv
  regions/profiles/insight.<region>.yml
  data/<region>/context_layers/{soil.csv, topography.csv, phenology_*.csv}

Writes:
  data/<region>/insights_daily.csv   (if daily available)
  data/<region>/insights_monthly.csv (otherwise)
"""

import argparse
from pathlib import Path
import shutil
import sys

try:
    from scripts.run_pipeline import require
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    from run_pipeline import require  # type: ignore

pd = require("pandas")
if pd is None:
    raise RuntimeError(
        "Pandas is required for build_region_insights. "
        "Re-run without OFFLINE_MODE to install missing dependencies."
    )

np = require("numpy")
if np is None:
    raise RuntimeError(
        "NumPy is required for build_region_insights. "
        "Re-run without OFFLINE_MODE to install missing dependencies."
    )

yaml = require("pyyaml", "yaml")
if yaml is None:
    raise RuntimeError(
        "PyYAML is required for build_region_insights. "
        "Re-run without OFFLINE_MODE to install missing dependencies."
    )

from _shared import ensure_region_workspace, load_region_profile
from datetime import datetime

try:
    from scripts.init_region import init_region
except ImportError:
    from init_region import init_region

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def compute_baseline(df, col, start_year, end_year):
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    base = df.loc[(df["year"] >= start_year) & (df["year"] <= end_year)]
    return base.groupby("month")[col].mean()


def anomaly_series(df, col, baseline):
    df = df.copy()
    df["month"] = df["date"].dt.month
    df[f"{col}_anomaly_clim"] = df[col] - df["month"].map(baseline)
    df[f"{col}_ratio_clim"] = df[col] / df["month"].map(baseline)
    return df


def rolling_anomaly(df, col, window=30):
    df = df.copy()
    roll_mean = df[col].rolling(window=window, min_periods=5, center=True).mean()
    df[f"{col}_anomaly_roll"] = df[col] - roll_mean
    return df


def zscore_series(df, col):
    mu, sigma = df[col].mean(), df[col].std()
    df[f"{col}_zscore"] = 0 if sigma == 0 or np.isnan(sigma) else (df[col] - mu) / sigma
    return df


def soil_deficit(df, col):
    q20 = df[col].quantile(0.2)
    df["soil_deficit_index"] = np.where(df[col] < q20, 1, 0)
    return df


def delta_features(df, col, lags=[5, 10]):
    """Compute short-term delta changes (difference over lag days)."""
    df = df.copy()
    for lag in lags:
        df[f"delta_{col}_{lag}d"] = df[col] - df[col].shift(lag)
    return df


def to_doy(val):
    if pd.isna(val):
        return np.nan
    try:
        f = float(val)
        if 1 <= f <= 366:
            return int(round(f))
    except Exception:
        pass
    s = str(val).strip()
    try:
        if len(s) == 5 and s[2] == "-":  # "MM-DD"
            dt = datetime(2001, int(s[:2]), int(s[3:5]))
            return int(dt.timetuple().tm_yday)
        if len(s) == 10 and s[4] == "-" and s[7] == "-":  # "YYYY-MM-DD"
            dt = datetime(int(s[:4]), int(s[5:7]), int(s[8:10]))
            return int(dt.timetuple().tm_yday)
    except Exception:
        return np.nan
    return np.nan


def merge_single_row_context(df, ctx_df, prefix=""):
    if ctx_df is None or ctx_df.empty:
        return df
    row = ctx_df.iloc[0].copy()
    for k, v in row.items():
        df[f"{prefix}{k}"] = v
    return df


def read_csv_safe(path):
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception as e:
        print(f"⚠️  Could not read {path}: {e}")
        return None


# ------------------------------------------------------------
# Main logic
# ------------------------------------------------------------

def build_region_insights(region):
    init_region(region)
    workspace = ensure_region_workspace(region)

    cfg = load_region_profile(region)
    baseline_cfg = cfg.get("baseline", {"start_year": 2010, "end_year": 2022})

    region_dir = DATA_DIR / region
    daily_file = region_dir / "daily_merged.csv"
    monthly_file = region_dir / "monthly_merged.csv"

    if daily_file.exists():
        df = pd.read_csv(daily_file, parse_dates=["date"])
        freq = "daily"
        print(f"📂 Loaded {len(df)} daily records for {region}")
    elif monthly_file.exists():
        df = pd.read_csv(monthly_file, parse_dates=["date"])
        freq = "monthly"
        print(f"📂 Loaded {len(df)} monthly records for {region}")
    else:
        print(f"❌ No merged data found for {region}.", file=sys.stderr)
        sys.exit(1)

    print(f"📅 Baseline: {baseline_cfg['start_year']}–{baseline_cfg['end_year']}")

    # ---- Temperature ----
    t_cols = [c for c in df.columns if "t2m_mean" in c.lower()]
    if t_cols:
        col = t_cols[0]
        baseline = compute_baseline(df, col, **baseline_cfg)
        df = anomaly_series(df, col, baseline)
        if freq == "daily":
            df = rolling_anomaly(df, col)
            df = delta_features(df, col)
        print(f"🌡  Added {col} anomalies, rolling, and deltas")

    # ---- Precipitation ----
    p_cols = [c for c in df.columns if "precip" in c.lower()]
    if p_cols:
        col = p_cols[0]
        baseline = compute_baseline(df, col, **baseline_cfg)
        df = anomaly_series(df, col, baseline)
        if freq == "daily":
            df = rolling_anomaly(df, col)
            df = delta_features(df, col)
        print(f"🌧  Added {col} anomalies, rolling, and deltas")

    # ---- NDVI ----
    n_cols = [c for c in df.columns if "ndvi" in c.lower()]
    if n_cols:
        col = n_cols[0]
        df = zscore_series(df, col)
        if freq == "daily":
            df = rolling_anomaly(df, col)
            df = delta_features(df, col)
        print("🌿  Added NDVI z-score, rolling, and deltas")

    # ---- Soil Moisture ----
    s_cols = [c for c in df.columns if "soil_surface" in c.lower()]
    if s_cols:
        col = s_cols[0]
        df = soil_deficit(df, col)
        if freq == "daily":
            df = rolling_anomaly(df, col)
            df = delta_features(df, col)
        print("🪱  Added soil deficit, rolling, and deltas")

    # ------------------------------------------------------------
    # Merge static context layers
    # ------------------------------------------------------------
    ctx_dir = region_dir / "context_layers"
    if ctx_dir.exists():
        soil_df = read_csv_safe(ctx_dir / "soil.csv")
        df = merge_single_row_context(df, soil_df, "soil_")

        topo_df = read_csv_safe(ctx_dir / "topography.csv")
        df = merge_single_row_context(df, topo_df, "topo_")

        phen_files = list(ctx_dir.glob("phenology_*.csv"))
        if phen_files:
            for pf in phen_files:
                crop = pf.stem.replace("phenology_", "")
                pdf = read_csv_safe(pf)
                if pdf is not None and not pdf.empty:
                    cols = {c.lower(): c for c in pdf.columns}
                    if "planting_date" in cols:
                        df[f"phen_planting_doy__{crop}"] = to_doy(pdf[cols["planting_date"]].iloc[0])
                    if "flowering_date" in cols:
                        df[f"phen_flowering_doy__{crop}"] = to_doy(pdf[cols["flowering_date"]].iloc[0])
                    if "harvest_date" in cols:
                        df[f"phen_harvest_doy__{crop}"] = to_doy(pdf[cols["harvest_date"]].iloc[0])
            print(f"🌾 Merged phenology DOY columns for {len(phen_files)} crop(s).")
        else:
            print("🌾 No phenology_*.csv files found; skipping phenology merge.")
    else:
        print("ℹ️  No context_layers directory; skipping context merge.")

    # ---- Output ----
    out_file = region_dir / f"insights_{freq}.csv"
    df.to_csv(out_file, index=False)
    print(f"✅ Wrote enriched insights → {out_file}")

    workspace_out = workspace / "insights" / out_file.name
    workspace_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_file, workspace_out)
    print(f"🗂️  Synced insights to workspace → {workspace_out.relative_to(workspace)}")


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute climate anomalies and insights for a region (Phase A enriched).")
    p.add_argument("--region", required=True)
    args = p.parse_args()
    build_region_insights(args.region)
