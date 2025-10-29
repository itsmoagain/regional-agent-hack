#!/usr/bin/env python3
"""
Build region cache by merging all fetcher outputs (CHIRPS, SMAP, NDVI, ERA5)
into unified daily and monthly CSVs, with mixed-granularity alignment.

Outputs (written to ``data/<region>/current``):
- daily_merged.csv
- monthly_merged.csv
- metadata.json
"""

import argparse
import subprocess
from pathlib import Path
import json
import sys
from pathlib import Path

import pandas as pd

from _shared import (
    get_region_current_dir,
    load_region_profile,
)

try:
    from scripts.run_pipeline import require
except ModuleNotFoundError:  # pragma: no cover - fallback when run directly
    from run_pipeline import require  # type: ignore

pd = require("pandas")
if pd is None:
    raise RuntimeError(
        "Pandas is required for build_region_cache. "
        "Re-run without OFFLINE_MODE to install missing dependencies."
    )

# ------------------------------------------------------------
# Helper: load CSVs
# ------------------------------------------------------------
def load_csv(path: Path, name: str) -> pd.DataFrame:
    print(f"📥 Loading {name} from {path}")
    return pd.read_csv(path, parse_dates=["date"])

# ------------------------------------------------------------
# Helper: load bbox from region YAML
# ------------------------------------------------------------
def load_bbox_from_yaml(region: str):
    """Load bounding box from the region profile."""
    cfg = load_region_profile(region)
    bbox = cfg.get("region_meta", {}).get("bbox") or []
    if not bbox or len(bbox) != 4:
        raise ValueError(f"Invalid bbox in config: {bbox}")
    return bbox

# ------------------------------------------------------------
# Harmonize to daily grid (mixed-granularity)
# ------------------------------------------------------------
def harmonize_to_daily(dfs: dict) -> tuple[pd.DataFrame, dict]:
    """Align mixed-frequency variables to common daily grid while preserving fidelity."""
    start = max(df["date"].min() for df in dfs.values())
    end = min(df["date"].max() for df in dfs.values())
    daily_index = pd.date_range(start, end, freq="D")

    out_df = pd.DataFrame({"date": daily_index})
    provenance = {}

    for var, df in dfs.items():
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        df = df.set_index("date").sort_index()
        col = [c for c in df.columns if var in c or var.replace("_mean", "") in c]
        col = col[0] if col else df.columns[-1]
        df = df[[col]].rename(columns={col: var})
        df = df.reindex(daily_index)

        if var.startswith(("precip", "t2m", "soil_")):
            df[var] = df[var].interpolate(limit=3, limit_direction="both")
            provenance[var] = {"fill_method": "linear", "max_gap_days": 3}
        elif "ndvi" in var.lower():
            df[var] = df[var].ffill().bfill()
            provenance[var] = {"fill_method": "ffill", "note": "held constant between observations"}
        else:
            provenance[var] = {"fill_method": "none"}

        out_df = out_df.merge(df[var], left_on="date", right_index=True, how="left")
        print(f"🔗 Merged {var} → {len(df)} rows")

    return out_df.reset_index(drop=True), provenance

# ------------------------------------------------------------
# Main build
# ------------------------------------------------------------
def build_region_cache(region: str, base_dir: Path | None = None):
    base = base_dir or get_region_current_dir(region)
    if not base.exists():
        raise FileNotFoundError(f"Region folder not found: {base}")

    valid_files = {
        "precip_mm_sum": "chirps_gee.csv",
        "soil_surface_moisture": "soil_gee.csv",
        "soil_rootzone_moisture": "soil_gee.csv",
        "ndvi": "ndvi_gee.csv",
        "t2m_mean": "openmeteo.csv",
        "t2m_max": "openmeteo.csv",
        "t2m_min": "openmeteo.csv",
    }

    dfs = {}
    for var, fname in valid_files.items():
        fp = base / fname
        if fp.exists():
            dfs[var] = load_csv(fp, fname)
        else:
            print(f"⚠️ Missing fetcher file for {var}: {fname}")

    # ------------------------------------------------------------
    # Auto-fetch missing datasets if needed
    # ------------------------------------------------------------
    if not dfs:
        print(f"⚠️ No valid fetcher files found for region {region}. Attempting to auto-fetch...")
        bbox = load_bbox_from_yaml(region)
        script_root = Path(__file__).resolve().parent
        required = {
            "chirps_gee.csv": script_root / "fetch_chirps_gee.py",
            "soil_gee.csv": script_root / "fetch_soil_gee.py",
            "ndvi_gee.csv": script_root / "fetch_ndvi_gee.py",
            "openmeteo.csv": script_root / "fetch_openmeteo.py",
        }

        for file, fetcher in required.items():
            target = base / file
            if not target.exists() and fetcher.exists():
                print(f"🌍 Fetching missing dataset: {file}")
                try:
                    subprocess.run(
                        [
                            sys.executable, str(fetcher),
                            "--bbox", *map(str, bbox),
                            "--out", str(target)
                        ],
                        check=True
                    )
                except Exception as e:
                    print(f"⚠️ Auto-fetch for {file} failed: {e}")

        # Try reloading after fetch
        for var, fname in valid_files.items():
            fp = base / fname
            if fp.exists():
                dfs[var] = load_csv(fp, fname)

        if not dfs:
            raise RuntimeError(f"No valid fetcher files found for region {region} after auto-fetch attempts")

    # ------------------------------------------------------------
    # Continue normal processing
    # ------------------------------------------------------------
    out_df, provenance = harmonize_to_daily(dfs)

    daily_out = base / "daily_merged.csv"
    out_df.to_csv(daily_out, index=False)
    print(f"✅ Saved daily merged → {daily_out.name} ({len(out_df)} rows)")

    out_df["year"], out_df["month"] = out_df["date"].dt.year, out_df["date"].dt.month
    agg_map = {c: ("sum" if c.startswith("precip") else "mean")
               for c in out_df.columns if c not in ["date", "year", "month"]}
    monthly = out_df.groupby(["year", "month"]).agg(agg_map).reset_index()
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    monthly = monthly[["date"] + [c for c in monthly.columns if c not in ("year", "month", "date")]]
    monthly_out = base / "monthly_merged.csv"
    monthly.to_csv(monthly_out, index=False)
    print(f"✅ Saved monthly merged → {monthly_out.name} ({len(monthly)} rows)")

    meta = {
        "region": region,
        "records_daily": len(out_df),
        "records_monthly": len(monthly),
        "frequency": "daily",
        "provenance": provenance,
    }
    with open(base / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"🧭 Metadata saved → metadata.json")
    print(f"🎉 Build complete for region {region}.")

# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build region cache from fetcher outputs.")
    parser.add_argument("--region", required=True)
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Override output directory (defaults to data/<region>/current)",
    )
    args = parser.parse_args()
    build_region_cache(args.region, base_dir=args.base_dir)
