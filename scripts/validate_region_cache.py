#!/usr/bin/env python3
"""
Validate region cache outputs — check for file presence, column alignment,
date continuity, and missing values.

Usage:
  python scripts/validate_region_cache.py --region hungary_farmland
"""

import argparse
from pathlib import Path

import pandas as pd

from _shared import get_region_current_dir

def validate_region_cache(region: str):
    base = get_region_current_dir(region)
    daily_path = base / "daily_merged.csv"
    monthly_path = base / "monthly_merged.csv"

    print(f"🔍 Validating cache for region: {region}")

    # --- Check existence
    if not daily_path.exists():
        print("❌ daily_merged.csv not found.")
        return
    if not monthly_path.exists():
        print("❌ monthly_merged.csv not found.")
        return

    # --- Load data
    daily = pd.read_csv(daily_path, parse_dates=["date"])
    monthly = pd.read_csv(monthly_path, parse_dates=["date"])

    print(f"📄 Daily columns:   {list(daily.columns)}")
    print(f"📄 Monthly columns: {list(monthly.columns)}")

    # --- Basic checks
    print(f"📅 Daily range:   {daily['date'].min().date()} → {daily['date'].max().date()} ({len(daily)} days)")
    print(f"📅 Monthly range: {monthly['date'].min().date()} → {monthly['date'].max().date()} ({len(monthly)} months)")

    # --- Missing values summary
    print("\n🕳 Missing values (daily):")
    print(daily.isna().sum()[daily.isna().sum() > 0].to_string())

    # --- Check date continuity
    daily_diff = daily["date"].diff().dropna().dt.days
    if daily_diff.max() > 1:
        print(f"⚠️ Gaps detected in daily timeline: max gap = {daily_diff.max()} days")
    else:
        print("✅ No missing days in daily timeline")

    # --- Quick correlation sanity check (if enough data)
    numeric_cols = [c for c in daily.columns if c != "date"]
    if len(numeric_cols) > 1:
        corr = daily[numeric_cols].corr().round(2)
        print("\n📈 Variable correlation snapshot:")
        print(corr)
    else:
        print("ℹ️ Not enough numeric columns for correlation matrix")

    print("\n✅ Validation complete.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Validate daily/monthly merged cache for a region.")
    p.add_argument("--region", required=True)
    args = p.parse_args()
    validate_region_cache(args.region)
