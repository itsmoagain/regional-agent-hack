from __future__ import annotations

#!/usr/bin/env python3
"""
Fetch all datasets for a region (CHIRPS, SMAP, NDVI, Open-Meteo).

- Ensures Earth Engine client is installed & authenticated BEFORE running GEE fetchers.
- Reads BBOX primarily from regions/profiles/insight.<region>.yml, falls back to internal map.
- Uses current venv's Python (sys.executable).

Usage:
  python scripts/fetch_all.py --region hungary_farmland --mode active
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import os

try:
    from scripts.run_pipeline import require
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    from run_pipeline import require  # type: ignore

yaml = require("pyyaml", "yaml")
if yaml is None:
    raise RuntimeError(
        "PyYAML is required for fetch_all. "
        "Re-run without OFFLINE_MODE to install missing dependencies."
    )

from _shared import load_region_profile

# ---------------------------------------------------------------------
# 🔧 Path safety — allows running this file directly OR via other scripts
# ---------------------------------------------------------------------
sys.path.append(os.path.dirname(__file__))                      # scripts/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))     # project root

try:
    from gee_setup import ensure_gee_ready
except ModuleNotFoundError:
    from scripts.gee_setup import ensure_gee_ready

# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SCRIPTS = ROOT / "scripts"

REGION_BBOXES = {
    "hungary_farmland": [16.0, 45.5, 23.0, 48.6],
    "jamaica_coffee":   [-78.2, 17.7, -76.0, 18.5],
    "austin_farmland":  [-97.708351, 30.16754, -97.447254, 30.28264],
}

# 🔁 Dynamic data fetchers (no NASA POWER)
FETCHERS = {
    "chirps": "fetch_chirps_gee.py",
    "soil":   "fetch_soil_gee.py",
    "ndvi":   "fetch_ndvi_gee.py",
    "temp":   "fetch_openmeteo.py",  # Open-Meteo replaces NASA POWER
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _read_bbox_from_yaml(region: str) -> Optional[List[float]]:
    """Read bounding box from region profile if available."""
    try:
        cfg = load_region_profile(region)
    except FileNotFoundError:
        return None
    bbox = cfg.get("region_meta", {}).get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(b) for b in bbox]
    return None


def run_fetch(script_name: str, bbox: List[float], out_path: Path, mode: str) -> bool:
    """Run a single fetcher script safely using the same Python executable."""
    script_path = SCRIPTS / script_name
    if not script_path.exists():
        print(f"⚠️  Missing script: {script_name}")
        return False

    if mode == "cached" and out_path.exists():
        print(f"⏭️  Cached dataset already present → {out_path.name}")
        return True

    cmd = [
        sys.executable,
        str(script_path),
        "--bbox", *map(str, bbox),
        "--mode", mode,
        "--out", str(out_path),
    ]

    print(f"\n🚀 Running {' '.join(cmd)}")
    start = time.time()
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {script_name} completed in {time.time() - start:.1f}s.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed ({e}) after {time.time() - start:.1f}s.")
        return False


# ---------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------
def main(region: str, mode: str, ee_project: Optional[str]) -> None:
    print(f"\n🌍 Fetching all datasets for region: {region}")
    print(f"📦 Mode: {mode}")

    bbox = _read_bbox_from_yaml(region) or REGION_BBOXES.get(region)
    if not bbox:
        print(f"❌ No BBOX found for '{region}'. Add it to regions/profiles/insight.{region}.yml or REGION_BBOXES.")
        sys.exit(1)
    print(f"🗺  Bounding Box: {bbox}")

    region_dir = DATA_DIR / region
    region_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Ensure Earth Engine is ready before GEE fetchers
    ee_ready = ensure_gee_ready(project=ee_project)

    # Step 2: Run each fetcher
    for key, script_name in FETCHERS.items():
        out_file = region_dir / (f"{key}_gee.csv" if key != "temp" else "openmeteo.csv")

        # Skip GEE-dependent fetchers if EE not ready
        if key in ("chirps", "soil", "ndvi") and not ee_ready:
            print(f"⚠️ Skipping {key} (Earth Engine not initialized).")
            continue

        success = run_fetch(script_name, bbox, out_file, mode)
        if not success:
            print(f"⚠️  {key} fetch failed — continuing.\n")

    # Step 3: Wrap-up summary
    print("\n🎉 All fetchers completed (with above status).")
    print(f"🕒 Finished at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"📁 Outputs in: {region_dir.resolve()}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run all fetchers for a region.")
    p.add_argument("--region", required=True, help="Region name (e.g. hungary_farmland)")
    p.add_argument("--mode", default="active", choices=["active", "cached"], help="Fetch mode")
    p.add_argument("--ee-project", default=None, help="Optional GCP project for EE Initialize()")
    args = p.parse_args()
    main(args.region, args.mode, args.ee_project)
