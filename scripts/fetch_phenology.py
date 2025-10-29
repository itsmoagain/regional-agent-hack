#!/usr/bin/env python3
"""
Fetch daily agroclimatic data from NASA POWER API to build phenology context layers.

These layers provide the environmental drivers (temperature, rainfall, radiation, humidity, etc.) 
that underpin phenological development for each crop in a region.

Includes automatic bioclimatic zone detection (tropical, arid, temperate, boreal)
to adjust variable emphasis regionally.

Output: data/<region>/context_layers/phenology_<crop>.csv
"""

import argparse
from pathlib import Path
import json
from datetime import datetime

try:
    from scripts.run_pipeline import require
except ModuleNotFoundError:  # pragma: no cover - fallback when executed directly
    from run_pipeline import require  # type: ignore

requests = require("requests")
if requests is None:
    raise RuntimeError(
        "Requests is required for fetch_phenology. "
        "Re-run without OFFLINE_MODE to install missing dependencies."
    )

pd = require("pandas")
if pd is None:
    raise RuntimeError(
        "Pandas is required for fetch_phenology. "
        "Re-run without OFFLINE_MODE to install missing dependencies."
    )

yaml = require("pyyaml", "yaml")
if yaml is None:
    raise RuntimeError(
        "PyYAML is required for fetch_phenology. "
        "Re-run without OFFLINE_MODE to install missing dependencies."
    )

from _shared import load_region_profile
from config.climate_zone_lookup import get_climate_zone

from config.crop_variable_map import (
    CROP_VARIABLE_MAP,
    FALLBACK_VARIABLES,
    classify_bioclimate,
    apply_regional_overrides,
)

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


# ------------------------------------------------------------
# Helper: Fetch data from NASA POWER
# ------------------------------------------------------------
def fetch_power_data(lat, lon, variables, start, end):
    params = {
        "latitude": lat,
        "longitude": lon,
        "parameters": ",".join(variables),
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
        "community": "AG",
    }
    r = requests.get(NASA_POWER_URL, params=params, timeout=30)
    if r.status_code == 200:
        data = r.json().get("properties", {}).get("parameter", {})
        if not data:
            raise ValueError("Empty NASA POWER response.")
        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df.index)
        df.reset_index(inplace=True)
        df.rename(columns={"index": "date"}, inplace=True)
        return df
    else:
        raise RuntimeError(f"NASA POWER request failed ({r.status_code}): {r.text[:200]}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(region: str):
    # Load region config
    cfg = load_region_profile(region)
    bbox = cfg["region_meta"]["bbox"]
    crops = cfg["region_meta"]["crops"]
    country = cfg["region_meta"].get("country", "unknown")

    # Determine center point for API calls
    lat = (bbox[1] + bbox[3]) / 2
    lon = (bbox[0] + bbox[2]) / 2

    # Detect climate zone via Köppen–Geiger lookup
    climate_info = get_climate_zone(lat, lon)
    bioclimate = climate_info["zone"]
    koppen_code = climate_info["koppen"]
    if koppen_code:
        print(f"🌍 Detected climate zone: {bioclimate} (Köppen {koppen_code})")
    else:
        print(f"🌍 Estimated climate zone: {bioclimate} (latitude-based fallback)")

    start = datetime(2019, 1, 1)
    end = datetime.today()

    out_dir = Path("data") / region / "context_layers"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for crop in crops:
        crop_key = crop.lower().strip()

        # Determine variable set and apply regional overrides
        base_vars = CROP_VARIABLE_MAP.get(crop_key, FALLBACK_VARIABLES)
        vars_for_crop = apply_regional_overrides(base_vars, bioclimate)

        print(f"🌾 Fetching NASA POWER for {crop} ({','.join(vars_for_crop)}) ...")
        try:
            df = fetch_power_data(lat, lon, vars_for_crop, start, end)
            out_csv = out_dir / f"phenology_{crop_key}.csv"
            df.to_csv(out_csv, index=False)
            print(f"✅ Saved {len(df)} records → {out_csv.name}")
            summary[crop_key] = {
                "variables": vars_for_crop,
                "records": len(df),
                "bioclimate": bioclimate,
                "koppen_code": koppen_code,
                "status": "ok",
            }

        except Exception as e:
            print(f"⚠️ Fetch failed for {crop_key}: {e}")
            # Retry with fallback
            try:
                print(f"↩️ Retrying {crop_key} with fallback variables ...")
                df = fetch_power_data(lat, lon, FALLBACK_VARIABLES, start, end)
                out_csv = out_dir / f"phenology_{crop_key}.csv"
                df.to_csv(out_csv, index=False)
                print(f"✅ Fallback succeeded → {out_csv.name}")
                summary[crop_key] = {
                    "variables": FALLBACK_VARIABLES,
                    "records": len(df),
                    "bioclimate": bioclimate,
                    "koppen_code": koppen_code,
                    "status": "fallback",
                }
            except Exception as e2:
                print(f"❌ Fallback failed for {crop_key}: {e2}")
                summary[crop_key] = {
                    "variables": FALLBACK_VARIABLES,
                    "bioclimate": bioclimate,
                    "koppen_code": koppen_code,
                    "status": "failed",
                }

    # Write metadata
    meta_path = out_dir / "phenology_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"🧾 Metadata saved → {meta_path}")


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fetch NASA POWER agroclimatic data for phenology context.")
    p.add_argument("--region", required=True, help="Region name matching regions/profiles/insight.<region>.yml")
    args = p.parse_args()
    main(args.region)
