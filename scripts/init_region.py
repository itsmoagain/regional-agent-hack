#!/usr/bin/env python3
"""
Auto-initialize a region's folder structure and config YAML.

Creates:
  - data/<region>/flags/
  - data/<region>/plots/
  - data/<region>/context_layers/
  - regions/profiles/insight.<region>.yml (if missing)

Now includes:
  ✅ Full rolling/climatology variable structure
  ✅ Phase A rule dictionary for interpretive logic
  ✅ BBOX + crop list + metadata hooks for context layers
  ✅ Automatic context layer generation (soil, elevation, phenology)
"""
import os, sys, subprocess
import sys

# Ensure critical dependencies before doing anything
required = ["pyyaml", "pandas", "requests"]
missing = []
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"📦 Installing missing dependencies: {', '.join(missing)} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
else:
    print("✅ Environment ready — all dependencies found.")
import yaml
from pathlib import Path
import subprocess

from _shared import (
    get_region_cache_dir,
    get_region_current_dir,
    get_region_data_root,
)

# -------------------------
# Default Config Template
# -------------------------
DEFAULTS = {
    "extends": "insight.defaults.yml",
    "baseline": {"start_year": 2010, "end_year": 2022},
    "rolling": {"window_days": 30, "min_periods": 5},
    "windows": {"spi_short": 30, "spi_long": 90, "plot_context_days": 60},
    # 🌍 Core region metadata for later enrichment
    "region_meta": {
        "bbox": [None, None, None, None],  # [min_lon, min_lat, max_lon, max_lat]
        "crops": ["maize"],                # list of crop names
        "country": "unknown",
        "notes": "placeholder — auto-filled by user or agent",
    },
    # 🌾 Variable registry
    "variables": {
        "temperature": {"column": "t2m_mean", "enabled": True},
        "precipitation": {"column": "precip_mm_sum", "enabled": True},
        "vegetation": {"column": "ndvi", "enabled": True},
        "soil_moisture": {"column": "soil_surface_moisture", "enabled": True},
    },
    "outputs": {
        "daily": True,
        "monthly": True,
        "include_climatology": True,
        "include_rolling": True
    },
    # -------------------------------------------------
    # 🌍 Phase A interpretive rule dictionary
    # -------------------------------------------------
    "rules": [
        {
            "id": "{region}_irrigation_recovery",
            "label": "{REGION}: Possible irrigation recovery event",
            "when": {
                "all": [
                    {"var": "Δndvi_10d", "op": ">", "value": 0.1},
                    {"var": "Δsoil_surface_moisture_10d", "op": ">", "value": 0.02},
                    {"var": "precip_mm_sum_anomaly_roll", "op": "<", "value": 0},
                ]
            },
            "note": "NDVI and soil moisture increase without rainfall → possible irrigation response.",
        },
        {
            "id": "{region}_heat_stress",
            "label": "{REGION}: Vegetation heat stress",
            "when": {
                "all": [
                    {"var": "t2m_mean_anomaly_roll", "op": ">", "value": 2.0},
                    {"var": "Δndvi_5d", "op": "<", "value": -0.05},
                ]
            },
            "note": "Rapid NDVI decline with strong short-term heat anomaly.",
        },
        {
            "id": "{region}_drought_onset",
            "label": "{REGION}: Early drought signal",
            "when": {
                "all": [
                    {"var": "precip_mm_sum_anomaly_clim", "op": "<", "value": -0.5},
                    {"var": "soil_deficit_index", "op": "==", "value": 1},
                    {"var": "ndvi_zscore", "op": "<", "value": -1.0},
                ]
            },
            "note": "Low precipitation, dry soils, and suppressed vegetation.",
        },
        {
            "id": "{region}_moisture_rebound",
            "label": "{REGION}: Moisture rebound following rain",
            "when": {
                "all": [
                    {"var": "precip_mm_sum_anomaly_roll", "op": ">", "value": 0.3},
                    {"var": "Δsoil_surface_moisture_5d", "op": ">", "value": 0.02},
                ]
            },
            "note": "Soil moisture recovery after short-term rainfall anomaly.",
        },
        {
            "id": "{region}_persistent_stress",
            "label": "{REGION}: Persistent vegetation stress",
            "when": {
                "all": [
                    {"var": "ndvi_zscore", "op": "<", "value": -1.0},
                    {"var": "t2m_mean_anomaly_clim", "op": ">", "value": 1.5},
                    {"var": "precip_mm_sum_anomaly_clim", "op": "<", "value": -0.5},
                ]
            },
            "note": "Combined long-term heat and dryness driving vegetation stress.",
        },
        {
            "id": "{region}_rapid_growth_phase",
            "label": "{REGION}: Rapid vegetative growth",
            "when": {
                "all": [
                    {"var": "Δndvi_10d", "op": ">", "value": 0.1},
                    {"var": "ndvi_zscore", "op": ">", "value": 0.5},
                ]
            },
            "note": "Strong NDVI increase suggests active growth or green-up.",
        },
        {
            "id": "{region}_harvest_signal",
            "label": "{REGION}: Potential harvest or senescence",
            "when": {
                "all": [
                    {"var": "Δndvi_10d", "op": "<", "value": -0.1},
                    {"var": "EVI2", "op": "<", "value": 0.4},
                ]
            },
            "note": "Vegetation decline with low canopy greenness (possible harvest).",
        },
    ],
}


# -------------------------------------------------
# Region initialization logic
# -------------------------------------------------
def init_region(region_name: str, bbox=None, crops=None, country=None):
    """Create necessary folders, YAML config, and context layers for a given region."""

    config_dir = Path("regions") / "profiles"
    config_dir.mkdir(parents=True, exist_ok=True)  # ✅ ensure profile directory exists

    cfg_path = config_dir / f"insight.{region_name}.yml"
    data_root = get_region_data_root(region_name)
    get_region_cache_dir(region_name)
    get_region_current_dir(region_name)

    # Create directories
    for sub in ["flags", "plots"]:
        (data_root / sub).mkdir(parents=True, exist_ok=True)
    (data_root / "context_layers").mkdir(parents=True, exist_ok=True)

    # If YAML exists, don’t overwrite
    if cfg_path.exists():
        print(f"⚙️  Config already exists for {region_name} — skipping creation.")
    else:
        # Generate new config
        region_id = region_name.split("_")[0][:2].lower()
        config_content = yaml.safe_load(yaml.dump(DEFAULTS))

        # Apply dynamic metadata
        if bbox:
            config_content["region_meta"]["bbox"] = bbox
        if crops:
            config_content["region_meta"]["crops"] = crops
        if country:
            config_content["region_meta"]["country"] = country

        # Replace placeholders dynamically
        for rule in config_content.get("rules", []):
            rule["id"] = rule["id"].format(region=region_id)
            rule["label"] = rule["label"].format(REGION=region_name.upper())

        # Write to YAML
        cfg_path.write_text(yaml.safe_dump(config_content, sort_keys=False))
        print(f"✅ Created {cfg_path}")

    print(f"✅ Initialized data/{region_name}/flags, plots, context_layers, caches, and current directories")
    print(f"🗺️  Region meta saved in {cfg_path}")

    # -------------------------------------------------
    # Trigger automatic context layer creation
    # -------------------------------------------------
    print(f"🌎 Fetching soil, elevation, and phenology context for {region_name}...")
    try:
        subprocess.run(
            [sys.executable, "scripts/build_context_layers.py", "--region", region_name],
            check=True,
        )
        print("✅ Context layers generated successfully.")
    except Exception as e:
        print(f"⚠️  Skipped context layer generation: {e}")


# -------------------------------------------------
# Entrypoint
# -------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Initialize region structure and config.")
    p.add_argument("--region", required=True)
    p.add_argument("--bbox", nargs=4, type=float, help="Bounding box: min_lon min_lat max_lon max_lat")
    p.add_argument("--crops", nargs="+", help="List of primary crops (space-separated)")
    p.add_argument("--country", help="Country name for region metadata")
    a = p.parse_args()

    bbox = a.bbox if a.bbox else None
    crops = a.crops if a.crops else None
    init_region(a.region, bbox=bbox, crops=crops, country=a.country)
