#!/usr/bin/env python3
"""
Build RAG (Regional Aggregation Grid) context layers for a region.

Reads:
  - config/insight.<region>.yml  (to get bbox, crops, and metadata)
  - data/<region>/ (for caching)

Creates:
  - data/<region>/context_layers/phenology.csv
  - data/<region>/context_layers/soil.csv
  - data/<region>/context_layers/topography.csv
  - data/<region>/rags.csv (summary)

Each layer is lightweight and cached — ready for use in feature building.
"""

import argparse
import pandas as pd
import numpy as np
import yaml
import requests
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def load_region_meta(region):
    cfg_path = CONFIG_DIR / f"insight.{region}.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No YAML config found for region {region}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("region_meta", {})


def save_layer(region, name, df):
    out_dir = DATA_DIR / region / "context_layers"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {name} → {out_path.name} ({len(df)} rows)")
    return out_path


# -------------------------------------------------
# 1️⃣ Phenology Layer — from NASA POWER or CropCalendar
# -------------------------------------------------
def fetch_phenology(region_meta):
    crops = region_meta.get("crops", ["unknown"])
    bbox = region_meta.get("bbox")
    if bbox and None not in bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
    else:
        print("⚠️  Missing bbox; using placeholder coordinates for phenology.")
        min_lon, min_lat, max_lon, max_lat = -77.0, 18.0, -76.5, 18.5

    # Placeholder logic — later can query NASA POWER CropCalendar API
    data = []
    for crop in crops:
        data.append({
            "crop": crop,
            "planting_month": np.random.choice(range(1, 13)),
            "harvest_month": np.random.choice(range(1, 13)),
            "source": "placeholder",
            "lat_center": (min_lat + max_lat) / 2,
            "lon_center": (min_lon + max_lon) / 2
        })
    return pd.DataFrame(data)


# -------------------------------------------------
# 2️⃣ Soil Structure Layer — from SoilGrids or OpenLandMap
# -------------------------------------------------
def fetch_soil(region_meta):
    bbox = region_meta.get("bbox")
    if bbox and None not in bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
    else:
        print("⚠️  Missing bbox; using placeholder coordinates for soil layer.")
        min_lon, min_lat, max_lon, max_lat = -77.0, 18.0, -76.5, 18.5

    # Placeholder synthetic soil values
    soil_props = [
        {"var": "sand", "mean_pct": np.random.uniform(30, 60)},
        {"var": "clay", "mean_pct": np.random.uniform(10, 40)},
        {"var": "organic_carbon", "g_kg": np.random.uniform(5, 20)},
    ]
    return pd.DataFrame(soil_props)


# -------------------------------------------------
# 3️⃣ Topography Layer — from SRTM / OpenTopography
# -------------------------------------------------
def fetch_topography(region_meta):
    bbox = region_meta.get("bbox")
    if bbox and None not in bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
    else:
        print("⚠️  Missing bbox; using placeholder coordinates for topography.")
        min_lon, min_lat, max_lon, max_lat = -77.0, 18.0, -76.5, 18.5

    # Placeholder values; later fetch via API
    elev_mean = np.random.uniform(100, 1200)
    slope_mean = np.random.uniform(2, 25)
    aspect = np.random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    df = pd.DataFrame([{"elevation_m": elev_mean, "slope_deg": slope_mean, "dominant_aspect": aspect}])
    return df


# -------------------------------------------------
# 4️⃣ Summarize into rags.csv
# -------------------------------------------------
def build_rags_summary(region):
    context_dir = DATA_DIR / region / "context_layers"
    rag_path = DATA_DIR / region / "rags.csv"

    layers = {}
    for layer_name in ["phenology", "soil", "topography"]:
        layer_file = context_dir / f"{layer_name}.csv"
        if layer_file.exists():
            layers[layer_name] = pd.read_csv(layer_file)
        else:
            print(f"⚠️ Missing layer: {layer_name}.csv")

    # Merge summaries into flat table
    summary = {
        "region": region,
        "timestamp": datetime.utcnow().isoformat(),
        "n_crops": len(layers.get("phenology", [])),
        "soil_mean_sand": layers.get("soil", pd.DataFrame()).get("mean_pct", pd.Series([None]))[0],
        "elevation_mean": layers.get("topography", pd.DataFrame()).get("elevation_m", pd.Series([None]))[0],
    }
    df = pd.DataFrame([summary])
    df.to_csv(rag_path, index=False)
    print(f"✅ Wrote summary RAG → {rag_path}")
    return df


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------
def build_rag_context(region: str):
    region_meta = load_region_meta(region)
    print(f"🌍 Building RAG context layers for {region}")
    print(f"🗺️  BBOX: {region_meta.get('bbox')} | Crops: {region_meta.get('crops')}")

    pheno_df = fetch_phenology(region_meta)
    soil_df = fetch_soil(region_meta)
    topo_df = fetch_topography(region_meta)

    save_layer(region, "phenology", pheno_df)
    save_layer(region, "soil", soil_df)
    save_layer(region, "topography", topo_df)

    build_rags_summary(region)
    print(f"🎉 Context layers built successfully for {region}")


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG (context) layers for a region.")
    parser.add_argument("--region", required=True, help="Region name, e.g. hungary_farmland")
    args = parser.parse_args()
    build_rag_context(args.region)
