#!/usr/bin/env python3
"""
Build static context layers for a given region.

Reads from:
  regions/profiles/insight.<region>.yml → to get bbox + crop list

Fetches:
  - 🌾 Crop phenology (via Open-Meteo temperature/GDD)
  - 🪱 Soil properties (SoilGrids v2.0)
  - 🏔️ Elevation (Open-Elevation API)
  - 🔁 Dynamic layers via scripts/fetch_all.py (CHIRPS, NDVI, SMAP, OpenMeteo)

Outputs:
  data/<region>/context_layers/
    soil.csv, soil_metadata.json
    topography.csv, topography_metadata.json
    phenology_<crop>.csv, phenology_<crop>_metadata.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from subprocess import run
import sys

try:  # Local import to avoid circular dependency when used as a module
    from scripts.run_pipeline import require
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    from run_pipeline import require  # type: ignore

np = require("numpy")
if np is None:
    raise RuntimeError(
        "NumPy is required for build_context_layers. "
        "Re-run without OFFLINE_MODE to install missing dependencies."
    )

pd = require("pandas")
if pd is None:
    raise RuntimeError(
        "Pandas is required for build_context_layers. "
        "Re-run without OFFLINE_MODE to install missing dependencies."
    )

requests = require("requests")
if requests is None:
    raise RuntimeError(
        "Requests is required for build_context_layers. "
        "Re-run without OFFLINE_MODE to install missing dependencies."
    )

yaml = require("pyyaml", "yaml")
if yaml is None:
    raise RuntimeError(
        "PyYAML is required for build_context_layers. "
        "Re-run without OFFLINE_MODE to install missing dependencies."
    )

from _shared import load_region_profile, resolve_region_config_path


# -------------------------------------------------------
# 🌾 Phenology Helper (Open-Meteo → GDD)
# -------------------------------------------------------
def fetch_openmeteo_phenology(lat: float, lon: float, crop: str, out_dir: Path) -> pd.DataFrame:
    """
    Derive simple phenology proxy from Open-Meteo ERA5 reanalysis.
    Uses temperature-based GDD accumulation to estimate planting,
    flowering, and harvest DOY thresholds.
    """

    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2010-01-01",
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "timezone": "auto",
    }

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lat": lat,
        "lon": lon,
        "crop": crop,
        "source": None,
    }

    # Crop-specific base temps (°C) for GDD calculation
    base_temp_map = {
        "maize": 10, "corn": 10,
        "soybean": 8, "wheat": 0,
        "rice": 10, "coffee": 12,
        "cotton": 15, "sorghum": 8,
        "vegetables": 8, "tomato": 10,
        "generic_crop": 10
    }
    base_temp = base_temp_map.get(crop.lower(), 10)

    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        temps = pd.DataFrame({
            "date": data["daily"]["time"],
            "tmax": data["daily"]["temperature_2m_max"],
            "tmin": data["daily"]["temperature_2m_min"],
        })
        temps["tavg"] = (temps["tmax"] + temps["tmin"]) / 2
        temps["gdd"] = (temps["tavg"] - base_temp).clip(lower=0)
        temps["cumsum_gdd"] = temps["gdd"].cumsum()

        # Approx phenology thresholds (% of cumulative GDD)
        total_gdd = temps["cumsum_gdd"].iloc[-1]
        thresholds = {
            "planting": 0.05 * total_gdd,
            "flowering": 0.55 * total_gdd,
            "harvest": 0.95 * total_gdd,
        }

        phenology = {}
        for key, gdd_threshold in thresholds.items():
            subset = temps[temps["cumsum_gdd"] >= gdd_threshold]
            phenology[f"{key}_date"] = (
                subset["date"].iloc[0] if not subset.empty else np.nan
            )

        df = pd.DataFrame([{
            "crop": crop,
            "planting_date": phenology.get("planting_date"),
            "flowering_date": phenology.get("flowering_date"),
            "harvest_date": phenology.get("harvest_date"),
            "lat": lat,
            "lon": lon,
            "data_source": "OpenMeteo_GDD",
        }])
        meta["source"] = "OpenMeteo_GDD"

    except Exception as e:
        print(f"⚠️ Phenology fallback for {crop}: {e}")
        df = pd.DataFrame([{
            "crop": crop,
            "planting_date": np.nan,
            "flowering_date": np.nan,
            "harvest_date": np.nan,
            "lat": lat,
            "lon": lon,
            "data_source": "Fallback_None",
        }])
        meta["source"] = "Fallback_None"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"phenology_{crop}.csv"
    df.to_csv(out_file, index=False)
    print(f"✅ Saved {out_file.name} ({len(df)} rows, source={meta['source']})")

    meta_path = out_dir / f"phenology_{crop}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return df


# -------------------------------------------------------
# 🪱 SoilGrids v2.0 & Elevation Helpers
# -------------------------------------------------------
def fetch_soilgrids(lat: float, lon: float, out_path: Path) -> pd.DataFrame:
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lon": lon,
        "lat": lat,
        "property": ["clay", "silt", "sand", "soc", "phh2o", "bdod"],
        "depth": ["0-5cm", "5-15cm", "15-30cm"],
    }

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lat": lat,
        "lon": lon,
        "source": None,
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        props = data.get("properties", {}).get("layers", {})
        if isinstance(props, dict):
            layer_items = props.items()
        elif isinstance(props, list):
            layer_items = [(layer.get("name"), layer) for layer in props if isinstance(layer, dict)]
        else:
            raise ValueError("Unrecognized SoilGrids format")

        records = []
        for name, layer in layer_items:
            depths = layer.get("depths", [])
            values = [
                d.get("values", {}).get("mean")
                for d in depths
                if d.get("values", {}).get("mean") is not None
            ]
            if values:
                records.append({"property": name, "mean_value": sum(values) / len(values)})

        if not records:
            raise ValueError("No valid mean values extracted from SoilGrids response.")

        df = pd.DataFrame(records)
        df = df.pivot_table(values="mean_value", columns="property", aggfunc="first").reset_index(drop=True)
        df.rename(columns={
            "clay": "clay_pct", "silt": "silt_pct", "sand": "sand_pct",
            "soc": "soc_gkg", "phh2o": "ph", "bdod": "bulk_density",
        }, inplace=True)

        df["data_source"] = "SoilGrids_API"
        meta["source"] = "SoilGrids_API"

    except Exception as e:
        print(f"⚠️ SoilGrids error: {e}")
        df = pd.DataFrame([{
            "clay_pct": 25, "silt_pct": 35, "sand_pct": 40, "soc_gkg": 15,
            "ph": 6.5, "bulk_density": 1.3, "data_source": "Fallback_FAO_Defaults",
        }])
        meta["source"] = "Fallback_FAO_Defaults"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    with open(out_path.with_name("soil_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved {out_path.name} ({len(df)} rows, {df.shape[1]} cols, source={meta['source']})")
    return df


def fetch_elevation(lat: float, lon: float, out_dir: Path) -> pd.DataFrame:
    url = "https://api.open-elevation.com/api/v1/lookup"
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lat": lat,
        "lon": lon,
        "source": None,
    }

    try:
        r = requests.post(url, json={"locations": [{"latitude": lat, "longitude": lon}]}, timeout=30)
        r.raise_for_status()
        elev = r.json()["results"][0]["elevation"]
        df = pd.DataFrame([{"elevation_m": elev, "data_source": "Open_Elevation_API"}])
        meta["source"] = "Open_Elevation_API"
    except Exception as e:
        print(f"⚠️ Elevation fallback: {e}")
        df = pd.DataFrame([{"elevation_m": np.nan, "data_source": "Fallback_None"}])
        meta["source"] = "Fallback_None"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "topography.csv"
    df.to_csv(out_csv, index=False)
    with open(out_dir / "topography_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Saved {out_csv.name} ({len(df)} rows, source={meta['source']})")
    return df


# -------------------------------------------------------
# 🧩 Context Layer Builder
# -------------------------------------------------------
def build_context_layers(region_name: str):
    root = Path(__file__).resolve().parents[1]
    cfg_path = resolve_region_config_path(region_name)
    region_dir = root / "data" / region_name
    ctx_dir = region_dir / "context_layers"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_region_profile(region_name)
    bbox = cfg.get("region_meta", {}).get("bbox", [None, None, None, None])
    crops = cfg.get("region_meta", {}).get("crops", ["generic_crop"])

    if not bbox or len(bbox) != 4 or any(v is None for v in bbox):
        print("⚠️ Invalid or missing bbox; using (0,0).")
        lat, lon = 0.0, 0.0
    else:
        lat = float(np.mean([bbox[1], bbox[3]]))
        lon = float(np.mean([bbox[0], bbox[2]]))

    print(f"🌍 Building context layers for {region_name}")
    print(f"📍 Approx centroid: lat={lat:.3f}, lon={lon:.3f}")
    print("🪱 Fetching SoilGrids data...")
    fetch_soilgrids(lat, lon, ctx_dir / "soil.csv")

    print("🏔️ Fetching elevation data...")
    fetch_elevation(lat, lon, ctx_dir)

    for crop in crops:
        print(f"🌾 Fetching phenology for {crop}...")
        fetch_openmeteo_phenology(lat, lon, crop, ctx_dir)

    # 🔁 Run dynamic fetchers
    fetch_all = root / "scripts" / "fetch_all.py"
    if fetch_all.exists():
        print("🔁 Running fetch_all.py for dynamic datasets...")
        run([sys.executable, str(fetch_all), "--region", region_name, "--mode", "active"], check=False)

    cache_builder = root / "scripts" / "build_region_cache.py"
    if cache_builder.exists():
        print("📦 Building full regional cache...")
        run([sys.executable, str(cache_builder), "--region", region_name], check=False)

    print(f"🎉 Context layers built successfully for {region_name}")


# -------------------------------------------------------
# Entrypoint
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build static context layers for a region.")
    parser.add_argument("--region", required=True, help="Region name (e.g. hungary_farmland)")
    parser.add_argument("--crop", nargs="+", help="Optional crop list to override YAML (e.g. --crop maize wheat)")
    args = parser.parse_args()

    if args.crop:
        cfg_path = resolve_region_config_path(args.region)
        cfg = yaml.safe_load(open(cfg_path))
        cfg.setdefault("region_meta", {})["crops"] = args.crop
        yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
        print(f"🌾 Overrode crops in profile → {args.crop}")

    build_context_layers(args.region)
