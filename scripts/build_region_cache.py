import os
import time
import json
import datetime
import pandas as pd
from pathlib import Path

# Optional imports – your existing fetchers should already exist
from fetch_chirps import fetch_chirps
from fetch_openmeteo import fetch_openmeteo
from fetch_era5_recent import fetch_era5_recent

def log(msg: str):
    """Simple timestamped logger."""
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now} UTC] {msg}", flush=True)

def file_summary(path):
    """Return short size summary if file exists."""
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    return "missing"

def build_region_cache(region_name: str):
    start = time.time()
    region_path = Path("data") / region_name
    region_path.mkdir(parents=True, exist_ok=True)

    log(f"=== Starting cache build for {region_name} ===")

    # --- 1️⃣ CHIRPS ---
    log("Fetching CHIRPS rainfall data…")
    chirps_path = region_path / "chirps_cached.csv"
    fetch_chirps(region_name, output_path=chirps_path)
    log(f"✅ CHIRPS done → {file_summary(chirps_path)}")

    # --- 2️⃣ Open-Meteo ---
    log("Fetching Open-Meteo temperature/RH data…")
    meteo_path = region_path / "openmeteo_cached.csv"
    fetch_openmeteo(region_name, output_path=meteo_path)
    log(f"✅ Open-Meteo done → {file_summary(meteo_path)}")

    # --- 3️⃣ ERA5 Recent ---
    log("Fetching ERA5 short-term context…")
    era5_path = region_path / "era5_recent.csv"
    fetch_era5_recent(region_name, output_path=era5_path)
    log(f"✅ ERA5 done → {file_summary(era5_path)}")

    elapsed = time.time() - start
    log(f"🏁 Finished {region_name} in {elapsed/60:.1f} min")

    # --- 4️⃣ Write manifest for reproducibility ---
    manifest = {
        "region": region_name,
        "generated_utc": datetime.datetime.utcnow().isoformat(),
        "files": {
            "chirps_cached.csv": file_summary(chirps_path),
            "openmeteo_cached.csv": file_summary(meteo_path),
            "era5_recent.csv": file_summary(era5_path)
        },
        "runtime_minutes": round(elapsed / 60, 2)
    }
    with open(region_path / "cache_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log(f"📦 Wrote manifest → {region_path / 'cache_manifest.json'}")
    log(f"=== Cache build complete for {region_name} ===\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, help="Region folder name")
    args = parser.parse_args()
    build_region_cache(args.region)
