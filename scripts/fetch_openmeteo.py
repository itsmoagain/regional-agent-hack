#!/usr/bin/env python3
"""
Fetch daily temperature (mean, max, min) and optional precipitation using the Open-Meteo ERA5 archive API,
and write accompanying metadata.json for schema tracking.

Modes:
  --mode full    → fetch full archive (1950–present)
  --mode active  → fetch default window (e.g., 2019–present or recent N years)

Docs: https://open-meteo.com/en/docs/era5-api
"""

import argparse, requests, pandas as pd, yaml, json
from datetime import datetime
from pathlib import Path

CONFIG_PATH = Path("config/fetch_defaults.yml")

# ---------------------------------------------------------------------
def get_date_window(mode, start=None, end=None):
    """Resolve start/end dates depending on mode or config."""
    if mode == "full":
        # ERA5 data starts around 1950
        return "1950-01-01", datetime.today().strftime("%Y-%m-%d")

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
            d = cfg.get("default_window", {})
            start = start or d.get("start", "2019-01-01")
            end = end or datetime.today().strftime("%Y-%m-%d")
    else:
        start = start or "2019-01-01"
        end = end or datetime.today().strftime("%Y-%m-%d")

    return start, end


# ---------------------------------------------------------------------
def write_metadata(region_dir: Path):
    """Append or create metadata.json entry for ERA5 temperature + precipitation."""
    meta_file = region_dir / "metadata.json"
    openmeteo_meta = {
        "t2m_mean": {
            "description": "Daily mean 2-meter air temperature from ERA5",
            "units": "°C",
            "temporal_dynamics": "fast",
            "source": "Open-Meteo ERA5 API",
            "fill_policy": "NaN if missing"
        },
        "t2m_max": {
            "description": "Daily maximum 2-meter air temperature from ERA5",
            "units": "°C",
            "temporal_dynamics": "fast",
            "source": "Open-Meteo ERA5 API",
            "fill_policy": "NaN if missing"
        },
        "t2m_min": {
            "description": "Daily minimum 2-meter air temperature from ERA5",
            "units": "°C",
            "temporal_dynamics": "fast",
            "source": "Open-Meteo ERA5 API",
            "fill_policy": "NaN if missing"
        },
        "precip_mm": {
            "description": "Daily precipitation total from ERA5 via Open-Meteo (fallback source)",
            "units": "mm/day",
            "temporal_dynamics": "fast",
            "source": "Open-Meteo ERA5 API",
            "fill_policy": "NaN if missing"
        }
    }

    if meta_file.exists():
        with open(meta_file) as f:
            existing = json.load(f)
    else:
        existing = {}

    existing.update(openmeteo_meta)
    with open(meta_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"🧾 Metadata updated → {meta_file}")


# ---------------------------------------------------------------------
def fetch_openmeteo(bbox, start, end, out):
    lon_min, lat_min, lon_max, lat_max = bbox
    lat = (lat_min + lat_max) / 2
    lon = (lon_min + lon_max) / 2

    print(f"🌡 Fetching Open-Meteo ERA5 data for {start} → {end} [{lon_min}, {lat_min}, {lon_max}, {lat_max}]")

    url = (
        f"https://archive-api.open-meteo.com/v1/era5?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum"
        f"&timezone=UTC"
    )

    r = requests.get(url)
    if not r.ok:
        raise RuntimeError(f"Request failed: {r.status_code} {r.text}")

    data = r.json().get("daily", {})
    if not data:
        print("⚠️  No Open-Meteo data returned.")
        return

    df = pd.DataFrame({
        "date": data["time"],
        "t2m_max": data.get("temperature_2m_max"),
        "t2m_min": data.get("temperature_2m_min"),
        "t2m_mean": data.get("temperature_2m_mean"),
        "precip_mm": data.get("precipitation_sum")
    })

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df.to_csv(out, index=False)
    print(f"✅ Saved {len(df)} daily records → {out}")

    # Write metadata
    write_metadata(Path(out).parent)


# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ERA5 temperature + precipitation via Open-Meteo")
    parser.add_argument("--bbox", nargs=4, type=float, required=True,
                        help="Bounding box: min_lon min_lat max_lon max_lat")
    parser.add_argument("--start", required=False)
    parser.add_argument("--end", required=False)
    parser.add_argument("--mode", choices=["full", "active"], default="active",
                        help="Select 'full' for full ERA5 archive or 'active' for default window.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    start, end = get_date_window(args.mode, args.start, args.end)
    suffix = "_full.csv" if args.mode == "full" else ".csv"
    out_path = Path(args.out).with_suffix(suffix)

    fetch_openmeteo(args.bbox, start, end, out_path)
