import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#!/usr/bin/env python3
"""
Fetch daily soil moisture (surface + root zone) using Google Earth Engine (SMAP 10 km HSL dataset)
and write accompanying metadata.json for schema tracking.

Dataset: NASA_USDA/HSL/SMAP_soil_moisture
Bands:
  - ssm  (surface 0–5 cm)
  - susm (subsurface/root-zone 5–100 cm)
Docs: https://developers.google.com/earth-engine/datasets/catalog/NASA_USDA_HSL_SMAP_soil_moisture
"""

import ee, argparse, pandas as pd, yaml, json
from datetime import datetime
from pathlib import Path

CONFIG_PATH = Path("config/fetch_defaults.yml")

def get_date_window(mode, start=None, end=None):
    """Resolve start/end dates depending on mode or config."""
    if mode == "full":
        return "2015-03-31", datetime.today().strftime("%Y-%m-%d")

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


def write_metadata(region_dir: Path):
    """Append or create metadata.json entry for soil moisture."""
    meta_file = region_dir / "metadata.json"
    soil_meta = {
        "soil_surface_moisture": {
            "description": "Top 0–5 cm volumetric soil moisture (surface layer)",
            "units": "m3/m3",
            "temporal_dynamics": "fast",
            "source": "NASA_USDA/HSL/SMAP_soil_moisture",
            "fill_policy": "NaN if missing"
        },
        "soil_rootzone_moisture": {
            "description": "5–100 cm volumetric soil moisture (root-zone layer)",
            "units": "m3/m3",
            "temporal_dynamics": "slow",
            "source": "NASA_USDA/HSL/SMAP_soil_moisture",
            "fill_policy": "NaN if missing"
        }
    }

    if meta_file.exists():
        with open(meta_file) as f:
            existing = json.load(f)
    else:
        existing = {}

    existing.update(soil_meta)
    with open(meta_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"🧾 Metadata updated → {meta_file}")


def fetch_soil(bbox, start, end, out):
    print(f"🪱 Fetching SMAP soil moisture (surface + root zone) from {start} to {end} for {bbox}")
    ee.Initialize(project="situated-insight-engine")

    region = ee.Geometry.Rectangle(bbox)
    collection = (
        ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture")
        .filterDate(start, end)
        .filterBounds(region)
    )

    def extract_mean(image):
        stats = image.select(["ssm", "susm"]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=10000
        )
        date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        return ee.Feature(None, {
            "date": date,
            "soil_surface_moisture": stats.get("ssm"),
            "soil_rootzone_moisture": stats.get("susm")
        })

    features = collection.map(extract_mean).filter(
        ee.Filter.notNull(["soil_surface_moisture", "soil_rootzone_moisture"])
    )

    dates = features.aggregate_array("date").getInfo()
    surf_vals = features.aggregate_array("soil_surface_moisture").getInfo()
    root_vals = features.aggregate_array("soil_rootzone_moisture").getInfo()

    if not dates or not surf_vals:
        print("⚠️ No soil data returned. Check your bbox or date range.")
        return

    df = pd.DataFrame({
        "date": dates,
        "soil_surface_moisture": surf_vals,
        "soil_rootzone_moisture": root_vals
    })
    df.to_csv(out, index=False)
    print(f"✅ Saved {len(df)} daily records → {out}")

    # Write metadata
    write_metadata(Path(out).parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch SMAP soil moisture via GEE")
    parser.add_argument("--bbox", nargs=4, type=float, required=True,
                        help="Bounding box: min_lon min_lat max_lon max_lat")
    parser.add_argument("--start", required=False)
    parser.add_argument("--end", required=False)
    parser.add_argument("--mode", choices=["full", "active"], default="active",
                        help="Select 'full' for complete archive or 'active' for default window.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    start, end = get_date_window(args.mode, args.start, args.end)
    suffix = "_full.csv" if args.mode == "full" else ".csv"
    out_path = Path(args.out).with_suffix(suffix)

    fetch_soil(args.bbox, start, end, out_path)
