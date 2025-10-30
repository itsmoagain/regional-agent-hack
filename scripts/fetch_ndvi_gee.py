import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#!/usr/bin/env python3
"""
Fetch MODIS NDVI (Normalized Difference Vegetation Index)
and write accompanying metadata.json for schema tracking.

Dataset: MODIS/061/MOD13Q1
"""

import ee, argparse, pandas as pd, yaml, json
from datetime import datetime
from pathlib import Path

CONFIG_PATH = Path("config/fetch_defaults.yml")

def get_date_window(mode, start=None, end=None):
    if mode == "full":
        return "2000-02-18", datetime.today().strftime("%Y-%m-%d")

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
    meta_file = region_dir / "metadata.json"
    ndvi_meta = {
        "ndvi_mean": {
            "description": "MODIS NDVI (vegetation greenness, scaled 0–1)",
            "units": "unitless (0–1)",
            "temporal_dynamics": "medium",
            "source": "MODIS/061/MOD13Q1",
            "fill_policy": "NaN if missing"
        }
    }

    existing = json.load(open(meta_file)) if meta_file.exists() else {}
    existing.update(ndvi_meta)
    json.dump(existing, open(meta_file, "w"), indent=2)
    print(f"🧾 Metadata updated → {meta_file}")


def fetch_ndvi(bbox, start, end, out):
    print(f"🌿 Fetching MODIS NDVI from {start} to {end} for {bbox}")
    ee.Initialize(project="situated-insight-engine")

    region = ee.Geometry.Rectangle(bbox)
    col = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(start, end)
        .filterBounds(region)
        .select("NDVI")
    )

    def region_mean(img):
        mean = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=250
        )
        return ee.Feature(None, mean.set("system:time_start", img.date().millis()))

    feats = col.map(region_mean).getInfo()["features"]
    records = []
    for f in feats:
        ts = f["properties"].get("system:time_start")
        if ts:
            date = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
            val = f["properties"].get("NDVI")
            if val is not None:
                records.append({"date": date, "ndvi_mean": val / 10000})  # scale factor

    df = pd.DataFrame(records).sort_values("date")
    df.to_csv(out, index=False)
    print(f"✅ Saved {len(df)} NDVI records → {out}")

    # ✅ Write metadata
    write_metadata(Path(out).parent)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bbox", nargs=4, type=float, required=True)
    p.add_argument("--start", type=str)
    p.add_argument("--end", type=str)
    p.add_argument("--mode", choices=["full", "active"], default="active")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    start, end = get_date_window(args.mode, args.start, args.end)
    suffix = "_full.csv" if args.mode == "full" else ".csv"
    out_path = Path(args.out).with_suffix(suffix)

    fetch_ndvi(args.bbox, start, end, out_path)
