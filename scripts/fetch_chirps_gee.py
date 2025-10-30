import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#!/usr/bin/env python3
"""
Fetch daily precipitation using CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)
and write accompanying metadata.json for schema tracking.

Dataset: UCSB/CHG/CHIRPS/DAILY
"""

import ee, argparse, pandas as pd, yaml, json
from datetime import datetime
from pathlib import Path

CONFIG_PATH = Path("config/fetch_defaults.yml")

def get_date_window(mode, start=None, end=None):
    if mode == "full":
        return "1981-01-01", datetime.today().strftime("%Y-%m-%d")

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
    chirps_meta = {
        "precip_mm_sum": {
            "description": "Daily precipitation total from CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)",
            "units": "mm/day",
            "temporal_dynamics": "fast",
            "source": "UCSB/CHG/CHIRPS/DAILY",
            "fill_policy": "NaN if missing"
        }
    }

    existing = json.load(open(meta_file)) if meta_file.exists() else {}
    existing.update(chirps_meta)
    json.dump(existing, open(meta_file, "w"), indent=2)
    print(f"🧾 Metadata updated → {meta_file}")


def fetch_chirps(bbox, start, end, out):
    print(f"🌧 Fetching CHIRPS daily precipitation from {start} to {end} for {bbox}")
    ee.Initialize(project="situated-insight-engine")

    region = ee.Geometry.Rectangle(bbox)
    col = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start, end)
        .filterBounds(region)
    )

    def daily_mean(img):
        mean = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=5000
        )
        return ee.Feature(None, mean.set("system:time_start", img.date().millis()))

    feats = col.map(daily_mean).getInfo()["features"]
    records = []
    for f in feats:
        ts = f["properties"].get("system:time_start")
        if ts:
            date = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
            val = f["properties"].get("precipitation")
            if val is not None:
                records.append({"date": date, "precip_mm_sum": val})

    df = pd.DataFrame(records).sort_values("date")
    df.to_csv(out, index=False)
    print(f"✅ Saved {len(df)} daily records → {out}")

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

    fetch_chirps(args.bbox, start, end, out_path)
