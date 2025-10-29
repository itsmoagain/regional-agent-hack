#!/usr/bin/env python3
"""
Build RAG (Regional Aggregation Grid) context layers for a region.

Reads:
  - regions/profiles/insight.<region>.yml  (to get bbox, crops, and metadata)
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
import shutil
from pathlib import Path
from datetime import datetime

from _shared import ensure_region_workspace, load_region_profile

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

try:
    from scripts.build_context_layers import build_context_layers
except ModuleNotFoundError:
    from build_context_layers import build_context_layers


def load_region_meta(region):
    cfg = load_region_profile(region)
    return cfg.get("region_meta", {})


def ensure_context_layers(region: str) -> Path:
    """Guarantee that context layer CSVs exist for the requested region."""
    ctx_dir = DATA_DIR / region / "context_layers"
    if not ctx_dir.exists() or not any(ctx_dir.glob("*.csv")):
        print("ℹ️  No cached context layers detected — building them now.")
        build_context_layers(region)
    return ctx_dir


def collect_layers(region: str) -> dict[str, pd.DataFrame]:
    ctx_dir = ensure_context_layers(region)
    layers: dict[str, pd.DataFrame] = {}

    phenology_frames = []
    for path in ctx_dir.glob("phenology_*.csv"):
        try:
            df = pd.read_csv(path)
            df["source_file"] = path.name
            phenology_frames.append(df)
        except Exception as exc:
            print(f"⚠️  Could not read {path}: {exc}")
    if phenology_frames:
        layers["phenology"] = pd.concat(phenology_frames, ignore_index=True)

    soil_path = ctx_dir / "soil.csv"
    if soil_path.exists():
        layers["soil"] = pd.read_csv(soil_path)

    topo_path = ctx_dir / "topography.csv"
    if topo_path.exists():
        layers["topography"] = pd.read_csv(topo_path)

    return layers


def build_rags_summary(region: str, layers: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rag_path = DATA_DIR / region / "rags.csv"

    summary = {
        "region": region,
        "timestamp": datetime.utcnow().isoformat(),
        "n_crops": len(layers.get("phenology", [])),
        "soil_mean_sand": layers.get("soil", pd.DataFrame()).get("sand_pct", pd.Series([None])).iloc[0]
        if "sand_pct" in layers.get("soil", pd.DataFrame()).columns
        else layers.get("soil", pd.DataFrame()).get("mean_pct", pd.Series([None])).iloc[0]
        if not layers.get("soil", pd.DataFrame()).empty
        else None,
        "elevation_mean": layers.get("topography", pd.DataFrame()).get("elevation_m", pd.Series([None])).iloc[0]
        if not layers.get("topography", pd.DataFrame()).empty
        else None,
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

    ensure_region_workspace(region)
    layers = collect_layers(region)
    if not layers:
        print("❌ No context layers available even after rebuild. Aborting.")
        return

    build_rags_summary(region, layers)
    workspace = ensure_region_workspace(region)
    workspace_out = workspace / "insights" / "rags.csv"
    workspace_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATA_DIR / region / "rags.csv", workspace_out)
    print(f"🗂️  Synced RAG summary to workspace → {workspace_out.relative_to(workspace)}")
    print(f"🎉 Context layers built successfully for {region}")


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG (context) layers for a region.")
    parser.add_argument("--region", required=True, help="Region name, e.g. hungary_farmland")
    args = parser.parse_args()
    build_rag_context(args.region)
