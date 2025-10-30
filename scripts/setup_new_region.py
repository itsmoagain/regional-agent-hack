#!/usr/bin/env python3
"""
🌎 Regional Setup Wizard (Open-Meteo Edition)
--------------------------------------------

Prepares a new region for analysis by:

  • Installing any missing dependencies automatically
  • Collecting region details (name, bounding box, crops, country)
  • Creating configuration files and folder structure
  • Generating soil, elevation, and phenology layers (via Open-Meteo GDD)
  • Building a unified data cache
  • Producing baseline climate insights
  • Computing rolling anomalies

You can type “exit” at any prompt to cancel safely.
"""

import os
import re
import subprocess
import sys
from pathlib import Path


# ------------------------------------------------------------
# Ensure rf_training_lib is installed and importable
# ------------------------------------------------------------
def ensure_rf_training_lib():
    """Ensure rf_training_lib is available; continue in demo mode if not."""
    import importlib
    from pathlib import Path

    lib_path = Path(__file__).resolve().parents[1] / "rf_training_lib"
    sys.path.append(str(lib_path))

    # Detect Kaggle/offline
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("OFFLINE_MODE") == "1":
        print("⚙️ Offline/Kaggle sandbox detected — skipping rf_training_lib installation.")
        try:
            importlib.import_module("rf_training_lib")
            print("✅ rf_training_lib imported directly from path.")
            return
        except Exception as e:
            print(f"⚠️ Could not import rf_training_lib: {e}")
            print("➡️ Continuing in limited DEMO mode (training disabled).")
            os.environ["DEMO_MODE"] = "1"
            return

    # Try to import or install
    try:
        importlib.import_module("rf_training_lib")
        print("✅ rf_training_lib already available.")
    except ModuleNotFoundError:
        print("📦 Installing rf_training_lib (editable mode)...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-e", str(lib_path)]
            )
            importlib.import_module("rf_training_lib")
            print("✅ rf_training_lib installed successfully.")
        except Exception as e:
            print(f"⚠️ Installation failed: {e}")
            print("➡️ Falling back to path import and continuing setup.")
            try:
                importlib.import_module("rf_training_lib")
                print("✅ rf_training_lib imported manually.")
            except Exception as e2:
                print(f"❌ Could not import rf_training_lib: {e2}")
                print("⚙️ Proceeding in DEMO mode — model training disabled.")
                os.environ["DEMO_MODE"] = "1"


def print_environment_banner():
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("OFFLINE_MODE") == "1":
        print(
            "⚙️ Running in offline/sandbox environment — fetchers disabled, demo caches will be used."
        )
    else:
        print("🌐 Running in online setup mode — full data initialization enabled.")


# ------------------------------------------------------------
# Imports after dependency check
# ------------------------------------------------------------
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from scripts.run_pipeline import require
except ModuleNotFoundError:
    from run_pipeline import require  # type: ignore


def _require_or_fail(pkg: str, import_name: str | None = None):
    module = require(pkg, import_name)
    if module is None:
        raise RuntimeError(
            f"Missing required dependency '{pkg}'. "
            "Disable OFFLINE_MODE or install the package manually."
        )
    return module


yaml = _require_or_fail("pyyaml", "yaml")
_require_or_fail("pandas")
_require_or_fail("numpy")
_require_or_fail("requests")
_require_or_fail("folium")

import folium
import numpy as np
import pandas as pd
from init_region import init_region
from build_context_layers import build_context_layers
from build_region_cache import build_region_cache
from build_region_insights import build_region_insights

try:
    from gee_setup import ensure_gee_ready
except ModuleNotFoundError:
    from scripts.gee_setup import ensure_gee_ready


# ------------------------------------------------------------
# Crop selector and input helpers
# ------------------------------------------------------------
def prompt_exit_check(value):
    if value.lower().strip() in {"exit", "quit", "q"}:
        print("🛑 Exiting setup.")
        sys.exit(0)


def select_crop_category():
    """Interactive selector for crop category and crop(s)."""
    config_dir = Path("config")
    library_path = config_dir / "crop_library.yml"
    last_file = config_dir / "last_category.txt"

    if not library_path.exists():
        print("❌ Crop library not found at config/crop_library.yml")
        return "Unknown", []

    with open(library_path, "r", encoding="utf-8") as f:
        crop_library = yaml.safe_load(f)

    categories = list(crop_library.keys())
    last_category = last_file.read_text().strip() if last_file.exists() else None

    print("\n🌾 Crop Selection Wizard")
    if last_category:
        print(f"(Last used category: {last_category})")
    print("Choose a category, or press ENTER to reuse the last one.\n")

    for i, cat in enumerate(categories, start=1):
        print(f"{i}) {cat}")

    choice = input("\nEnter number or category name (or press ENTER): ").strip()
    if not choice and last_category and last_category in crop_library:
        category = last_category
        print(f"🔁 Using previously selected category: {category}")
    elif choice.isdigit() and 1 <= int(choice) <= len(categories):
        category = categories[int(choice) - 1]
    else:
        category = categories[0]
        print(f"⚠️  Defaulting to {category}")

    try:
        last_file.write_text(category)
    except Exception:
        pass

    crops = crop_library[category]
    print(f"\n🌱 Available crops in '{category}':")
    for i, crop in enumerate(crops, start=1):
        name = crop["name"] if isinstance(crop, dict) else crop
        print(f"{i}) {name}")

    # ---- FIXED INDENTATION AND LOGIC ----
    crop_input = input("\nSelect crop(s) (comma/space separated, ENTER for all): ").strip()
    if not crop_input:
        selected = crops
    else:
        entries = [x.strip().lower() for x in crop_input.replace(",", " ").split()]
        selected = []
        for entry in entries:
            if entry.isdigit():
                idx = int(entry)
                if 1 <= idx <= len(crops):
                    selected.append(crops[idx - 1])
                    continue
            for c in crops:
                cname = c["name"].lower() if isinstance(c, dict) else c.lower()
                if entry == cname:
                    selected.append(c)
                    break
        if not selected:
            print("⚠️  None matched, using all crops by default.")
            selected = crops

    enriched = []
    for c in selected:
        if isinstance(c, dict):
            enriched.append({
                "name": c.get("name"),
                "sci_name": c.get("sci_name"),
                "fao_code": c.get("fao_code"),
                "cycle_days": c.get("cycle_days"),
                "season_start": c.get("season_start"),
                "season_end": c.get("season_end"),
                "source": c.get("source", "crop_library.yml")
            })
        else:
            enriched.append({"name": c, "source": "crop_library.yml"})

    return category, enriched


def prompt_bbox():
    print("\n=== Define your Area of Interest (AOI) ===")
    print("1) Go to: https://bboxfinder.com")
    print("2) Copy min_lon, min_lat, max_lon, max_lat")
    while True:
        raw = input("\nPaste your BBOX here (or type 'exit' to quit): ").strip()
        prompt_exit_check(raw)
        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', raw)
        if len(nums) != 4:
            print("Invalid format. Please enter exactly 4 numbers.")
            continue
        try:
            lon1, lat1, lon2, lat2 = map(float, nums)
            min_lon, max_lon = sorted([lon1, lon2])
            min_lat, max_lat = sorted([lat1, lat2])
            bbox = [min_lon, min_lat, max_lon, max_lat]
            print(f"Bounding box set to {bbox}")
            try:
                center_lat = (min_lat + max_lat) / 2
                center_lon = (min_lon + max_lon) / 2
                m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
                folium.Rectangle(bounds=[[min_lat, min_lon], [max_lat, max_lon]], color="blue", fill=True).add_to(m)
                out_path = Path("bbox_preview.html")
                m.save(str(out_path))
                os.startfile(out_path.resolve())
            except Exception as e:
                print(f"⚠️  Map preview skipped ({e}).")
            return bbox
        except Exception:
            print("Could not parse numbers. Try again.")


def confirm_inputs(region_name, bbox, crops, country):
    print("\nYou entered:")
    print(f"  Region: {region_name}")
    print(f"  BBOX: {bbox}")
    print(f"  Crops: {', '.join([c['name'] for c in crops])}")
    print(f"  Country: {country or 'N/A'}")
    confirm = input("\nIs this correct? (Y/n): ").strip().lower()
    if confirm and confirm != "y":
        print("Setup cancelled.")
        sys.exit(0)


def create_demo_caches(region_dir):
    """Generate placeholder data files for offline demo use."""
    import pandas as pd
    from pathlib import Path
    demo_text = "⚠️ DEMO DATA — generated offline for illustration only"
    placeholders = {
        "daily_anomalies.csv": pd.DataFrame({"demo_note": [demo_text]}),
        "daily_merged.csv": pd.DataFrame({"demo_note": [demo_text]}),
        "monthly_merged.csv": pd.DataFrame({"demo_note": [demo_text]}),
        "chirps_gee.csv": pd.DataFrame({"demo_note": [demo_text]}),
        "soil_gee.csv": pd.DataFrame({"demo_note": [demo_text]}),
        "openmeteo.csv": pd.DataFrame({"demo_note": [demo_text]}),
        "ndvi_gee.csv": pd.DataFrame({"demo_note": [demo_text]}),
    }
    current_dir = Path(region_dir) / "current"
    current_dir.mkdir(parents=True, exist_ok=True)
    for name, df in placeholders.items():
        df.to_csv(current_dir / name, index=False)
    print(f"🧪 Created demo placeholder cache files in {current_dir}")


# ------------------------------------------------------------
# Full setup sequence
# ------------------------------------------------------------
def run_region_setup(region_name=None):
    print_environment_banner()
    if not region_name:
        print("\n🌱 === Regional Setup Wizard ===")
        region_name = input("Enter region name (e.g., transdanubia_farmland): ").strip()

    prompt_exit_check(region_name)
    bbox = prompt_bbox()
    category, crops = select_crop_category()
    country = input("Enter country (optional): ").strip()
    confirm_inputs(region_name, bbox, crops, country)

    print("\n🚀 Starting setup...\n")

    # Step 1
    try:
        print("Step 1: Preparing region folders and config...")
        init_region(region_name, bbox=bbox, crops=crops, country=country)
        print("✅ Region structure initialized.\n")
    except Exception as e:
        print(f"❌ Region initialization failed: {e}")

    # Step 1.5
    try:
        print("Step 1.5: Checking Google Earth Engine setup...")
        ee_ok = ensure_gee_ready(project=os.environ.get("EE_PROJECT"))
        if ee_ok:
            print("✅ Earth Engine is ready.\n")
        else:
            print("⚠️ Earth Engine not initialized; skipping.\n")
    except Exception as e:
        print(f"⚠️ Earth Engine setup error: {e}\n")

    # Step 2
    print("Step 2: Fetching soil, elevation, and phenology layers (Open-Meteo)...")
    for attempt in range(2):
        try:
            subprocess.run([sys.executable, "scripts/build_context_layers.py", "--region", region_name], check=True)
            print("✅ Context layers built.\n")
            break
        except Exception as e:
            print(f"⚠️ Context layer build failed: {e}")
            if attempt == 0:
                print("🔁 Retrying once...")
            else:
                print("❌ Skipped after second failure.\n")

    # Step 3
    print("Step 3: Building regional cache...")
    for attempt in range(2):
        try:
            subprocess.run([sys.executable, "scripts/build_region_cache.py", "--region", region_name], check=True)
            print("✅ Regional cache built.\n")
            break
        except Exception as e:
            print(f"⚠️ Cache build failed: {e}")
            if attempt == 0:
                print("🔁 Retrying once...")
            else:
                print("❌ Skipped after second failure.\n")

    # Step 4
    print("Step 4: Generating climate insights...")
    for attempt in range(2):
        try:
            subprocess.run([sys.executable, "scripts/build_region_insights.py", "--region", region_name], check=True)
            print("✅ Climate insights created.\n")
            break
        except Exception as e:
            print(f"⚠️ Insight generation failed: {e}")
            if attempt == 0:
                print("🔁 Retrying once...")
            else:
                print("❌ Skipped after second failure.\n")

    # Step 5
    print("Step 5: Computing rolling anomalies...")
    for attempt in range(2):
        try:
            subprocess.run([sys.executable, "scripts/compute_anomalies.py", "--region", region_name], check=True)
            print("✅ Rolling anomalies computed.\n")
            break
        except Exception as e:
            print(f"⚠️ Anomaly computation failed: {e}")
            if attempt == 0:
                print("🔁 Retrying once...")
            else:
                print("❌ Skipped after second failure.\n")

    if os.environ.get("DEMO_MODE") == "1":
        create_demo_caches(f"data/{region_name}")

    print(f"\n🎉 Region '{region_name}' setup complete!")
    print("Includes:")
    print("  - Config + metadata YAML")
    print("  - Context layers (soil, elevation, phenology)")
    print("  - Merged cache CSVs")
    print("  - Insight summaries")
    print("  - Rolling anomalies\n")
    print("Next: Train your model with:")
    print(f"  python scripts/train_region_model.py --region {region_name} --tier 1\n")


# ------------------------------------------------------------
# CLI entry
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Initialize a new region for Situated Insight.")
    parser.add_argument("--region", help="Optional region name (if omitted, you’ll be prompted).")
    args = parser.parse_args()

    ensure_rf_training_lib()
    run_region_setup(args.region)
