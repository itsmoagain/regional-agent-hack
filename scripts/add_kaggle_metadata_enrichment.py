import json, os, datetime

TAGS = ["climate", "agriculture", "green-ai", "cpu-efficient", "spatial-data", "time-series"]

def enrich_metadata(folder, region_name):
    path = os.path.join(folder, "dataset-metadata.json")
    if not os.path.exists(path):
        print(f"⚠️ No metadata found in {folder}, skipping.")
        return

    with open(path) as f:
        meta = json.load(f)

    meta["subtitle"] = f"{region_name} regional climate cache ({datetime.date.today().year})"
    meta["keywords"] = TAGS
    meta["resources"] = [
        {"path": "chirps_cached.csv", "description": "Daily rainfall (mm)"},
        {"path": "openmeteo_cached.csv", "description": "Temperature & RH"},
        {"path": "era5_recent.csv", "description": "Short-term ERA5 reanalysis"},
    ]
    meta["usabilityRating"] = 1.0
    meta["collaborators"] = ["morganurich"]

    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Enriched metadata for {region_name}")

for region in ["hungary_farmland", "jamaica_coffee"]:
    enrich_metadata(os.path.join("data", region), region)
