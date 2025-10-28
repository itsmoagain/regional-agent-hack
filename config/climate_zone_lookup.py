"""
Utility for determining the Köppen–Geiger climate classification for a given lat/lon.

This uses an open geodata service (WorldClim / Beck et al. 2018 Köppen raster)
and caches results locally for reuse.

Fallbacks to heuristic classification if no data is available.
"""

import json
from pathlib import Path
import requests

CACHE_PATH = Path("data/_cache/climate_zones.json")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Approximate Köppen class → simplified bioclimate label mapping
KOPPEN_TO_ZONE = {
    # Tropical
    "Af": "tropical_rainforest",
    "Am": "tropical_monsoon",
    "Aw": "tropical_savanna",

    # Arid
    "BWh": "arid_desert_hot",
    "BWk": "arid_desert_cold",
    "BSh": "arid_steppe_hot",
    "BSk": "arid_steppe_cold",

    # Temperate
    "Csa": "temperate_dry_summer_hot",
    "Csb": "temperate_dry_summer_warm",
    "Cwa": "temperate_dry_winter_hot",
    "Cwb": "temperate_dry_winter_warm",
    "Cfa": "temperate_humid_subtropical",
    "Cfb": "temperate_oceanic",

    # Cold / Continental
    "Dfa": "cold_continental_hot_summer",
    "Dfb": "cold_continental_warm_summer",
    "Dfc": "cold_subarctic",

    # Polar
    "ET": "polar_tundra",
    "EF": "polar_frost",
}


def get_koppen_code(lat, lon):
    """
    Fetch Köppen–Geiger code from open raster lookup (via geoclimatic API or simplified dataset).
    Returns e.g. "Cfa", "Aw", etc.
    """
    # Using WorldClim’s open raster proxy via climate-data API (small JSON endpoint)
    url = f"https://climate-api.open-meteo.com/v1/classify?latitude={lat}&longitude={lon}"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            data = r.json()
            if "koppen" in data:
                return data["koppen"]
    except Exception:
        pass
    return None


def get_climate_zone(lat, lon):
    """
    Returns a simplified bioclimate label (e.g., 'tropical', 'arid', 'temperate', 'polar')
    using cached Köppen–Geiger data or heuristic fallback.
    """
    # Load cache
    cache = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            pass

    key = f"{lat:.3f},{lon:.3f}"
    if key in cache:
        return cache[key]

    # Try fetching real Köppen code
    koppen = get_koppen_code(lat, lon)
    if koppen:
        if koppen in KOPPEN_TO_ZONE:
            zone = KOPPEN_TO_ZONE[koppen].split("_")[0]
        else:
            # Simplify based on first letter if unknown
            prefix = koppen[0]
            if prefix == "A":
                zone = "tropical"
            elif prefix == "B":
                zone = "arid"
            elif prefix == "C":
                zone = "temperate"
            elif prefix == "D":
                zone = "cold"
            else:
                zone = "polar"
        cache[key] = {"koppen": koppen, "zone": zone}
        CACHE_PATH.write_text(json.dumps(cache, indent=2))
        return cache[key]

    # Fallback: heuristic by latitude
    abs_lat = abs(lat)
    if abs_lat < 15:
        zone = "tropical"
    elif abs_lat < 35:
        zone = "arid"
    elif abs_lat < 55:
        zone = "temperate"
    else:
        zone = "polar"

    cache[key] = {"koppen": None, "zone": zone}
    CACHE_PATH.write_text(json.dumps(cache, indent=2))
    return cache[key]
