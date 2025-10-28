

"""
Expanded mapping between crop types and NASA POWER agroclimatic variables.
Includes regional overrides to adjust variable emphasis for tropical,
temperate, and arid zones.
"""

# --- Base variable map ---
CROP_VARIABLE_MAP = {
    # --- Cereal grains ---
    "maize": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "GWETROOT"],
    "corn": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "GWETROOT"],
    "wheat": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "T2MDEW", "ALLSKY_SFC_SW_DWN"],
    "barley": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],
    "oats": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "T2MDEW", "ALLSKY_SFC_SW_DWN"],
    "sorghum": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "GWETROOT"],
    "rice": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN", "GWETROOT"],

    # --- Legumes & pulses ---
    "soybean": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "RH2M"],
    "beans": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],
    "chickpea": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],
    "lentil": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "T2MDEW", "ALLSKY_SFC_SW_DWN"],
    "peanut": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "GWETROOT", "ALLSKY_SFC_SW_DWN"],
    "cowpea": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],

    # --- Fiber crops ---
    "cotton": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "RH2M"],
    "flax": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "T2MDEW", "ALLSKY_SFC_SW_DWN"],
    "hemp": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],
    "jute": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],

    # --- Root & tuber crops ---
    "potato": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],
    "cassava": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "GWETROOT", "ALLSKY_SFC_SW_DWN"],
    "sweet_potato": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "GWETROOT", "ALLSKY_SFC_SW_DWN"],
    "yam": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "GWETROOT", "ALLSKY_SFC_SW_DWN"],

    # --- Oil & bioenergy crops ---
    "sunflower": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],
    "canola": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],
    "rapeseed": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "T2MDEW", "ALLSKY_SFC_SW_DWN"],
    "oil_palm": ["T2M_MAX", "T2M_MIN", "RH2M", "ALLSKY_SFC_SW_DWN", "GWETROOT"],
    "sugarcane": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "GWETROOT"],
    "miscanthus": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],

    # --- Horticultural / vegetable crops ---
    "vegetables": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],
    "tomato": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],
    "onion": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "T2MDEW", "ALLSKY_SFC_SW_DWN"],
    "garlic": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "T2MDEW", "ALLSKY_SFC_SW_DWN"],
    "pepper": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],
    "lettuce": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],
    "carrot": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "T2MDEW", "ALLSKY_SFC_SW_DWN"],
    "spinach": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],
    "cabbage": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],

    # --- Fruit & perennial crops ---
    "apple": ["T2M_MAX", "T2M_MIN", "T2MDEW", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"],
    "grape": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"],
    "citrus": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],
    "banana": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "GWETROOT"],
    "mango": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"],
    "avocado": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"],
    "coffee": ["T2M_MAX", "T2M_MIN", "RH2M", "ALLSKY_SFC_SW_DWN", "GWETROOT"],
    "cocoa": ["T2M_MAX", "T2M_MIN", "RH2M", "ALLSKY_SFC_SW_DWN", "GWETROOT"],
    "tea": ["T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"],

    # --- Forage, pasture, and rangeland ---
    "pasture": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"],
    "alfalfa": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],
    "clover": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "T2MDEW", "ALLSKY_SFC_SW_DWN"],

    # --- Tree crops & agroforestry ---
    "olive": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "T2MDEW", "ALLSKY_SFC_SW_DWN"],
    "date_palm": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"],
    "rubber": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "GWETROOT"],
    "coconut": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "GWETROOT"],
    "timber": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "GWETROOT", "ALLSKY_SFC_SW_DWN"],

    # --- Miscellaneous and broad categories ---
    "mixed": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "RH2M"],
    "horticulture": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"],
    "grain": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"],
}

FALLBACK_VARIABLES = ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"]

# --- Regional modifiers ---
REGIONAL_OVERRIDES = {
    "tropical": {
        "add": ["RH2M", "GWETROOT"],
        "remove": ["T2MDEW"],
    },
    "temperate": {
        "add": ["T2MDEW"],
        "remove": ["RH2M", "GWETROOT"],
    },
    "arid": {
        "add": ["GWETROOT", "PRECTOTCORR"],
        "remove": ["RH2M"],
    },
}

def classify_bioclimate(lat, lon):
    """
    Very simple heuristic classification of region by latitude band.
    Could later be replaced by Köppen zone lookup or WorldClim climatology.
    """
    abs_lat = abs(lat)
    if abs_lat < 15:
        return "tropical"
    elif abs_lat < 35:
        return "arid"
    elif abs_lat < 55:
        return "temperate"
    else:
        return "boreal"


def apply_regional_overrides(base_vars, bioclimate):
    """
    Adjusts the crop variable list based on regional climate zone.
    """
    if bioclimate not in REGIONAL_OVERRIDES:
        return base_vars
    rules = REGIONAL_OVERRIDES[bioclimate]
    updated = [v for v in base_vars if v not in rules["remove"]]
    for v in rules["add"]:
        if v not in updated:
            updated.append(v)
    return updated
