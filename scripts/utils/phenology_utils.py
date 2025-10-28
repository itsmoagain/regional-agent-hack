import yaml
from pathlib import Path

def load_phenology_library(path: str | Path = "config/phenology_library.yml") -> dict:
    """Load and normalize the full phenology library."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Phenology library missing: {p}")
    data = yaml.safe_load(p.read_text())
    library = data.get("phenology_library", data)
    # normalize aliases
    alias_map = {}
    for k, v in library.items():
        for a in v.get("aliases", []):
            alias_map[a.lower()] = k
    return library, alias_map

def resolve_crop(name: str, library: dict, alias_map: dict) -> str:
    """Return canonical crop key from alias or original name."""
    name = name.lower().strip()
    return alias_map.get(name, name if name in library else None)
