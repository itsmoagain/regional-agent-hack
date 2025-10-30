import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# scripts/utils_metadata.py
import json
from pathlib import Path

def update_metadata(region_dir: Path, new_entries: dict):
    """Append or create metadata.json with new variable entries."""
    meta_file = region_dir / "metadata.json"

    if meta_file.exists():
        with open(meta_file) as f:
            existing = json.load(f)
    else:
        existing = {}

    existing.update(new_entries)

    with open(meta_file, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"🧾 Metadata updated → {meta_file}")
