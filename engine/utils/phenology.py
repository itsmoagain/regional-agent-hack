"""Lightweight phenology heuristics used when richer data is unavailable."""

from __future__ import annotations

from datetime import datetime
from typing import Mapping


def build_phenology_hints(region: str, metadata: Mapping[str, object] | None = None) -> Mapping[str, object]:
    metadata = metadata or {}
    crops = [c.lower() for c in metadata.get("dominant_crops", []) or []]
    centroid = metadata.get("centroid")

    lat = 0.0
    if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
        lat = float(centroid[0])

    hemisphere = "south" if lat < 0 else "north"
    stage_map: dict[str, str] = {}
    for month in range(1, 13):
        stage_map[f"{month:02d}"] = _month_to_stage(month, hemisphere, crops)

    current_month = datetime.utcnow().month
    current_stage = stage_map[f"{current_month:02d}"]

    return {
        "hemisphere": hemisphere,
        "stage_by_month": stage_map,
        "current_stage": current_stage,
    }


def _month_to_stage(month: int, hemisphere: str, crops: list[str]) -> str:
    """Map ``month`` to a broad phenological stage."""

    # Coffee tends to have longer, overlapping stages. Provide a friendlier mapping.
    if any("coffee" in crop for crop in crops):
        return _coffee_stage(month, hemisphere)

    offset = 0 if hemisphere == "north" else 6
    phase = ((month + offset - 1) % 12) + 1
    if phase in (12, 1, 2):
        return "dormancy / field prep"
    if phase in (3, 4):
        return "emergence"
    if phase in (5, 6, 7):
        return "vegetative growth"
    if phase in (8, 9):
        return "reproductive"
    return "harvest / post-harvest"


def _coffee_stage(month: int, hemisphere: str) -> str:
    offset = 0 if hemisphere == "north" else 6
    phase = ((month + offset - 1) % 12) + 1
    if phase in (11, 12, 1):
        return "flowering"
    if phase in (2, 3, 4):
        return "fruit set"
    if phase in (5, 6, 7):
        return "bean fill"
    if phase in (8, 9):
        return "ripening"
    return "harvest / resting"


__all__ = ["build_phenology_hints"]
diff --git a/README.md b/README.md
