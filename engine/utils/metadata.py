"""Metadata helpers used throughout the insight engine pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from regional_agent.config import (
    get_region_current_dir,
    load_region_profile,
)

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover - Python <3.11 fallback
    TypedDict = dict  # type: ignore[misc,assignment]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


class RegionMetadata(TypedDict, total=False):
    """Normalized metadata payload for a region."""

    region_key: str
    region_name: str
    country: str | None
    bbox: list[float]
    dominant_crops: list[str]
    soil_type: str | None
    management_system: str | None
    notes: str | None
    rule_overrides: list[dict[str, Any]]
    practice_overrides: Mapping[str, Any]


def load_region_metadata(region: str) -> RegionMetadata:
    """Load metadata for *region*, applying opinionated defaults."""

    profile = load_region_profile(region)
    meta: Mapping[str, Any] = profile.get("region_meta", {})

    region_key = str(meta.get("key", region))
    region_name = str(meta.get("name", region.replace("_", " ").title()))
    country = meta.get("country")
    bbox = list(meta.get("bbox", []))
    notes = meta.get("notes")

    dominant_crops = _as_list(meta.get("crops") or profile.get("dominant_crops"))
    soil_type = meta.get("soil_type") or profile.get("soil_type")
    management = meta.get("management_system") or profile.get("management_system")

    payload: RegionMetadata = {
        "region_key": region_key,
        "region_name": region_name,
        "country": country,
        "bbox": bbox,
        "dominant_crops": dominant_crops,
        "soil_type": soil_type,
        "management_system": management,
        "notes": notes,
        "rule_overrides": list(profile.get("rules", [])),
        "practice_overrides": profile.get("practices", {}) or {},
    }

    centroid = _compute_centroid(bbox) if bbox else None
    if centroid:
        payload["centroid"] = centroid  # type: ignore[index]

    return payload


def _compute_centroid(bbox: Iterable[float]) -> tuple[float, float]:
    values = list(bbox)
    if len(values) != 4:
        raise ValueError("bbox must contain four values: [min_lon, min_lat, max_lon, max_lat]")
    min_lon, min_lat, max_lon, max_lat = map(float, values)
    return ((min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0)


def update_metadata(region: str, new_entries: Mapping[str, Any]) -> Path:
    """Persist *new_entries* into ``metadata.json`` under the region cache.

    This is the central hook for recording auxiliary metadata derived during
    experimentation (e.g., regional mesh associations or prospective training
    features). Downstream jobs can read the cached `metadata.json` alongside the
    anomaly time series to build richer model corpora without requiring a
    separate metadata export step.
    """

    target = get_region_current_dir(region) / "metadata.json"
    existing: Dict[str, Any] = {}
    if target.exists():
        try:
            existing = json.loads(target.read_text())
        except Exception:  # pragma: no cover - resilient to partial writes
            existing = {}
    existing.update(dict(new_entries))
    target.write_text(json.dumps(existing, indent=2, sort_keys=True))
    return target


def load_phenology_hints(region: str, metadata: RegionMetadata | None = None) -> Mapping[str, Any]:
    """Load lightweight phenology hints for *region* if available."""

    try:
        from .phenology import build_phenology_hints
    except Exception:
        return {}

    if metadata is None:
        metadata = load_region_metadata(region)
    return build_phenology_hints(region, metadata)


__all__ = [
    "RegionMetadata",
    "load_region_metadata",
    "load_phenology_hints",
    "update_metadata",
]
