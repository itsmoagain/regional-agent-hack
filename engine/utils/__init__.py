"""Utility helpers consumed by the insight engine."""

from __future__ import annotations

from .metadata import load_region_metadata, load_phenology_hints, update_metadata

__all__ = ["load_region_metadata", "load_phenology_hints", "update_metadata"]
diff --git a/engine/utils/metadata.py b/engine/utils/metadata.py
