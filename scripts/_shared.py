from __future__ import annotations


import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
"""Shared helpers for script entrypoints to access the packaged utilities."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.regional_agent.config import (  # noqa: E402
    resolve_region_config_path,
    load_region_profile,
    ensure_region_workspace,
    get_region_cache_dir,
    get_region_current_dir,
    get_region_data_root,
    load_layer_registry,
)

__all__ = [
    "ROOT",
    "resolve_region_config_path",
    "load_region_profile",
    "ensure_region_workspace",
    "get_region_cache_dir",
    "get_region_current_dir",
    "get_region_data_root",
    "load_layer_registry",
]
