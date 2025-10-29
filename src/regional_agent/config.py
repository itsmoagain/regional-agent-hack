from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

try:
    from scripts.run_pipeline import require
except Exception:  # pragma: no cover - fallback for isolated package usage
    import importlib

    def require(pkg: str, import_name: str | None = None):  # type: ignore
        module_name = import_name or pkg
        return importlib.import_module(module_name)


yaml = require("pyyaml", "yaml")
if yaml is None:
    raise RuntimeError(
        "PyYAML is required for regional_agent.config. "
        "Re-run without OFFLINE_MODE to install missing dependencies."
    )

ROOT = Path(__file__).resolve().parents[2]
REGION_PROFILES_DIR = ROOT / "regions" / "profiles"
LEGACY_CONFIG_DIR = ROOT / "config"
WORKSPACES_DIR = ROOT / "regions" / "workspaces"
DATA_ROOT = ROOT / "data"


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_region_config_path(region: str) -> Path:
    """Return the path to the region profile, searching new and legacy layouts."""
    REGION_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [
        REGION_PROFILES_DIR / f"insight.{region}.yml",
        REGION_PROFILES_DIR / f"{region}.yml",
        LEGACY_CONFIG_DIR / f"insight.{region}.yml",
        LEGACY_CONFIG_DIR / f"{region}.yml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No profile found for '{region}'. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def load_region_profile(region: str) -> Dict[str, Any]:
    """Load a region profile and resolve inheritance."""
    profile_path = resolve_region_config_path(region)
    cfg = yaml.safe_load(profile_path.read_text())
    if "extends" in cfg:
        base_path = Path(cfg["extends"])
        if not base_path.is_absolute():
            base_path = profile_path.parent / base_path
        base = yaml.safe_load(base_path.read_text())
        base.update({k: v for k, v in cfg.items() if k != "extends"})
        cfg = base
    cfg.setdefault("region_meta", {})
    cfg["region_meta"].setdefault("key", region)
    cfg["region_meta"].setdefault("profile_path", str(profile_path))
    return cfg


def ensure_region_workspace(region: str) -> Path:
    """Ensure a dedicated workspace directory for interactive exploration."""
    workspace = _ensure_directory(WORKSPACES_DIR / region)
    (workspace / "logs").mkdir(exist_ok=True)
    (workspace / "insights").mkdir(exist_ok=True)
    (workspace / "models").mkdir(exist_ok=True)
    (workspace / "cache").mkdir(exist_ok=True)
    # Drop a lightweight manifest for discoverability.
    manifest = {
        "region": region,
        "profile": str(resolve_region_config_path(region)),
    }
    (workspace / "workspace.json").write_text(json.dumps(manifest, indent=2))
    return workspace


def get_region_data_root(region: str) -> Path:
    """Return the root data directory for a region, creating it if needed."""
    return _ensure_directory(DATA_ROOT / region)


def get_region_cache_dir(region: str) -> Path:
    """Return the caches directory for a region."""
    return _ensure_directory(get_region_data_root(region) / "caches")


def get_region_current_dir(region: str) -> Path:
    """Return the working "current" directory that downstream steps should read from."""
    return _ensure_directory(get_region_data_root(region) / "current")


@dataclass
class LayerSpec:
    """Normalized definition for a configured data layer."""

    name: str
    fetcher: str
    cache_file: str
    ttl_days: int
    required: bool = True
    source_url: str | None = None
    fetch: Dict[str, Any] | None = None


def _resolve_layer_config(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    if not raw:
        return {}
    if "extends" in raw:
        base_path = Path(raw["extends"])
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base = _resolve_layer_config(base_path)
        base.update({k: v for k, v in raw.items() if k != "extends"})
        raw = base
    return raw


def load_layer_registry(region: str) -> Dict[str, LayerSpec]:
    """Load the per-region data layer registry."""

    candidates: Iterable[Path] = (
        Path(LEGACY_CONFIG_DIR) / f"insight.{region}.yml",
        REGION_PROFILES_DIR / f"insight.{region}.yml",
    )
    config_path: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            config_path = candidate
            break
    if config_path is None:
        raise FileNotFoundError(
            f"No data layer registry found for '{region}'. Expected one of: "
            + ", ".join(str(c) for c in candidates)
        )

    payload = _resolve_layer_config(config_path)
    layers: Mapping[str, Any] = payload.get("data_layers", {})
    registry: Dict[str, LayerSpec] = {}
    for name, cfg in layers.items():
        try:
            registry[name] = LayerSpec(
                name=name,
                fetcher=cfg["fetcher"],
                cache_file=cfg.get("cache_file", f"{name}.csv"),
                ttl_days=int(cfg.get("ttl_days", 30)),
                required=bool(cfg.get("required", True)),
                source_url=cfg.get("source_url"),
                fetch=cfg.get("fetch"),
            )
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError(f"Layer '{name}' missing required config key: {exc}") from exc
    return registry


__all__ = [
    "REGION_PROFILES_DIR",
    "resolve_region_config_path",
    "load_region_profile",
    "ensure_region_workspace",
    "get_region_data_root",
    "get_region_cache_dir",
    "get_region_current_dir",
    "load_layer_registry",
    "LayerSpec",
]
