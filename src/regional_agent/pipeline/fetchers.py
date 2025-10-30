"""Fetcher adapters used by the consolidated pipeline runner."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional

from src.regional_agent.config import LayerSpec

FetchMode = Literal["bootstrap", "refresh"]


@dataclass
class FetchCommand:
    name: str
    script: str


FETCHER_SCRIPTS: Dict[str, FetchCommand] = {
    "chirps_gee": FetchCommand("chirps_gee", "fetch_chirps_gee.py"),
    "openmeteo_api": FetchCommand("openmeteo_api", "fetch_openmeteo.py"),
    "modis_gee": FetchCommand("modis_gee", "fetch_ndvi_gee.py"),
    "soil_gee": FetchCommand("soil_gee", "fetch_soil_gee.py"),
}


def _resolve_mode(spec: LayerSpec, mode: FetchMode) -> str:
    cfg = spec.fetch or {}
    if mode == "bootstrap":
        return cfg.get("bootstrap_mode", cfg.get("mode", "full"))
    return cfg.get("mode", "active")


def _append_if_present(args: list[str], flag: str, value: Optional[str]) -> None:
    if value:
        args.extend([flag, value])


def _maybe_cleanup(destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    alt = destination.with_name(destination.stem + "_full.csv")
    if alt.exists():
        alt.unlink()


def _resolve_output(destination: Path) -> Path:
    if destination.exists():
        return destination
    alt = destination.with_name(destination.stem + "_full.csv")
    if alt.exists():
        alt.rename(destination)
        return destination
    raise FileNotFoundError(f"Fetcher did not create expected output: {destination}")


def run_fetcher(
    spec: LayerSpec,
    *,
    mode: FetchMode,
    destination: Path,
    bbox: Iterable[float],
) -> Dict[str, str]:
    """Execute the configured fetcher and return provenance metadata."""

    command = FETCHER_SCRIPTS.get(spec.fetcher)
    if command is None:
        raise KeyError(f"Unknown fetcher '{spec.fetcher}' for layer '{spec.name}'")

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    script_path = scripts_dir / command.script
    if not script_path.exists():  # pragma: no cover - defensive guard
        raise FileNotFoundError(f"Fetcher script not found: {script_path}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    _maybe_cleanup(destination)

    fetch_mode = _resolve_mode(spec, mode)
    bbox_vals = [str(v) for v in bbox]

    args = [
        sys.executable,
        str(script_path),
        "--bbox",
        *bbox_vals,
        "--mode",
        fetch_mode,
        "--out",
        str(destination),
    ]

    cfg = spec.fetch or {}
    _append_if_present(args, "--start", cfg.get("start"))
    _append_if_present(args, "--end", cfg.get("end"))
    extra = cfg.get("extra_args", [])
    if extra:
        args.extend(extra)

    started = datetime.utcnow()
    subprocess.run(args, check=True)
    finished = datetime.utcnow()

    output_path = _resolve_output(destination)

    return {
        "started_at": started.isoformat() + "Z",
        "finished_at": finished.isoformat() + "Z",
        "command": " ".join(shlex.quote(part) for part in args),
        "output": str(output_path),
    }


__all__ = ["run_fetcher"]
