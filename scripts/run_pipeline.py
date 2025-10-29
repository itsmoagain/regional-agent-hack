#!/usr/bin/env python3
"""Unified regional pipeline runner with explicit online/offline modes."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

from _shared import (
    get_region_cache_dir,
    get_region_current_dir,
    load_layer_registry,
    load_region_profile,
)
from regional_agent.config import LayerSpec
from regional_agent.pipeline.fetchers import run_fetcher

MODE = Literal["analyze", "bootstrap", "refresh"]


@dataclass
class LayerState:
    spec: LayerSpec
    path: Path
    entry: Optional[Dict[str, object]]
    fetched_at: Optional[datetime]
    expires_at: Optional[datetime]
    age_days: Optional[float]
    expired: bool
    exists: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regional insight pipeline")
    parser.add_argument("--region", required=True, help="Region key (e.g. austin_farmland)")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--bootstrap", action="store_true", help="Fetch all required layers from scratch")
    mode_group.add_argument("--refresh", action="store_true", help="Refresh layers whose TTL has expired")
    mode_group.add_argument("--analyze", action="store_true", help="Analyze using only cached data (default)")
    parser.add_argument("--allow-stale", action="store_true", help="Permit using expired caches without refreshing")
    parser.add_argument(
        "--max-staleness",
        type=int,
        help="Fail if any required layer is older than this many days",
    )
    return parser.parse_args()


def determine_mode(args: argparse.Namespace) -> MODE:
    if args.bootstrap:
        return "bootstrap"
    if args.refresh:
        return "refresh"
    return "analyze"


def parse_iso(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            return datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def build_layer_state(
    spec: LayerSpec,
    entry: Optional[Dict[str, object]],
    current_path: Path,
    now: datetime,
) -> LayerState:
    fetched_at = parse_iso(entry.get("fetched_at")) if entry else None
    if fetched_at is None and current_path.exists():
        fetched_at = datetime.fromtimestamp(current_path.stat().st_mtime, tz=timezone.utc)

    expires_at = parse_iso(entry.get("expires_at")) if entry else None
    if expires_at is None and fetched_at is not None:
        expires_at = fetched_at + timedelta(days=spec.ttl_days)

    age_days = None
    if fetched_at is not None:
        age_days = (now - fetched_at).total_seconds() / 86400

    expired = bool(expires_at and expires_at <= now)

    return LayerState(
        spec=spec,
        path=current_path,
        entry=entry,
        fetched_at=fetched_at,
        expires_at=expires_at,
        age_days=age_days,
        expired=expired,
        exists=current_path.exists(),
    )


def load_manifest(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def layer_entries_by_file(manifest: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    for entry in manifest.get("layers", []):
        cache_file = entry.get("cache_file")
        if cache_file:
            result[cache_file] = entry
    return result


def summarize_state(state: LayerState, now: datetime) -> str:
    if not state.exists:
        if state.spec.required:
            return "❌ missing"
        return "⚠️ optional missing"
    if state.age_days is None:
        return "ℹ️ age unknown"
    ttl = state.spec.ttl_days
    age_str = f"{state.age_days:.1f}d (TTL {ttl})"
    if state.expired:
        return f"⚠️ {age_str} → stale"
    return f"✅ {age_str}"


def check_staleness(
    states: Iterable[LayerState],
    *,
    allow_stale: bool,
    max_staleness: Optional[int],
    mode: MODE,
) -> None:
    violations: List[str] = []
    for state in states:
        if not state.exists and state.spec.required:
            violations.append(f"Required layer '{state.spec.name}' is missing")
            continue
        if state.age_days is None:
            continue
        if max_staleness is not None and state.age_days > max_staleness and state.spec.required:
            violations.append(
                f"Layer '{state.spec.name}' is {state.age_days:.1f} days old (> {max_staleness} days max)"
            )
        if mode == "analyze" and state.expired and state.spec.required and not allow_stale:
            violations.append(
                f"Layer '{state.spec.name}' is stale ({state.age_days:.1f} days) — rerun with --refresh or --allow-stale"
            )
    if violations:
        raise SystemExit("\n".join(violations))


def compute_file_stats(path: Path) -> Dict[str, object]:
    import hashlib

    data = path.read_bytes()
    rows = 0
    if path.suffix == ".csv" and path.stat().st_size:
        with path.open("r", encoding="utf-8") as fh:
            rows = sum(1 for _ in fh) - 1
            if rows < 0:
                rows = 0
    return {
        "bytes": path.stat().st_size,
        "hash": hashlib.sha256(data).hexdigest(),
        "rows": rows,
    }


def ensure_snapshot_dir(region: str, now: datetime) -> Path:
    caches_dir = get_region_cache_dir(region)
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    candidate = caches_dir / ts
    suffix = 1
    while candidate.exists():
        candidate = caches_dir / f"{ts}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def copy_previous_layer(
    source_dirs: List[Path],
    destination: Path,
) -> bool:
    for base in source_dirs:
        candidate = base / destination.name
        if candidate.exists():
            shutil.copy2(candidate, destination)
            return True
    return False


def update_current_view(region: str, snapshot_dir: Path) -> Path:
    current_dir = get_region_current_dir(region)
    tmp_dir = current_dir.parent / "_current_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    shutil.copytree(snapshot_dir, tmp_dir, dirs_exist_ok=True)
    if current_dir.exists():
        shutil.rmtree(current_dir)
    tmp_dir.rename(current_dir)
    return current_dir


def write_manifest(
    snapshot_dir: Path,
    *,
    region: str,
    mode: MODE,
    created_at: datetime,
    entries: List[Dict[str, object]],
) -> Path:
    manifest = {
        "region": region,
        "mode": mode,
        "created_at": created_at.isoformat() + "Z",
        "snapshot": snapshot_dir.name,
        "layers": entries,
    }
    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def run_offline_steps(region: str) -> None:
    commands = [
        [sys.executable, "scripts/build_region_cache.py", "--region", region],
        [sys.executable, "scripts/build_region_insights.py", "--region", region],
        [sys.executable, "scripts/compute_anomalies.py", "--region", region, "--plot-format", "png"],
        [sys.executable, "scripts/build_training_window.py", "--region", region],
    ]
    for cmd in commands:
        print(f"➡️  Running {' '.join(cmd)}")
        try:
            subprocess_env = os.environ.copy()
            subprocess_env.setdefault("PYTHONUNBUFFERED", "1")
            subprocess_env.setdefault("OFFLINE_MODE", os.environ.get("OFFLINE_MODE", "0"))
            subprocess.run(cmd, check=True, env=subprocess_env)
        except subprocess.CalledProcessError as exc:
            print(f"⚠️  Step failed ({' '.join(cmd)}): {exc}")
            break


def main() -> None:
    args = parse_args()
    mode = determine_mode(args)

    if mode == "analyze":
        os.environ["OFFLINE_MODE"] = "1"
    else:
        os.environ.pop("OFFLINE_MODE", None)

    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    region = args.region

    print(f"🛰  Running pipeline for {region} in {mode.upper()} mode")

    profile = load_region_profile(region)
    bbox = profile.get("region_meta", {}).get("bbox")
    if not bbox or len(bbox) != 4:
        raise SystemExit("Region profile missing bbox coordinates")

    registry = load_layer_registry(region)
    current_dir = get_region_current_dir(region)
    manifest_path = current_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    entry_by_file = layer_entries_by_file(manifest)

    previous_snapshot = manifest.get("snapshot") if isinstance(manifest, dict) else None
    previous_dirs: List[Path] = []
    if previous_snapshot:
        previous_dir = get_region_cache_dir(region) / str(previous_snapshot)
        if previous_dir.exists():
            previous_dirs.append(previous_dir)
    if current_dir.exists():
        previous_dirs.append(current_dir)

    states: List[LayerState] = []
    for spec in registry.values():
        state = build_layer_state(
            spec,
            entry_by_file.get(spec.cache_file),
            current_dir / spec.cache_file,
            now,
        )
        print(f" • {spec.name}: {summarize_state(state, now)}")
        states.append(state)

    check_staleness(states, allow_stale=args.allow_stale, max_staleness=args.max_staleness, mode=mode)

    if mode == "analyze":
        run_offline_steps(region)
        return

    snapshot_dir = ensure_snapshot_dir(region, now)
    manifest_entries: List[Dict[str, object]] = []

    for state in states:
        destination = snapshot_dir / state.spec.cache_file
        should_fetch = False
        if mode == "bootstrap":
            should_fetch = state.spec.required or not state.exists
            if state.expired and not args.allow_stale:
                should_fetch = True
        else:  # refresh
            if not state.exists:
                should_fetch = True
            elif state.expired and not args.allow_stale:
                should_fetch = True

        metadata: Dict[str, object]
        provenance: Dict[str, str] | None = None

        if should_fetch:
            print(f"🌐 Fetching {state.spec.name} via {state.spec.fetcher}")
            provenance = run_fetcher(
                state.spec,
                mode="bootstrap" if mode == "bootstrap" else "refresh",
                destination=destination,
                bbox=bbox,
            )
            fetched_at = now
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            copied = copy_previous_layer(previous_dirs, destination)
            if not copied:
                if state.spec.required:
                    raise SystemExit(f"No cached data available for required layer '{state.spec.name}'")
                else:
                    print(f"⚠️  Optional layer '{state.spec.name}' missing; skipping")
                    continue
            fetched_at = state.fetched_at or now

        metadata = compute_file_stats(destination)
        expires_at = fetched_at + timedelta(days=state.spec.ttl_days)

        entry = {
            "layer": state.spec.name,
            "cache_file": state.spec.cache_file,
            "fetcher": state.spec.fetcher,
            "ttl_days": state.spec.ttl_days,
            "required": state.spec.required,
            "fetched": should_fetch,
            "fetched_at": fetched_at.isoformat() + "Z",
            "expires_at": expires_at.isoformat() + "Z",
            "source_url": state.spec.source_url,
            **metadata,
        }
        if provenance:
            entry["provenance"] = provenance
        manifest_entries.append(entry)

    manifest_path = write_manifest(snapshot_dir, region=region, mode=mode, created_at=now, entries=manifest_entries)
    current_dir = update_current_view(region, snapshot_dir)
    shutil.copy2(manifest_path, current_dir / "manifest.json")

    print(f"🗂  Snapshot ready at {snapshot_dir}")

    run_offline_steps(region)


if __name__ == "__main__":
    main()
