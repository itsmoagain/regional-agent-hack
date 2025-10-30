#!/usr/bin/env python3
"""Wrap any command with CodeCarbon to log Green-AI metrics.

Example
-------
python engine/evaluate_greenai.py \
    --region hungary_transdanubia \
    --label baseline \
    --command "python scripts/train_region_model.py --region hungary_transdanubia --tier 1"
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from codecarbon import EmissionsTracker

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.regional_agent import ensure_region_workspace


def _default_hardware_metadata() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "python_version": sys.version.split()[0],
    }


def run_with_tracking(command: str, project_name: str | None = None) -> Dict[str, Any]:
    tracker = EmissionsTracker(
        project_name=project_name,
        save_to_file=False,
        measure_power_secs=1,
        log_level="error",
    )
    tracker.start()
    started = time.time()
    returncode = 0
    emissions = 0.0
    elapsed = 0.0
    data: Dict[str, Any] = {}
    try:
        completed = subprocess.run(command, shell=True, check=True)
        returncode = completed.returncode
    except subprocess.CalledProcessError as exc:
        returncode = exc.returncode
        raise
    finally:
        emissions = tracker.stop() or 0.0
        elapsed = time.time() - started
        emissions_data = getattr(tracker, "final_emissions_data", None)
        if hasattr(emissions_data, "as_dict"):
            data = emissions_data.as_dict()
        else:
            data = {}
        data.setdefault("emissions_kg", emissions)
        data.setdefault("duration_sec", elapsed)
        data.setdefault("energy_consumed_kwh", data.get("energy_consumed"))
        data.setdefault("returncode", returncode)

    record = {
        "runtime_sec": round(elapsed, 3),
        "co2e_kg": round(emissions or 0.0, 6),
        "energy_kwh": round((data.get("energy_consumed_kwh") or 0.0), 6),
        "raw": data,
        "returncode": returncode,
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a command with CodeCarbon.")
    parser.add_argument("--region", required=True, help="Region key (e.g. hungary_transdanubia)")
    parser.add_argument("--label", default="baseline", help="Run label (baseline, optimized, etc.)")
    parser.add_argument(
        "--command",
        required=True,
        help="Command to execute; wrap in quotes if it has spaces",
    )
    parser.add_argument(
        "--output",
        help="Optional custom output path; defaults to models/<region>/greenai_runs.jsonl",
    )
    args = parser.parse_args()

    project = f"{args.region}-{args.label}"
    timestamp = datetime.now(tz=timezone.utc).isoformat()

    try:
        metrics = run_with_tracking(args.command, project_name=project)
    except subprocess.CalledProcessError as exc:
        metrics = {
            "runtime_sec": None,
            "co2e_kg": None,
            "energy_kwh": None,
            "raw": {"error": str(exc)},
            "returncode": exc.returncode,
        }

    entry = {
        "timestamp": timestamp,
        "region": args.region,
        "label": args.label,
        "command": args.command,
        "hardware": _default_hardware_metadata(),
        **metrics,
    }

    output_path = Path(args.output) if args.output else Path("models") / args.region / "greenai_runs.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    workspace = ensure_region_workspace(args.region)
    workspace_out = workspace / "logs" / output_path.name
    workspace_out.parent.mkdir(parents=True, exist_ok=True)
    with workspace_out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
