#!/usr/bin/env python3
"""Convenient entry point for running the full regional pipeline.

Example
-------
    python scripts/run_pipeline.py --region hungary_farmland

This will:
    1. Initialize region folders and profile if missing.
    2. Fetch remote datasets (CHIRPS, NDVI, soil moisture, Open-Meteo).
    3. Build the merged daily/monthly cache.
    4. Compute regional insight features.
    5. Train the default Random Forest model.

Each stage can be skipped with the corresponding ``--skip-*`` flag.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Ensure the project root and scripts directory are importable when executed
# as ``python scripts/run_pipeline.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports (deferred until after sys.path adjustments)
from importlib import import_module
from subprocess import CalledProcessError


def _env_flag(name: str) -> bool:
    """Return True when an environment variable is truthy."""

    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_scripts():
    """Import pipeline modules lazily to avoid hard failures during ``--help``."""

    try:
        init_region = import_module("init_region").init_region
        fetch_all = import_module("fetch_all").main
        build_region_cache = import_module("build_region_cache").build_region_cache
        build_region_insights = import_module("build_region_insights").build_region_insights
        train_region_model = import_module("train_region_model").train_region_model
    except ModuleNotFoundError as exc:  # pragma: no cover - CLI path
        missing = exc.name or "a required module"
        raise SystemExit(
            f"Missing dependency '{missing}'. "
            "Ensure `pip install -r requirements.txt` (or the Kaggle environment) has been run "
            "before executing the pipeline."
        ) from exc

    return (
        init_region,
        fetch_all,
        build_region_cache,
        build_region_insights,
        train_region_model,
    )


def _step(
    label: str,
    func,
    *args,
    fail_fast: bool = True,
    strict_subprocess: bool = False,
    **kwargs,
) -> Dict[str, str]:
    """Run a pipeline step and capture status for later reporting."""

    start = datetime.utcnow()
    summary: Dict[str, str] = {
        "step": label,
        "started": start.isoformat() + "Z",
    }

    try:
        func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - lightweight CLI
        summary["status"] = "error"
        summary["message"] = str(exc)
        if fail_fast and not (
            isinstance(exc, CalledProcessError) and not strict_subprocess
        ):
            raise
    else:
        summary["status"] = "ok"
    finally:
        summary["finished"] = datetime.utcnow().isoformat() + "Z"

    return summary


def run_pipeline(
    *,
    region: str,
    fetch_mode: str = "active",
    ee_project: str | None = None,
    tier: int = 1,
    target: str = "ndvi_zscore",
    freq: str = "monthly",
    skip_fetch: bool = False,
    skip_cache: bool = False,
    skip_insights: bool = False,
    skip_train: bool = False,
    fail_fast: bool = True,
    strict_subprocess: bool = False,
) -> List[Dict[str, str]]:
    """Execute the requested steps for a region and return a status report."""

    (
        init_region,
        fetch_all,
        build_region_cache,
        build_region_insights,
        train_region_model,
    ) = _load_scripts()

    report: List[Dict[str, str]] = []

    # Always ensure the region workspace exists before other steps.
    report.append(
        _step(
            "init_region",
            init_region,
            region,
            fail_fast=fail_fast,
            strict_subprocess=strict_subprocess,
        )
    )

    if not skip_fetch:
        report.append(
            _step(
                "fetch_all",
                fetch_all,
                region,
                fetch_mode,
                ee_project,
                fail_fast=fail_fast,
                strict_subprocess=strict_subprocess,
            )
        )

    if not skip_cache:
        report.append(
            _step(
                "build_region_cache",
                build_region_cache,
                region,
                fail_fast=fail_fast,
                strict_subprocess=strict_subprocess,
            )
        )

    if not skip_insights:
        report.append(
            _step(
                "build_region_insights",
                build_region_insights,
                region,
                fail_fast=fail_fast,
                strict_subprocess=strict_subprocess,
            )
        )

    if not skip_train:
        report.append(
            _step(
                "train_region_model",
                train_region_model,
                region,
                tier,
                target,
                freq,
                fail_fast=fail_fast,
                strict_subprocess=strict_subprocess,
            )
        )

    return report


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full regional pipeline (fetch, cache, insights, train)."
    )
    parser.add_argument("--region", required=True, help="Region slug, e.g. hungary_farmland")
    parser.add_argument(
        "--mode",
        default="active",
        choices=["active", "cached"],
        help="Fetch mode to pass through to fetchers.",
    )
    parser.add_argument("--ee-project", default=None, help="Optional GEE project ID.")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3], help="Model tier")
    parser.add_argument(
        "--target",
        default="ndvi_zscore",
        help="Target variable for model training.",
    )
    parser.add_argument(
        "--freq",
        choices=["daily", "monthly"],
        default="monthly",
        help="Frequency for model training data.",
    )
    parser.add_argument("--skip-fetch", action="store_true", help="Skip dataset fetching.")
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip cache building (uses existing merged files).",
    )
    parser.add_argument(
        "--skip-insights",
        action="store_true",
        help="Skip insight generation (requires existing insights CSV).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training stage.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on first error instead of continuing remaining steps.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode (skip internet-dependent fetchers).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Re-raise subprocess failures (useful in CI).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path to write a JSON report summarizing each step.",
    )

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    offline_env = _env_flag("OFFLINE_MODE")
    offline_mode = offline_env or args.offline
    if offline_mode:
        print(
            "🌐 Skipping fetchers — running in offline analysis mode. "
            "Cached data will be used."
        )

    strict_subprocess = args.strict or (
        os.getenv("CI", "").strip().lower() == "true"
    )

    report = run_pipeline(
        region=args.region,
        fetch_mode=args.mode,
        ee_project=args.ee_project,
        tier=args.tier,
        target=args.target,
        freq=args.freq,
        skip_fetch=args.skip_fetch or offline_mode,
        skip_cache=args.skip_cache,
        skip_insights=args.skip_insights,
        skip_train=args.skip_train,
        fail_fast=args.fail_fast,
        strict_subprocess=strict_subprocess,
    )

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        print(f"📝 Report written to {report_path}")

    print("\nPipeline summary:")
    for step in report:
        status = step.get("status", "ok")
        msg = step.get("message", "")
        print(f" • {step['step']}: {status}" + (f" — {msg}" if msg else ""))


if __name__ == "__main__":
    main()
