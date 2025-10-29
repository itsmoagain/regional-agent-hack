#!/usr/bin/env python3
"""Unified entry point for the regional insights pipeline."""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Dependency helper
# ---------------------------------------------------------------------------

def require(pkg: str, import_name: Optional[str] = None):
    """Import *pkg*, installing it on-demand when permitted.

    Parameters
    ----------
    pkg:
        Name passed to ``pip install`` when the import fails.
    import_name:
        Module name to import via :func:`importlib.import_module`.  When not
        provided the value of ``pkg`` is used.

    Returns
    -------
    module or ``None``
        The imported module object.  ``None`` is returned when the dependency is
        missing and ``OFFLINE_MODE=1`` has been set in the environment.
    """

    module_name = import_name or pkg
    try:
        return importlib.import_module(module_name)
    except ImportError:
        if os.environ.get("OFFLINE_MODE") == "1":
            print(f"⚠️ Offline mode: '{pkg}' not installed, skipping auto-install.")
            return None

        print(f"📦 Installing missing dependency: {pkg}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except subprocess.CalledProcessError as exc:
            print(f"❌ Failed to install '{pkg}': {exc}")
            return None
        _log_auto_install(pkg)
        return importlib.import_module(module_name)


_RUNTIME_LOG = Path(__file__).resolve().parents[1] / ".runtime_log.txt"


def _log_auto_install(pkg: str) -> None:
    """Record auto-installed packages for observability."""

    try:
        timestamp = datetime.utcnow().isoformat()
        with _RUNTIME_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp}\t{pkg}\n")
    except Exception:
        # Logging should never block the pipeline – ignore filesystem errors.
        pass


# ---------------------------------------------------------------------------
# Pipeline orchestration helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent

# The core libraries required by downstream scripts.  The tuple structure is
# (package-name, import-name).
ESSENTIAL_PACKAGES: Tuple[Tuple[str, Optional[str]], ...] = (
    ("pyyaml", "yaml"),
    ("pandas", None),
    ("numpy", None),
    ("requests", None),
    ("folium", None),
    ("matplotlib", None),
    ("scikit-learn", None),
    ("joblib", None),
    ("earthengine-api", "ee"),
)


def bootstrap_dependencies(packages: Iterable[Tuple[str, Optional[str]]]) -> list[str]:
    """Ensure that all required dependencies are present.

    Returns a list of packages that are still missing (usually because the
    environment is offline).
    """

    missing: list[str] = []
    for pkg, import_name in packages:
        module = require(pkg, import_name)
        if module is None:
            missing.append(pkg)
    if missing:
        if os.environ.get("OFFLINE_MODE") == "1":
            print(
                "⚠️ Some dependencies could not be installed in offline mode: "
                + ", ".join(missing)
            )
        else:
            # ``require`` should have installed all dependencies already.  If we
            # still have missing entries it means pip failed and raised
            # ``ImportError`` again.  Surface a friendly error.
            raise RuntimeError(
                "Unable to import required packages: " + ", ".join(missing)
            )
    else:
        print("✅ All pipeline dependencies available.")
    return missing


def build_command(script: str, *args: str) -> Sequence[str]:
    """Construct a Python command that reuses the current interpreter."""

    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline script missing: {script_path}")
    return (sys.executable, str(script_path), *args)


def run_step(name: str, command: Sequence[str], allow_failure: bool = False) -> None:
    """Execute a pipeline step with helpful console output."""

    print(f"\n🚀 {name}")
    print("   " + " ".join(command))
    try:
        subprocess.check_call(command, cwd=str(ROOT))
        print(f"✅ {name} complete.")
    except subprocess.CalledProcessError as exc:
        if allow_failure:
            print(f"⚠️ {name} failed (allow-stale enabled): {exc}")
        else:
            raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the regional insights pipeline.")
    parser.add_argument("--region", help="Region slug to operate on (e.g. hungary_farmland)")
    parser.add_argument(
        "--mode",
        default="active",
        choices=["active", "cached"],
        help="Fetch mode forwarded to fetch_all.py (ignored in offline mode)",
    )
    parser.add_argument(
        "--ee-project",
        default=None,
        help="Optional Google Earth Engine project identifier",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Install all known dependencies and exit.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run in analysis-only mode (no new downloads; implies offline mode).",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="Continue even if intermediate steps fail (use with caution).",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip the dynamic data fetch step.",
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip rebuilding the region cache.",
    )
    parser.add_argument(
        "--skip-insights",
        action="store_true",
        help="Skip the insight generation phase.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.analyze:
        os.environ["OFFLINE_MODE"] = "1"

    missing = bootstrap_dependencies(ESSENTIAL_PACKAGES)
    if args.bootstrap:
        if missing and os.environ.get("OFFLINE_MODE") != "1":
            return 1
        return 0

    region = args.region or os.environ.get("REGION")
    if not region:
        print("❌ Region is required unless --bootstrap is provided.")
        return 2

    steps: list[Tuple[str, Sequence[str]]] = []
    offline = os.environ.get("OFFLINE_MODE") == "1"

    if not (args.skip_fetch or offline or args.analyze):
        steps.append(
            (
                "Fetch remote datasets",
                build_command(
                    "fetch_all.py",
                    "--region",
                    region,
                    "--mode",
                    args.mode,
                    *(["--ee-project", args.ee_project] if args.ee_project else []),
                ),
            )
        )

    if not args.skip_cache:
        steps.append(("Build region cache", build_command("build_region_cache.py", "--region", region)))

    if not args.skip_insights:
        steps.append(
            (
                "Generate region insights",
                build_command("build_region_insights.py", "--region", region),
            )
        )

    for name, command in steps:
        run_step(name, command, allow_failure=args.allow_stale)

    if steps:
        print("\n🎉 Pipeline finished.")
    else:
        print("ℹ️ No steps were executed (all phases skipped or offline analyze mode).")

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
