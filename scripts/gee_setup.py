#!/usr/bin/env python3
"""
gee_setup.py — Ensure Google Earth Engine (GEE) is installed & authenticated.

Usage:
    from scripts.gee_setup import ensure_gee_ready
    ensure_gee_ready(project="your-gcp-project-id")  # project optional
"""

from __future__ import annotations
import os
import sys
import json
import subprocess
from typing import Optional

def _pip_install(pkg: str) -> None:
    """Install or upgrade a package inside the current venv."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg])

def _print_box(title: str, lines: list[str]) -> None:
    """Pretty console box output."""
    print("\n" + "═" * 70)
    print(f"🔧 {title}")
    print("─" * 70)
    for ln in lines:
        print(ln)
    print("═" * 70 + "\n")

def ensure_gee_ready(project: Optional[str] = None) -> bool:
    """
    Ensures:
      - earthengine-api is installed
      - user/service auth exists
      - ee.Initialize() succeeds (optionally with project)
    Returns True if initialized; False otherwise.
    """
    # 1) Install core dependencies
    _pip_install("packaging")   # ✅ ensure packaging is available
    try:
        import ee  # noqa: F401
    except Exception:
        _print_box("Installing earthengine-api", [
            "Installing Google Earth Engine Python client into your current venv...",
        ])
        _pip_install("earthengine-api")
        _pip_install("packaging")   # ✅ install packaging again after earthengine
        import ee  # type: ignore  # noqa: F401

    # 2) Try initializing Earth Engine
    import ee  # type: ignore

    def _try_init(tag: str) -> bool:
        try:
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
            print(f"✅ Earth Engine initialized ({tag}).")
            return True
        except Exception as e:
            print(f"⚠️ EE init failed ({tag}): {e}")
            return False

    # a) If Application Default Credentials or Service Account exist
    adc_hint = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if adc_hint and os.path.isfile(adc_hint):
        if _try_init("ADC / service account"):
            return True

    # b) Try existing user credentials
    if _try_init("existing user creds"):
        return True

    # 3) Perform user authentication (browser flow)
    _print_box("Earth Engine authentication required", [
        "You need a Google Earth Engine account and to complete a one-time sign-in.",
        "",
        "If you haven't **signed up** for GEE yet, do this first:",
        "  👉 https://signup.earthengine.google.com/",
        "",
        "Then we'll launch the auth flow. If a browser doesn't open automatically,",
        "copy-paste the URL shown in the terminal into your browser.",
    ])

    try:
        subprocess.check_call(["earthengine", "authenticate"], env=os.environ)
    except FileNotFoundError:
        subprocess.check_call([sys.executable, "-m", "ee", "authenticate"], env=os.environ)

    # 4) Retry initialization after successful authentication
    if _try_init("post-auth"):
        if project:
            _try_init(f"post-auth with project={project}")
        return True

    _print_box("Earth Engine initialization still failing", [
        "Common fixes:",
        " • Ensure you completed the sign-in in the browser and returned to the terminal.",
        " • If using a service account, set GOOGLE_APPLICATION_CREDENTIALS to your JSON key.",
        " • Some orgs require enabling Earth Engine API in GCP project:",
        "     https://console.cloud.google.com/earth-engine",
        f" • Project used (optional): {project or '(none)'}",
    ])
    return False


if __name__ == "__main__":
    ok = ensure_gee_ready(project=os.environ.get("EE_PROJECT"))
    sys.exit(0 if ok else 1)
