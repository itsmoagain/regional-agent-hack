"""High-level CLI for the context-aware insight engine."""

from __future__ import annotations

import argparse
import importlib
import traceback
from datetime import datetime
from pathlib import Path
from typing import Sequence

from agent.insight_engine import build_insights
from engine.distill_insights import distill_region


LOG_FILE = Path("logs/monthly_check.log")


def check_rf_library_health(region: str = "test_region") -> None:
    """Monthly maintenance: verify RF library health and retraining ability."""

    print(f"\n🧠 Running rf_training_lib monthly check ({datetime.now().date()})...")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    status_message = "completed."
    success = True

    try:
        rf_lib = importlib.import_module("rf_training_lib")
        print("✅ rf_training_lib import successful.")
        if hasattr(rf_lib, "train_from_cache"):
            print("🧩 Found train_from_cache — testing retrain function...")
            rf_lib.train_from_cache(region)
            print(f"✅ Model retrained successfully for {region}.")
            status_message = f"success for {region}."
        else:
            print("⚠️ rf_training_lib missing train_from_cache method.")
            status_message = "missing train_from_cache method."
            success = False
    except Exception:
        print("🚨 rf_training_lib check failed:\n")
        traceback.print_exc()
        status_message = "failed (see console for traceback)."
        success = False
    finally:
        timestamp = datetime.now()
        with LOG_FILE.open("a", encoding="utf-8") as log_file:
            outcome = "SUCCESS" if success else "FAILURE"
            log_file.write(
                f"{timestamp} — rf_training_lib check {outcome}: {status_message}\n"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the regional insight engine pipeline.")
    parser.add_argument("--region", required=True, help="Region slug (e.g. jamaica_coffee)")
    parser.add_argument("--distill", action="store_true", help="Run the distillation stage only.")
    parser.add_argument(
        "--insight",
        action="store_true",
        help="Produce insight feed and alerts (runs distillation implicitly).",
    )
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="Run monthly rf_training_lib health check for the specified region.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    region = args.region

    distill_path: str | None = None
    feed_path: str | None = None
    alerts_path: str | None = None

    if args.distill or args.insight:
        distill_path = distill_region(region)
        print(f"✅ Distilled monthly summary → {distill_path}")

    if args.insight:
        feed_path, alerts_path = build_insights(region)
        print(f"✅ Insight feed ready → {feed_path}")
        if alerts_path:
            print(f"📣 Alerts issued → {alerts_path}")

    if getattr(args, "monthly", False):
        check_rf_library_health(region=args.region or "global")

    if not any([args.distill, args.insight, args.monthly]):
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
diff --git a/scripts/prep_kaggle_export.py b/scripts/prep_kaggle_export.py
