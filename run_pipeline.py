"""High-level CLI for the context-aware insight engine."""

from __future__ import annotations

import argparse
from typing import Sequence

from agent.insight_engine import build_insights
from engine.distill_insights import distill_region


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the regional insight engine pipeline.")
    parser.add_argument("--region", required=True, help="Region slug (e.g. jamaica_coffee)")
    parser.add_argument("--distill", action="store_true", help="Run the distillation stage only.")
    parser.add_argument(
        "--insight",
        action="store_true",
        help="Produce insight feed and alerts (runs distillation implicitly).",
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

    if not any([args.distill, args.insight]):
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
diff --git a/scripts/prep_kaggle_export.py b/scripts/prep_kaggle_export.py
