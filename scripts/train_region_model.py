#!/usr/bin/env python3
"""Train a region-specific Random Forest model using cached features."""
from __future__ import annotations

import argparse

from rf_training_lib import train_from_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, help="Region slug (e.g. jamaica_coffee)")
    parser.add_argument("--tier", type=int, default=1, help="Feature tier depth to use")
    parser.add_argument("--target", default="ndvi_zscore", help="Target column to predict")
    parser.add_argument("--freq", default="monthly", choices=["daily", "monthly"], help="Insight dataset frequency")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Holdout ratio for evaluation")
    parser.add_argument("--force-refresh", action="store_true", help="Regenerate feature cache before training")
    args = parser.parse_args()

    artifacts = train_from_cache(
        args.region,
        tier=args.tier,
        target=args.target,
        freq=args.freq,
        test_ratio=args.test_ratio,
        force_refresh=args.force_refresh,
    )

    print(f"✅ Model trained for {args.region} → {artifacts.model_path}")
    print(f"📊 Feature importances → {artifacts.feature_importances}")
    print(f"📈 Metrics → {artifacts.metrics_path}")
    print(f"ℹ️  Metrics summary: {artifacts.metrics}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
