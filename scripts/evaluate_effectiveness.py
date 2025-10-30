from __future__ import annotations

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#!/usr/bin/env python3
"""Evaluate the latest Random Forest model for a region."""

import argparse
import json
from pathlib import Path

from rf_training_lib import evaluate_random_forest, load_feature_cache, load_or_train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, help="Region slug (e.g. jamaica_coffee)")
    parser.add_argument("--tier", type=int, default=1, help="Feature tier depth used during training")
    parser.add_argument("--target", default="ndvi_zscore", help="Target column to evaluate")
    parser.add_argument("--freq", default="monthly", choices=["daily", "monthly"], help="Insight dataset frequency")
    parser.add_argument("--force-refresh", action="store_true", help="Recompute feature cache before evaluation")
    args = parser.parse_args()

    cache = load_feature_cache(
        args.region,
        tier=args.tier,
        target=args.target,
        freq=args.freq,
        force_refresh=args.force_refresh,
    )
    if cache.target is None or cache.features.empty:
        raise ValueError("No labelled data available for evaluation.")

    model = load_or_train_model(
        args.region,
        tier=args.tier,
        target=args.target,
        freq=args.freq,
    )

    metrics = evaluate_random_forest(model, cache.features, cache.target)
    output_dir = Path("outputs") / args.region
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model_evaluation.json"
    payload = {
        "region": args.region,
        "tier": args.tier,
        "target": args.target,
        "freq": args.freq,
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2))

    print(f"✅ Evaluation complete for {args.region} → {path}")
    print(f"📊 Metrics: {metrics}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
