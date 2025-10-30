"""Feature cache management for Random Forest training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

try:
    import joblib
except ImportError:  # pragma: no cover - offline fallback
    joblib = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - offline fallback
    pd = None

from agent.features import build_features

try:
    from scripts import _shared as scripts_shared  # pragma: no cover
except ImportError:  # pragma: no cover - package usage
    scripts_shared = None

if scripts_shared is not None:
    ensure_region_workspace = scripts_shared.ensure_region_workspace
    get_region_current_dir = scripts_shared.get_region_current_dir
else:
    from regional_agent.config import ensure_region_workspace, get_region_current_dir  # type: ignore  # pragma: no cover


CACHE_NAME = "feature_cache.joblib"


@dataclass
class FeatureCache:
    features: pd.DataFrame
    target: Optional[pd.Series]
    metadata: dict


def _cache_path(region: str) -> Path:
    region_dir = get_region_current_dir(region)
    return region_dir / CACHE_NAME


def _build_insight_path(region: str, freq: str = "monthly") -> Path:
    """Return the expected path to the insight dataset; fall back gracefully if monthly missing."""
    data_dir = Path("data") / region / "current"
    candidates = [
        data_dir / "insights_monthly.csv",
        data_dir / "insight_monthly.csv",
        data_dir / "distilled_summary.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    # --- fallback for daily insights ---
    daily_fallback = data_dir / "insights_daily.csv"
    if daily_fallback.exists():
        print(f"⚙️ Fallback to daily insights → {daily_fallback}")
        return daily_fallback

    raise FileNotFoundError(
        f"No insight dataset found for {region}. "
        f"Checked: {', '.join(str(p) for p in candidates)} and {daily_fallback}"
    )


def load_feature_cache(
    region: str,
    *,
    tier: int = 1,
    target: str = "ndvi_zscore",
    freq: str = "monthly",
    force_refresh: bool = False,
) -> FeatureCache:
    """Return cached features/targets, rebuilding if necessary."""

    ensure_region_workspace(region)
    path = _cache_path(region)
    metadata = {
        "tier": int(tier),
        "target": target,
        "freq": freq,
    }

    if path.exists() and not force_refresh:
        if joblib is not None:
            try:
                payload: FeatureCache = joblib.load(path)
                if payload.metadata == metadata:
                    return payload
            except Exception:
                pass

    if pd is None:
        raise RuntimeError(
            "pandas is required for feature preprocessing; install dependencies or use offline fallback."
        )

    insight_file = _build_insight_path(region, freq)
    X, y = build_features(region, tier=tier, insight_file=insight_file, target=target)

    cache = FeatureCache(features=X, target=y, metadata=metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    if joblib is not None:
        joblib.dump(cache, path)
    return cache


__all__ = ["FeatureCache", "load_feature_cache"]
