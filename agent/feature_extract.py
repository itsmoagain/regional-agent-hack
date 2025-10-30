"""Feature extraction helpers for anomaly calculations."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def compute_spi(series: Iterable[float], window: int = 30) -> pd.Series:
    """Compute a simple Standardized Precipitation Index approximation."""

    s = pd.Series(series, dtype="float64")
    if s.empty:
        return s

    window = max(int(window), 1)
    accum = s.rolling(window, min_periods=max(1, window // 2)).sum()
    mean = accum.rolling(window, min_periods=1).mean()
    std = accum.rolling(window, min_periods=1).std(ddof=0)

    spi = (accum - mean) / std.replace(0, np.nan)
    spi = spi.replace([np.inf, -np.inf], np.nan)
    return spi.fillna(0.0)


__all__ = ["compute_spi"]
