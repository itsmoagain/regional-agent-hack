"""Lightweight engine utilities for the regional insight pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists():
    src_path = str(_SRC)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
diff --git a/engine/distill_insights.py b/engine/distill_insights.py
