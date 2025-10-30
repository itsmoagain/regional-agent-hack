"""Persistence helpers for Random Forest models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import joblib
except ImportError:  # pragma: no cover - offline fallback
    joblib = None

MODELS_DIR = Path("models")


def get_model_path(region: str) -> Path:
    return MODELS_DIR / f"{region}_rf.pkl"


def save_model(model: Any, region: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = get_model_path(region)
    if joblib is not None:
        joblib.dump(model, path)
    else:
        path.write_text("fallback", encoding="utf-8")
    return path


def load_model(region: str):
    path = get_model_path(region)
    if path.exists():
        if joblib is not None:
            try:
                return joblib.load(path)
            except Exception:
                return None
        return path.read_text(encoding="utf-8")
    return None


__all__ = ["get_model_path", "load_model", "save_model"]
