"""Lightweight retrieval-augmented generation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from scripts import _shared as scripts_shared  # pragma: no cover - optional path helper
except ImportError:  # pragma: no cover - package usage
    scripts_shared = None

if scripts_shared is not None:
    load_region_profile = scripts_shared.load_region_profile
else:
    from src.regional_agent.config import load_region_profile  # type: ignore  # pragma: no cover


CONTEXT_DIR = Path("data") / "context"


def retrieve_context(region: str, topics: Sequence[str] | None = None, limit: int = 5) -> list[str]:
    """Return contextual strings for *region* filtered by *topics*."""

    topics = [t.lower() for t in (topics or [])]
    snippets: list[str] = []

    profile = load_region_profile(region)
    meta = profile.get("region_meta", {})
    region_name = meta.get("name") or region.replace("_", " ").title()
    country = meta.get("country") or "Unknown"
    crops = meta.get("crops") or meta.get("dominant_crops") or []

    base_snippet = f"Region {region_name} in {country}. Dominant crops: {', '.join(map(str, crops)) or 'unspecified'}."
    snippets.append(base_snippet)

    context_dir = CONTEXT_DIR
    if context_dir.exists():
        for path in sorted(context_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            text = str(payload.get("text") or payload.get("summary") or "").strip()
            if not text:
                continue
            label = str(payload.get("label") or path.stem).lower()
            if topics and not any(topic in label for topic in topics):
                continue
            snippets.append(text)
            if len(snippets) >= limit:
                break

    return snippets[:limit]


def generate_recommendations(
    region: str,
    rule_ids: Iterable[str] | None = None,
    crop: str | None = None,
) -> list[str]:
    """Return agronomic recommendations keyed by rule identifiers."""

    rule_ids = [r.lower() for r in (rule_ids or [])]
    crop = (crop or "crop").lower()

    templates = {
        "drought": "Increase soil cover and evaluate supplemental irrigation scheduling.",
        "soil": "Incorporate organic matter or apply light irrigation to rebuild soil moisture.",
        "vegetation": "Inspect fields for pests and consider foliar feeding.",
        "heat": "Schedule irrigation during cooler hours and consider shade or windbreaks.",
    }

    recs: list[str] = []
    for rid in rule_ids:
        for key, message in templates.items():
            if key in rid:
                recs.append(message)
                break

    if not recs:
        context = retrieve_context(region, topics=[crop])
        if context:
            recs.append(context[0])

    return recs


__all__ = ["generate_recommendations", "retrieve_context"]
