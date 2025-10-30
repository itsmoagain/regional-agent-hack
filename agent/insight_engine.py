"""Orchestrates distillation, rule tagging, and insight text generation."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from engine.distill_insights import distill_region
from engine.utils.metadata import load_phenology_hints, load_region_metadata
from agent.rules.engine import CRITICAL_RULES, evaluate_rules_row, parse_rule_hit

OUTPUT_DIR = Path("outputs")
TEMPLATE_PATH = Path("data/context/insight_templates.json")


def build_insights(region: str) -> tuple[str, str | None]:
    summary_path = Path(distill_region(region))
    df = pd.read_csv(summary_path)

    metadata = load_region_metadata(region)
    phenology = load_phenology_hints(region, metadata)
    templates = _load_templates()

    crops = metadata.get("dominant_crops") or ["mixed cropping"]
    crop_type = str(crops[0])
    rule_overrides = metadata.get("rule_overrides")
    critical_rules = set(CRITICAL_RULES)
    if isinstance(rule_overrides, Sequence):
        critical_rules.update(
            rule.get("id")
            for rule in rule_overrides
            if isinstance(rule, Mapping) and rule.get("critical")
        )

    records = []
    alerts: list[str] = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        hits = evaluate_rules_row(row_dict, rule_overrides)
        rule_ids: list[str] = []
        rule_messages: list[str] = []
        for hit in hits:
            rid, message = parse_rule_hit(hit)
            rule_ids.append(rid)
            rule_messages.append(message)
            if rid in critical_rules:
                alerts.append(_format_alert(region, rid, message, row_dict))

        recommendation = _maybe_fetch_practices(
            crop_type,
            rule_ids,
            row_dict,
            metadata,
            phenology,
        )
        insight_text = _render_insight(
            row_dict,
            metadata,
            crop_type,
            rule_messages,
            recommendation,
            templates,
            phenology,
        )

        records.append(
            {
                "month": str(row_dict.get("month")),
                "crop_type": crop_type,
                "region_name": metadata.get("region_name"),
                "spi": row_dict.get("spi"),
                "ndvi_anomaly": row_dict.get("ndvi_anomaly"),
                "soil_surface_moisture": row_dict.get("soil_surface_moisture"),
                "temp_mean": row_dict.get("temp_mean"),
                "rule_hits": ",".join(rule_ids),
                "model_signal": row_dict.get("model_signal"),
                "insight_text": insight_text,
            }
        )

    feed_df = pd.DataFrame.from_records(records)
    output_dir = OUTPUT_DIR / region
    output_dir.mkdir(parents=True, exist_ok=True)
    feed_path = output_dir / "insight_feed.csv"
    feed_df.to_csv(feed_path, index=False)

    alerts_path: Path | None = None
    if alerts:
        alerts_path = output_dir / "alerts.txt"
        unique_alerts = _unique_preserve_order(alerts)
        alerts_path.write_text("\n".join(unique_alerts) + "\n")

    return str(feed_path), str(alerts_path) if alerts_path else None


def _render_insight(
    row: Mapping[str, float],
    metadata: Mapping[str, object],
    crop_type: str,
    rule_messages: Sequence[str],
    practice_recommendations: Sequence[str],
    templates: Mapping[str, str],
    phenology: Mapping[str, object],
) -> str:
    template = templates.get("base") or (
        "{region_name} ({crop_type}) — {month}: SPI {spi}, NDVI anomaly {ndvi}, "
        "soil moisture {soil}, temp {temp}. {rule_summary} {model_desc}{practice}"
    )

    spi = _format_metric(row.get("spi"))
    ndvi = _format_metric(row.get("ndvi_anomaly"))
    soil = _format_metric(row.get("soil_surface_moisture"))
    temp = _format_metric(row.get("temp_mean"))
    rule_summary = _format_rule_summary(rule_messages)
    model_desc = _describe_model_signal(row.get("model_signal"))
    practice = "" if not practice_recommendations else " " + " ".join(practice_recommendations)
    month = str(row.get("month"))
    stage = _phenology_stage(month, phenology)

    return template.format(
        region_name=metadata.get("region_name", "Unknown region"),
        crop_type=crop_type,
        month=month,
        spi=spi,
        ndvi=ndvi,
        soil=soil,
        temp=temp,
        rule_summary=rule_summary,
        model_desc=model_desc,
        practice=practice,
        phenology_stage=stage,
    ).strip()


def _format_metric(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value:.2f}"


def _format_rule_summary(messages: Sequence[str]) -> str:
    if not messages:
        return "No major rule triggers this month."
    return "Triggers: " + "; ".join(messages)


def _describe_model_signal(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Model signal unavailable."
    if value >= 0.75:
        level = "high risk"
    elif value >= 0.5:
        level = "moderate risk"
    else:
        level = "low risk"
    return f"Model signal {level} ({value:.2f})."


def _phenology_stage(month: str, phenology: Mapping[str, object]) -> str:
    stage_map = phenology.get("stage_by_month") if isinstance(phenology, Mapping) else None
    if isinstance(stage_map, Mapping):
        key = str(month)[-2:]
        stage = stage_map.get(key)
        if stage:
            return str(stage)
    return str(phenology.get("current_stage", "")) if phenology else ""


def _unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _format_alert(region: str, rule_id: str, message: str, row: Mapping[str, float]) -> str:
    prefix = f"[{region}] {rule_id}:"
    summary_parts = [prefix, message]
    spi = row.get("spi")
    if spi is not None and not pd.isna(spi):
        summary_parts.append(f"SPI={spi:.2f}")
    ndvi = row.get("ndvi_anomaly")
    if ndvi is not None and not pd.isna(ndvi):
        summary_parts.append(f"NDVI={ndvi:.2f}")
    return " ".join(summary_parts)


class PracticeLibraryAdapter:
    _callable = None
    _loaded = False
    _warned = False

    @classmethod
    def fetch(
        cls,
        crop: str,
        rule_ids: Sequence[str],
        row: Mapping[str, float],
        metadata: Mapping[str, object],
        phenology: Mapping[str, object],
    ) -> list[str]:
        func = cls._resolve_callable()
        if func is None:
            return []
        anomaly_type = rule_ids[0] if rule_ids else "baseline"
        kwargs = {
            "crop": crop,
            "anomaly_type": anomaly_type,
            "spi": row.get("spi"),
            "ndvi_anomaly": row.get("ndvi_anomaly"),
            "soil_type": metadata.get("soil_type"),
            "stage": _phenology_stage(str(row.get("month")), phenology) or None,
        }
        try:
            recs = func(**kwargs)
        except Exception:
            return []
        if isinstance(recs, str):
            return [recs]
        return [str(rec) for rec in recs or []]

    @classmethod
    def _resolve_callable(cls):
        if cls._loaded:
            return cls._callable
        cls._loaded = True
        try:
            module = importlib.import_module("practice_library")
        except Exception:
            url = os.getenv("PRACTICE_LIB_URL")
            if url and not cls._warned:
                cls._warned = True
                print(
                    "⚠️ PRACTICE_LIB_URL is set but the optional practice_library package "
                    "is unavailable; falling back to templates."
                )
            cls._callable = None
            return None
        func = getattr(module, "get_recommendations", None)
        if callable(func):
            cls._callable = func
        else:
            cls._callable = None
        return cls._callable


def _maybe_fetch_practices(
    crop: str,
    rule_ids: Sequence[str],
    row: Mapping[str, float],
    metadata: Mapping[str, object],
    phenology: Mapping[str, object],
) -> list[str]:
    return PracticeLibraryAdapter.fetch(crop, rule_ids, row, metadata, phenology)


def _load_templates() -> Mapping[str, str]:
    if TEMPLATE_PATH.exists():
        try:
            return json.loads(TEMPLATE_PATH.read_text())
        except Exception:
            pass
    return {
        "base": (
            "{region_name} ({crop_type}) — {month}: SPI {spi}, NDVI {ndvi}, "
            "soil moisture {soil}, temperature {temp}. {rule_summary} {model_desc}{practice}"
        )
    }


__all__ = ["build_insights"]
diff --git a/agent/rules/engine.py b/agent/rules/engine.py
