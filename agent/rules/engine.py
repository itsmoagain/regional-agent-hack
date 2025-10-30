"""Rule evaluation primitives for deterministic insight tagging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import math


@dataclass
class RuleCondition:
    var: str
    op: str
    value: Any


@dataclass
class Rule:
    id: str
    label: str
    conditions_all: Sequence[RuleCondition]
    conditions_any: Sequence[RuleCondition]
    critical: bool = False


DEFAULT_RULES: list[Rule] = [
    Rule(
        id="drought",
        label="SPI indicates severe dryness",
        conditions_all=[RuleCondition("spi", "<", -1.5)],
        conditions_any=[],
        critical=True,
    ),
    Rule(
        id="vegetation_stress",
        label="NDVI anomaly signals canopy stress",
        conditions_all=[RuleCondition("ndvi_anomaly", "<", -0.2)],
        conditions_any=[],
    ),
    Rule(
        id="soil_dryness",
        label="Soil moisture below sustainable threshold",
        conditions_all=[RuleCondition("soil_surface_moisture", "<", 0.12)],
        conditions_any=[],
        critical=True,
    ),
    Rule(
        id="heat_extreme",
        label="High mean temperatures",
        conditions_all=[RuleCondition("temp_mean", ">", 35.0)],
        conditions_any=[],
    ),
]


CRITICAL_RULES = {rule.id for rule in DEFAULT_RULES if rule.critical}


def evaluate_rules_row(row: Mapping[str, Any], rule_overrides: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None) -> list[str]:
    """Evaluate deterministic rules against a *row* and return hits."""

    values = {str(k): row[k] for k in row.keys()}
    rules = _merge_rules(rule_overrides)

    hits: list[str] = []
    for rule in rules:
        if _rule_matches(rule, values):
            message = _format_rule_message(rule, values)
            hits.append(f"{rule.id}|{message}")
    return hits


def _merge_rules(rule_overrides: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None) -> list[Rule]:
    overrides: list[Rule] = []
    if rule_overrides:
        raw_rules: Iterable[Mapping[str, Any]]
        if isinstance(rule_overrides, Mapping) and "rules" in rule_overrides:
            raw_rules = rule_overrides["rules"]  # type: ignore[index]
        elif isinstance(rule_overrides, Mapping):
            raw_rules = [rule_overrides]
        else:
            raw_rules = rule_overrides  # type: ignore[assignment]

        for spec in raw_rules:
            try:
                overrides.append(_normalise_rule(spec))
            except Exception:
                continue

    merged: dict[str, Rule] = {rule.id: rule for rule in DEFAULT_RULES}
    for rule in overrides:
        merged[rule.id] = rule
    return list(merged.values())


def _normalise_rule(spec: Mapping[str, Any]) -> Rule:
    rid = str(spec.get("id"))
    label = str(spec.get("label", rid))
    critical = bool(spec.get("critical", False))
    conditions = spec.get("when", {})
    all_conditions = [_parse_condition(entry) for entry in conditions.get("all", [])]
    any_conditions = [_parse_condition(entry) for entry in conditions.get("any", [])]
    return Rule(rid, label, all_conditions, any_conditions, critical)


def _parse_condition(payload: Mapping[str, Any]) -> RuleCondition:
    return RuleCondition(
        var=str(payload.get("var")),
        op=str(payload.get("op", "<")),
        value=payload.get("value"),
    )


def _rule_matches(rule: Rule, row: Mapping[str, Any]) -> bool:
    if not _conditions_match(rule.conditions_all, row, require_all=True):
        return False
    if rule.conditions_any and not _conditions_match(rule.conditions_any, row, require_all=False):
        return False
    return True


def _conditions_match(conditions: Sequence[RuleCondition], row: Mapping[str, Any], *, require_all: bool) -> bool:
    if not conditions:
        return True
    results = [_condition_matches(cond, row) for cond in conditions]
    return all(results) if require_all else any(results)


def _condition_matches(cond: RuleCondition, row: Mapping[str, Any]) -> bool:
    value = row.get(cond.var)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    try:
        value = float(value)
    except Exception:
        return False

    target = cond.value
    if isinstance(target, (list, tuple)) and len(target) == 2 and cond.op == "between":
        lower, upper = float(target[0]), float(target[1])
        return lower <= value <= upper

    try:
        threshold = float(target)
    except Exception:
        threshold = None

    match cond.op:
        case "<":
            return threshold is not None and value < threshold
        case "<=":
            return threshold is not None and value <= threshold
        case ">":
            return threshold is not None and value > threshold
        case ">=":
            return threshold is not None and value >= threshold
        case "abs>":
            return threshold is not None and abs(value) > threshold
        case _:
            return False


def _format_rule_message(rule: Rule, row: Mapping[str, Any]) -> str:
    parts: list[str] = [rule.label]
    for cond in rule.conditions_all:
        if cond.var in row:
            parts.append(f"{cond.var}={row[cond.var]:.2f}")
    return "; ".join(parts)


def parse_rule_hit(hit: str) -> tuple[str, str]:
    if "|" in hit:
        rid, message = hit.split("|", 1)
        return rid, message
    return hit, hit


__all__ = [
    "CRITICAL_RULES",
    "DEFAULT_RULES",
    "evaluate_rules_row",
    "parse_rule_hit",
]
