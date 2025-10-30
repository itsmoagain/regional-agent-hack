"""Simple wrapper around deterministic rule evaluation."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from agent.rules.engine import CRITICAL_RULES, evaluate_rules_row, parse_rule_hit


def apply_rules(
    features_row: Mapping[str, Any] | Sequence[tuple[str, Any]],
    agentspec: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate rules and return structured recommendation hints."""

    if isinstance(features_row, Mapping):
        row_dict = dict(features_row)
    else:
        row_dict = {str(k): v for k, v in features_row}

    overrides = None
    if agentspec:
        overrides = agentspec.get("rule_overrides") or agentspec.get("rules")

    hits = evaluate_rules_row(row_dict, overrides)
    results: list[dict[str, Any]] = []
    for hit in hits:
        rid, message = parse_rule_hit(hit)
        results.append(
            {
                "id": rid,
                "message": message,
                "critical": rid in CRITICAL_RULES,
            }
        )
    return results


__all__ = ["apply_rules"]
