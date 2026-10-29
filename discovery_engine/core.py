"""Transparent evidence triage primitives; not a truth oracle."""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Iterable

VALID_STANCES = {"support", "refute", "neutral"}

def _validate_evidence(evidence: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    items = list(evidence)
    for index, item in enumerate(items):
        if item.get("stance") not in VALID_STANCES:
            raise ValueError(f"evidence[{index}].stance must be one of {sorted(VALID_STANCES)}")
        if not item.get("hypothesis_id"):
            raise ValueError(f"evidence[{index}] is missing hypothesis_id")
    return items

def analyze_hypotheses(hypotheses: Iterable[dict[str, Any]], evidence: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize explicitly supplied evidence. Scores are not probabilities."""
    hypothesis_list = list(hypotheses)
    evidence_list = _validate_evidence(evidence)
    ids = [item.get("id") for item in hypothesis_list]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Each hypothesis needs a unique, non-empty id")
    unknown = sorted({e["hypothesis_id"] for e in evidence_list} - set(ids))
    if unknown:
        raise ValueError(f"Evidence references unknown hypothesis ids: {unknown}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in evidence_list:
        grouped[item["hypothesis_id"]].append(item)
    results = []
    for hypothesis in hypothesis_list:
        records = grouped[hypothesis["id"]]
        support = sum(row["stance"] == "support" for row in records)
        refute = sum(row["stance"] == "refute" for row in records)
        neutral = sum(row["stance"] == "neutral" for row in records)
        decisive = support + refute
        score = (support - refute) / decisive if decisive else None
        if decisive == 0:
            status = "insufficient_evidence"
        elif support == refute:
            status = "contested"
        elif support > refute:
            status = "more_support_than_refutation"
        else:
            status = "more_refutation_than_support"
        results.append({
            "id": hypothesis["id"], "claim": hypothesis["claim"],
            "support_count": support, "refute_count": refute, "neutral_count": neutral,
            "evidence_count": len(records), "descriptive_score": score,
            "status": status, "evidence": records,
        })
    return sorted(results, key=lambda row: (
        row["descriptive_score"] is None,
        -(row["descriptive_score"] if row["descriptive_score"] is not None else 0),
        row["id"],
    ))

def find_contradictions(observations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find opposite explicit stances sharing the same caller-normalized claim key."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, item in enumerate(observations):
        if not item.get("claim_key"):
            raise ValueError(f"observations[{index}] is missing claim_key")
        if item.get("stance") not in {"support", "refute"}:
            raise ValueError(f"observations[{index}].stance must be support or refute")
        grouped[item["claim_key"]].append(item)
    conflicts = []
    for claim_key, records in sorted(grouped.items()):
        supports = [row for row in records if row["stance"] == "support"]
        refutations = [row for row in records if row["stance"] == "refute"]
        if supports and refutations:
            conflicts.append({
                "claim_key": claim_key, "supporting_observations": supports,
                "refuting_observations": refutations, "status": "unresolved_contradiction",
            })
    return conflicts

def suggest_next_test(analysis: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Triage hypotheses with no decisive evidence or balanced evidence."""
    rows = list(analysis)
    if not rows:
        return None
    def priority(row: dict[str, Any]) -> tuple[float, int, str]:
        support, refute = row.get("support_count", 0), row.get("refute_count", 0)
        decisive = support + refute
        uncertainty = 1.0 if decisive == 0 else 1.0 - abs(support - refute) / decisive
        return uncertainty, -row.get("evidence_count", 0), row.get("id", "")
    chosen = max(rows, key=priority)
    no_decisive = chosen.get("support_count", 0) + chosen.get("refute_count", 0) == 0
    return {
        "hypothesis_id": chosen["id"], "claim": chosen["claim"],
        "reason": "No decisive evidence is recorded." if no_decisive else "Support and refutation are comparatively balanced.",
        "suggestion": "Define a falsifiable test that could produce different observations under competing explanations.",
        "caution": "Review feasibility, safety, and evidence quality before running the test.",
    }
