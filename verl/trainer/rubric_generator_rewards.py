"""Pure reward accounting for on-policy rubric evolution proposals.

A completion is the action, not each accepted criterion. Invalid, duplicate,
and over-budget slots stay in the denominator. Vertical targets are averaged
equally, including explicitly empty targets. No aggregate Solver-variance gate
or LLM-estimated discriminability is used here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


GAP_ROLE = "gap"
VERTICAL_ROLE = "vertical"


def criterion_text_key(item: Mapping[str, Any]) -> str:
    """Exact textual dedup independent of title and weight (semantic audit is separate)."""
    return " ".join(str(item.get("description", "")).split()).casefold()


@dataclass
class CandidateAssessment:
    target_id: str
    slot: int
    candidate: dict[str, Any] | None
    semantic: dict[str, Any] = field(default_factory=dict)
    empirical: dict[str, Any] = field(default_factory=dict)
    rejection: str = ""
    meta_payload: str = ""

    @property
    def valid(self) -> bool:
        return self.candidate is not None and not self.rejection and bool(self.semantic.get("valid", False))

    @property
    def accepted(self) -> bool:
        return self.valid and bool(self.empirical.get("passed", False))


@dataclass
class ProposalAssessment:
    target_ids: tuple[str, ...]
    candidates: list[CandidateAssessment]
    raw_counts: dict[str, int]
    budgets: dict[str, int]
    format_valid: bool = True
    error: str = ""
    audit_error: str = ""


def parse_rg_proposal(
    text: str,
    *,
    role: str,
    operations: Mapping[str, str],
    budgets: Mapping[str, int],
    weights: Mapping[str, float],
    existing_rubric: Sequence[Mapping[str, Any]],
) -> ProposalAssessment:
    """Parse the current GAP/vertical prompt contracts without silently dropping slots.

    Oversized outputs are counted in full but only the first budgeted slots are
    audited/executed. Peers means other criteria in THIS completion, never other
    sampled completions. Thus candidate evaluation order cannot affect reward.
    """
    targets = tuple(operations)
    result = ProposalAssessment(targets, [], {key: 0 for key in targets}, dict(budgets))
    if not targets or set(budgets) != set(targets) or set(weights) != set(targets):
        raise ValueError("RG target operations, budgets and weights must match and be nonempty.")
    if any(value < 1 for value in budgets.values()):
        raise ValueError("RG per-target budgets must be positive.")
    try:
        payload = json.loads(text)
        if role == GAP_ROLE:
            if targets != ("gap",) or not isinstance(payload, dict):
                raise ValueError("GAP requires one gap target and a JSON object.")
            if set(payload) != {"operation", "criteria"} or payload["operation"] != "gap_expand":
                raise ValueError("Invalid GAP routing/schema.")
            arrays = {"gap": payload["criteria"]}
        elif role == VERTICAL_ROLE:
            if not isinstance(payload, dict) or set(payload) != {"targets"}:
                raise ValueError("Vertical proposal requires a targets array.")
            rows = payload["targets"]
            if not isinstance(rows, list) or [row.get("parent_id") for row in rows if isinstance(row, dict)] != list(targets):
                raise ValueError("Missing, duplicate, reordered or unknown parent IDs.")
            arrays = {}
            for row in rows:
                key = row["parent_id"]
                if set(row) != {"parent_id", "operation", "candidates"} or row["operation"] != operations[key]:
                    raise ValueError("Invalid vertical target operation/schema.")
                arrays[key] = row["candidates"]
        else:
            raise ValueError(f"Unknown RG role: {role}")
        if any(not isinstance(value, list) for value in arrays.values()):
            raise ValueError("Candidate entries must be arrays.")
    except (ValueError, TypeError, KeyError) as exc:
        result.format_valid = False
        result.error = str(exc)
        return result

    seen = {criterion_text_key(item) for item in existing_rubric}
    for target_id in targets:
        rows = arrays[target_id]
        result.raw_counts[target_id] = len(rows)
        for slot, raw in enumerate(rows):
            assessment = CandidateAssessment(target_id, slot, None)
            result.candidates.append(assessment)
            if slot >= budgets[target_id]:
                assessment.rejection = "over_budget"
                continue
            # GAP uses the already validated envelope operation. Keep accepting
            # legacy repeated fields, but reject explicit conflicting values.
            required = ({"title", "description", "weight"} if role == GAP_ROLE
                        else {"title", "description", "operation"})
            allowed = required | {"operation"}
            if (not isinstance(raw, dict) or not required <= set(raw) <= allowed
                    or raw.get("operation", operations[target_id]) != operations[target_id]
                    or not all(isinstance(raw.get(key), str) and raw[key].strip() for key in ("title", "description"))):
                assessment.rejection = "invalid_criterion_schema"
                continue
            weight = weights[target_id]
            if role == GAP_ROLE:
                if type(raw["weight"]) not in (int, float) or raw["weight"] not in (-1, 1):
                    assessment.rejection = "invalid_polarity"
                    continue
                weight = abs(weight) * raw["weight"]
            item = {"title": raw["title"].strip(), "description": raw["description"].strip(),
                    "operation": operations[target_id], "weight": float(weight)}
            key = criterion_text_key(item)
            if key in seen:
                assessment.rejection = "duplicate_criterion"
                continue
            seen.add(key)
            assessment.candidate = item
    return result


def score_rg_proposal(
    proposal: ProposalAssessment,
    *,
    output_tokens: int,
    token_budget: int,
    invalid_reward: float = -1.0,
) -> dict[str, float]:
    """V + D - P using the paper's minimum-support eligibility reward.

    `r_meta = V` is the Meta-Verifier validity ratio in [0, 1], standing in
    for the proportion of structurally and semantically valid criteria.
    `r_disc = D` is the fraction passing both semantic and empirical checks.
    All eligible splits receive the same credit; a 50/50 split is not preferred.
    `P = p_num + p_len` is the structural budget penalty (count + token).
    Both V and D carry equal weight one; `discrimination` is reported as the
    raw unscaled D so split quality stays readable as a separate metric.
    """
    if proposal.audit_error:
        raise ValueError("Unavailable audit must be excluded before assigning RG reward")
    if token_budget < 1 or output_tokens < 0:
        raise ValueError("Invalid RG token budget/count.")
    if not proposal.format_valid:
        return {"reward": invalid_reward, "validity": 0.0, "discrimination": 0.0,
                "budget_penalty": 0.0, "format_valid": 0.0, "accepted": 0.0}
    validity, discrimination, count_penalties = [], [], []
    for target in proposal.target_ids:
        denominator = max(1, proposal.raw_counts[target])
        v_sum, d_sum = 0.0, 0.0
        for item in proposal.candidates:
            if item.target_id != target or not item.valid:
                continue
            v_sum += 1.0
            if item.accepted:
                p = float(item.empirical["pass_rate"])
                if not math.isfinite(p) or not 0.0 <= p <= 1.0:
                    raise ValueError("Non-finite or out-of-range empirical pass rate.")
                d_sum += 1.0
        validity.append(v_sum / denominator)
        discrimination.append(d_sum / denominator)
        count_penalties.append(max(0, proposal.raw_counts[target] - proposal.budgets[target]) / proposal.budgets[target])
    v = sum(validity) / len(validity)
    d = sum(discrimination) / len(discrimination)
    penalty = sum(count_penalties) / len(count_penalties) + max(0, output_tokens - token_budget) / token_budget
    return {"reward": v + d - penalty,
            "validity": v, "discrimination": d, "budget_penalty": penalty,
            "format_valid": 1.0, "accepted": float(sum(item.accepted for item in proposal.candidates))}


def group_relative_advantages(rewards: Sequence[float], group_ids: Sequence[str], *, expected_n: int) -> list[float]:
    """Normalize each exact input group independently; tied groups get zero PG signal."""
    if len(rewards) != len(group_ids) or expected_n < 2:
        raise ValueError("RG rewards/groups must align and expected_n must be >= 2.")
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(group_ids):
        if not math.isfinite(float(rewards[index])):
            raise ValueError("RG rewards must be finite.")
        groups.setdefault(key, []).append(index)
    advantages = [0.0] * len(rewards)
    for indices in groups.values():
        if len(indices) != expected_n:
            raise ValueError("Incomplete RG group; do not train a filtered subset of completions.")
        mean = sum(rewards[index] for index in indices) / len(indices)
        std = math.sqrt(sum((rewards[index] - mean) ** 2 for index in indices) / len(indices))
        for index in indices:
            advantages[index] = (rewards[index] - mean) / (std + 1e-6) if std > 1e-8 else 0.0
    return advantages
