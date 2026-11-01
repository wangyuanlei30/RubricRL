# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Policy-aware rubric evolution for Rubric-Guided RLSD."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ..protocol import DataProto
from .opsd_config import OPSDConfig
from .rubric_empirical_batch import CandidateJudgeGroup, judge_candidate_groups
from .rubric_meta_verifier import CHILD_EVOLUTION, parse_meta_verifier_verdict
from .rubric_weight_allocation import equal_family_weight

CRITERION_STATE_DEAD = "dead"
CRITERION_STATE_MIXED = "mixed"
CRITERION_STATE_SATURATED = "saturated"
CRITERION_POLARITY_POSITIVE = "positive"
CRITERION_POLARITY_NEGATIVE = "negative"

CANDIDATE_PENDING = "pending"
CANDIDATE_ACTIVE = "active"
CANDIDATE_REJECTED = "rejected"

META_VERIFIER_DIMENSIONS = (
    "target_relevance",
    "polarity_consistency",
    "factual_grounding",
    "measurability",
    "novelty_and_non_duplication",
    "specificity_and_non_triviality",
    "style_relevance",
    "rollout_generalization",
    "atomicity",
)

_META_DIMENSION_REJECTION_TYPES = {
    "target_relevance": "invalid_relation",
    "polarity_consistency": "polarity_mismatch",
    "factual_grounding": "unsupported_fact",
    "measurability": "not_judgeable",
    "novelty_and_non_duplication": "redundant",
    "specificity_and_non_triviality": "trivial",
    "style_relevance": "length_style",
    "rollout_generalization": "rollout_overfit",
    "atomicity": "invalid_relation",
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truncate_for_log(text: Any, max_len: int = 400) -> str:
    value = str(text or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def find_unsupported_numeric_claims(candidate_text: str, source_text: str) -> list[str]:
    """Return numeric claims introduced outside the question/selected parent.

    In medical rubrics, an invented stricter dose or frequency is not a safe
    refinement.  This deterministic gate complements (rather than trusts) the
    policy's factual-grounding self-report.
    """
    number_pattern = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
    source_numbers = set(number_pattern.findall(str(source_text or "")))
    candidate_numbers = number_pattern.findall(str(candidate_text or ""))
    return list(
        dict.fromkeys(
            number for number in candidate_numbers if number not in source_numbers
        )
    )


def _normalized_evidence_text(text: Any) -> str:
    return " ".join(str(text or "").split()).casefold()


def validate_proposal_evidence(
    candidate: dict[str, Any],
    rollout_responses: list[str],
    hit_vector: list[int],
) -> dict[str, Any]:
    """Validate that proposer citations refer to the scored rollout split.

    The proposer is asked for target-local positive and negative evidence, but a
    prompt instruction alone is not a verification gate.  Rollout ids must be
    in range, positive quotes must occur verbatim (modulo whitespace/case), and
    the cited sides must agree with the empirical candidate judge.
    """

    positive = candidate.get("positive_evidence")
    negative = candidate.get("negative_evidence")
    if not isinstance(positive, list) or not positive:
        return {"valid": False, "reason": "missing_positive_evidence"}
    if not isinstance(negative, list) or not negative:
        return {"valid": False, "reason": "missing_negative_evidence"}
    if len(rollout_responses) != len(hit_vector):
        return {"valid": False, "reason": "evidence_score_length_mismatch"}

    positive_ids: list[int] = []
    negative_ids: list[int] = []
    for evidence in positive:
        if not isinstance(evidence, dict):
            return {"valid": False, "reason": "invalid_positive_evidence_item"}
        try:
            rollout_id = int(evidence.get("rollout_id"))
        except (TypeError, ValueError):
            return {"valid": False, "reason": "invalid_positive_rollout_id"}
        if rollout_id < 1 or rollout_id > len(rollout_responses):
            return {"valid": False, "reason": "positive_rollout_id_out_of_range"}
        quote = _normalized_evidence_text(evidence.get("quote"))
        response = _normalized_evidence_text(rollout_responses[rollout_id - 1])
        if not quote or quote not in response:
            return {"valid": False, "reason": "positive_quote_not_in_rollout"}
        if hit_vector[rollout_id - 1] != 1:
            return {"valid": False, "reason": "positive_evidence_disagrees_with_judge"}
        positive_ids.append(rollout_id)

    for evidence in negative:
        if not isinstance(evidence, dict):
            return {"valid": False, "reason": "invalid_negative_evidence_item"}
        try:
            rollout_id = int(evidence.get("rollout_id"))
        except (TypeError, ValueError):
            return {"valid": False, "reason": "invalid_negative_rollout_id"}
        if rollout_id < 1 or rollout_id > len(rollout_responses):
            return {"valid": False, "reason": "negative_rollout_id_out_of_range"}
        if hit_vector[rollout_id - 1] != 0:
            return {"valid": False, "reason": "negative_evidence_disagrees_with_judge"}
        negative_ids.append(rollout_id)

    if set(positive_ids) & set(negative_ids):
        return {"valid": False, "reason": "same_rollout_cited_on_both_sides"}
    return {
        "valid": True,
        "reason": "evidence_matches_empirical_split",
        "positive_rollout_ids": positive_ids,
        "negative_rollout_ids": negative_ids,
    }


def normalize_rubric_items(
    raw_rubric: Any, raw_rubric_list: Any
) -> list[dict[str, Any]]:
    if isinstance(raw_rubric, np.ndarray):
        raw_rubric = raw_rubric.tolist()
    if isinstance(raw_rubric, list) and raw_rubric:
        items = []
        for idx, item in enumerate(raw_rubric):
            if isinstance(item, dict):
                items.append(
                    {
                        "title": str(item.get("title") or f"Criterion {idx}"),
                        "description": str(item.get("description") or ""),
                        "weight": _as_float(item.get("weight", 1.0), default=1.0),
                        "criterion_id": str(
                            item.get("criterion_id") or f"parent_{idx}"
                        ),
                        "parent_id": str(item.get("parent_id") or ""),
                        "source": str(item.get("source") or "parent"),
                    }
                )
            else:
                items.append(
                    {
                        "title": f"Criterion {idx}",
                        "description": str(item),
                        "weight": 1.0,
                        "criterion_id": f"parent_{idx}",
                        "parent_id": "",
                        "source": "parent",
                    }
                )
        return items

    if isinstance(raw_rubric_list, np.ndarray):
        raw_rubric_list = raw_rubric_list.tolist()
    if isinstance(raw_rubric_list, list):
        return [
            {
                "title": f"Criterion {idx}",
                "description": str(item),
                "weight": 1.0,
                "criterion_id": f"parent_{idx}",
                "parent_id": "",
                "source": "parent",
            }
            for idx, item in enumerate(raw_rubric_list)
        ]
    return []


def format_complete_rubric(rubric_items: list[dict[str, Any]]) -> str:
    requirements = []
    penalties = []
    neutral = []

    for idx, item in enumerate(rubric_items, start=1):
        title = str(item.get("title") or f"Criterion {idx}").strip()
        description = str(item.get("description") or "").strip()
        weight = _as_float(item.get("weight", 1.0), default=1.0)
        criterion = f"{idx}. [{title}] {description} (points: {weight:g})"
        if weight > 0:
            requirements.append(criterion)
        elif weight < 0:
            penalties.append(criterion)
        else:
            neutral.append(criterion)

    sections = []
    if requirements:
        sections.append("Requirements to satisfy:\n" + "\n".join(requirements))
    if penalties:
        sections.append(
            "Penalty conditions to avoid:\n"
            "The response should avoid triggering the failures, errors, or "
            "undesired behaviors described below.\n" + "\n".join(penalties)
        )
    if neutral:
        sections.append("Additional evaluation criteria:\n" + "\n".join(neutral))
    return "\n\n".join(sections)


def make_prompt_key(prompt_text: str, parent_rubric: list[dict[str, Any]]) -> str:
    payload = {
        "prompt_text": prompt_text,
        "parent_rubric": parent_rubric,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def classify_pass_rate(
    pass_rate: float,
    *,
    tau_dead: float,
    tau_sat: float,
) -> str:
    if pass_rate <= tau_dead:
        return CRITERION_STATE_DEAD
    if pass_rate >= tau_sat:
        return CRITERION_STATE_SATURATED
    return CRITERION_STATE_MIXED


def criterion_polarity(rubric_item: dict[str, Any]) -> str:
    weight = _as_float(rubric_item.get("weight", 1.0), default=1.0)
    return CRITERION_POLARITY_NEGATIVE if weight < 0 else CRITERION_POLARITY_POSITIVE


def classify_criterion_state(
    pass_rate: float,
    rubric_item: dict[str, Any],
    *,
    tau_dead: float,
    tau_sat: float,
) -> str:
    """Classify signal state while respecting criterion polarity.

    The judge emits raw criterion-hit rates. For positive criteria, a hit is a
    desirable pass. For negative criteria, a hit means the penalty/pitfall was
    triggered, so the semantic direction is reversed.
    """
    state = classify_pass_rate(pass_rate, tau_dead=tau_dead, tau_sat=tau_sat)
    if criterion_polarity(rubric_item) == CRITERION_POLARITY_POSITIVE:
        return state
    if state == CRITERION_STATE_DEAD:
        return CRITERION_STATE_SATURATED
    if state == CRITERION_STATE_SATURATED:
        return CRITERION_STATE_DEAD
    return state


def is_example_criterion(rubric_item: dict[str, Any]) -> bool:
    return "(example)" in str(rubric_item.get("title", "")).lower()


def rubric_item_identity(rubric_item: dict[str, Any], fallback_idx: int) -> str:
    criterion_id = str(rubric_item.get("criterion_id") or "").strip()
    if criterion_id:
        return f"id:{criterion_id}"
    title = str(rubric_item.get("title") or "").strip()
    description = str(rubric_item.get("description") or "").strip()
    weight = _as_float(rubric_item.get("weight", 1.0), default=1.0)
    return f"fallback:{fallback_idx}:{title}\n{description}\n{weight:g}"


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _extract_json_object_or_array(text: str) -> dict[str, Any] | list[Any]:
    """Extract the first JSON object or array from a model response."""

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    raise json.JSONDecodeError("No JSON object or array found", text, 0)


def parse_proposed_criteria(text: str, *, max_items: int) -> list[dict[str, Any]]:
    payload = _extract_json_object(text)
    raw_items = payload.get("subcriteria")
    if raw_items is None:
        raw_items = payload.get("refinements")
    if raw_items is None:
        raw_items = payload.get("criteria")
    if not isinstance(raw_items, list):
        return []

    parsed: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items[:max_items]):
        if isinstance(item, str):
            parsed.append(
                {
                    "title": f"Criterion {idx}",
                    "description": item.strip(),
                    "weight": 1.0,
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or item.get("text") or "").strip()
        if not description:
            continue
        parsed.append(
            {
                "title": str(item.get("title") or f"Criterion {idx}").strip(),
                "description": description,
                "weight": _as_float(item.get("weight", 1.0), default=1.0),
                # Keep the proposal's evidence and lineage claims.  They are
                # not trusted as verification, but they make both the policy
                # verifier and the evolution logs auditable.
                "operation": str(item.get("operation") or "").strip(),
                "parent_link": str(
                    item.get("parent_link") or item.get("parent_preservation") or ""
                ).strip(),
                "inherited_parent_span": str(
                    item.get("inherited_parent_span") or ""
                ).strip(),
                "positive_evidence": (
                    item.get("positive_evidence")
                    if isinstance(item.get("positive_evidence"), list)
                    else []
                ),
                "negative_evidence": (
                    item.get("negative_evidence")
                    if isinstance(item.get("negative_evidence"), list)
                    else []
                ),
                "factual_support": (
                    item.get("factual_support")
                    if isinstance(item.get("factual_support"), list)
                    else []
                ),
            }
        )
    return parsed


def parse_batched_proposal_payload(
    text: str,
    *,
    expected_operations: dict[str, str],
    max_items_by_parent: dict[str, int],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Parse one multi-parent proposal response and fail closed per parent.

    Parent ids are supplied by the caller and are the only accepted routing
    keys. A missing, duplicated, unknown, or operation-mismatched bundle never
    gets assigned to another parent by position.
    """

    expected_ids = list(expected_operations)
    parsed_by_parent: dict[str, list[dict[str, Any]]] = {
        parent_id: [] for parent_id in expected_ids
    }
    diagnostics: dict[str, Any] = {
        "parse_valid": False,
        "expected_count": len(expected_ids),
        "returned_count": 0,
        "missing_parent_ids": list(expected_ids),
        "duplicate_parent_ids": [],
        "unknown_parent_ids": [],
        "operation_mismatch_parent_ids": [],
    }

    try:
        payload = _extract_json_object_or_array(text)
    except json.JSONDecodeError:
        diagnostics["reason"] = "invalid_proposal_batch_json"
        return parsed_by_parent, diagnostics

    raw_targets = payload if isinstance(payload, list) else payload.get("targets")
    if not isinstance(raw_targets, list):
        diagnostics["reason"] = "missing_proposal_targets_list"
        return parsed_by_parent, diagnostics

    diagnostics["parse_valid"] = True
    diagnostics["returned_count"] = len(raw_targets)
    bundles_by_parent: dict[str, list[dict[str, Any]]] = {}
    unknown_ids: list[str] = []
    for raw_bundle in raw_targets:
        if not isinstance(raw_bundle, dict):
            unknown_ids.append("<non_object>")
            continue
        parent_id = str(raw_bundle.get("parent_id") or "").strip()
        if parent_id not in expected_operations:
            unknown_ids.append(parent_id or "<missing>")
            continue
        bundles_by_parent.setdefault(parent_id, []).append(raw_bundle)

    duplicate_ids = [
        parent_id
        for parent_id, bundles in bundles_by_parent.items()
        if len(bundles) != 1
    ]
    operation_mismatches: list[str] = []
    for parent_id, bundles in bundles_by_parent.items():
        if len(bundles) != 1:
            continue
        bundle = bundles[0]
        expected_operation = str(expected_operations[parent_id]).strip().casefold()
        returned_operation = str(bundle.get("operation") or "").strip().casefold()
        if returned_operation != expected_operation:
            operation_mismatches.append(parent_id)
            continue
        raw_candidates = bundle.get("candidates")
        if not isinstance(raw_candidates, list):
            continue
        max_items = max(0, int(max_items_by_parent.get(parent_id, 0)))
        if max_items == 0:
            continue
        candidates = parse_proposed_criteria(
            json.dumps({"criteria": raw_candidates}, ensure_ascii=False),
            max_items=max_items,
        )
        for candidate in candidates:
            candidate["operation"] = expected_operation
        parsed_by_parent[parent_id] = candidates

    missing_ids = [
        parent_id for parent_id in expected_ids if parent_id not in bundles_by_parent
    ]
    diagnostics.update(
        {
            "missing_parent_ids": missing_ids,
            "duplicate_parent_ids": duplicate_ids,
            "unknown_parent_ids": list(dict.fromkeys(unknown_ids)),
            "operation_mismatch_parent_ids": operation_mismatches,
        }
    )
    return parsed_by_parent, diagnostics


def _invalid_batched_verifier_result(candidate_id: str, reason: str) -> str:
    return json.dumps(
        {
            "candidate_id": candidate_id,
            "verdict": "INVALID: Invalid Target Evolution",
            "reason": reason,
        },
        ensure_ascii=False,
    )


def parse_batched_candidate_verifier_payloads(
    text: str,
    *,
    expected_candidate_ids: list[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Route one batched meta-verifier response by explicit candidate id.

    Missing and duplicated ids receive synthetic invalid payloads. An unknown
    id invalidates the complete batch because it indicates unreliable routing;
    no model-authored result is ever reassigned by output order.
    """

    expected_ids = list(dict.fromkeys(str(item) for item in expected_candidate_ids))
    routed = {
        candidate_id: _invalid_batched_verifier_result(
            candidate_id,
            "missing_batched_meta_verifier_result",
        )
        for candidate_id in expected_ids
    }
    diagnostics: dict[str, Any] = {
        "parse_valid": False,
        "expected_count": len(expected_ids),
        "returned_count": 0,
        "missing_candidate_ids": list(expected_ids),
        "duplicate_candidate_ids": [],
        "unknown_candidate_ids": [],
    }

    try:
        payload = _extract_json_object_or_array(text)
    except json.JSONDecodeError:
        diagnostics["reason"] = "invalid_batched_meta_verifier_json"
        return routed, diagnostics

    raw_results = payload if isinstance(payload, list) else payload.get("results")
    if not isinstance(raw_results, list):
        diagnostics["reason"] = "missing_batched_meta_results_list"
        return routed, diagnostics

    diagnostics["parse_valid"] = True
    diagnostics["returned_count"] = len(raw_results)
    results_by_id: dict[str, list[dict[str, Any]]] = {}
    unknown_ids: list[str] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            unknown_ids.append("<non_object>")
            continue
        candidate_id = str(raw_result.get("candidate_id") or "").strip()
        if candidate_id not in routed:
            unknown_ids.append(candidate_id or "<missing>")
            continue
        results_by_id.setdefault(candidate_id, []).append(raw_result)

    duplicate_ids = [
        candidate_id
        for candidate_id, results in results_by_id.items()
        if len(results) != 1
    ]
    for candidate_id, results in results_by_id.items():
        if len(results) == 1:
            routed[candidate_id] = json.dumps(results[0], ensure_ascii=False)
        else:
            routed[candidate_id] = _invalid_batched_verifier_result(
                candidate_id,
                "duplicate_batched_meta_verifier_result",
            )

    missing_ids = [
        candidate_id
        for candidate_id in expected_ids
        if candidate_id not in results_by_id
    ]
    unique_unknown_ids = list(dict.fromkeys(unknown_ids))
    if unique_unknown_ids:
        routed = {
            candidate_id: _invalid_batched_verifier_result(
                candidate_id,
                "unknown_candidate_id_in_batched_meta_verifier_result",
            )
            for candidate_id in expected_ids
        }
    diagnostics.update(
        {
            "missing_candidate_ids": missing_ids,
            "duplicate_candidate_ids": duplicate_ids,
            "unknown_candidate_ids": unique_unknown_ids,
        }
    )
    return routed, diagnostics


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {
            "true",
            "yes",
            "y",
            "1",
            "pass",
            "passed",
            "accept",
            "accepted",
        }:
            return True
        if normalized in {
            "false",
            "no",
            "n",
            "0",
            "fail",
            "failed",
            "reject",
            "rejected",
        }:
            return False
    return default


def parse_candidate_verifier_verdict(text: str) -> dict[str, Any]:
    if not str(text or "").strip():
        return {
            "valid": False,
            "reason": "missing_verifier_payload",
            "rejection_type": "invalid_json",
        }

    try:
        payload = _extract_json_object(text)
    except json.JSONDecodeError:
        return {
            "valid": False,
            "reason": "invalid_verifier_json",
            "rejection_type": "invalid_json",
            "raw": _truncate_for_log(text, 400),
        }

    # Batched production verification returns every dimension for an auditable
    # chain of reasoning, but the model-authored final verdict is authoritative.
    # Code validates only the response schema and candidate routing; it does not
    # replace the model's semantic judgment with a programmatic conjunction.
    if "checks" in payload:
        raw_checks = payload.get("checks")
        if not isinstance(raw_checks, dict):
            return {
                "valid": False,
                "reason": "invalid_multidimensional_checks",
                "rejection_type": "invalid_json",
                "raw": _truncate_for_log(text, 400),
            }

        normalized_checks: dict[str, dict[str, str]] = {}
        missing_dimensions: list[str] = []
        invalid_dimensions: list[str] = []
        failed_dimensions: list[str] = []
        uncertain_dimensions: list[str] = []
        for dimension in META_VERIFIER_DIMENSIONS:
            raw_check = raw_checks.get(dimension)
            if not isinstance(raw_check, dict):
                missing_dimensions.append(dimension)
                continue
            verdict = str(raw_check.get("verdict") or "").strip().upper()
            reason = str(raw_check.get("reason") or "").strip()
            normalized_checks[dimension] = {"verdict": verdict, "reason": reason}
            if verdict == "FAIL":
                failed_dimensions.append(dimension)
            elif verdict == "UNCERTAIN":
                uncertain_dimensions.append(dimension)
            elif verdict != "PASS":
                invalid_dimensions.append(dimension)

        if missing_dimensions or invalid_dimensions:
            problems = []
            if missing_dimensions:
                problems.append("missing=" + ",".join(missing_dimensions))
            if invalid_dimensions:
                problems.append("invalid=" + ",".join(invalid_dimensions))
            return {
                "valid": False,
                "reason": "invalid_multidimensional_checks:" + ";".join(problems),
                "rejection_type": "invalid_json",
                "checks": normalized_checks,
                "missing_dimensions": missing_dimensions,
                "invalid_dimensions": invalid_dimensions,
                "raw": _truncate_for_log(text, 400),
            }

        model_final_verdict = str(payload.get("final_verdict") or "").strip().upper()
        if model_final_verdict not in {"VALID", "INVALID", "REVIEW"}:
            return {
                "valid": False,
                "reason": "invalid_or_missing_model_final_verdict",
                "rejection_type": "invalid_json",
                "checks": normalized_checks,
                "failed_dimensions": failed_dimensions,
                "uncertain_dimensions": uncertain_dimensions,
                "raw": _truncate_for_log(text, 400),
            }

        valid = model_final_verdict == "VALID"
        non_pass_dimensions = failed_dimensions or uncertain_dimensions
        rejection_type = (
            ""
            if valid
            else (
                _META_DIMENSION_REJECTION_TYPES.get(
                    non_pass_dimensions[0],
                    "verifier_rejected",
                )
                if non_pass_dimensions
                else "verifier_rejected"
            )
        )
        reason = str(payload.get("final_reason") or "").strip()
        if not reason:
            if valid:
                reason = "all_multidimensional_checks_passed"
            else:
                reason_parts = []
                for dimension in non_pass_dimensions:
                    check_reason = normalized_checks[dimension]["reason"]
                    reason_parts.append(
                        f"{dimension}:{check_reason or normalized_checks[dimension]['verdict']}"
                    )
                reason = "; ".join(reason_parts)
        return {
            "valid": valid,
            "reason": reason,
            "rejection_type": rejection_type,
            "candidate_id": str(payload.get("candidate_id") or "").strip(),
            "checks": normalized_checks,
            "failed_dimensions": failed_dimensions,
            "uncertain_dimensions": uncertain_dimensions,
            "model_final_verdict": model_final_verdict,
            "raw": _truncate_for_log(text, 400),
        }

    # Legacy EvoRubric verification used one categorical verdict. Retain it for
    # old artifacts/tests; the batched production prompt uses checks above.
    if "verdict" in payload:
        verdict = str(payload.get("verdict") or "").strip()
        normalized = verdict.casefold()
        valid = normalized == "valid"
        category = ""
        if normalized.startswith("invalid:"):
            category = verdict.split(":", 1)[1].strip()
        elif not valid:
            category = "Invalid JSON"
        category_key = category.casefold()
        if "target evolution" in category_key:
            rejection_type = "invalid_relation"
        elif "factual conflict" in category_key:
            rejection_type = "unsupported_fact"
        elif "unmeasurable" in category_key:
            rejection_type = "not_judgeable"
        elif "semantic duplication" in category_key:
            rejection_type = "redundant"
        elif "trivial" in category_key:
            rejection_type = "trivial"
        elif "length" in category_key or "style" in category_key:
            rejection_type = "length_style"
        elif "rollout overfitting" in category_key:
            rejection_type = "rollout_overfit"
        else:
            rejection_type = "" if valid else "invalid_json"
        return {
            "valid": valid,
            "verdict": verdict,
            "category": category,
            "reason": str(
                payload.get("reason") or verdict or "meta_verifier_rejected"
            ).strip(),
            "rejection_type": rejection_type,
            "raw": _truncate_for_log(text, 400),
        }

    parent_anchored = _as_bool(payload.get("parent_anchored"), default=False)
    operation_valid = _as_bool(payload.get("operation_valid"), default=False)
    polarity_preserving = _as_bool(payload.get("polarity_preserving"), default=False)
    behavior_direction_preserved = _as_bool(
        payload.get("behavior_direction_preserved"),
        default=False,
    )
    required_implication_holds = _as_bool(
        payload.get("required_implication_holds"),
        default=False,
    )
    if "parent_paraphrase" in payload:
        parent_paraphrase = _as_bool(payload.get("parent_paraphrase"), default=True)
        not_redundant_with_parent = not parent_paraphrase
    else:
        not_redundant_with_parent = _as_bool(
            payload.get(
                "not_paraphrase_of_parent", payload.get("not_redundant_with_parent")
            ),
            default=False,
        )
    judgeable = _as_bool(payload.get("judgeable"), default=False)
    factual_grounded = _as_bool(payload.get("factual_grounded"), default=False)
    counterexample_found = _as_bool(payload.get("counterexample_found"), default=False)
    checks_valid = (
        parent_anchored
        and operation_valid
        and polarity_preserving
        and behavior_direction_preserved
        and required_implication_holds
        and not_redundant_with_parent
        and judgeable
        and factual_grounded
        and not counterexample_found
    )
    model_valid = _as_bool(payload.get("valid"), default=checks_valid)
    valid = model_valid and checks_valid

    reason = str(payload.get("reason") or "").strip()
    if not reason:
        reason = "accepted" if valid else "verifier_rejected"

    if valid:
        rejection_type = ""
    elif not parent_anchored:
        rejection_type = "parent_drift"
    elif not operation_valid or not required_implication_holds or counterexample_found:
        rejection_type = "invalid_relation"
    elif not polarity_preserving or not behavior_direction_preserved:
        rejection_type = "polarity_mismatch"
    elif not not_redundant_with_parent:
        rejection_type = "parent_paraphrase"
    elif not judgeable:
        rejection_type = "not_judgeable"
    elif not factual_grounded:
        rejection_type = "unsupported_fact"
    else:
        rejection_type = "verifier_rejected"

    return {
        "valid": valid,
        "reason": reason,
        "rejection_type": rejection_type,
        "parent_anchored": parent_anchored,
        "operation_valid": operation_valid,
        "polarity_preserving": polarity_preserving,
        "behavior_direction_preserved": behavior_direction_preserved,
        "required_implication_holds": required_implication_holds,
        "not_redundant_with_parent": not_redundant_with_parent,
        "parent_paraphrase": not not_redundant_with_parent,
        "judgeable": judgeable,
        "factual_grounded": factual_grounded,
        "counterexample_found": counterexample_found,
        "normalized_parent_behavior": str(
            payload.get("normalized_parent_behavior") or ""
        ).strip(),
        "normalized_candidate_behavior": str(
            payload.get("normalized_candidate_behavior") or ""
        ).strip(),
        "inherited_parent_span": str(
            payload.get("inherited_parent_span") or ""
        ).strip(),
        "unsupported_claims": (
            payload.get("unsupported_claims")
            if isinstance(payload.get("unsupported_claims"), list)
            else []
        ),
        "raw": _truncate_for_log(text, 400),
    }


def parse_collision_verifier_verdict(text: str) -> dict[str, Any]:
    """Parse the independent candidate-vs-rubric collision audit.

    Generation and global collision checking intentionally use separate model
    calls.  Missing fields fail closed so malformed self-verification cannot
    silently activate a candidate.
    """
    if not str(text or "").strip():
        return {
            "valid": False,
            "reason": "missing_collision_verifier_payload",
            "rejection_type": "invalid_json",
        }
    try:
        payload = _extract_json_object(text)
    except json.JSONDecodeError:
        return {
            "valid": False,
            "reason": "invalid_collision_verifier_json",
            "rejection_type": "invalid_json",
            "raw": _truncate_for_log(text, 400),
        }

    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list):
        comparisons = []
    normalized_comparisons: list[dict[str, Any]] = []
    detected_conflicts: list[str] = []
    for item in comparisons:
        if not isinstance(item, dict):
            continue
        criterion_id = str(item.get("criterion_id") or "").strip()
        relation = str(item.get("relation") or "").strip().lower()
        conflict = _as_bool(item.get("conflict"), default=False) or relation in {
            "duplicate",
            "near_duplicate",
            "sibling_dimension",
            "cross_criterion_leakage",
        }
        normalized_comparisons.append(
            {
                "criterion_id": criterion_id,
                "relation": relation,
                "conflict": conflict,
                "reason": str(item.get("reason") or "").strip(),
            }
        )
        if conflict:
            detected_conflicts.append(criterion_id or relation or "unknown")

    explicit_conflict_ids = payload.get("conflict_ids")
    if isinstance(explicit_conflict_ids, list):
        detected_conflicts.extend(
            str(item) for item in explicit_conflict_ids if str(item).strip()
        )
    detected_conflicts = list(dict.fromkeys(detected_conflicts))
    all_compared = _as_bool(payload.get("all_compared"), default=False)
    no_conflict = _as_bool(payload.get("no_conflict"), default=False)
    model_valid = _as_bool(payload.get("valid"), default=False)
    valid = model_valid and all_compared and no_conflict and not detected_conflicts
    reason = str(payload.get("reason") or "").strip() or (
        "accepted" if valid else "rubric_collision"
    )
    return {
        "valid": valid,
        "reason": reason,
        "rejection_type": "" if valid else "collision",
        "no_conflict": no_conflict,
        "all_compared": all_compared,
        "checked_count": int(_as_float(payload.get("checked_count"), default=0.0)),
        "conflict_ids": detected_conflicts,
        "comparisons": normalized_comparisons,
        "raw": _truncate_for_log(text, 600),
    }


def parse_collision_pair_verdict(text: str) -> dict[str, Any]:
    """Parse a focused confirmation for one suspected rubric collision."""
    if not str(text or "").strip():
        return {
            "valid": False,
            "conflict": True,
            "reason": "missing_collision_pair_payload",
        }
    try:
        payload = _extract_json_object(text)
    except json.JSONDecodeError:
        return {
            "valid": False,
            "conflict": True,
            "reason": "invalid_collision_pair_json",
            "raw": _truncate_for_log(text, 300),
        }
    duplicate = _as_bool(payload.get("duplicate"), default=False)
    candidate_matches_existing = _as_bool(
        payload.get("candidate_matches_existing"),
        default=False,
    )
    drifted_to_existing_dimension = _as_bool(
        payload.get("drifted_to_existing_dimension"),
        default=False,
    )
    # The selected parent is excluded from the collision catalog, so matching
    # any listed criterion is already a duplicate/sibling collision.  Do not
    # trust a contradictory drift boolean to clear an explicit B-C match.
    relation = str(payload.get("relation") or "").strip().lower()
    conflict = (
        _as_bool(payload.get("conflict"), default=False)
        or duplicate
        or candidate_matches_existing
        or drifted_to_existing_dimension
        or relation in {"duplicate", "near_duplicate", "sibling_dimension"}
    )
    model_valid = _as_bool(payload.get("valid"), default=False)
    # Invalid/malformed confirmations fail closed and preserve the suspected
    # conflict instead of silently clearing it.
    return {
        "valid": model_valid,
        "conflict": conflict if model_valid else True,
        "duplicate": duplicate,
        "candidate_matches_existing": candidate_matches_existing,
        "drifted_to_existing_dimension": drifted_to_existing_dimension,
        "relation": relation,
        "reason": str(payload.get("reason") or "").strip(),
        "raw": _truncate_for_log(text, 400),
    }


def parse_negative_polarity_verdict(text: str) -> dict[str, Any]:
    """Parse a minimal literal-trigger audit for a negative-weight child."""
    if not str(text or "").strip():
        return {
            "valid": False,
            "reason": "missing_negative_polarity_payload",
        }
    try:
        payload = _extract_json_object(text)
    except json.JSONDecodeError:
        return {
            "valid": False,
            "reason": "invalid_negative_polarity_json",
            "raw": _truncate_for_log(text, 300),
        }
    same_behavior_family = _as_bool(payload.get("same_behavior_family"), default=False)
    candidate_trigger_implies_parent_trigger = _as_bool(
        payload.get("candidate_trigger_implies_parent_trigger"),
        default=False,
    )
    penalizes_opposite_or_avoidance = _as_bool(
        payload.get("penalizes_opposite_or_avoidance"),
        default=True,
    )
    model_valid = _as_bool(payload.get("valid"), default=False)
    valid = (
        model_valid
        and same_behavior_family
        and candidate_trigger_implies_parent_trigger
        and not penalizes_opposite_or_avoidance
    )
    return {
        "valid": valid,
        "same_behavior_family": same_behavior_family,
        "candidate_trigger_implies_parent_trigger": candidate_trigger_implies_parent_trigger,
        "penalizes_opposite_or_avoidance": penalizes_opposite_or_avoidance,
        "parent_penalty_trigger": str(
            payload.get("parent_penalty_trigger") or ""
        ).strip(),
        "candidate_penalty_trigger": str(
            payload.get("candidate_penalty_trigger") or ""
        ).strip(),
        "reason": str(payload.get("reason") or "").strip(),
        "raw": _truncate_for_log(text, 400),
    }


def _semantic_content_tokens(text: str) -> set[str]:
    presentation_tokens = {
        "a",
        "an",
        "and",
        "the",
        "to",
        "of",
        "in",
        "with",
        "both",
        "explicit",
        "explicitly",
        "clear",
        "clearly",
        "mention",
        "mentions",
        "mentioned",
        "reference",
        "references",
        "specify",
        "specifies",
        "specified",
        "state",
        "states",
        "stating",
        "clarify",
        "clarifies",
        "emphasize",
        "emphasizes",
        "include",
        "includes",
        "including",
        "use",
        "uses",
        "usage",
        "source",
        "sources",
        "requirement",
        "response",
        "intake",
        "dosage",
        "amount",
        "target",
        "minimum",
        "at",
        "least",
        "per",
    }
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", str(text or "").lower()):
        if token in presentation_tokens:
            continue
        if token == "daily":
            token = "day"
        elif token == "supplementation":
            token = "supplement"
        if len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        tokens.add(token)
    return tokens


def parse_parent_novelty_verdict(
    text: str,
    *,
    source: str,
    parent_text: str = "",
    candidate_text: str = "",
) -> dict[str, Any]:
    """Parse a focused parent-vs-child novelty/atomicity audit."""
    if not str(text or "").strip():
        return {"valid": False, "reason": "missing_parent_novelty_payload"}
    try:
        payload = _extract_json_object(text)
    except json.JSONDecodeError:
        return {
            "valid": False,
            "reason": "invalid_parent_novelty_json",
            "raw": _truncate_for_log(text, 300),
        }
    uncovered_constraints = payload.get("uncovered_constraints")
    if not isinstance(uncovered_constraints, list):
        uncovered_constraints = []
    uncovered_constraints = [
        str(item).strip() for item in uncovered_constraints if str(item).strip()
    ]
    parent_tokens = _semantic_content_tokens(parent_text)
    candidate_tokens = _semantic_content_tokens(candidate_text)
    substantive_uncovered_constraints = []
    for constraint in uncovered_constraints:
        constraint_tokens = _semantic_content_tokens(constraint)
        if (
            constraint_tokens
            and not constraint_tokens.issubset(parent_tokens)
            and candidate_tokens
            and constraint_tokens.issubset(candidate_tokens)
        ):
            substantive_uncovered_constraints.append(constraint)
    is_paraphrase = _as_bool(payload.get("is_paraphrase"), default=True)
    proper_atomic_subset = _as_bool(payload.get("proper_atomic_subset"), default=False)
    if source == "refine":
        valid = bool(substantive_uncovered_constraints) and not is_paraphrase
    else:
        valid = proper_atomic_subset and not is_paraphrase
    return {
        "valid": valid,
        "source": source,
        "is_paraphrase": is_paraphrase,
        "proper_atomic_subset": proper_atomic_subset,
        "uncovered_constraints": uncovered_constraints,
        "substantive_uncovered_constraints": substantive_uncovered_constraints,
        "reason": str(payload.get("reason") or "").strip(),
        "raw": _truncate_for_log(text, 500),
    }


@dataclass
class CriterionMeta:
    parent_index: int
    state: str = CRITERION_STATE_MIXED
    pass_rate: float = 0.0
    last_observed_step: int = 0
    last_evolved_step: Optional[int] = None
    active_child_ids: list[str] = field(default_factory=list)


@dataclass
class EvolvedCriterion:
    criterion_id: str
    parent_index: int
    parent_criterion_id: str
    title: str
    description: str
    weight: float
    source: str
    status: str = CANDIDATE_PENDING
    proposed_step: int = 0
    verify_history: list[dict[str, Any]] = field(default_factory=list)

    def to_rubric_item(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "weight": self.weight,
            "criterion_id": self.criterion_id,
            "parent_id": self.parent_criterion_id,
            "source": self.source,
        }


@dataclass
class PromptRubricState:
    prompt_key: str
    parent_rubric: list[dict[str, Any]]
    candidates: list[EvolvedCriterion] = field(default_factory=list)
    criterion_meta: dict[int, CriterionMeta] = field(default_factory=dict)
    transfer_parent_weight_to_children: bool = False
    equal_parent_child_weights: bool = False

    def active_children(self) -> list[EvolvedCriterion]:
        return [item for item in self.candidates if item.status == CANDIDATE_ACTIVE]

    def active_children_for_parent(self, parent_index: int) -> list[EvolvedCriterion]:
        return [
            item
            for item in self.candidates
            if item.status == CANDIDATE_ACTIVE and item.parent_index == parent_index
        ]

    def pending_children(self) -> list[EvolvedCriterion]:
        return [item for item in self.candidates if item.status == CANDIDATE_PENDING]

    def rebalance_active_child_weights(
        self, parent_index: int, child_weight_ratio: float
    ) -> None:
        children = self.active_children_for_parent(parent_index)
        if not children or parent_index >= len(self.parent_rubric):
            return
        parent_weight = _as_float(
            self.parent_rubric[parent_index].get("weight", 1.0), default=1.0
        )
        if self.equal_parent_child_weights:
            child_weight = equal_family_weight(parent_weight, len(children))
        else:
            if parent_weight == 0.0:
                return
            effective_ratio = (
                1.0 if self.transfer_parent_weight_to_children else child_weight_ratio
            )
            child_magnitude = abs(parent_weight) * effective_ratio / float(len(children))
            child_weight = -child_magnitude if parent_weight < 0 else child_magnitude
        for child in children:
            child.weight = child_weight

    def build_active_rubric(self) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        for parent_index, parent_item in enumerate(self.parent_rubric):
            # The original rubric remains the task-definition anchor.  Evolved
            # children may take over its reward weight without deleting its
            # semantic definition from the Teacher/verifier context.
            children = self.active_children_for_parent(parent_index)
            active_parent = deepcopy(parent_item)
            if self.equal_parent_child_weights:
                # Preserve the original weight in parent_rubric. Always derive
                # effective weights afresh so repeated reads cannot shrink them.
                share = equal_family_weight(_as_float(parent_item.get("weight", 1.0), default=1.0), len(children))
                active_parent["weight"] = share
            elif children and self.transfer_parent_weight_to_children:
                active_parent["weight"] = 0.0
            active.append(active_parent)
            if children:
                active.extend(
                    {**child.to_rubric_item(), "weight": share}
                    if self.equal_parent_child_weights else child.to_rubric_item()
                    for child in children
                )
        return active


class RubricStore:
    def __init__(self, *, transfer_parent_weight_to_children: bool = False,
                 equal_parent_child_weights: bool = False) -> None:
        if transfer_parent_weight_to_children and equal_parent_child_weights:
            raise ValueError("Conflicting rubric weight allocation modes")
        self._states: dict[str, PromptRubricState] = {}
        self.equal_parent_child_weights = bool(equal_parent_child_weights)
        self.transfer_parent_weight_to_children = bool(
            transfer_parent_weight_to_children
        )

    def get(self, prompt_key: str) -> Optional[PromptRubricState]:
        return self._states.get(prompt_key)

    def get_or_create(
        self, prompt_key: str, parent_rubric: list[dict[str, Any]]
    ) -> PromptRubricState:
        if prompt_key not in self._states:
            parents = deepcopy(parent_rubric)
            for idx, item in enumerate(parents):
                item.setdefault("criterion_id", f"parent_{idx}")
                item.setdefault("parent_id", "")
                item.setdefault("source", "parent")
            self._states[prompt_key] = PromptRubricState(
                prompt_key=prompt_key,
                parent_rubric=parents,
                transfer_parent_weight_to_children=(
                    self.transfer_parent_weight_to_children
                ),
                equal_parent_child_weights=self.equal_parent_child_weights,
            )
        return self._states[prompt_key]

    def get_active_rubric(
        self, prompt_key: str, parent_rubric: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        state = self.get_or_create(prompt_key, parent_rubric)
        return state.build_active_rubric()

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of all persistent rubric state."""
        return {
            "version": 1,
            "states": {
                prompt_key: {
                    "prompt_key": state.prompt_key,
                    "parent_rubric": deepcopy(state.parent_rubric),
                    "equal_parent_child_weights": state.equal_parent_child_weights,
                    "transfer_parent_weight_to_children": (
                        state.transfer_parent_weight_to_children
                    ),
                    "candidates": [asdict(candidate) for candidate in state.candidates],
                    "criterion_meta": {
                        str(parent_index): asdict(meta)
                        for parent_index, meta in state.criterion_meta.items()
                    },
                }
                for prompt_key, state in self._states.items()
            },
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`state_dict`.

        Checkpoint corruption must fail loudly: silently starting with the
        original rubrics would change the reward function after resume.
        """
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("Unsupported rubric evolution checkpoint format.")
        raw_states = payload.get("states")
        if not isinstance(raw_states, dict):
            raise ValueError("Rubric evolution checkpoint is missing states.")

        restored: dict[str, PromptRubricState] = {}
        valid_statuses = {CANDIDATE_PENDING, CANDIDATE_ACTIVE, CANDIDATE_REJECTED}
        for prompt_key, raw_state in raw_states.items():
            if not isinstance(raw_state, dict):
                raise ValueError(f"Invalid rubric state for prompt {prompt_key!r}.")
            parent_rubric = raw_state.get("parent_rubric")
            raw_candidates = raw_state.get("candidates", [])
            raw_meta = raw_state.get("criterion_meta", {})
            if not isinstance(parent_rubric, list):
                raise ValueError(f"Invalid parent rubric for prompt {prompt_key!r}.")
            if not isinstance(raw_candidates, list) or not isinstance(raw_meta, dict):
                raise ValueError(
                    f"Invalid candidate/meta state for prompt {prompt_key!r}."
                )
            stored_equal = raw_state.get("equal_parent_child_weights", False)
            if not isinstance(stored_equal, bool):
                raise ValueError("Invalid equal parent/child weight state")
            if stored_equal != self.equal_parent_child_weights and any(
                isinstance(item, dict) and item.get("status") == CANDIDATE_ACTIVE
                for item in raw_candidates
            ):
                raise ValueError("Checkpoint equal parent/child weight mode differs with active children; explicit migration required.")
            # A checkpoint with no active children has identical effective
            # weights under either mode, including earlier GAP-only runs.
            stored_transfer = raw_state.get(
                "transfer_parent_weight_to_children",
                self.transfer_parent_weight_to_children,
            )
            if not isinstance(stored_transfer, bool):
                raise ValueError(
                    f"Invalid weight-transfer state for prompt {prompt_key!r}."
                )
            if stored_transfer != self.transfer_parent_weight_to_children:
                raise ValueError(
                    "Rubric evolution checkpoint weight-transfer mode does not "
                    f"match the current config for prompt {prompt_key!r}."
                )

            candidates: list[EvolvedCriterion] = []
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, dict):
                    raise ValueError(
                        f"Invalid evolved criterion for prompt {prompt_key!r}."
                    )
                status = str(raw_candidate.get("status") or "")
                if status not in valid_statuses:
                    raise ValueError(
                        f"Invalid evolved criterion status {status!r} for prompt {prompt_key!r}."
                    )
                verify_history = raw_candidate.get("verify_history", [])
                if not isinstance(verify_history, list):
                    raise ValueError(
                        f"Invalid verify history for prompt {prompt_key!r}."
                    )
                candidates.append(
                    EvolvedCriterion(
                        criterion_id=str(raw_candidate.get("criterion_id") or ""),
                        parent_index=int(raw_candidate.get("parent_index", -1)),
                        parent_criterion_id=str(
                            raw_candidate.get("parent_criterion_id") or ""
                        ),
                        title=str(raw_candidate.get("title") or ""),
                        description=str(raw_candidate.get("description") or ""),
                        weight=_as_float(raw_candidate.get("weight"), default=0.0),
                        source=str(raw_candidate.get("source") or ""),
                        status=status,
                        proposed_step=int(raw_candidate.get("proposed_step", 0)),
                        verify_history=deepcopy(verify_history),
                    )
                )

            criterion_meta: dict[int, CriterionMeta] = {}
            for raw_parent_index, raw_item in raw_meta.items():
                if not isinstance(raw_item, dict):
                    raise ValueError(
                        f"Invalid criterion metadata for prompt {prompt_key!r}."
                    )
                parent_index = int(raw_parent_index)
                active_child_ids = raw_item.get("active_child_ids", [])
                if not isinstance(active_child_ids, list):
                    raise ValueError(
                        f"Invalid active child ids for prompt {prompt_key!r}."
                    )
                last_evolved_step = raw_item.get("last_evolved_step")
                criterion_meta[parent_index] = CriterionMeta(
                    parent_index=int(raw_item.get("parent_index", parent_index)),
                    state=str(raw_item.get("state") or CRITERION_STATE_MIXED),
                    pass_rate=_as_float(raw_item.get("pass_rate"), default=0.0),
                    last_observed_step=int(raw_item.get("last_observed_step", 0)),
                    last_evolved_step=(
                        None if last_evolved_step is None else int(last_evolved_step)
                    ),
                    active_child_ids=[str(item) for item in active_child_ids],
                )

            restored_state = PromptRubricState(
                prompt_key=str(raw_state.get("prompt_key") or prompt_key),
                parent_rubric=deepcopy(parent_rubric),
                candidates=candidates,
                criterion_meta=criterion_meta,
                transfer_parent_weight_to_children=stored_transfer,
                equal_parent_child_weights=self.equal_parent_child_weights,
            )
            if stored_transfer or self.equal_parent_child_weights:
                for parent_index in range(len(parent_rubric)):
                    restored_state.rebalance_active_child_weights(
                        parent_index,
                        child_weight_ratio=1.0,
                    )
            restored[str(prompt_key)] = restored_state

        self._states = restored


@dataclass
class EvolutionTarget:
    prompt_key: str
    prompt_text: str
    parent_index: int
    parent_item: dict[str, Any]
    state: str
    pass_rate: float
    group_indices: list[int]
    group_responses: list[str]


class RubricEvolver:
    def __init__(
        self,
        config: OPSDConfig,
        *,
        judge_fn: Optional[Callable[[dict[str, Any]], dict[str, float]]] = None,
        judge_batch_fn: Optional[
            Callable[[list[dict[str, Any]]], list[dict[str, float]]]
        ] = None,
    ) -> None:
        self.config = config
        self.store = RubricStore(
            transfer_parent_weight_to_children=(
                config.rubric_evolution_transfer_parent_weight_to_children
            ),
            equal_parent_child_weights=config.rubric_evolution_equal_parent_child_weights,
        )
        self.judge_fn = judge_fn
        self.judge_batch_fn = judge_batch_fn
        self._last_group_cache: dict[str, dict[str, Any]] = {}
        self._recent_evolution_examples: list[dict[str, Any]] = []
        self._current_score_cache_step: Optional[int] = None
        self._current_score_cache: dict[tuple[str, str], dict[int, float]] = {}

    def pop_recent_evolution_examples(self) -> list[dict[str, Any]]:
        events = list(self._recent_evolution_examples)
        self._recent_evolution_examples.clear()
        max_examples = int(self.config.rubric_evolution_wandb_max_examples)
        if len(events) <= max_examples:
            return events
        accepted = [event for event in events if event.get("outcome") == "accepted"]
        rejected = [event for event in events if event.get("outcome") == "rejected"]
        others = [
            event
            for event in events
            if event.get("outcome") not in {"accepted", "rejected"}
        ]
        # Keep failures visible instead of filling every WandB table with
        # accepted rows first.  The accepted-only table is derived downstream.
        selected: list[dict[str, Any]] = []
        buckets = [accepted, rejected, others]
        while len(selected) < max_examples and any(buckets):
            for bucket in buckets:
                if bucket and len(selected) < max_examples:
                    selected.append(bucket.pop(0))
        return selected

    def pop_recent_accepted_events(self) -> list[dict[str, Any]]:
        """Backward-compatible alias for accepted rows only."""
        return [
            event
            for event in self.pop_recent_evolution_examples()
            if event.get("outcome") == "accepted"
        ]

    def observed_groups(self) -> dict[str, dict[str, Any]]:
        """Return the fixed current-step rollout groups for evolution operations."""

        return deepcopy(self._last_group_cache)

    def cache_current_criterion_scores(
        self,
        *,
        prompt_key: str,
        criterion_id: str,
        group_indices: list[int],
        score_vector: list[Any],
    ) -> None:
        """Cache scores already produced by the empirical pass-rate gate."""

        if not criterion_id or len(group_indices) != len(score_vector):
            return
        self._current_score_cache[(str(prompt_key), str(criterion_id))] = {
            int(sample_idx): _as_float(score, default=0.0)
            for sample_idx, score in zip(group_indices, score_vector)
        }

    def current_criterion_score(
        self,
        *,
        prompt_key: str,
        criterion_id: str,
        sample_index: int,
    ) -> Optional[float]:
        """Return an already judged new-criterion score for this batch sample."""

        scores = self._current_score_cache.get((str(prompt_key), str(criterion_id)))
        if scores is None:
            return None
        return scores.get(int(sample_index))

    def _build_evolution_example(
        self,
        target: EvolutionTarget,
        *,
        global_step: int,
        contrast_response: str = "",
        contrast_parent_ok: Optional[bool] = None,
    ) -> dict[str, Any]:
        return {
            "step": global_step,
            "prompt": _truncate_for_log(target.prompt_text, 400),
            "parent_state": target.state,
            "parent_title": str(target.parent_item.get("title", "")),
            "parent_description": _truncate_for_log(
                target.parent_item.get("description", ""),
                300,
            ),
            "parent_weight": _as_float(
                target.parent_item.get("weight", 1.0), default=1.0
            ),
            "parent_polarity": criterion_polarity(target.parent_item),
            "parent_pass_rate": float(target.pass_rate),
            "contrast": _truncate_for_log(contrast_response, 800),
            "contrast_parent_ok": contrast_parent_ok,
            "rollout_sample": _truncate_for_log(
                target.group_responses[0] if target.group_responses else "",
                300,
            ),
        }

    def _record_evolution_example(self, event: dict[str, Any]) -> None:
        self._recent_evolution_examples.append(event)

    def collect_global_metrics(self) -> dict[str, float]:
        """Aggregate store-level evolution stats for WandB monitoring."""
        if not self.store._states:
            return {"rubric_evolution/store_prompts": 0.0}

        active_children_total = 0.0
        pending_children_total = 0.0
        active_rubric_lengths: list[float] = []
        parent_rubric_lengths: list[float] = []
        dead_parents = 0.0
        mixed_parents = 0.0
        saturated_parents = 0.0
        pass_rates: list[float] = []
        gap_roots_total = 0.0

        for state in self.store._states.values():
            active_children_total += float(len(state.active_children()))
            pending_children_total += float(len(state.pending_children()))
            active_rubric_lengths.append(float(len(state.build_active_rubric())))
            parent_rubric_lengths.append(float(len(state.parent_rubric)))
            gap_roots_total += float(
                sum(
                    1
                    for item in state.parent_rubric
                    if str(item.get("source") or "") == "gap_expand"
                )
            )
            for meta in state.criterion_meta.values():
                pass_rates.append(float(meta.pass_rate))
                if meta.state == CRITERION_STATE_DEAD:
                    dead_parents += 1.0
                elif meta.state == CRITERION_STATE_MIXED:
                    mixed_parents += 1.0
                else:
                    saturated_parents += 1.0

        num_prompts = float(len(self.store._states))
        parent_state_total = dead_parents + mixed_parents + saturated_parents
        metrics = {
            "rubric_evolution/store_prompts": num_prompts,
            "rubric_evolution/store_active_children_total": active_children_total,
            "rubric_evolution/store_pending_children_total": pending_children_total,
            "rubric_evolution/store_mean_active_rubric_len": float(
                np.mean(active_rubric_lengths)
            ),
            "rubric_evolution/store_mean_parent_rubric_len": float(
                np.mean(parent_rubric_lengths)
            ),
            "rubric_evolution/store_mean_children_per_prompt": active_children_total
            / num_prompts,
            "rubric_evolution/store_gap_roots_total": gap_roots_total,
            "rubric_evolution/store_mean_gap_roots_per_prompt": gap_roots_total
            / num_prompts,
            "rubric_evolution/store_rubric_growth": float(
                np.mean(active_rubric_lengths)
            )
            - float(np.mean(parent_rubric_lengths)),
        }
        if parent_state_total > 0:
            metrics["rubric_evolution/store_dead_ratio"] = (
                dead_parents / parent_state_total
            )
            metrics["rubric_evolution/store_mixed_ratio"] = (
                mixed_parents / parent_state_total
            )
            metrics["rubric_evolution/store_saturated_ratio"] = (
                saturated_parents / parent_state_total
            )
        if pass_rates:
            metrics["rubric_evolution/store_mean_pass_rate"] = float(
                np.mean(pass_rates)
            )
        return metrics

    def apply_active_rubric_to_batch(self, batch: DataProto) -> None:
        rubric_batch = batch.non_tensor_batch.get("rubric")
        rubric_list_batch = batch.non_tensor_batch.get("rubric_list")
        prompt_texts = batch.non_tensor_batch.get("prompt_text")
        parent_rubric_batch = batch.non_tensor_batch.get("parent_rubric")
        prompt_keys = batch.non_tensor_batch.get("rubric_evolution_prompt_key")

        if prompt_keys is None:
            prompt_keys = np.empty(len(batch), dtype=object)

        if parent_rubric_batch is None:
            parent_rubric_batch = np.empty(len(batch), dtype=object)

        active_rubrics = []
        for i in range(len(batch)):
            raw_existing_prompt_key = prompt_keys[i]
            existing_prompt_key = str(raw_existing_prompt_key or "").strip()
            existing_state = self.store.get(existing_prompt_key)
            if existing_state is not None:
                # Keep the original store key stable after GAP_EXPAND appends
                # new roots. Re-hashing the expanded parent rubric would create
                # a second empty state during same-step reward aggregation.
                prompt_keys[i] = existing_prompt_key
                parent_rubric_batch[i] = deepcopy(existing_state.parent_rubric)
                active_rubrics.append(existing_state.build_active_rubric())
                continue

            existing_parent_rubric = (
                None if parent_rubric_batch is None else parent_rubric_batch[i]
            )
            if isinstance(existing_parent_rubric, np.ndarray):
                existing_parent_rubric = existing_parent_rubric.tolist()
            if isinstance(existing_parent_rubric, list) and existing_parent_rubric:
                parent_items = deepcopy(existing_parent_rubric)
            else:
                raw_rubric = None if rubric_batch is None else rubric_batch[i]
                raw_rubric_list = (
                    None if rubric_list_batch is None else rubric_list_batch[i]
                )
                parent_items = normalize_rubric_items(raw_rubric, raw_rubric_list)
            if not parent_items:
                active_rubrics.append([])
                prompt_keys[i] = ""
                parent_rubric_batch[i] = []
                continue

            prompt_text = "" if prompt_texts is None else str(prompt_texts[i])
            prompt_key = make_prompt_key(prompt_text, parent_items)
            state = self.store.get_or_create(prompt_key, parent_items)
            prompt_keys[i] = prompt_key
            parent_rubric_batch[i] = deepcopy(state.parent_rubric)
            active_rubrics.append(state.build_active_rubric())

        batch.non_tensor_batch["rubric_evolution_prompt_key"] = prompt_keys
        batch.non_tensor_batch["parent_rubric"] = parent_rubric_batch
        batch.non_tensor_batch["rubric"] = np.array(active_rubrics, dtype=object)

    def observe(
        self,
        batch: DataProto,
        reward_metric_values: Optional[dict[str, list[Any]]],
        global_step: int,
    ) -> dict[str, float]:
        if self._current_score_cache_step != int(global_step):
            self._current_score_cache.clear()
            self._current_score_cache_step = int(global_step)

        # Targets must be tied to the batch observed in this training step.
        # Leaving older prompt indices here can pair a stale prompt with the
        # current batch's responses during evolution proposal logging.
        self._last_group_cache.clear()

        if reward_metric_values is None:
            return {}

        uids = batch.non_tensor_batch.get("uid")
        if uids is None:
            return {}

        prompt_keys = batch.non_tensor_batch.get("rubric_evolution_prompt_key")
        parent_rubric_batch = batch.non_tensor_batch.get("parent_rubric")
        active_rubric_batch = batch.non_tensor_batch.get("rubric")
        if prompt_keys is None or parent_rubric_batch is None:
            return {}

        metrics = {
            "rubric_evolution/dead_count": 0.0,
            "rubric_evolution/mixed_count": 0.0,
            "rubric_evolution/saturated_count": 0.0,
            "rubric_evolution/active_children": 0.0,
            "rubric_evolution/pending_children": 0.0,
            "rubric_evolution/criteria_observed": 0.0,
            "rubric_evolution/prompts_observed": 0.0,
            "rubric_evolution/incomplete_score_groups": 0.0,
        }
        step_pass_rates: list[float] = []

        unique_uids = np.unique(uids)
        for uid in unique_uids:
            group_indices = np.where(uids == uid)[0].tolist()
            if not group_indices:
                continue

            first_idx = group_indices[0]
            prompt_key = str(prompt_keys[first_idx])
            parent_items = parent_rubric_batch[first_idx]
            if not prompt_key or not parent_items:
                continue

            state = self.store.get_or_create(prompt_key, parent_items)
            num_criteria = len(parent_items)
            active_items = (
                active_rubric_batch[first_idx]
                if active_rubric_batch is not None
                else parent_items
            )
            if isinstance(active_items, np.ndarray):
                active_items = active_items.tolist()
            if not isinstance(active_items, list) or not active_items:
                active_items = parent_items
            active_index_by_identity = {
                rubric_item_identity(item, active_idx): active_idx
                for active_idx, item in enumerate(active_items)
                if isinstance(item, dict)
            }
            pass_rates = []
            for criterion_idx in range(num_criteria):
                parent_item = parent_items[criterion_idx]
                active_idx = active_index_by_identity.get(
                    rubric_item_identity(parent_item, criterion_idx)
                )
                if active_idx is None:
                    existing_meta = state.criterion_meta.get(criterion_idx)
                    pass_rates.append(
                        existing_meta.pass_rate if existing_meta is not None else 0.0
                    )
                    continue

                values = reward_metric_values.get(f"criterion_{active_idx}_score")
                scores = []
                for sample_idx in group_indices:
                    if values is None or sample_idx >= len(values):
                        continue
                    score = _as_float(values[sample_idx], default=float("nan"))
                    if score == score:
                        scores.append(score)
                if len(scores) != len(group_indices):
                    # Missing judge outputs are unknown, not evidence that a
                    # positive criterion is dead (or a penalty is avoided).
                    existing_meta = state.criterion_meta.get(criterion_idx)
                    pass_rates.append(
                        existing_meta.pass_rate if existing_meta is not None else 0.0
                    )
                    metrics["rubric_evolution/incomplete_score_groups"] += 1.0
                    continue
                pass_rate = float(np.mean(scores))
                pass_rates.append(pass_rate)
                criterion_state = classify_criterion_state(
                    pass_rate,
                    parent_item,
                    tau_dead=self.config.rubric_evolution_tau_dead,
                    tau_sat=self.config.rubric_evolution_tau_sat,
                )
                state.criterion_meta[criterion_idx] = CriterionMeta(
                    parent_index=criterion_idx,
                    state=criterion_state,
                    pass_rate=pass_rate,
                    last_observed_step=global_step,
                    last_evolved_step=(
                        state.criterion_meta.get(criterion_idx).last_evolved_step
                        if criterion_idx in state.criterion_meta
                        else None
                    ),
                    active_child_ids=list(
                        state.criterion_meta.get(criterion_idx).active_child_ids
                        if criterion_idx in state.criterion_meta
                        else []
                    ),
                )
                metrics[f"rubric_evolution/state_{criterion_state}"] = (
                    metrics.get(f"rubric_evolution/state_{criterion_state}", 0.0) + 1.0
                )
                metrics["rubric_evolution/criteria_observed"] += 1.0
                step_pass_rates.append(pass_rate)
                if criterion_state == CRITERION_STATE_DEAD:
                    metrics["rubric_evolution/dead_count"] += 1.0
                elif criterion_state == CRITERION_STATE_MIXED:
                    metrics["rubric_evolution/mixed_count"] += 1.0
                else:
                    metrics["rubric_evolution/saturated_count"] += 1.0

            metrics["rubric_evolution/prompts_observed"] += 1.0
            metrics["rubric_evolution/active_children"] += float(
                len(state.active_children())
            )
            metrics["rubric_evolution/pending_children"] += float(
                len(state.pending_children())
            )

            prompt_text = str(
                batch.non_tensor_batch.get("prompt_text", [""])[first_idx]
            )
            self._last_group_cache[prompt_key] = {
                "prompt_text": prompt_text,
                "parent_rubric": deepcopy(parent_items),
                "group_indices": group_indices,
                "pass_rates": pass_rates,
                "criterion_meta": deepcopy(state.criterion_meta),
            }

        criteria_observed = metrics["rubric_evolution/criteria_observed"]
        if criteria_observed > 0:
            metrics["rubric_evolution/dead_ratio"] = (
                metrics["rubric_evolution/dead_count"] / criteria_observed
            )
            metrics["rubric_evolution/mixed_ratio"] = (
                metrics["rubric_evolution/mixed_count"] / criteria_observed
            )
            metrics["rubric_evolution/saturated_ratio"] = (
                metrics["rubric_evolution/saturated_count"] / criteria_observed
            )
        if step_pass_rates:
            metrics["rubric_evolution/step_mean_pass_rate"] = float(
                np.mean(step_pass_rates)
            )
        metrics.update(self.collect_global_metrics())
        return metrics

    def select_targets(self, global_step: int) -> list[EvolutionTarget]:
        targets: list[EvolutionTarget] = []
        max_targets = int(self.config.rubric_evolution_max_targets_per_step)
        max_targets_per_prompt = int(
            self.config.rubric_evolution_max_targets_per_prompt
        )
        retry_cooldown = int(self.config.rubric_evolution_retry_cooldown_steps)

        for prompt_key, cache in self._last_group_cache.items():
            state = self.store.get(prompt_key)
            if state is None:
                continue
            selected_for_prompt = 0

            for criterion_idx, meta in sorted(
                state.criterion_meta.items(), key=lambda item: item[0]
            ):
                if selected_for_prompt >= max_targets_per_prompt:
                    break
                if meta.state not in {CRITERION_STATE_DEAD, CRITERION_STATE_SATURATED}:
                    continue

                parent_item = state.parent_rubric[criterion_idx]
                if (
                    criterion_polarity(parent_item) == CRITERION_POLARITY_NEGATIVE
                    and meta.state == CRITERION_STATE_SATURATED
                ):
                    # A parent-preserving negative refinement has trigger(C)
                    # subset trigger(P).  If no current rollout triggers P,
                    # such a child cannot create a real within-group split.
                    continue
                parent_weight = _as_float(parent_item.get("weight", 1.0), default=1.0)
                if parent_weight == 0.0:
                    continue
                if (
                    self.config.rubric_evolution_only_example_criteria
                    and not is_example_criterion(parent_item)
                ):
                    continue

                has_unresolved_child = any(
                    child.parent_index == criterion_idx
                    and child.status in {CANDIDATE_PENDING, CANDIDATE_ACTIVE}
                    for child in state.candidates
                )
                if has_unresolved_child:
                    continue

                if (
                    meta.last_evolved_step is not None
                    and retry_cooldown > 0
                    and global_step - meta.last_evolved_step < retry_cooldown
                ):
                    continue

                targets.append(
                    EvolutionTarget(
                        prompt_key=prompt_key,
                        prompt_text=str(cache["prompt_text"]),
                        parent_index=criterion_idx,
                        parent_item=parent_item,
                        state=meta.state,
                        pass_rate=meta.pass_rate,
                        group_indices=list(cache["group_indices"]),
                        group_responses=[],
                    )
                )
                selected_for_prompt += 1
                if max_targets > 0 and len(targets) >= max_targets:
                    return targets
        return targets

    def activate_pending(self, global_step: int) -> dict[str, float]:
        delayed_steps = int(self.config.rubric_evolution_delayed_steps)
        activated = 0.0
        rejected = 0.0
        for state in self.store._states.values():
            for candidate in state.pending_children():
                if global_step - candidate.proposed_step < delayed_steps:
                    continue
                if not candidate.verify_history:
                    candidate.status = CANDIDATE_REJECTED
                    rejected += 1.0
                    continue
                last_verify = candidate.verify_history[-1]
                if not last_verify.get("passed", False):
                    candidate.status = CANDIDATE_REJECTED
                    rejected += 1.0
                    continue
                candidate.status = CANDIDATE_ACTIVE
                meta = state.criterion_meta.get(candidate.parent_index)
                if meta is not None:
                    meta.active_child_ids.append(candidate.criterion_id)
                    meta.last_evolved_step = global_step
                state.rebalance_active_child_weights(
                    candidate.parent_index,
                    self.config.rubric_evolution_child_weight_ratio,
                )
                activated += 1.0
        return {
            "rubric_evolution/activated_candidates": activated,
            "rubric_evolution/rejected_candidates": rejected,
        }

    def _score_response_on_rubric(
        self,
        *,
        prompt_text: str,
        response: str,
        rubric_items: list[dict[str, Any]],
    ) -> dict[str, float]:
        return self._score_response_requests(
            [
                {
                    "prompt_text": prompt_text,
                    "response": response,
                    "rubric": rubric_items,
                }
            ]
        )[0]

    def _score_response_requests(
        self, requests: list[dict[str, Any]]
    ) -> list[dict[str, float]]:
        if not requests:
            return []
        if self.judge_batch_fn is not None:
            return self.judge_batch_fn(requests)
        if self.judge_fn is None:
            return [{"overall": 0.0} for _ in requests]
        return [self.judge_fn(request) for request in requests]

    def judge_new_criterion_groups(self, groups: list[CandidateJudgeGroup]) -> list[list[list[float]]]:
        """Public batch API: independent criterion scores, one request per answer."""
        return judge_candidate_groups(groups, judge_batch=self._score_response_requests)

    def _parent_matches_contrast_direction(
        self,
        *,
        prompt_text: str,
        response: str,
        parent_item: dict[str, Any],
    ) -> bool:
        score = self._score_response_on_rubric(
            prompt_text=prompt_text,
            response=response,
            rubric_items=[parent_item],
        )
        criterion_score = score.get("criterion_0_score", 0.0)
        if criterion_polarity(parent_item) == CRITERION_POLARITY_NEGATIVE:
            return criterion_score <= 0.0
        return criterion_score >= 1.0 or score.get("overall", 0.0) >= 1.0

    def _verify_candidate(
        self,
        *,
        prompt_text: str,
        parent_item: dict[str, Any],
        candidate: dict[str, Any],
        source: str,
        rollout_responses: list[str],
        expected_response_count: Optional[int] = None,
        precomputed_scores: Optional[list[dict[str, float]]] = None,
    ) -> tuple[bool, dict[str, Any]]:
        candidate_item = {
            "title": candidate["title"],
            "description": candidate["description"],
            "weight": candidate["weight"],
        }
        # Empirical discrimination is measured only on the current on-policy
        # group.  A synthetic answer must never manufacture variance.
        # Preserve every member, including an empty model response: an empty
        # answer is still a real policy sample and must be judged as such.
        responses = [str(response) for response in rollout_responses]
        if (
            expected_response_count is not None
            and len(responses) != expected_response_count
        ):
            return False, {
                "passed": False,
                "reason": "incomplete_rollout_group",
                "has_training_signal": False,
                "response_count": len(responses),
                "expected_response_count": expected_response_count,
            }
        candidate_requests = [
            {
                "prompt_text": prompt_text,
                "response": response,
                "rubric": [candidate_item],
            }
            for response in responses
        ]
        candidate_score_dicts = (self._score_response_requests(candidate_requests)
                                 if precomputed_scores is None else precomputed_scores)
        if not candidate_score_dicts or len(candidate_score_dicts) != len(responses):
            return False, {
                "passed": False,
                "reason": "no_scores",
                "has_training_signal": False,
            }

        judge_errors = [
            _as_float(score.get("judge_error", 0.0), default=1.0)
            for score in candidate_score_dicts
        ]
        missing_scores = [
            idx
            for idx, score in enumerate(candidate_score_dicts)
            if "criterion_0_score" not in score
        ]
        if any(error > 0.0 for error in judge_errors) or missing_scores:
            return False, {
                "passed": False,
                "reason": "judge_error",
                "has_training_signal": False,
                "judge_errors": judge_errors,
                "missing_score_indices": missing_scores,
                "response_count": len(responses),
                "expected_response_count": expected_response_count,
            }

        candidate_scores = [
            _as_float(score["criterion_0_score"], default=0.0)
            for score in candidate_score_dicts
        ]
        candidate_hits = [1 if score >= 0.5 else 0 for score in candidate_scores]
        response_count = len(candidate_hits)
        hit_count = int(sum(candidate_hits))
        pass_rate = float(hit_count / response_count) if response_count else 0.0
        variance = float(pass_rate * (1.0 - pass_rate))
        # Require support on both sides of the split. Merely separating one
        # rollout from the other seven is usually a rollout-specific detail,
        # not a stable policy capability. Small synthetic/unit-test groups use
        # the largest feasible symmetric requirement.
        required_support = min(
            int(self.config.rubric_evolution_min_split_count),
            response_count // 2,
        )
        has_training_signal = (
            required_support >= 1
            and required_support <= hit_count <= response_count - required_support
        )
        discriminative = has_training_signal
        passed = has_training_signal
        return passed, {
            "passed": passed,
            "pass_rate": pass_rate,
            "score_vector": candidate_scores,
            "hit_vector": candidate_hits,
            "hit_count": hit_count,
            "response_count": response_count,
            "expected_response_count": expected_response_count,
            "variance": variance,
            "discriminative": discriminative,
            "has_training_signal": has_training_signal,
            "required_split_support": required_support,
            "source": source,
        }

    def verify_candidate_on_group(
        self,
        *,
        prompt_text: str,
        candidate: dict[str, Any],
        source: str,
        rollout_responses: list[str],
        expected_response_count: Optional[int] = None,
        precomputed_scores: Optional[list[dict[str, float]]] = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Empirical gate; optional batch scores avoid a second Judge request."""

        return self._verify_candidate(
            prompt_text=prompt_text,
            parent_item={},
            candidate=candidate,
            source=source,
            rollout_responses=rollout_responses,
            expected_response_count=expected_response_count,
            precomputed_scores=precomputed_scores,
        )

    def add_active_gap_root(
        self,
        *,
        prompt_key: str,
        prompt_text: str,
        candidate: dict[str, Any],
        global_step: int,
        semantic_verifier_info: dict[str, Any],
        empirical_verify_info: dict[str, Any],
        group_indices: Optional[list[int]] = None,
    ) -> bool:
        """Append one verified GAP_EXPAND candidate as a persistent root."""

        state = self.store.get(prompt_key)
        if state is None:
            return False
        title = " ".join(str(candidate.get("title") or "").split()).strip()
        description = " ".join(str(candidate.get("description") or "").split()).strip()
        if not title or not description:
            return False
        candidate_key = (title.casefold(), description.casefold())
        for item in state.build_active_rubric():
            existing_key = (
                " ".join(str(item.get("title") or "").split()).casefold(),
                " ".join(str(item.get("description") or "").split()).casefold(),
            )
            if existing_key == candidate_key:
                return False

        root_item = {
            "title": title,
            "description": description,
            "weight": _as_float(candidate.get("weight", 1.0), default=1.0),
            "criterion_id": f"root_{uuid.uuid4().hex[:12]}",
            "parent_id": "",
            "source": "gap_expand",
            "proposed_step": global_step,
        }
        state.parent_rubric.append(root_item)
        self.cache_current_criterion_scores(
            prompt_key=prompt_key,
            criterion_id=root_item["criterion_id"],
            group_indices=list(group_indices or []),
            score_vector=list(empirical_verify_info.get("score_vector", [])),
        )
        self._record_evolution_example(
            {
                "step": global_step,
                "prompt": _truncate_for_log(prompt_text, 400),
                "parent_state": "gap_expand",
                "parent_title": "",
                "parent_description": "",
                "parent_weight": None,
                "parent_polarity": None,
                "parent_pass_rate": None,
                "outcome": "accepted",
                "source": "gap_expand",
                "child_title": title,
                "child_description": description,
                "candidate_status": CANDIDATE_ACTIVE,
                "verify_pass_rate": float(empirical_verify_info.get("pass_rate", 0.0)),
                "verify_detail": "accepted",
                "candidate_score_vector": empirical_verify_info.get("score_vector", []),
                "semantic_verify_reason": semantic_verifier_info.get("reason", ""),
                "semantic_verify_checks": semantic_verifier_info.get("checks", {}),
            }
        )
        return True

    def evolve_targets(
        self,
        targets: list[EvolutionTarget],
        *,
        global_step: int,
        contrast_responses: dict[str, str],
        proposal_payloads: dict[tuple[str, int], str],
        proposal_verifier_payloads: Optional[dict[tuple[str, int, int], str]] = None,
        parent_novelty_verifier_payloads: Optional[
            dict[tuple[str, int, int], str]
        ] = None,
        polarity_verifier_payloads: Optional[dict[tuple[str, int, int], str]] = None,
        collision_verifier_payloads: Optional[dict[tuple[str, int, int], str]] = None,
        empirical_verifier_payloads: Optional[dict[tuple[str, int, int], dict[str, Any]]] = None,
        activate_immediately: bool = False,
    ) -> dict[str, float]:
        metrics = {
            "rubric_evolution/targets": float(len(targets)),
            "rubric_evolution/proposed_candidates": 0.0,
            "rubric_evolution/accepted_candidates": 0.0,
            "rubric_evolution/skipped_no_contrast": 0.0,
            "rubric_evolution/dead_contrast_parent_failed": 0.0,
            "rubric_evolution/decompose_proposed": 0.0,
            "rubric_evolution/decompose_accepted": 0.0,
            "rubric_evolution/refine_proposed": 0.0,
            "rubric_evolution/refine_accepted": 0.0,
            "rubric_evolution/verify_failed_no_signal": 0.0,
            "rubric_evolution/verify_failed_not_parent_preserving": 0.0,
            "rubric_evolution/verifier_candidates": 0.0,
            "rubric_evolution/verifier_rejected": 0.0,
            "rubric_evolution/verifier_failed_invalid_json": 0.0,
            "rubric_evolution/verify_failed_parent_drift": 0.0,
            "rubric_evolution/verify_failed_invalid_relation": 0.0,
            "rubric_evolution/verify_failed_polarity_mismatch": 0.0,
            "rubric_evolution/verify_failed_parent_paraphrase": 0.0,
            "rubric_evolution/verify_failed_redundant": 0.0,
            "rubric_evolution/verify_failed_not_judgeable": 0.0,
            "rubric_evolution/verify_failed_unsupported_fact": 0.0,
            "rubric_evolution/verify_failed_trivial": 0.0,
            "rubric_evolution/verify_failed_length_style": 0.0,
            "rubric_evolution/verify_failed_rollout_overfit": 0.0,
            "rubric_evolution/verify_failed_unsupported_numeric": 0.0,
            "rubric_evolution/verify_failed_proposal_evidence": 0.0,
            "rubric_evolution/collision_verifier_candidates": 0.0,
            "rubric_evolution/collision_verifier_rejected": 0.0,
            "rubric_evolution/negative_polarity_verifier_candidates": 0.0,
            "rubric_evolution/negative_polarity_verifier_rejected": 0.0,
            "rubric_evolution/parent_novelty_verifier_candidates": 0.0,
            "rubric_evolution/parent_novelty_verifier_rejected": 0.0,
        }
        accepted_pass_rates: list[float] = []

        for target in targets:
            state = self.store.get(target.prompt_key)
            cache = self._last_group_cache.get(target.prompt_key)
            if state is None or cache is None:
                continue

            # Kept in the event schema for compatibility with old runs.  New
            # evolution does not generate or score synthetic contrast answers.
            contrast_response = contrast_responses.get(target.prompt_key, "")
            contrast_parent_ok: Optional[bool] = None

            proposal_text = proposal_payloads.get(
                (target.prompt_key, target.parent_index), ""
            )
            if not proposal_text.strip():
                self._record_evolution_example(
                    {
                        **self._build_evolution_example(
                            target,
                            global_step=global_step,
                            contrast_response=contrast_response,
                            contrast_parent_ok=contrast_parent_ok,
                        ),
                        "outcome": "skipped_no_proposal",
                        "source": (
                            "decompose"
                            if target.state == CRITERION_STATE_DEAD
                            else "refine"
                        ),
                        "child_title": "",
                        "child_description": "",
                        "verify_pass_rate": None,
                        "verify_detail": "missing_proposal",
                    }
                )
                continue

            max_items = (
                self.config.rubric_evolution_max_subcriteria
                if target.state == CRITERION_STATE_DEAD
                else self.config.rubric_evolution_max_refinements
            )
            try:
                proposed_items = parse_proposed_criteria(
                    proposal_text, max_items=max_items
                )
            except json.JSONDecodeError:
                self._record_evolution_example(
                    {
                        **self._build_evolution_example(
                            target,
                            global_step=global_step,
                            contrast_response=contrast_response,
                            contrast_parent_ok=contrast_parent_ok,
                        ),
                        "outcome": "parse_failed",
                        "source": (
                            "decompose"
                            if target.state == CRITERION_STATE_DEAD
                            else "refine"
                        ),
                        "child_title": "",
                        "child_description": _truncate_for_log(proposal_text, 400),
                        "verify_pass_rate": None,
                        "verify_detail": "invalid_proposal_json",
                    }
                )
                continue

            source = "decompose" if target.state == CRITERION_STATE_DEAD else "refine"
            parent_weight = _as_float(
                target.parent_item.get("weight", 1.0), default=1.0
            )
            child_magnitude = (
                abs(parent_weight) * self.config.rubric_evolution_child_weight_ratio
            )
            child_weight = -child_magnitude if parent_weight < 0 else child_magnitude

            for item_idx, item in enumerate(proposed_items):
                metrics["rubric_evolution/proposed_candidates"] += 1.0
                if source == "decompose":
                    metrics["rubric_evolution/decompose_proposed"] += 1.0
                else:
                    metrics["rubric_evolution/refine_proposed"] += 1.0
                candidate_dict = {
                    "title": item["title"],
                    "description": item["description"],
                    "weight": child_weight,
                }
                semantic_verifier_info: dict[str, Any] = {
                    "valid": True,
                    "reason": "meta_verifier_not_provided",
                }
                if proposal_verifier_payloads is not None:
                    verifier_payload = proposal_verifier_payloads.get(
                        (target.prompt_key, target.parent_index, item_idx),
                        "",
                    )
                    semantic_verifier_info = parse_meta_verifier_verdict(
                        verifier_payload,
                        operation_type=CHILD_EVOLUTION,
                    )
                    if (
                        semantic_verifier_info.get("reason")
                        == "meta_verifier_operation_mismatch"
                    ):
                        # Compatibility path for checkpoints/configs that still
                        # use the former generic nine-dimension verifier.
                        semantic_verifier_info = parse_candidate_verifier_verdict(
                            verifier_payload
                        )
                    metrics["rubric_evolution/verifier_candidates"] += 1.0
                    if not semantic_verifier_info.get("valid", False):
                        metrics["rubric_evolution/verifier_rejected"] += 1.0
                        rejection_type = str(
                            semantic_verifier_info.get("rejection_type")
                            or "verifier_rejected"
                        )
                        if rejection_type == "invalid_json":
                            metrics[
                                "rubric_evolution/verifier_failed_invalid_json"
                            ] += 1.0
                        elif rejection_type == "parent_drift":
                            metrics[
                                "rubric_evolution/verify_failed_parent_drift"
                            ] += 1.0
                        elif rejection_type == "invalid_relation":
                            metrics[
                                "rubric_evolution/verify_failed_invalid_relation"
                            ] += 1.0
                        elif rejection_type == "polarity_mismatch":
                            metrics[
                                "rubric_evolution/verify_failed_polarity_mismatch"
                            ] += 1.0
                        elif rejection_type == "parent_paraphrase":
                            metrics[
                                "rubric_evolution/verify_failed_parent_paraphrase"
                            ] += 1.0
                            metrics["rubric_evolution/verify_failed_redundant"] += 1.0
                        elif rejection_type == "redundant":
                            metrics["rubric_evolution/verify_failed_redundant"] += 1.0
                        elif rejection_type == "not_judgeable":
                            metrics[
                                "rubric_evolution/verify_failed_not_judgeable"
                            ] += 1.0
                        elif rejection_type == "unsupported_fact":
                            metrics[
                                "rubric_evolution/verify_failed_unsupported_fact"
                            ] += 1.0
                        elif rejection_type == "trivial":
                            metrics["rubric_evolution/verify_failed_trivial"] += 1.0
                        elif rejection_type == "length_style":
                            metrics[
                                "rubric_evolution/verify_failed_length_style"
                            ] += 1.0
                        elif rejection_type == "rollout_overfit":
                            metrics[
                                "rubric_evolution/verify_failed_rollout_overfit"
                            ] += 1.0
                        verify_detail = (
                            f"verifier_{rejection_type}:"
                            f"{_truncate_for_log(semantic_verifier_info.get('reason', ''), 160)}"
                        )
                        self._record_evolution_example(
                            {
                                **self._build_evolution_example(
                                    target,
                                    global_step=global_step,
                                    contrast_response=contrast_response,
                                    contrast_parent_ok=contrast_parent_ok,
                                ),
                                "outcome": "rejected",
                                "source": source,
                                "child_title": item["title"],
                                "child_description": item["description"],
                                "verify_pass_rate": None,
                                "verify_detail": verify_detail,
                                "semantic_verify_checks": semantic_verifier_info.get(
                                    "checks", {}
                                ),
                                "semantic_failed_dimensions": semantic_verifier_info.get(
                                    "failed_dimensions", []
                                ),
                                "semantic_uncertain_dimensions": semantic_verifier_info.get(
                                    "uncertain_dimensions", []
                                ),
                            }
                        )
                        continue

                parent_novelty_info: dict[str, Any] = {
                    "valid": True,
                    "reason": "parent_novelty_verifier_not_provided",
                }
                if parent_novelty_verifier_payloads is not None:
                    novelty_payload = parent_novelty_verifier_payloads.get(
                        (target.prompt_key, target.parent_index, item_idx),
                        "",
                    )
                    parent_novelty_info = parse_parent_novelty_verdict(
                        novelty_payload,
                        source=source,
                        parent_text=str(target.parent_item.get("description") or ""),
                        candidate_text=str(item.get("description") or ""),
                    )
                    metrics[
                        "rubric_evolution/parent_novelty_verifier_candidates"
                    ] += 1.0
                    if not parent_novelty_info.get("valid", False):
                        metrics[
                            "rubric_evolution/parent_novelty_verifier_rejected"
                        ] += 1.0
                        metrics[
                            "rubric_evolution/verify_failed_parent_paraphrase"
                        ] += 1.0
                        metrics["rubric_evolution/verify_failed_redundant"] += 1.0
                        self._record_evolution_example(
                            {
                                **self._build_evolution_example(
                                    target,
                                    global_step=global_step,
                                ),
                                "outcome": "rejected",
                                "source": source,
                                "child_title": item["title"],
                                "child_description": item["description"],
                                "verify_pass_rate": None,
                                "verify_detail": "parent_novelty:"
                                + _truncate_for_log(
                                    parent_novelty_info.get("reason", ""), 180
                                ),
                            }
                        )
                        continue

                negative_polarity_info: dict[str, Any] = {
                    "valid": True,
                    "reason": "not_negative_or_verifier_not_provided",
                }
                if (
                    criterion_polarity(target.parent_item)
                    == CRITERION_POLARITY_NEGATIVE
                    and polarity_verifier_payloads is not None
                ):
                    polarity_payload = polarity_verifier_payloads.get(
                        (target.prompt_key, target.parent_index, item_idx),
                        "",
                    )
                    negative_polarity_info = parse_negative_polarity_verdict(
                        polarity_payload
                    )
                    metrics[
                        "rubric_evolution/negative_polarity_verifier_candidates"
                    ] += 1.0
                    if not negative_polarity_info.get("valid", False):
                        metrics[
                            "rubric_evolution/negative_polarity_verifier_rejected"
                        ] += 1.0
                        metrics[
                            "rubric_evolution/verify_failed_polarity_mismatch"
                        ] += 1.0
                        self._record_evolution_example(
                            {
                                **self._build_evolution_example(
                                    target,
                                    global_step=global_step,
                                ),
                                "outcome": "rejected",
                                "source": source,
                                "child_title": item["title"],
                                "child_description": item["description"],
                                "verify_pass_rate": None,
                                "verify_detail": "negative_polarity:"
                                + _truncate_for_log(
                                    negative_polarity_info.get("reason", ""), 180
                                ),
                            }
                        )
                        continue

                collision_verifier_info: dict[str, Any] = {
                    "valid": True,
                    "reason": "collision_verifier_not_provided",
                }
                if collision_verifier_payloads is not None:
                    collision_payload = collision_verifier_payloads.get(
                        (target.prompt_key, target.parent_index, item_idx),
                        "",
                    )
                    collision_verifier_info = parse_collision_verifier_verdict(
                        collision_payload
                    )
                    metrics["rubric_evolution/collision_verifier_candidates"] += 1.0
                    if not collision_verifier_info.get("valid", False):
                        metrics["rubric_evolution/collision_verifier_rejected"] += 1.0
                        metrics["rubric_evolution/verify_failed_redundant"] += 1.0
                        verify_detail = (
                            "collision:"
                            f"{_truncate_for_log(collision_verifier_info.get('reason', ''), 160)}"
                        )
                        self._record_evolution_example(
                            {
                                **self._build_evolution_example(
                                    target,
                                    global_step=global_step,
                                ),
                                "outcome": "rejected",
                                "source": source,
                                "child_title": item["title"],
                                "child_description": item["description"],
                                "verify_pass_rate": None,
                                "verify_detail": verify_detail,
                                "collision_ids": collision_verifier_info.get(
                                    "conflict_ids", []
                                ),
                            }
                        )
                        continue

                if empirical_verifier_payloads is not None:
                    # Only the trusted evaluate-before-commit RG path supplies
                    # these results. No second Judge call on the winning patch.
                    empirical_key = (target.prompt_key, target.parent_index, item_idx)
                    if empirical_key not in empirical_verifier_payloads:
                        raise ValueError("Missing pre-evaluated RG criterion result.")
                    empirical_verify_info = deepcopy(empirical_verifier_payloads[empirical_key])
                    if len(empirical_verify_info.get("score_vector", [])) != len(target.group_responses):
                        raise ValueError("Pre-evaluated RG score vector does not match the fixed Solver group.")
                    passed = bool(empirical_verify_info.get("passed", False))
                else:
                    passed, empirical_verify_info = self._verify_candidate(
                        prompt_text=target.prompt_text,
                        parent_item=target.parent_item,
                        candidate=candidate_dict,
                        source=source,
                        rollout_responses=target.group_responses,
                        expected_response_count=(
                            len(target.group_indices)
                            if target.group_indices
                            else len(target.group_responses)
                        ),
                    )
                empirical_verify_info["semantic_verifier"] = semantic_verifier_info
                empirical_verify_info["parent_novelty_verifier"] = parent_novelty_info
                empirical_verify_info["negative_polarity_verifier"] = (
                    negative_polarity_info
                )
                empirical_verify_info["collision_verifier"] = collision_verifier_info
                if not passed:
                    if not empirical_verify_info.get("has_training_signal", False):
                        metrics["rubric_evolution/verify_failed_no_signal"] += 1.0
                        verify_detail = "no_signal"
                    else:
                        verify_detail = "verify_failed"
                    self._record_evolution_example(
                        {
                            **self._build_evolution_example(
                                target,
                                global_step=global_step,
                                contrast_response=contrast_response,
                                contrast_parent_ok=contrast_parent_ok,
                            ),
                            "outcome": "rejected",
                            "source": source,
                            "child_title": item["title"],
                            "child_description": item["description"],
                            "verify_pass_rate": float(
                                empirical_verify_info.get("pass_rate", 0.0)
                            ),
                            "verify_detail": verify_detail,
                            "candidate_score_vector": empirical_verify_info.get(
                                "score_vector", []
                            ),
                        }
                    )
                    continue

                active_rubric_before = state.build_active_rubric()
                candidate_status = (
                    CANDIDATE_ACTIVE if activate_immediately else CANDIDATE_PENDING
                )
                candidate = EvolvedCriterion(
                    criterion_id=f"child_{uuid.uuid4().hex[:12]}",
                    parent_index=target.parent_index,
                    parent_criterion_id=str(
                        target.parent_item.get("criterion_id")
                        or f"parent_{target.parent_index}"
                    ),
                    title=item["title"],
                    description=item["description"],
                    weight=child_weight,
                    source=source,
                    status=candidate_status,
                    proposed_step=global_step,
                    verify_history=[empirical_verify_info],
                )
                state.candidates.append(candidate)
                self.cache_current_criterion_scores(
                    prompt_key=target.prompt_key,
                    criterion_id=candidate.criterion_id,
                    group_indices=target.group_indices,
                    score_vector=list(empirical_verify_info.get("score_vector", [])),
                )
                if activate_immediately:
                    meta = state.criterion_meta.get(target.parent_index)
                    if meta is not None:
                        meta.active_child_ids.append(candidate.criterion_id)
                    state.rebalance_active_child_weights(
                        target.parent_index,
                        self.config.rubric_evolution_child_weight_ratio,
                    )
                replacement_children = [
                    child.to_rubric_item()
                    for child in state.active_children_for_parent(target.parent_index)
                ]
                active_rubric_after = state.build_active_rubric()
                metrics["rubric_evolution/accepted_candidates"] += 1.0
                if source == "decompose":
                    metrics["rubric_evolution/decompose_accepted"] += 1.0
                else:
                    metrics["rubric_evolution/refine_accepted"] += 1.0
                accepted_pass_rates.append(
                    float(empirical_verify_info.get("pass_rate", 0.0))
                )
                self._record_evolution_example(
                    {
                        **self._build_evolution_example(
                            target,
                            global_step=global_step,
                            contrast_response=contrast_response,
                            contrast_parent_ok=contrast_parent_ok,
                        ),
                        "outcome": "accepted",
                        "source": source,
                        "child_title": item["title"],
                        "child_description": item["description"],
                        "candidate_status": candidate_status,
                        "verify_pass_rate": float(
                            empirical_verify_info.get("pass_rate", 0.0)
                        ),
                        "verify_detail": "accepted",
                        "candidate_score_vector": empirical_verify_info.get(
                            "score_vector", []
                        ),
                        "semantic_verify_reason": semantic_verifier_info.get(
                            "reason", ""
                        ),
                        "semantic_verify_checks": semantic_verifier_info.get(
                            "checks", {}
                        ),
                        "semantic_failed_dimensions": semantic_verifier_info.get(
                            "failed_dimensions", []
                        ),
                        "semantic_uncertain_dimensions": semantic_verifier_info.get(
                            "uncertain_dimensions", []
                        ),
                        "collision_verify_reason": collision_verifier_info.get(
                            "reason", ""
                        ),
                        "active_rubric_before_count": len(active_rubric_before),
                        "active_rubric_before": format_complete_rubric(
                            active_rubric_before
                        ),
                        "active_rubric_replaced_parent": "",
                        "active_rubric_added": (
                            format_complete_rubric([candidate.to_rubric_item()])
                            if candidate_status == CANDIDATE_ACTIVE
                            else ""
                        ),
                        "active_rubric_replacement_children": (
                            format_complete_rubric(replacement_children)
                            if candidate_status == CANDIDATE_ACTIVE
                            else ""
                        ),
                        "active_rubric_after_count": len(active_rubric_after),
                        "active_rubric_after": format_complete_rubric(
                            active_rubric_after
                        ),
                    }
                )

            meta = state.criterion_meta.get(target.parent_index)
            if meta is not None:
                meta.last_evolved_step = global_step

        proposed = metrics["rubric_evolution/proposed_candidates"]
        if proposed > 0:
            metrics["rubric_evolution/acceptance_rate"] = (
                metrics["rubric_evolution/accepted_candidates"] / proposed
            )
        if accepted_pass_rates:
            metrics["rubric_evolution/accepted_candidate_pass_rate_mean"] = float(
                np.mean(accepted_pass_rates)
            )
        metrics.update(self.collect_global_metrics())
        return metrics
