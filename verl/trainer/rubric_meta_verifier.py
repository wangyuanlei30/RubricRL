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
"""Operation-aware semantic verification for evolving rubric candidates.

This module deliberately does not inspect rollout scores or pass rates.  It
audits only whether a proposed criterion is semantically and structurally valid.
Empirical discriminativeness remains the responsibility of RubricEvolver's
independent pass-rate gate.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


GAP_EXPAND = "gap_expand"
CHILD_EVOLUTION = "child_evolution"

GAP_META_VERIFY_DIMENSIONS = (
    "target_relevance",
    "polarity_consistency",
    "factual_grounding",
    "measurability",
    "global_non_duplication",
    "specificity_and_non_triviality",
    "style_relevance",
    "answer_independence",
    "atomicity",
)

CHILD_META_VERIFY_DIMENSIONS = (
    "target_relevance",
    "parent_relation",
    "operation_consistency",
    "polarity_consistency",
    "factual_grounding",
    "measurability",
    "non_paraphrase",
    "branch_non_collision",
    "specificity_and_non_triviality",
    "style_relevance",
    "answer_independence",
    "atomicity",
)

_DIMENSIONS_BY_OPERATION = {
    GAP_EXPAND: GAP_META_VERIFY_DIMENSIONS,
    CHILD_EVOLUTION: CHILD_META_VERIFY_DIMENSIONS,
}

_REJECTION_TYPE_BY_DIMENSION = {
    "target_relevance": "parent_drift",
    "parent_relation": "invalid_relation",
    "operation_consistency": "invalid_relation",
    "polarity_consistency": "polarity_mismatch",
    "factual_grounding": "unsupported_fact",
    "measurability": "not_judgeable",
    "global_non_duplication": "redundant",
    "non_paraphrase": "parent_paraphrase",
    "branch_non_collision": "redundant",
    "specificity_and_non_triviality": "trivial",
    "style_relevance": "length_style",
    "answer_independence": "rollout_overfit",
    "atomicity": "not_judgeable",
}


def meta_verify_dimensions(operation_type: str) -> tuple[str, ...]:
    """Return the exact semantic audit dimensions for one operation."""

    try:
        return _DIMENSIONS_BY_OPERATION[operation_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported rubric meta-verify operation: {operation_type!r}."
        ) from exc


def _extract_json_value(text: str) -> Any:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    starts = [
        position
        for position in (stripped.find("{"), stripped.find("["))
        if position >= 0
    ]
    if not starts:
        raise json.JSONDecodeError("No JSON object or array found", stripped, 0)
    value, _ = json.JSONDecoder().raw_decode(stripped[min(starts) :])
    return value


def _invalid_result_payload(candidate_id: str, operation_type: str, reason: str) -> str:
    return json.dumps(
        {
            "candidate_id": candidate_id,
            "operation_type": operation_type,
            "checks": {},
            "final_verdict": "INVALID",
            "failed_dimensions": [],
            "final_reason": reason,
            "_routing_error": reason,
        },
        ensure_ascii=False,
    )


def route_batched_meta_verifier_payloads(
    text: str,
    *,
    expected_candidate_ids: list[str],
    operation_type: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Route a batched response without ever relying on result ordering.

    The router fails closed for missing or duplicated candidates.  An unknown id
    invalidates the entire batch because the remaining assignments are no longer
    trustworthy.
    """

    meta_verify_dimensions(operation_type)
    expected_ids = [str(candidate_id) for candidate_id in expected_candidate_ids]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected meta-verifier candidate ids must be unique.")

    routed = {
        candidate_id: _invalid_result_payload(
            candidate_id,
            operation_type,
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
        "operation_mismatch_candidate_ids": [],
    }

    try:
        payload = _extract_json_value(text)
    except (json.JSONDecodeError, TypeError):
        diagnostics["reason"] = "invalid_batched_meta_verifier_json"
        return routed, diagnostics

    raw_results = payload if isinstance(payload, list) else payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(raw_results, list):
        diagnostics["reason"] = "missing_batched_meta_results_list"
        return routed, diagnostics

    diagnostics["parse_valid"] = True
    diagnostics["returned_count"] = len(raw_results)
    results_by_id: dict[str, list[dict[str, Any]]] = {}
    unknown_ids: list[str] = []
    operation_mismatch_ids: list[str] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            unknown_ids.append("<non_object>")
            continue
        candidate_id = str(raw_result.get("candidate_id") or "").strip()
        if candidate_id not in routed:
            unknown_ids.append(candidate_id or "<missing>")
            continue
        returned_operation = str(raw_result.get("operation_type") or "").strip()
        # An omitted operation_type is routed for compatibility with the old
        # generic verifier. The candidate parser will accept it only if it also
        # matches that complete legacy schema. An explicitly wrong operation is
        # always rejected here.
        if returned_operation and returned_operation != operation_type:
            operation_mismatch_ids.append(candidate_id)
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
            routed[candidate_id] = _invalid_result_payload(
                candidate_id,
                operation_type,
                "duplicate_batched_meta_verifier_result",
            )

    for candidate_id in operation_mismatch_ids:
        routed[candidate_id] = _invalid_result_payload(
            candidate_id,
            operation_type,
            "meta_verifier_operation_mismatch",
        )

    missing_ids = [
        candidate_id
        for candidate_id in expected_ids
        if candidate_id not in results_by_id
        and candidate_id not in operation_mismatch_ids
    ]
    unique_unknown_ids = list(dict.fromkeys(unknown_ids))
    if unique_unknown_ids:
        routed = {
            candidate_id: _invalid_result_payload(
                candidate_id,
                operation_type,
                "unknown_candidate_id_in_batched_meta_verifier_result",
            )
            for candidate_id in expected_ids
        }
    diagnostics.update(
        {
            "missing_candidate_ids": missing_ids,
            "duplicate_candidate_ids": duplicate_ids,
            "unknown_candidate_ids": unique_unknown_ids,
            "operation_mismatch_candidate_ids": list(
                dict.fromkeys(operation_mismatch_ids)
            ),
        }
    )
    return routed, diagnostics


def _normalize_compact_verdict(verdict: str) -> tuple[str, str]:
    """Canonicalize case/spacing only: " invalid : Factual Grounding" -> ("INVALID", "factual_grounding").

    The status and dimension themselves are never inferred or remapped.
    """
    status, _, category = verdict.partition(":")
    category = re.sub(r"[\s\-]+", "_", category.strip().lower())
    return status.strip().upper(), category


PEER_CANDIDATES_RULE = ("Same-completion peer candidates (not accepted rubric). Reject redundant "
                        "later candidates; preserve expected parent overlap. Do not evaluate pass rates:")


def format_peer_candidates_hint(peers: list[dict[str, Any]]) -> str:
    """Peer section shared by batched and single-candidate audits."""

    return PEER_CANDIDATES_RULE + "\n" + json.dumps(peers, ensure_ascii=False)


def parse_meta_verifier_verdict(text: str, *, operation_type: str) -> dict[str, Any]:
    """Validate one operation-specific semantic verdict and fail closed."""

    dimensions = meta_verify_dimensions(operation_type)
    try:
        payload = _extract_json_value(text)
    except (json.JSONDecodeError, TypeError):
        return {
            "valid": False,
            "reason": "invalid_verifier_json",
            "rejection_type": "invalid_json",
        }
    if not isinstance(payload, dict):
        return {
            "valid": False,
            "reason": "verifier_payload_is_not_an_object",
            "rejection_type": "invalid_json",
        }
    if payload.get("_routing_error"):
        return {
            "valid": False,
            "reason": str(payload["_routing_error"]),
            "rejection_type": "invalid_json",
        }
    if "verdict" in payload:
        allowed = {"candidate_id", "verdict", "operation_type"}
        verdict = payload.get("verdict")
        if (not set(payload) <= allowed or not isinstance(verdict, str)
                or not isinstance(payload.get("candidate_id"), str)
                or not payload["candidate_id"].strip()
                or payload.get("operation_type", operation_type) != operation_type):
            return {"valid": False, "reason": "invalid_compact_schema", "rejection_type": "invalid_json"}
        status, category = _normalize_compact_verdict(verdict)
        if status == "VALID" and not category:
            return {"valid": True, "reason": "VALID", "checks": {},
                    "final_verdict": "VALID", "failed_dimensions": [], "uncertain_dimensions": [],
                    "response_format": "compact"}
        verdict = f"{status}: {category}"
        if status not in {"INVALID", "REVIEW"} or category not in dimensions:
            return {"valid": False, "reason": "invalid_compact_verdict", "rejection_type": "invalid_json"}
        return {"valid": False, "reason": verdict, "checks": {}, "final_verdict": status,
                "failed_dimensions": [category] if status == "INVALID" else [],
                "uncertain_dimensions": [category] if status == "REVIEW" else [],
                "rejection_type": _REJECTION_TYPE_BY_DIMENSION[category] if status == "INVALID" else "uncertain",
                "response_format": "compact"}
    if str(payload.get("operation_type") or "").strip() != operation_type:
        return {
            "valid": False,
            "reason": "meta_verifier_operation_mismatch",
            "rejection_type": "invalid_json",
        }

    raw_checks = payload.get("checks")
    if not isinstance(raw_checks, dict):
        return {
            "valid": False,
            "reason": "invalid_multidimensional_checks",
            "rejection_type": "invalid_json",
        }

    missing_dimensions = [
        dimension for dimension in dimensions if dimension not in raw_checks
    ]
    extra_dimensions = [
        dimension for dimension in raw_checks if dimension not in dimensions
    ]
    normalized_checks: dict[str, dict[str, Any]] = {}
    failed_dimensions: list[str] = []
    uncertain_dimensions: list[str] = []
    invalid_dimensions: list[str] = []
    for dimension in dimensions:
        raw_check = raw_checks.get(dimension)
        if not isinstance(raw_check, dict):
            if dimension not in missing_dimensions:
                invalid_dimensions.append(dimension)
            continue
        verdict = str(raw_check.get("verdict") or "").strip().upper()
        reason = str(raw_check.get("reason") or "").strip()
        normalized = {"verdict": verdict, "reason": reason}
        if "p_pass" in raw_check:
            p = raw_check["p_pass"]
            if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
                invalid_dimensions.append(dimension)
            else:
                normalized["p_pass"] = p
        if dimension == "factual_grounding":
            introduced_numbers = raw_check.get("introduced_numbers", [])
            if not isinstance(introduced_numbers, list):
                invalid_dimensions.append(dimension)
                introduced_numbers = []
            normalized["introduced_numbers"] = introduced_numbers
        normalized_checks[dimension] = normalized
        if verdict not in {"PASS", "FAIL", "UNCERTAIN"} or not reason:
            invalid_dimensions.append(dimension)
        elif verdict == "FAIL":
            failed_dimensions.append(dimension)
        elif verdict == "UNCERTAIN":
            uncertain_dimensions.append(dimension)

    if missing_dimensions or extra_dimensions or invalid_dimensions:
        return {
            "valid": False,
            "reason": "invalid_multidimensional_checks",
            "rejection_type": "invalid_json",
            "checks": normalized_checks,
            "missing_dimensions": missing_dimensions,
            "extra_dimensions": extra_dimensions,
            "invalid_dimensions": list(dict.fromkeys(invalid_dimensions)),
        }

    final_verdict = str(payload.get("final_verdict") or "").strip().upper()
    expected_final_verdict = (
        "INVALID"
        if failed_dimensions
        else "REVIEW" if uncertain_dimensions else "VALID"
    )
    reported_failed_dimensions = payload.get("failed_dimensions")
    if (
        final_verdict != expected_final_verdict
        or not isinstance(reported_failed_dimensions, list)
        or set(str(item) for item in reported_failed_dimensions)
        != set(failed_dimensions)
    ):
        return {
            "valid": False,
            "reason": "inconsistent_meta_verifier_aggregation",
            "rejection_type": "invalid_json",
            "checks": normalized_checks,
            "failed_dimensions": failed_dimensions,
            "uncertain_dimensions": uncertain_dimensions,
            "model_final_verdict": final_verdict,
        }

    rejection_dimension = (
        failed_dimensions[0]
        if failed_dimensions
        else uncertain_dimensions[0] if uncertain_dimensions else None
    )
    return {
        "valid": expected_final_verdict == "VALID",
        "reason": str(payload.get("final_reason") or expected_final_verdict).strip(),
        "rejection_type": (
            _REJECTION_TYPE_BY_DIMENSION.get(rejection_dimension, "verifier_rejected")
            if rejection_dimension is not None
            else None
        ),
        "operation_type": operation_type,
        "checks": normalized_checks,
        "failed_dimensions": failed_dimensions,
        "uncertain_dimensions": uncertain_dimensions,
        "model_final_verdict": final_verdict,
    }


@dataclass(frozen=True)
class RubricMetaVerifierRouter:
    """One interface that selects prompt and schema by evolution operation."""

    gap_template_path: Optional[str]
    child_template_path: Optional[str]

    def template_for(self, operation_type: str) -> str:
        meta_verify_dimensions(operation_type)
        path = (
            self.gap_template_path
            if operation_type == GAP_EXPAND
            else self.child_template_path
        )
        if not path:
            raise ValueError(
                f"No meta-verifier template configured for {operation_type!r}."
            )
        return str(Path(path))

    def build_batch_hint(
        self,
        *,
        operation_type: str,
        existing_rubric: list[dict[str, Any]],
        candidate_records: list[dict[str, Any]],
    ) -> str:
        dimensions = meta_verify_dimensions(operation_type)
        if not candidate_records:
            raise ValueError(
                "Batched rubric meta-verifier requires at least one candidate."
            )
        candidate_ids = [
            str(record.get("candidate_id") or "") for record in candidate_records
        ]
        if any(not candidate_id for candidate_id in candidate_ids):
            raise ValueError(
                "Every meta-verifier candidate requires a non-empty candidate_id."
            )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(
                "Meta-verifier candidate ids must be unique within a batch."
            )

        context = {
            "operation_type": operation_type,
            "required_dimensions": list(dimensions),
            "existing_rubric": existing_rubric,
            "candidates": candidate_records,
        }
        return "\n\n".join(
            [
                "Required output routing:\n"
                f"Expected candidate count: {len(candidate_ids)}\n"
                "Expected Candidate IDs in exact output order: "
                f"{json.dumps(candidate_ids, ensure_ascii=False)}\n"
                f"Required operation_type: {operation_type}\n"
                "Return the exact ID sequence once each; do not add, omit, rename, "
                "or reorder IDs.",
                "Semantic verification context (JSON):\n"
                + json.dumps(context, ensure_ascii=False, indent=2),
            ]
        )

    def route_batch_response(
        self,
        text: str,
        *,
        expected_candidate_ids: list[str],
        operation_type: str,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        return route_batched_meta_verifier_payloads(
            text,
            expected_candidate_ids=expected_candidate_ids,
            operation_type=operation_type,
        )

    def parse_candidate(self, text: str, *, operation_type: str) -> dict[str, Any]:
        return parse_meta_verifier_verdict(text, operation_type=operation_type)
