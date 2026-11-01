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
"""EvoRubric-style horizontal coverage expansion utilities.

The proposer inspects one fixed on-policy rollout group and the complete active
rubric, then proposes missing top-level dimensions.  Semantic verification and
empirical pass-rate verification are deliberately handled by separate stages in
the trainer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .rubric_evolution import _extract_json_object, parse_proposed_criteria


GAP_PROPOSAL_OPERATION = "gap_expand"


@dataclass(frozen=True)
class GapExpansionContext:
    prompt_key: str
    prompt_text: str
    current_rubric: list[dict[str, Any]]
    group_indices: list[int]
    group_responses: list[str]


def build_gap_proposal_hint(
    context: GapExpansionContext,
    *,
    max_candidates: int,
) -> str:
    """Build a complete, non-truncated horizontal coverage-discovery context."""

    if max_candidates < 1:
        raise ValueError("Gap proposal max_candidates must be at least one.")
    rubric_rows = [
        {
            "criterion_index": index,
            "title": str(item.get("title") or ""),
            "description": str(item.get("description") or ""),
            "weight": item.get("weight", 1.0),
            "criterion_id": str(item.get("criterion_id") or ""),
            "parent_id": str(item.get("parent_id") or ""),
        }
        for index, item in enumerate(context.current_rubric)
        if isinstance(item, dict)
    ]
    response_rows = [
        {"response_id": f"r{index}", "response": str(response)}
        for index, response in enumerate(context.group_responses)
    ]
    payload = {
        "operation": GAP_PROPOSAL_OPERATION,
        "max_candidates": max_candidates,
        "current_rubric": rubric_rows,
        "fixed_solver_rollouts": response_rows,
    }
    return "\n\n".join(
        [
            "Discover rubric coverage gaps for this one question. The training "
            "system assigns candidate IDs after generation; do not invent IDs.",
            "Complete gap-discovery context (JSON):\n"
            + json.dumps(payload, ensure_ascii=False, indent=2),
        ]
    )


def parse_gap_candidates(
    text: str,
    *,
    max_items: int,
    root_weight: float,
) -> list[dict[str, Any]]:
    """Parse only explicit GAP_EXPAND candidates and normalize weight magnitude."""

    if max_items < 1:
        return []
    magnitude = abs(float(root_weight))
    if magnitude <= 0.0:
        raise ValueError("Gap root weight magnitude must be positive.")

    payload = _extract_json_object(text)
    if not isinstance(payload, dict) or payload.get("operation") != GAP_PROPOSAL_OPERATION:
        return []
    raw_items = payload.get("criteria")
    if not isinstance(raw_items, list):
        return []
    # Normalize the compact prompt schema for the non-joint evolution path too.
    # Explicit null/empty/conflicting operations remain invalid, not defaulted.
    payload = dict(payload, criteria=[
        dict(item, operation=item.get("operation", GAP_PROPOSAL_OPERATION))
        if isinstance(item, dict) else item for item in raw_items
    ])
    parsed = parse_proposed_criteria(json.dumps(payload), max_items=max_items)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in parsed:
        if str(item.get("operation") or "").strip().lower() != GAP_PROPOSAL_OPERATION:
            continue
        title = " ".join(str(item.get("title") or "").split()).strip()
        description = " ".join(str(item.get("description") or "").split()).strip()
        if not title or not description:
            continue
        semantic_key = (title.casefold(), description.casefold())
        if semantic_key in seen:
            continue
        seen.add(semantic_key)
        proposed_weight = float(item.get("weight", 1.0) or 1.0)
        candidates.append(
            {
                "title": title,
                "description": description,
                "weight": -magnitude if proposed_weight < 0.0 else magnitude,
                "operation": GAP_PROPOSAL_OPERATION,
            }
        )
    return candidates
