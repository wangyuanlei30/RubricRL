"""Normalize explicit meta-verifier structure mistakes without changing verdicts.

Only complete fenced JSON, misplaced summary fields, and omitted object closers
at array boundaries in that same misplaced-summary pattern are supported.
The original strict routing and verdict checks run after normalization.
"""
from __future__ import annotations

import json
import re

from .rubric_meta_response_validation import validate_meta_response

SUMMARY_KEYS = ('final_verdict', 'failed_dimensions', 'final_reason')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _close_objects_before_array_end(text: str) -> tuple[str, int]:
    """Insert only missing `}` before `]`; never close strings or complete EOF."""
    stack, output = [], []
    in_string = escaped = False
    inserted = 0
    for char in text:
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in '{[':
            stack.append(char)
        elif char == ']':
            while stack and stack[-1] == '{':
                output.append('}')
                stack.pop()
                inserted += 1
            if not stack or stack.pop() != '[':
                raise ValueError('Unmatched array delimiter')
        elif char == '}':
            if not stack or stack.pop() != '{':
                raise ValueError('Unmatched object delimiter')
        output.append(char)
    if in_string or stack or not inserted:
        raise ValueError('JSON error is not the supported missing-object-closer pattern')
    return ''.join(output), inserted


def normalize_meta_response(text: str, context: dict | None) -> tuple[str, list[str]]:
    changes = []
    candidate = text.strip()
    fence = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', candidate, flags=re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
        changes.append('removed_json_code_fence')
    inserted = 0
    try:
        payload = json.loads(candidate, object_pairs_hook=_unique_object)
    except json.JSONDecodeError:
        candidate, inserted = _close_objects_before_array_end(candidate)
        payload = json.loads(candidate, object_pairs_hook=_unique_object)
    moved = 0
    if isinstance(payload, dict) and isinstance(payload.get('results'), list):
        for index, result in enumerate(payload['results']):
            if not isinstance(result, dict) or not isinstance(result.get('checks'), dict):
                continue
            checks = result['checks']
            for key in SUMMARY_KEYS:
                if key not in checks:
                    continue
                if key in result and result[key] != checks[key]:
                    raise ValueError(f'Conflicting nested and top-level {key}')
                result[key] = checks.pop(key)
                changes.append(f'results[{index}].checks.{key}->results[{index}].{key}')
                moved += 1
    if inserted:
        if not moved or not context:
            raise ValueError('Brace correction requires the known misplaced-summary pattern and audit context')
        changes.append(f'inserted_{inserted}_object_closers_before_array_end')
    normalized = json.dumps(payload, ensure_ascii=False) if changes else text
    validate_meta_response(normalized, context)
    return normalized, changes
