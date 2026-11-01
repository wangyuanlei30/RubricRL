"""Validate structured external meta responses before assigning any RG reward."""
from __future__ import annotations

import json
from .rubric_meta_verifier import route_batched_meta_verifier_payloads, parse_meta_verifier_verdict


def validate_meta_response(text: str, context: dict | None) -> None:
    json.loads(text)
    if not context:
        return
    operation = context['operation_type']
    ids = [item['candidate_id'] for item in context['candidates']]
    routed, diagnostic = route_batched_meta_verifier_payloads(text,
        expected_candidate_ids=ids, operation_type=operation)
    if not diagnostic['parse_valid'] or any(diagnostic.get(key) for key in (
        'missing_candidate_ids', 'duplicate_candidate_ids', 'unknown_candidate_ids',
        'operation_mismatch_candidate_ids')):
        raise ValueError('Invalid candidate routing: ' + json.dumps(diagnostic))
    for key in ids:
        verdict = parse_meta_verifier_verdict(routed[key], operation_type=operation)
        if verdict.get('rejection_type') == 'invalid_json':
            raise ValueError(f'Malformed verdict for {key}: ' + json.dumps(verdict))
