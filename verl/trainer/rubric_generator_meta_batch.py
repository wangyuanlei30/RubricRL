"""Submit independent RG proposal audits to one bounded verifier pool.

Only requests are batched: candidates and peer context remain local to each
proposal. The backend must return responses in input order (as its pool.map
does). Reward assignment and persistent rubric mutation stay on the driver.
"""
from typing import Any, Callable


def audit_proposal_batches(entry_groups: list[list[dict[str, Any]]], *,
                           audit: Callable, template_path: str,
                           max_tokens: int) -> list[list[str]]:
    entries = [entry for group in entry_groups for entry in group]
    if not entries:
        return [[] for _ in entry_groups]
    responses = audit(entries, template_path=template_path, max_tokens=max_tokens)
    if len(responses) != len(entries):
        raise RuntimeError("Meta-verifier batch count mismatch.")
    grouped, offset = [], 0
    for group in entry_groups:
        grouped.append(responses[offset:offset + len(group)])
        offset += len(group)
    return grouped
