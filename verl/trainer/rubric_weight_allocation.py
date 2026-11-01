"""Allocate one root's original signed reward weight across its active family."""
from __future__ import annotations

import math


def equal_family_weight(parent_weight: float, active_child_count: int) -> float:
    """Split the original W equally among the parent and its active children.

    A parent without active children keeps W. Always pass the original weight,
    so rebuilding a rubric does not repeatedly discount it.
    """
    if not math.isfinite(parent_weight):
        raise ValueError("Parent weight must be finite")
    if not isinstance(active_child_count, int) or active_child_count < 0:
        raise ValueError("Active child count must be a nonnegative integer")
    if active_child_count == 0:
        return parent_weight
    return parent_weight / (active_child_count + 1)
