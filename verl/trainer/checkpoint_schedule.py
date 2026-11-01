"""Combine periodic checkpoint saves with explicit training-step milestones."""

from collections.abc import Sequence


def should_save_checkpoint(step: int, *, save_freq: int, save_steps: Sequence[int] = ()) -> bool:
    """One save at each periodic or explicit step, even if both coincide.

    Retention is controlled separately by trainer.save_limit. This predicate
    also lets the final-save hook avoid saving the same final step twice.
    """
    return step > 0 and ((save_freq > 0 and step % save_freq == 0) or step in save_steps)
