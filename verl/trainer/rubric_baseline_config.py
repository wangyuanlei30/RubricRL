"""Opt-in settings for fixed-rubric GRPO baselines."""

from dataclasses import dataclass, field
from typing import Any
import math


@dataclass
class RubricBaselineConfig:
    method: str = "none"  # none | rar | category_balanced | pow3r
    alpha_min: float = 0.67
    alpha_max: float = 1.5
    epsilon: float = 1.0e-4
    contrast_mix: float = 0.5
    ema_beta: float = 0.2
    min_valid_fraction: float = 0.75
    # Eq. 4--7 use epoch-t observations for epoch t+1, not the same group.
    update_schedule: str = "dataset_epoch"
    eval_judge_overrides: dict[str, Any] = field(default_factory=dict)

    def post_init(self):
        if self.method not in {"none", "rar", "category_balanced", "pow3r"}:
            raise ValueError(f"Unknown rubric baseline: {self.method}")
        values = (self.alpha_min, self.alpha_max, self.epsilon,
                  self.contrast_mix, self.ema_beta, self.min_valid_fraction)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("POW3R hyperparameters must be finite.")
        if not 0 < self.alpha_min <= 1 <= self.alpha_max:
            raise ValueError("Require 0 < alpha_min <= 1 <= alpha_max.")
        if self.epsilon <= 0 or not 0 <= self.contrast_mix <= 1:
            raise ValueError("Require epsilon > 0 and contrast_mix in [0,1].")
        if not 0 < self.ema_beta <= 1 or not 0 < self.min_valid_fraction <= 1:
            raise ValueError("EMA beta and valid fraction must be in (0,1].")
        if self.update_schedule != "dataset_epoch":
            raise ValueError("This reproduction uses dataset_epoch factor updates only.")
