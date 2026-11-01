"""POW3R Eq. 3--8: pure reward arithmetic and checkpointable epoch state.

Reference: https://arxiv.org/html/2605.20164v1
No policy, judge, torch, or distributed execution dependency is needed here.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Sequence

import numpy as np

from .rubric_baseline_config import RubricBaselineConfig


@dataclass(frozen=True)
class Criterion:
    description: str
    title: str
    category: str
    weight: float  # Original signed HealthBench points, never overwritten.


def parse_rubric(rubric: Any) -> list[Criterion]:
    """Recover native categories; never silently merge missing categories."""
    if isinstance(rubric, np.ndarray):
        rubric = rubric.tolist()
    if not isinstance(rubric, (list, tuple)) or not rubric:
        raise ValueError("A nonempty structured rubric is required.")
    result = []
    for item in rubric:
        if not isinstance(item, dict):
            raise ValueError("Each rubric criterion must be a mapping.")
        title = str(item.get("title", ""))
        category = item.get("category")
        if not category:
            # Existing HealthBench conversion: 'accuracy (cluster/example)'.
            match = re.fullmatch(r"(.+?)\s*\((?:cluster|example)\)", title.strip())
            category = match.group(1).strip() if match else None
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Missing native category for rubric title {title!r}.")
        description = str(item.get("description", "")).strip()
        weight = float(item.get("weight", 0))
        if not description or not math.isfinite(weight) or weight == 0:
            raise ValueError("Rubric text must be nonempty and weight finite/nonzero.")
        result.append(Criterion(description, title, category.strip(), weight))
    return result


def good_behavior_item(criterion: Criterion) -> dict[str, Any]:
    """Appendix A.4: judge avoidance of a negative criterion, with |weight|."""
    text = criterion.description
    if criterion.weight < 0:
        text = ("The response does NOT exhibit the following undesirable behavior. "
                "Evaluate the logical negation of the entire quoted criterion: "
                + json.dumps(text, ensure_ascii=False))
    return {"description": text, "title": criterion.title,
            "category": criterion.category, "weight": abs(criterion.weight)}


def native_healthbench_score(verdicts: Sequence[float], criteria: Sequence[Criterion],
                             *, good_behavior: bool = False, clip_zero: bool = False) -> float:
    """Signed points / positive points. Evaluation retains negative scores."""
    values = np.asarray(verdicts, dtype=np.float64)
    weights = np.asarray([c.weight for c in criteria], dtype=np.float64)
    if values.shape != weights.shape or not np.isfinite(values).all():
        raise ValueError("Incomplete or non-finite judge verdicts.")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Verdicts must be in [0,1].")
    if good_behavior:
        values = np.where(weights < 0, 1 - values, values)
    denominator = weights[weights > 0].sum()
    if denominator <= 0:
        raise ValueError("Native HealthBench scoring requires a positive-point criterion.")
    score = float(np.dot(weights, values) / denominator)
    return max(0.0, score) if clip_zero else score


def category_reward(scores: np.ndarray, criteria: Sequence[Criterion],
                    alpha: Sequence[float] | None = None) -> np.ndarray:
    """Eq. 3/8, on good-behavior verdicts, one shared alpha vector per group."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(criteria) or not len(criteria):
        raise ValueError("Expected [rollout, criterion] score matrix.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        # Missing verdicts are not failures and must not silently become rewards.
        raise ValueError("Cannot aggregate missing, non-finite or out-of-range verdicts.")
    factors = np.ones(len(criteria)) if alpha is None else np.asarray(alpha, dtype=np.float64)
    if factors.shape != (len(criteria),) or not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError("Expected positive finite factors, one per criterion.")
    weights = np.asarray([abs(c.weight) for c in criteria]) * factors
    categories = list(dict.fromkeys(c.category for c in criteria))
    result = np.zeros(scores.shape[0], dtype=np.float64)
    for category in categories:
        indices = [i for i, c in enumerate(criteria) if c.category == category]
        result += scores[:, indices] @ weights[indices] / weights[indices].sum()
    return result / len(categories)


def update_factors(alpha: Sequence[float], variances: Sequence[float],
                   valid: Sequence[bool], criteria: Sequence[Criterion],
                   config: RubricBaselineConfig) -> np.ndarray:
    """Eq. 5--7; invalid criteria retain their old factor."""
    old = np.asarray(alpha, dtype=np.float64)
    variance = np.asarray(variances, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    n = len(criteria)
    if old.shape != (n,) or variance.shape != (n,) or valid.shape != (n,):
        raise ValueError("Factor, variance and valid-mask dimensions must match rubric.")
    if not np.isfinite(old).all() or np.any((old < config.alpha_min) | (old > config.alpha_max)):
        raise ValueError("Stored factor is outside its finite bounds.")
    if not np.isfinite(variance[valid]).all() or np.any(variance[valid] < 0):
        raise ValueError("Valid variance must be finite and nonnegative.")
    target = old.copy()
    for category in dict.fromkeys(c.category for c in criteria):
        idx = np.array([i for i, c in enumerate(criteria) if c.category == category and valid[i]], dtype=int)
        if not len(idx):
            continue
        if np.all(variance[idx] == 0):
            target[idx] = 1.0
        else:
            g = np.sqrt(variance[idx] + config.epsilon)
            weights = np.array([abs(criteria[i].weight) for i in idx])
            mean_g = float(np.dot(weights, g) / weights.sum())
            target[idx] = np.clip(1 - config.contrast_mix + config.contrast_mix * g / mean_g,
                                  config.alpha_min, config.alpha_max)
    result = old.copy()
    result[valid] = np.clip((1 - config.ema_beta) * old[valid] + config.ema_beta * target[valid],
                            config.alpha_min, config.alpha_max)
    return result


def rubric_key(prompt: Any, criteria: Sequence[Criterion]) -> str:
    """Stable across rollout UUIDs, balancing, epochs and restarts."""
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    payload = {"prompt": prompt, "criteria": [asdict(c) for c in criteria]}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class POW3RState:
    """Freeze weights within a dataset epoch; commit observations on next entry.

    Pending sufficient statistics are saved too, making mid-epoch resume exact.
    A repeated identical prompt within an epoch pools its verdicts, without
    changing the factors used by any group in that epoch.
    """
    VERSION = 1

    def __init__(self, config: RubricBaselineConfig):
        config.post_init()
        self.config = config
        self.epoch = -1
        self.entries: dict[str, dict[str, Any]] = {}

    def begin_epoch(self, epoch: int) -> None:
        if epoch < 0 or epoch < self.epoch:
            raise ValueError("POW3R epoch cannot move backwards.")
        if epoch == self.epoch:
            return
        if self.epoch >= 0 and epoch != self.epoch + 1:
            raise ValueError("POW3R epoch cannot skip an unrecorded data pass.")
        for entry in self.entries.values():
            count = np.array(entry["count"], dtype=np.float64)
            valid = count > 0
            mean = np.divide(entry["sum"], count, out=np.zeros_like(count), where=valid)
            variance = np.maximum(0.0, np.divide(entry["sum_sq"], count,
                                                out=np.zeros_like(count), where=valid) - mean ** 2)
            criteria = [Criterion(**c) for c in entry["criteria"]]
            entry["alpha"] = update_factors(entry["alpha"], variance, valid, criteria, self.config).tolist()
            for name in ("count", "sum", "sum_sq"):
                entry[name] = [0.0] * len(criteria)
        self.epoch = epoch

    def factors(self, key: str, criteria: Sequence[Criterion]) -> np.ndarray:
        encoded = [asdict(c) for c in criteria]
        if key not in self.entries:
            self.entries[key] = {"criteria": encoded, "alpha": [1.0] * len(criteria),
                                 "count": [0.0] * len(criteria), "sum": [0.0] * len(criteria),
                                 "sum_sq": [0.0] * len(criteria)}
        if self.entries[key]["criteria"] != encoded:
            raise ValueError("A persistent rubric key was reused for different criteria.")
        return np.array(self.entries[key]["alpha"], dtype=np.float64)

    def observe(self, key: str, criteria: Sequence[Criterion], scores: np.ndarray) -> None:
        """Record valid statistics only, honoring ceil(0.75 * group size)."""
        self.factors(key, criteria)
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != len(criteria) or not scores.shape[0]:
            raise ValueError("Observation matrix shape does not match rubric.")
        finite = np.isfinite(scores)
        if np.any((scores[finite] < 0) | (scores[finite] > 1)):
            raise ValueError("Observed verdicts must be in [0,1].")
        valid = finite.sum(axis=0) >= math.ceil(self.config.min_valid_fraction * len(scores))
        safe = np.where(finite, scores, 0.0)
        entry = self.entries[key]
        for name, value in (("count", finite.sum(axis=0)), ("sum", safe.sum(axis=0)),
                            ("sum_sq", (safe ** 2).sum(axis=0))):
            entry[name] = (np.array(entry[name]) + np.where(valid, value, 0)).tolist()

    def state_dict(self) -> dict[str, Any]:
        params = asdict(self.config)
        params.pop("eval_judge_overrides")
        return {"version": self.VERSION, "config": params, "epoch": self.epoch,
                "entries": json.loads(json.dumps(self.entries, allow_nan=False))}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = self.state_dict()["config"]
        if state.get("version") != self.VERSION or state.get("config") != expected:
            raise ValueError("POW3R checkpoint version/config does not match this run.")
        epoch = state.get("epoch")
        if not isinstance(epoch, int) or epoch < -1 or not isinstance(state.get("entries"), dict):
            raise ValueError("Malformed POW3R checkpoint.")
        entries = json.loads(json.dumps(state["entries"], allow_nan=False))
        for entry in entries.values():
            criteria = [Criterion(**c) for c in entry["criteria"]]
            n = len(criteria)
            update_factors(entry["alpha"], np.zeros(n), np.zeros(n, dtype=bool), criteria, self.config)
            for name in ("count", "sum", "sum_sq"):
                arr = np.asarray(entry[name])
                if arr.shape != (n,) or not np.isfinite(arr).all() or np.any(arr < 0):
                    raise ValueError("Malformed POW3R pending statistics.")
        self.epoch, self.entries = epoch, entries
