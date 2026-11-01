"""Judge all new criteria together per answer, across a whole RG stage.

Each sampled RG proposal has its own candidate list and cache namespace. Column order is retained when
mapping results back to criteria; original-rubric scores are merged later
from the training batch. Only presence scores feed the empirical gate.
"""
from dataclasses import dataclass
import math
from typing import Any, Callable


@dataclass
class CandidateJudgeGroup:
    question: str
    responses: list[str]
    candidates: list[dict[str, Any]]
    cache_scope: str = ""


def judge_candidate_groups(groups: list[CandidateJudgeGroup], *, judge_batch: Callable) -> list[list[list[float]]]:
    """Return [proposal][criterion][answer] scores, or raise on incomplete data.

    One batch call submits all proposal/answer requests to the existing bounded
Judge pool and shared RPM limiter. No per-criterion network calls are made.
"""
    requests, owners = [], []
    vectors = [[[] for _ in group.candidates] for group in groups]
    for group_index, group in enumerate(groups):
        if not group.candidates:
            continue
        if not group.responses:
            raise RuntimeError("Cannot judge new criteria without Solver responses")
        rubric = [{key: candidate[key] for key in ('title', 'description', 'weight')}
                  for candidate in group.candidates]
        for response in group.responses:
            requests.append(dict(prompt_text=group.question, response=response, rubric=rubric,
                                 reward_eval_mode='rubric_batch', judge_cache_scope=group.cache_scope))
            owners.append(group_index)
    if not requests:
        return vectors
    results = judge_batch(requests)
    if len(results) != len(requests):
        raise RuntimeError("New-criterion Judge response count mismatch")
    for group_index, result in zip(owners, results, strict=True):
        count = len(groups[group_index].candidates)
        try:
            if float(result.get('judge_error', 1)) != 0 or float(result.get('criterion_count', -1)) != count:
                raise ValueError('Judge error or criterion count mismatch')
            scores = [float(result[f'criterion_{index}_score']) for index in range(count)]
            if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
                raise ValueError('Invalid presence score')
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("Incomplete/invalid batched criterion scores; refusing synthetic RG reward") from exc
        for vector, score in zip(vectors[group_index], scores, strict=True):
            vector.append(score)
    return vectors
