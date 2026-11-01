"""Thin fixed-rubric adapter on the ordinary (teacher-free) GRPO trainer."""

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from .pow3r_rewards import POW3RState, category_reward, parse_rubric, rubric_key
from .ray_trainer import RayPPOTrainer
from ..utils.logger import Tracker


def validate_baseline_config(config) -> None:
    config.baseline.post_init()
    if config.baseline.method == "none":
        raise ValueError("A fixed-rubric baseline method must be selected.")
    if config.algorithm.adv_estimator != "grpo":
        raise ValueError("Rubric baselines require GRPO.")
    if config.algorithm.pipeline_reward or config.algorithm.online_filtering:
        raise ValueError("Baseline grouping requires pipeline_reward=false and online_filtering=false.")
    if config.data.mini_rollout_batch_size is not None:
        raise ValueError("Baseline epoch accounting requires mini_rollout_batch_size=null.")
    if config.worker.actor.ppo_epochs != 1:
        raise ValueError("The POW3R reproduction uses one PPO epoch per rollout batch.")
    if config.worker.rollout.n < 2:
        raise ValueError("GRPO requires at least two responses per prompt.")
    flags = ("enable_grpo_opsd_simple", "enable_casd", "enable_rubric_guided_rlsd",
             "enable_rubric_initialization", "enable_rubric_evolution",
             "enable_rubric_gap_expansion", "enable_rubric_generator_training",
             "enable_dead_criteria_guided_rollout")
    if any(getattr(config.opsd, key, False) for key in flags):
        raise ValueError("Fixed-rubric baselines cannot enable OPSD, rubric evolution or RG training.")
    expected = "rubric_baseline_judge.py"
    if Path(config.worker.reward.reward_function).name != expected:
        raise ValueError(f"Rubric baselines require the polarity-aware {expected}.")


def baseline_reward_configs(config):
    """Independent train/eval role flags; never infer eval from a generic marker."""
    validate_baseline_config(config)
    train = deepcopy(config.worker.reward)
    evaluation = deepcopy(config.worker.reward)
    train.reward_function_kwargs.update(baseline_method=config.baseline.method, is_eval=False)
    evaluation.reward_function_kwargs.update(config.baseline.eval_judge_overrides)
    evaluation.reward_function_kwargs.update(baseline_method=config.baseline.method, is_eval=True)
    return train, evaluation


def apply_baseline_rewards(batch, reward_metrics: dict[str, list[float]], *, config,
                           state: POW3RState, epoch: int) -> dict[str, float]:
    """Run on the driver after batch balancing, with sample-aligned judge metrics."""
    method = config.baseline.method
    if method == "rar":
        return {"baseline/rar": 1.0}
    groups: dict[str, list[int]] = defaultdict(list)
    for index, uid in enumerate(batch.non_tensor_batch["uid"]):
        groups[str(uid)].append(index)
    pending = []
    for indices in groups.values():
        if len(indices) != config.worker.rollout.n:
            raise ValueError("Incomplete rollout group: refusing biased POW3R statistics.")
        i = indices[0]
        rubric = parse_rubric(batch.non_tensor_batch["rubric"][i])
        question_key = "teacher_question" if "teacher_question" in batch.non_tensor_batch else "prompt_text"
        key = rubric_key(batch.non_tensor_batch[question_key][i], rubric)
        scores = []
        for row in indices:
            row_rubric = parse_rubric(batch.non_tensor_batch["rubric"][row])
            if rubric_key(batch.non_tensor_batch[question_key][row], row_rubric) != key:
                raise ValueError("A rollout group contains different questions/rubrics.")
            if reward_metrics["baseline_good_behavior"][row] != 1.0:
                raise ValueError("POW3R expects good-behavior judge verdicts; refusing double/wrong inversion.")
            if reward_metrics["criterion_count"][row] != len(rubric):
                raise ValueError("Judge criterion count does not match rubric.")
            if reward_metrics["judge_error"][row] != 0.0:
                raise ValueError("Judge failure cannot become a POW3R training reward.")
            scores.append([reward_metrics[f"criterion_{j}_score"][row] for j in range(len(rubric))])
        scores = np.asarray(scores, dtype=np.float64)
        # Validate the entire batch before mutating state at the epoch boundary.
        category_reward(scores, rubric)
        pending.append((indices, rubric, key, scores))

    response_mask = batch.batch["response_mask"]
    if not torch.all(response_mask.sum(-1) > 0):
        raise ValueError("Cannot place terminal reward on an empty response.")
    state.begin_epoch(epoch)
    tensor = torch.zeros_like(batch.batch["token_level_scores"])
    spreads, original_spreads, factors, rewards = [], [], [], []
    for indices, rubric, key, scores in pending:
        alpha = state.factors(key, rubric) if method == "pow3r" else np.ones(len(rubric))
        group_rewards = category_reward(scores, rubric, alpha)
        baseline_rewards = category_reward(scores, rubric)
        for row, reward in zip(indices, group_rewards):
            last = int(torch.nonzero(response_mask[row], as_tuple=False)[-1].item())
            tensor[row, last] = float(reward)
        spreads.append(float(np.std(group_rewards)))
        original_spreads.append(float(np.std(baseline_rewards)))
        factors.extend(alpha.tolist())
        rewards.extend(group_rewards.tolist())
        if method == "pow3r":
            # Observations do NOT change this epoch's applied alpha.
            state.observe(key, rubric, scores)
    batch.batch["token_level_scores"] = tensor
    return {"baseline/data_epoch": float(epoch), "baseline/reward_mean": float(np.mean(rewards)),
            "reward/overall": float(np.mean(rewards)),
            "baseline/native_clipped_reward_mean": float(np.mean(reward_metrics.get("overall", rewards))),
            "baseline/reward_std_group_mean": float(np.mean(spreads)),
            "baseline/category_reward_std_group_mean": float(np.mean(original_spreads)),
            "baseline/tied_group_fraction": float(np.mean(np.asarray(spreads) < 1e-12)),
            "baseline/alpha_mean": float(np.mean(factors)),
            "baseline/alpha_min": float(min(factors)), "baseline/alpha_max": float(max(factors)),
            "baseline/state_prompts": float(len(state.entries))}


class RayRubricBaselineTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        validate_baseline_config(self.config)
        self.pow3r_state = POW3RState(self.config.baseline)
        if not len(self.train_dataloader):
            raise ValueError("Empty baseline training dataloader.")
        digest = hashlib.sha256()
        with Path(self.config.data.train_files).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(json.dumps({"seed": self.config.data.seed, "shuffle": self.config.data.shuffle,
            "batch": self.config.data.rollout_batch_size, "prompt_key": self.config.data.prompt_key},
            sort_keys=True).encode())
        self.training_data_signature = digest.hexdigest()

    def _create_logger(self):
        # Preserve previous logs: the existing FileLogger opens files with 'w'.
        root = Path(self.config.trainer.save_checkpoint_path)
        previous = [root / name for name in
                    ("experiment_config.json", "experiment_log.jsonl", "generations.log")]
        if "file" in self.config.trainer.logger and any(p.exists() for p in previous):
            if not (self.config.trainer.find_last_checkpoint or self.config.trainer.load_checkpoint_path):
                raise FileExistsError("Baseline output already exists; choose a new path or explicit resume.")
            history = root / "log_history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            history.mkdir(parents=True, exist_ok=False)
            for path in previous:
                if path.exists():
                    shutil.copy2(path, history / path.name)
        return Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())

    def _postprocess_train_rewards(self, batch, reward_metrics, metrics):
        epoch = (self.global_step - 1) // len(self.train_dataloader)
        metrics.update(apply_baseline_rewards(batch, reward_metrics, config=self.config,
                                             state=self.pow3r_state, epoch=epoch))

    def _save_extra_checkpoint_state(self, folder_path: str) -> None:
        payload = {"global_step": self.global_step, "epoch_batches": len(self.train_dataloader),
                   "rollout_n": self.config.worker.rollout.n, "state": self.pow3r_state.state_dict(),
                   "training_data_signature": self.training_data_signature}
        path = Path(folder_path) / "pow3r_state.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    def _load_extra_checkpoint_state(self, folder_path: str) -> None:
        path = Path(folder_path) / "pow3r_state.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing baseline state: {path}; do not resume from an unrelated checkpoint.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_step = int(Path(folder_path).name.removeprefix("global_step_"))
        if (payload["global_step"] != expected_step or
                payload["epoch_batches"] != len(self.train_dataloader) or
                payload["rollout_n"] != self.config.worker.rollout.n or
                payload.get("training_data_signature") != self.training_data_signature):
            raise ValueError("Baseline resume changed step, training data/order, epoch length or group size.")
        self.pow3r_state.load_state_dict(payload["state"])
