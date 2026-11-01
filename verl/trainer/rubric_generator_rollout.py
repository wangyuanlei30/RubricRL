"""Retain RG trajectories and pack separately normalized roles for one actor update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import uuid

import numpy as np
import torch
import torch.nn.functional as F

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from .rubric_generator_rewards import group_relative_advantages


ROLE_IDS = {"solver": 0, "gap": 1, "vertical": 2}
MODEL_KEYS = ("input_ids", "attention_mask", "position_ids")
TEACHER_KEYS = tuple("grpo_opsd_teacher_" + key for key in MODEL_KEYS)
RESPONSE_KEYS = ("responses", "response_mask", "old_log_probs", "advantages", "ref_log_probs", "stgca_thought_mask")


@dataclass
class RubricGeneratorRollouts:
    """One raw completion per row, in prompt-major then sample-major order."""
    data: DataProto
    texts: list[str]
    group_ids: list[str]
    n: int

    @classmethod
    def from_generation(cls, data: DataProto, texts: list[str], *, prompt_count: int, n: int, temperature: float):
        if n < 2 or len(data) != prompt_count * n or len(texts) != len(data):
            raise ValueError("RG rollout shape mismatch or n < 2.")
        if temperature <= 0:
            raise ValueError("RG training needs positive sampling temperature.")
        groups = np.repeat(["rg_" + uuid.uuid4().hex for _ in range(prompt_count)], n).tolist()
        data.meta_info["temperature"] = float(temperature)
        data.meta_info["log_prob_temperature"] = float(temperature)
        return cls(data, texts, groups, n)

    def assign_rewards(self, rewards: list[float], *, role: str) -> None:
        advantages = group_relative_advantages(rewards, self.group_ids, expected_n=self.n)
        mask = self.data.batch["response_mask"]
        scalar = torch.tensor(advantages, dtype=torch.float32, device=mask.device)
        self.data.batch["advantages"] = scalar[:, None] * mask
        self.data.batch["rg_reward"] = torch.tensor(rewards, dtype=torch.float32, device=mask.device)
        self.data.batch["joint_role"] = torch.full((len(self.data),), ROLE_IDS[role], dtype=torch.long, device=mask.device)
        self.data.non_tensor_batch["uid"] = np.asarray(self.group_ids, dtype=object)
        self.data.non_tensor_batch["rg_raw_completion"] = np.asarray(self.texts, dtype=object)


def attach_behavior_log_probs(rollouts: RubricGeneratorRollouts, worker_group: Any, *, use_reference: bool) -> DataProto:
    """Compute on raw sampled token IDs before ANY optimizer update; pad only for dispatch."""
    if "advantages" not in rollouts.data.batch:
        raise ValueError("Assign RG rewards before preparing the training batch.")
    padded, count = pad_dataproto_to_divisor(rollouts.data, worker_group.world_size)
    old = unpad_dataproto(worker_group.compute_log_probs(padded), pad_size=count)
    result = rollouts.data.union(old)
    if use_reference:
        ref = unpad_dataproto(worker_group.compute_ref_log_probs(padded), pad_size=count)
        result = result.union(ref)
    return result


def _pad_sequence(tensor: torch.Tensor, *, prompt_width: int, response_width: int, old_response_width: int, value: int = 0):
    old_prompt_width = tensor.size(-1) - old_response_width
    if old_prompt_width < 1 or prompt_width < old_prompt_width or response_width < old_response_width:
        raise ValueError("Invalid prompt/response repadding lengths.")
    # Pad the PROMPT on the left and the RESPONSE on the right. Padding the
    # entire concatenated sequence on one side would shift response labels.
    return F.pad(tensor, (prompt_width - old_prompt_width, response_width - old_response_width), value=value)


def build_joint_update_batch(
    solver: DataProto,
    rg_batches: list[DataProto],
    *,
    world_size: int,
    pad_token_id: int,
    rg_loss_coef: float,
    gap_weight: float,
    vertical_weight: float,
) -> DataProto:
    """Pack equal role counts on every DP rank without putting dummy rows in the loss.

    Role-specific padding happens BEFORE rank-major packing. Every rank then
    executes the same Solver/GAP/vertical forward schedule, including when a
    role has fewer real samples than GPUs. This avoids FSDP collective mismatch.
    The original Solver batch is untouched for metrics/checkpoint orchestration.
    This initial implementation is text-only and uses one joint step per PPO epoch.
    """
    if world_size < 1:
        raise ValueError("world_size must be positive.")
    for key in ("multi_modal_data", "multi_modal_inputs"):
        if any(bool(value) for value in solver.non_tensor_batch.get(key, [])):
            raise ValueError("RG joint update currently supports text-only Solver inputs.")
    sources: dict[int, list[DataProto]] = {0: [solver]}
    for data in rg_batches:
        roles = set(data.batch["joint_role"].cpu().tolist())
        if len(roles) != 1 or not roles <= {1, 2}:
            raise ValueError("Each RG source batch must have exactly one known role.")
        sources.setdefault(next(iter(roles)), []).append(data)
    all_sources = [data for group in sources.values() for data in group]
    response_width = max(data.batch["responses"].size(-1) for data in all_sources)
    prompt_width = max(data.batch["input_ids"].size(-1) - data.batch["responses"].size(-1) for data in all_sources)
    teacher_prompt_width = max(solver.batch[key].size(-1) - solver.batch["responses"].size(-1) for key in TEACHER_KEYS)
    use_ref = "ref_log_probs" in solver.batch
    use_thought = "stgca_thought_mask" in solver.batch
    prepared: dict[int, DataProto] = {}
    rg_temperatures: set[float] = set()

    for role, batches in sorted(sources.items()):
        parts = []
        for data in batches:
            width = data.batch["responses"].size(-1)
            tensors = {}
            for key in MODEL_KEYS:
                tensors[key] = _pad_sequence(data.batch[key], prompt_width=prompt_width,
                    response_width=response_width, old_response_width=width,
                    value=pad_token_id if key == "input_ids" else 0)
            for key in RESPONSE_KEYS:
                if key == "ref_log_probs" and not use_ref or key == "stgca_thought_mask" and not use_thought:
                    continue
                if key not in data.batch:
                    if key != "stgca_thought_mask":
                        raise ValueError(f"Missing {key} in role {role}.")
                    source = torch.zeros_like(data.batch["response_mask"])
                else:
                    source = data.batch[key]
                tensors[key] = F.pad(source, (0, response_width - width), value=pad_token_id if key == "responses" else 0)
            for key in TEACHER_KEYS:
                if role == 0:
                    tensors[key] = _pad_sequence(data.batch[key], prompt_width=teacher_prompt_width,
                        response_width=response_width, old_response_width=width,
                        value=pad_token_id if key.endswith("input_ids") else 0)
                else:
                    shape = (*solver.batch[key].shape[1:-1], teacher_prompt_width + response_width)
                    tensors[key] = torch.zeros((len(data), *shape), dtype=solver.batch[key].dtype)
            tensors["joint_role"] = torch.full((len(data),), role, dtype=torch.long)
            tensors["joint_real_row"] = torch.ones(len(data), dtype=torch.float32)
            parts.append(DataProto.from_dict(tensors=tensors))
            if role:
                rg_temperatures.add(float(data.meta_info["temperature"]))
        merged = DataProto.concat(parts)
        padded, padding = pad_dataproto_to_divisor(merged, world_size)
        # pad_dataproto repeats real rows for safe forwards; remove their loss,
        # not their attention, because an all-padding FSDP forward is unsafe.
        if padding:
            padded.batch["joint_real_row"][-padding:] = 0
            padded.batch["response_mask"][-padding:] = 0
            padded.batch["advantages"][-padding:] = 0
        prepared[role] = padded
    if len(rg_temperatures) > 1:
        raise ValueError("GAP/vertical training temperatures must match.")
    by_role_rank = {role: data.chunk(world_size) for role, data in prepared.items()}
    rank_batches = [DataProto.concat([by_role_rank[role][rank] for role in sorted(prepared)]) for rank in range(world_size)]
    result = DataProto.concat(rank_batches)
    result.meta_info = dict(solver.meta_info)
    result.meta_info.update({
        "loss_type": "rubric_joint", "joint_roles": sorted(prepared),
        "rg_temperature": next(iter(rg_temperatures), float(solver.meta_info["temperature"])),
        "joint_role_weights": {0: 1.0, 1: rg_loss_coef * gap_weight, 2: rg_loss_coef * vertical_weight},
        "global_token_num": result.batch["attention_mask"].sum(-1).tolist(),
    })
    return result
