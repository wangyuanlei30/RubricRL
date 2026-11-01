"""One shared-parameter update from Solver, GAP and vertical GRPO trajectories.

Each role has its own loss denominator and temperature. Only Solver receives
rubric-conditioned token advantage reweighting. All micro-batches accumulate
before ONE optimizer step per PPO epoch; global_batch_size does not split this
joint mode into separate optimizer steps. FSDP CPU parameter offload for the
trainable actor is unsupported (the normal worker offload-between-phases is OK).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist

from ...protocol import DataProto
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch


def _global_sum(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def _micro_batches(actor: Any, data: DataProto, *, solver: bool) -> list[DataProto]:
    if not actor.config.dynamic_batching:
        return actor._split_with_remainder(data, actor.config.micro_batch_size_per_device_for_update)
    original_mask = data.batch["attention_mask"]
    lengths = original_mask.sum(-1)
    max_width = original_mask.size(-1)
    if solver:
        teacher_mask = data.batch["grpo_opsd_teacher_attention_mask"]
        lengths = torch.maximum(lengths, teacher_mask.sum(-1))
        max_width = max(max_width, teacher_mask.size(-1))
    # The balancing mask measures forward cost only, never replaces the actual
    # attention layout used by either model. Teacher and Student widths may differ.
    data.batch["attention_mask"] = (torch.arange(max_width, device=lengths.device)[None, :] < lengths[:, None]).to(original_mask.dtype)
    budget = actor.config.max_token_len_per_gpu
    if budget is None:
        budget = actor.config.micro_batch_size_per_device_for_update * max_width
    try:
        batches, indices = prepare_dynamic_batch(data, max_token_len=budget)
    finally:
        data.batch["attention_mask"] = original_mask
    for batch, rows in zip(batches, indices, strict=True):
        batch.batch["attention_mask"] = original_mask[rows]
    return batches


def compute_joint_role_loss(actor: Any, data: DataProto, *, role: int, temperature: float, opsd: dict[str, Any]):
    """No optimizer actions here: return this micro-batch's differentiable loss."""
    inputs = {**data.batch, **data.non_tensor_batch}
    mask = inputs["response_mask"]
    log_probs = actor._forward_micro_batch(inputs, temperature=temperature)
    advantages = inputs["advantages"]
    metrics = {}
    if role == 0 and opsd.get("stgca_lambda", 0.5) > 0:
        teacher = actor._get_teacher_module()
        was_training = teacher.training
        teacher.eval()
        teacher_inputs = {key: inputs["grpo_opsd_teacher_" + key] for key in ("input_ids", "attention_mask", "position_ids")}
        teacher_inputs["responses"] = inputs["responses"]
        try:
            with torch.no_grad():
                teacher_lp = actor._forward_micro_batch_with_module(teacher_inputs, temperature=temperature, module=teacher)
        finally:
            teacher.train(was_training)
        clip_range = opsd.get("stgca_reweight_clip_range")
        advantages, metrics = actor._build_stgca_advantages(
            teacher_log_probs=teacher_lp, student_log_probs=log_probs,
            advantages=advantages, response_mask=mask, lam=opsd.get("stgca_lambda", 0.5),
            clip_range=actor.config.clip_ratio_low if clip_range is None else clip_range,
            negative_only=opsd.get("stgca_negative_only", False),
            thought_mask=inputs.get("stgca_thought_mask") if opsd.get("stgca_thought_only", False) else None,
        )
    # RG is completion-level GRPO: long completions must not get extra weight
    # merely for containing more tokens. Solver keeps its configured averaging.
    mode = actor.config.loss_avg_mode if role == 0 else "seq"
    pg_loss, pg_metrics = compute_policy_loss(
        old_log_probs=inputs["old_log_probs"], log_probs=log_probs,
        advantages=advantages, response_mask=mask,
        clip_ratio_low=actor.config.clip_ratio_low, clip_ratio_high=actor.config.clip_ratio_high,
        clip_ratio_dual=actor.config.clip_ratio_dual,
        loss_type=actor.config.loss_type, loss_avg_mode=mode,
    )
    loss = pg_loss
    metrics.update(pg_metrics)
    metrics["pg_loss"] = pg_loss.detach().item()
    if actor.config.use_kl_loss:
        if "ref_log_probs" not in inputs:
            raise ValueError("Joint KL loss requires fixed-reference log probabilities for every role.")
        kl = average_loss(compute_kl(log_probs=log_probs, ref_log_probs=inputs["ref_log_probs"],
            kl_penalty=actor.config.kl_penalty), mask, mode=mode)
        loss = loss + actor.config.kl_coef * kl
        metrics["kl_loss"] = kl.detach().item()
    return loss, metrics


def update_rubric_joint_policy(actor: Any, data: DataProto) -> dict[str, Any]:
    """Accumulate all three roles into the same optimizer before stepping."""
    actor.actor_module.train()
    opsd = data.meta_info["opsd_config"]
    if opsd.get("enable_casd", False):
        raise ValueError("Rubric joint update is for the single full-rubric Teacher, not CASD.")
    if getattr(getattr(actor.config, "fsdp", None), "enable_cpu_offload", False):
        raise ValueError("Trainable FSDP CPU offload is incompatible with joint gradient accumulation.")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    names = {0: "solver", 1: "gap", 2: "vertical"}
    roles = data.meta_info["joint_roles"]
    weights = data.meta_info["joint_role_weights"]
    metrics = defaultdict(list)
    role_batches = {}
    denominators = {}
    for role in roles:
        indices = torch.nonzero(data.batch["joint_role"] == role, as_tuple=False).flatten()
        if not len(indices):
            raise ValueError("Every rank must contain every enabled joint role, including masked padding.")
        batch = data.index_select(indices)
        role_batches[role] = batch
        denominator = batch.batch["response_mask"].sum() if role == 0 and actor.config.loss_avg_mode == "token" else batch.batch["joint_real_row"].sum()
        denominators[role] = _global_sum(denominator).clamp_min(1.0)

    optimizer_steps = 0
    for _ in range(actor.config.ppo_epochs):
        actor.actor_optimizer.zero_grad()
        for role in roles:
            temperature = data.meta_info["temperature"] if role == 0 else data.meta_info["rg_temperature"]
            coefficient = float(weights.get(role, weights.get(str(role), 0.0)))
            for batch in _micro_batches(actor, role_batches[role], solver=role == 0):
                loss, batch_metrics = compute_joint_role_loss(actor, batch, role=role, temperature=temperature, opsd=opsd)
                # seq averaging includes zero-valued dummy rows. Multiplying by
                # ALL micro-batch rows first removes that local denominator;
                # the global denominator counts REAL rows only.
                numerator = batch.batch["response_mask"].sum() if role == 0 and actor.config.loss_avg_mode == "token" else len(batch)
                scale = coefficient * world_size * numerator / denominators[role]
                (loss * scale).backward()
                append_to_dict(metrics, {f"joint/{names[role]}/{key}": value for key, value in batch_metrics.items()})
        # There is deliberately no optimizer.step between GAP and vertical.
        norm = actor._optimizer_step()
        optimizer_steps += int(torch.isfinite(norm).item())
        append_to_dict(metrics, {"actor/grad_norm": float(norm.detach().item())})
    interval = int(opsd.get("stgca_teacher_sync_interval", 0))
    step = int(opsd.get("global_step", 0))
    synced = actor.teacher_module is not None and interval > 0 and step > 0 and step % interval == 0
    if synced:
        from ..rubric_teacher_state import sync_teacher_parameters
        sync_teacher_parameters(actor.teacher_module, actor.actor_module)
    append_to_dict(metrics, {"grpo_opsd/teacher_synced": float(synced)})
    metrics["joint/optimizer_steps"] = [float(optimizer_steps)]
    return metrics
