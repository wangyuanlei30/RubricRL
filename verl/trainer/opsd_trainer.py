# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
OPSD Trainer: Hint-Guided On-Policy Self-Distillation with Self-Evolving Navigator.

Extends RayPPOTrainer with:
  1. Multi-round generation: Student → Navigator → Teacher
  2. Navigator GRPO advantage from Teacher trajectory correctness
  3. OPSD forward batch assembly for joint training
  4. Curriculum learning for ground-truth dropping
"""

import json
import time
import uuid
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm

from rubric_generation.rubric_initializer import (
    EVORUBRIC_BOOTSTRAP,
    REFERENCE_GROUNDED,
    QUESTION_ONLY,
    InitialRubricPromptBuilder,
    InitialRubricRequest,
    ReferenceAnswerIndex,
    RubricInitializationStore,
    make_initialization_key,
    parse_initial_rubric,
    reference_answer_text,
)

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..utils import torch_functional as VF
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from .checkpoint_schedule import should_save_checkpoint
from .config import PPOConfig
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from .core_algos import AdvantageEstimator, compute_advantage_return
from .opsd_algos import (
    compute_curriculum_p_drop,
    compute_navigator_advantage,
    construct_navigator_prompts,
    construct_sdpo_teacher_prompts,
    construct_teacher_prompts,
    decode_navigator_hints,
    truncate_text_by_tokens,
)
from .opsd_config import OPSDConfig
from .rubric_gap_evolution import (
    GAP_PROPOSAL_OPERATION,
    GapExpansionContext,
    build_gap_proposal_hint,
    parse_gap_candidates,
)
from .rubric_evolution import (
    CRITERION_POLARITY_NEGATIVE,
    CRITERION_STATE_DEAD,
    CRITERION_STATE_SATURATED,
    RubricEvolver,
    criterion_polarity,
    parse_batched_proposal_payload,
    parse_proposed_criteria,
    rubric_item_identity,
)
from .rubric_meta_verifier import (
    CHILD_EVOLUTION,
    GAP_EXPAND,
    RubricMetaVerifierRouter,
)
from .rubric_generator_rollout import (
    RubricGeneratorRollouts,
    attach_behavior_log_probs,
    build_joint_update_batch,
)
from .ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
    apply_kl_penalty,
    compute_advantage,
)


DEAD_CRITERIA_GUIDED_ROLLOUT_TEMPLATE = """
{% if question | trim -%}
{{ question | trim }}

{% endif -%}
Your previous attempts either missed required criteria or triggered penalty conditions:

{{ hint | trim }}

Use these criteria as private guidance while answering the original user request. Satisfy the requirements and avoid the penalty conditions. Answer naturally and directly. Do not mention the criteria, point values, or any private guidance.
"""


def _concat_prompt_and_response(
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    prompt_position_ids: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate prompt and response following the rollout engine's position-id logic."""
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, response_mask], dim=1)

    response_length = response_ids.size(1)
    batch_size = prompt_ids.size(0)
    delta_position_id = torch.arange(
        1, response_length + 1, device=prompt_position_ids.device
    )
    delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
    if prompt_position_ids.ndim == 3:
        delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
            batch_size, prompt_position_ids.size(1), -1
        )
    response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
    position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
    return input_ids, attention_mask, position_ids


def _mask_prefix_response_tokens(
    response_mask: torch.Tensor, prefix_tokens: int
) -> torch.Tensor:
    """Zero out the first N valid response tokens in each sequence."""
    if prefix_tokens <= 0:
        return response_mask
    masked = response_mask.clone()
    batch_size = masked.shape[0]
    for idx in range(batch_size):
        valid_positions = torch.nonzero(masked[idx] > 0, as_tuple=False).squeeze(-1)
        if valid_positions.numel() == 0:
            continue
        cutoff = min(prefix_tokens, int(valid_positions.numel()))
        masked[idx, valid_positions[:cutoff]] = 0
    return masked


class RayOPSDTrainer(RayPPOTrainer):
    """OPSD Trainer with multi-round generation and joint Navigator + Reasoner update."""

    def __init__(self, config, *args, **kwargs):
        # OPSD does not rely on PPO/GRPO's rollout.n validation in parent trainer.
        # Temporarily set n=2 to bypass the parent check, then restore it.
        original_n = config.worker.rollout.n
        if original_n == 1:
            config.worker.rollout.n = 2  # bypass parent check
        super().__init__(config, *args, **kwargs)
        config.worker.rollout.n = original_n  # restore
        self.opsd_config: OPSDConfig = self.config.opsd

        # Set opsd_enabled flag on actor config so workers instantiate the right actor class
        self.config.worker.actor.opsd_enabled = True
        self.config.worker.actor.freeze_teacher_model = (
            self.opsd_config.freeze_teacher_model
        )
        self.config.worker.actor.separate_teacher_model = (
            self.opsd_config.enable_rubric_generator_training
            and self.opsd_config.freeze_teacher_model
        )

        # Load navigator and teacher templates
        self._nav_template_path = self.opsd_config.navigator_prompt_template
        self._teacher_template_path = self.opsd_config.teacher_hint_template
        self._sdpo_teacher_template_path = self.opsd_config.sdpo_teacher_template

        if self.opsd_config.enable_sdpo and self.opsd_config.use_gt_as_hint:
            raise ValueError(
                "opsd.enable_sdpo and opsd.use_gt_as_hint cannot both be enabled."
            )

        if self.opsd_config.enable_sdpo:
            self.opsd_config.num_hints = 1
            self.opsd_config.lambda_nav = 0.0
            self.opsd_config.alpha = 1.0
            print("[OPSD] Mode: SDPO self-distillation")
            print("[OPSD]   - Navigator rollout: SKIPPED")
            print(
                f"[OPSD]   - candidate samples per prompt: {self.opsd_config.sdpo_num_candidates}"
            )
            print(
                "[OPSD]   - num_hints forced to 1, lambda_nav forced to 0.0, alpha forced to 1.0"
            )
        elif self.opsd_config.use_gt_as_hint:
            # Actually force the config values, not just print about it
            self.opsd_config.num_hints = 1
            self.opsd_config.lambda_nav = 0.0
            self.opsd_config.alpha = 1.0
            print("[OPSD] Mode: Self-Distilled Reasoner (use_gt_as_hint=True)")
            print("[OPSD]   - Navigator rollout: SKIPPED (using GT answers as hints)")
            print(
                "[OPSD]   - num_hints forced to 1, lambda_nav forced to 0.0, alpha forced to 1.0"
            )
            print("[OPSD]   - p_drop forced to 0 (Teacher always sees GT answer)")
            if self.opsd_config.self_distill_negative_off_policy:
                print(
                    "[OPSD]   - negative off-policy enabled: "
                    f"coef={self.opsd_config.self_distill_negative_off_policy_coef}"
                )
        else:
            print(
                f"[OPSD] Navigator hints per question (K): {self.opsd_config.num_hints}"
            )
            print(f"[OPSD] Lambda_nav: {self.opsd_config.lambda_nav}")
        print(f"[OPSD] Alpha (on/off-policy weight): {self.opsd_config.alpha}")
        print(
            f"[OPSD] Distillation loss type: {self.opsd_config.distillation_loss_type}"
        )
        print(f"[OPSD] Student rollout n: {self.opsd_config.student_rollout_n}")
        print(
            f"[OPSD] Navigator include student trajectory: "
            f"{self.opsd_config.navigator_include_student_trajectory}"
        )
        print(
            f"[OPSD] Navigator max trajectory length: "
            f"{self.opsd_config.navigator_max_trajectory_length}"
        )
        print(
            f"[OPSD] Teacher max trajectory length: "
            f"{self.opsd_config.teacher_max_trajectory_length}"
        )
        print(f"[OPSD] Teacher context mode: {self.opsd_config.teacher_context_mode}")
        print(
            f"[OPSD] Teacher reference context length: "
            f"{self.opsd_config.teacher_reference_context_length}"
        )
        print(
            f"[OPSD] Teacher student context length: "
            f"{self.opsd_config.teacher_student_context_length}"
        )
        print(f"[OPSD] Dynamic sample hint: {self.opsd_config.dynamic_sample_hint}")
        print(
            f"[OPSD] Teacher rollout n per hint: {self.opsd_config.teacher_rollout_n}"
        )
        print(
            f"[OPSD] Distill token selection: {self.opsd_config.distill_token_selection}"
        )
        print(
            f"[OPSD] Distill token keep ratio: {self.opsd_config.distill_token_keep_ratio}"
        )
        print(
            f"[OPSD] Curriculum warmup steps: {self.opsd_config.curriculum_warmup_steps}"
        )
        print(
            f"[OPSD] Off-policy include GT answer: {self.opsd_config.off_policy_include_gt_answer}"
        )
        print(
            "[OPSD] Self-distill negative off-policy: "
            f"{self.opsd_config.self_distill_negative_off_policy}"
        )
        print(
            "[OPSD] Self-distill masked prefix tokens: "
            f"{self.opsd_config.self_distill_mask_prefix_tokens}"
        )
        print(f"[OPSD] KL penalty type: {self.opsd_config.opsd_kl_penalty}")
        if self.opsd_config.distillation_topk is not None:
            print(
                f"[OPSD] Top-k KL approximation: k={self.opsd_config.distillation_topk}"
            )
            print(f"[OPSD] Top-k source: {self.opsd_config.distillation_topk_source}")

        self._nav_hint_log_samples = 4
        self._nav_hint_log_history_limit = 20
        self._nav_hint_log_history: list[dict[str, Any]] = []
        self._nav_hint_log_path = (
            Path(self.config.trainer.save_checkpoint_path)
            / "navigator_hint_samples.json"
        )
        self._student_generation_log_samples = 4
        self._student_generation_log_dir = (
            Path(self.config.trainer.save_checkpoint_path)
            / "student_generation_samples"
        )

    def _get_teacher_questions(self, batch: DataProto) -> np.ndarray:
        questions = batch.non_tensor_batch.get("teacher_question")
        if questions is None:
            questions = batch.non_tensor_batch["prompt_text"]
        return questions

    def _log_navigator_hints(self, batch: DataProto, hints: list[list[str]]) -> None:
        """Persist a small rolling window of navigator hint samples for inspection."""
        if not hints:
            return

        prompt_text = batch.non_tensor_batch.get("prompt_text")
        ground_truth = batch.non_tensor_batch.get("ground_truth")
        uids = batch.non_tensor_batch.get("uid")

        sample_count = min(self._nav_hint_log_samples, len(hints))
        samples = []
        for idx in range(sample_count):
            question = "" if prompt_text is None else str(prompt_text[idx])
            samples.append(
                {
                    "uid": None if uids is None else str(uids[idx]),
                    "question": question[:500],
                    "ground_truth": (
                        None if ground_truth is None else str(ground_truth[idx])
                    ),
                    "hints": [str(h) for h in hints[idx]],
                }
            )

        self._nav_hint_log_history.append(
            {
                "step": self.global_step,
                "num_questions": len(hints),
                "num_hints_per_question": len(hints[0]) if hints else 0,
                "samples": samples,
            }
        )
        self._nav_hint_log_history = self._nav_hint_log_history[
            -self._nav_hint_log_history_limit :
        ]

        self._nav_hint_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._nav_hint_log_path.write_text(
            json.dumps(self._nav_hint_log_history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _log_student_generations(
        self, batch: DataProto, student_output: DataProto
    ) -> None:
        """Persist a few sampled student generations for the current step."""
        if len(student_output) == 0:
            return

        prompt_text = batch.non_tensor_batch.get("prompt_text")
        ground_truth = batch.non_tensor_batch.get("ground_truth")
        uids = batch.non_tensor_batch.get("uid")
        problem_type = batch.non_tensor_batch.get("problem_type")

        sample_count = min(self._student_generation_log_samples, len(student_output))
        sampled_indices = np.random.choice(
            len(student_output), size=sample_count, replace=False
        )

        samples = []
        response_ids = student_output.batch["responses"]
        response_mask = student_output.batch["response_mask"]
        for idx in sampled_indices.tolist():
            cur_response_length = int(response_mask[idx].sum().item())
            valid_response_ids = response_ids[idx][:cur_response_length]
            response_text = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=False
            )
            question = "" if prompt_text is None else str(prompt_text[idx])
            samples.append(
                {
                    "uid": None if uids is None else str(uids[idx]),
                    "problem_type": (
                        None if problem_type is None else str(problem_type[idx])
                    ),
                    "ground_truth": (
                        None if ground_truth is None else str(ground_truth[idx])
                    ),
                    "response_length": cur_response_length,
                    "question": question,
                    "response": response_text,
                }
            )

        payload = {
            "step": self.global_step,
            "num_samples_in_batch": len(student_output),
            "num_logged_samples": sample_count,
            "samples": samples,
        }

        self._student_generation_log_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            self._student_generation_log_dir / f"step_{self.global_step:06d}.json"
        )
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _decode_student_trajectories(
        self,
        student_output: DataProto,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        """Decode student rollout responses for optional Navigator conditioning."""
        trajectories: list[str] = []
        response_ids = student_output.batch["responses"]
        response_mask = student_output.batch["response_mask"]

        for idx in range(len(student_output)):
            cur_response_length = int(response_mask[idx].sum().item())
            valid_response_ids = response_ids[idx][:cur_response_length]
            response_text = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=False
            )
            trajectories.append(
                truncate_text_by_tokens(response_text, self.tokenizer, max_tokens)
            )

        return trajectories

    def _extract_teacher_context_snippet(
        self,
        text: Any,
        max_tokens: Optional[int],
        *,
        core_only: bool,
    ) -> str:
        if text is None:
            return ""
        text_str = str(text)
        if not text_str:
            return ""
        if not max_tokens or max_tokens <= 0:
            return ""
        if not core_only:
            return truncate_text_by_tokens(text_str, self.tokenizer, max_tokens)

        token_ids = self.tokenizer.encode(text_str, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text_str
        if max_tokens <= 32:
            return self.tokenizer.decode(
                token_ids[-max_tokens:], skip_special_tokens=False
            )

        head_len = max(1, max_tokens // 2)
        tail_len = max(1, max_tokens - head_len)
        head_text = self.tokenizer.decode(
            token_ids[:head_len], skip_special_tokens=False
        ).strip()
        tail_text = self.tokenizer.decode(
            token_ids[-tail_len:], skip_special_tokens=False
        ).strip()
        return (
            "[Earlier Key Step]\n"
            f"{head_text}\n\n"
            "[Conclusion / Final Segment]\n"
            f"{tail_text}"
        )

    def _build_teacher_context_trajectories(
        self,
        batch: DataProto,
        student_output: Optional[DataProto] = None,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        mode = self.opsd_config.teacher_context_mode
        if mode == "hint":
            return None, None

        core_only = mode == "hint_plus_core_trajectories"
        ref_max_tokens = self.opsd_config.teacher_reference_context_length
        student_max_tokens = self.opsd_config.teacher_student_context_length

        reference_trajectories: list[str] = []
        raw_references = batch.non_tensor_batch.get("trajectory")
        for idx in range(len(batch)):
            raw_reference = "" if raw_references is None else raw_references[idx]
            reference_trajectories.append(
                self._extract_teacher_context_snippet(
                    raw_reference,
                    ref_max_tokens,
                    core_only=core_only,
                )
            )

        student_trajectories = None
        if student_output is not None:
            decoded_student_trajectories = self._decode_student_trajectories(
                student_output
            )
            student_trajectories = [
                self._extract_teacher_context_snippet(
                    trajectory,
                    student_max_tokens,
                    core_only=core_only,
                )
                for trajectory in decoded_student_trajectories
            ]

        return (
            np.array(reference_trajectories, dtype=object),
            (
                None
                if student_trajectories is None
                else np.array(student_trajectories, dtype=object)
            ),
        )

    def _get_self_distill_hints(self, batch: DataProto) -> list[list[str]]:
        """Use trajectory as teacher context when available, otherwise fall back to GT."""
        trajectories = batch.non_tensor_batch.get("trajectory")
        hints = []
        for idx, gt in enumerate(batch.non_tensor_batch["ground_truth"]):
            trajectory = ""
            if trajectories is not None:
                trajectory = truncate_text_by_tokens(
                    trajectories[idx],
                    self.tokenizer,
                    self.opsd_config.teacher_max_trajectory_length,
                )
            hints.append([trajectory] if trajectory else [str(gt)])
        return hints

    def _construct_teacher_fallback_prompts(
        self,
        batch: DataProto,
        repeat_k: int,
        student_output: Optional[DataProto] = None,
    ) -> tuple[DataProto, int]:
        """Build fallback teacher prompts using trajectory first and GT only as a final fallback."""
        trajectories = batch.non_tensor_batch.get("trajectory")
        fallback_hints: list[list[str]] = []
        fallback_gt_answers: list[str] = []
        trajectory_missing = 0

        for idx, gt in enumerate(batch.non_tensor_batch["ground_truth"]):
            trajectory = ""
            if trajectories is not None:
                trajectory = truncate_text_by_tokens(
                    trajectories[idx],
                    self.tokenizer,
                    self.opsd_config.teacher_max_trajectory_length,
                )

            if trajectory:
                fallback_hints.append([trajectory] * repeat_k)
                fallback_gt_answers.append("")
            else:
                fallback_hints.append([""] * repeat_k)
                fallback_gt_answers.append(str(gt))
                trajectory_missing += 1

        reference_trajectories, student_trajectories = (
            self._build_teacher_context_trajectories(
                batch,
                student_output=student_output,
            )
        )
        fallback_teacher_gen = construct_teacher_prompts(
            questions=self._get_teacher_questions(batch),
            hints=fallback_hints,
            gt_answers=np.array(fallback_gt_answers, dtype=object),
            p_drop=0.0,
            tokenizer=self.tokenizer,
            processor=self.processor,
            reference_trajectories=reference_trajectories,
            student_trajectories=student_trajectories,
            template_path=self._teacher_template_path,
            max_prompt_length=self.config.data.max_prompt_length,
            multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
            image_min_pixels=self.config.data.image_min_pixels,
            image_max_pixels=self.config.data.image_max_pixels,
            video_min_pixels=self.config.data.video_min_pixels,
            video_max_pixels=self.config.data.video_max_pixels,
            video_fps=self.config.data.video_fps,
            video_max_frames=self.config.data.video_max_frames,
            apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
        )
        return fallback_teacher_gen, trajectory_missing

    def _construct_on_policy_teacher_prompts(
        self,
        batch: DataProto,
        hints: list[list[str]],
        student_output: Optional[DataProto] = None,
    ) -> DataProto:
        """Build teacher prompts for on-policy distillation with GT always retained."""
        reference_trajectories, student_trajectories = (
            self._build_teacher_context_trajectories(
                batch,
                student_output=student_output,
            )
        )
        return construct_teacher_prompts(
            questions=self._get_teacher_questions(batch),
            hints=hints,
            gt_answers=batch.non_tensor_batch["ground_truth"],
            p_drop=0.0,
            tokenizer=self.tokenizer,
            processor=self.processor,
            reference_trajectories=reference_trajectories,
            student_trajectories=student_trajectories,
            template_path=self._teacher_template_path,
            max_prompt_length=self.config.data.max_prompt_length,
            multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
            image_min_pixels=self.config.data.image_min_pixels,
            image_max_pixels=self.config.data.image_max_pixels,
            video_min_pixels=self.config.data.video_min_pixels,
            video_max_pixels=self.config.data.video_max_pixels,
            video_fps=self.config.data.video_fps,
            video_max_frames=self.config.data.video_max_frames,
            apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
        )

    def _select_dynamic_hint_targets(
        self,
        rollout_correctness: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select the best hint group and one correct rollout from it for each question."""
        batch_size, _, _ = rollout_correctness.shape
        device = rollout_correctness.device
        hint_group_scores = rollout_correctness.float().mean(dim=-1)
        selected_hint_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        selected_rollout_indices = torch.full(
            (batch_size,), -1, dtype=torch.long, device=device
        )
        selected_has_correct = torch.zeros(batch_size, dtype=torch.bool, device=device)
        selected_group_scores = torch.zeros(
            batch_size, dtype=hint_group_scores.dtype, device=device
        )

        for i in range(batch_size):
            hint_scores = hint_group_scores[i]
            max_score = hint_scores.max()
            best_hint_candidates = torch.nonzero(
                torch.isclose(hint_scores, max_score), as_tuple=True
            )[0]
            chosen_hint = best_hint_candidates[
                np.random.randint(best_hint_candidates.numel())
            ]
            selected_hint_indices[i] = chosen_hint
            selected_group_scores[i] = hint_scores[chosen_hint]

            correct_rollout_candidates = torch.nonzero(
                rollout_correctness[i, chosen_hint] > 0, as_tuple=True
            )[0]
            if correct_rollout_candidates.numel() > 0:
                selected_has_correct[i] = True
                selected_rollout_indices[i] = correct_rollout_candidates[
                    np.random.randint(correct_rollout_candidates.numel())
                ]

        return (
            hint_group_scores,
            selected_hint_indices,
            selected_rollout_indices,
            selected_has_correct,
            selected_group_scores,
        )

    def _build_student_gen_batch(self, batch: DataProto) -> DataProto:
        """Build rollout prompts from the current batch without removing prompt text fields."""
        student_gen = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[
                "raw_prompt_ids",
                "multi_modal_data",
                "has_offline_trajectory",
                "offline_output",
            ],
            meta_info_keys=[
                "image_min_pixels",
                "image_max_pixels",
                "video_min_pixels",
                "video_max_pixels",
                "video_fps",
                "video_max_frames",
            ],
        )
        if "raw_prompt" in batch.non_tensor_batch:
            student_gen.non_tensor_batch["raw_prompt"] = batch.non_tensor_batch[
                "raw_prompt"
            ].copy()
        return student_gen

    def _compute_response_rewards(
        self,
        source_batch: DataProto,
        rollout_output: DataProto,
        repeat: int = 1,
    ) -> tuple[torch.Tensor, dict[str, list[float]]]:
        reward_batch = DataProto.from_dict(
            tensors={
                "responses": rollout_output.batch["responses"],
                "response_mask": rollout_output.batch["response_mask"],
            },
            non_tensors={},
        )
        for key, value in source_batch.non_tensor_batch.items():
            if key in ("multi_modal_data", "raw_prompt_ids"):
                continue
            reward_batch.non_tensor_batch[key] = (
                np.repeat(value, repeat) if repeat > 1 else value
            )
        return ray.get(self.reward_fn.compute_reward.remote(reward_batch))

    def _select_sdpo_feedback(
        self,
        candidate_output: DataProto,
        candidate_scores: torch.Tensor,
        batch_size: int,
    ) -> tuple[list[str], list[str], dict[str, float]]:
        candidate_n = self.opsd_config.sdpo_num_candidates
        response_ids = candidate_output.batch["responses"]
        response_mask = candidate_output.batch["response_mask"]
        response_lengths = response_mask.sum(dim=-1).to(torch.long)

        correct_solutions: list[str] = []
        incorrect_solutions: list[str] = []
        num_with_correct = 0
        num_with_incorrect = 0

        for i in range(batch_size):
            start = i * candidate_n
            end = start + candidate_n
            group_scores = candidate_scores[start:end]
            group_lengths = response_lengths[start:end]

            correct_idx = (group_scores > 0).nonzero(as_tuple=True)[0]
            incorrect_idx = (group_scores <= 0).nonzero(as_tuple=True)[0]

            selected_correct = ""
            if len(correct_idx) > 0:
                best_rel = correct_idx[group_lengths[correct_idx].argmin()].item()
                best_idx = start + best_rel
                cur_len = int(response_lengths[best_idx].item())
                selected_correct = self.tokenizer.decode(
                    response_ids[best_idx][:cur_len], skip_special_tokens=True
                ).strip()
                selected_correct = truncate_text_by_tokens(
                    selected_correct,
                    self.tokenizer,
                    self.opsd_config.teacher_max_trajectory_length,
                )
                num_with_correct += 1

            selected_incorrect = ""
            if len(incorrect_idx) > 0:
                best_rel = incorrect_idx[group_lengths[incorrect_idx].argmin()].item()
                best_idx = start + best_rel
                cur_len = int(response_lengths[best_idx].item())
                selected_incorrect = self.tokenizer.decode(
                    response_ids[best_idx][:cur_len], skip_special_tokens=True
                ).strip()
                selected_incorrect = truncate_text_by_tokens(
                    selected_incorrect,
                    self.tokenizer,
                    self.opsd_config.teacher_max_trajectory_length,
                )
                num_with_incorrect += 1

            correct_solutions.append(selected_correct)
            incorrect_solutions.append(selected_incorrect)

        metrics = {
            "sdpo/num_prompts_with_correct_candidate": float(num_with_correct),
            "sdpo/num_prompts_with_incorrect_candidate": float(num_with_incorrect),
            "sdpo/correct_candidate_coverage": float(
                num_with_correct / max(batch_size, 1)
            ),
            "sdpo/incorrect_candidate_coverage": float(
                num_with_incorrect / max(batch_size, 1)
            ),
        }
        return correct_solutions, incorrect_solutions, metrics

    def _make_opsd_batch(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate a full OPSD batch with multi-round generation.

        Steps:
            1. Load data batch with trajectories
            2. Student on-policy rollout (n=opsd.student_rollout_n)
            3. Navigator hint generation (n=K)
            4. Teacher trajectory generation (n=1, B*K prompts)
            5. Evaluate Teacher trajectories for binary rewards
            6. Compute Navigator GRPO advantage
            7. Assemble OPSD forward pass batch

        Returns:
            dict with all data needed for joint update.
        """
        timing_raw = {}

        # === Step 0: Load data ===
        try:
            batch_dict = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.train_dataloader)
            batch_dict = next(self.data_iterator)

        meta_info = {
            "image_min_pixels": self.config.data.image_min_pixels,
            "image_max_pixels": self.config.data.image_max_pixels,
            "video_min_pixels": self.config.data.video_min_pixels,
            "video_max_pixels": self.config.data.video_max_pixels,
            "video_fps": self.config.data.video_fps,
            "video_max_frames": self.config.data.video_max_frames,
        }
        batch = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
        batch_size = len(batch.batch)
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(batch_size)], dtype=object
        )

        use_gt_as_hint = self.opsd_config.use_gt_as_hint
        enable_sdpo = self.opsd_config.enable_sdpo
        self_distill_negative_off_policy = (
            use_gt_as_hint and self.opsd_config.self_distill_negative_off_policy
        )
        student_rollout_n = int(self.opsd_config.student_rollout_n)
        dynamic_sample_hint = bool(self.opsd_config.dynamic_sample_hint)
        teacher_rollout_n = int(self.opsd_config.teacher_rollout_n)
        K = 1 if (use_gt_as_hint or enable_sdpo) else self.opsd_config.num_hints
        skip_nav_grpo = (
            (self.opsd_config.lambda_nav == 0) or use_gt_as_hint or enable_sdpo
        )
        skip_off_policy = (
            ((self.opsd_config.alpha >= 1.0) and (not self_distill_negative_off_policy))
            or (use_gt_as_hint and (not self_distill_negative_off_policy))
            or enable_sdpo
        )

        student_gen_base = self._build_student_gen_batch(batch)
        sdpo_correct_solutions: Optional[list[str]] = None
        sdpo_incorrect_solutions: Optional[list[str]] = None

        # === Step 1: Student On-Policy Rollout ===
        if enable_sdpo:
            candidate_n = self.opsd_config.sdpo_num_candidates
            with timer("sdpo_candidate_rollout", timing_raw):
                sdpo_candidate_gen = deepcopy(student_gen_base)
                sdpo_candidate_gen.meta_info.update(
                    {
                        "n": candidate_n,
                        "temperature": self.config.worker.rollout.temperature,
                        "top_p": self.config.worker.rollout.top_p,
                        "max_tokens": self.config.data.max_response_length,
                        "image_min_pixels": self.config.data.image_min_pixels,
                        "image_max_pixels": self.config.data.image_max_pixels,
                        "video_min_pixels": self.config.data.video_min_pixels,
                        "video_max_pixels": self.config.data.video_max_pixels,
                        "video_fps": self.config.data.video_fps,
                        "video_max_frames": self.config.data.video_max_frames,
                    }
                )
                sdpo_candidate_gen, sdpo_candidate_pad_size = pad_dataproto_to_divisor(
                    sdpo_candidate_gen, self.actor_rollout_ref_wg.world_size
                )
                sdpo_candidate_output = self.actor_rollout_ref_wg.generate_sequences(
                    sdpo_candidate_gen
                )
                sdpo_candidate_output = unpad_dataproto(
                    sdpo_candidate_output,
                    pad_size=sdpo_candidate_pad_size * candidate_n,
                )

            with timer("sdpo_candidate_reward", timing_raw):
                sdpo_candidate_reward_tensor, sdpo_candidate_reward_metrics = (
                    self._compute_response_rewards(
                        batch, sdpo_candidate_output, repeat=candidate_n
                    )
                )
                metrics.update(
                    self._reduce_reward_metrics_for_logging(
                        sdpo_candidate_reward_metrics,
                        prefix="sdpo_candidate_reward/",
                    )
                )
                sdpo_candidate_scores = sdpo_candidate_reward_tensor.sum(dim=-1)
                (
                    sdpo_correct_solutions,
                    sdpo_incorrect_solutions,
                    sdpo_selection_metrics,
                ) = self._select_sdpo_feedback(
                    sdpo_candidate_output, sdpo_candidate_scores, batch_size
                )
                metrics.update(sdpo_selection_metrics)

            with timer("student_rollout", timing_raw):
                student_gen = deepcopy(student_gen_base)
                student_gen.meta_info.update(
                    {
                        "n": student_rollout_n,
                        "temperature": self.config.worker.rollout.temperature,
                        "top_p": self.config.worker.rollout.top_p,
                        "max_tokens": self.config.data.max_response_length,
                        "image_min_pixels": self.config.data.image_min_pixels,
                        "image_max_pixels": self.config.data.image_max_pixels,
                        "video_min_pixels": self.config.data.video_min_pixels,
                        "video_max_pixels": self.config.data.video_max_pixels,
                        "video_fps": self.config.data.video_fps,
                        "video_max_frames": self.config.data.video_max_frames,
                    }
                )
                student_gen, student_pad_size = pad_dataproto_to_divisor(
                    student_gen, self.actor_rollout_ref_wg.world_size
                )
                student_output = self.actor_rollout_ref_wg.generate_sequences(
                    student_gen
                )
                student_output = unpad_dataproto(
                    student_output, pad_size=student_pad_size * student_rollout_n
                )
                student_gen = unpad_dataproto(student_gen, pad_size=student_pad_size)
        else:
            with timer("student_rollout", timing_raw):
                student_gen = deepcopy(student_gen_base)
                student_gen.meta_info.update(
                    {
                        "n": student_rollout_n,
                        "temperature": self.config.worker.rollout.temperature,
                        "top_p": self.config.worker.rollout.top_p,
                        "max_tokens": self.config.data.max_response_length,
                        "image_min_pixels": self.config.data.image_min_pixels,
                        "image_max_pixels": self.config.data.image_max_pixels,
                        "video_min_pixels": self.config.data.video_min_pixels,
                        "video_max_pixels": self.config.data.video_max_pixels,
                        "video_fps": self.config.data.video_fps,
                        "video_max_frames": self.config.data.video_max_frames,
                    }
                )
                student_gen, student_pad_size = pad_dataproto_to_divisor(
                    student_gen, self.actor_rollout_ref_wg.world_size
                )
                student_output = self.actor_rollout_ref_wg.generate_sequences(
                    student_gen
                )
                student_output = unpad_dataproto(
                    student_output, pad_size=student_pad_size * student_rollout_n
                )
                student_gen = unpad_dataproto(student_gen, pad_size=student_pad_size)

        expected_student_bs = batch_size * student_rollout_n
        if len(student_output) != expected_student_bs:
            raise RuntimeError(
                f"Unexpected student rollout batch size. expected={expected_student_bs}, got={len(student_output)}. "
                f"base_batch_size={batch_size}, student_rollout_n={student_rollout_n}"
            )

        if student_rollout_n > 1:
            batch = batch.repeat(repeat_times=student_rollout_n, interleave=True)
            # Keep each repeated student sample independent in downstream grouping/metrics.
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )
            if enable_sdpo:
                if sdpo_correct_solutions is not None:
                    sdpo_correct_solutions = np.repeat(
                        np.array(sdpo_correct_solutions, dtype=object),
                        student_rollout_n,
                    ).tolist()
                if sdpo_incorrect_solutions is not None:
                    sdpo_incorrect_solutions = np.repeat(
                        np.array(sdpo_incorrect_solutions, dtype=object),
                        student_rollout_n,
                    ).tolist()

        # Merge student output back
        batch = batch.union(student_output)
        batch_size = len(batch.batch)
        metrics["opsd/student_rollout_n"] = float(student_rollout_n)
        metrics["opsd/student_rollout_effective_batch"] = float(batch_size)
        self._log_student_generations(batch, student_output)

        print(
            f"[OPSD] Step 1: Student rollout complete. "
            f"base_batch_size={expected_student_bs // student_rollout_n}, "
            f"student_rollout_n={student_rollout_n}, effective_batch_size={batch_size}"
        )

        student_old_lp = None
        student_outcome_advantages = None
        student_outcome_response_mask = None
        student_distill_response_mask = None
        reward_tensor = None
        reward_metrics = None
        if use_gt_as_hint or enable_sdpo:
            with timer("student_reward", timing_raw):
                reward_tensor, reward_metrics = self._compute_response_rewards(
                    batch, student_output
                )
                metrics.update(
                    self._reduce_reward_metrics_for_logging(
                        reward_metrics,
                        prefix="student_reward/",
                    )
                )

            if self.opsd_config.outcome_ppo_coef > 0:
                with timer("student_old_logprobs", timing_raw):
                    student_logprob_batch = DataProto.from_dict(
                        tensors={
                            "input_ids": student_output.batch["input_ids"],
                            "attention_mask": student_output.batch["attention_mask"],
                            "position_ids": student_output.batch["position_ids"],
                            "responses": student_output.batch["responses"],
                        },
                        non_tensors={
                            "multi_modal_data": batch.non_tensor_batch.get(
                                "multi_modal_data",
                                np.array(
                                    [{}] * student_output.batch["input_ids"].shape[0],
                                    dtype=object,
                                ),
                            ),
                            "uid": batch.non_tensor_batch["uid"],
                        },
                        meta_info={
                            "temperature": self.config.worker.rollout.temperature,
                            "image_min_pixels": self.config.data.image_min_pixels,
                            "image_max_pixels": self.config.data.image_max_pixels,
                            "video_min_pixels": self.config.data.video_min_pixels,
                            "video_max_pixels": self.config.data.video_max_pixels,
                            "video_fps": self.config.data.video_fps,
                            "video_max_frames": self.config.data.video_max_frames,
                        },
                    )
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(
                        student_logprob_batch
                    )
                    student_old_lp = old_log_probs.batch["old_log_probs"]

                student_outcome_advantages, _ = compute_advantage_return(
                    AdvantageEstimator.REINFORCE_PLUS_PLUS,
                    token_level_rewards=reward_tensor,
                    response_mask=student_output.batch["response_mask"],
                    gamma=torch.as_tensor(
                        self.config.algorithm.gamma,
                        dtype=reward_tensor.dtype,
                        device=reward_tensor.device,
                    ),
                )
                student_outcome_response_mask = student_output.batch["response_mask"]
                student_distill_response_mask = student_output.batch["response_mask"]
                if self.opsd_config.outcome_ppo_incorrect_only:
                    if "accuracy" in reward_metrics:
                        accuracy = torch.as_tensor(
                            reward_metrics["accuracy"],
                            dtype=student_outcome_advantages.dtype,
                            device=student_outcome_advantages.device,
                        )
                        incorrect_mask = (accuracy <= 0).to(
                            student_output.batch["response_mask"].dtype
                        )
                    else:
                        outcome_scores = reward_tensor.sum(dim=-1)
                        incorrect_mask = (outcome_scores <= 0).to(
                            student_output.batch["response_mask"].dtype
                        )
                    student_outcome_response_mask = student_output.batch[
                        "response_mask"
                    ] * incorrect_mask.unsqueeze(-1)
                    student_distill_response_mask = student_output.batch[
                        "response_mask"
                    ] * (1 - incorrect_mask).unsqueeze(-1)
                    metrics["opsd/outcome_num_incorrect"] = float(
                        incorrect_mask.sum().item()
                    )
                    metrics["opsd/outcome_num_correct"] = float(
                        incorrect_mask.numel() - incorrect_mask.sum().item()
                    )
            if use_gt_as_hint and self.opsd_config.self_distill_mask_prefix_tokens > 0:
                student_distill_response_mask = _mask_prefix_response_tokens(
                    (
                        student_distill_response_mask
                        if student_distill_response_mask is not None
                        else student_output.batch["response_mask"]
                    ),
                    self.opsd_config.self_distill_mask_prefix_tokens,
                )
                masked_prefix_tokens = (
                    student_output.batch["response_mask"].sum()
                    - student_distill_response_mask.sum()
                ).item()
                metrics["opsd/self_distill_masked_prefix_tokens"] = float(
                    masked_prefix_tokens
                )
                metrics["opsd/self_distill_mask_prefix_tokens"] = float(
                    self.opsd_config.self_distill_mask_prefix_tokens
                )
            metrics["opsd/outcome_reward_mean"] = (
                (reward_tensor * student_output.batch["response_mask"]).sum()
                / student_output.batch["response_mask"].sum().clamp(min=1)
            ).item()

        # === Step 2: Navigator Hint Generation (or SDPO/GT-as-hint) ===
        if enable_sdpo:
            nav_output = None
            print(
                "[OPSD] Step 2: SDPO candidate selection complete. "
                f"correct_cov={metrics['sdpo/correct_candidate_coverage']:.3f}, "
                f"incorrect_cov={metrics['sdpo/incorrect_candidate_coverage']:.3f}"
            )
        elif use_gt_as_hint:
            # Self-Distilled Reasoner: skip Navigator rollout. Use the preprocessed
            # reference trajectory directly as teacher-side extra context; fall back
            # to GT only if the dataset does not provide one.
            trajectories = batch.non_tensor_batch.get("trajectory")
            hints = []
            used_trajectory = 0
            for idx, gt in enumerate(batch.non_tensor_batch["ground_truth"]):
                trajectory = ""
                if trajectories is not None:
                    trajectory = truncate_text_by_tokens(
                        trajectories[idx],
                        self.tokenizer,
                        self.opsd_config.teacher_max_trajectory_length,
                    )
                if trajectory:
                    hints.append([trajectory])
                    used_trajectory += 1
                else:
                    hints.append([str(gt)])
            nav_output = None
            print(
                f"[OPSD] Step 2: Using reference trajectories as teacher context where available "
                f"(fallback to GT for {batch_size - used_trajectory}/{batch_size}). B={batch_size}"
            )
        else:
            navigator_student_trajectories = None
            if self.opsd_config.navigator_include_student_trajectory:
                navigator_student_trajectories = self._decode_student_trajectories(
                    student_output,
                    max_tokens=self.opsd_config.navigator_max_trajectory_length,
                )

            with timer("navigator_rollout", timing_raw):
                nav_gen = construct_navigator_prompts(
                    questions=batch.non_tensor_batch["prompt_text"],
                    trajectories=batch.non_tensor_batch["trajectory"],
                    student_trajectories=navigator_student_trajectories,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    template_path=self._nav_template_path,
                    max_traj_len=self.opsd_config.navigator_max_trajectory_length,
                    max_prompt_length=self.config.data.max_prompt_length,
                    multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                    image_min_pixels=self.config.data.image_min_pixels,
                    image_max_pixels=self.config.data.image_max_pixels,
                    video_min_pixels=self.config.data.video_min_pixels,
                    video_max_pixels=self.config.data.video_max_pixels,
                    video_fps=self.config.data.video_fps,
                    video_max_frames=self.config.data.video_max_frames,
                    apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
                )
                nav_gen.meta_info.update(
                    {
                        "n": K,
                        "temperature": self.config.worker.rollout.temperature,
                        "top_p": self.config.worker.rollout.top_p,
                        "max_tokens": self.opsd_config.navigator_response_length,
                        "image_min_pixels": self.config.data.image_min_pixels,
                        "image_max_pixels": self.config.data.image_max_pixels,
                        "video_min_pixels": self.config.data.video_min_pixels,
                        "video_max_pixels": self.config.data.video_max_pixels,
                        "video_fps": self.config.data.video_fps,
                        "video_max_frames": self.config.data.video_max_frames,
                    }
                )

                nav_gen, nav_pad_size = pad_dataproto_to_divisor(
                    nav_gen, self.actor_rollout_ref_wg.world_size
                )
                nav_output = self.actor_rollout_ref_wg.generate_sequences(nav_gen)
                nav_output = unpad_dataproto(nav_output, pad_size=nav_pad_size * K)

            # Decode hints
            nav_output.meta_info["n"] = K
            hints = decode_navigator_hints(nav_output, self.tokenizer)  # B × K
            self._log_navigator_hints(batch, hints)

            print(f"[OPSD] Step 2: Navigator generated {len(hints)} × {K} hints")

        # === Step 3: Teacher ===
        if enable_sdpo:
            p_drop = 0.0
            metrics["curriculum/p_drop"] = p_drop

            with timer("teacher_prompt_tokenize", timing_raw):
                teacher_gen = construct_sdpo_teacher_prompts(
                    questions=self._get_teacher_questions(batch),
                    correct_solutions=sdpo_correct_solutions or [""] * batch_size,
                    incorrect_solutions=sdpo_incorrect_solutions or [""] * batch_size,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    template_path=self._sdpo_teacher_template_path,
                    max_prompt_length=self.config.data.max_prompt_length,
                    multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                    image_min_pixels=self.config.data.image_min_pixels,
                    image_max_pixels=self.config.data.image_max_pixels,
                    video_min_pixels=self.config.data.video_min_pixels,
                    video_max_pixels=self.config.data.video_max_pixels,
                    video_fps=self.config.data.video_fps,
                    video_max_frames=self.config.data.video_max_frames,
                    apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
                )
            teacher_output = None
            teacher_uid = batch.non_tensor_batch["uid"]
            hint_correct = torch.ones(batch_size)
            metrics["nav/hint_accuracy"] = 1.0
            print(
                f"[OPSD] Step 3: SDPO teacher prompt tokenized (no generation). B={batch_size}"
            )
            print("[OPSD] Step 4: Skipped (SDPO, no Teacher trajectories)")
        elif use_gt_as_hint:
            # Self-Distilled Reasoner: skip Teacher generation entirely.
            # Only tokenize Teacher prompts (with GT answer as hint, p_drop=0).
            # The Teacher forward pass happens during actor update, not here.
            p_drop = 0.0
            metrics["curriculum/p_drop"] = p_drop

            with timer("teacher_prompt_tokenize", timing_raw):
                # In self-distillation mode, GT is already provided as `hint`.
                # Pass empty reference answers so the teacher prompt cannot
                # accidentally include a second copy of the ground truth.
                empty_gt_answers = np.array([""] * batch_size, dtype=object)
                teacher_gen = construct_teacher_prompts(
                    questions=self._get_teacher_questions(batch),
                    hints=hints,
                    gt_answers=empty_gt_answers,
                    p_drop=p_drop,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    template_path=self._teacher_template_path,
                    max_prompt_length=self.config.data.max_prompt_length,
                    multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                    image_min_pixels=self.config.data.image_min_pixels,
                    image_max_pixels=self.config.data.image_max_pixels,
                    video_min_pixels=self.config.data.video_min_pixels,
                    video_max_pixels=self.config.data.video_max_pixels,
                    video_fps=self.config.data.video_fps,
                    video_max_frames=self.config.data.video_max_frames,
                    apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
                )
            teacher_on_policy_gen = teacher_gen
            teacher_output = None
            if self_distill_negative_off_policy:
                with timer("teacher_rollout", timing_raw):
                    teacher_gen.meta_info.update(
                        {
                            "n": 1,
                            "temperature": self.config.worker.rollout.temperature,
                            "top_p": self.config.worker.rollout.top_p,
                            "image_min_pixels": self.config.data.image_min_pixels,
                            "image_max_pixels": self.config.data.image_max_pixels,
                            "video_min_pixels": self.config.data.video_min_pixels,
                            "video_max_pixels": self.config.data.video_max_pixels,
                            "video_fps": self.config.data.video_fps,
                            "video_max_frames": self.config.data.video_max_frames,
                        }
                    )
                    teacher_gen, teacher_pad_size = pad_dataproto_to_divisor(
                        teacher_gen, self.actor_rollout_ref_wg.world_size
                    )
                    teacher_output = self.actor_rollout_ref_wg.generate_sequences(
                        teacher_gen
                    )
                    teacher_output = unpad_dataproto(
                        teacher_output, pad_size=teacher_pad_size
                    )
                print(
                    "[OPSD] Step 3: Teacher rollout complete for self-distill negative off-policy. "
                    f"B={batch_size}"
                )
            else:
                print(
                    f"[OPSD] Step 3: Teacher prompt tokenized (no generation). B={batch_size}"
                )

            # Step 4: Skip Teacher evaluation. Self-distill does not reward/filter
            # Teacher trajectories, even when a negative off-policy branch is enabled.
            teacher_uid = batch.non_tensor_batch["uid"]
            hint_correct = torch.ones(batch_size)  # GT hint → assume correct
            metrics["nav/hint_accuracy"] = 1.0
            if self_distill_negative_off_policy:
                print(
                    "[OPSD] Step 4: Skipped (Self-Distilled Reasoner negative off-policy, no Teacher filtering)"
                )
            else:
                print(
                    "[OPSD] Step 4: Skipped (Self-Distilled Reasoner, no Teacher trajectories)"
                )

        else:
            # Full OPSD / Static Navigator: Teacher generates trajectories
            # Curriculum is temporarily disabled: always keep GT available to Teacher.
            p_drop = 0.0
            metrics["curriculum/p_drop"] = p_drop
            teacher_reference_trajectories, teacher_student_trajectories = (
                self._build_teacher_context_trajectories(
                    batch, student_output=student_output
                )
            )
            rollout_gt_answers = (
                batch.non_tensor_batch["ground_truth"]
                if self.opsd_config.off_policy_include_gt_answer
                else np.array([""] * batch_size, dtype=object)
            )

            with timer("teacher_rollout", timing_raw):
                teacher_gen = construct_teacher_prompts(
                    questions=self._get_teacher_questions(batch),
                    hints=hints,
                    gt_answers=rollout_gt_answers,
                    p_drop=p_drop,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    reference_trajectories=teacher_reference_trajectories,
                    student_trajectories=teacher_student_trajectories,
                    template_path=self._teacher_template_path,
                    max_prompt_length=self.config.data.max_prompt_length,
                    multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                    image_min_pixels=self.config.data.image_min_pixels,
                    image_max_pixels=self.config.data.image_max_pixels,
                    video_min_pixels=self.config.data.video_min_pixels,
                    video_max_pixels=self.config.data.video_max_pixels,
                    video_fps=self.config.data.video_fps,
                    video_max_frames=self.config.data.video_max_frames,
                    apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
                )
                teacher_gen.meta_info.update(
                    {
                        "n": teacher_rollout_n,
                        "temperature": self.config.worker.rollout.temperature,
                        "top_p": self.config.worker.rollout.top_p,
                        "image_min_pixels": self.config.data.image_min_pixels,
                        "image_max_pixels": self.config.data.image_max_pixels,
                        "video_min_pixels": self.config.data.video_min_pixels,
                        "video_max_pixels": self.config.data.video_max_pixels,
                        "video_fps": self.config.data.video_fps,
                        "video_max_frames": self.config.data.video_max_frames,
                    }
                )

                teacher_gen, teacher_pad_size = pad_dataproto_to_divisor(
                    teacher_gen, self.actor_rollout_ref_wg.world_size
                )
                teacher_output = self.actor_rollout_ref_wg.generate_sequences(
                    teacher_gen
                )
                teacher_output = unpad_dataproto(
                    teacher_output, pad_size=teacher_pad_size * teacher_rollout_n
                )

            print(
                f"[OPSD] Step 3: Teacher rollout complete. B*K*M = {batch_size * K * teacher_rollout_n}"
            )

            teacher_on_policy_gen = teacher_gen
            if not self.opsd_config.off_policy_include_gt_answer:
                with timer("teacher_on_policy_prompt_tokenize", timing_raw):
                    teacher_on_policy_gen = self._construct_on_policy_teacher_prompts(
                        batch,
                        hints,
                        student_output=student_output,
                    )

            # === Step 4: Evaluate Teacher Trajectories ===
            with timer("teacher_reward", timing_raw):
                teacher_repeat = K * teacher_rollout_n
                teacher_gt = np.repeat(
                    batch.non_tensor_batch["ground_truth"], teacher_repeat
                )
                teacher_rollout_uid = np.repeat(
                    batch.non_tensor_batch["uid"], teacher_repeat
                )

                teacher_reward_batch = DataProto.from_dict(
                    tensors={
                        "responses": teacher_output.batch["responses"],
                        "response_mask": teacher_output.batch["response_mask"],
                    },
                    non_tensors={
                        "ground_truth": teacher_gt,
                        "uid": teacher_rollout_uid,
                    },
                )
                for key in batch.non_tensor_batch:
                    if (
                        key not in teacher_reward_batch.non_tensor_batch
                        and key not in ("multi_modal_data", "raw_prompt_ids")
                    ):
                        teacher_reward_batch.non_tensor_batch[key] = np.repeat(
                            batch.non_tensor_batch[key], teacher_repeat
                        )

                reward_tensor, reward_metrics = ray.get(
                    self.reward_fn.compute_reward.remote(teacher_reward_batch)
                )

                teacher_rollout_rewards = reward_tensor.sum(dim=-1)
                if "accuracy" in reward_metrics:
                    rollout_correctness = torch.as_tensor(
                        reward_metrics["accuracy"],
                        dtype=teacher_rollout_rewards.dtype,
                        device=teacher_rollout_rewards.device,
                    )
                else:
                    rollout_correctness = (teacher_rollout_rewards > 0).to(
                        teacher_rollout_rewards.dtype
                    )

                rollout_correctness = rollout_correctness.view(
                    batch_size, K, teacher_rollout_n
                )
                if dynamic_sample_hint:
                    (
                        hint_group_scores,
                        selected_hint_indices,
                        selected_rollout_indices,
                        selected_has_correct,
                        selected_group_scores,
                    ) = self._select_dynamic_hint_targets(rollout_correctness)
                    hint_correct = hint_group_scores.reshape(-1)
                    teacher_uid = np.repeat(batch.non_tensor_batch["uid"], K)
                else:
                    hint_correct = rollout_correctness.reshape(
                        batch_size * K, teacher_rollout_n
                    ).mean(dim=-1)
                    teacher_uid = np.repeat(batch.non_tensor_batch["uid"], K)

                reward_metrics_reduced = self._reduce_reward_metrics_for_logging(
                    reward_metrics,
                    prefix="teacher_reward/",
                )
                metrics.update(reward_metrics_reduced)

            metrics["nav/hint_accuracy"] = hint_correct.mean().item()
            if dynamic_sample_hint:
                metrics["nav/best_hint_accuracy"] = selected_group_scores.mean().item()
                metrics["opsd/dynamic_selected_hint_has_correct_rate"] = (
                    selected_has_correct.float().mean().item()
                )
                print(
                    "[OPSD] Step 4: Teacher reward computed. "
                    f"hint_group_acc={metrics['nav/hint_accuracy']:.3f}, "
                    f"best_hint_acc={metrics['nav/best_hint_accuracy']:.3f}"
                )
            else:
                print(
                    f"[OPSD] Step 4: Teacher reward computed. Hint accuracy: {metrics['nav/hint_accuracy']:.3f}"
                )

        # === Step 5: Navigator GRPO Advantage ===
        if nav_output is not None:
            nav_resp_len = nav_output.batch["responses"].shape[1]
            nav_response_mask = nav_output.batch["response_mask"]
        else:
            # SDPO / use_gt_as_hint: no Navigator output, use minimal placeholders
            nav_resp_len = 1
            nav_response_mask = torch.zeros(batch_size * K, nav_resp_len)

        if not skip_nav_grpo:
            nav_advantages = compute_navigator_advantage(
                hint_rewards=hint_correct,
                hint_uids=teacher_uid,
                K=K,
            )
            nav_advantages_expanded = (
                nav_advantages.unsqueeze(-1).expand(-1, nav_resp_len)
                * nav_response_mask
            )
            metrics["nav/advantage_std"] = nav_advantages.std().item()
            metrics["nav/advantage_mean"] = nav_advantages.mean().item()
        else:
            # Placeholder: zero advantages (won't be used)
            nav_advantages_expanded = torch.zeros(batch_size * K, nav_resp_len)
            metrics["nav/advantage_std"] = 0.0
            metrics["nav/advantage_mean"] = 0.0

        # === Step 6: Compute Navigator old_log_probs ===
        if not skip_nav_grpo:
            with timer("nav_old_logprobs", timing_raw):
                nav_logprob_batch = DataProto.from_dict(
                    tensors={
                        "input_ids": nav_output.batch["input_ids"],
                        "attention_mask": nav_output.batch["attention_mask"],
                        "position_ids": nav_output.batch["position_ids"],
                        "responses": nav_output.batch["responses"],
                    },
                    non_tensors={
                        "multi_modal_data": nav_output.non_tensor_batch.get(
                            "multi_modal_data",
                            np.array(
                                [{}] * nav_output.batch["input_ids"].shape[0],
                                dtype=object,
                            ),
                        ),
                        "uid": np.repeat(batch.non_tensor_batch["uid"], K),
                    },
                    meta_info={
                        "temperature": self.config.worker.rollout.temperature,
                        "image_min_pixels": self.config.data.image_min_pixels,
                        "image_max_pixels": self.config.data.image_max_pixels,
                        "video_min_pixels": self.config.data.video_min_pixels,
                        "video_max_pixels": self.config.data.video_max_pixels,
                        "video_fps": self.config.data.video_fps,
                        "video_max_frames": self.config.data.video_max_frames,
                    },
                )
                nav_old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(
                    nav_logprob_batch
                )
                nav_old_lp = nav_old_log_probs.batch["old_log_probs"]
        else:
            # Placeholder: zeros (won't be used)
            nav_old_lp = torch.zeros(batch_size * K, nav_resp_len)
            if enable_sdpo:
                print("[OPSD] Skipping Navigator (enable_sdpo=True)")
            elif use_gt_as_hint:
                print(
                    "[OPSD] Skipping Navigator (use_gt_as_hint=True, Self-Distilled Reasoner)"
                )
            else:
                print(
                    "[OPSD] Skipping Navigator old_log_probs (lambda_nav=0, static navigator)"
                )

        # === Step 7: Assemble combined batch for joint update ===
        # We pack all data with prefix keys into a single DataProto

        # --- Navigator data (included even when static, but lambda_nav=0 skips gradient) ---
        if nav_output is not None:
            nav_tensors = {
                "nav_input_ids": nav_output.batch["input_ids"],
                "nav_attention_mask": nav_output.batch["attention_mask"],
                "nav_position_ids": nav_output.batch["position_ids"],
                "nav_responses": nav_output.batch["responses"],
                "nav_response_mask": nav_response_mask,
                "nav_old_log_probs": nav_old_lp,
                "nav_advantages": nav_advantages_expanded,
            }
        else:
            # use_gt_as_hint: no Navigator data, use minimal placeholders
            nav_tensors = {
                "nav_input_ids": torch.zeros(batch_size * K, 1, dtype=torch.long),
                "nav_attention_mask": torch.zeros(batch_size * K, 1, dtype=torch.long),
                "nav_position_ids": torch.zeros(batch_size * K, 1, dtype=torch.long),
                "nav_responses": torch.zeros(batch_size * K, 1, dtype=torch.long),
                "nav_response_mask": nav_response_mask,
                "nav_old_log_probs": nav_old_lp,
                "nav_advantages": nav_advantages_expanded,
            }

        if dynamic_sample_hint and teacher_output is not None:
            on_indices = torch.arange(
                batch_size, device=student_output.batch["responses"].device
            )
            on_question_indices = on_indices
            num_valid_on_policy = len(on_indices)

            student_input_ids_rep = student_output.batch["input_ids"]
            student_attention_mask_rep = student_output.batch["attention_mask"]
            student_position_ids_rep = student_output.batch["position_ids"]
            student_responses_rep = student_output.batch["responses"]
            student_response_mask_rep = student_output.batch["response_mask"]

            selected_hint_indices_dev = selected_hint_indices.to(
                device=on_indices.device
            )
            selected_has_correct_dev = selected_has_correct.to(device=on_indices.device)
            selected_rollout_indices_dev = selected_rollout_indices.to(
                device=on_indices.device
            )
            selected_prompt_rows = on_question_indices * K + selected_hint_indices_dev

            teacher_prompts = teacher_on_policy_gen.batch["input_ids"][
                selected_prompt_rows
            ]
            teacher_prompt_mask = teacher_on_policy_gen.batch["attention_mask"][
                selected_prompt_rows
            ]
            teacher_prompt_position_ids = teacher_on_policy_gen.batch["position_ids"][
                selected_prompt_rows
            ]

            invalid_hint_mask = ~selected_has_correct_dev
            if torch.any(invalid_hint_mask):
                with timer("teacher_on_policy_fallback_prompt_tokenize", timing_raw):
                    fallback_teacher_gen, trajectory_missing = (
                        self._construct_teacher_fallback_prompts(
                            batch,
                            repeat_k=1,
                            student_output=student_output,
                        )
                    )

                fallback_prompts = fallback_teacher_gen.batch["input_ids"]
                fallback_prompt_mask = fallback_teacher_gen.batch["attention_mask"]
                fallback_prompt_position_ids = fallback_teacher_gen.batch[
                    "position_ids"
                ]

                teacher_prompts = teacher_prompts.clone()
                teacher_prompt_mask = teacher_prompt_mask.clone()
                teacher_prompt_position_ids = teacher_prompt_position_ids.clone()
                teacher_prompts[invalid_hint_mask] = fallback_prompts[invalid_hint_mask]
                teacher_prompt_mask[invalid_hint_mask] = fallback_prompt_mask[
                    invalid_hint_mask
                ]
                teacher_prompt_position_ids[invalid_hint_mask] = (
                    fallback_prompt_position_ids[invalid_hint_mask]
                )
                metrics["opsd/on_policy_trajectory_fallback_count"] = float(
                    invalid_hint_mask.sum().item()
                )
                metrics["opsd/on_policy_fallback_missing_trajectory_count"] = float(
                    trajectory_missing
                )
                metrics["opsd/on_policy_gt_fallback_count"] = float(trajectory_missing)
            else:
                metrics["opsd/on_policy_trajectory_fallback_count"] = 0.0
                metrics["opsd/on_policy_fallback_missing_trajectory_count"] = 0.0
                metrics["opsd/on_policy_gt_fallback_count"] = 0.0

            on_teacher_input_ids, on_teacher_attention_mask, on_teacher_position_ids = (
                _concat_prompt_and_response(
                    prompt_ids=teacher_prompts,
                    prompt_attention_mask=teacher_prompt_mask,
                    prompt_position_ids=teacher_prompt_position_ids,
                    response_ids=student_responses_rep,
                    response_mask=student_response_mask_rep,
                )
            )
            on_policy_tensors = {
                "on_student_input_ids": student_input_ids_rep,
                "on_student_attention_mask": student_attention_mask_rep,
                "on_student_position_ids": student_position_ids_rep,
                "on_student_responses": student_responses_rep,
                "on_student_response_mask": student_response_mask_rep,
                "on_teacher_input_ids": on_teacher_input_ids,
                "on_teacher_attention_mask": on_teacher_attention_mask,
                "on_teacher_position_ids": on_teacher_position_ids,
                "on_teacher_responses": student_responses_rep,
                "on_teacher_response_mask": student_response_mask_rep,
                "on_student_distill_response_mask": (
                    student_distill_response_mask
                    if student_distill_response_mask is not None
                    else student_response_mask_rep
                ),
            }
            if student_old_lp is not None and student_outcome_advantages is not None:
                on_policy_tensors["on_student_old_log_probs"] = student_old_lp
                on_policy_tensors["on_student_outcome_advantages"] = (
                    student_outcome_advantages
                )
                on_policy_tensors["on_student_outcome_response_mask"] = (
                    student_outcome_response_mask
                )

            nonempty_on_mask = (
                (student_attention_mask_rep.sum(dim=-1) > 0)
                & (student_response_mask_rep.sum(dim=-1) > 0)
                & (on_teacher_attention_mask.sum(dim=-1) > 0)
            )
            if not torch.all(nonempty_on_mask):
                on_indices = on_indices[nonempty_on_mask]
                on_question_indices = on_question_indices[nonempty_on_mask]
                selected_has_correct_dev = selected_has_correct_dev[nonempty_on_mask]
                selected_rollout_indices_dev = selected_rollout_indices_dev[
                    nonempty_on_mask
                ]
                selected_hint_indices_dev = selected_hint_indices_dev[nonempty_on_mask]
                num_valid_on_policy = int(nonempty_on_mask.sum().item())
                on_policy_tensors = {
                    key: tensor[nonempty_on_mask]
                    for key, tensor in on_policy_tensors.items()
                }

            on_teacher_attention_mask = on_policy_tensors["on_teacher_attention_mask"]
            off_policy_tensors = {}
            valid_question_indices = on_question_indices[selected_has_correct_dev]
            num_valid = len(valid_question_indices)
            if not skip_off_policy and num_valid > 0:
                valid_hint_rows = (
                    valid_question_indices * K
                    + selected_hint_indices_dev[selected_has_correct_dev]
                )
                valid_rollout_rows = (
                    valid_hint_rows * teacher_rollout_n
                    + selected_rollout_indices_dev[selected_has_correct_dev]
                )
                teacher_correct_responses = teacher_output.batch["responses"][
                    valid_rollout_rows
                ]
                teacher_correct_resp_mask = teacher_output.batch["response_mask"][
                    valid_rollout_rows
                ]

                student_prompts_valid = student_output.batch["prompts"][
                    valid_question_indices
                ]
                student_prompt_len = student_prompts_valid.shape[1]
                student_prompt_mask_valid = student_output.batch["attention_mask"][
                    valid_question_indices, :student_prompt_len
                ]
                student_prompt_position_ids_valid = student_output.batch[
                    "position_ids"
                ][valid_question_indices, ..., :student_prompt_len]
                (
                    off_student_input_ids,
                    off_student_attention_mask,
                    off_student_position_ids,
                ) = _concat_prompt_and_response(
                    prompt_ids=student_prompts_valid,
                    prompt_attention_mask=student_prompt_mask_valid,
                    prompt_position_ids=student_prompt_position_ids_valid,
                    response_ids=teacher_correct_responses,
                    response_mask=teacher_correct_resp_mask,
                )

                off_policy_tensors = {
                    "off_student_input_ids": off_student_input_ids,
                    "off_student_attention_mask": off_student_attention_mask,
                    "off_student_position_ids": off_student_position_ids,
                    "off_student_responses": teacher_correct_responses,
                    "off_student_response_mask": teacher_correct_resp_mask,
                    "off_teacher_input_ids": teacher_output.batch["input_ids"][
                        valid_rollout_rows
                    ],
                    "off_teacher_attention_mask": teacher_output.batch[
                        "attention_mask"
                    ][valid_rollout_rows],
                    "off_teacher_position_ids": teacher_output.batch["position_ids"][
                        valid_rollout_rows
                    ],
                    "off_teacher_responses": teacher_correct_responses,
                    "off_teacher_response_mask": teacher_correct_resp_mask,
                }
            elif skip_off_policy:
                print(
                    "[OPSD] Skipping off-policy batch assembly (alpha=1.0, on-policy only)"
                )

            metrics["opsd/num_valid_hints"] = num_valid
            metrics["opsd/num_total_hints"] = batch_size * K
            metrics["opsd/dynamic_selected_hint_count"] = float(batch_size)
        else:
            on_indices = torch.arange(
                batch_size * K, device=student_output.batch["responses"].device
            )
            if use_gt_as_hint or enable_sdpo:
                on_positive_mask = torch.ones_like(on_indices, dtype=torch.bool)
            else:
                on_positive_mask = hint_correct.to(device=on_indices.device) > 0
            num_valid_on_policy = len(on_indices)

            student_input_ids_rep = student_output.batch["input_ids"].repeat_interleave(
                K, dim=0
            )[on_indices]
            student_attention_mask_rep = student_output.batch[
                "attention_mask"
            ].repeat_interleave(K, dim=0)[on_indices]
            student_position_ids_rep = student_output.batch[
                "position_ids"
            ].repeat_interleave(K, dim=0)[on_indices]
            student_responses_rep = student_output.batch["responses"].repeat_interleave(
                K, dim=0
            )[on_indices]
            student_response_mask_rep = student_output.batch[
                "response_mask"
            ].repeat_interleave(K, dim=0)[on_indices]
            on_question_indices = (
                on_indices // K if num_valid_on_policy > 0 else on_indices
            )

            on_policy_use_fallback_prompt = False
            if teacher_output is not None:
                teacher_prompts = teacher_on_policy_gen.batch["input_ids"][on_indices]
                teacher_prompt_mask = teacher_on_policy_gen.batch["attention_mask"][
                    on_indices
                ]
                teacher_prompt_position_ids = teacher_on_policy_gen.batch[
                    "position_ids"
                ][on_indices]

                invalid_hint_mask = ~on_positive_mask
                if (
                    (not use_gt_as_hint)
                    and (not enable_sdpo)
                    and torch.any(invalid_hint_mask)
                ):
                    with timer(
                        "teacher_on_policy_fallback_prompt_tokenize", timing_raw
                    ):
                        fallback_teacher_gen, trajectory_missing = (
                            self._construct_teacher_fallback_prompts(
                                batch,
                                repeat_k=K,
                                student_output=student_output,
                            )
                        )

                    fallback_prompts = fallback_teacher_gen.batch["input_ids"][
                        on_indices
                    ]
                    fallback_prompt_mask = fallback_teacher_gen.batch["attention_mask"][
                        on_indices
                    ]
                    fallback_prompt_position_ids = fallback_teacher_gen.batch[
                        "position_ids"
                    ][on_indices]

                    teacher_prompts = teacher_prompts.clone()
                    teacher_prompt_mask = teacher_prompt_mask.clone()
                    teacher_prompt_position_ids = teacher_prompt_position_ids.clone()
                    teacher_prompts[invalid_hint_mask] = fallback_prompts[
                        invalid_hint_mask
                    ]
                    teacher_prompt_mask[invalid_hint_mask] = fallback_prompt_mask[
                        invalid_hint_mask
                    ]
                    teacher_prompt_position_ids[invalid_hint_mask] = (
                        fallback_prompt_position_ids[invalid_hint_mask]
                    )
                    on_policy_use_fallback_prompt = True
                    metrics["opsd/on_policy_trajectory_fallback_count"] = float(
                        invalid_hint_mask.sum().item()
                    )
                    metrics["opsd/on_policy_fallback_missing_trajectory_count"] = float(
                        trajectory_missing
                    )
                    metrics["opsd/on_policy_gt_fallback_count"] = float(
                        trajectory_missing
                    )
                else:
                    metrics["opsd/on_policy_trajectory_fallback_count"] = 0.0
                    metrics["opsd/on_policy_fallback_missing_trajectory_count"] = 0.0
                    metrics["opsd/on_policy_gt_fallback_count"] = 0.0
            else:
                teacher_prompts = teacher_gen.batch["input_ids"]
                teacher_prompt_mask = teacher_gen.batch["attention_mask"]
                teacher_prompt_position_ids = teacher_gen.batch["position_ids"]
                metrics["opsd/on_policy_trajectory_fallback_count"] = 0.0
                metrics["opsd/on_policy_fallback_missing_trajectory_count"] = 0.0
                metrics["opsd/on_policy_gt_fallback_count"] = 0.0

            student_resp_for_teacher = (
                student_responses_rep
                if teacher_output is not None
                else student_output.batch["responses"]
            )
            student_resp_mask_for_teacher = (
                student_response_mask_rep
                if teacher_output is not None
                else student_output.batch["response_mask"]
            )

            on_teacher_input_ids, on_teacher_attention_mask, on_teacher_position_ids = (
                _concat_prompt_and_response(
                    prompt_ids=teacher_prompts,
                    prompt_attention_mask=teacher_prompt_mask,
                    prompt_position_ids=teacher_prompt_position_ids,
                    response_ids=student_resp_for_teacher,
                    response_mask=student_resp_mask_for_teacher,
                )
            )
            on_teacher_responses = student_resp_for_teacher
            on_teacher_response_mask = student_resp_mask_for_teacher

            if student_distill_response_mask is not None:
                on_student_distill_response_mask = student_distill_response_mask[
                    on_indices
                ]
            else:
                if on_policy_use_fallback_prompt:
                    on_student_distill_response_mask = student_response_mask_rep
                else:
                    on_student_distill_response_mask = (
                        student_response_mask_rep
                        * on_positive_mask[on_indices]
                        .unsqueeze(-1)
                        .to(student_response_mask_rep.dtype)
                    )

            on_policy_tensors = {
                "on_student_input_ids": student_input_ids_rep,
                "on_student_attention_mask": student_attention_mask_rep,
                "on_student_position_ids": student_position_ids_rep,
                "on_student_responses": student_responses_rep,
                "on_student_response_mask": student_response_mask_rep,
                "on_teacher_input_ids": on_teacher_input_ids,
                "on_teacher_attention_mask": on_teacher_attention_mask,
                "on_teacher_position_ids": on_teacher_position_ids,
                "on_teacher_responses": on_teacher_responses,
                "on_teacher_response_mask": on_teacher_response_mask,
                "on_student_distill_response_mask": on_student_distill_response_mask,
            }
            if student_old_lp is not None and student_outcome_advantages is not None:
                on_policy_tensors["on_student_old_log_probs"] = student_old_lp[
                    on_indices
                ]
                on_policy_tensors["on_student_outcome_advantages"] = (
                    student_outcome_advantages[on_indices]
                )
                on_policy_tensors["on_student_outcome_response_mask"] = (
                    student_outcome_response_mask[on_indices]
                )

            if num_valid_on_policy > 0:
                nonempty_on_mask = (
                    (student_attention_mask_rep.sum(dim=-1) > 0)
                    & (student_response_mask_rep.sum(dim=-1) > 0)
                    & (on_teacher_attention_mask.sum(dim=-1) > 0)
                    & (on_teacher_response_mask.sum(dim=-1) > 0)
                )
                if not torch.all(nonempty_on_mask):
                    on_indices = on_indices[nonempty_on_mask]
                    on_question_indices = on_question_indices[nonempty_on_mask]
                    on_positive_mask = on_positive_mask[on_indices]
                    num_valid_on_policy = int(nonempty_on_mask.sum().item())
                    on_policy_tensors = {
                        key: tensor[nonempty_on_mask]
                        for key, tensor in on_policy_tensors.items()
                    }
                else:
                    on_positive_mask = on_positive_mask[on_indices]

            on_teacher_attention_mask = on_policy_tensors["on_teacher_attention_mask"]
            valid_indices = on_indices[on_positive_mask]
            valid_question_indices = on_question_indices[on_positive_mask]

            off_policy_tensors = {}
            num_valid = 0
            if not skip_off_policy:
                if self_distill_negative_off_policy:
                    valid_indices = torch.arange(
                        batch_size,
                        device=student_output.batch["responses"].device,
                        dtype=torch.long,
                    )
                    valid_question_indices = valid_indices
                num_valid = len(valid_indices)

                if num_valid > 0:
                    teacher_correct_responses = teacher_output.batch["responses"][
                        valid_indices
                    ]
                    teacher_correct_resp_mask = teacher_output.batch["response_mask"][
                        valid_indices
                    ]

                    student_prompts_valid = student_output.batch["prompts"][
                        valid_question_indices
                    ]
                    student_prompt_len = student_prompts_valid.shape[1]
                    student_prompt_mask_valid = student_output.batch["attention_mask"][
                        valid_question_indices, :student_prompt_len
                    ]
                    student_prompt_position_ids_valid = student_output.batch[
                        "position_ids"
                    ][valid_question_indices, ..., :student_prompt_len]
                    (
                        off_student_input_ids,
                        off_student_attention_mask,
                        off_student_position_ids,
                    ) = _concat_prompt_and_response(
                        prompt_ids=student_prompts_valid,
                        prompt_attention_mask=student_prompt_mask_valid,
                        prompt_position_ids=student_prompt_position_ids_valid,
                        response_ids=teacher_correct_responses,
                        response_mask=teacher_correct_resp_mask,
                    )

                    off_policy_tensors = {
                        "off_student_input_ids": off_student_input_ids,
                        "off_student_attention_mask": off_student_attention_mask,
                        "off_student_position_ids": off_student_position_ids,
                        "off_student_responses": teacher_correct_responses,
                        "off_student_response_mask": teacher_correct_resp_mask,
                        "off_teacher_input_ids": teacher_output.batch["input_ids"][
                            valid_indices
                        ],
                        "off_teacher_attention_mask": teacher_output.batch[
                            "attention_mask"
                        ][valid_indices],
                        "off_teacher_position_ids": teacher_output.batch[
                            "position_ids"
                        ][valid_indices],
                        "off_teacher_responses": teacher_correct_responses,
                        "off_teacher_response_mask": teacher_correct_resp_mask,
                    }
            else:
                print(
                    "[OPSD] Skipping off-policy batch assembly (alpha=1.0, on-policy only)"
                )

            metrics["opsd/num_valid_hints"] = num_valid
            metrics["opsd/num_total_hints"] = batch_size * K

        # Pad all tensors to same batch size (nav: B*K, on: B*K, off: num_valid)
        # We'll handle different sizes in the actor by processing each group separately
        # For now, pack them all into a single DataProto with separate key prefixes

        # Find the max sequence length across all groups to pad
        all_tensors = {}
        all_tensors.update(nav_tensors)
        all_tensors.update(on_policy_tensors)
        all_tensors.update(off_policy_tensors)

        # We need all tensors to have the same batch size for DataProto
        # Strategy: pad smaller batches to B*K (the largest group)
        target_bs = batch_size * K

        def pad_batch_to_size(tensor_dict, target_size, prefix=""):
            """Pad a dict of tensors along batch dim to target_size."""
            result = {}
            for key, tensor in tensor_dict.items():
                if tensor.shape[0] < target_size:
                    pad_size = target_size - tensor.shape[0]
                    pad_shape = list(tensor.shape)
                    pad_shape[0] = pad_size
                    padding = torch.zeros(
                        pad_shape, dtype=tensor.dtype, device=tensor.device
                    )
                    result[key] = torch.cat([tensor, padding], dim=0)
                elif tensor.shape[0] > target_size:
                    result[key] = tensor[:target_size]
                else:
                    result[key] = tensor
            return result

        # Pad all smaller groups to B*K so DataProto sees a consistent batch
        # dimension. The actor uses num_valid_on_policy/num_valid_off_policy
        # to ignore the padded tail.
        if on_policy_tensors:
            on_policy_tensors = pad_batch_to_size(on_policy_tensors, target_bs)
        if off_policy_tensors:
            off_policy_tensors = pad_batch_to_size(off_policy_tensors, target_bs)

        # Rebuild all_tensors
        all_tensors = {}
        all_tensors.update(nav_tensors)
        all_tensors.update(on_policy_tensors)
        if off_policy_tensors:
            all_tensors.update(off_policy_tensors)

        # --- Multi-modal data for OPSD forward passes ---
        # All three roles (Nav, Student, Teacher) may need pixel_values because
        # the question text contains <image> tags. We pass multi_modal_data from
        # the original batch so that _process_multi_modal_inputs in fsdp_workers
        # can construct multi_modal_inputs on the worker side.
        #
        # Layout: on-policy Student/Teacher share the same image (repeated K times).
        # Nav and off-policy similarly reference the same original images.
        original_mmd = batch.non_tensor_batch.get("multi_modal_data")
        original_uid = batch.non_tensor_batch.get("uid")

        def _repeat_and_pad(arr, repeat_k, target_size, pad_value):
            """Repeat each element K times, then pad to target_size."""
            if arr is None:
                return np.array([pad_value] * target_size, dtype=object)
            repeated = np.repeat(arr, repeat_k)
            if len(repeated) < target_size:
                pad = np.array(
                    [pad_value] * (target_size - len(repeated)), dtype=object
                )
                repeated = np.concatenate([repeated, pad])
            return repeated[:target_size]

        nav_mmd = _repeat_and_pad(original_mmd, K, target_bs, {})
        nav_uid = _repeat_and_pad(original_uid, K, target_bs, "__nav_pad__")
        if num_valid_on_policy > 0:
            on_question_indices_np = on_question_indices.detach().cpu().numpy()
            on_mmd = (
                original_mmd[on_question_indices_np]
                if original_mmd is not None
                else None
            )
            on_uid = (
                original_uid[on_question_indices_np]
                if original_uid is not None
                else None
            )
        else:
            on_mmd = None
            on_uid = None
        on_student_mmd = _repeat_and_pad(on_mmd, 1, target_bs, {})
        on_student_uid = _repeat_and_pad(on_uid, 1, target_bs, "__on_pad__")
        if num_valid > 0:
            valid_question_indices_np = valid_question_indices.detach().cpu().numpy()
            off_mmd = (
                original_mmd[valid_question_indices_np]
                if original_mmd is not None
                else None
            )
            off_uid = (
                original_uid[valid_question_indices_np]
                if original_uid is not None
                else None
            )
        else:
            off_mmd = None
            off_uid = None
        off_mmd = _repeat_and_pad(off_mmd, 1, target_bs, {})
        off_uid = _repeat_and_pad(off_uid, 1, target_bs, "__off_pad__")

        non_tensors = {
            "nav_multi_modal_data": nav_mmd,
            "nav_uid": nav_uid,
            "on_multi_modal_data": on_student_mmd,
            "on_uid": on_student_uid,
            "off_multi_modal_data": off_mmd,
            "off_uid": off_uid,
        }

        combined_batch = DataProto.from_dict(
            tensors=all_tensors,
            non_tensors=non_tensors,
            meta_info={
                "temperature": self.config.worker.rollout.temperature,
                "loss_type": "opsd",
                "opsd_config": {
                    "alpha": self.opsd_config.alpha,
                    "lambda_nav": self.opsd_config.lambda_nav,
                    "distillation_loss_type": self.opsd_config.distillation_loss_type,
                    "opsd_kl_penalty": self.opsd_config.opsd_kl_penalty,
                    "distillation_topk": self.opsd_config.distillation_topk,
                    "distillation_topk_source": self.opsd_config.distillation_topk_source,
                    "distillation_add_tail": self.opsd_config.distillation_add_tail,
                    "distill_token_selection": self.opsd_config.distill_token_selection,
                    "distill_token_keep_ratio": self.opsd_config.distill_token_keep_ratio,
                    "dynamic_sample_hint": self.opsd_config.dynamic_sample_hint,
                    "teacher_rollout_n": self.opsd_config.teacher_rollout_n,
                    "outcome_ppo_coef": self.opsd_config.outcome_ppo_coef,
                    "outcome_ppo_incorrect_only": self.opsd_config.outcome_ppo_incorrect_only,
                    "nav_clip_ratio_low": self.opsd_config.nav_clip_ratio_low,
                    "nav_clip_ratio_high": self.opsd_config.nav_clip_ratio_high,
                    "nav_clip_ratio_dual": self.opsd_config.nav_clip_ratio_dual,
                    "self_distill_negative_off_policy": self.opsd_config.self_distill_negative_off_policy,
                    "self_distill_negative_off_policy_coef": self.opsd_config.self_distill_negative_off_policy_coef,
                    "self_distill_mask_prefix_tokens": self.opsd_config.self_distill_mask_prefix_tokens,
                },
                "num_valid_off_policy": num_valid,
                "num_valid_on_policy": num_valid_on_policy,
                "global_token_num": torch.sum(
                    on_teacher_attention_mask, dim=-1
                ).tolist(),
                # Pixel config for _process_multi_modal_inputs
                "image_min_pixels": self.config.data.image_min_pixels,
                "image_max_pixels": self.config.data.image_max_pixels,
                "video_min_pixels": self.config.data.video_min_pixels,
                "video_max_pixels": self.config.data.video_max_pixels,
                "video_fps": self.config.data.video_fps,
                "video_max_frames": self.config.data.video_max_frames,
            },
        )

        # Timing metrics
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

        return combined_batch, batch, metrics

    def fit(self):
        """The OPSD training loop.

        Per step:
          1. Student rollout, Navigator hint generation, Teacher rollout
          2. Evaluate Teacher trajectories → binary rewards
          3. Navigator GRPO advantage
          4. Joint update: Navigator GRPO + OPSD distillation
        """
        self.logger = self._create_logger()
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="OPSD Training", position=0)
        val_metrics = None

        # Load checkpoint
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # Validation before training
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)

        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # === GENERATION PHASE ===
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    combined_batch, original_batch, gen_metrics = self._make_opsd_batch(
                        metrics
                    )
                    self.actor_rollout_ref_wg.release_rollout_engine()
                    metrics.update(gen_metrics)

                # === JOINT UPDATE ===
                with timer("update_actor", timing_raw):
                    actor_output = self.actor_rollout_ref_wg.update_actor(
                        combined_batch
                    )

                actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                metrics.update(actor_metrics)

                # === VALIDATION ===
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)

                # === CHECKPOINT ===
                if should_save_checkpoint(
                    self.global_step,
                    save_freq=self.config.trainer.save_freq,
                    save_steps=self.config.trainer.save_steps,
                ):
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # Collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(
                compute_timing_metrics(batch=original_batch, timing_raw=timing_raw)
            )

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # Final validation
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(
                f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}"
            )

        if not should_save_checkpoint(
            self.global_step,
            save_freq=self.config.trainer.save_freq,
            save_steps=self.config.trainer.save_steps,
        ):
            self._save_checkpoint()

    def _create_logger(self):
        """Create logger instance."""
        from ..utils.logger import Tracker

        return Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())


class RaySimpleGRPOOPSDTrainer(RayOPSDTrainer):
    """Separate minimal GRPO+OPSD trainer without Navigator logic."""

    RUBRIC_INITIALIZATION_CHECKPOINT = "rubric_initialization_state.json"
    RUBRIC_EVOLUTION_CHECKPOINT = "rubric_evolution_state.json"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if self.opsd_config.enable_rubric_generator_training:
            if self.config.algorithm.disable_kl or not self.config.algorithm.use_kl_loss:
                raise ValueError("RG joint mode currently requires fixed-reference KL as a loss term, not reward shaping.")
            if self.config.worker.actor.ulysses_size != 1:
                raise ValueError("RG joint mode currently supports text-only FSDP with ulysses_size=1.")
            if self.config.worker.actor.fsdp.enable_cpu_offload:
                raise ValueError("RG joint gradient accumulation cannot use trainable FSDP CPU offload.")
        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise ValueError(
                "Simple GRPO+OPSD mode requires algorithm.adv_estimator=grpo."
            )
        if self.config.worker.rollout.n <= 1:
            raise ValueError("Simple GRPO+OPSD mode requires worker.rollout.n > 1.")
        if (
            self.opsd_config.enable_rubric_evolution
            and self.config.algorithm.pipeline_reward
        ):
            raise ValueError(
                "opsd.enable_rubric_evolution requires algorithm.pipeline_reward=False "
                "so judge scoring uses the active rubric applied before reward computation."
            )
        if (
            self.opsd_config.enable_rubric_initialization
            and self.config.algorithm.pipeline_reward
        ):
            raise ValueError(
                "opsd.enable_rubric_initialization requires algorithm.pipeline_reward=False "
                "so the generated root rubric is applied before reward computation."
            )
        print("[OPSD] Mode: Simple GRPO+RLSD (no Navigator)")
        print(f"[OPSD]   - rollout.n: {self.config.worker.rollout.n}")
        print(f"[OPSD]   - stgca_lambda: {self.opsd_config.stgca_lambda}")
        print(
            f"[OPSD]   - stgca_lambda_warmup_steps: {self.opsd_config.stgca_lambda_warmup_steps}"
        )
        print(f"[OPSD]   - stgca_thought_only: {self.opsd_config.stgca_thought_only}")
        print(
            f"[OPSD]   - stgca_teacher_sync_interval: {self.opsd_config.stgca_teacher_sync_interval}"
        )
        print(f"[OPSD]   - stgca_negative_only: {self.opsd_config.stgca_negative_only}")
        print(
            f"[OPSD]   - stgca_lambda_decay_steps: {self.opsd_config.stgca_lambda_decay_steps}"
        )
        print(
            "[OPSD]   - enable_rubric_guided_rlsd: "
            f"{self.opsd_config.enable_rubric_guided_rlsd}"
        )
        if self.opsd_config.enable_rubric_guided_rlsd:
            print(
                "[OPSD]   - rubric_guided_teacher_template: "
                f"{self.opsd_config.rubric_guided_teacher_template}"
            )
        print(f"[OPSD]   - enable_casd: {self.opsd_config.enable_casd}")
        if self.opsd_config.enable_casd:
            print(f"[OPSD]   - casd_max_criteria: {self.opsd_config.casd_max_criteria}")
            print(
                f"[OPSD]   - casd_teacher_forward_chunk_size: "
                f"{self.opsd_config.casd_teacher_forward_chunk_size}"
            )
            print(
                f"[OPSD]   - casd_use_variance_weights: {self.opsd_config.casd_use_variance_weights}"
            )
            print(
                f"[OPSD]   - casd_teacher_template: {self.opsd_config.casd_teacher_template}"
            )
        print(
            "[OPSD]   - enable_dead_criteria_guided_rollout: "
            f"{self.opsd_config.enable_dead_criteria_guided_rollout}"
        )
        if self.opsd_config.enable_dead_criteria_guided_rollout:
            print(
                "[OPSD]   - dead_criteria_guided_rollout_n: "
                f"{self.opsd_config.dead_criteria_guided_rollout_n}"
            )
            print(
                "[OPSD]   - dead_criteria_guided_rollout_template: "
                f"{self.opsd_config.dead_criteria_guided_rollout_template}"
            )
        print(
            f"[OPSD]   - stgca_reweight_clip_range: "
            f"{self.opsd_config.stgca_reweight_clip_range}"
        )
        self.initial_rubric_prompt_builder: Optional[InitialRubricPromptBuilder] = None
        self.rubric_initialization_store: Optional[RubricInitializationStore] = None
        self.rubric_reference_index: Optional[ReferenceAnswerIndex] = None
        if self.opsd_config.enable_rubric_initialization:
            initialization_mode = self.opsd_config.rubric_initialization_mode
            self.initial_rubric_prompt_builder = InitialRubricPromptBuilder(
                self.opsd_config.rubric_initialization_prompt_dir,
                few_shot_examples=(
                    self.opsd_config.rubric_initialization_few_shot_examples
                ),
            )
            self.rubric_initialization_store = RubricInitializationStore(
                initialization_mode
            )
            self.rubric_reference_index = ReferenceAnswerIndex(
                self.opsd_config.rubric_initialization_reference_source
                if initialization_mode == REFERENCE_GROUNDED
                else None
            )
            print("[OPSD]   - enable_rubric_initialization: True")
            print("[OPSD]   - rubric_initialization_mode: " f"{initialization_mode}")
            if initialization_mode == REFERENCE_GROUNDED:
                print(
                    "[OPSD]   - rubric_initialization_reference_source_entries: "
                    f"{len(self.rubric_reference_index)}"
                )
        self.rubric_evolver: Optional[RubricEvolver] = None
        self.rubric_meta_verifier: Optional[RubricMetaVerifierRouter] = None
        if self.opsd_config.enable_rubric_evolution:
            judge_fn, judge_batch_fn = self._build_rubric_evolution_judge_fns()
            self.rubric_evolver = RubricEvolver(
                self.opsd_config,
                judge_fn=judge_fn,
                judge_batch_fn=judge_batch_fn,
            )
            self.rubric_meta_verifier = RubricMetaVerifierRouter(
                gap_template_path=self.opsd_config.rubric_gap_meta_verify_template,
                child_template_path=(
                    self.opsd_config.rubric_child_meta_verify_template
                    or self.opsd_config.rubric_evolution_batch_verify_template
                ),
            )
            print("[OPSD]   - enable_rubric_evolution: True")
            print(
                "[OPSD]   - enable_rubric_vertical_evolution: "
                f"{self.opsd_config.enable_rubric_vertical_evolution}"
            )
            print(
                f"[OPSD]   - rubric_evolution_interval: "
                f"{self.opsd_config.rubric_evolution_interval}"
            )
            delayed_steps = int(self.opsd_config.rubric_evolution_delayed_steps)
            if delayed_steps == 0:
                print(
                    "[OPSD]   - rubric_evolution_activation: current_step "
                    "(accepted children update current reward and Teacher context)"
                )
            else:
                print(
                    "[OPSD]   - rubric_evolution_activation: delayed_future_step "
                    f"({delayed_steps} steps)"
                )
            print(
                f"[OPSD]   - rubric_evolution_max_targets_per_step: "
                f"{self.opsd_config.rubric_evolution_max_targets_per_step}"
            )
            print(
                f"[OPSD]   - rubric_evolution_judge_max_workers: "
                f"{self.opsd_config.rubric_evolution_judge_max_workers}"
            )
            print(
                "[OPSD]   - rubric_evolution_transfer_parent_weight_to_children: "
                f"{self.opsd_config.rubric_evolution_transfer_parent_weight_to_children}"
            )

        self.rubric_generator_training = None
        if self.opsd_config.enable_rubric_generator_training:
            from .rubric_generator_training import RubricGeneratorTraining
            from .rubric_generator_verifier import FixedRubricMetaVerifier

            fixed_meta = FixedRubricMetaVerifier(self.config.worker.reward,
                overrides=self.opsd_config.rubric_generator_meta_overrides,
                max_workers=self.opsd_config.rubric_evolution_judge_max_workers,
                log_dir=str(Path(self.opsd_config.rubric_generator_log_dir) / "meta"))
            self.rubric_generator_training = RubricGeneratorTraining(
                config=self.opsd_config, evolver=self.rubric_evolver,
                meta_router=self.rubric_meta_verifier,
                generate=self._generate_rubric_evolution_texts, audit=fixed_meta.generate,
                decode=self._decode_batch_responses,
                build_vertical_hint=self._build_rubric_evolution_batch_proposal_hint,
                parent_id=self._rubric_evolution_parent_id,
                operation=self._rubric_evolution_operation,
            )
            rg_roles = "GAP + vertical" if self.opsd_config.enable_rubric_vertical_evolution else "GAP only"
            print(f"[RubricGenerator] Shared-policy {rg_roles} GRPO enabled; fixed meta-verifier; one joint optimizer step per PPO epoch.")

    def _save_extra_checkpoint_state(self, folder_path: str) -> None:
        super()._save_extra_checkpoint_state(folder_path)
        rubric_initialization_store = getattr(self, "rubric_initialization_store", None)
        rubric_evolver = getattr(self, "rubric_evolver", None)
        if rubric_initialization_store is None and rubric_evolver is None:
            return

        checkpoint_dir = Path(folder_path)
        if rubric_initialization_store is not None:
            state_path = checkpoint_dir / self.RUBRIC_INITIALIZATION_CHECKPOINT
            temporary_path = state_path.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(
                    rubric_initialization_store.state_dict(),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary_path.replace(state_path)
        if rubric_evolver is not None:
            state_path = checkpoint_dir / self.RUBRIC_EVOLUTION_CHECKPOINT
            temporary_path = state_path.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(
                    rubric_evolver.store.state_dict(),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary_path.replace(state_path)

        from liverubric.checkpoints import publish_checkpoint
        publish_checkpoint(checkpoint_dir,
            world_size=self.config.trainer.nnodes * self.config.trainer.n_gpus_per_node,
            teacher=bool(self.config.worker.actor.separate_teacher_model),
            initialization=rubric_initialization_store is not None,
            evolution=rubric_evolver is not None)

    def _load_extra_checkpoint_state(self, folder_path: str) -> None:
        super()._load_extra_checkpoint_state(folder_path)
        rubric_initialization_store = getattr(self, "rubric_initialization_store", None)
        rubric_evolver = getattr(self, "rubric_evolver", None)
        if rubric_initialization_store is None and rubric_evolver is None:
            return

        checkpoint_dir = Path(folder_path)
        from liverubric.checkpoints import validate_checkpoint
        validate_checkpoint(checkpoint_dir,
            world_size=self.config.trainer.nnodes * self.config.trainer.n_gpus_per_node,
            teacher=bool(self.config.worker.actor.separate_teacher_model),
            initialization=rubric_initialization_store is not None,
            evolution=rubric_evolver is not None)
        if rubric_initialization_store is not None:
            state_path = checkpoint_dir / self.RUBRIC_INITIALIZATION_CHECKPOINT
            if not state_path.exists():
                raise FileNotFoundError(
                    "Rubric initialization is enabled, but the resumed checkpoint "
                    f"has no {self.RUBRIC_INITIALIZATION_CHECKPOINT}: {state_path}. "
                    "Refusing to regenerate root rubrics with different samples."
                )
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            rubric_initialization_store.load_state_dict(payload)
            print(
                "[RubricInitialization] Restored persistent state from "
                f"{state_path} ({len(rubric_initialization_store)} prompts)."
            )

        if rubric_evolver is not None:
            state_path = checkpoint_dir / self.RUBRIC_EVOLUTION_CHECKPOINT
            if not state_path.exists():
                raise FileNotFoundError(
                    "Rubric evolution is enabled, but the resumed checkpoint has no "
                    f"{self.RUBRIC_EVOLUTION_CHECKPOINT}: {state_path}. Refusing to "
                    "silently reset the reward rubric. Disable checkpoint resume or "
                    "provide a checkpoint created with rubric evolution state saving."
                )

            payload = json.loads(state_path.read_text(encoding="utf-8"))
            rubric_evolver.store.load_state_dict(payload)
            print(
                "[RubricEvolution] Restored persistent state from "
                f"{state_path} ({len(rubric_evolver.store._states)} prompts)."
            )

    def _build_rubric_evolution_judge_fns(self):
        import importlib.util
        import sys

        reward_config = self.config.worker.reward
        if reward_config.reward_function is None:
            raise ValueError("Rubric evolution requires worker.reward.reward_function.")

        spec = importlib.util.spec_from_file_location(
            "rubric_evolution_reward_fn",
            reward_config.reward_function,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["rubric_evolution_reward_fn"] = module
        spec.loader.exec_module(module)
        reward_fn = getattr(module, reward_config.reward_function_name)
        judge_kwargs = dict(reward_config.reward_function_kwargs)
        judge_kwargs["fail_on_error"] = False
        judge_kwargs["max_workers"] = min(
            int(judge_kwargs.get("max_workers", 1)),
            int(self.opsd_config.rubric_evolution_judge_max_workers),
        )

        def _judge_batch(inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
            return reward_fn(inputs, **judge_kwargs)

        def _judge_one(inp: dict[str, Any]) -> dict[str, float]:
            return _judge_batch([inp])[0]

        return _judge_one, _judge_batch

    @staticmethod
    def _format_full_responses(responses: list[str]) -> str:
        """Format every rollout verbatim for target-conditioned discovery.

        Rubric proposal quality depends on comparing distinctions across the
        complete response group. Do not normalize whitespace, select members,
        or take head/tail excerpts here. Prompt length is checked separately and
        fails closed instead of silently discarding rollout evidence.
        """
        return "\n\n".join(
            f"Rollout {idx}:\n{str(response)}"
            for idx, response in enumerate(responses, start=1)
        )

    def _decode_batch_responses(
        self, batch: DataProto, indices: list[int]
    ) -> list[str]:
        response_ids = batch.batch["responses"]
        response_length = torch.sum(batch.batch["response_mask"], dim=-1)
        decoded: list[str] = []
        for idx in indices:
            cur_response_length = int(response_length[idx].item())
            valid_response_ids = response_ids[idx][:cur_response_length]
            decoded.append(
                self.tokenizer.decode(
                    valid_response_ids,
                    skip_special_tokens=self.config.worker.reward.skip_special_tokens,
                )
            )
        return decoded

    def _resolve_rubric_reference_answer(
        self,
        batch: DataProto,
        *,
        sample_index: int,
        question: str,
    ) -> str:
        reference_key = str(self.opsd_config.rubric_initialization_reference_key)
        reference_batch = batch.non_tensor_batch.get(reference_key)
        answer = ""
        if reference_batch is not None:
            answer = reference_answer_text(reference_batch[sample_index])
        if not answer and self.rubric_reference_index is not None:
            answer = self.rubric_reference_index.lookup(question)
        if not answer:
            source = self.opsd_config.rubric_initialization_reference_source
            raise ValueError(
                "reference_grounded rubric initialization could not resolve a "
                f"reference answer for sample {sample_index}. Add batch field "
                f"{reference_key!r} or configure a matching "
                "opsd.rubric_initialization_reference_source. "
                f"Configured source={source!r}; question={question[:300]!r}."
            )
        return answer

    @staticmethod
    def _object_array(values: list[Any]) -> np.ndarray:
        result = np.empty(len(values), dtype=object)
        for index, value in enumerate(values):
            result[index] = value
        return result

    def _initialize_rubrics_for_batch(
        self,
        batch: DataProto,
        timing_raw: dict[str, Any],
    ) -> dict[str, float]:
        """Generate missing prompt roots once, then apply cached roots to the batch."""
        if (
            self.initial_rubric_prompt_builder is None
            or self.rubric_initialization_store is None
        ):
            return {}

        prompt_texts = batch.non_tensor_batch.get("prompt_text")
        if prompt_texts is None:
            raise KeyError(
                "Rubric initialization requires batch.non_tensor_batch['prompt_text']."
            )
        mode = self.opsd_config.rubric_initialization_mode
        indices_by_key: dict[str, list[int]] = {}
        questions_by_key: dict[str, str] = {}
        for sample_index in range(len(batch)):
            question = str(prompt_texts[sample_index] or "").strip()
            key = make_initialization_key(question, mode)
            indices_by_key.setdefault(key, []).append(sample_index)
            questions_by_key.setdefault(key, question)

        missing_entries: list[dict[str, Any]] = []
        missing_metadata: list[tuple[str, str]] = []
        cached_prompts = 0
        for key, sample_indices in indices_by_key.items():
            if self.rubric_initialization_store.get(key) is not None:
                cached_prompts += 1
                continue
            question = questions_by_key[key]
            if mode == QUESTION_ONLY:
                request = InitialRubricRequest(question=question, mode=mode)
            elif mode == REFERENCE_GROUNDED:
                request = InitialRubricRequest(
                    question=question,
                    mode=mode,
                    reference_answer=self._resolve_rubric_reference_answer(
                        batch,
                        sample_index=sample_indices[0],
                        question=question,
                    ),
                )
            elif mode == EVORUBRIC_BOOTSTRAP:
                request = InitialRubricRequest(
                    question=question,
                    mode=mode,
                    responses=tuple(
                        self._decode_batch_responses(batch, sample_indices)
                    ),
                )
            else:
                raise ValueError(f"Unsupported rubric initialization mode: {mode}")
            rendered = self.initial_rubric_prompt_builder.build(request)
            missing_entries.append({"question": "", "hint": rendered.flat_prompt})
            missing_metadata.append((key, question))

        if missing_entries:
            generated_texts = self._generate_rubric_evolution_texts(
                missing_entries,
                template_path=self.opsd_config.rubric_initialization_template,
                max_tokens=self.opsd_config.rubric_initialization_max_tokens,
                timing_raw=timing_raw,
                timer_name="rubric_initialization_generate",
                temperature=self.opsd_config.rubric_initialization_temperature,
                prompt_limit=(self.opsd_config.rubric_initialization_max_prompt_length),
                context_label="Rubric initialization",
                prompt_limit_config_name=(
                    "opsd.rubric_initialization_max_prompt_length"
                ),
            )
            if len(generated_texts) != len(missing_metadata):
                raise RuntimeError(
                    "Initial-rubric generation count mismatch: "
                    f"expected {len(missing_metadata)}, got {len(generated_texts)}."
                )
            for (key, question), generated_text in zip(
                missing_metadata,
                generated_texts,
                strict=True,
            ):
                roots = parse_initial_rubric(
                    generated_text,
                    question=question,
                    mode=mode,
                    min_criteria=(self.opsd_config.rubric_initialization_min_criteria),
                    max_criteria=(self.opsd_config.rubric_initialization_max_criteria),
                )
                self.rubric_initialization_store.put(
                    key=key,
                    question=question,
                    rubric=roots,
                    created_step=self.global_step,
                )

        if "rubric_before_initialization" not in batch.non_tensor_batch:
            source_rubrics = batch.non_tensor_batch.get("rubric")
            original_rubrics = [
                deepcopy(source_rubrics[index]) if source_rubrics is not None else []
                for index in range(len(batch))
            ]
            batch.non_tensor_batch["rubric_before_initialization"] = self._object_array(
                original_rubrics
            )

        active_rubrics: list[list[dict[str, Any]]] = []
        active_rubric_lists: list[list[str]] = []
        prompt_keys: list[str] = []
        for sample_index in range(len(batch)):
            question = str(prompt_texts[sample_index] or "").strip()
            key = make_initialization_key(question, mode)
            roots = self.rubric_initialization_store.get(key)
            if roots is None:
                raise RuntimeError(
                    f"Missing cached initial rubric after generation for key {key}."
                )
            active_rubrics.append(roots)
            active_rubric_lists.append(
                [str(item.get("description") or "") for item in roots]
            )
            prompt_keys.append(key)

        batch.non_tensor_batch["rubric_initialization_prompt_key"] = self._object_array(
            prompt_keys
        )
        batch.non_tensor_batch["rubric"] = self._object_array(active_rubrics)
        batch.non_tensor_batch["rubric_list"] = self._object_array(active_rubric_lists)
        root_counts = [len(items) for items in active_rubrics]
        return {
            "rubric_initialization/generated_prompts": float(len(missing_entries)),
            "rubric_initialization/cached_prompts": float(cached_prompts),
            "rubric_initialization/applied_samples": float(len(batch)),
            "rubric_initialization/root_criteria_mean": float(
                np.mean(root_counts) if root_counts else 0.0
            ),
            "rubric_initialization/store_size": float(
                len(self.rubric_initialization_store)
            ),
        }

    def _generate_rubric_evolution_texts(
        self,
        entries: list[dict[str, Any]],
        *,
        template_path: str,
        max_tokens: int,
        timing_raw: dict[str, Any],
        manage_rollout_engine: bool = True,
        timer_name: str = "rubric_evolution_generate",
        temperature: Optional[float] = None,
        prompt_limit: Optional[int] = None,
        context_label: str = "Rubric evolution",
        prompt_limit_config_name: str = "opsd.rubric_evolution_max_prompt_length",
        n: int = 1,
        keep_rollouts: bool = False,
    ) -> list[str] | RubricGeneratorRollouts:
        if not entries:
            return []
        if n < 1 or (n > 1 and not keep_rollouts):
            raise ValueError("Multi-sample rubric generation must retain its original rollout trajectories.")

        questions = np.array([entry["question"] for entry in entries], dtype=object)
        hints = [[entry["hint"]] for entry in entries]
        empty_gt_answers = np.array([""] * len(entries), dtype=object)

        with timer(timer_name, timing_raw):
            evolution_prompt_limit = int(
                self.opsd_config.rubric_evolution_max_prompt_length
                if prompt_limit is None
                else prompt_limit
            )
            teacher_gen = construct_teacher_prompts(
                questions=questions,
                hints=hints,
                gt_answers=empty_gt_answers,
                p_drop=0.0,
                tokenizer=self.tokenizer,
                processor=self.processor,
                template_path=template_path,
                max_prompt_length=evolution_prompt_limit,
                apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
            )
            # ``construct_teacher_prompts`` normally left-truncates overlong
            # prompts. That is unacceptable for rubric evolution because it can
            # silently remove the question, parent, or early rollout members.
            # Re-tokenize its retained raw strings and fail before generation if
            # even one complete prompt cannot fit.
            raw_prompt_lengths = [
                len(self.tokenizer.encode(str(prompt), add_special_tokens=False))
                for prompt in teacher_gen.non_tensor_batch["raw_prompt"]
            ]
            overlong = [
                (idx, length)
                for idx, length in enumerate(raw_prompt_lengths)
                if length > evolution_prompt_limit
            ]
            if overlong:
                raise ValueError(
                    f"{context_label} requires complete context, but "
                    f"{len(overlong)} prompt(s) exceed "
                    f"{prompt_limit_config_name}="
                    f"{evolution_prompt_limit}; examples={overlong[:4]}. "
                    "Increase the dedicated prompt budget. Refusing "
                    "to silently truncate rollout responses."
                )
            rollout_model_limit = self.config.worker.rollout.max_model_len
            if rollout_model_limit is None:
                rollout_model_limit = int(self.config.data.max_prompt_length) + int(
                    self.config.data.max_response_length
                )
            required_model_length = max(raw_prompt_lengths, default=0) + int(max_tokens)
            if required_model_length > int(rollout_model_limit):
                raise ValueError(
                    f"{context_label} complete prompt plus generation requires "
                    f"{required_model_length} tokens, but "
                    f"worker.rollout.max_model_len={rollout_model_limit}. "
                    "Increase the rollout engine context limit; refusing silent "
                    "truncation."
                )
            teacher_gen.meta_info.update(
                {
                    "n": n,
                    "temperature": (
                        self.config.worker.rollout.temperature
                        if temperature is None
                        else temperature
                    ),
                    "top_p": self.config.worker.rollout.top_p,
                    "max_tokens": max_tokens,
                    "image_min_pixels": self.config.data.image_min_pixels,
                    "image_max_pixels": self.config.data.image_max_pixels,
                    "video_min_pixels": self.config.data.video_min_pixels,
                    "video_max_pixels": self.config.data.video_max_pixels,
                    "video_fps": self.config.data.video_fps,
                    "video_max_frames": self.config.data.video_max_frames,
                }
            )
            teacher_gen, pad_size = pad_dataproto_to_divisor(
                teacher_gen,
                self.actor_rollout_ref_wg.world_size,
            )
            if manage_rollout_engine:
                self.actor_rollout_ref_wg.prepare_rollout_engine()
            try:
                gen_output = self.actor_rollout_ref_wg.generate_sequences(teacher_gen)
            finally:
                if manage_rollout_engine:
                    self.actor_rollout_ref_wg.release_rollout_engine()
            gen_output = unpad_dataproto(gen_output, pad_size=pad_size * n)

        if len(gen_output) != len(entries) * n:
            raise RuntimeError("Rubric rollout count/order does not match prompt_count * n.")

        response_ids = gen_output.batch["responses"]
        response_mask = gen_output.batch["response_mask"]
        response_length = torch.sum(response_mask, dim=-1)
        texts: list[str] = []
        for idx in range(len(entries) * n):
            cur_response_length = int(response_length[idx].item())
            valid_response_ids = response_ids[idx][:cur_response_length]
            texts.append(
                self.tokenizer.decode(
                    valid_response_ids,
                    skip_special_tokens=(True if keep_rollouts else self.config.worker.reward.skip_special_tokens),
                )
            )
        if keep_rollouts:
            return RubricGeneratorRollouts.from_generation(gen_output, texts,
                prompt_count=len(entries), n=n, temperature=float(teacher_gen.meta_info["temperature"]))
        return texts

    def _build_rubric_evolution_proposal_hint(
        self,
        *,
        target,
        rollout_responses: list[str],
    ) -> str:
        polarity = criterion_polarity(target.parent_item)
        score_label = (
            "Raw penalty trigger rate"
            if polarity == CRITERION_POLARITY_NEGATIVE
            else "Raw positive hit rate"
        )
        if polarity == CRITERION_POLARITY_NEGATIVE:
            direction_note = (
                "This is a negative penalty criterion. A raw score of 1 means the "
                "undesired pitfall is triggered; a raw score of 0 means it is avoided. "
                "Any proposed child criteria must remain negative penalty conditions, "
                "not positive requirements."
            )
        else:
            direction_note = (
                "This is a positive requirement criterion. A raw score of 1 means the "
                "desired behavior is present. Any proposed child criteria must remain "
                "positive requirements."
            )
        parent_text = (
            f"Title: {target.parent_item.get('title', '')}\n"
            f"Description: {target.parent_item.get('description', '')}\n"
            f"Weight: {target.parent_item.get('weight', 1.0)}"
        )
        rubric_state = self.rubric_evolver.store.get(target.prompt_key)
        existing_rubric = (
            [] if rubric_state is None else rubric_state.build_active_rubric()
        )
        existing_rows = []
        selected_identity = rubric_item_identity(
            target.parent_item, target.parent_index
        )
        for idx, item in enumerate(existing_rubric):
            if not isinstance(item, dict):
                continue
            marker = (
                " [SELECTED PARENT]"
                if rubric_item_identity(item, idx) == selected_identity
                else ""
            )
            existing_rows.append(
                f"- {item.get('title', '')}{marker}: {item.get('description', '')}"
            )
        sections = [
            f"Parent criterion:\n{parent_text}",
            f"Criterion polarity: {polarity}\n{direction_note}",
            f"Criterion state: {target.state}",
            f"{score_label}: {target.pass_rate:.3f}",
            "Existing rubric coverage (do not duplicate):\n"
            + ("\n".join(existing_rows) if existing_rows else "(none)"),
            (
                "All current rollout members (complete, verbatim; compare the "
                "whole group and use only target-local evidence):\n"
                + self._format_full_responses(rollout_responses)
            ),
        ]
        return "\n\n".join(sections)

    @staticmethod
    def _rubric_evolution_parent_id(target) -> str:
        return f"p{target.parent_index}"

    @staticmethod
    def _rubric_evolution_operation(target) -> str:
        return "decompose" if target.state == CRITERION_STATE_DEAD else "refine"

    def _build_rubric_evolution_batch_proposal_hint(self, targets: list[Any]) -> str:
        if not targets:
            raise ValueError("Batched rubric proposal requires at least one target.")
        prompt_key = targets[0].prompt_key
        if any(target.prompt_key != prompt_key for target in targets):
            raise ValueError("Batched rubric proposal cannot mix different questions.")

        expected_parent_ids = [
            self._rubric_evolution_parent_id(target) for target in targets
        ]
        expected_operations = {
            self._rubric_evolution_parent_id(target): self._rubric_evolution_operation(
                target
            )
            for target in targets
        }

        parent_blocks = []
        selected_identities: dict[str, str] = {}
        for target in targets:
            parent_id = self._rubric_evolution_parent_id(target)
            selected_identities[
                rubric_item_identity(target.parent_item, target.parent_index)
            ] = parent_id
            polarity = criterion_polarity(target.parent_item)
            score_label = (
                "Raw penalty trigger rate"
                if polarity == CRITERION_POLARITY_NEGATIVE
                else "Raw positive hit rate"
            )
            max_candidates = (
                self.opsd_config.rubric_evolution_max_subcriteria
                if target.state == CRITERION_STATE_DEAD
                else self.opsd_config.rubric_evolution_max_refinements
            )
            parent_blocks.append(
                "\n".join(
                    [
                        f"Parent ID: {parent_id}",
                        f"Criterion index: {target.parent_index}",
                        f"Requested operation: {self._rubric_evolution_operation(target)}",
                        f"Maximum candidates: {max_candidates}",
                        f"State: {target.state}",
                        f"Polarity: {polarity}",
                        f"{score_label}: {target.pass_rate:.3f}",
                        f"Title: {target.parent_item.get('title', '')}",
                        f"Description: {target.parent_item.get('description', '')}",
                        f"Weight: {target.parent_item.get('weight', 1.0)}",
                    ]
                )
            )

        rubric_state = self.rubric_evolver.store.get(prompt_key)
        existing_rubric = (
            [] if rubric_state is None else rubric_state.build_active_rubric()
        )
        existing_rows = []
        for index, item in enumerate(existing_rubric):
            if not isinstance(item, dict):
                continue
            selected_parent_id = selected_identities.get(
                rubric_item_identity(item, index)
            )
            selected_marker = (
                f" [SELECTED AS {selected_parent_id}]"
                if selected_parent_id is not None
                else ""
            )
            existing_rows.append(
                f"- criterion_index={index}{selected_marker}; "
                f"title={item.get('title', '')}; weight={item.get('weight', 1.0)}; "
                f"description={item.get('description', '')}"
            )

        return "\n\n".join(
            [
                (
                    "Required output routing:\n"
                    f"Expected parent count: {len(expected_parent_ids)}\n"
                    "Expected Parent IDs in exact output order: "
                    f"{json.dumps(expected_parent_ids, ensure_ascii=False)}\n"
                    "Expected operations by Parent ID: "
                    f"{json.dumps(expected_operations, ensure_ascii=False)}\n"
                    "The targets array must contain this exact Parent ID sequence once "
                    "each, including parents whose candidates array is empty. Do not "
                    "omit, add, rename, or reorder a Parent ID. Candidate IDs are "
                    "assigned by the training system from list position; do not return "
                    "a candidate_id field."
                ),
                "Selected parent criteria:\n\n" + "\n\n".join(parent_blocks),
                "Existing rubric coverage (do not duplicate):\n"
                + ("\n".join(existing_rows) if existing_rows else "(none)"),
                (
                    "All current rollout members (complete and verbatim; shared by every "
                    "selected parent):\n"
                    + self._format_full_responses(targets[0].group_responses)
                ),
            ]
        )

    def _build_rubric_evolution_batch_verifier_hint(
        self,
        candidate_records: list[dict[str, Any]],
    ) -> str:
        if not candidate_records:
            raise ValueError("Batched rubric verifier requires at least one candidate.")
        first_target = candidate_records[0]["target"]
        prompt_key = first_target.prompt_key
        if any(
            record["target"].prompt_key != prompt_key for record in candidate_records
        ):
            raise ValueError("Batched rubric verifier cannot mix different questions.")

        meta_verifier = getattr(self, "rubric_meta_verifier", None)
        if meta_verifier is None:
            # Hint construction itself is path-independent. Keeping this
            # fallback makes the helper usable by lightweight unit fixtures;
            # the live generation path still owns the configured router.
            meta_verifier = RubricMetaVerifierRouter(None, None)

        rubric_state = self.rubric_evolver.store.get(prompt_key)
        active_rubric = (
            [] if rubric_state is None else rubric_state.build_active_rubric()
        )
        existing_rubric = []
        for index, item in enumerate(active_rubric):
            if not isinstance(item, dict):
                continue
            existing_rubric.append(
                {
                    "criterion_index": index,
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "weight": item.get("weight", 1.0),
                }
            )

        verifier_records = []
        for record in candidate_records:
            target = record["target"]
            candidate = record["candidate"]
            verifier_records.append(
                {
                    "candidate_id": str(record["candidate_id"]),
                    "requested_operation": self._rubric_evolution_operation(target),
                    "parent_state": target.state,
                    "parent": {
                        "parent_id": self._rubric_evolution_parent_id(target),
                        "title": target.parent_item.get("title", ""),
                        "description": target.parent_item.get("description", ""),
                        "weight": target.parent_item.get("weight", 1.0),
                        "polarity": criterion_polarity(target.parent_item),
                    },
                    "candidate": {
                        "title": candidate.get("title", ""),
                        "description": candidate.get("description", ""),
                    },
                }
            )

        return meta_verifier.build_batch_hint(
            operation_type=CHILD_EVOLUTION,
            existing_rubric=existing_rubric,
            candidate_records=verifier_records,
        )

    @staticmethod
    def _format_candidate_criteria(
        items: list[dict[str, Any]],
        *,
        skip_index: Optional[int] = None,
    ) -> str:
        chunks = []
        for idx, item in enumerate(items):
            if skip_index is not None and idx == skip_index:
                continue
            chunks.append(
                f"{idx}. Title: {item.get('title', '')}\n"
                f"Description: {item.get('description', '')}"
            )
        return "\n\n".join(chunks) if chunks else "(none)"

    def _build_rubric_evolution_verifier_hint(
        self,
        *,
        target,
        candidate: dict[str, Any],
        existing_rubric: list[dict[str, Any]],
        source: str,
    ) -> str:
        polarity = criterion_polarity(target.parent_item)
        parent_text = (
            f"Title: {target.parent_item.get('title', '')}\n"
            f"Description: {target.parent_item.get('description', '')}\n"
            f"Weight: {target.parent_item.get('weight', 1.0)}"
        )
        candidate_text = (
            f"Title: {candidate.get('title', '')}\n"
            f"Description: {candidate.get('description', '')}"
        )
        other_items = []
        selected_identity = rubric_item_identity(
            target.parent_item, target.parent_index
        )
        for idx, item in enumerate(existing_rubric):
            if not isinstance(item, dict):
                continue
            if rubric_item_identity(item, idx) == selected_identity:
                continue
            other_items.append(
                f"- {item.get('title', '')}: {item.get('description', '')}"
            )
        sections = [
            "Verifier task: apply the six EvoRubric-style error categories to "
            "this targeted candidate.",
            f"Requested operation: {source.upper()}",
            f"Parent state: {target.state}",
            f"Parent criterion:\n{parent_text}",
            f"Parent polarity: {polarity}",
            f"Candidate child criterion:\n{candidate_text}",
            "Other existing criteria (the candidate must not duplicate these):\n"
            + ("\n".join(other_items) if other_items else "(none)"),
        ]
        return "\n\n".join(sections)

    @staticmethod
    def _build_rubric_evolution_collision_items(
        *,
        target,
        parent_rubric: list[dict[str, Any]],
        active_rubric: list[dict[str, Any]],
        sibling_candidates: list[dict[str, Any]],
        candidate_index: int,
    ) -> list[dict[str, str]]:
        """Collect a compact, de-duplicated global collision catalog."""
        selected_parent_identity = rubric_item_identity(
            target.parent_item, target.parent_index
        )
        seen: set[str] = set()
        items: list[dict[str, str]] = []

        def add_item(item: dict[str, Any], fallback_idx: int, prefix: str) -> None:
            if "(cluster)" in str(item.get("title") or "").lower():
                return
            identity = rubric_item_identity(item, fallback_idx)
            if identity == selected_parent_identity or identity in seen:
                return
            seen.add(identity)
            criterion_id = str(item.get("criterion_id") or f"{prefix}_{fallback_idx}")
            description = " ".join(str(item.get("description") or "").split())
            if len(description) > 300:
                description = description[:297] + "..."
            items.append(
                {
                    "criterion_id": criterion_id,
                    "title": str(item.get("title") or ""),
                    "description": description,
                    "weight": str(item.get("weight", 1.0)),
                }
            )

        for idx, item in enumerate(parent_rubric):
            if isinstance(item, dict):
                add_item(item, idx, "parent")
        for idx, item in enumerate(active_rubric):
            if isinstance(item, dict):
                add_item(item, idx, "active")
        for idx, item in enumerate(sibling_candidates):
            if idx == candidate_index:
                continue
            sibling_item = {
                **item,
                "criterion_id": f"proposal_sibling_{idx}",
            }
            add_item(sibling_item, idx, "proposal_sibling")
        return items

    @staticmethod
    def _format_rubric_evolution_collision_items(items: list[dict[str, str]]) -> str:
        rows = [
            f"- criterion_id={item['criterion_id']}\n"
            f"  title={item['title']}\n"
            f"  weight={item.get('weight', '1.0')}\n"
            f"  description={item['description']}"
            for item in items
        ]
        return "\n".join(rows) if rows else "(none)"

    def _build_rubric_evolution_collision_verifier_hint(
        self,
        *,
        target,
        candidate: dict[str, Any],
        parent_rubric: list[dict[str, Any]],
        active_rubric: list[dict[str, Any]],
        sibling_candidates: list[dict[str, Any]],
        candidate_index: int,
    ) -> str:
        candidate_text = (
            f"Title: {candidate.get('title', '')}\n"
            f"Description: {candidate.get('description', '')}"
        )
        catalog_items = self._build_rubric_evolution_collision_items(
            target=target,
            parent_rubric=parent_rubric,
            active_rubric=active_rubric,
            sibling_candidates=sibling_candidates,
            candidate_index=candidate_index,
        )
        catalog = self._format_rubric_evolution_collision_items(catalog_items)
        return (
            "Selected parent (allowed and expected overlap):\n"
            f"Title: {target.parent_item.get('title', '')}\n"
            f"Description: {target.parent_item.get('description', '')}\n\n"
            "Candidate to audit:\n"
            f"{candidate_text}\n\n"
            "Existing criteria to compare one by one:\n"
            f"{catalog}"
        )

    def _run_rubric_gap_expansion(
        self,
        batch: DataProto,
        timing_raw: dict[str, Any],
    ) -> dict[str, float]:
        """Run horizontal coverage expansion on the fixed current rollout groups."""

        rg_training = getattr(self, "rubric_generator_training", None)
        if rg_training is not None:
            # Vertical RG runs in the same stage so both roles share one Judge pass.
            return rg_training.run_step(batch, timing_raw, step=self.global_step,
                vertical=self.opsd_config.enable_rubric_vertical_evolution)

        metrics = {
            "rubric_gap/contexts": 0.0,
            "rubric_gap/proposed_candidates": 0.0,
            "rubric_gap/meta_rejected": 0.0,
            "rubric_gap/pass_rate_rejected": 0.0,
            "rubric_gap/accepted_roots": 0.0,
            "rubric_gap/proposal_parse_failures": 0.0,
            "rubric_gap/meta_parse_failures": 0.0,
        }
        if (
            not self.opsd_config.enable_rubric_gap_expansion
            or self.rubric_evolver is None
            or self.rubric_meta_verifier is None
        ):
            return metrics

        contexts: list[GapExpansionContext] = []
        max_gap_roots = int(self.opsd_config.rubric_gap_max_roots_per_prompt)
        for prompt_key, cache in self.rubric_evolver.observed_groups().items():
            state = self.rubric_evolver.store.get(prompt_key)
            if state is None:
                continue
            current_gap_roots = sum(
                1
                for item in state.parent_rubric
                if str(item.get("source") or "") == GAP_PROPOSAL_OPERATION
            )
            if current_gap_roots >= max_gap_roots:
                continue
            group_indices = [int(index) for index in cache.get("group_indices", [])]
            if not group_indices:
                continue
            contexts.append(
                GapExpansionContext(
                    prompt_key=prompt_key,
                    prompt_text=str(cache.get("prompt_text") or ""),
                    current_rubric=state.build_active_rubric(),
                    group_indices=group_indices,
                    group_responses=self._decode_batch_responses(batch, group_indices),
                )
            )

        metrics["rubric_gap/contexts"] = float(len(contexts))
        if not contexts:
            return metrics

        max_candidates = int(self.opsd_config.rubric_gap_max_candidates_per_prompt)
        proposal_entries = [
            {
                "question": context.prompt_text,
                "hint": build_gap_proposal_hint(
                    context,
                    max_candidates=max_candidates,
                ),
            }
            for context in contexts
        ]
        proposal_texts = self._generate_rubric_evolution_texts(
            proposal_entries,
            template_path=self.opsd_config.rubric_gap_proposal_template,
            max_tokens=self.opsd_config.rubric_evolution_proposer_max_tokens,
            timing_raw=timing_raw,
            timer_name="rubric_gap_proposal_generate",
            context_label="Rubric horizontal gap proposal",
            prompt_limit_config_name="opsd.rubric_evolution_max_prompt_length",
        )
        if len(proposal_texts) != len(contexts):
            raise RuntimeError(
                "Rubric gap proposal count mismatch: "
                f"expected {len(contexts)}, got {len(proposal_texts)}."
            )

        candidate_records_by_prompt: list[list[dict[str, Any]]] = []
        for context_index, (context, proposal_text) in enumerate(
            zip(contexts, proposal_texts)
        ):
            try:
                candidates = parse_gap_candidates(
                    proposal_text,
                    max_items=max_candidates,
                    root_weight=self.opsd_config.rubric_gap_root_weight,
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                metrics["rubric_gap/proposal_parse_failures"] += 1.0
                candidates = []
            records = [
                {
                    "candidate_id": f"gap_p{context_index}_c{candidate_index}",
                    "context": context,
                    "candidate": candidate,
                    "candidate_index": candidate_index,
                }
                for candidate_index, candidate in enumerate(candidates)
            ]
            metrics["rubric_gap/proposed_candidates"] += float(len(records))
            if records:
                candidate_records_by_prompt.append(records)

        max_meta_batch = int(
            self.opsd_config.rubric_evolution_max_candidates_per_meta_batch
        )
        meta_batches: list[list[dict[str, Any]]] = []
        for records in candidate_records_by_prompt:
            for start in range(0, len(records), max_meta_batch):
                meta_batches.append(records[start : start + max_meta_batch])
        if not meta_batches:
            return metrics

        meta_entries = []
        for records in meta_batches:
            context = records[0]["context"]
            meta_entries.append(
                {
                    "question": context.prompt_text,
                    "hint": self.rubric_meta_verifier.build_batch_hint(
                        operation_type=GAP_EXPAND,
                        existing_rubric=context.current_rubric,
                        candidate_records=[
                            {
                                "candidate_id": record["candidate_id"],
                                "candidate": record["candidate"],
                            }
                            for record in records
                        ],
                    ),
                }
            )

        meta_texts = self._generate_rubric_evolution_texts(
            meta_entries,
            template_path=self.rubric_meta_verifier.template_for(GAP_EXPAND),
            max_tokens=self.opsd_config.rubric_evolution_verifier_max_tokens,
            timing_raw=timing_raw,
            timer_name="rubric_gap_meta_verify_generate",
            temperature=0.0,
            context_label="Rubric horizontal gap meta-verification",
            prompt_limit_config_name="opsd.rubric_evolution_max_prompt_length",
        )
        if len(meta_texts) != len(meta_batches):
            raise RuntimeError(
                "Rubric gap meta-verifier count mismatch: "
                f"expected {len(meta_batches)}, got {len(meta_texts)}."
            )

        for records, meta_text in zip(meta_batches, meta_texts):
            expected_candidate_ids = [record["candidate_id"] for record in records]
            routed, diagnostics = self.rubric_meta_verifier.route_batch_response(
                meta_text,
                expected_candidate_ids=expected_candidate_ids,
                operation_type=GAP_EXPAND,
            )
            if not diagnostics["parse_valid"]:
                metrics["rubric_gap/meta_parse_failures"] += 1.0
            for record in records:
                context = record["context"]
                candidate = record["candidate"]
                semantic_info = self.rubric_meta_verifier.parse_candidate(
                    routed[record["candidate_id"]],
                    operation_type=GAP_EXPAND,
                )
                if not semantic_info.get("valid", False):
                    metrics["rubric_gap/meta_rejected"] += 1.0
                    continue
                passed, empirical_info = self.rubric_evolver.verify_candidate_on_group(
                    prompt_text=context.prompt_text,
                    candidate=candidate,
                    source=GAP_PROPOSAL_OPERATION,
                    rollout_responses=context.group_responses,
                    expected_response_count=len(context.group_indices),
                )
                if not passed:
                    metrics["rubric_gap/pass_rate_rejected"] += 1.0
                    continue
                state = self.rubric_evolver.store.get(context.prompt_key)
                if state is None:
                    continue
                accepted_gap_roots = sum(
                    1
                    for item in state.parent_rubric
                    if str(item.get("source") or "") == GAP_PROPOSAL_OPERATION
                )
                if accepted_gap_roots >= max_gap_roots:
                    continue
                accepted = self.rubric_evolver.add_active_gap_root(
                    prompt_key=context.prompt_key,
                    prompt_text=context.prompt_text,
                    candidate=candidate,
                    global_step=self.global_step,
                    semantic_verifier_info=semantic_info,
                    empirical_verify_info=empirical_info,
                    group_indices=context.group_indices,
                )
                metrics["rubric_gap/accepted_roots"] += float(accepted)
        return metrics

    @staticmethod
    def _build_rubric_evolution_collision_pair_hint(
        *,
        target,
        candidate: dict[str, Any],
        existing_item: dict[str, str],
    ) -> str:
        return (
            "Selected parent:\n"
            f"Title: {target.parent_item.get('title', '')}\n"
            f"Weight: {target.parent_item.get('weight', 1.0)}\n"
            f"Description: {target.parent_item.get('description', '')}\n\n"
            "Candidate:\n"
            f"Title: {candidate.get('title', '')}\n"
            f"Inherited weight/polarity: {target.parent_item.get('weight', 1.0)}\n"
            f"Description: {candidate.get('description', '')}\n\n"
            "One suspected conflicting criterion:\n"
            f"criterion_id={existing_item['criterion_id']}\n"
            f"Title: {existing_item['title']}\n"
            f"Weight: {existing_item.get('weight', '1.0')}\n"
            f"Description: {existing_item['description']}"
        )

    @staticmethod
    def _build_rubric_evolution_negative_polarity_hint(
        *,
        target,
        candidate: dict[str, Any],
    ) -> str:
        return (
            "Negative-weight parent penalty condition:\n"
            f"{target.parent_item.get('description', '')}\n\n"
            "Negative-weight candidate penalty condition:\n"
            f"{candidate.get('description', '')}"
        )

    @staticmethod
    def _build_rubric_evolution_parent_novelty_hint(
        *,
        target,
        candidate: dict[str, Any],
        source: str,
    ) -> str:
        return (
            f"Operation: {source.upper()}\n\n"
            "PARENT (P):\n"
            f"{target.parent_item.get('description', '')}\n\n"
            "CANDIDATE (C):\n"
            f"{candidate.get('description', '')}"
        )

    def _run_rubric_evolution(
        self,
        batch: DataProto,
        metrics: dict[str, Any],
        timing_raw: dict[str, Any],
        *,
        activate_immediately: bool = False,
    ) -> dict[str, float]:
        if not self.opsd_config.enable_rubric_vertical_evolution:
            return {
                "rubric_evolution/targets": 0.0,
                "rubric_evolution/accepted_candidates": 0.0,
            }
        if self.rubric_evolver is None:
            return {}
        rg_training = getattr(self, "rubric_generator_training", None)
        if rg_training is not None:
            if not activate_immediately:
                raise ValueError("RG joint training requires immediate rubric activation.")
            # Already run together with GAP in _run_rubric_gap_expansion.
            return {}

        targets = self.rubric_evolver.select_targets(self.global_step)
        if not targets:
            return {"rubric_evolution/targets": 0.0}

        for target in targets:
            target.group_responses = self._decode_batch_responses(
                batch, target.group_indices
            )

        print(
            f"[RubricEvolution] Step {self.global_step}: selected {len(targets)} targets "
            f"for target-conditioned proposal generation."
        )

        with timer("rubric_evolution_rollout", timing_raw):
            self.actor_rollout_ref_wg.prepare_rollout_engine()
            try:
                batch_metrics = {
                    "rubric_evolution/proposal_batch_calls": 0.0,
                    "rubric_evolution/proposal_batch_parse_failures": 0.0,
                    "rubric_evolution/proposal_batch_missing_targets": 0.0,
                    "rubric_evolution/proposal_batch_duplicate_targets": 0.0,
                    "rubric_evolution/proposal_batch_unknown_targets": 0.0,
                    "rubric_evolution/proposal_batch_operation_mismatches": 0.0,
                    "rubric_evolution/meta_verifier_batch_calls": 0.0,
                    "rubric_evolution/meta_verifier_batch_parse_failures": 0.0,
                    "rubric_evolution/meta_verifier_batch_missing_candidates": 0.0,
                    "rubric_evolution/meta_verifier_batch_duplicate_candidates": 0.0,
                    "rubric_evolution/meta_verifier_batch_unknown_candidates": 0.0,
                    "rubric_evolution/meta_verifier_batch_operation_mismatches": 0.0,
                }

                targets_by_prompt: dict[str, list[Any]] = {}
                for target in targets:
                    targets_by_prompt.setdefault(target.prompt_key, []).append(target)

                max_targets_per_prompt = int(
                    self.opsd_config.rubric_evolution_max_targets_per_prompt
                )
                proposal_batches: list[list[Any]] = []
                for prompt_targets in targets_by_prompt.values():
                    for start in range(0, len(prompt_targets), max_targets_per_prompt):
                        proposal_batches.append(
                            prompt_targets[start : start + max_targets_per_prompt]
                        )

                proposal_entries: list[dict[str, Any]] = []
                for proposal_batch in proposal_batches:
                    proposal_entries.append(
                        {
                            "question": proposal_batch[0].prompt_text,
                            "hint": self._build_rubric_evolution_batch_proposal_hint(
                                proposal_batch
                            ),
                        }
                    )

                proposal_payloads: dict[tuple[str, int], str] = {}
                proposal_texts = self._generate_rubric_evolution_texts(
                    proposal_entries,
                    template_path=self.opsd_config.rubric_evolution_batch_proposal_template,
                    max_tokens=self.opsd_config.rubric_evolution_proposer_max_tokens,
                    timing_raw=timing_raw,
                    manage_rollout_engine=False,
                    timer_name="rubric_evolution_batched_proposal_generate",
                )
                batch_metrics["rubric_evolution/proposal_batch_calls"] = float(
                    len(proposal_texts)
                )
                for proposal_batch, proposal_text in zip(
                    proposal_batches, proposal_texts
                ):
                    expected_operations = {
                        self._rubric_evolution_parent_id(
                            target
                        ): self._rubric_evolution_operation(target)
                        for target in proposal_batch
                    }
                    max_items_by_parent = {
                        self._rubric_evolution_parent_id(target): (
                            self.opsd_config.rubric_evolution_max_subcriteria
                            if target.state == CRITERION_STATE_DEAD
                            else self.opsd_config.rubric_evolution_max_refinements
                        )
                        for target in proposal_batch
                    }
                    candidates_by_parent, diagnostics = parse_batched_proposal_payload(
                        proposal_text,
                        expected_operations=expected_operations,
                        max_items_by_parent=max_items_by_parent,
                    )
                    if not diagnostics["parse_valid"]:
                        batch_metrics[
                            "rubric_evolution/proposal_batch_parse_failures"
                        ] += 1.0
                    batch_metrics[
                        "rubric_evolution/proposal_batch_missing_targets"
                    ] += float(len(diagnostics["missing_parent_ids"]))
                    batch_metrics[
                        "rubric_evolution/proposal_batch_duplicate_targets"
                    ] += float(len(diagnostics["duplicate_parent_ids"]))
                    batch_metrics[
                        "rubric_evolution/proposal_batch_unknown_targets"
                    ] += float(len(diagnostics["unknown_parent_ids"]))
                    batch_metrics[
                        "rubric_evolution/proposal_batch_operation_mismatches"
                    ] += float(len(diagnostics["operation_mismatch_parent_ids"]))

                    for target in proposal_batch:
                        parent_id = self._rubric_evolution_parent_id(target)
                        output_key = (
                            "subcriteria"
                            if target.state == CRITERION_STATE_DEAD
                            else "refinements"
                        )
                        proposal_payloads[(target.prompt_key, target.parent_index)] = (
                            json.dumps(
                                {output_key: candidates_by_parent[parent_id]},
                                ensure_ascii=False,
                            )
                        )

                # Verify all candidates for one question together. Explicit ids
                # route results back to the old per-candidate evolver interface;
                # missing/duplicate ids are converted into invalid payloads.
                candidate_records_by_prompt: dict[str, list[dict[str, Any]]] = {}
                for target in targets:
                    proposal_text = proposal_payloads.get(
                        (target.prompt_key, target.parent_index),
                        "",
                    )
                    max_items = (
                        self.opsd_config.rubric_evolution_max_subcriteria
                        if target.state == CRITERION_STATE_DEAD
                        else self.opsd_config.rubric_evolution_max_refinements
                    )
                    try:
                        proposed_items = parse_proposed_criteria(
                            proposal_text,
                            max_items=max_items,
                        )
                    except json.JSONDecodeError:
                        continue
                    for item_idx, item in enumerate(proposed_items):
                        candidate_records_by_prompt.setdefault(
                            target.prompt_key, []
                        ).append(
                            {
                                "candidate_id": (
                                    f"{self._rubric_evolution_parent_id(target)}_c{item_idx}"
                                ),
                                "target": target,
                                "candidate": item,
                                "item_idx": item_idx,
                            }
                        )

                max_candidates_per_batch = int(
                    self.opsd_config.rubric_evolution_max_candidates_per_meta_batch
                )
                meta_batches: list[list[dict[str, Any]]] = []
                for candidate_records in candidate_records_by_prompt.values():
                    for start in range(
                        0, len(candidate_records), max_candidates_per_batch
                    ):
                        meta_batches.append(
                            candidate_records[start : start + max_candidates_per_batch]
                        )

                meta_verifier_entries = [
                    {
                        "question": meta_batch[0]["target"].prompt_text,
                        "hint": self._build_rubric_evolution_batch_verifier_hint(
                            meta_batch
                        ),
                    }
                    for meta_batch in meta_batches
                ]

                proposal_verifier_payloads: dict[tuple[str, int, int], str] = {}
                verifier_texts = self._generate_rubric_evolution_texts(
                    meta_verifier_entries,
                    template_path=self.rubric_meta_verifier.template_for(
                        CHILD_EVOLUTION
                    ),
                    max_tokens=self.opsd_config.rubric_evolution_verifier_max_tokens,
                    timing_raw=timing_raw,
                    manage_rollout_engine=False,
                    timer_name="rubric_evolution_batched_meta_verifier_generate",
                    temperature=0.0,
                )
                batch_metrics["rubric_evolution/meta_verifier_batch_calls"] = float(
                    len(verifier_texts)
                )
                for meta_batch, verifier_text in zip(meta_batches, verifier_texts):
                    expected_candidate_ids = [
                        record["candidate_id"] for record in meta_batch
                    ]
                    routed_payloads, diagnostics = (
                        self.rubric_meta_verifier.route_batch_response(
                            verifier_text,
                            expected_candidate_ids=expected_candidate_ids,
                            operation_type=CHILD_EVOLUTION,
                        )
                    )
                    if not diagnostics["parse_valid"]:
                        batch_metrics[
                            "rubric_evolution/meta_verifier_batch_parse_failures"
                        ] += 1.0
                    batch_metrics[
                        "rubric_evolution/meta_verifier_batch_missing_candidates"
                    ] += float(len(diagnostics["missing_candidate_ids"]))
                    batch_metrics[
                        "rubric_evolution/meta_verifier_batch_duplicate_candidates"
                    ] += float(len(diagnostics["duplicate_candidate_ids"]))
                    batch_metrics[
                        "rubric_evolution/meta_verifier_batch_unknown_candidates"
                    ] += float(len(diagnostics["unknown_candidate_ids"]))
                    batch_metrics[
                        "rubric_evolution/meta_verifier_batch_operation_mismatches"
                    ] += float(len(diagnostics["operation_mismatch_candidate_ids"]))
                    for record in meta_batch:
                        target = record["target"]
                        key = (
                            target.prompt_key,
                            target.parent_index,
                            record["item_idx"],
                        )
                        proposal_verifier_payloads[key] = routed_payloads[
                            record["candidate_id"]
                        ]

                evolution_metrics = self.rubric_evolver.evolve_targets(
                    targets,
                    global_step=self.global_step,
                    contrast_responses={},
                    proposal_payloads=proposal_payloads,
                    proposal_verifier_payloads=proposal_verifier_payloads,
                    parent_novelty_verifier_payloads=None,
                    polarity_verifier_payloads=None,
                    collision_verifier_payloads=None,
                    activate_immediately=activate_immediately,
                )
                evolution_metrics.update(batch_metrics)
                print(
                    f"[RubricEvolution] Step {self.global_step}: "
                    f"proposal_batches={batch_metrics['rubric_evolution/proposal_batch_calls']:.0f}, "
                    f"meta_batches={batch_metrics['rubric_evolution/meta_verifier_batch_calls']:.0f}, "
                    f"proposed={evolution_metrics.get('rubric_evolution/proposed_candidates', 0.0):.0f}, "
                    f"accepted={evolution_metrics.get('rubric_evolution/accepted_candidates', 0.0):.0f}, "
                    f"meta_rejected={evolution_metrics.get('rubric_evolution/verifier_rejected', 0.0):.0f}, "
                    f"split_support_rejected={evolution_metrics.get('rubric_evolution/verify_failed_no_signal', 0.0):.0f}."
                )
                return evolution_metrics

            finally:
                self.actor_rollout_ref_wg.release_rollout_engine()

    @staticmethod
    def _rubric_evolution_wandb_json_cell(value: Any) -> str:
        """Serialize structured values so every W&B table row has one stable type."""
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    def _append_rubric_evolution_jsonl(
        self,
        examples: list[dict[str, Any]],
    ) -> None:
        """Persist every evolution case independently of optional W&B logging."""
        save_dir = getattr(self.config.trainer, "save_checkpoint_path", None)
        if not save_dir:
            return
        try:
            output_path = Path(save_dir) / "rubric_evolution_cases.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as output_file:
                for event in examples:
                    output_file.write(
                        json.dumps(
                            event, ensure_ascii=False, sort_keys=True, default=str
                        )
                        + "\n"
                    )
        except Exception as exc:
            print(
                "[RubricEvolution][WARN] Failed to persist evolution JSONL; "
                f"training continues: {type(exc).__name__}: {exc}"
            )

    def _log_rubric_evolution_wandb_table(self, step: int) -> None:
        if self.rubric_evolver is None:
            return

        examples = self.rubric_evolver.pop_recent_evolution_examples()
        if not examples:
            return
        self._append_rubric_evolution_jsonl(examples)

        if "wandb" not in self.config.trainer.logger:
            return
        try:
            import wandb
        except ImportError:
            return

        try:
            examples_table = wandb.Table(
                columns=[
                    "step",
                    "outcome",
                    "source",
                    "parent_state",
                    "parent_title",
                    "parent_description",
                    "parent_weight",
                    "parent_polarity",
                    "parent_pass_rate",
                    "contrast_parent_ok",
                    "prompt",
                    "contrast",
                    "rollout_sample",
                    "child_title",
                    "child_description",
                    "candidate_status",
                    "verify_pass_rate",
                    "verify_detail",
                    "candidate_score_vector",
                    "semantic_verify_reason",
                    "semantic_failed_dimensions",
                    "semantic_uncertain_dimensions",
                    "semantic_verify_checks",
                    "collision_verify_reason",
                    "collision_ids",
                    "active_rubric_before_count",
                    "active_rubric_before",
                    "active_rubric_replaced_parent",
                    "active_rubric_added",
                    "active_rubric_replacement_children",
                    "active_rubric_after_count",
                    "active_rubric_after",
                ]
            )
            accepted_table = wandb.Table(
                columns=[
                    "step",
                    "source",
                    "parent_title",
                    "parent_weight",
                    "parent_polarity",
                    "title",
                    "description",
                    "candidate_pass_rate",
                    "parent_pass_rate",
                    "active_rubric_before_count",
                    "active_rubric_before",
                    "active_rubric_replaced_parent",
                    "active_rubric_added",
                    "active_rubric_replacement_children",
                    "active_rubric_after_count",
                    "active_rubric_after",
                ]
            )
            for event in examples:
                examples_table.add_data(
                    event.get("step"),
                    event.get("outcome"),
                    event.get("source"),
                    event.get("parent_state"),
                    event.get("parent_title"),
                    event.get("parent_description"),
                    event.get("parent_weight"),
                    event.get("parent_polarity"),
                    event.get("parent_pass_rate"),
                    event.get("contrast_parent_ok"),
                    event.get("prompt"),
                    event.get("contrast"),
                    event.get("rollout_sample"),
                    event.get("child_title"),
                    event.get("child_description"),
                    event.get("candidate_status"),
                    event.get("verify_pass_rate"),
                    event.get("verify_detail"),
                    self._rubric_evolution_wandb_json_cell(
                        event.get("candidate_score_vector")
                    ),
                    event.get("semantic_verify_reason"),
                    self._rubric_evolution_wandb_json_cell(
                        event.get("semantic_failed_dimensions")
                    ),
                    self._rubric_evolution_wandb_json_cell(
                        event.get("semantic_uncertain_dimensions")
                    ),
                    self._rubric_evolution_wandb_json_cell(
                        event.get("semantic_verify_checks")
                    ),
                    event.get("collision_verify_reason"),
                    self._rubric_evolution_wandb_json_cell(event.get("collision_ids")),
                    event.get("active_rubric_before_count"),
                    event.get("active_rubric_before"),
                    event.get("active_rubric_replaced_parent"),
                    event.get("active_rubric_added"),
                    event.get("active_rubric_replacement_children"),
                    event.get("active_rubric_after_count"),
                    event.get("active_rubric_after"),
                )
                if event.get("outcome") == "accepted":
                    accepted_table.add_data(
                        event.get("step"),
                        event.get("source"),
                        event.get("parent_title"),
                        event.get("parent_weight"),
                        event.get("parent_polarity"),
                        event.get("child_title"),
                        event.get("child_description"),
                        event.get("verify_pass_rate"),
                        event.get("parent_pass_rate"),
                        event.get("active_rubric_before_count"),
                        event.get("active_rubric_before"),
                        event.get("active_rubric_replaced_parent"),
                        event.get("active_rubric_added"),
                        event.get("active_rubric_replacement_children"),
                        event.get("active_rubric_after_count"),
                        event.get("active_rubric_after"),
                    )

            log_payload: dict[str, Any] = {
                "rubric_evolution/examples": examples_table,
                "rubric_evolution/accepted_candidates_table": accepted_table,
            }
            featured = next(
                (event for event in examples if event.get("outcome") == "accepted"),
                examples[0],
            )
            log_payload["rubric_evolution/featured_example"] = wandb.Html(
                self._format_rubric_evolution_featured_example(featured)
            )
            wandb.log(log_payload, step=step)
        except Exception as exc:
            print(
                "[RubricEvolution][WARN] Failed to log evolution tables to W&B; "
                f"training continues: {type(exc).__name__}: {exc}"
            )

    @staticmethod
    def _format_rubric_evolution_featured_example(event: dict[str, Any]) -> str:
        def esc(value: Any) -> str:
            return (
                str(value if value is not None else "")
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

        return (
            "<pre style='white-space:pre-wrap;font-size:12px;'>"
            f"outcome: {esc(event.get('outcome'))}\n"
            f"source: {esc(event.get('source'))} | parent_state: {esc(event.get('parent_state'))}\n"
            f"parent: {esc(event.get('parent_title'))} | "
            f"weight={esc(event.get('parent_weight'))} | "
            f"polarity={esc(event.get('parent_polarity'))} | "
            f"raw_score_rate={esc(event.get('parent_pass_rate'))}\n"
            f"{esc(event.get('parent_description'))}\n\n"
            f"prompt:\n{esc(event.get('prompt'))}\n\n"
            f"contrast:\n{esc(event.get('contrast'))}\n\n"
            f"rollout sample:\n{esc(event.get('rollout_sample'))}\n\n"
            f"proposed child:\n{esc(event.get('child_title'))}\n"
            f"{esc(event.get('child_description'))}\n\n"
            f"verify: pass_rate={esc(event.get('verify_pass_rate'))} "
            f"detail={esc(event.get('verify_detail'))}\n\n"
            f"meta failed dimensions: "
            f"{esc(event.get('semantic_failed_dimensions'))}\n"
            f"meta uncertain dimensions: "
            f"{esc(event.get('semantic_uncertain_dimensions'))}\n"
            f"meta checks:\n{esc(event.get('semantic_verify_checks'))}\n\n"
            f"active rubric before evolution "
            f"(count={esc(event.get('active_rubric_before_count'))}):\n"
            f"{esc(event.get('active_rubric_before'))}\n\n"
            f"replaced parent:\n"
            f"{esc(event.get('active_rubric_replaced_parent'))}\n\n"
            f"new child from this proposal:\n"
            f"{esc(event.get('active_rubric_added'))}\n\n"
            f"all active replacement children for this parent:\n"
            f"{esc(event.get('active_rubric_replacement_children'))}\n\n"
            f"active rubric after evolution "
            f"(count={esc(event.get('active_rubric_after_count'))}):\n"
            f"{esc(event.get('active_rubric_after'))}"
            "</pre>"
        )

    @staticmethod
    def _metric_value(
        metric_values: Optional[dict[str, list[Any]]],
        key: str,
        index: int,
        default: float = 0.0,
    ) -> float:
        if metric_values is None:
            return default
        values = metric_values.get(key)
        if values is None or index >= len(values):
            return default
        try:
            value = float(values[index])
        except (TypeError, ValueError):
            return default
        return value if value == value else default

    @staticmethod
    def _ensure_metric_values(
        metric_values: dict[str, list[Any]],
        key: str,
        batch_size: int,
    ) -> list[Any]:
        if key not in metric_values:
            metric_values[key] = [float("nan")] * batch_size
        elif len(metric_values[key]) < batch_size:
            metric_values[key].extend(
                [float("nan")] * (batch_size - len(metric_values[key]))
            )
        return metric_values[key]

    def _compute_weighted_rubric_score(
        self,
        rubric_items: list[dict[str, Any]],
        metric_values: dict[str, list[Any]],
        sample_idx: int,
    ) -> float:
        positive_weight = 0.0
        weighted_score = 0.0
        for criterion_idx, item in enumerate(rubric_items):
            try:
                weight = float(item.get("weight", 1.0) or 0.0)
            except (TypeError, ValueError):
                weight = 0.0
            if weight > 0:
                positive_weight += weight
            criterion_score = self._metric_value(
                metric_values,
                f"criterion_{criterion_idx}_score",
                sample_idx,
                default=0.0,
            )
            weighted_score += criterion_score * weight
        if positive_weight <= 0:
            return 0.0
        return max(0.0, weighted_score / positive_weight)

    def _merge_immediate_evolved_rewards(
        self,
        batch: DataProto,
        *,
        base_rubric_batch: np.ndarray,
        reward_metric_values: dict[str, list[Any]],
        metrics: dict[str, Any],
        timing_raw: dict[str, Any],
    ) -> None:
        """Incrementally fold evolved criteria into the current reward matrix.

        Existing criterion scores come from the diagnostic Judge pass. Scores
        produced while empirically verifying new criteria are reused as well;
        only cache misses trigger a fallback Judge call on the missing new
        criteria.
        """
        if self.rubric_evolver is None:
            return

        active_rubric_batch = batch.non_tensor_batch.get("rubric")
        if active_rubric_batch is None:
            return

        prompt_keys = batch.non_tensor_batch.get("rubric_evolution_prompt_key")
        updated_sample_indices: list[int] = []
        judge_sample_indices: list[int] = []
        evolved_rubrics: list[list[dict[str, Any]]] = []
        cached_new_scores: dict[int, dict[int, float]] = {}
        new_active_indices: dict[int, list[int]] = {}
        batch_size = len(batch)
        reused_old_score_values = 0

        for sample_idx in range(batch_size):
            base_items = base_rubric_batch[sample_idx]
            if isinstance(base_items, np.ndarray):
                base_items = base_items.tolist()
            active_items = active_rubric_batch[sample_idx]
            if isinstance(active_items, np.ndarray):
                active_items = active_items.tolist()
            if not isinstance(base_items, list) or not isinstance(active_items, list):
                continue

            base_identities = {
                rubric_item_identity(item, base_idx)
                for base_idx, item in enumerate(base_items)
                if isinstance(item, dict)
            }
            missing_items: list[dict[str, Any]] = []
            missing_active_indices: list[int] = []
            sample_cached_scores: dict[int, float] = {}
            has_new_item = False
            prompt_key = (
                "" if prompt_keys is None else str(prompt_keys[sample_idx] or "")
            )
            for active_idx, item in enumerate(active_items):
                if not isinstance(item, dict):
                    continue
                if rubric_item_identity(item, active_idx) in base_identities:
                    reused_old_score_values += 1
                    continue
                has_new_item = True
                cached_score = self.rubric_evolver.current_criterion_score(
                    prompt_key=prompt_key,
                    criterion_id=str(item.get("criterion_id") or ""),
                    sample_index=sample_idx,
                )
                if cached_score is not None:
                    sample_cached_scores[active_idx] = cached_score
                else:
                    missing_items.append(item)
                    missing_active_indices.append(active_idx)
            if not has_new_item:
                continue
            updated_sample_indices.append(sample_idx)
            cached_new_scores[sample_idx] = sample_cached_scores
            if missing_items:
                judge_sample_indices.append(sample_idx)
                evolved_rubrics.append(deepcopy(missing_items))
                new_active_indices[sample_idx] = missing_active_indices

        if not updated_sample_indices:
            return

        evolved_reward_metrics: dict[str, list[Any]] = {}
        if judge_sample_indices:
            evolved_reward_batch = batch.index_select(judge_sample_indices).select(
                batch_keys=["responses", "response_mask"],
                non_tensor_batch_keys=[
                    key for key in batch.non_tensor_batch if key != "multi_modal_data"
                ],
                deepcopy=True,
            )
            evolved_reward_batch.non_tensor_batch["rubric"] = np.array(
                evolved_rubrics, dtype=object
            )
            evolved_reward_batch.non_tensor_batch["rubric_list"] = np.array(
                [None] * len(judge_sample_indices), dtype=object
            )

            with timer("rubric_evolution_reward", timing_raw):
                evolved_reward_tensor, evolved_reward_metrics = ray.get(
                    self.reward_fn.compute_reward.remote(evolved_reward_batch)
                )
            del evolved_reward_tensor
        judge_local_index = {
            sample_idx: local_idx
            for local_idx, sample_idx in enumerate(judge_sample_indices)
        }

        response_lengths = torch.sum(batch.batch["response_mask"], dim=-1)
        reward_delta_values: list[float] = []

        for sample_idx in updated_sample_indices:
            base_items = base_rubric_batch[sample_idx]
            if isinstance(base_items, np.ndarray):
                base_items = base_items.tolist()
            active_items = active_rubric_batch[sample_idx]
            if isinstance(active_items, np.ndarray):
                active_items = active_items.tolist()
            if not isinstance(base_items, list) or not isinstance(active_items, list):
                continue

            old_overall = self._metric_value(
                reward_metric_values, "overall", sample_idx, default=0.0
            )
            base_scores = [
                self._metric_value(
                    reward_metric_values,
                    f"criterion_{base_idx}_score",
                    sample_idx,
                    default=0.0,
                )
                for base_idx in range(len(base_items))
            ]
            base_index_by_identity = {
                rubric_item_identity(item, base_idx): base_idx
                for base_idx, item in enumerate(base_items)
                if isinstance(item, dict)
            }
            local_new_scores = dict(cached_new_scores.get(sample_idx, {}))

            for child_idx, active_idx in enumerate(
                new_active_indices.get(sample_idx, [])
            ):
                local_score = self._metric_value(
                    evolved_reward_metrics,
                    f"criterion_{child_idx}_score",
                    judge_local_index[sample_idx],
                    default=0.0,
                )
                local_new_scores[active_idx] = local_score

            active_scores: list[float] = []
            for active_idx, item in enumerate(active_items):
                if active_idx in local_new_scores:
                    score = local_new_scores[active_idx]
                elif isinstance(item, dict):
                    base_idx = base_index_by_identity.get(
                        rubric_item_identity(item, active_idx)
                    )
                    score = base_scores[base_idx] if base_idx is not None else 0.0
                else:
                    score = 0.0
                active_scores.append(score)

            for active_idx, score in enumerate(active_scores):
                absolute_key = f"criterion_{active_idx}_score"
                self._ensure_metric_values(
                    reward_metric_values, absolute_key, batch_size
                )[sample_idx] = score
            for stale_idx in range(len(active_scores), len(base_items)):
                stale_key = f"criterion_{stale_idx}_score"
                self._ensure_metric_values(reward_metric_values, stale_key, batch_size)[
                    sample_idx
                ] = float("nan")

            self._ensure_metric_values(
                reward_metric_values, "criterion_count", batch_size
            )[sample_idx] = float(len(active_items))
            new_overall = self._compute_weighted_rubric_score(
                active_items, reward_metric_values, sample_idx
            )

            self._ensure_metric_values(reward_metric_values, "overall", batch_size)[
                sample_idx
            ] = new_overall
            self._ensure_metric_values(reward_metric_values, "accuracy", batch_size)[
                sample_idx
            ] = new_overall

            response_end = int(response_lengths[sample_idx].item()) - 1
            if response_end >= 0:
                batch.batch["token_level_scores"][
                    sample_idx, response_end
                ] = new_overall
            reward_delta_values.append(new_overall - old_overall)

        if evolved_reward_metrics:
            metrics.update(
                self._reduce_reward_metrics_for_logging(
                    evolved_reward_metrics,
                    prefix="rubric_evolution/new_",
                )
            )
        metrics["rubric_evolution/immediate_reward_samples"] = float(
            len(updated_sample_indices)
        )
        metrics["rubric_evolution/immediate_fallback_judge_samples"] = float(
            len(judge_sample_indices)
        )
        metrics["rubric_evolution/immediate_old_score_values_reused"] = float(
            reused_old_score_values
        )
        metrics["rubric_evolution/immediate_new_score_values_reused"] = float(
            sum(len(scores) for scores in cached_new_scores.values())
        )
        metrics["rubric_evolution/immediate_reward_delta_mean"] = float(
            np.mean(reward_delta_values)
        )

    def _build_thought_span_mask(
        self, responses: torch.Tensor, response_mask: torch.Tensor
    ) -> torch.Tensor:
        """Build a mask that is 1 only for tokens inside <thought>...</thought> spans."""
        thought_start_ids = self.tokenizer.encode("<thought>", add_special_tokens=False)
        thought_end_ids = self.tokenizer.encode("</thought>", add_special_tokens=False)
        start_len = len(thought_start_ids)
        end_len = len(thought_end_ids)

        batch_size, seq_len = responses.shape
        mask = torch.zeros_like(response_mask)

        for i in range(batch_size):
            tokens = responses[i].tolist()
            in_thought = False
            t = 0
            while t < seq_len:
                if not in_thought:
                    if (
                        t + start_len <= seq_len
                        and tokens[t : t + start_len] == thought_start_ids
                    ):
                        in_thought = True
                        t += start_len
                        continue
                else:
                    if (
                        t + end_len <= seq_len
                        and tokens[t : t + end_len] == thought_end_ids
                    ):
                        in_thought = False
                        t += end_len
                        continue
                    mask[i, t] = 1.0
                t += 1

        return mask * response_mask

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_rubric_items(
        raw_rubric: Any, raw_rubric_list: Any
    ) -> list[dict[str, Any]]:
        if isinstance(raw_rubric, np.ndarray):
            raw_rubric = raw_rubric.tolist()
        if isinstance(raw_rubric, list) and raw_rubric:
            items = []
            for idx, item in enumerate(raw_rubric):
                if isinstance(item, dict):
                    items.append(
                        {
                            "title": str(item.get("title") or f"Criterion {idx}"),
                            "description": str(item.get("description") or ""),
                            "weight": item.get("weight", 1.0),
                        }
                    )
                else:
                    items.append(
                        {
                            "title": f"Criterion {idx}",
                            "description": str(item),
                            "weight": 1.0,
                        }
                    )
            return items

        if isinstance(raw_rubric_list, np.ndarray):
            raw_rubric_list = raw_rubric_list.tolist()
        if isinstance(raw_rubric_list, list):
            return [
                {"title": f"Criterion {idx}", "description": str(item), "weight": 1.0}
                for idx, item in enumerate(raw_rubric_list)
            ]
        return []

    def _criterion_score_matrix(
        self,
        batch: DataProto,
        reward_metric_values: Optional[dict[str, list[Any]]],
        num_criteria: int,
        rubric_counts: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # `presence` is derived from each sample's true rubric length (criterion k
        # exists for sample i iff k < rubric_counts[i]) rather than from list
        # membership, because the reward metric lists are sample-aligned and pad
        # absent criteria with NaN. Scores are read by sample index i directly.
        batch_size = len(batch)
        score_matrix = torch.zeros((batch_size, num_criteria), dtype=torch.float32)
        presence_matrix = torch.zeros((batch_size, num_criteria), dtype=torch.float32)
        for k in range(num_criteria):
            values = (
                None
                if reward_metric_values is None
                else reward_metric_values.get(f"criterion_{k}_score")
            )
            for i in range(batch_size):
                if k >= rubric_counts[i]:
                    continue
                presence_matrix[i, k] = 1.0
                if values is not None and i < len(values):
                    score = self._as_float(values[i], default=float("nan"))
                    if score == score:  # skip NaN (absent value)
                        score_matrix[i, k] = score
        return score_matrix, presence_matrix

    def _criterion_variance_weights(
        self,
        scores: torch.Tensor,
        presence: torch.Tensor,
        uids: np.ndarray,
    ) -> torch.Tensor:
        weights = torch.zeros_like(scores)
        eps = float(self.opsd_config.casd_weight_eps)
        for uid in np.unique(uids):
            group_indices = np.where(uids == uid)[0]
            if len(group_indices) == 0:
                continue
            group_scores = scores[group_indices]
            group_presence = presence[group_indices]
            active = group_presence.sum(dim=0) > 0
            if not torch.any(active):
                continue
            if len(group_indices) > 1:
                variance = (
                    torch.var(group_scores, dim=0, unbiased=False) * active.float()
                )
            else:
                variance = torch.zeros(scores.size(1), dtype=scores.dtype)
            variance_sum = variance.sum()
            if not self.opsd_config.casd_use_variance_weights or variance_sum <= eps:
                group_weights = active.float() / active.float().sum().clamp(min=1.0)
            else:
                group_weights = variance / variance_sum.clamp(min=eps)
            weights[group_indices] = group_weights
        return weights

    @classmethod
    def _format_complete_rubric(cls, rubric_items: list[dict[str, Any]]) -> str:
        requirements = []
        penalties = []
        neutral = []

        for idx, item in enumerate(rubric_items, start=1):
            title = str(item.get("title") or f"Criterion {idx}").strip()
            description = str(item.get("description") or "").strip()
            weight = cls._as_float(item.get("weight", 1.0), default=1.0)
            criterion = f"{idx}. [{title}] {description} (points: {weight:g})"
            if weight > 0:
                requirements.append(criterion)
            elif weight < 0:
                penalties.append(criterion)
            else:
                neutral.append(criterion)

        sections = []
        if requirements:
            sections.append("Requirements to satisfy:\n" + "\n".join(requirements))
        if penalties:
            sections.append(
                "Penalty conditions to avoid:\n"
                "The response should avoid triggering the failures, errors, or "
                "undesired behaviors described below.\n" + "\n".join(penalties)
            )
        if neutral:
            sections.append("Additional evaluation criteria:\n" + "\n".join(neutral))
        return "\n\n".join(sections)

    def _build_rubric_guided_hints(self, batch: DataProto) -> list[list[str]]:
        """Build one complete-rubric privileged hint for each sampled response."""
        rubric_batch = batch.non_tensor_batch.get("rubric")
        rubric_list_batch = batch.non_tensor_batch.get("rubric_list")
        hints: list[list[str]] = []

        for i in range(len(batch)):
            raw_rubric = None if rubric_batch is None else rubric_batch[i]
            raw_rubric_list = (
                None if rubric_list_batch is None else rubric_list_batch[i]
            )
            rubric_items = self._normalize_rubric_items(raw_rubric, raw_rubric_list)
            if not rubric_items:
                raise ValueError(
                    "Rubric-Guided RLSD requires rubric or rubric_list fields "
                    f"for every sample; sample {i} has no rubric."
                )
            hints.append([self._format_complete_rubric(rubric_items)])

        return hints

    def _build_casd_hints(
        self,
        batch: DataProto,
        reward_metric_values: Optional[dict[str, list[Any]]],
    ) -> tuple[list[list[str]], torch.Tensor, int]:
        rubric_batch = batch.non_tensor_batch.get("rubric")
        rubric_list_batch = batch.non_tensor_batch.get("rubric_list")
        rubric_items_per_sample = []
        max_criteria = 0
        for i in range(len(batch)):
            raw_rubric = None if rubric_batch is None else rubric_batch[i]
            raw_rubric_list = (
                None if rubric_list_batch is None else rubric_list_batch[i]
            )
            rubric_items = self._normalize_rubric_items(raw_rubric, raw_rubric_list)
            rubric_items_per_sample.append(rubric_items)
            max_criteria = max(max_criteria, len(rubric_items))

        num_criteria = min(max_criteria, int(self.opsd_config.casd_max_criteria))
        if num_criteria <= 0:
            raise ValueError(
                "CASD requires rubric or rubric_list fields in the training data."
            )

        rubric_counts = [len(items) for items in rubric_items_per_sample]
        score_matrix, presence_matrix = self._criterion_score_matrix(
            batch, reward_metric_values, num_criteria, rubric_counts
        )
        criterion_weights = self._criterion_variance_weights(
            scores=score_matrix,
            presence=presence_matrix,
            uids=batch.non_tensor_batch["uid"],
        )

        hints: list[list[str]] = []
        for i, rubric_items in enumerate(rubric_items_per_sample):
            sample_hints = []
            for k in range(num_criteria):
                if k < len(rubric_items):
                    item = rubric_items[k]
                    sample_hints.append(
                        "Rubric criterion:\n"
                        f"Title: {item['title']}\n"
                        f"Description: {item['description']}\n"
                        f"Weight: {item['weight']}"
                    )
                else:
                    sample_hints.append(
                        "Rubric criterion:\n"
                        f"Title: Criterion {k}\n"
                        "Description: unavailable\n"
                        "Weight: 0"
                    )
            hints.append(sample_hints)
        return hints, criterion_weights, num_criteria

    def _build_dead_criteria_guided_targets(
        self,
        batch: DataProto,
        reward_metric_values: Optional[dict[str, list[Any]]],
    ) -> list[dict[str, Any]]:
        if reward_metric_values is None:
            return []

        uids = batch.non_tensor_batch.get("uid")
        rubric_batch = batch.non_tensor_batch.get("rubric")
        rubric_list_batch = batch.non_tensor_batch.get("rubric_list")
        if uids is None or rubric_batch is None and rubric_list_batch is None:
            return []

        threshold = float(self.opsd_config.dead_criteria_guided_score_threshold)
        max_criteria = int(self.opsd_config.dead_criteria_guided_max_criteria)

        targets: list[dict[str, Any]] = []
        seen: set[str] = set()
        for first_idx, uid_value in enumerate(uids):
            uid = str(uid_value)
            if uid in seen:
                continue
            seen.add(uid)
            group_indices = np.where(uids == uid_value)[0]
            if len(group_indices) == 0:
                continue

            raw_rubric = None if rubric_batch is None else rubric_batch[first_idx]
            raw_rubric_list = (
                None if rubric_list_batch is None else rubric_list_batch[first_idx]
            )
            rubric_items = self._normalize_rubric_items(raw_rubric, raw_rubric_list)
            if not rubric_items:
                continue

            dead_items = []
            for criterion_idx, item in enumerate(rubric_items):
                weight = self._as_float(item.get("weight", 1.0), default=1.0)

                values = reward_metric_values.get(f"criterion_{criterion_idx}_score")
                if values is None:
                    continue

                scores = []
                has_missing_score = False
                for sample_idx in group_indices:
                    if sample_idx >= len(values):
                        has_missing_score = True
                        break
                    score = self._as_float(values[sample_idx], default=float("nan"))
                    if score != score:
                        has_missing_score = True
                        break
                    scores.append(score)

                if has_missing_score or not scores:
                    continue
                if weight > 0:
                    should_guide = all(score <= threshold for score in scores)
                elif weight < 0:
                    # For penalty criteria, score=1 means the undesired condition
                    # is present. Guide only when every rollout triggered it.
                    should_guide = all(score >= 1.0 - threshold for score in scores)
                else:
                    should_guide = False

                if should_guide:
                    dead_items.append(item)

            if max_criteria > 0:
                dead_items = dead_items[:max_criteria]
            if dead_items:
                targets.append(
                    {
                        "source_index": int(first_idx),
                        "source_uid": uid,
                        "dead_items": dead_items,
                        "hint": self._format_complete_rubric(dead_items),
                    }
                )

        max_groups = int(self.opsd_config.dead_criteria_guided_max_groups_per_step)
        max_ratio = float(self.opsd_config.dead_criteria_guided_max_group_ratio)
        if max_ratio > 0:
            max_by_ratio = int(len(seen) * max_ratio)
            max_groups = (
                max_by_ratio if max_groups <= 0 else min(max_groups, max_by_ratio)
            )
            targets = targets[:max_groups]
        elif max_groups > 0:
            targets = targets[:max_groups]
        return targets

    @staticmethod
    def _concat_metric_values(
        left: Optional[dict[str, list[Any]]],
        right: Optional[dict[str, list[Any]]],
        left_size: int,
        right_size: int,
    ) -> Optional[dict[str, list[Any]]]:
        if left is None and right is None:
            return None
        left = left or {}
        right = right or {}
        keys = set(left) | set(right)
        merged: dict[str, list[Any]] = {}
        for key in keys:
            left_values = list(left.get(key, [float("nan")] * left_size))
            right_values = list(right.get(key, [float("nan")] * right_size))
            if len(left_values) < left_size:
                left_values.extend([float("nan")] * (left_size - len(left_values)))
            if len(right_values) < right_size:
                right_values.extend([float("nan")] * (right_size - len(right_values)))
            merged[key] = left_values[:left_size] + right_values[:right_size]
        return merged

    def _limit_guided_targets_for_world_size(
        self, targets: list[dict[str, Any]], metrics: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if not targets:
            return targets
        guided_n = int(self.opsd_config.dead_criteria_guided_rollout_n)
        world_size = int(self.actor_rollout_ref_wg.world_size)
        if world_size <= 1 or (len(targets) * guided_n) % world_size == 0:
            return targets

        keep = len(targets)
        while keep > 0 and (keep * guided_n) % world_size != 0:
            keep -= 1
        skipped = len(targets) - keep
        metrics["dead_criteria_guided/skipped_world_size_alignment_groups"] = float(
            skipped
        )
        return targets[:keep]

    def _maybe_append_dead_criteria_guided_rollouts(
        self,
        batch: DataProto,
        reward_metric_values: Optional[dict[str, list[Any]]],
        metrics: dict[str, Any],
        timing_raw: dict[str, Any],
    ) -> tuple[DataProto, Optional[dict[str, list[Any]]]]:
        if not self.opsd_config.enable_dead_criteria_guided_rollout:
            return batch, reward_metric_values

        targets = self._build_dead_criteria_guided_targets(batch, reward_metric_values)
        metrics["dead_criteria_guided/candidate_groups"] = float(len(targets))
        targets = self._limit_guided_targets_for_world_size(targets, metrics)
        if not targets:
            metrics["dead_criteria_guided/generated_groups"] = 0.0
            return batch, reward_metric_values

        guided_n = int(self.opsd_config.dead_criteria_guided_rollout_n)
        source_indices = [target["source_index"] for target in targets]
        source_batch = batch.index_select(source_indices)
        question_values = self._get_teacher_questions(source_batch)
        hints = [[target["hint"]] for target in targets]
        empty_gt_answers = np.array([""] * len(targets), dtype=object)

        with timer("dead_criteria_guided_prompt", timing_raw):
            guided_gen = construct_teacher_prompts(
                questions=question_values,
                hints=hints,
                gt_answers=empty_gt_answers,
                p_drop=0.0,
                tokenizer=self.tokenizer,
                processor=self.processor,
                template_path=self.opsd_config.dead_criteria_guided_rollout_template,
                template_text=(
                    None
                    if self.opsd_config.dead_criteria_guided_rollout_template
                    else DEAD_CRITERIA_GUIDED_ROLLOUT_TEMPLATE
                ),
                max_prompt_length=self.config.data.max_prompt_length,
                multi_modal_data=source_batch.non_tensor_batch.get("multi_modal_data"),
                image_min_pixels=self.config.data.image_min_pixels,
                image_max_pixels=self.config.data.image_max_pixels,
                video_min_pixels=self.config.data.video_min_pixels,
                video_max_pixels=self.config.data.video_max_pixels,
                video_fps=self.config.data.video_fps,
                video_max_frames=self.config.data.video_max_frames,
                apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
            )
            guided_gen.meta_info.update(
                {
                    "n": guided_n,
                    "temperature": self.config.worker.rollout.temperature,
                    "top_p": self.config.worker.rollout.top_p,
                    "max_tokens": self.config.data.max_response_length,
                    "image_min_pixels": self.config.data.image_min_pixels,
                    "image_max_pixels": self.config.data.image_max_pixels,
                    "video_min_pixels": self.config.data.video_min_pixels,
                    "video_max_pixels": self.config.data.video_max_pixels,
                    "video_fps": self.config.data.video_fps,
                    "video_max_frames": self.config.data.video_max_frames,
                }
            )

        with timer("dead_criteria_guided_rollout", timing_raw):
            self.actor_rollout_ref_wg.prepare_rollout_engine()
            try:
                guided_gen, guided_pad_size = pad_dataproto_to_divisor(
                    guided_gen, self.actor_rollout_ref_wg.world_size
                )
                guided_output = self.actor_rollout_ref_wg.generate_sequences(guided_gen)
                guided_output = unpad_dataproto(
                    guided_output, pad_size=guided_pad_size * guided_n
                )
            finally:
                self.actor_rollout_ref_wg.release_rollout_engine()

        expected_guided_size = len(targets) * guided_n
        if len(guided_output) != expected_guided_size:
            raise RuntimeError(
                "Unexpected dead-criteria guided rollout batch size. "
                f"expected={expected_guided_size}, got={len(guided_output)}."
            )

        guided_non_tensors = {
            key: np.repeat(value, guided_n, axis=0)
            for key, value in source_batch.non_tensor_batch.items()
        }
        guided_group_uids = np.array([str(uuid.uuid4()) for _ in targets], dtype=object)
        guided_non_tensors["uid"] = np.repeat(guided_group_uids, guided_n, axis=0)
        if "multi_modal_data" in guided_output.non_tensor_batch:
            guided_non_tensors["multi_modal_data"] = guided_output.non_tensor_batch[
                "multi_modal_data"
            ]

        guided_batch = DataProto(
            batch=guided_output.batch,
            non_tensor_batch=guided_non_tensors,
            meta_info=batch.meta_info,
        )

        with timer("dead_criteria_guided_old", timing_raw):
            guided_old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(
                guided_batch
            )
            guided_batch = guided_batch.union(guided_old_log_probs)

        if self.use_reference_policy:
            with timer("dead_criteria_guided_ref", timing_raw):
                guided_ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(
                    guided_batch
                )
                guided_batch = guided_batch.union(guided_ref_log_probs)

        with timer("dead_criteria_guided_reward", timing_raw):
            guided_reward_batch = guided_batch.select(
                batch_keys=["responses", "response_mask"],
                non_tensor_batch_keys=[
                    k for k in guided_batch.non_tensor_batch if k != "multi_modal_data"
                ],
            )
            guided_reward_tensor, guided_reward_metrics = ray.get(
                self.reward_fn.compute_reward.remote(guided_reward_batch)
            )
            guided_batch.batch["token_level_scores"] = guided_reward_tensor
            metrics.update(
                self._reduce_reward_metrics_for_logging(
                    guided_reward_metrics,
                    prefix="dead_criteria_guided/reward_",
                )
            )

        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
            guided_batch, guided_kl_metrics = apply_kl_penalty(
                guided_batch, self.kl_ctrl, self.config.algorithm.kl_penalty
            )
            metrics.update(
                {f"dead_criteria_guided/{k}": v for k, v in guided_kl_metrics.items()}
            )
        else:
            guided_batch.batch["token_level_rewards"] = guided_batch.batch[
                "token_level_scores"
            ]

        guided_batch = compute_advantage(
            guided_batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
        )

        combined_reward_metric_values = self._concat_metric_values(
            reward_metric_values,
            guided_reward_metrics,
            len(batch),
            len(guided_batch),
        )
        combined_batch = DataProto.concat([batch, guided_batch])
        balance_idx = self._balance_batch(
            combined_batch,
            metrics=metrics,
            logging_prefix="dead_criteria_guided_global_seqlen",
        )
        if combined_reward_metric_values is not None:
            combined_reward_metric_values = self._select_metric_values(
                combined_reward_metric_values,
                balance_idx,
                len(combined_batch),
            )

        combined_batch.meta_info["global_token_num"] = torch.sum(
            combined_batch.batch["attention_mask"], dim=-1
        ).tolist()
        metrics["dead_criteria_guided/generated_groups"] = float(len(targets))
        metrics["dead_criteria_guided/generated_rollouts"] = float(len(guided_batch))
        metrics["dead_criteria_guided/dead_criteria_per_group_mean"] = float(
            np.mean([len(target["dead_items"]) for target in targets])
        )
        return combined_batch, combined_reward_metric_values

    @staticmethod
    def _select_metric_values(
        metric_values: dict[str, list[Any]],
        indices: torch.Tensor,
        target_size: int,
    ) -> dict[str, list[Any]]:
        selected: dict[str, list[Any]] = {}
        index_list = indices.detach().cpu().tolist()
        for key, values in metric_values.items():
            trimmed = list(values)[:target_size]
            # Lists are sample-aligned; pad short ones with NaN so the reordering by
            # `indices` is always applied and never silently skipped (which would
            # leave the metric misaligned with the reordered batch).
            if len(trimmed) < target_size:
                trimmed = trimmed + [float("nan")] * (target_size - len(trimmed))
            selected[key] = [trimmed[idx] for idx in index_list]
        return selected

    def _augment_batch_with_grpo_opsd(
        self,
        batch: DataProto,
        metrics: dict[str, Any],
        timing_raw: dict[str, Any],
        reward_metric_values: Optional[dict[str, list[Any]]] = None,
    ) -> DataProto:
        """Attach teacher on-policy tensors aligned with the current GRPO batch."""
        with timer("grpo_opsd_teacher_prompt", timing_raw):
            if self.opsd_config.enable_casd:
                hints, criterion_weights, num_criteria = self._build_casd_hints(
                    batch, reward_metric_values
                )
                empty_gt_answers = np.array([""] * len(batch), dtype=object)
                template_path = (
                    self.opsd_config.casd_teacher_template
                    or self._teacher_template_path
                )
            elif self.opsd_config.enable_rubric_guided_rlsd:
                hints = self._build_rubric_guided_hints(batch)
                criterion_weights = None
                num_criteria = 1
                template_path = (
                    self.opsd_config.rubric_guided_teacher_template
                    or self._teacher_template_path
                )
            else:
                hints = self._get_self_distill_hints(batch)
                criterion_weights = None
                num_criteria = 1
                template_path = self._teacher_template_path
            empty_gt_answers = np.array([""] * len(batch), dtype=object)
            teacher_gen = construct_teacher_prompts(
                questions=self._get_teacher_questions(batch),
                hints=hints,
                gt_answers=empty_gt_answers,
                p_drop=0.0,
                tokenizer=self.tokenizer,
                processor=self.processor,
                template_path=template_path,
                max_prompt_length=(self.opsd_config.rubric_generator_teacher_max_prompt_length
                    if self.opsd_config.enable_rubric_generator_training else self.config.data.max_prompt_length),
                multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                image_min_pixels=self.config.data.image_min_pixels,
                image_max_pixels=self.config.data.image_max_pixels,
                video_min_pixels=self.config.data.video_min_pixels,
                video_max_pixels=self.config.data.video_max_pixels,
                video_fps=self.config.data.video_fps,
                video_max_frames=self.config.data.video_max_frames,
                apply_chat_template_kwargs=self.config.data.apply_chat_template_kwargs,
            )

            if self.opsd_config.enable_rubric_generator_training:
                lengths = [len(self.tokenizer.encode(str(text), add_special_tokens=False))
                    for text in teacher_gen.non_tensor_batch["raw_prompt"]]
                if max(lengths, default=0) > self.opsd_config.rubric_generator_teacher_max_prompt_length:
                    raise ValueError("Complete final rubric exceeds RG Teacher prompt budget; refusing silent truncation.")

        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        if self.opsd_config.enable_casd:
            responses_for_teacher = responses.repeat_interleave(num_criteria, dim=0)
            response_mask_for_teacher = response_mask.repeat_interleave(
                num_criteria, dim=0
            )
        else:
            responses_for_teacher = responses
            response_mask_for_teacher = response_mask

        teacher_input_ids, teacher_attention_mask, teacher_position_ids = (
            _concat_prompt_and_response(
                prompt_ids=teacher_gen.batch["input_ids"],
                prompt_attention_mask=teacher_gen.batch["attention_mask"],
                prompt_position_ids=teacher_gen.batch["position_ids"],
                response_ids=responses_for_teacher,
                response_mask=response_mask_for_teacher,
            )
        )

        if self.opsd_config.enable_casd:
            batch_size = len(batch)
            teacher_input_ids = teacher_input_ids.reshape(batch_size, num_criteria, -1)
            teacher_attention_mask = teacher_attention_mask.reshape(
                batch_size, num_criteria, -1
            )
            if teacher_position_ids.dim() == 3:
                teacher_position_ids = teacher_position_ids.reshape(
                    batch_size,
                    num_criteria,
                    teacher_position_ids.size(1),
                    teacher_position_ids.size(2),
                )
            else:
                teacher_position_ids = teacher_position_ids.reshape(
                    batch_size, num_criteria, -1
                )
            batch.batch["casd_criterion_weights"] = criterion_weights

        batch.batch["grpo_opsd_teacher_input_ids"] = teacher_input_ids
        batch.batch["grpo_opsd_teacher_attention_mask"] = teacher_attention_mask
        batch.batch["grpo_opsd_teacher_position_ids"] = teacher_position_ids

        if self.opsd_config.stgca_thought_only:
            batch.batch["stgca_thought_mask"] = self._build_thought_span_mask(
                batch.batch["responses"], batch.batch["response_mask"]
            )
        warmup_steps = self.opsd_config.stgca_lambda_warmup_steps
        decay_steps = self.opsd_config.stgca_lambda_decay_steps
        target_lambda = self.opsd_config.stgca_lambda
        if warmup_steps > 0 and self.global_step < warmup_steps:
            effective_lambda = target_lambda * (self.global_step / warmup_steps)
        elif decay_steps > 0 and self.global_step >= warmup_steps:
            decay_progress = (self.global_step - warmup_steps) / decay_steps
            effective_lambda = target_lambda * max(1.0 - decay_progress, 0.0)
        else:
            effective_lambda = target_lambda

        batch.meta_info["loss_type"] = "grpo_opsd"
        batch.meta_info["opsd_config"] = {
            "stgca_lambda": effective_lambda,
            "stgca_reweight_clip_range": self.opsd_config.stgca_reweight_clip_range,
            "stgca_negative_only": self.opsd_config.stgca_negative_only,
            "stgca_thought_only": self.opsd_config.stgca_thought_only,
            "stgca_teacher_sync_interval": self.opsd_config.stgca_teacher_sync_interval,
            "global_step": self.global_step,
            "enable_casd": self.opsd_config.enable_casd,
            "casd_num_criteria": num_criteria,
            "casd_teacher_forward_chunk_size": self.opsd_config.casd_teacher_forward_chunk_size,
            "casd_normalize_delta": self.opsd_config.casd_normalize_delta,
        }
        metrics["grpo_opsd/stgca_effective_lambda"] = effective_lambda
        if self.opsd_config.enable_casd:
            metrics["grpo_opsd/casd_num_criteria"] = float(num_criteria)
            metrics["grpo_opsd/casd_use_variance_weights"] = float(
                self.opsd_config.casd_use_variance_weights
            )
            metrics["grpo_opsd/casd_criterion_weight_max"] = float(
                criterion_weights.max().item()
            )
            metrics["grpo_opsd/casd_criterion_weight_min"] = float(
                criterion_weights.min().item()
            )
        elif self.opsd_config.enable_rubric_guided_rlsd:
            metrics["grpo_opsd/rubric_guided_rlsd"] = 1.0
        else:
            metrics["grpo_opsd/teacher_context_from_trajectory"] = float(
                sum(
                    1
                    for idx, hint in enumerate(hints)
                    if hint
                    and str(hint[0]).strip()
                    and str(hint[0]).strip()
                    != str(batch.non_tensor_batch["ground_truth"][idx]).strip()
                )
            )
        return batch

    def fit(self):
        self.logger = self._create_logger()
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                reward_futures = [] if self.config.algorithm.pipeline_reward else None
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(
                        metrics=metrics, reward_futures=reward_futures
                    )
                    self.actor_rollout_ref_wg.release_rollout_engine()

                global_idx = self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = torch.sum(
                    batch.batch["attention_mask"], dim=-1
                ).tolist()

                if self.rubric_initialization_store is not None:
                    with timer("rubric_initialization_apply", timing_raw):
                        # Initialization sees the fixed Solver rollout group and
                        # replaces the dataset rubric before both reward judging
                        # and persistent hierarchical evolution.
                        metrics.update(
                            self._initialize_rubrics_for_batch(batch, timing_raw)
                        )

                if self.rubric_evolver is not None:
                    with timer("rubric_evolution_apply", timing_raw):
                        # Restore any legacy/delayed candidates before this
                        # step's diagnostic Judge pass. The configured joint
                        # pipeline activates newly verified criteria later in
                        # this same step and scores only those additions.
                        metrics.update(
                            self.rubric_evolver.activate_pending(self.global_step)
                        )
                        self.rubric_evolver.apply_active_rubric_to_batch(batch)

                if "token_level_scores" not in batch.batch and not (
                    reward_futures and len(reward_futures) > 0
                ):
                    with timer("reward", timing_raw):
                        reward_batch = batch.select(
                            batch_keys=["responses", "response_mask"],
                            non_tensor_batch_keys=[
                                k
                                for k in batch.non_tensor_batch
                                if k != "multi_modal_data"
                            ],
                        )
                        reward_ref = self.reward_fn.compute_reward.remote(reward_batch)

                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(
                            batch
                        )
                        batch = batch.union(ref_log_probs)

                reward_metric_values = None
                with timer("adv", timing_raw):
                    if reward_futures and len(reward_futures) > 0:
                        with timer("reward", timing_raw):
                            all_reward_tensors = []
                            all_reward_metrics: dict[str, list[Any]] = {}
                            running = 0
                            for ref, _mini_size in reward_futures:
                                reward_tensor_i, reward_metrics_i = ray.get(ref)
                                all_reward_tensors.append(reward_tensor_i)
                                mb = reward_tensor_i.shape[0]
                                # Concatenate while preserving sample alignment: keys
                                # absent from a given mini-batch (e.g. higher
                                # `criterion_{k}_score`) are back-filled with NaN so
                                # every list stays aligned to the global sample index.
                                for k, v in reward_metrics_i.items():
                                    if k not in all_reward_metrics:
                                        all_reward_metrics[k] = [float("nan")] * running
                                    if len(all_reward_metrics[k]) < running:
                                        all_reward_metrics[k].extend(
                                            [float("nan")]
                                            * (running - len(all_reward_metrics[k]))
                                        )
                                    all_reward_metrics[k].extend(v)
                                running += mb
                            for k in all_reward_metrics:
                                if len(all_reward_metrics[k]) < running:
                                    all_reward_metrics[k].extend(
                                        [float("nan")]
                                        * (running - len(all_reward_metrics[k]))
                                    )
                            reward_tensor_all = torch.cat(all_reward_tensors, dim=0)
                            target_size = (
                                self.config.data.rollout_batch_size
                                * self.config.worker.rollout.n
                            )
                            reward_tensor_all = reward_tensor_all[:target_size]
                            reward_tensor_all = reward_tensor_all[global_idx]
                            batch.batch["token_level_scores"] = reward_tensor_all
                            metrics.update(
                                self._reduce_reward_metrics_for_logging(
                                    all_reward_metrics,
                                    prefix="reward/",
                                )
                            )
                            reward_metric_values = self._select_metric_values(
                                all_reward_metrics,
                                global_idx,
                                target_size,
                            )
                    elif "token_level_scores" not in batch.batch:
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        metrics.update(
                            self._reduce_reward_metrics_for_logging(
                                reward_metrics,
                                prefix="reward/",
                            )
                        )
                        reward_metric_values = reward_metrics

                    if self.rubric_generator_training is not None:
                        self.rubric_generator_training.begin_step()

                    if self.rubric_evolver is not None:
                        with timer("rubric_evolution_observe", timing_raw):
                            metrics.update(
                                self.rubric_evolver.observe(
                                    batch,
                                    reward_metric_values,
                                    self.global_step,
                                )
                            )
                        if (
                            self.global_step
                            % self.opsd_config.rubric_evolution_interval
                            == 0
                        ):
                            activate_immediately = (
                                int(self.opsd_config.rubric_evolution_delayed_steps)
                                == 0
                            )
                            base_rubric_batch = (
                                deepcopy(batch.non_tensor_batch.get("rubric"))
                                if activate_immediately
                                else None
                            )
                            # Horizontal expansion runs first on the same fixed
                            # rollout group. It appends new roots but leaves the
                            # diagnostic scores of all existing criteria intact.
                            gap_metrics = (
                                self._run_rubric_gap_expansion(batch, timing_raw)
                                if activate_immediately
                                else {}
                            )
                            metrics.update(gap_metrics)

                            # Vertical DECOMPOSE/REFINE uses the criterion states
                            # diagnosed by the first judge pass. New gap roots are
                            # appended, so existing parent indices stay stable.
                            evolution_metrics = self._run_rubric_evolution(
                                batch,
                                metrics,
                                timing_raw,
                                activate_immediately=activate_immediately,
                            )
                            metrics.update(evolution_metrics)
                            stage_metrics = {**gap_metrics, **evolution_metrics}
                            accepted_updates = float(
                                stage_metrics.get("rubric_gap/accepted_roots", 0.0)
                            ) + float(
                                stage_metrics.get(
                                    "rubric_evolution/accepted_candidates", 0.0
                                )
                            )
                            if (
                                activate_immediately
                                and base_rubric_batch is not None
                                and reward_metric_values is not None
                                and accepted_updates > 0.0
                            ):
                                # Reuse every first-pass criterion score. Judge
                                # only newly accepted roots/children, then
                                # recompute the final weighted overall reward.
                                self.rubric_evolver.apply_active_rubric_to_batch(batch)
                                self._merge_immediate_evolved_rewards(
                                    batch,
                                    base_rubric_batch=base_rubric_batch,
                                    reward_metric_values=reward_metric_values,
                                    metrics=metrics,
                                    timing_raw=timing_raw,
                                )
                                metrics.update(
                                    self._reduce_reward_metrics_for_logging(
                                        reward_metric_values,
                                        prefix="reward/",
                                    )
                                )
                                # Refresh persistent pass rates from the merged
                                # final score matrix so new roots become eligible
                                # for vertical evolution on later steps.
                                with timer(
                                    "rubric_evolution_observe_final",
                                    timing_raw,
                                ):
                                    metrics.update(
                                        self.rubric_evolver.observe(
                                            batch,
                                            reward_metric_values,
                                            self.global_step,
                                        )
                                    )
                            self._log_rubric_evolution_wandb_table(self.global_step)

                    if (
                        not self.config.algorithm.use_kl_loss
                        and self.use_reference_policy
                    ):
                        batch, kl_metrics = apply_kl_penalty(
                            batch, self.kl_ctrl, self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch[
                            "token_level_scores"
                        ]

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                batch, reward_metric_values = (
                    self._maybe_append_dead_criteria_guided_rollouts(
                        batch,
                        reward_metric_values,
                        metrics,
                        timing_raw,
                    )
                )

                batch = self._augment_batch_with_grpo_opsd(
                    batch, metrics, timing_raw, reward_metric_values
                )

                update_batch = batch
                if self.rubric_generator_training is not None:
                    with timer("rg_behavior_log_probs", timing_raw):
                        rg_batches = [attach_behavior_log_probs(rollouts, self.actor_rollout_ref_wg,
                            use_reference=self.use_reference_policy)
                            for rollouts in self.rubric_generator_training.rollouts]
                    update_batch = build_joint_update_batch(batch, rg_batches,
                        world_size=self.actor_rollout_ref_wg.world_size,
                        pad_token_id=self.tokenizer.pad_token_id,
                        rg_loss_coef=self.opsd_config.rubric_generator_loss_coef,
                        gap_weight=self.opsd_config.rubric_generator_gap_loss_weight,
                        vertical_weight=self.opsd_config.rubric_generator_vertical_loss_weight)
                    self.rubric_generator_training.write_diagnostics(self.global_step)

                with timer("update_actor", timing_raw):
                    actor_output = self.actor_rollout_ref_wg.update_actor(update_batch)
                actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                metrics.update(actor_metrics)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)

                if should_save_checkpoint(
                    self.global_step,
                    save_freq=self.config.trainer.save_freq,
                    save_steps=self.config.trainer.save_steps,
                ):
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=False))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(
                compute_throughout_metrics(
                    batch=batch, timing_raw=timing_raw, num_gpus=num_gpus
                )
            )

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

        if not should_save_checkpoint(
            self.global_step,
            save_freq=self.config.trainer.save_freq,
            save_steps=self.config.trainer.save_steps,
        ):
            self._save_checkpoint()
