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

import importlib.util
import os
import sys
from functools import partial
from typing import Any, Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict, total=False):
    response: str
    response_length: int
    ground_truth: str
    data_type: Optional[str]
    problem_type: Optional[str]
    problem: Optional[str]
    problem_id: Optional[int]
    prompt_text: Optional[str]
    teacher_question: Optional[str]
    question_source: Optional[str]
    rubric: Optional[Any]
    rubric_list: Optional[Any]
    rubric_count: Optional[int]
    reward_eval_mode: Optional[str]


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


def _get_non_tensor_value(non_tensor: dict[str, Any], key: str, index: int, default: Any = None) -> Any:
    if key not in non_tensor:
        return default
    value = non_tensor[key][index]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _build_reward_context(non_tensor: dict[str, Any], index: int) -> dict[str, Any]:
    problem = _get_non_tensor_value(non_tensor, "problem_reserved_text", index)
    if problem is None:
        problem = _get_non_tensor_value(non_tensor, "teacher_question", index)
    if problem is None:
        problem = _get_non_tensor_value(non_tensor, "prompt_text", index)

    return {
        "data_type": _get_non_tensor_value(non_tensor, "data_type", index),
        "problem_type": _get_non_tensor_value(non_tensor, "problem_type", index),
        "problem": problem,
        "problem_id": _get_non_tensor_value(non_tensor, "problem_id", index),
        "prompt_text": _get_non_tensor_value(non_tensor, "prompt_text", index),
        "teacher_question": _get_non_tensor_value(non_tensor, "teacher_question", index),
        "question_source": _get_non_tensor_value(non_tensor, "question_source", index),
        "rubric": _get_non_tensor_value(non_tensor, "rubric", index),
        "rubric_list": _get_non_tensor_value(non_tensor, "rubric_list", index),
        "rubric_count": _get_non_tensor_value(non_tensor, "rubric_count", index),
        "reward_eval_mode": _get_non_tensor_value(non_tensor, "reward_eval_mode", index),
    }


class SequentialFunctionRewardManagerMixin:
    reward_fn: SequentialRewardFunction

    def compute_reward_sequential(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        batch_size = len(data)
        # Keep every metric list aligned to the sample index. Some keys (e.g. the
        # per-criterion rubric scores `criterion_{k}_score`) are only emitted for a
        # subset of samples (those whose rubric is long enough). Appending only when
        # present would compact those lists and silently break sample alignment for
        # any consumer that indexes by sample (e.g. CASD criterion weights). Missing
        # entries are filled with NaN so list[i] always maps to sample i.
        reward_metrics: dict[str, list[float]] = {}
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(batch_size):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    **_build_reward_context(data.non_tensor_batch, i),
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                if key not in reward_metrics:
                    reward_metrics[key] = [float("nan")] * batch_size
                reward_metrics[key][i] = value

        return reward_tensor, reward_metrics


class BatchFunctionRewardManagerMixin:
    reward_fn: BatchRewardFunction

    def compute_reward_batch(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            non_tensor = data.non_tensor_batch
            reward_inputs.append(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    **_build_reward_context(non_tensor, i),
                }
            )

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        batch_size = len(data)
        # See compute_reward_sequential: keep every metric list aligned to the sample
        # index, filling absent per-sample keys (e.g. higher `criterion_{k}_score`
        # for short-rubric samples) with NaN so list[i] always maps to sample i.
        reward_metrics: dict[str, list[float]] = {}
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                if key not in reward_metrics:
                    reward_metrics[key] = [float("nan")] * batch_size
                reward_metrics[key][i] = value

        return reward_tensor, reward_metrics


class AutoRewardManager(BatchFunctionRewardManagerMixin, SequentialFunctionRewardManagerMixin):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        reward_name = getattr(module, "REWARD_NAME", "unknown")
        reward_type = getattr(module, "REWARD_TYPE", "batch")
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        print(f"Reward name: {reward_name}, reward type: {reward_type}.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.reward_type = reward_type
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        if self.reward_type == "batch":
            return self.compute_reward_batch(data)
        elif self.reward_type == "sequential":
            return self.compute_reward_sequential(data)
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}.")
