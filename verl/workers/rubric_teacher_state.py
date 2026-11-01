"""Independent periodically refreshed Teacher state; never overwrite the KL reference."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init


def _clear_teacher_full_parameter_cache(teacher: torch.nn.Module) -> None:
    """Called only outside forwards/backwards on the frozen Teacher.

    FSDP can retain the root's all-gathered weights after a no-grad forward.
    Copying only local shards then leaves that full tensor stale, including for
    state_dict(). Explicitly reshard/free it before refresh/save/load. This uses
    the same pinned FSDP handle API as the worker's reference-log-prob path.
    """
    if isinstance(teacher, FSDP):
        _lazy_init(teacher, teacher)
        torch.cuda.synchronize()
        for handle in teacher._all_handles:
            handle.reshard(handle.uses_sharded_strategy)


def sync_teacher_parameters(teacher: torch.nn.Module, actor: torch.nn.Module) -> None:
    if teacher is None or teacher is actor:
        raise ValueError("Periodic refresh requires an independent Teacher module.")
    _clear_teacher_full_parameter_cache(teacher)
    source = dict(actor.named_parameters())
    target = dict(teacher.named_parameters())
    if source.keys() != target.keys() or any(source[key].shape != target[key].shape for key in source):
        raise ValueError("Actor/Teacher parameter shards do not match; refusing partial zip-based sync.")
    with torch.no_grad():
        for key, parameter in target.items():
            parameter.copy_(source[key])
        # Buffers that change shape at runtime (e.g. cached rotary embeddings)
        # are not model parameters; only copy matching registered buffers.
        source_buffers = dict(actor.named_buffers())
        for key, buffer in teacher.named_buffers():
            if key in source_buffers and source_buffers[key].shape == buffer.shape:
                buffer.copy_(source_buffers[key])


def _state_path(path: str) -> Path:
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    return Path(path) / "rubric_teacher" / f"model_world_size_{world}_rank_{rank}.pt"


def save_teacher_state(teacher: torch.nn.Module, path: str) -> None:
    _clear_teacher_full_parameter_cache(teacher)
    destination = _state_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = get_model_state_dict(teacher, options=StateDictOptions(cpu_offload=True))
    torch.save(state, destination)


def load_teacher_state(teacher: torch.nn.Module, path: str) -> None:
    source = _state_path(path)
    if not source.exists():
        raise FileNotFoundError(f"Missing independent Teacher checkpoint: {source}. Start a fresh joint run or resume a joint checkpoint; do not silently reset Teacher weights.")
    state = torch.load(source, map_location="cpu", weights_only=False)
    _clear_teacher_full_parameter_cache(teacher)
    set_model_state_dict(teacher, state, options=StateDictOptions(cpu_offload=True))
    _clear_teacher_full_parameter_cache(teacher)
