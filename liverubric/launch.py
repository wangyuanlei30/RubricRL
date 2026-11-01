"""Explicit local paths, immutable run configuration, and checked resume."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import yaml

from .checkpoints import atomic_json, resolve_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/liverubric_4gpu.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_identity() -> dict[str, str]:
    paths = [ROOT / "train.py"]
    for directory in ("liverubric", "verl", "rubric_generation", "rewards", "prompts"):
        paths.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".jinja", ".txt", ".json"})
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def validate_settings(config: dict) -> None:
    o, t = config["opsd"], config["trainer"]
    if o["enable_rubric_initialization"]:
        raise ValueError("Use offline initialization; online initialization changes the base model")
    for key in ("enable_rubric_generator_training", "enable_rubric_evolution",
                "enable_rubric_gap_expansion", "enable_rubric_vertical_evolution",
                "enable_rubric_guided_rlsd", "freeze_teacher_model"):
        if not o[key]:
            raise ValueError(f"This joint-training entry requires opsd.{key}=true")
    if t["save_model_only"] or t["nnodes"] != 1:
        raise ValueError("This launcher supports resumable, single-node distributed runs")
    if config["worker"]["rollout"]["n"] < 2 * o["rubric_evolution_min_split_count"]:
        raise ValueError("Solver rollout group is too small for minimum split support")
    if o["rubric_generator_rollout_n"] < 2:
        raise ValueError("Generator GRPO needs at least two proposals per group")
    if not o["rubric_evolution_equal_parent_child_weights"]:
        raise ValueError("Enable conserved parent/child family weights for this release")
    env = config["worker"]["reward"]["reward_function_kwargs"]["api_key_env"]
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", env):
        raise ValueError("api_key_env must name an environment variable, never contain a key")


def resolved_config(args: argparse.Namespace, init_dir: Path) -> dict:
    config = yaml.safe_load(args.config.read_text())
    config["data"].update(train_files=str(init_dir / "train.jsonl"),
        val_files=str(args.val_file) if args.val_file else "")
    config["worker"]["actor"]["model"]["model_path"] = str(args.model)
    config["trainer"].update(n_gpus_per_node=args.gpus,
        enable_validation=args.val_file is not None,
        save_checkpoint_path=str(args.output / "checkpoints"),
        find_last_checkpoint=False, load_checkpoint_path=None)
    config["opsd"]["rubric_generator_log_dir"] = str(args.output / "rubric_logs")
    # Original templates are copied byte-for-byte; only their filesystem paths change.
    for section in (config["data"], config["opsd"]):
        for key, value in section.items():
            if isinstance(value, str) and value.startswith("prompts/"):
                section[key] = str(ROOT / value)
    config["opsd"]["rubric_initialization_prompt_dir"] = str(ROOT / "prompts")
    config["worker"]["reward"]["reward_function"] = str(ROOT / "rewards/rubric_llm_judge.py") + ":compute_score"
    validate_settings(config)
    return config


def make_plan(args: argparse.Namespace) -> dict:
    if args.resume:
        if any((args.model, args.train_file, args.val_file, args.init_dir)) or args.config != DEFAULT_CONFIG:
            raise ValueError("Resume uses the saved configuration; pass only --output and --resume")
        metadata = json.loads((args.output / "launch.json").read_text())
        saved_config = args.output / "resolved_config.json"
        if sha256(saved_config) != metadata["config_sha256"]:
            raise ValueError("Saved run configuration changed")
        if code_identity() != metadata["code_sha256"]:
            raise ValueError("Training code or prompts changed since this run started")
        config = json.loads(saved_config.read_text())
        validate_settings(config)
        checkpoint = resolve_checkpoint(args.output / "checkpoints", args.resume,
            world_size=config["trainer"]["n_gpus_per_node"])
        if checkpoint.parent != (args.output / "checkpoints").resolve():
            raise ValueError("Resume must use a checkpoint belonging to this --output run")
        command = [sys.executable, "-m", "verl.trainer.opsd_main",
            f"config={saved_config}", f"trainer.load_checkpoint_path={checkpoint}"]
        return dict(config=config, init_command=None, train_command=command,
            metadata=metadata, resume=str(checkpoint))
    if not args.model or not args.train_file:
        raise ValueError("A new run needs --model and --train-file")
    if args.gpus < 1 or args.init_tp < 1 or args.gpus % args.init_tp:
        raise ValueError("--init-tp must be a positive divisor of --gpus")
    init_dir = args.init_dir or args.output / "initialization"
    config = resolved_config(args, init_dir)
    metadata = dict(version=1, model=str(args.model), source=str(args.train_file),
        val_file=str(args.val_file) if args.val_file else None,
        initialization=str(init_dir), init_tp=args.init_tp, code_sha256=code_identity())
    init_command = [sys.executable, "-m", "rubric_generation.offline_question_rubrics",
        "--source", str(args.train_file), "--model", str(args.model),
        "--prompt-dir", str(ROOT / "prompts"), "--output", str(init_dir),
        "--tp", str(args.init_tp)]
    train_command = [sys.executable, "-m", "verl.trainer.opsd_main",
        f"config={args.output / 'resolved_config.json'}"]
    return dict(config=config, init_command=init_command,
        train_command=train_command, metadata=metadata, resume=None)


def validate_source(path: Path, *, batch_size: int) -> str:
    count = 0
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, (str, list)) or not prompt or "answer" not in row:
                raise ValueError(f"{path}:{line_number}: expected prompt and answer fields")
            if isinstance(prompt, list) and any(not isinstance(m, dict) or
                    not isinstance(m.get("content"), str) or "role" not in m for m in prompt):
                raise ValueError(f"{path}:{line_number}: malformed conversation")
            count += 1
    if count < batch_size:
        raise ValueError(f"Need at least {batch_size} questions; the training loader drops incomplete batches")
    return sha256(path)


def validate_runtime(config: dict) -> None:
    demonstrations = ROOT / "prompts/healthbench_example_level_fewshot.json"
    if not demonstrations.is_file():
        raise RuntimeError("Prepare local HealthBench demonstrations first; see prompts/README.md")
    if sys.platform != "linux":
        raise RuntimeError("Distributed training requires Linux with CUDA; use --dry-run to inspect locally")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < config["trainer"]["n_gpus_per_node"]:
        raise RuntimeError("Not enough visible CUDA GPUs; check CUDA_VISIBLE_DEVICES")
    for name, expected in (("vllm", "0.11.0"), ("transformers", "4.57.3")):
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"Expected {name}=={expected}; use the supplied requirements")
    # The real dataclass schema and post_init checks run BEFORE expensive initialization.
    from omegaconf import OmegaConf
    from verl.trainer.config import PPOConfig
    full = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(PPOConfig()), config))
    full.deep_post_init()
    env = config["worker"]["reward"]["reward_function_kwargs"]["api_key_env"]
    if not os.environ.get(env):
        raise RuntimeError(f"Set the judge credential environment variable {env}")


def execute(plan: dict, args: argparse.Namespace) -> None:
    config, metadata = plan["config"], deepcopy(plan["metadata"])
    validate_runtime(config)
    source_hash = validate_source(Path(metadata["source"]),
        batch_size=config["data"]["rollout_batch_size"])
    val_hash = sha256(Path(metadata["val_file"])) if metadata["val_file"] else None
    if plan["resume"]:
        if metadata["source_sha256"] != source_hash or metadata["val_sha256"] != val_hash:
            raise ValueError("Training or validation data changed since the run started")
    else:
        metadata.update(source_sha256=source_hash, val_sha256=val_hash)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not plan["resume"]:
            if list((args.output / "checkpoints").glob("global_step_*")):
                raise ValueError("Run already contains checkpoints; use --resume or a new --output")
            config_path = args.output / "resolved_config.json"
            # A pre-training interruption can reuse the same initialization cache.
            if (args.output / "launch.json").exists():
                old = json.loads((args.output / "launch.json").read_text())
                metadata["config_sha256"] = sha256(config_path)
                if old != metadata or json.loads(config_path.read_text()) != config:
                    raise ValueError("Existing run has different inputs or settings; use a new output")
            else:
                atomic_json(config_path, config)
                metadata["config_sha256"] = sha256(config_path)
                atomic_json(args.output / "launch.json", metadata)
            subprocess.run(plan["init_command"], cwd=ROOT, check=True)
        from rubric_generation.prepared_rubric_dataset import validate_prepared_dataset
        validate_prepared_dataset(Path(metadata["initialization"]),
            source=Path(metadata["source"]), model=Path(metadata["model"]),
            prompt_dir=ROOT / "prompts", tensor_parallel_size=metadata["init_tp"])
        subprocess.run(plan["train_command"], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Local Hugging Face base model with safetensors")
    parser.add_argument("--train-file", type=Path, help="Question JSONL; initial rubrics are generated offline")
    parser.add_argument("--val-file", type=Path, help="Optional JSONL with evaluation rubrics")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--init-tp", type=int, default=4)
    parser.add_argument("--init-dir", type=Path, help="Reuse an identity-checked initialization cache")
    parser.add_argument("--resume", help="latest or an explicit global_step_<N> directory")
    parser.add_argument("--dry-run", action="store_true", help="Print config and commands; no writes or model/API calls")
    args = parser.parse_args()
    for name in ("model", "train_file", "val_file", "output", "config", "init_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    try:
        plan = make_plan(args)
        if args.dry_run:
            print(yaml.safe_dump(plan["config"], sort_keys=False))
            if plan["init_command"]:
                print("INITIALIZE:", shlex.join(plan["init_command"]))
            print("TRAIN:", shlex.join(plan["train_command"]))
            print("Dry run: paths and execution plan only; no GPU, API, or training validation performed.")
        else:
            execute(plan, args)
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"LiveRubric: {exc}\n")
