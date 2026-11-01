"""Publish complete distributed checkpoints and validate them before resume."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MANIFEST = "liverubric_checkpoint.json"


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def required_files(world_size: int, *, teacher: bool, initialization: bool,
                   evolution: bool) -> list[str]:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    files = ["dataloader.pt"]
    if initialization:
        files.append("rubric_initialization_state.json")
    if evolution:
        files.append("rubric_evolution_state.json")
    for rank in range(world_size):
        for kind in ("model", "optim", "extra_state"):
            files.append(f"actor/{kind}_world_size_{world_size}_rank_{rank}.pt")
        if teacher:
            files.append(f"actor/rubric_teacher/model_world_size_{world_size}_rank_{rank}.pt")
    return files


def publish_checkpoint(directory: Path, *, world_size: int, teacher: bool,
                       initialization: bool, evolution: bool) -> None:
    """Called after worker barriers and rubric writes, before the latest tracker."""
    modes = dict(teacher=teacher, initialization=initialization, evolution=evolution)
    files = {}
    for name in required_files(world_size, **modes):
        path = directory / name
        size = path.stat().st_size
        if size == 0:
            raise ValueError(f"Empty checkpoint state: {path}")
        files[name] = {"size": size}
        if name.endswith(".json"):
            files[name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(directory / MANIFEST, dict(version=1, world_size=world_size,
        **modes, files=files))


def validate_checkpoint(directory: Path, *, world_size: int, teacher: bool = True,
                        initialization: bool = False, evolution: bool = True) -> dict:
    directory = Path(directory)
    if not (directory / MANIFEST).is_file():
        raise ValueError(f"No completed LiveRubric checkpoint at {directory}; partial or legacy checkpoint")
    payload = json.loads((directory / MANIFEST).read_text())
    modes = dict(teacher=teacher, initialization=initialization, evolution=evolution)
    expected = dict(version=1, world_size=world_size, **modes)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint version, world size, or enabled state differs from this run")
    names = required_files(world_size, **modes)
    if set(payload.get("files", {})) != set(names):
        raise ValueError("Incomplete checkpoint manifest")
    for name in names:
        path = directory / name
        record = payload["files"][name]
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError(f"Missing or truncated checkpoint state: {path}")
        if "sha256" in record and hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Changed rubric checkpoint: {path}")
    return payload


def resolve_checkpoint(root: Path, value: str, *, world_size: int) -> Path:
    if value == "latest":
        tracker = root / "checkpoint_tracker.json"
        if not tracker.is_file():
            raise ValueError(f"No published latest checkpoint: {tracker}")
        step = json.loads(tracker.read_text())["last_global_step"]
        path = root / f"global_step_{int(step)}"
    else:
        path = Path(value).expanduser().resolve()
    if not path.name.startswith("global_step_") or not path.name[12:].isdigit():
        raise ValueError("Resume path must end with global_step_<integer>")
    validate_checkpoint(path, world_size=world_size)
    return path.resolve()
