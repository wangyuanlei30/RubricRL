"""Unified initial-rubric generation for reference and rollout bootstrapping.

This module deliberately contains no trainer or model-serving dependencies.  It
owns prompt construction, output parsing, deterministic root-node identities,
reference-answer lookup, and checkpointable initialization state.  The trainer
only has to provide model outputs for the rendered prompts.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from jinja2 import Environment, StrictUndefined


REFERENCE_GROUNDED = "reference_grounded"
QUESTION_ONLY = "question_only"
EVORUBRIC_BOOTSTRAP = "evorubric_bootstrap"
INITIAL_RUBRIC_MODES = (REFERENCE_GROUNDED, QUESTION_ONLY, EVORUBRIC_BOOTSTRAP)

_DEFAULT_LIMITS = {
    REFERENCE_GROUNDED: (7, 20),
    QUESTION_ONLY: (7, 20),
    EVORUBRIC_BOOTSTRAP: (1, 12),
}


def validate_initialization_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized not in INITIAL_RUBRIC_MODES:
        raise ValueError(
            f"Unknown rubric initialization mode {mode!r}; expected one of "
            f"{', '.join(INITIAL_RUBRIC_MODES)}."
        )
    return normalized


def resolve_criteria_limits(
    mode: str,
    *,
    min_criteria: int = 0,
    max_criteria: int = 0,
) -> tuple[int, int]:
    """Resolve zero-valued limits to mode-specific defaults."""
    mode = validate_initialization_mode(mode)
    default_min, default_max = _DEFAULT_LIMITS[mode]
    resolved_min = default_min if int(min_criteria) == 0 else int(min_criteria)
    resolved_max = default_max if int(max_criteria) == 0 else int(max_criteria)
    if resolved_min < 1:
        raise ValueError(f"min_criteria must be >= 1, got {resolved_min}.")
    if resolved_max < resolved_min:
        raise ValueError(
            f"max_criteria ({resolved_max}) must be >= min_criteria ({resolved_min})."
        )
    return resolved_min, resolved_max


def _conversation_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("prompt_text", "prompt", "messages", "conversation", "question"):
            if key in value:
                return _conversation_to_text(value[key])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        lines: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                role = str(item.get("role") or item.get("from") or "user").strip()
                content = item.get("content", item.get("value", ""))
                lines.append(f"{role}: {_conversation_to_text(content)}")
            else:
                lines.append(_conversation_to_text(item))
        return "\n".join(line for line in lines if line).strip()
    return str(value).strip()


def canonical_question_text(value: Any) -> str:
    return re.sub(r"\s+", " ", _conversation_to_text(value)).strip()


def _question_lookup_keys(value: Any) -> tuple[str, ...]:
    canonical = canonical_question_text(value)
    keys = [canonical]
    if canonical.lower().startswith("user:"):
        keys.append(canonical[5:].strip())
    return tuple(dict.fromkeys(key for key in keys if key))


def make_initialization_key(question: Any, mode: str) -> str:
    payload = f"{validate_initialization_mode(mode)}\n{canonical_question_text(question)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_reference_answer(row: Mapping[str, Any]) -> str:
    for key in (
        "reference_answer",
        "ideal_answer",
        "ideal_completion",
        "reference_response",
    ):
        value = row.get(key)
        if value is not None and str(value).strip():
            return _conversation_to_text(value)
    nested = row.get("ideal_completions_data")
    if isinstance(nested, Mapping):
        value = nested.get("ideal_completion")
        if value is not None and str(value).strip():
            return _conversation_to_text(value)
    return ""


def reference_answer_text(value: Any) -> str:
    """Normalize either a direct answer or a HealthBench-style answer mapping."""
    if isinstance(value, Mapping):
        return _extract_reference_answer(value)
    return _conversation_to_text(value)


class ReferenceAnswerIndex:
    """Question-to-reference lookup loaded from JSON or JSONL data."""

    def __init__(self, source_path: Optional[str] = None):
        self.source_path = str(source_path or "").strip()
        self._answers: dict[str, str] = {}
        if self.source_path:
            self._load(Path(self.source_path))

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Rubric reference source does not exist: {path}")
        text = path.read_text(encoding="utf-8")
        stripped = text.lstrip()
        if stripped.startswith("["):
            rows = json.loads(text)
        else:
            rows = [
                json.loads(line)
                # JSON strings may legally contain Unicode line-separator
                # characters. ``str.splitlines`` would incorrectly split a
                # valid JSONL record at those characters; JSONL records here
                # are delimited specifically by LF.
                for line in text.split("\n")
                if line.strip()
            ]
        if not isinstance(rows, list):
            raise ValueError(f"Reference source must contain a list/JSONL of rows: {path}")

        for row in rows:
            if not isinstance(row, Mapping):
                continue
            question = None
            for key in ("prompt_text", "prompt", "messages", "conversation", "question"):
                if key in row:
                    question = row[key]
                    break
            answer = _extract_reference_answer(row)
            if question is None or not answer:
                continue
            for key in _question_lookup_keys(question):
                existing = self._answers.get(key)
                if existing is not None and existing != answer:
                    raise ValueError(
                        "Reference source contains conflicting answers for question key "
                        f"{key[:160]!r}."
                    )
                self._answers[key] = answer

    def lookup(self, question: Any) -> str:
        for key in _question_lookup_keys(question):
            answer = self._answers.get(key)
            if answer:
                return answer
        return ""

    def __len__(self) -> int:
        return len(self._answers)


@dataclass(frozen=True)
class InitialRubricRequest:
    question: str
    mode: str
    reference_answer: str = ""
    responses: tuple[str, ...] = ()


@dataclass(frozen=True)
class InitialRubricPrompt:
    mode: str
    messages: tuple[dict[str, str], ...]
    flat_prompt: str


class InitialRubricPromptBuilder:
    """Build either multi-message or flattened prompts from shared assets."""

    def __init__(
        self,
        prompt_dir: Optional[str] = None,
        *,
        few_shot_examples: int = 5,
    ):
        self.prompt_dir = (
            Path(prompt_dir)
            if prompt_dir
            else Path(__file__).resolve().parent / "prompts"
        )
        self.few_shot_examples = int(few_shot_examples)
        if self.few_shot_examples < 0:
            raise ValueError("few_shot_examples must be >= 0.")
        self._environment = Environment(
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        )

    def _read(self, name: str) -> str:
        path = self.prompt_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing initial-rubric prompt asset: {path}")
        return path.read_text(encoding="utf-8").strip()

    def _render(self, name: str, **kwargs: Any) -> str:
        return self._environment.from_string(self._read(name)).render(**kwargs).strip()

    def _reference_few_shot_messages(self, *, question_only: bool = False) -> list[dict[str, str]]:
        if self.few_shot_examples == 0:
            return []
        path = self.prompt_dir / "healthbench_example_level_fewshot.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Few-shot prompt asset must be a JSON list: {path}")
        messages: list[dict[str, str]] = []
        for example in payload[: self.few_shot_examples]:
            example_messages = example.get("messages") if isinstance(example, Mapping) else None
            if not isinstance(example_messages, list):
                raise ValueError(f"Malformed few-shot example in {path}")
            for message in example_messages:
                if not isinstance(message, Mapping):
                    raise ValueError(f"Malformed few-shot message in {path}")
                role = str(message.get("role") or "").strip()
                content = str(message.get("content") or "").strip()
                if role not in {"user", "assistant"} or not content:
                    raise ValueError(f"Malformed few-shot role/content in {path}")
                if question_only and role == "user":
                    if "### Target Conversation\n" not in content or "### Reference Answer\n" not in content:
                        raise ValueError("Unexpected few-shot schema; refusing possible reference leakage.")
                    conversation = content.split("### Target Conversation\n", 1)[1].split("### Reference Answer\n", 1)[0].strip()
                    content = "### Target Conversation\n" + conversation + "\n\nGenerate the example-level grading rubric as a JSON array."
                messages.append({"role": role, "content": content})
        return messages

    @staticmethod
    def _flatten(messages: Sequence[Mapping[str, str]]) -> str:
        blocks: list[str] = []
        last_index = len(messages) - 1
        demonstration = 0
        for index, message in enumerate(messages):
            role = str(message["role"]).upper()
            if index == 0 and role == "SYSTEM":
                label = "SYSTEM INSTRUCTIONS"
            elif index == last_index and role == "USER":
                label = "TARGET REQUEST"
            else:
                if role == "USER":
                    demonstration += 1
                label = f"FEW-SHOT DEMONSTRATION {demonstration} {role}"
            blocks.append(f"### {label}\n{str(message['content']).strip()}")
        blocks.append("### RESPONSE\nReturn only the requested JSON array, with no markdown fence.")
        return "\n\n".join(blocks)

    def build(self, request: InitialRubricRequest) -> InitialRubricPrompt:
        mode = validate_initialization_mode(request.mode)
        question = str(request.question or "").strip()
        if not question:
            raise ValueError("Initial-rubric generation requires a non-empty question.")

        if mode == QUESTION_ONLY:
            messages = [
                {"role": "system", "content": self._read("question_only_initial_rubric_system.txt")},
                *self._reference_few_shot_messages(question_only=True),
                {"role": "user", "content": "### Target Conversation\n" + question +
                 "\n\nGenerate the complete initial rubric for this conversation. Return only the JSON list."},
            ]
        elif mode == REFERENCE_GROUNDED:
            reference_answer = str(request.reference_answer or "").strip()
            if not reference_answer:
                raise ValueError(
                    "reference_grounded initialization requires a non-empty reference answer."
                )
            messages = [
                {
                    "role": "system",
                    "content": self._read(
                        "rar_reference_initial_rubric_generator_system.txt"
                    ),
                },
                *self._reference_few_shot_messages(),
                {
                    "role": "user",
                    "content": self._render(
                        "reference_initial_rubric_user_template.jinja",
                        question=question,
                        reference_answer=reference_answer,
                    ),
                },
            ]
        else:
            responses = [str(item or "").strip() for item in request.responses]
            responses = [item for item in responses if item]
            if not responses:
                raise ValueError(
                    "evorubric_bootstrap initialization requires at least one Solver response."
                )
            messages = [
                {
                    "role": "system",
                    "content": self._read("evorubric_generator_system.txt"),
                },
                {
                    "role": "user",
                    "content": self._render(
                        "evorubric_bootstrap_user_template.jinja",
                        conversation=question,
                        responses=responses,
                    ),
                },
            ]

        return InitialRubricPrompt(
            mode=mode,
            messages=tuple(messages),
            flat_prompt=self._flatten(messages),
        )


def _iter_json_values(text: str) -> Iterable[Any]:
    cleaned = re.sub(r"```(?:json)?", "", str(text or ""), flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        yield value


def _extract_rubric_array(text: str) -> list[Any]:
    for value in _iter_json_values(text):
        if isinstance(value, list):
            return value
        if isinstance(value, Mapping):
            for key in ("rubrics", "criteria", "items"):
                nested = value.get(key)
                if isinstance(nested, list):
                    return nested
    raise ValueError("Initial-rubric generator output contains no JSON rubric array.")


def parse_initial_rubric(
    text: str,
    *,
    question: str,
    mode: str,
    min_criteria: int = 0,
    max_criteria: int = 0,
) -> list[dict[str, Any]]:
    """Parse model output into deterministic depth-0 hierarchical rubric nodes."""
    mode = validate_initialization_mode(mode)
    resolved_min, resolved_max = resolve_criteria_limits(
        mode,
        min_criteria=min_criteria,
        max_criteria=max_criteria,
    )
    raw_items = _extract_rubric_array(text)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            continue
        description = str(
            raw_item.get("criterion")
            or raw_item.get("description")
            or raw_item.get("text")
            or ""
        ).strip()
        if not description:
            continue
        dedup_key = re.sub(r"\s+", " ", description).strip().lower()
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        raw_weight = raw_item.get(
            "points", raw_item.get("weight", raw_item.get("score", 0))
        )
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid rubric weight {raw_weight!r}: {description}") from exc
        if weight == 0 or abs(weight) > 10:
            raise ValueError(
                f"Initial rubric weights must be non-zero and in [-10, 10], got "
                f"{weight} for {description!r}."
            )

        index = len(normalized)
        digest_payload = f"{mode}\n{canonical_question_text(question)}\n{description}\n{weight}"
        digest = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()[:12]
        title = str(
            raw_item.get("title") or f"Initial criterion {index + 1} (example)"
        ).strip()
        normalized.append(
            {
                "title": title,
                "description": description,
                "weight": weight,
                "criterion_id": f"root_{digest}_{index:02d}",
                "parent_id": "",
                "depth": 0,
                "source": f"initial_{mode}",
                "initialization_mode": mode,
            }
        )
        if len(normalized) >= resolved_max:
            break

    if len(normalized) < resolved_min:
        raise ValueError(
            f"{mode} initialization produced {len(normalized)} valid unique criteria; "
            f"at least {resolved_min} are required. Raw output: {str(text)[:800]}"
        )
    return normalized


class RubricInitializationStore:
    """Checkpointable prompt-level cache; each question is initialized once."""

    VERSION = 1

    def __init__(self, mode: str):
        self.mode = validate_initialization_mode(mode)
        self._states: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> Optional[list[dict[str, Any]]]:
        state = self._states.get(str(key))
        if state is None:
            return None
        return deepcopy(state["rubric"])

    def put(
        self,
        *,
        key: str,
        question: str,
        rubric: list[dict[str, Any]],
        created_step: int,
    ) -> None:
        key = str(key)
        existing = self._states.get(key)
        if existing is not None:
            if existing["rubric"] != rubric:
                raise ValueError(
                    f"Refusing to overwrite initialized rubric for prompt key {key}."
                )
            return
        self._states[key] = {
            "question": str(question),
            "mode": self.mode,
            "rubric": deepcopy(rubric),
            "created_step": int(created_step),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "mode": self.mode,
            "states": deepcopy(self._states),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        version = int(payload.get("version", 0))
        if version != self.VERSION:
            raise ValueError(
                f"Unsupported rubric initialization checkpoint version {version}; "
                f"expected {self.VERSION}."
            )
        checkpoint_mode = validate_initialization_mode(str(payload.get("mode") or ""))
        if checkpoint_mode != self.mode:
            raise ValueError(
                "Rubric initialization checkpoint mode mismatch: "
                f"checkpoint={checkpoint_mode}, config={self.mode}."
            )
        states = payload.get("states")
        if not isinstance(states, Mapping):
            raise ValueError("Rubric initialization checkpoint has no state mapping.")
        restored: dict[str, dict[str, Any]] = {}
        for key, state in states.items():
            if not isinstance(state, Mapping) or not isinstance(state.get("rubric"), list):
                raise ValueError(f"Malformed initialized-rubric state for key {key!r}.")
            restored[str(key)] = deepcopy(dict(state))
        self._states = restored

    def __len__(self) -> int:
        return len(self._states)
