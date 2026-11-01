"""Fixed, non-trainable meta-verification using the configured Judge backend.

Uses the existing judge's provider/client/request compatibility helpers. Actor
weights and optimizer updates never enter this backend. Exhausted format errors
use batch-to-single fallback, then an internal unavailable marker. Widespread
audit failures and API transport/authentication failures raise explicitly.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import threading
import uuid
from typing import Any

from jinja2 import Environment, StrictUndefined
from .rubric_meta_response_normalization import normalize_meta_response
from .rubric_request_control import controlled_completion
from .rubric_meta_recovery import UnavailableAudit, single_candidate_entry
from .rubric_meta_response_validation import validate_meta_response

_FAILURE_LOCK = threading.Lock()


class FixedRubricMetaVerifier:
    def __init__(self, reward_config: Any, *, overrides: dict[str, Any] | None = None, max_workers: int = 8,
                 log_dir: str | None = None):
        """log_dir: per-run directory for audit logs (default: shared tmp/rubric_meta_logs).

        Only unrecoverable responses keep the full audit input (one JSON per
        audit under failures/). Normalized and retry-recovered responses append
        compact records to normalizations.jsonl / recovered.jsonl.
        """
        self.kwargs = {**dict(reward_config.reward_function_kwargs), **dict(overrides or {})}
        root = Path(log_dir or "tmp/rubric_meta_logs")
        self.failure_dir = Path(self.kwargs.get("meta_failure_log_dir") or root / "failures")
        self.normalization_dir = Path(self.kwargs.get("meta_normalization_log_dir") or root)
        self.jev = None
        if str(self.kwargs.get("judge_provider", "")).lower() in {"jev", "typesafe"}:
            from .rubric_jev_meta_verifier import JevRubricMetaVerifier
            self.jev = JevRubricMetaVerifier(self.kwargs, max_workers=max_workers)
            return
        spec = importlib.util.spec_from_file_location("rubric_generator_fixed_judge", reward_config.reward_function)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for method in ("resolve_judge_settings", "_resolve_base_urls", "_get_client", "_build_judge_request"):
            if not hasattr(module, method):
                raise ValueError(f"Fixed meta backend needs the rubric_llm_judge interface: missing {method}.")
        self.module = module
        self.settings = module.resolve_judge_settings(**{key: self.kwargs.get(key) for key in
            ("judge_provider", "model", "base_url", "api_key_env", "reasoning_effort", "enable_thinking")})
        self.urls = module._resolve_base_urls(base_url=self.settings["base_url"], base_urls=self.kwargs.get("base_urls"))
        self.max_workers = max(1, min(max_workers, int(self.kwargs.get("max_workers", max_workers))))

    def _append_jsonl(self, name: str, record: dict[str, Any], stats: dict[str, int], counter: str) -> None:
        with _FAILURE_LOCK:
            stats[counter] += 1
            self.normalization_dir.mkdir(parents=True, exist_ok=True)
            with (self.normalization_dir / name).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def generate(self, entries: list[dict[str, Any]], *, template_path: str, max_tokens: int) -> list[str]:
        if not entries:
            return []
        if self.jev is not None:
            return self.jev.generate(entries, template_path=template_path, max_tokens=max_tokens)
        template = Environment(undefined=StrictUndefined, autoescape=False).from_string(Path(template_path).read_text(encoding="utf-8"))

        def run(task) -> str:
            index, entry = task
            prompt = template.render(question=entry["question"], hint=entry["hint"])
            if entry.get("meta_context"):
                prompt += ("\n\nReturn one results array. Each candidate is a separate JSON object "
                    "with only candidate_id and verdict. Close each object before the next one. "
                    "Do not output checks or explanations.")
            url = self.urls[index % len(self.urls)]
            client = self.module._get_client(self.settings["api_key_env"], url,
                float(self.kwargs.get("timeout", 120.0)), int(self.kwargs.get("max_retries", 2)))
            request = self.module._build_judge_request({}, [], model=self.settings["model"], base_url=url,
                reasoning_effort=self.settings["reasoning_effort"], enable_thinking=self.settings["enable_thinking"],
                temperature=0.0, max_tokens=max_tokens,
                chat_template_kwargs=self.kwargs.get("chat_template_kwargs"), extra_body=self.kwargs.get("extra_body"),
                response_format=self.kwargs.get("response_format", {"type": "json_object"}))
            request["messages"] = [{"role": "user", "content": prompt}]
            retries = int(self.kwargs.get("response_retries", 2))
            if retries < 0:
                raise ValueError("response_retries must be nonnegative")
            audit_id = uuid.uuid4().hex
            context = entry.get("meta_context") or {}
            summary = {"audit_id": audit_id, "operation_type": context.get("operation_type"),
                       "candidate_ids": [item.get("candidate_id") for item in context.get("candidates", [])]}
            attempts = []
            for attempt in range(retries + 1):
                response = controlled_completion(client, request, control=self.kwargs.get("request_control"),
                    transport_retries=int(self.kwargs.get("max_retries", 2)))
                choice = response.choices[0]
                text = choice.message.content
                try:
                    if getattr(choice, "finish_reason", None) == "length":
                        raise ValueError("Meta-verifier response truncated")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("Empty meta-verifier response")
                    normalized, changes = normalize_meta_response(text, entry.get("meta_context"))
                    if changes:
                        self._append_jsonl("normalizations.jsonl", {**summary, "attempt": attempt + 1,
                            "changes": changes, "response": text}, stats, "normalized")
                    if attempts:
                        self._append_jsonl("recovered.jsonl", {**summary, "attempts": attempts,
                            "recovered_on_attempt": attempt + 1}, stats, "recovered")
                    return normalized
                except ValueError as exc:
                    attempts.append({"attempt": attempt + 1, "error": str(exc), "response": text})
                    with _FAILURE_LOCK:
                        stats["invalid_attempts"] += 1
                    if attempt == retries:
                        path = self.failure_dir / f"{audit_id}.json"
                        with _FAILURE_LOCK:
                            stats["unrecoverable"] += 1
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(json.dumps({**summary, "entry": entry, "attempts": attempts},
                                ensure_ascii=False, indent=2))
                        return UnavailableAudit(f"Meta-verifier schema failed after retries; see {path}")
                    request["messages"] = [{"role": "user", "content": prompt +
                        "\n\nYour previous response failed schema validation: " + str(exc) +
                        "\nReturn corrected compact JSON with exactly one candidate_id and verdict "
                        "per results object. Use VALID, INVALID: <dimension>, or REVIEW: <dimension>."}]
            raise AssertionError("Unreachable meta retry state")

        stats = {"normalized": 0, "recovered": 0, "invalid_attempts": 0, "unrecoverable": 0, "split_batches": 0}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            responses = list(pool.map(run, enumerate(entries)))
            tasks, owners = [], []
            for index, response in enumerate(responses):
                candidates = entries[index].get("meta_context", {}).get("candidates", [])
                if isinstance(response, UnavailableAudit) and len(candidates) > 1:
                    for candidate in candidates:
                        tasks.append((index, single_candidate_entry(entries[index], candidate)))
                        owners.append(index)
            if tasks:
                stats["split_batches"] = len(set(owners))
                single_results = list(pool.map(run, tasks))
                grouped = {}
                for owner, result in zip(owners, single_results, strict=True):
                    grouped.setdefault(owner, []).append(result)
                for owner, results in grouped.items():
                    if any(isinstance(result, UnavailableAudit) for result in results):
                        responses[owner] = UnavailableAudit("Single-candidate fallback exhausted for batch " + str(owner))
                        continue
                    merged = []
                    for result in results:
                        payload = json.loads(result)
                        merged.extend(payload if isinstance(payload, list) else payload['results'])
                    text = json.dumps({'results': merged}, ensure_ascii=False)
                    validate_meta_response(text, entries[owner].get('meta_context'))
                    responses[owner] = text
        if any(stats.values()):
            print(f"[MetaVerifier] {len(entries)} requests (+{len(tasks)} single-candidate fallbacks): "
                  f"{stats['normalized']} normalized, "
                  f"{stats['invalid_attempts']} invalid attempts, {stats['recovered']} recovered by retry, "
                  f"{stats['unrecoverable']} unrecoverable, "
                  f"{stats['split_batches']} batches split to single audits; "
                  f"logs in {self.normalization_dir} and {self.failure_dir}.", flush=True)
        unavailable = sum(isinstance(result, UnavailableAudit) for result in responses)
        if unavailable:
            print(f"[MetaVerifier] {unavailable}/{len(entries)} audits unavailable after fallback; affected RG groups will be excluded.", flush=True)
            # Widespread service failure is actionable; don't silently train
            # indefinitely without verification. Small batches may skip safely.
            if len(entries) >= 8 and unavailable / len(entries) > float(self.kwargs.get('meta_max_unavailable_fraction', .1)):
                raise RuntimeError(f"Meta-verifier unavailable for {unavailable}/{len(entries)} requests after fallback")
        return responses
