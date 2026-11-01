"""Single-host, cross-process pacing for Judge and Meta-Verify API calls.

All callers sharing an API quota must use the same local SQLite path and scope.
Admission is evenly spaced (no burst bucket); every retry needs admission.
The database stores only quota timing, never prompts, responses, or API keys.
Multiple hosts require a distributed limiter instead of this local database.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
import math
from pathlib import Path
import random
import sqlite3
import time
from typing import Any


class RequestControlError(RuntimeError):
    """Transport retries exhausted; never treat this as a model-quality label."""


@dataclass(frozen=True)
class RequestControl:
    rpm: float
    state_path: str
    scope: str
    rate_limit_retries: int = 6
    backoff_initial: float = 60.0
    backoff_max: float = 120.0

    def __post_init__(self):
        if not math.isfinite(self.rpm) or self.rpm <= 0:
            raise ValueError("request_control.rpm must be positive and finite")
        if not self.scope or not Path(self.state_path).is_absolute():
            raise ValueError("request_control requires a scope and absolute state_path")
        if self.rate_limit_retries < 0 or not isinstance(self.rate_limit_retries, int):
            raise ValueError("rate_limit_retries must be a nonnegative integer")
        if not all(math.isfinite(v) and v > 0 for v in (self.backoff_initial, self.backoff_max)):
            raise ValueError("request_control backoff must be positive and finite")
        if self.backoff_max < self.backoff_initial:
            raise ValueError("backoff_max must be >= backoff_initial")


class SharedRequestGate:
    def __init__(self, config: RequestControl):
        self.config = config
        Path(config.state_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS request_quota "
                       "(scope TEXT PRIMARY KEY, next_at REAL NOT NULL, blocked_until REAL NOT NULL)")
            db.execute("INSERT OR IGNORE INTO request_quota VALUES (?, 0, 0)", (config.scope,))

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.config.state_path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def acquire(self) -> None:
        # Never sleep with a database lock, nor reserve future slots: a new
        # shared cooldown must take effect for all callers still waiting.
        with self._connect() as db:
            while True:
                db.execute("BEGIN IMMEDIATE")
                next_at, blocked = db.execute(
                    "SELECT next_at, blocked_until FROM request_quota WHERE scope=?",
                    (self.config.scope,)).fetchone()
                now = time.time()
                delay = max(next_at, blocked) - now
                if delay <= 0:
                    db.execute("UPDATE request_quota SET next_at=? WHERE scope=?",
                               (now + 60.0 / self.config.rpm, self.config.scope))
                    db.commit()
                    return
                db.commit()
                time.sleep(min(delay, 1.0))

    def cooldown(self, seconds: float) -> None:
        with self._connect() as db:
            db.execute("UPDATE request_quota SET blocked_until=MAX(blocked_until, ?) WHERE scope=?",
                       (time.time() + seconds, self.config.scope))


def retry_after_seconds(exc: Exception) -> float:
    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return 0.0
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = parsedate_to_datetime(str(value)).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0., seconds) if math.isfinite(seconds) else 0.0


def controlled_completion(client: Any, request: dict, *, control: dict | None = None,
                          transport_retries: int = 2) -> Any:
    """Preserve legacy SDK behavior unless request control is explicitly set."""
    if not control:
        return client.chat.completions.create(**request)
    config = RequestControl(**dict(control))
    if transport_retries < 0:
        raise ValueError("transport_retries must be nonnegative")
    gate = SharedRequestGate(config)
    # SDK retries would bypass our quota accounting and shared cooldown.
    caller = client.with_options(max_retries=0)
    rate_failures = transport_failures = 0
    while True:
        gate.acquire()
        try:
            return caller.chat.completions.create(**request)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status == 429:
                delay = max(retry_after_seconds(exc),
                    min(config.backoff_max, config.backoff_initial * 2 ** rate_failures))
                delay += random.uniform(0, min(1., delay * .1))
                gate.cooldown(delay)
                rate_failures += 1
                retry_status = (f"retry {rate_failures}/{config.rate_limit_retries}"
                    if rate_failures <= config.rate_limit_retries else "retries exhausted")
                print(f"[RequestControl] 429: shared cooldown {delay:.1f}s; "
                      f"{retry_status}, cap {config.rpm:g} RPM", flush=True)
                if rate_failures > config.rate_limit_retries:
                    raise RequestControlError("API rate-limit retries exhausted") from exc
                continue
            # Preserve bounded retries for SDK transport errors and transient HTTP failures.
            transient = status in (408, 409) or (isinstance(status, int) and status >= 500)
            transient = transient or type(exc).__name__ in ("APIConnectionError", "APITimeoutError")
            if not transient:
                raise
            if transport_failures >= transport_retries:
                raise RequestControlError("API transport retries exhausted") from exc
            transport_failures += 1
            time.sleep(max(retry_after_seconds(exc), min(8., 2 ** (transport_failures - 1))))
