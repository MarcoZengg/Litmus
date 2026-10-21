"""Adapters call an app and return the same normalized response (docs/design.md, section 4).

Error messages name the kind of failure only. They never include request or
response bodies, so they are safe to show for private datasets.
"""

import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

import httpx
import jmespath

from litmus.appconfig import AppConfig, resolve_env
from litmus.dataset import Record
from litmus.templates import render


class AdapterError(Exception):
    """The adapter can't be set up, e.g. a missing environment variable."""


class BudgetExhausted(Exception):
    """The run reached limits.max_requests; no further calls are made."""


@dataclass
class AppResponse:
    actual_output: str | None = None
    retrieved_context_ids: list[str] | None = None
    retrieval_context: list[str] | None = None
    status: str | None = None
    observed_behavior: str | None = None
    behavior_source: str | None = None  # "status" now; "judge" once the judge exists
    cached: bool | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    kb_version: str | None = None
    latency_ms: float | None = None
    error: dict | None = None  # {"type": ..., "message": ...}

    def to_dict(self) -> dict:
        return asdict(self)


class _FieldError(Exception):
    pass


class HttpAdapter:
    def __init__(self, config: AppConfig, env: Mapping[str, str] = os.environ,
                 transport: httpx.BaseTransport | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        if config.request is None:
            raise AdapterError("the http adapter needs a 'request' section")
        resolved, missing = resolve_env(config.raw["request"], env)
        if missing:
            raise AdapterError(f"environment variable(s) not set: {', '.join(missing)}")
        self.config = config
        self._url = resolved["url"]
        self._headers = resolved.get("headers", {})
        self._body = resolved.get("body")
        self._paths = {name: jmespath.compile(expression) for name, expression in config.response.items()}
        self._client = httpx.Client(timeout=config.request.timeout_s, transport=transport, follow_redirects=False)
        self._clock = clock
        self._sleep = sleep
        self._slots = threading.Semaphore(config.limits.concurrency)
        self._lock = threading.Lock()
        self._calls = 0
        self._next_start = 0.0

    @property
    def calls_made(self) -> int:
        return self._calls

    def close(self) -> None:
        self._client.close()

    def call(self, record: Record) -> AppResponse:
        """Send one record. Raises BudgetExhausted before a call that would exceed max_requests."""
        attempts = 1 + self.config.retries
        for attempt in range(attempts):
            self._take_budget()
            self._wait_for_rate_limit()
            with self._slots:
                response = self._attempt(record)
            if attempt == attempts - 1 or not _retryable(response.error):
                return response
            self._sleep(min(2 ** attempt, 10))
        raise AssertionError("unreachable")

    def _take_budget(self) -> None:
        with self._lock:
            limit = self.config.limits.max_requests
            if limit is not None and self._calls >= limit:
                raise BudgetExhausted(f"reached max_requests ({limit})")
            self._calls += 1

    def _wait_for_rate_limit(self) -> None:
        per_minute = self.config.limits.requests_per_minute
        if per_minute is None:
            return
        with self._lock:
            now = self._clock()
            start = max(now, self._next_start)
            self._next_start = start + 60 / per_minute
        if start > now:
            self._sleep(start - now)

    def _attempt(self, record: Record) -> AppResponse:
        request = self.config.request
        body = render(self._body, record)
        if request.method == "GET":
            payload = {"params": body or {}}
        else:
            payload = {} if body is None else {"json": body}
        started = self._clock()
        try:
            reply = self._client.request(request.method, render(self._url, record),
                                         headers=render(self._headers, record), **payload)
        except httpx.TimeoutException:
            return _failed("timeout", f"no reply within {request.timeout_s} s", self._since(started))
        except httpx.HTTPError as error:
            return _failed("connection_error", type(error).__name__, self._since(started))
        latency_ms = self._since(started)
        if not 200 <= reply.status_code < 300:
            failed = _failed("http_status", f"HTTP {reply.status_code}", latency_ms)
            failed.error["status_code"] = reply.status_code
            return failed
        try:
            data = reply.json()
        except ValueError:
            return _failed("invalid_json", "the reply isn't JSON", latency_ms)
        return self._parse(data, latency_ms)

    def _since(self, started: float) -> float:
        return round((self._clock() - started) * 1000, 1)

    def _parse(self, data: object, latency_ms: float) -> AppResponse:
        found = {name: path.search(data) for name, path in self._paths.items()}
        try:
            output = found.get("actual_output")
            if output is None:
                return _failed("no_answer", "the actual_output expression found nothing", latency_ms)
            response = AppResponse(
                actual_output=_text(output, "actual_output"),
                retrieved_context_ids=_text_list(found.get("retrieved_context_ids"), "retrieved_context_ids",
                                                 allow_numbers=True),
                retrieval_context=_text_list(found.get("retrieval_context"), "retrieval_context"),
                status=_scalar_text(found.get("status"), "status"),
                cached=_boolean(found.get("cached"), "cached"),
                input_tokens=_count(found.get("input_tokens"), "input_tokens"),
                output_tokens=_count(found.get("output_tokens"), "output_tokens"),
                cost=_number(found.get("cost"), "cost"),
                kb_version=_scalar_text(found.get("kb_version"), "kb_version"),
                latency_ms=latency_ms,
            )
        except _FieldError as error:
            return _failed("bad_response_field", str(error), latency_ms)
        if response.status is not None and response.status in self.config.behavior_map:
            response.observed_behavior = self.config.behavior_map[response.status]
            response.behavior_source = "status"
        return response


def create_adapter(config: AppConfig, env: Mapping[str, str] = os.environ) -> HttpAdapter:
    if config.adapter == "http":
        return HttpAdapter(config, env)
    raise AdapterError(f"the {config.adapter} adapter isn't implemented yet")


def _failed(kind: str, message: str, latency_ms: float | None) -> AppResponse:
    return AppResponse(latency_ms=latency_ms, error={"type": kind, "message": message})


def _retryable(error: dict | None) -> bool:
    if error is None:
        return False
    if error["type"] in ("timeout", "connection_error"):
        return True
    code = error.get("status_code", 0)
    return error["type"] == "http_status" and (code == 429 or code >= 500)


def _text(value: object, name: str) -> str:
    if isinstance(value, (dict, list, bool)):
        raise _FieldError(f"{name} isn't text")
    return str(value)


def _scalar_text(value: object, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _text_list(value: object, name: str, allow_numbers: bool = False) -> list[str] | None:
    if value is None:
        return None
    allowed = (str, int, float) if allow_numbers else (str,)
    if not isinstance(value, list) or not all(isinstance(i, allowed) and not isinstance(i, bool) for i in value):
        raise _FieldError(f"{name} isn't a list of {'IDs' if allow_numbers else 'text'}")
    return [str(item) for item in value]


def _boolean(value: object, name: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise _FieldError(f"{name} isn't true or false")
    return value


def _count(value: object, name: str) -> int | None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise _FieldError(f"{name} isn't a whole number")
    return value


def _number(value: object, name: str) -> float | None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise _FieldError(f"{name} isn't a number")
    return value
