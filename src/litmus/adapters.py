"""Adapters call an app and return the same normalized response (docs/design.md, section 4).

Error messages name the kind of failure only. They never include request or
response bodies, so they are safe to show for private datasets.
"""

import json
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import TypeVar

import httpx
import jmespath

from litmus.appconfig import AppConfig, Limits, resolve_env
from litmus.dataset import Record
from litmus.passages import PassagesError, load_passages
from litmus.templates import format_passages, render

T = TypeVar("T")


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


class Limiter:
    """concurrency, requests_per_minute and max_requests for one run, shared by every thread."""

    def __init__(self, limits: Limits, retries: int = 0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.limits = limits
        self.retries = retries
        self.clock = clock
        self.sleep = sleep
        self._slots = threading.Semaphore(limits.concurrency)
        self._lock = threading.Lock()
        self.calls = 0
        self._next_start = 0.0

    def call(self, attempt: Callable[[], T], failed: Callable[[T], dict | None]) -> T:
        """Run attempt() within the limits, retrying when failed() returns a retryable error.
        Raises BudgetExhausted before a call that would exceed max_requests."""
        attempts = 1 + self.retries
        for number in range(attempts):
            self._take_budget()
            self._wait_for_rate_limit()
            with self._slots:
                result = attempt()
            if number == attempts - 1 or not _retryable(failed(result)):
                return result
            self.sleep(min(2 ** number, 10))
        raise AssertionError("unreachable")

    def _take_budget(self) -> None:
        with self._lock:
            limit = self.limits.max_requests
            if limit is not None and self.calls >= limit:
                raise BudgetExhausted(f"reached max_requests ({limit})")
            self.calls += 1

    def _wait_for_rate_limit(self) -> None:
        per_minute = self.limits.requests_per_minute
        if per_minute is None:
            return
        with self._lock:
            now = self.clock()
            start = max(now, self._next_start)
            self._next_start = start + 60 / per_minute
        if start > now:
            self.sleep(start - now)


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
        self._limiter = Limiter(config.limits, config.retries, clock, sleep)

    @property
    def calls_made(self) -> int:
        return self._limiter.calls

    def close(self) -> None:
        self._client.close()

    def call(self, record: Record) -> AppResponse:
        """Send one record. Raises BudgetExhausted before a call that would exceed max_requests."""
        return self._limiter.call(lambda: self._attempt(record), lambda response: response.error)

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
        return _read_fields(self._paths, data, latency_ms, self.config.behavior_map)


class OpenAIAdapter:
    """Calls an OpenAI-compatible chat completions API: the model is the app. With passages_from, each
    question's saved passages go into the prompt and into the response, so answer models can be
    compared on exactly the same retrieval."""

    def __init__(self, config: AppConfig, env: Mapping[str, str] = os.environ,
                 transport: httpx.BaseTransport | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        settings = config.openai
        if settings is None:
            raise AdapterError("the openai adapter needs base_url and model")
        headers = {}
        if settings.api_key_env:
            key = env.get(settings.api_key_env)
            if not key:
                raise AdapterError(f"environment variable(s) not set: {settings.api_key_env}")
            headers["Authorization"] = f"Bearer {key}"
        try:
            self._passages = load_passages(settings.passages_from).by_id if settings.passages_from else None
        except PassagesError as error:
            raise AdapterError(str(error)) from None
        self.config = config
        self._settings = settings
        self._paths = {name: jmespath.compile(expression) for name, expression in config.response.items()}
        self._client = httpx.Client(base_url=settings.base_url.rstrip("/") + "/", headers=headers,
                                    timeout=settings.timeout_s, transport=transport, follow_redirects=False)
        self._clock = clock
        self._limiter = Limiter(config.limits, config.retries, clock, sleep)

    @property
    def calls_made(self) -> int:
        return self._limiter.calls

    def close(self) -> None:
        self._client.close()

    def call(self, record: Record) -> AppResponse:
        """Send one record. Raises BudgetExhausted before a call that would exceed max_requests."""
        return self._limiter.call(lambda: self._attempt(record), lambda response: response.error)

    def request_body(self, record: Record) -> dict:
        settings = self._settings
        passages = None
        if self._passages is not None:
            saved = self._passages.get(record.id)
            passages = format_passages(saved.texts, saved.ids) if saved else ""
        messages = []
        if settings.system_prompt:
            messages.append({"role": "system", "content": _prompt(settings.system_prompt, record, passages)})
        messages.append({"role": "user", "content": _prompt(settings.user_template, record, passages)})
        body = {"model": settings.model, "messages": messages}
        if settings.temperature is not None:
            body["temperature"] = settings.temperature
        body.update(settings.extra)
        if settings.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def _attempt(self, record: Record) -> AppResponse:
        body = self.request_body(record)
        started = self._clock()
        try:
            reply = self._client.post("chat/completions", json=body)
        except httpx.TimeoutException:
            return _failed("timeout", f"no reply within {self._settings.timeout_s} s", self._since(started))
        except httpx.HTTPError as error:
            return _failed("connection_error", type(error).__name__, self._since(started))
        latency_ms = self._since(started)
        if not 200 <= reply.status_code < 300:
            failed = _failed("http_status", f"HTTP {reply.status_code}", latency_ms)
            failed.error["status_code"] = reply.status_code
            return failed
        try:
            data = reply.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            return _failed("invalid_reply", "not a chat completions reply", latency_ms)
        if not isinstance(content, str) or not content.strip():
            return _failed("no_answer", "the model's reply has no text", latency_ms)
        if self._paths:
            try:
                parsed = _json_in(content)
            except ValueError:
                return _failed("invalid_json", "the model's reply isn't JSON", latency_ms)
            response = _read_fields(self._paths, parsed, latency_ms, self.config.behavior_map)
            if response.error is not None:
                return response
        else:
            response = AppResponse(actual_output=content, latency_ms=latency_ms)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        if response.input_tokens is None:
            response.input_tokens = _usage_count(usage.get("prompt_tokens"))
        if response.output_tokens is None:
            response.output_tokens = _usage_count(usage.get("completion_tokens"))
        saved = self._passages.get(record.id) if self._passages is not None else None
        if saved is not None and response.retrieval_context is None and response.retrieved_context_ids is None:
            response.retrieved_context_ids = list(saved.ids) if saved.ids is not None else None
            response.retrieval_context = list(saved.texts)
        return response

    def _since(self, started: float) -> float:
        return round((self._clock() - started) * 1000, 1)


Adapter = HttpAdapter | OpenAIAdapter


def create_adapter(config: AppConfig, env: Mapping[str, str] = os.environ) -> Adapter:
    if config.adapter == "http":
        return HttpAdapter(config, env)
    if config.adapter == "openai":
        return OpenAIAdapter(config, env)
    raise AdapterError(f"the {config.adapter} adapter isn't implemented yet")


def _read_fields(paths: dict, data: object, latency_ms: float, behavior_map: dict[str, str]) -> AppResponse:
    """The normalized response, from JMESPath expressions into a JSON reply."""
    found = {name: path.search(data) for name, path in paths.items()}
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
    if response.status is not None and response.status in behavior_map:
        response.observed_behavior = behavior_map[response.status]
        response.behavior_source = "status"
    return response


def _prompt(template: str, record: Record, passages: str | None) -> str:
    text = render(template, record, passages)
    return "" if text is None else str(text)


def _json_in(content: str) -> object:
    """The JSON in a model's reply, also when it's wrapped in a code fence or a sentence."""
    try:
        return json.loads(content)
    except ValueError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _usage_count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


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
