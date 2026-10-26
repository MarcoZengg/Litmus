"""The judge: a model that grades replies pass/fail with a one-sentence reason (docs/design.md, section 6).

It works on a finished run folder, so a run can be judged, or re-judged with new
prompts, without calling the app again. Results go next to records.jsonl:
judgments.jsonl (one line per question, repeat and metric) and judging.yaml
(which model and prompts were used, without secrets).
"""

import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from string import Template

import httpx
import yaml

from litmus import __version__
from litmus.adapters import BudgetExhausted, Limiter
from litmus.appconfig import Limits, _is_int, _limits
from litmus.dataset import BEHAVIORS, Dataset, Issue, Manifest, Record
from litmus.metrics import NOT_ANSWERED

JUDGE_METRICS = ("correctness", "faithfulness", "rubric", "behavior")
DEFAULT_METRICS = ["correctness", "faithfulness", "behavior"]
CONFIG_FIELDS = {"provider", "base_url", "model", "api_key_env", "temperature", "timeout_s", "json_mode", "extra",
                 "metrics", "rubric", "prompts", "limits", "retries"}
JUDGMENTS_FILE = "judgments.jsonl"
JUDGING_FILE = "judging.yaml"
MAX_REASON = 500


class JudgeError(Exception):
    """The judge can't be set up, e.g. a missing API key."""


@dataclass
class JudgeConfig:
    path: Path
    raw: dict  # as written; the API key is only named (api_key_env), never stored
    provider: str
    base_url: str
    model: str
    api_key_env: str | None = None
    temperature: float = 0.0
    timeout_s: float = 60.0
    json_mode: bool = True
    extra: dict = field(default_factory=dict)
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    rubric: str | None = None
    prompts: dict[str, str] = field(default_factory=dict)  # metric -> prompt text
    limits: Limits = field(default_factory=Limits)
    retries: int = 0

    @property
    def prompt_sha256(self) -> dict[str, str]:
        return {metric: hashlib.sha256(text.encode("utf-8")).hexdigest() for metric, text in self.prompts.items()}


def default_prompt(metric: str) -> str:
    return resources.files("litmus").joinpath("prompts", f"{metric}.txt").read_text(encoding="utf-8")


def load_judge_config(path: Path) -> tuple[JudgeConfig | None, list[Issue]]:
    where = path.name
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (yaml.YAMLError, UnicodeDecodeError):
        return None, [Issue(where, "not valid YAML")]
    if not isinstance(data, dict):
        return None, [Issue(where, "must be a list of settings, e.g. 'model: ...'")]

    problems = [f"unknown setting '{key}'" for key in sorted(set(map(str, data)) - CONFIG_FIELDS)]
    for key in ("provider", "model"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            problems.append(f"'{key}' is required")
    base_url = data.get("base_url")
    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        problems.append("'base_url' must start with http:// or https:// (an OpenAI-compatible API)")
    if "api_key_env" in data and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(data["api_key_env"])):
        problems.append("'api_key_env' must be the name of an environment variable")
    for key, default in (("temperature", 0), ("timeout_s", 60)):
        value = data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or (key == "timeout_s" and value == 0):
            problems.append(f"'{key}' must be a {'positive ' if key == 'timeout_s' else ''}number")
    if not isinstance(data.get("json_mode", True), bool):
        problems.append("'json_mode' must be true or false")
    if not isinstance(data.get("extra", {}), dict):
        problems.append("'extra' must be a list of settings to add to every request")
    metrics = data.get("metrics", DEFAULT_METRICS)
    if not isinstance(metrics, list) or not metrics or any(m not in JUDGE_METRICS for m in metrics):
        problems.append(f"'metrics' must list some of: {', '.join(JUDGE_METRICS)}")
        metrics = []
    rubric = data.get("rubric")
    if "rubric" in metrics and not (isinstance(rubric, str) and rubric.strip()):
        problems.append("the rubric metric needs 'rubric': the criteria, written out")
    prompts = {}
    overrides = data.get("prompts", {})
    if not isinstance(overrides, dict) or any(key not in JUDGE_METRICS for key in overrides):
        problems.append(f"'prompts' must map some of {', '.join(JUDGE_METRICS)} to prompt files")
        overrides = {}
    for metric in metrics:
        if metric in overrides:
            prompt_path = path.parent / str(overrides[metric])
            if not prompt_path.is_file():
                problems.append(f"prompt file for {metric} not found: {overrides[metric]}")
                continue
            prompts[metric] = prompt_path.read_text(encoding="utf-8-sig")
        else:
            prompts[metric] = default_prompt(metric)
    limits = _limits(data.get("limits", {}), problems)
    retries = data.get("retries", 0)
    if not _is_int(retries, minimum=0):
        problems.append("'retries' must be a whole number, 0 or more")

    if problems:
        return None, [Issue(where, problem) for problem in problems]
    return JudgeConfig(path=path, raw=data, provider=data["provider"], base_url=base_url, model=data["model"],
                       api_key_env=data.get("api_key_env"), temperature=data.get("temperature", 0),
                       timeout_s=data.get("timeout_s", 60), json_mode=data.get("json_mode", True),
                       extra=data.get("extra", {}), metrics=list(metrics), rubric=rubric, prompts=prompts,
                       limits=limits, retries=retries), []


def check_judge(manifest: Manifest | None, config: JudgeConfig, env: Mapping[str, str]) -> list[Issue]:
    """The dataset's allowed_providers, and the API key being set (docs/design.md, section 8)."""
    issues = []
    where = config.path.name
    if manifest is not None and manifest.allowed_providers is not None \
            and config.provider not in manifest.allowed_providers:
        issues.append(Issue(where, f"provider {config.provider} isn't in the dataset's allowed_providers"))
    if config.api_key_env and not env.get(config.api_key_env):
        issues.append(Issue(where, f"environment variable {config.api_key_env} is not set"))
    return issues


class JudgeClient:
    """Calls an OpenAI-compatible chat completions API, within the judge's limits."""

    def __init__(self, config: JudgeConfig, env: Mapping[str, str] = os.environ,
                 transport: httpx.BaseTransport | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        headers = {}
        if config.api_key_env:
            key = env.get(config.api_key_env)
            if not key:
                raise JudgeError(f"environment variable {config.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        self.config = config
        self._client = httpx.Client(base_url=config.base_url.rstrip("/") + "/", headers=headers,
                                    timeout=config.timeout_s, transport=transport, follow_redirects=False)
        self._limiter = Limiter(config.limits, config.retries, clock, sleep)

    @property
    def calls_made(self) -> int:
        return self._limiter.calls

    def close(self) -> None:
        self._client.close()

    def ask(self, prompt: str) -> dict:
        """{"content", "input_tokens", "output_tokens", "error"}. Raises BudgetExhausted at max_requests."""
        return self._limiter.call(lambda: self._attempt(prompt), lambda reply: reply["error"])

    def _attempt(self, prompt: str) -> dict:
        config = self.config
        body = {"model": config.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": config.temperature, **config.extra}
        if config.json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            reply = self._client.post("chat/completions", json=body)
        except httpx.TimeoutException:
            return _judge_failed("timeout", f"no reply within {config.timeout_s} s")
        except httpx.HTTPError as error:
            return _judge_failed("connection_error", type(error).__name__)
        if not 200 <= reply.status_code < 300:
            failed = _judge_failed("http_status", f"HTTP {reply.status_code}")
            failed["error"]["status_code"] = reply.status_code
            return failed
        try:
            data = reply.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            return _judge_failed("invalid_reply", "not a chat completions reply")
        usage = data.get("usage") or {}
        return {"content": content, "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"), "error": None}


def _judge_failed(kind: str, message: str) -> dict:
    return {"content": None, "input_tokens": None, "output_tokens": None, "error": {"type": kind, "message": message}}


def parse_verdict(content: str, metric: str) -> tuple[str, str]:
    """(verdict, reason). The verdict is pass/fail, or a behavior for the behavior metric."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        match = re.search(r"\{.*\}", content or "", re.DOTALL)
        if not match:
            raise ValueError("no JSON in the judge's reply")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("the judge's reply isn't a JSON object")
    reason = str(data.get("reason", ""))[:MAX_REASON]
    if metric == "behavior":
        behavior = str(data.get("behavior", "")).strip().lower()
        if behavior not in BEHAVIORS:
            raise ValueError("the judge's behavior isn't one of the five")
        return behavior, reason
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in ("pass", "fail"):
        raise ValueError("the judge's verdict isn't pass or fail")
    return verdict, reason


@dataclass
class Task:
    id: str
    repeat: int
    metric: str
    prompt: str | None = None  # None: decided without asking the judge
    verdict: str | None = None
    reason: str | None = None


CALL = "call"


def decide(record: Record, row: dict, metric: str, glass_box: bool) -> str | None:
    """How a judge metric applies to one reply: None (it doesn't), CALL (ask the judge), or the reason
    it fails without a call. Failed calls and non-answers need no judge: they fail correctness, and
    have no claims to check for faithfulness."""
    failed = row["error"] is not None
    no_answer = row["observed_behavior"] in NOT_ANSWERED
    if metric == "correctness":
        if record.expected_behavior != "answer" or not record.expected_output:
            return None
        if failed or no_answer:
            return "the app call failed" if failed else "the app didn't answer"
    elif metric == "faithfulness":
        if not glass_box or (not failed and (no_answer or not row.get("retrieval_context"))):
            return None
        if failed:
            return "the app call failed"
    elif metric == "rubric":
        if failed:
            return "the app call failed"
    elif metric == "behavior":
        if failed or row["observed_behavior"] is not None:
            return None
    return CALL


def plan_judgments(dataset: Dataset, rows: list[dict], config: JudgeConfig, glass_box: bool) -> list[Task]:
    """One task per question, repeat and metric that applies."""
    records = {record.id: record for record in dataset.records}
    tasks = []
    for row in rows:
        record = records.get(row["id"])
        if record is None:
            continue
        for metric in config.metrics:
            decision = decide(record, row, metric, glass_box)
            if decision is None:
                continue
            task = Task(record.id, row["repeat"], metric)
            if decision == CALL:
                task.prompt = _prompt(config, metric, record, row)
            else:
                task.verdict, task.reason = "fail", decision
            tasks.append(task)
    return tasks


def format_passages(row: dict) -> str:
    texts = row.get("retrieval_context") or []
    ids = row.get("retrieved_context_ids") or []
    return "\n".join(f"[{n}] ({ids[n - 1]}) {text}" if len(ids) == len(texts) else f"[{n}] {text}"
                     for n, text in enumerate(texts, 1))


def _prompt(config: JudgeConfig, metric: str, record: Record, row: dict) -> str:
    return Template(config.prompts[metric]).safe_substitute(
        question=record.input, reference=record.expected_output or "", answer=row["actual_output"] or "",
        passages=format_passages(row), criteria=config.rubric or "")


@dataclass
class JudgeResult:
    planned_calls: int
    automatic: int
    judged: int = 0
    errors: int = 0
    stop_reason: str | None = None

    @property
    def complete(self) -> bool:
        return self.stop_reason is None and self.judged + self.errors == self.planned_calls


def judge_run(folder: Path, tasks: list[Task], config: JudgeConfig, client: JudgeClient,
              progress: Callable[[int, int, int], None] | None = None) -> JudgeResult:
    """Write automatic decisions, then ask the judge for the rest, writing each line as it finishes."""
    calls = [task for task in tasks if task.prompt is not None]
    result = JudgeResult(planned_calls=len(calls), automatic=len(tasks) - len(calls))
    pending = iter(calls)
    lock = threading.Lock()
    stop = threading.Event()
    started_at = _now()

    with open(folder / JUDGMENTS_FILE, "w", encoding="utf-8") as out:
        def write(task: Task, source: str, reply: dict | None = None, error: dict | None = None) -> None:
            line = {"id": task.id, "repeat": task.repeat, "metric": task.metric, "verdict": task.verdict,
                    "reason": task.reason, "source": source,
                    "input_tokens": reply["input_tokens"] if reply else None,
                    "output_tokens": reply["output_tokens"] if reply else None, "error": error}
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
            out.flush()

        for task in tasks:
            if task.prompt is None:
                write(task, "auto")

        def worker() -> None:
            while not stop.is_set():
                with lock:
                    task = next(pending, None)
                if task is None:
                    return
                try:
                    reply = client.ask(task.prompt)
                except BudgetExhausted:
                    with lock:
                        result.stop_reason = "max_requests reached"
                    stop.set()
                    return
                error = reply["error"]
                if error is None:
                    try:
                        task.verdict, task.reason = parse_verdict(reply["content"], task.metric)
                    except ValueError as problem:
                        error = {"type": "unreadable_verdict", "message": str(problem)}
                with lock:
                    write(task, "judge", reply, error)
                    if error is None:
                        result.judged += 1
                    else:
                        result.errors += 1
                    if progress:
                        progress(result.judged + result.errors, result.planned_calls, result.errors)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(config.limits.concurrency)]
        for thread in threads:
            thread.start()
        try:
            while any(thread.is_alive() for thread in threads):
                for thread in threads:
                    thread.join(0.2)
        except KeyboardInterrupt:
            result.stop_reason = "interrupted"
            stop.set()
            for thread in threads:
                thread.join()

    judging = {
        "litmus_version": __version__,
        "status": "complete" if result.complete else "incomplete",
        "stop_reason": result.stop_reason,
        "started_at": started_at,
        "finished_at": _now(),
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "metrics": config.metrics,
        "prompt_sha256": config.prompt_sha256,
        "counts": {"judge_calls": client.calls_made, "judged": result.judged, "errors": result.errors,
                   "automatic": result.automatic},
        "config": config.raw,
    }
    (folder / JUDGING_FILE).write_text(yaml.safe_dump(judging, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return result


def load_judgments(folder: Path) -> tuple[dict | None, list[dict]]:
    """(judging.yaml, judgments.jsonl lines), or (None, []) when the run hasn't been judged."""
    judging_file, judgments_file = folder / JUDGING_FILE, folder / JUDGMENTS_FILE
    if not judging_file.is_file() or not judgments_file.is_file():
        return None, []
    judging = yaml.safe_load(judging_file.read_text(encoding="utf-8"))
    lines = [json.loads(line) for line in judgments_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return judging, lines


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
