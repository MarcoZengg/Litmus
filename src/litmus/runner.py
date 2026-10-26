"""Run a dataset against an app and save the results (docs/design.md, sections 3 and 5).

A run folder holds run.yaml (what was run, without secrets) and records.jsonl
(one line per question per repeat, written as each call finishes, so a
stopped run keeps everything done so far).
"""

import hashlib
import json
import re
import secrets
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from litmus import __version__
from litmus.adapters import AppResponse, BudgetExhausted, HttpAdapter
from litmus.appconfig import AppConfig
from litmus.dataset import Dataset, Issue

RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
RUN_FILE = "run.yaml"
RECORDS_FILE = "records.jsonl"


def is_glass_box(run: dict) -> bool:
    """Whether the run's app config reads retrieved passages from the reply (docs/design.md, section 4)."""
    response = (run.get("app") or {}).get("config", {}).get("response") or {}
    return "retrieved_context_ids" in response


def load_run(folder: Path) -> dict:
    return yaml.safe_load((folder / RUN_FILE).read_text(encoding="utf-8"))


def load_rows(folder: Path) -> list[dict]:
    return [json.loads(line) for line in (folder / RECORDS_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def run_folder(dataset: Dataset, name: str, today: datetime | None = None) -> Path:
    date = (today or datetime.now()).strftime("%Y-%m-%d")
    return dataset.folder / "runs" / f"{date}_{name}"


@dataclass
class RunResult:
    folder: Path
    planned: int
    written: int = 0
    errors: int = 0
    app_calls: int = 0
    stop_reason: str | None = None
    warnings: list[Issue] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.stop_reason is None and self.written == self.planned


def execute_run(dataset: Dataset, config: AppConfig, saved_config: dict, adapter: HttpAdapter, folder: Path,
                name: str, repeats: int,
                progress: Callable[[int, int, int], None] | None = None) -> RunResult:
    """Call the app for every record and repeat; repeat 1 for all records first, so a run
    stopped early still covers as many questions as possible."""
    folder.mkdir(parents=True, exist_ok=False)
    jobs = iter([(repeat, record) for repeat in range(1, repeats + 1) for record in dataset.records])
    result = RunResult(folder=folder, planned=len(dataset.records) * repeats)
    responses: dict[str, list[AppResponse]] = defaultdict(list)
    lock = threading.Lock()
    stop = threading.Event()
    started_at = _now()

    with open(folder / RECORDS_FILE, "w", encoding="utf-8") as out:
        def worker() -> None:
            while not stop.is_set():
                with lock:
                    job = next(jobs, None)
                if job is None:
                    return
                repeat, record = job
                try:
                    response = adapter.call(record)
                except BudgetExhausted:
                    with lock:
                        result.stop_reason = "max_requests reached"
                    stop.set()
                    return
                except Exception as error:  # a bug must not lose the records already written
                    response = AppResponse(error={"type": "internal_error", "message": type(error).__name__})
                with lock:
                    out.write(json.dumps({"id": record.id, "repeat": repeat, **response.to_dict()},
                                         ensure_ascii=False) + "\n")
                    out.flush()
                    result.written += 1
                    result.errors += response.error is not None
                    responses[record.id].append(response)
                    if progress:
                        progress(result.written, result.planned, result.errors)

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

    result.app_calls = adapter.calls_made
    result.warnings = _warnings(dataset, responses, repeats)
    _write_run_file(folder, dataset, config, saved_config, name, repeats, result, started_at)
    return result


def _write_run_file(folder: Path, dataset: Dataset, config: AppConfig, saved_config: dict, name: str,
                    repeats: int, result: RunResult, started_at: str) -> None:
    manifest = dataset.manifest
    run = {
        "litmus_version": __version__,
        "name": name,
        "status": "complete" if result.complete else "incomplete",
        "stop_reason": result.stop_reason,
        "started_at": started_at,
        "finished_at": _now(),
        "repeats": repeats,
        "seed": secrets.randbits(32),  # for the bootstrap, so comparisons can be reproduced
        "dataset": {
            "name": manifest.name,
            "version": manifest.version,
            "kb_version": manifest.kb_version,
            "sha256": dataset.sha256,
            "records": len(dataset.records),
            "private": manifest.private,
        },
        "app": {
            "file": config.path.name,
            "sha256": hashlib.sha256(config.path.read_bytes()).hexdigest(),
            "config": saved_config,
        },
        "judge": None,
        "counts": {
            "planned": result.planned,
            "written": result.written,
            "errors": result.errors,
            "app_calls": result.app_calls,
        },
    }
    (folder / RUN_FILE).write_text(yaml.safe_dump(run, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _warnings(dataset: Dataset, responses: dict[str, list[AppResponse]], repeats: int) -> list[Issue]:
    issues = []
    every = [r for rs in responses.values() for r in rs]
    cached = sum(r.cached is True for r in every)
    if cached:
        issues.append(Issue("cache", f"{cached} replies came from the app's cache, so those repeats "
                                     "aren't independent"))
    if repeats > 1 and responses:
        answered = [rs for rs in responses.values() if len(rs) > 1 and all(r.error is None for r in rs)]
        same = sum(len({r.actual_output for r in rs}) == 1 for rs in answered)
        if answered and same / len(answered) >= 0.9:
            issues.append(Issue("repeats", f"{same} of {len(answered)} questions gave identical answers on "
                                           "every repeat; if that's unexpected, check for caching"))
    unmapped = sum(r.status is not None and r.observed_behavior is None for r in every)
    if unmapped:
        issues.append(Issue("behavior_map", f"{unmapped} replies had a status missing from behavior_map; "
                                            "their behavior is left for the judge"))
    expected = dataset.manifest.kb_version
    reported = {r.kb_version for r in every if r.kb_version is not None}
    if expected is not None and reported - {expected}:
        issues.append(Issue("kb_version", f"the app reported kb_version {', '.join(sorted(reported))} but the "
                                          f"dataset says {expected}; passage labels may be stale"))
    return issues


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
