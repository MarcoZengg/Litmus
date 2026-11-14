"""Saved passages for the openai adapter's passages_from (docs/design.md, section 4).

Replaying the passages an app retrieved in an earlier run gives every answer
model exactly the same evidence, so a comparison measures the model and not
the retriever. The source is a run folder (its records.jsonl) or any JSONL
file with the same fields: id, retrieved_context_ids and retrieval_context.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from litmus.appconfig import AppConfig
from litmus.dataset import Dataset, Issue

SHOWN_IDS = 10


class PassagesError(Exception):
    """passages_from can't be read."""


@dataclass(frozen=True)
class Passages:
    texts: list[str]
    ids: list[str] | None = None  # None when the saved IDs don't line up with the texts


@dataclass
class SavedPassages:
    by_id: dict[str, Passages]
    source: str  # for summaries: "run <name>" or the file's name
    dataset_sha256: str | None = None  # the run's dataset hash, when the source is a run folder


def load_passages(path: Path) -> SavedPassages:
    """Each question's passages. Replies without passage texts, and failed calls, are skipped;
    when a question has several repeats, the lowest repeat wins."""
    from litmus.runner import RECORDS_FILE, RUN_FILE, load_run  # the runner imports the adapters

    source, sha256 = path.name, None
    if path.is_dir():
        if not (path / RUN_FILE).is_file() or not (path / RECORDS_FILE).is_file():
            raise PassagesError(f"passages_from: {path.name} isn't a run folder: it needs {RUN_FILE} and {RECORDS_FILE}")
        run = load_run(path)
        source = f"run {run.get('name') or path.name}"
        sha256 = (run.get("dataset") or {}).get("sha256")
        path = path / RECORDS_FILE
    elif not path.is_file():
        raise PassagesError(f"passages_from: {path} not found")
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError):
        raise PassagesError(f"passages_from: can't read {path.name} as UTF-8 text") from None

    best: dict[str, tuple[int, Passages]] = {}
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise PassagesError(f"passages_from: line {number} of {path.name} isn't JSON") from None
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise PassagesError(f"passages_from: line {number} of {path.name} has no id")
        texts = row.get("retrieval_context")
        if row.get("error") or not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            continue
        ids = row.get("retrieved_context_ids")
        ids = [str(i) for i in ids] if isinstance(ids, list) and len(ids) == len(texts) else None
        repeat = row["repeat"] if isinstance(row.get("repeat"), int) else 1
        if row["id"] not in best or repeat < best[row["id"]][0]:
            best[row["id"]] = (repeat, Passages(texts, ids))
    return SavedPassages({key: passages for key, (_, passages) in best.items()}, source, sha256)


def check_passages(config: AppConfig, dataset: Dataset) -> tuple[list[str], list[Issue], list[Issue]]:
    """Summary lines, errors and warnings for an openai app config's passages_from."""
    if config.openai is None or config.openai.passages_from is None:
        return [], [], []
    where = config.path.name
    try:
        saved = load_passages(config.openai.passages_from)
    except PassagesError as error:
        return [], [Issue(where, str(error))], []
    errors, warnings = [], []
    if saved.dataset_sha256 and dataset.sha256 and saved.dataset_sha256 != dataset.sha256:
        errors.append(Issue(where, f"passages_from: {saved.source} used another version of the dataset, "
                                   "so its passages may not match these questions"))
    missing = [record.id for record in dataset.records if record.id not in saved.by_id]
    if missing:
        shown = ", ".join(missing[:SHOWN_IDS]) + (f" and {len(missing) - SHOWN_IDS} more"
                                                   if len(missing) > SHOWN_IDS else "")
        warnings.append(Issue(where, f"passages_from has no passages for {len(missing)} record(s), "
                                     f"so their {{{{passages}}}} is empty: {shown}"))
    found = len(dataset.records) - len(missing)
    return [f"Passages   from {saved.source}, for {found} of {len(dataset.records)} records"], errors, warnings
