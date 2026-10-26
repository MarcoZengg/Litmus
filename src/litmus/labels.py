"""Hand labels that check the judge (docs/design.md, section 7).

litmus label writes a sheet of the replies the judge grades for one metric, for a
person to label without seeing the judge's verdicts. Reports then compare those
labels with the judge's verdicts on the same replies, pooled over every run
judged with the same model and prompt.

The sheet holds question and answer text, so it stays in the run folder, next to
the dataset. Reports show only counts and rates from it.
"""

import csv
import io
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from litmus.dataset import BEHAVIORS, Dataset, Issue
from litmus.judge import CALL, decide, format_passages, load_judgments
from litmus.runner import RUN_FILE, load_run

SUGGESTED_LABELS = 50
KAPPA_WARNING = 0.6
VERDICTS = ("pass", "fail")
TEXT_COLUMNS = ("question", "reference", "passages", "answer")
FORMULA_START = ("=", "+", "-", "@")  # spreadsheet apps read cells starting with these as formulas


def labels_file(folder: Path, metric: str) -> Path:
    return folder / f"labels-{metric}.csv"


def label_values(metric: str) -> tuple[str, ...]:
    return BEHAVIORS if metric == "behavior" else VERDICTS


def pick_replies(dataset: Dataset, rows: list[dict], metric: str, glass_box: bool, sample: int,
                 seed: int) -> tuple[list[dict], int]:
    """A random sample of the replies the judge grades for a metric, taking one repeat of every
    question before a second one: (the sample in dataset order, how many replies were eligible)."""
    records = {record.id: record for record in dataset.records}
    eligible = [row for row in rows
                if row["id"] in records and decide(records[row["id"]], row, metric, glass_box) == CALL]
    first, more, seen = [], [], set()
    for row in random.Random(seed).sample(eligible, len(eligible)):
        (more if row["id"] in seen else first).append(row)
        seen.add(row["id"])
    order = {record.id: n for n, record in enumerate(dataset.records)}
    picked = sorted((first + more)[:sample], key=lambda row: (order[row["id"]], row["repeat"]))
    return picked, len(eligible)


def write_sheet(path: Path, dataset: Dataset, replies: list[dict], metric: str) -> None:
    """A CSV that spreadsheet apps open directly: UTF-8 with a byte-order mark, one row per reply."""
    records = {record.id: record for record in dataset.records}
    columns = ["id", "repeat", "question"]
    columns += {"correctness": ["reference"], "faithfulness": ["passages"]}.get(metric, [])
    columns += ["answer", "label", "note"]
    with open(path, "w", encoding="utf-8-sig", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(columns)
        for row in replies:
            record = records[row["id"]]
            cells = {"id": row["id"], "repeat": row["repeat"], "question": record.input,
                     "reference": record.expected_output or "", "passages": format_passages(row),
                     "answer": row["actual_output"] or "", "label": "", "note": ""}
            writer.writerow([_as_text(cells[c]) if c in TEXT_COLUMNS else cells[c] for c in columns])


def _as_text(value: str) -> str:
    # A leading apostrophe keeps "- step one" from turning into a broken formula.
    return "'" + value if value.startswith(FORMULA_START) else value


def load_labels(path: Path, metric: str) -> tuple[dict[tuple[str, int], str], list[Issue]]:
    """{(id, repeat): label} for the rows filled in, plus problems with the rest. Blank labels are skipped.
    Only id, repeat and label are read, so a sheet resaved in another encoding still works."""
    text = path.read_bytes().decode("utf-8-sig", errors="replace")
    header = text.split("\n", 1)[0]
    delimiter = max(",;\t", key=header.count)  # spreadsheet apps in some locales save with ';'
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
    where = f"{path.parent.name}/{path.name}"
    if not {"id", "repeat", "label"} <= set(reader.fieldnames or []):
        return {}, [Issue(where, "needs id, repeat and label columns")]
    allowed = label_values(metric)
    labels, issues = {}, []
    for number, line in enumerate(reader, 2):  # row numbers as a spreadsheet app shows them
        value = (line["label"] or "").strip().lower()
        if not value:
            continue
        if value not in allowed:
            issues.append(Issue(f"{where} row {number}", f"label '{value}' isn't {' or '.join(allowed)}; left out"))
            continue
        try:
            repeat = int(line["repeat"])
        except (TypeError, ValueError):
            issues.append(Issue(f"{where} row {number}", "repeat isn't a number; left out"))
            continue
        labels[((line["id"] or "").strip(), repeat)] = value
    return labels, issues


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Agreement beyond chance; None when chance agreement is already 1 (both sides gave one label only)."""
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(first[label] * second[label] for label in first) / n ** 2
    return None if expected == 1 else (observed - expected) / (1 - expected)


@dataclass
class Agreement:
    labels: int  # replies with both a hand label and a judge verdict
    agreement: float
    kappa: float | None
    lenient: int | None  # pass/fail metrics: the judge passed what you failed
    strict: int | None  # ... and failed what you passed
    runs: list[str]
    dataset_versions: list[str]


def judge_key(judging: dict, metric: str) -> tuple:
    return judging["provider"], judging["model"], (judging.get("prompt_sha256") or {}).get(metric)


def check_judge_metric(runs_folder: Path, judging: dict, metric: str) -> tuple[Agreement | None, list[Issue]]:
    """The judge's agreement with hand labels for one metric, over every run in the folder that
    has labels and was judged with the same provider, model and prompt."""
    key = judge_key(judging, metric)
    pairs, runs, versions, issues = [], [], [], []
    for folder in sorted(path for path in runs_folder.iterdir() if (path / RUN_FILE).is_file()):
        path = labels_file(folder, metric)
        other, judgments = load_judgments(folder)
        if not path.is_file() or other is None or judge_key(other, metric) != key:
            continue
        labels, problems = load_labels(path, metric)
        issues += problems
        verdicts = {(j["id"], j["repeat"]): j["verdict"] for j in judgments
                    if j["metric"] == metric and j["source"] == "judge" and j["error"] is None}
        found = [(labels[k], verdicts[k]) for k in labels if k in verdicts]
        if found:
            pairs += found
            run = load_run(folder)
            runs.append(run["name"])
            version = str(run["dataset"].get("version", "unknown"))
            if version not in versions:
                versions.append(version)
    if not pairs:
        return None, issues
    pass_fail = metric != "behavior"
    return Agreement(
        labels=len(pairs),
        agreement=sum(mine == judge for mine, judge in pairs) / len(pairs),
        kappa=cohen_kappa(pairs),
        lenient=sum(pair == ("fail", "pass") for pair in pairs) if pass_fail else None,
        strict=sum(pair == ("pass", "fail") for pair in pairs) if pass_fail else None,
        runs=runs,
        dataset_versions=versions,
    ), issues
