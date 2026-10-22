"""Turn one run into metrics: printed for people, saved as metrics.json for tools.

Both show numbers and question IDs only, never question or answer text, so
they are safe for private datasets. metrics.json leaves out IDs too, so it can
be shared as aggregate numbers.
"""

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from litmus.dataset import BEHAVIORS, SOURCES, Dataset, Issue, load_dataset
from litmus.metrics import (Metric, behavior_matrix, behavior_match, breakdown, call_metrics, missed_answer,
                            question_metrics, unsafe_answer, worst_questions)
from litmus.runner import RECORDS_FILE, RUN_FILE

METRICS_FILE = "metrics.json"
SHOWN_IDS = 10


class ReportError(Exception):
    pass


@dataclass
class Report:
    run: dict
    dataset: Dataset
    metrics: list[Metric]
    matrix: dict[str, dict[str, int]]
    by_source: dict[str, dict]
    by_tag: dict[str, dict]
    errors_by_type: dict[str, int]
    to_check: dict[str, list[tuple[str, int, int]]]
    warnings: list[Issue] = field(default_factory=list)


def build_report(folder: Path) -> Report:
    run_file, records_file = folder / RUN_FILE, folder / RECORDS_FILE
    if not run_file.is_file() or not records_file.is_file():
        raise ReportError(f"{folder} isn't a run folder: it needs {RUN_FILE} and {RECORDS_FILE}")
    run = yaml.safe_load(run_file.read_text(encoding="utf-8"))
    dataset = load_dataset(folder.parent.parent)  # runs live in <dataset>/runs/<run>/
    if not dataset.ok:
        raise ReportError(f"the run's dataset ({dataset.folder}) has errors; fix them with litmus validate")

    warnings = []
    if dataset.sha256 != run["dataset"]["sha256"]:
        warnings.append(Issue("dataset", "the dataset changed since this run, so the labels used here may "
                                         "differ from the ones the run was made against"))
    if run["status"] != "complete":
        warnings.append(Issue("run", f"the run is incomplete ({run.get('stop_reason') or 'unknown reason'}); "
                                     "metrics cover only the calls that finished"))

    rows = [json.loads(line) for line in records_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    known = {record.id for record in dataset.records}
    unknown = {row["id"] for row in rows} - known
    if unknown:
        warnings.append(Issue("dataset", f"{len(unknown)} question(s) in the run are no longer in the dataset "
                                         "and were left out"))
    rows = [row for row in rows if row["id"] in known]
    rows_by_id = defaultdict(list)
    for row in rows:
        rows_by_id[row["id"]].append(row)

    records = dataset.records
    return Report(
        run=run,
        dataset=dataset,
        metrics=question_metrics(records, rows_by_id) + call_metrics(rows),
        matrix=behavior_matrix(records, rows_by_id),
        by_source=breakdown(records, rows_by_id, lambda r: [r.source or "unlabeled"]),
        by_tag=breakdown(records, rows_by_id, lambda r: r.tags),
        errors_by_type=dict(Counter(row["error"]["type"] for row in rows if row["error"] is not None)),
        to_check={
            "Missed answers": worst_questions(records, rows_by_id, missed_answer, lambda v: v == 1),
            "Unsafe answers": worst_questions(records, rows_by_id, unsafe_answer, lambda v: v == 1),
            "Wrong behavior": worst_questions(records, rows_by_id, behavior_match, lambda v: v == 0),
        },
        warnings=warnings,
    )


def save_metrics(report: Report, folder: Path) -> Path:
    """Aggregate numbers only: no question IDs or text."""
    data = {
        "run": report.run["name"],
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_sha256": report.dataset.sha256,
        "dataset_matches_run": report.dataset.sha256 == report.run["dataset"]["sha256"],
        "metrics": {m.name: {k: v for k, v in asdict(m).items() if k != "name"} for m in report.metrics},
        "behavior_matrix": report.matrix,
        "by_source": report.by_source,
        "by_tag": report.by_tag,
        "errors_by_type": report.errors_by_type,
    }
    path = folder / METRICS_FILE
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def format_value(metric_name: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    if metric_name.startswith("latency"):
        return f"{value:.0f} ms"
    if metric_name == "mrr" or metric_name.endswith("per_call"):
        return f"{value:.2f}"
    if metric_name == "cost_per_1000":
        return f"{value:.4g}"
    return f"{value * 100:.1f}%"


def format_report(report: Report) -> list[str]:
    run, manifest = report.run, report.dataset.manifest
    lines = [
        f"Run        {run['name']} ({run['status']}, {run['repeats']} repeats, {run['counts']['written']} calls)",
        f"Dataset    {manifest.name} (version {manifest.version})",
        f"App        {run['app']['config'].get('name', run['app']['file'])}",
        "",
        f"{'Metric':<24}{'Value':>10}   Coverage",
    ]
    for metric in report.metrics:
        note = "  (lower is better)" if metric.lower_is_better else ""
        coverage = f"{metric.scored} of {metric.total} {metric.unit}"
        lines.append(f"{metric.name:<24}{format_value(metric.name, metric.value):>10}   {coverage}{note}")
    if report.errors_by_type:
        lines.append("Errors by type: " + ", ".join(f"{k} {v}" for k, v in sorted(report.errors_by_type.items())))

    if report.matrix:
        columns = [*BEHAVIORS, "unknown", "error"]
        lines += ["", "Behavior (rows: expected, columns: observed, every repeat)",
                  " " * 10 + "".join(f"{c:>9}" for c in columns)]
        for expected in BEHAVIORS:
            if expected in report.matrix:
                row = report.matrix[expected]
                lines.append(f"{expected:<10}" + "".join(f"{row.get(c, 0):>9}" for c in columns))

    for title, groups, order in (("By source", report.by_source, SOURCES), ("By tag", report.by_tag, ())):
        if not groups or (title == "By source" and len(groups) == 1):
            continue
        names = [n for n in order if n in groups] + sorted(set(groups) - set(order))
        shown = ("behavior_accuracy", "unsafe_answer_rate", "missed_answer_rate", "recall@5", "mrr", "errors")
        lines += ["", title, f"{'':<24}" + "".join(f"{n[:14]:>15}" for n in names),
                  f"{'questions':<24}" + "".join(f"{groups[n]['questions']:>15}" for n in names)]
        for metric in shown:
            lines.append(f"{metric:<24}" + "".join(f"{format_value(metric, groups[n][metric]):>15}" for n in names))

    flagged = {title: items for title, items in report.to_check.items() if items}
    if flagged:
        lines += ["", "Questions to look at (IDs only; bad repeats of scored repeats)"]
        for title, items in flagged.items():
            shown_items = ", ".join(f"{qid} ({bad}/{of})" for qid, bad, of in items[:SHOWN_IDS])
            more = f" and {len(items) - SHOWN_IDS} more" if len(items) > SHOWN_IDS else ""
            lines.append(f"  {title + ':':<17}{shown_items}{more}")
    return lines
