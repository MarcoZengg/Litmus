"""Turn one run into metrics: printed for people, saved as metrics.json for tools.

Both show numbers and question IDs only, never question or answer text, so
they are safe for private datasets. metrics.json leaves out IDs too, so it can
be shared as aggregate numbers.
"""

import json
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


from litmus.dataset import BEHAVIORS, SOURCES, Dataset, Issue, load_dataset
from litmus.judge import load_judgments
from litmus.labels import KAPPA_WARNING, SUGGESTED_LABELS, Agreement, check_judge_metric
from litmus.metrics import (RETRIEVAL_METRICS, Metric, Values, behavior_matrix, behavior_match, breakdown,
                            call_metric_values, call_metrics, missed_answer, question_metric_values, question_metrics,
                            question_values, unsafe_answer, worst_questions)
from litmus.runner import RECORDS_FILE, RUN_FILE, is_glass_box, load_rows, load_run
from litmus.stats import flips, interval

METRICS_FILE = "metrics.json"
SHOWN_IDS = 10
JUDGE_SCORED = ("correctness", "faithfulness", "rubric")


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
    judging: dict | None = None
    judge_tokens: tuple[int, int] = (0, 0)
    judge_checks: dict[str, Agreement | None] = field(default_factory=dict)
    # Each metric's measurements by question ID, for litmus compare. Never saved: it holds IDs.
    values: dict[str, Values] = field(default_factory=dict, repr=False)


def build_report(folder: Path) -> Report:
    if not (folder / RUN_FILE).is_file() or not (folder / RECORDS_FILE).is_file():
        raise ReportError(f"{folder} isn't a run folder: it needs {RUN_FILE} and {RECORDS_FILE}")
    run = load_run(folder)
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

    rows = load_rows(folder)
    known = {record.id for record in dataset.records}
    unknown = {row["id"] for row in rows} - known
    if unknown:
        warnings.append(Issue("dataset", f"{len(unknown)} question(s) in the run are no longer in the dataset "
                                         "and were left out"))
    rows = [row for row in rows if row["id"] in known]

    judging, judgments = load_judgments(folder)
    judge_checks = {}
    if judging is not None:
        warnings += _judge_warnings(judging, judgments)
        rows = _with_judged_behavior(rows, judgments)
        judge_checks, check_warnings = _judge_checks(folder.parent, judging, judgments)
        warnings += check_warnings
    rows_by_id = defaultdict(list)
    for row in rows:
        rows_by_id[row["id"]].append(row)

    records = dataset.records
    glass_box = is_glass_box(run)
    scorers = _judge_scorers(judgments)
    values = question_metric_values(records, rows_by_id, glass_box)
    judge_metrics = []
    for metric in JUDGE_SCORED:
        if judging is not None and metric in judging["metrics"]:
            values[metric] = Values("mean", question_values(records, rows_by_id, scorers[metric]))
            applied = len({j["id"] for j in judgments if j["metric"] == metric})
            judge_metrics.append(Metric(metric, values[metric].value(), len(values[metric].per_question), applied))
    values.update(call_metric_values(rows))
    metrics = question_metrics(records, rows_by_id, glass_box) + judge_metrics + call_metrics(rows)
    _add_ranges(metrics, values, run.get("seed") or 0)
    to_check = {
        "Missed answers": worst_questions(records, rows_by_id, missed_answer, lambda v: v == 1),
        "Unsafe answers": worst_questions(records, rows_by_id, unsafe_answer, lambda v: v == 1),
        "Wrong behavior": worst_questions(records, rows_by_id, behavior_match, lambda v: v == 0),
    }
    if any(metric.name == "correctness" for metric in judge_metrics):
        to_check["Incorrect answers"] = worst_questions(records, rows_by_id, scorers["correctness"], lambda v: v == 0)
    return Report(
        run=run,
        dataset=dataset,
        metrics=metrics,
        matrix=behavior_matrix(records, rows_by_id),
        by_source=breakdown(records, rows_by_id, lambda r: [r.source or "unlabeled"], glass_box),
        by_tag=breakdown(records, rows_by_id, lambda r: r.tags, glass_box),
        errors_by_type=dict(Counter(row["error"]["type"] for row in rows if row["error"] is not None)),
        to_check=to_check,
        warnings=warnings,
        judging=judging,
        judge_tokens=(sum(j["input_tokens"] or 0 for j in judgments), sum(j["output_tokens"] or 0 for j in judgments)),
        judge_checks=judge_checks,
        values=values,
    )


def _add_ranges(metrics: list[Metric], values: dict[str, Values], seed: int) -> None:
    """95% ranges for every metric, and flip counts for the ones scored per repeat. The same seed for
    every metric, so a report can be reproduced exactly."""
    for metric in metrics:
        measured = values.get(metric.name)
        if measured is None or metric.value is None:
            continue
        metric.low, metric.high = interval(measured, seed)
        if measured.kind == "mean":
            metric.flipped, metric.repeated = flips(measured)


def _judge_checks(runs_folder: Path, judging: dict, judgments: list[dict]
                  ) -> tuple[dict[str, Agreement | None], list[Issue]]:
    """Hand-label agreement for each judge metric the judge actually graded in this run."""
    graded = {j["metric"] for j in judgments if j["source"] == "judge"}
    checks, warnings = {}, []
    for metric in judging["metrics"]:
        if metric in graded:
            checks[metric], problems = check_judge_metric(runs_folder, judging, metric)
            warnings += problems
    unchecked = [metric for metric, check in checks.items() if check is None]
    if unchecked:
        warnings.append(Issue("judge", f"{', '.join(unchecked)}: this judge model and prompt haven't been checked "
                                       "against hand labels; see litmus label"))
    for metric, check in checks.items():
        if check is not None and check.kappa is not None and check.kappa < KAPPA_WARNING:
            warnings.append(Issue("judge", f"{metric}: the judge agrees poorly with your labels (kappa "
                                           f"{check.kappa:.2f}), so treat {metric} as rough"))
    return checks, warnings


def _judge_warnings(judging: dict, judgments: list[dict]) -> list[Issue]:
    warnings = []
    if judging["status"] != "complete":
        warnings.append(Issue("judge", f"judging is incomplete ({judging.get('stop_reason') or 'unknown reason'}); "
                                       "judge metrics cover only what was judged"))
    failed = sum(j["error"] is not None for j in judgments)
    if failed:
        warnings.append(Issue("judge", f"{failed} judge call(s) failed; those repeats aren't scored"))
    return warnings


def _with_judged_behavior(rows: list[dict], judgments: list[dict]) -> list[dict]:
    """Fill in behavior the app's status didn't give, from the judge."""
    judged = {(j["id"], j["repeat"]): j["verdict"] for j in judgments
              if j["metric"] == "behavior" and j["verdict"] in BEHAVIORS}
    filled = []
    for row in rows:
        key = (row["id"], row["repeat"])
        if row["error"] is None and row["observed_behavior"] is None and key in judged:
            row = {**row, "observed_behavior": judged[key], "behavior_source": "judge"}
        filled.append(row)
    return filled


def _judge_scorers(judgments: list[dict]) -> dict:
    verdicts = {(j["id"], j["repeat"], j["metric"]): j["verdict"] for j in judgments if j["verdict"] in ("pass", "fail")}

    def scorer_for(metric: str):
        def score(record, row) -> float | None:
            verdict = verdicts.get((record.id, row["repeat"], metric))
            return None if verdict is None else float(verdict == "pass")
        return score
    return {metric: scorer_for(metric) for metric in JUDGE_SCORED}


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
        "judge": None if report.judging is None else {
            key: report.judging[key] for key in ("provider", "model", "status", "prompt_sha256", "counts")},
        "judge_checks": {metric: None if check is None else asdict(check)
                         for metric, check in report.judge_checks.items()},
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


def format_range(metric_name: str, low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return ""
    low_text, high_text = format_value(metric_name, low), format_value(metric_name, high)
    for unit in ("%", " ms"):
        if low_text.endswith(unit) and high_text.endswith(unit):
            return f"{low_text[:-len(unit)]}-{high_text}"
    return f"{low_text}-{high_text}"


def _width(text: str) -> int:
    """Terminal width: Chinese, Japanese and Korean characters take two columns."""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _pad(text: str, size: int, right: bool = False) -> str:
    fill = " " * max(0, size - _width(text))
    return fill + text if right else text + fill


GROUP_COLUMNS = (("behavior_accuracy", "behavior"), ("unsafe_answer_rate", "unsafe"), ("missed_answer_rate", "missed"),
                 ("recall@5", "recall@5"), ("mrr", "mrr"), ("errors", "errors"))


def _group_table(title: str, groups: dict[str, dict], order: tuple[str, ...], glass_box: bool) -> list[str]:
    """One row per group, so many groups (or long names) stay readable."""
    names = [n for n in order if n in groups] + sorted(set(groups) - set(order))
    columns = [(key, label) for key, label in GROUP_COLUMNS if glass_box or key not in RETRIEVAL_METRICS]
    name_width = max(24, *(_width(n) + 2 for n in names))
    lines = ["", title, _pad("", name_width) + _pad("questions", 11, right=True)
             + "".join(_pad(label, 10, right=True) for _, label in columns)]
    for name in names:
        values = groups[name]
        lines.append(_pad(name, name_width) + _pad(str(values["questions"]), 11, right=True)
                     + "".join(_pad(format_value(key, values[key]), 10, right=True) for key, _ in columns))
    return lines


def _check_lines(metric: str, check: Agreement | None) -> list[str]:
    if check is None:
        return [f"  {metric:<22}not checked"]
    kappa = "n/a" if check.kappa is None else f"{check.kappa:.2f}"
    few = f" (aim for {SUGGESTED_LABELS})" if check.labels < SUGGESTED_LABELS else ""
    line = f"  {metric:<22}{check.agreement * 100:.1f}% agreement, kappa {kappa}, {check.labels} labels{few}"
    if check.lenient is not None:
        line += f"; judge too lenient {check.lenient}, too strict {check.strict}"
    versions = ", ".join(check.dataset_versions)
    return [line, f"  {'':<22}labels from {', '.join(check.runs)} (dataset version {versions})"]


def format_report(report: Report) -> list[str]:
    run, manifest = report.run, report.dataset.manifest
    glass_box = is_glass_box(run)
    lines = [
        f"Run        {run['name']} ({run['status']}, {run['repeats']} repeats, {run['counts']['written']} calls)",
        f"Dataset    {manifest.name} (version {manifest.version})",
        f"App        {run['app']['config'].get('name', run['app']['file'])}",
    ]
    if not glass_box:
        lines.append("Retrieval  not measured: the app config doesn't read retrieved passages (black-box)")
    judging = report.judging
    if judging is None:
        lines.append("Judge      none: run litmus judge for correctness and the other judge metrics")
    else:
        counts = judging["counts"]
        tokens_in, tokens_out = report.judge_tokens
        lines.append(f"Judge      {judging['model']} ({judging['provider']}): {counts['judged']} judged, "
                     f"{counts['errors']} failed, {counts['automatic']} decided without a call; "
                     f"{tokens_in} tokens in, {tokens_out} out")
    show_flips = any(metric.repeated for metric in report.metrics)
    lines += ["", f"{'Metric':<24}{'Value':>10}{'95% range':>17}" + (f"{'Flips':>8}" if show_flips else "")
              + "   Coverage"]
    for metric in report.metrics:
        if metric.name in RETRIEVAL_METRICS and not glass_box:
            continue
        note = "  (lower is better)" if metric.lower_is_better else ""
        coverage = f"{metric.scored} of {metric.total} {metric.unit}"
        flipped = f"{metric.flipped / metric.repeated * 100:.1f}%" if metric.repeated else ""
        lines.append(f"{metric.name:<24}{format_value(metric.name, metric.value):>10}"
                     f"{format_range(metric.name, metric.low, metric.high):>17}"
                     + (f"{flipped:>8}" if show_flips else "") + f"   {coverage}{note}")
    if show_flips:
        lines.append("Flips: the share of questions with 2+ scored repeats whose score changed between repeats.")
    if report.errors_by_type:
        lines.append("Errors by type: " + ", ".join(f"{k} {v}" for k, v in sorted(report.errors_by_type.items())))
    if report.judge_checks:
        lines += ["", "Judge check (your hand labels vs this judge model and prompt)"]
        for metric, check in report.judge_checks.items():
            lines += _check_lines(metric, check)

    if report.matrix:
        columns = [*BEHAVIORS, "unknown", "error"]
        lines += ["", "Behavior (rows: expected, columns: observed, every repeat)",
                  " " * 10 + "".join(f"{c:>9}" for c in columns)]
        for expected in BEHAVIORS:
            if expected in report.matrix:
                row = report.matrix[expected]
                lines.append(f"{expected:<10}" + "".join(f"{row.get(c, 0):>9}" for c in columns))

    for title, groups, order in (("By source", report.by_source, SOURCES), ("By tag", report.by_tag, ())):
        if groups and not (title == "By source" and len(groups) == 1):
            lines += _group_table(title, groups, order, glass_box)

    flagged = {title: items for title, items in report.to_check.items() if items}
    if flagged:
        lines += ["", "Questions to look at (IDs only; bad repeats of scored repeats)"]
        width = max(len(title) for title in flagged) + 3  # the colon and two spaces
        for title, items in flagged.items():
            shown_items = ", ".join(f"{qid} ({bad}/{of})" for qid, bad, of in items[:SHOWN_IDS])
            more = f" and {len(items) - SHOWN_IDS} more" if len(items) > SHOWN_IDS else ""
            lines.append(f"  {title + ':':<{width}}{shown_items}{more}")
    return lines
