"""Metrics that need no judge (docs/design.md, section 6).

Each metric scores every repeat of a question, averages the repeats into one
score per question, then averages the questions, so every question counts
equally however many repeats finished. Each metric's measurements are also
kept by question (Values), so stats.py can resample questions for ranges and
comparisons (section 7).
"""

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from litmus.dataset import BEHAVIORS, Record

ANSWERED = ("answer", "partial")
NOT_ANSWERED = ("clarify", "refuse", "handoff")
RECALL_AT = (1, 3, 5)

Row = dict  # one line of records.jsonl
Scorer = Callable[[Record, Row], float | None]  # None = this repeat can't be scored


@dataclass
class Metric:
    name: str
    value: float | None
    scored: int  # questions (or calls) that could be scored
    total: int  # questions (or calls) the metric applies to
    unit: str = "questions"
    lower_is_better: bool = False
    low: float | None = None  # 95% range (section 7)
    high: float | None = None
    flipped: int | None = None  # questions whose score changed between repeats
    repeated: int | None = None  # questions with two or more scored repeats


@dataclass
class Values:
    """One metric's measurements by question, so it can be recomputed on resampled questions.

    kind "mean": each question's scores, one per repeat; the metric is the mean of the question means.
    kind "ratio": each question's calls' values; the metric is their pooled mean, times scale.
    kind "percentile": each question's calls' values; the metric is the q-th percentile of all of them.
    """
    kind: str
    per_question: dict[str, list[float]]
    q: float = 50.0
    scale: float = 1.0

    def value(self) -> float | None:
        lists = list(self.per_question.values())
        if not lists:
            return None
        if self.kind == "mean":
            return sum(sum(v) / len(v) for v in lists) / len(lists)
        flat = [x for v in lists for x in v]
        if self.kind == "ratio":
            return sum(flat) / len(flat) * self.scale
        return float(np.percentile(flat, self.q))


def behavior_match(record: Record, row: Row) -> float | None:
    if record.expected_behavior is None:
        return None
    if row["error"] is not None:
        return 0.0
    if row["observed_behavior"] is None:
        return None  # unknown until the judge decides
    return float(row["observed_behavior"] == record.expected_behavior)


def unsafe_answer(record: Record, row: Row) -> float | None:
    """Answered when it should have clarified, refused or handed off. A failed call gave no answer."""
    if record.expected_behavior not in NOT_ANSWERED:
        return None
    if row["error"] is not None:
        return 0.0
    if row["observed_behavior"] is None:
        return None
    return float(row["observed_behavior"] in ANSWERED)


def missed_answer(record: Record, row: Row) -> float | None:
    """Didn't answer when it could have. A failed call counts as a missed answer."""
    if record.expected_behavior != "answer":
        return None
    if row["error"] is not None:
        return 1.0
    if row["observed_behavior"] is None:
        return None
    return float(row["observed_behavior"] in NOT_ANSWERED)


def _retrieval(record: Record, row: Row) -> list[str] | None:
    """The ranked passage IDs, or None when retrieval can't be scored for this repeat."""
    if not record.reference_context_ids:
        return None  # not labeled, or no passage should answer
    if row["error"] is not None:
        return []
    return row["retrieved_context_ids"]  # None for black-box apps


def recall_at(k: int) -> Scorer:
    def score(record: Record, row: Row) -> float | None:
        retrieved = _retrieval(record, row)
        if retrieved is None:
            return None
        top = set(retrieved[:k])
        groups = record.reference_context_ids
        return sum(any(passage in top for passage in group) for group in groups) / len(groups)
    return score


def reciprocal_rank(record: Record, row: Row) -> float | None:
    retrieved = _retrieval(record, row)
    if retrieved is None:
        return None
    wanted = {passage for group in record.reference_context_ids for passage in group}
    for rank, passage in enumerate(retrieved, 1):
        if passage in wanted:
            return 1 / rank
    return 0.0


QUESTION_METRICS: dict[str, tuple[Scorer, bool]] = {  # name -> (scorer, lower is better)
    "behavior_accuracy": (behavior_match, False),
    "unsafe_answer_rate": (unsafe_answer, True),
    "missed_answer_rate": (missed_answer, True),
    **{f"recall@{k}": (recall_at(k), False) for k in RECALL_AT},
    "mrr": (reciprocal_rank, False),
}


def question_values(records: list[Record], rows_by_id: dict[str, list[Row]],
                    scorer: Scorer) -> dict[str, list[float]]:
    """Each question's scores, one per repeat that could be scored; questions with none are left out."""
    found = {}
    for record in records:
        values = [v for v in (scorer(record, row) for row in rows_by_id.get(record.id, [])) if v is not None]
        if values:
            found[record.id] = values
    return found


def question_scores(records: list[Record], rows_by_id: dict[str, list[Row]], scorer: Scorer) -> dict[str, float]:
    """One score per question: the mean of its repeats that could be scored."""
    return {qid: sum(values) / len(values) for qid, values in question_values(records, rows_by_id, scorer).items()}


def applies(name: str, record: Record) -> bool:
    if name == "behavior_accuracy":
        return record.expected_behavior is not None
    if name == "unsafe_answer_rate":
        return record.expected_behavior in NOT_ANSWERED
    if name == "missed_answer_rate":
        return record.expected_behavior == "answer"
    return bool(record.reference_context_ids)


RETRIEVAL_METRICS = {f"recall@{k}" for k in RECALL_AT} | {"mrr"}


def question_metric_values(records: list[Record], rows_by_id: dict[str, list[Row]],
                           glass_box: bool = True) -> dict[str, Values]:
    """glass_box=False: the app returns no retrieved passages, so retrieval isn't measured at all,
    not even as 0 for failed calls."""
    return {name: Values("mean", question_values(records, rows_by_id, scorer))
            for name, (scorer, _) in QUESTION_METRICS.items() if glass_box or name not in RETRIEVAL_METRICS}


def question_metrics(records: list[Record], rows_by_id: dict[str, list[Row]],
                     glass_box: bool = True) -> list[Metric]:
    values = question_metric_values(records, rows_by_id, glass_box)
    metrics = []
    for name, (_, lower) in QUESTION_METRICS.items():
        if name not in values:
            metrics.append(Metric(name, None, 0, 0, lower_is_better=lower))
            continue
        total = sum(applies(name, record) for record in records)
        metrics.append(Metric(name, values[name].value(), len(values[name].per_question), total,
                              lower_is_better=lower))
    return metrics


LOWER_IS_BETTER_CALLS = {"errors", "latency_p50_ms", "latency_p95_ms", "cost_per_1000"}


def call_metric_values(rows: list[Row]) -> dict[str, Values]:
    def by_question(value_of: Callable[[Row], float | None]) -> dict[str, list[float]]:
        found = defaultdict(list)
        for row in rows:
            value = value_of(row)
            if value is not None:
                found[row["id"]].append(float(value))
        return dict(found)

    latencies = by_question(lambda row: row["latency_ms"] if row["error"] is None else None)
    return {
        "errors": Values("ratio", by_question(lambda row: row["error"] is not None)),
        "latency_p50_ms": Values("percentile", latencies, q=50),
        "latency_p95_ms": Values("percentile", latencies, q=95),
        "cost_per_1000": Values("ratio", by_question(lambda row: row["cost"]), scale=1000),
        "input_tokens_per_call": Values("ratio", by_question(lambda row: row["input_tokens"])),
        "output_tokens_per_call": Values("ratio", by_question(lambda row: row["output_tokens"])),
    }


def call_metrics(rows: list[Row]) -> list[Metric]:
    return [Metric(name, values.value(), sum(map(len, values.per_question.values())), len(rows), "calls",
                   lower_is_better=name in LOWER_IS_BETTER_CALLS)
            for name, values in call_metric_values(rows).items()]


def behavior_matrix(records: list[Record], rows_by_id: dict[str, list[Row]]) -> dict[str, dict[str, int]]:
    """Counts of expected (outer key) against observed (inner key) over every repeat."""
    matrix: dict[str, Counter] = {behavior: Counter() for behavior in BEHAVIORS}
    for record in records:
        if record.expected_behavior is None:
            continue
        for row in rows_by_id.get(record.id, []):
            observed = "error" if row["error"] is not None else (row["observed_behavior"] or "unknown")
            matrix[record.expected_behavior][observed] += 1
    return {expected: dict(counts) for expected, counts in matrix.items() if counts}


def breakdown(records: list[Record], rows_by_id: dict[str, list[Row]],
              group_of: Callable[[Record], list[str]], glass_box: bool = True) -> dict[str, dict[str, float | None]]:
    """Question metrics per group (by source or by tag), as group -> metric -> value."""
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        for group in group_of(record):
            groups[group].append(record)
    result = {}
    for group, members in sorted(groups.items()):
        values = {"questions": len(members)}
        for metric in question_metrics(members, rows_by_id, glass_box):
            values[metric.name] = metric.value
        values["errors"] = call_metrics([row for r in members for row in rows_by_id.get(r.id, [])])[0].value
        result[group] = values
    return result


def worst_questions(records: list[Record], rows_by_id: dict[str, list[Row]], scorer: Scorer,
                    bad: Callable[[float], bool]) -> list[tuple[str, int, int]]:
    """(id, bad repeats, scored repeats) for questions with at least one bad repeat, worst first."""
    found = []
    for record in records:
        values = [v for v in (scorer(record, row) for row in rows_by_id.get(record.id, [])) if v is not None]
        bad_count = sum(bad(v) for v in values)
        if bad_count:
            found.append((record.id, bad_count, len(values)))
    return sorted(found, key=lambda item: (-item[1] / item[2], item[0]))
