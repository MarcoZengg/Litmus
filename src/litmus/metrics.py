"""Metrics that need no judge (docs/design.md, section 6).

Each metric scores every repeat of a question, averages the repeats into one
score per question, then averages the questions, so every question counts
equally however many repeats finished. Ranges (intervals) come with the
statistics in section 7.
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


def question_scores(records: list[Record], rows_by_id: dict[str, list[Row]], scorer: Scorer) -> dict[str, float]:
    """One score per question: the mean of its repeats that could be scored."""
    scores = {}
    for record in records:
        values = [v for v in (scorer(record, row) for row in rows_by_id.get(record.id, [])) if v is not None]
        if values:
            scores[record.id] = sum(values) / len(values)
    return scores


def applies(name: str, record: Record) -> bool:
    if name == "behavior_accuracy":
        return record.expected_behavior is not None
    if name == "unsafe_answer_rate":
        return record.expected_behavior in NOT_ANSWERED
    if name == "missed_answer_rate":
        return record.expected_behavior == "answer"
    return bool(record.reference_context_ids)


def question_metrics(records: list[Record], rows_by_id: dict[str, list[Row]]) -> list[Metric]:
    metrics = []
    for name, (scorer, lower) in QUESTION_METRICS.items():
        scores = question_scores(records, rows_by_id, scorer)
        total = sum(applies(name, record) for record in records)
        value = sum(scores.values()) / len(scores) if scores else None
        metrics.append(Metric(name, value, len(scores), total, lower_is_better=lower))
    return metrics


def call_metrics(rows: list[Row]) -> list[Metric]:
    errors = sum(row["error"] is not None for row in rows)
    latencies = [row["latency_ms"] for row in rows if row["error"] is None and row["latency_ms"] is not None]
    costs = [row["cost"] for row in rows if row["cost"] is not None]
    input_tokens = [row["input_tokens"] for row in rows if row["input_tokens"] is not None]
    output_tokens = [row["output_tokens"] for row in rows if row["output_tokens"] is not None]

    def percentile(q: float) -> float | None:
        return float(np.percentile(latencies, q)) if latencies else None

    def mean(values: list) -> float | None:
        return sum(values) / len(values) if values else None

    return [
        Metric("errors", errors / len(rows) if rows else None, len(rows), len(rows), "calls", lower_is_better=True),
        Metric("latency_p50_ms", percentile(50), len(latencies), len(rows), "calls", lower_is_better=True),
        Metric("latency_p95_ms", percentile(95), len(latencies), len(rows), "calls", lower_is_better=True),
        Metric("cost_per_1000", mean(costs) * 1000 if costs else None, len(costs), len(rows), "calls",
               lower_is_better=True),
        Metric("input_tokens_per_call", mean(input_tokens), len(input_tokens), len(rows), "calls"),
        Metric("output_tokens_per_call", mean(output_tokens), len(output_tokens), len(rows), "calls"),
    ]


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
              group_of: Callable[[Record], list[str]]) -> dict[str, dict[str, float | None]]:
    """Question metrics per group (by source or by tag), as group -> metric -> value."""
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        for group in group_of(record):
            groups[group].append(record)
    result = {}
    for group, members in sorted(groups.items()):
        values = {"questions": len(members)}
        for metric in question_metrics(members, rows_by_id):
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
