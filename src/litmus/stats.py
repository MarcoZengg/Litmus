"""Ranges, flip rates and paired comparisons by bootstrap (docs/design.md, section 7).

The bootstrap resamples whole questions, never single repeats or calls, because
the repeats and calls of one question aren't independent. A resample is kept as
how many times each question was drawn, so every metric kind can be recomputed
on it without copying data: means and ratios by matrix products, percentiles by
counting.
"""

from collections.abc import Iterator

import numpy as np

from litmus.metrics import Values

RESAMPLES = 10_000
LEVEL = 0.95
MAX_CELLS = 2_000_000  # memory bound for one chunk of resamples (resamples x calls)


def interval(values: Values, seed) -> tuple[float, float] | None:
    """The 95% range of a metric, or None when nothing was scored."""
    ids = list(values.per_question)
    if not ids:
        return None
    stats = [_statistic(values, ids, counts) for counts in _resamples(len(ids), _cells(values, ids), seed)]
    return _ends(np.concatenate(stats))


def paired_change(baseline: Values, candidate: Values, seed) -> tuple[float, float, float, float, float, int] | None:
    """(baseline, candidate, change, low, high, questions) over the questions both runs scored:
    each resample draws the same questions from both runs."""
    ids = [qid for qid in baseline.per_question if qid in candidate.per_question]
    if not ids:
        return None
    before, after = _subset(baseline, ids).value(), _subset(candidate, ids).value()
    cells = max(_cells(baseline, ids), _cells(candidate, ids))
    stats = [_statistic(candidate, ids, counts) - _statistic(baseline, ids, counts)
             for counts in _resamples(len(ids), cells, seed)]
    low, high = _ends(np.concatenate(stats))
    return before, after, after - before, low, high, len(ids)


def flips(values: Values) -> tuple[int, int]:
    """(questions whose score changed between repeats, questions with two or more scored repeats)."""
    repeated = [scores for scores in values.per_question.values() if len(scores) > 1]
    return sum(len(set(scores)) > 1 for scores in repeated), len(repeated)


def _subset(values: Values, ids: list[str]) -> Values:
    return Values(values.kind, {qid: values.per_question[qid] for qid in ids}, values.q, values.scale)


def _cells(values: Values, ids: list[str]) -> int:
    return sum(len(values.per_question[qid]) for qid in ids) if values.kind == "percentile" else len(ids)


def _resamples(n: int, cells: int, seed) -> Iterator[np.ndarray]:
    """RESAMPLES draws of n questions with replacement, as counts per question, in chunks."""
    rng = np.random.default_rng(seed)
    chunk = max(1, min(RESAMPLES, MAX_CELLS // max(cells, 1)))
    for start in range(0, RESAMPLES, chunk):
        yield rng.multinomial(n, np.full(n, 1 / n), size=min(chunk, RESAMPLES - start))


def _ends(stats: np.ndarray) -> tuple[float, float]:
    tail = (1 - LEVEL) / 2 * 100
    low, high = np.percentile(stats, [tail, 100 - tail])
    return float(low), float(high)


def _statistic(values: Values, ids: list[str], counts: np.ndarray) -> np.ndarray:
    """The metric on each resample (one row of counts per resample)."""
    lists = [values.per_question[qid] for qid in ids]
    if values.kind == "mean":
        means = np.array([sum(scores) / len(scores) for scores in lists])
        return counts @ means / counts.sum(axis=1)
    if values.kind == "ratio":
        sums = np.array([sum(calls) for calls in lists])
        sizes = np.array([len(calls) for calls in lists])
        return counts @ sums / (counts @ sizes) * values.scale
    # Percentile: np.percentile's default (linear) method on the resampled calls, without building them.
    flat = np.array([x for calls in lists for x in calls])
    owner = np.repeat(np.arange(len(lists)), [len(calls) for calls in lists])
    order = np.argsort(flat, kind="stable")
    flat, owner = flat[order], owner[order]
    cumulative = np.cumsum(counts[:, owner], axis=1)  # position of each sorted call in the resampled list
    position = values.q / 100 * (cumulative[:, -1] - 1)
    below = np.floor(position)
    # The k-th resampled call is the first sorted call whose cumulative count passes k.
    low = flat[(cumulative <= below[:, None]).sum(axis=1)]
    high = flat[(cumulative <= np.ceil(position)[:, None]).sum(axis=1)]
    return low + (position - below) * (high - low)
