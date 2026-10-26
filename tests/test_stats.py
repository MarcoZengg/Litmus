import json

import numpy as np
import pytest
import yaml

from litmus.metrics import Values
from litmus.report import build_report
from litmus.stats import _resamples, _statistic, flips, interval, paired_change
from test_report import ROWS, make_run


def brute_force(values, ids, counts):
    """The metric on each resample, by building the resampled data the slow way."""
    out = []
    for row in counts:
        chosen = {f"{qid}#{n}": values.per_question[qid] for qid, c in zip(ids, row) for n in range(c)}
        out.append(Values(values.kind, chosen, values.q, values.scale).value())
    return np.array(out)


@pytest.mark.parametrize("kind, q, scale", [("mean", 50, 1), ("ratio", 50, 1000), ("percentile", 50, 1),
                                           ("percentile", 95, 1), ("percentile", 0, 1), ("percentile", 100, 1)])
def test_resampled_statistics_match_brute_force(kind, q, scale):
    rng = np.random.default_rng(3)
    per_question = {f"q{i}": [float(x) for x in rng.integers(0, 500, size=rng.integers(1, 4))] for i in range(25)}
    values = Values(kind, per_question, q, scale)
    ids = list(per_question)
    counts = next(_resamples(len(ids), 1, seed=1))[:200]
    assert np.allclose(_statistic(values, ids, counts), brute_force(values, ids, counts))


def test_interval_brackets_the_value_and_is_reproducible():
    values = Values("mean", {f"q{i}": [float(i % 3 == 0)] for i in range(60)})
    low, high = interval(values, seed=42)
    assert low < values.value() < high
    assert interval(values, seed=42) == (low, high)
    assert 0.2 < high - low < 0.3  # about 2 x 1.96 x sqrt(p(1-p)/n) for p=1/3, n=60
    smooth = Values("mean", {f"q{i}": [i / 7 % 1] for i in range(60)})  # continuous, so the seed shows
    assert interval(smooth, seed=42) != interval(smooth, seed=43)


def test_interval_edge_cases():
    assert interval(Values("mean", {}), seed=1) is None
    assert interval(Values("mean", {"q1": [1.0], "q2": [1.0]}), seed=1) == (1.0, 1.0)


def test_paired_change_uses_questions_scored_in_both_runs():
    baseline = Values("mean", {"q1": [0.0], "q2": [0.0], "q3": [1.0]})
    candidate = Values("mean", {"q1": [1.0], "q2": [1.0], "q4": [0.0]})
    before, after, change, low, high, questions = paired_change(baseline, candidate, seed=[1, 2])
    assert (before, after, change, low, high, questions) == (0.0, 1.0, 1.0, 1.0, 1.0, 2)
    assert paired_change(baseline, Values("mean", {"q9": [1.0]}), seed=1) is None


def test_pairing_cancels_question_difficulty():
    # Hard and easy questions, and a candidate that's 0.1 better on every one of them.
    rng = np.random.default_rng(0)
    base = {f"q{i}": [float(x)] for i, x in enumerate(rng.uniform(0, 0.9, size=40))}
    cand = {qid: [scores[0] + 0.1] for qid, scores in base.items()}
    _, _, change, low, high, _ = paired_change(Values("mean", base), Values("mean", cand), seed=5)
    assert change == pytest.approx(0.1) and low == pytest.approx(0.1) and high == pytest.approx(0.1)
    separate = [interval(Values("mean", v), seed=5) for v in (base, cand)]
    assert separate[0][1] > separate[1][0]  # unpaired ranges overlap, so they couldn't show it


def test_flips():
    assert flips(Values("mean", {"a": [1.0, 0.0, 1.0], "b": [1.0, 1.0], "c": [0.0]})) == (1, 2)


def test_report_has_ranges_and_flips(x200):
    report = build_report(make_run(x200))
    m = {metric.name: metric for metric in report.metrics}
    behavior = m["behavior_accuracy"]
    assert behavior.low <= behavior.value <= behavior.high
    # q-0001 matched on repeats 1 and 2 and failed on 3; the other questions have one repeat.
    assert (behavior.flipped, behavior.repeated) == (1, 1)
    assert m["errors"].low == 0.0 and m["errors"].flipped is None
    assert m["cost_per_1000"].low is None
    saved = json.loads(json.dumps({k: v for k, v in vars(behavior).items()}))
    assert saved["high"] == behavior.high


def test_report_ranges_use_the_run_seed(x200):
    folder = make_run(x200)
    run = yaml.safe_load((folder / "run.yaml").read_text(encoding="utf-8"))
    run["seed"] = 12345
    (folder / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
    report = build_report(folder)
    for metric in report.metrics:
        if metric.low is not None:
            assert (metric.low, metric.high) == interval(report.values[metric.name], 12345)
