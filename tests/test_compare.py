import json

import pytest
import yaml
from typer.testing import CliRunner

from litmus.cli import app
from litmus.compare import Change, load_gates
from litmus.dataset import load_dataset
from test_report import GLASS_BOX, ok, row

runner = CliRunner()

# The X200 examples: q-0001 and q-0002 expect answer, q-0003 clarify, q-0004 handoff.
GOOD = [row("q-0001", 1, **ok("answer", ["x200-manual#p12"], 100)),
        row("q-0002", 1, **ok("answer", ["x200-manual#p4", "x200-manual#p15"], 100)),
        row("q-0003", 1, **ok("clarify", None, 100)),
        row("q-0004", 1, **ok("handoff", None, 100))]
BAD = [row("q-0001", 1, **ok("handoff", [], 100)),
       row("q-0002", 1, **ok("handoff", [], 100)),
       row("q-0003", 1, **ok("answer", None, 100)),
       row("q-0004", 1, **ok("answer", None, 100))]


def slower(rows, ms):
    return [{**r, "latency_ms": r["latency_ms"] + ms} for r in rows]


def make_run(x200, name, rows, seed=1, sha256=None, correctness_prompt=None):
    folder = x200 / "runs" / f"2026-10-23_{name}"
    folder.mkdir(parents=True)
    (folder / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    run = {"name": name, "status": "complete", "stop_reason": None, "repeats": 1, "seed": seed,
           "dataset": {"sha256": sha256 or load_dataset(x200).sha256, "version": 1},
           "app": {"file": "app.yaml", "config": {"name": "x200-support-bot", "response": GLASS_BOX}},
           "counts": {"written": len(rows)}}
    (folder / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
    if correctness_prompt is not None:  # judged: correctness passes when the app answered
        judging = {"status": "complete", "provider": "example-provider", "model": "judge-large",
                   "metrics": ["correctness"], "prompt_sha256": {"correctness": correctness_prompt},
                   "counts": {"judge_calls": 2, "judged": 2, "errors": 0, "automatic": 0}}
        (folder / "judging.yaml").write_text(yaml.safe_dump(judging), encoding="utf-8")
        lines = [{"id": r["id"], "repeat": r["repeat"], "metric": "correctness", "source": "judge",
                  "verdict": "pass" if r["observed_behavior"] == "answer" else "fail", "reason": "r",
                  "input_tokens": 10, "output_tokens": 2, "error": None}
                 for r in rows if r["id"] in ("q-0001", "q-0002")]
        (folder / "judgments.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
    return folder


def compare(*args):
    return runner.invoke(app, ["compare", *map(str, args)])


def table(output):
    """{metric: verdict} from the comparison table."""
    lines = output.split("Change (95% range)")[1].split("\n\n")[0].splitlines()[1:]
    return {line.split()[0]: line[80:].strip() for line in lines}


def write_gates(x200, text):
    path = x200 / "gates.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --- verdicts ---

@pytest.mark.parametrize("change, low, high, lower, verdict", [
    (-0.10, -0.15, -0.05, False, "regression"),
    (-0.01, -0.015, -0.005, False, "small but real drop"),
    (0.01, -0.02, 0.04, False, "no detectable change"),
    (0.01, 0.005, 0.015, False, "small but real gain"),
    (0.10, 0.05, 0.15, False, "improvement"),
    (0.10, 0.05, 0.15, True, "regression"),  # up is worse for errors, latency, cost...
    (-0.10, -0.15, -0.05, True, "improvement"),
    (-0.02, -0.03, -0.01, False, "regression"),  # exactly the minimum change
])
def test_verdicts(change, low, high, lower, verdict):
    assert Change("m", 0.5, 0.5 + change, change, low, high, 70, 0.02, lower).verdict == verdict


def test_a_run_compared_with_itself_has_no_change(x200):
    folder = make_run(x200, "baseline", GOOD)
    result = compare(folder, folder)
    assert result.exit_code == 0, result.output
    assert set(table(result.output).values()) == {"no detectable change"}
    assert "+0.0 pts (+0.0 to +0.0)" in result.output


def test_regression_and_improvement(x200):
    good, bad = make_run(x200, "good", GOOD), make_run(x200, "bad", BAD, seed=2)
    worse = table(compare(good, bad).output)
    assert worse["behavior_accuracy"] == worse["unsafe_answer_rate"] == worse["missed_answer_rate"] == "regression"
    assert worse["latency_p50_ms"] == "no detectable change"
    better = compare(bad, good).output
    assert table(better)["behavior_accuracy"] == "improvement"
    assert "-100.0 pts (-100.0 to -100.0)" in better  # unsafe answers went from all to none
    assert "q-000" not in better  # numbers only


def test_small_but_real_drop_in_latency(x200):
    # Every call 5 ms slower: certainly worse, but less than the default 10% of the baseline.
    base, slow = make_run(x200, "base", GOOD), make_run(x200, "slow", slower(GOOD, 5))
    output = compare(base, slow).output
    assert table(output)["latency_p50_ms"] == "small but real drop"
    assert "+5 ms (+5 to +5)" in output


def test_comparisons_are_reproducible(x200):
    base = make_run(x200, "base", GOOD + [row("q-0001", 2, **ok("handoff", [], 300))])
    cand = make_run(x200, "cand", BAD[:2] + GOOD[2:], seed=5)
    assert compare(base, cand).output == compare(base, cand).output


# --- comparability ---

def test_different_datasets_need_force(x200):
    base, cand = make_run(x200, "base", GOOD), make_run(x200, "cand", GOOD, sha256="0" * 64)
    result = compare(base, cand)
    assert result.exit_code == 1
    assert "the runs used different versions of dataset.jsonl" in result.output
    assert "Not comparable. Add --force to compare anyway." in result.output
    forced = compare(base, cand, "--force")
    assert forced.exit_code == 0 and "(compared anyway, --force)" in forced.output


def test_different_judge_prompts_need_force(x200):
    base = make_run(x200, "base", GOOD, correctness_prompt="a" * 64)
    cand = make_run(x200, "cand", BAD, correctness_prompt="b" * 64)
    result = compare(base, cand)
    assert result.exit_code == 1
    assert "correctness was judged with a different provider, model or prompt in each run" in result.output


def test_judge_metrics_compare_when_both_runs_used_the_same_judge(x200):
    base = make_run(x200, "base", GOOD, correctness_prompt="a" * 64)
    cand = make_run(x200, "cand", BAD, correctness_prompt="a" * 64)
    assert table(compare(base, cand).output)["correctness"] == "regression"
    unjudged = make_run(x200, "unjudged", BAD)
    output = compare(base, unjudged).output
    assert "correctness" not in table(output)
    assert "only the baseline run was judged, so judge metrics aren't compared" in output


# --- gates ---

def test_gates_pass(x200):
    gates = write_gates(x200, "gates:\n  behavior_accuracy: {}\n  errors: {max: 0.05}\n")
    folder = make_run(x200, "base", GOOD)
    result = compare(folder, folder, "--gates", gates)
    assert result.exit_code == 0, result.output
    assert "  behavior_accuracy       pass  no detectable change (a regression needs a worsening of at least 2.0 pts)" \
        in result.output
    assert "  errors                  pass  0.0% is within the cap of 5.0%" in result.output
    assert "Result     PASS: 0 of 2 gate(s) failed" in result.output


def test_gates_fail_on_regression_cap_or_missing_metric(x200):
    gates = write_gates(x200, "gates:\n  behavior_accuracy: {}\n  errors: {max: 0.05}\n  correctness: {}\n")
    errored = BAD[:3] + [row("q-0004", 1, error={"type": "timeout", "message": "no reply"})]
    result = compare(make_run(x200, "good", GOOD), make_run(x200, "bad", errored), "--gates", gates)
    assert result.exit_code == 1
    assert "  behavior_accuracy       FAIL  regression of -100.0 pts (range -100.0 to -100.0)" in result.output
    assert "  errors                  FAIL  25.0% is above the cap of 5.0%" in result.output
    assert "  correctness             FAIL  not measured in both runs" in result.output
    assert "Result     FAIL: 3 of 3 gate(s) failed" in result.output


def test_gate_min_effect_turns_a_small_drop_into_a_regression(x200):
    base, slow = make_run(x200, "base", GOOD), make_run(x200, "slow", slower(GOOD, 5))
    gates = write_gates(x200, "gates:\n  latency_p50_ms: {min_effect: 4}\n")
    result = compare(base, slow, "--gates", gates)
    assert result.exit_code == 1
    assert table(result.output)["latency_p50_ms"] == "regression"
    assert "regression of +5 ms (range +5 to +5)" in result.output


@pytest.mark.parametrize("text, expected", [
    ("correctness: {}\n", "needs a 'gates' mapping"),
    ("gates:\n  accuracy: {}\n", "'accuracy' isn't a metric Litmus reports"),
    ("gates:\n  errors: {cap: 0.05}\n", "errors: unknown setting 'cap'"),
    ("gates:\n  errors: {max: lots}\n", "errors: 'max' must be a number"),
    ("gates:\n  correctness: {min_effect: -1}\n", "correctness: 'min_effect' must be a number of 0 or more"),
    ("gates:\n  correctness: 0.02\n", "correctness: use a mapping"),
    ("gates: [\n", "isn't valid YAML"),
])
def test_gate_file_errors(tmp_path, text, expected):
    path = tmp_path / "gates.yaml"
    path.write_text(text, encoding="utf-8")
    gates, issues = load_gates(path)
    assert any(expected in str(issue) for issue in issues), [str(i) for i in issues]


def test_bad_gate_file_stops_compare(x200):
    folder = make_run(x200, "base", GOOD)
    result = compare(folder, folder, "--gates", write_gates(x200, "gates:\n  accuracy: {}\n"))
    assert result.exit_code == 1 and "Not compared: the gates file has 1 error(s)." in result.output
