import json

import pytest
import yaml
from typer.testing import CliRunner

from litmus.adapters import AppResponse
from litmus.cli import app
from litmus.dataset import load_dataset
from litmus.report import build_report

runner = CliRunner()


def row(qid, repeat, **fields):
    return {"id": qid, "repeat": repeat, **{**AppResponse().to_dict(), **fields}}


def ok(behavior, retrieved=None, latency=100.0, **fields):
    return {"observed_behavior": behavior, "behavior_source": "status", "actual_output": "text",
            "retrieved_context_ids": retrieved, "latency_ms": latency, **fields}


# Hand-checked expectations are in each test. The X200 examples are:
# q-0001 answer, passages [[p12]]            (real_traffic, tags battery)
# q-0002 answer, passages [[p4, faq#7], [p15]] (staff, tags noise, maintenance)
# q-0003 clarify, passages []                (real_traffic)
# q-0004 handoff, passages []                (staff)
ROWS = [
    row("q-0001", 1, **ok("answer", ["x200-manual#p12", "x200-manual#p4"], 100)),
    row("q-0001", 2, **ok("answer", ["x200-manual#p4", "x200-manual#p12"], 200)),
    row("q-0001", 3, error={"type": "timeout", "message": "no reply within 60 s"}),
    row("q-0002", 1, **ok("partial", ["x200-faq#7", "x200-manual#p9", "x200-manual#p15"], 300)),
    row("q-0003", 1, **ok("answer", None, 400)),
    row("q-0004", 1, **ok(None, None, 500, status="odd")),
]


GLASS_BOX = {"actual_output": "reply.answer", "retrieved_context_ids": "debug.retrieved[*].id"}
BLACK_BOX = {"actual_output": "reply.answer"}


def make_run(x200, rows=ROWS, status="complete", sha256=None, response=GLASS_BOX):
    folder = x200 / "runs" / "2026-10-21_baseline"
    folder.mkdir(parents=True)
    (folder / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    run = {"name": "baseline", "status": status, "stop_reason": None if status == "complete" else "interrupted",
           "repeats": 3, "dataset": {"sha256": sha256 or load_dataset(x200).sha256},
           "app": {"file": "app.yaml", "config": {"name": "x200-support-bot", "response": response}},
           "counts": {"written": len(rows)}}
    (folder / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
    return folder


def metrics(report):
    return {m.name: m for m in report.metrics}


def test_question_metrics(x200):
    m = metrics(build_report(make_run(x200)))
    # q-0001 matches 2 of 3 repeats (the timeout fails); q-0002 says partial; q-0003 answers instead of
    # clarifying; q-0004's status isn't mapped, so it waits for the judge.
    assert m["behavior_accuracy"].value == pytest.approx((2 / 3 + 0 + 0) / 3)
    assert (m["behavior_accuracy"].scored, m["behavior_accuracy"].total) == (3, 4)
    # Only q-0003 can be scored for unsafe answers, and it answered.
    assert m["unsafe_answer_rate"].value == 1.0
    assert (m["unsafe_answer_rate"].scored, m["unsafe_answer_rate"].total) == (1, 2)
    # The timeout is a missed answer for q-0001; a partial answer isn't missed.
    assert m["missed_answer_rate"].value == pytest.approx((1 / 3 + 0) / 2)


def test_retrieval_metrics(x200):
    m = metrics(build_report(make_run(x200)))
    # q-0001: p12 at rank 1, then rank 2, then a timeout. q-0002: faq#7 (group 1) rank 1, p15 (group 2) rank 3.
    assert m["recall@1"].value == pytest.approx((1 / 3 + 1 / 2) / 2)
    assert m["recall@3"].value == pytest.approx((2 / 3 + 1) / 2)
    assert m["mrr"].value == pytest.approx(((1 + 1 / 2 + 0) / 3 + 1) / 2)
    assert (m["mrr"].scored, m["mrr"].total) == (2, 2)


def test_call_metrics(x200):
    report = build_report(make_run(x200))
    m = metrics(report)
    assert m["errors"].value == pytest.approx(1 / 6)
    assert m["latency_p50_ms"].value == 300
    assert m["latency_p95_ms"].value == pytest.approx(480)
    assert m["cost_per_1000"].value is None
    assert report.errors_by_type == {"timeout": 1}


def test_glass_box_replies_without_passages_are_skipped(x200):
    rows = [{**r, "retrieved_context_ids": None} for r in ROWS if r["error"] is None]
    m = metrics(build_report(make_run(x200, rows)))
    assert m["recall@5"].value is None and (m["recall@5"].scored, m["recall@5"].total) == (0, 2)


def test_black_box_apps_have_no_retrieval_metrics_even_with_errors(x200):
    # Found by the first real smoke test: a failed call used to score 0 recall for a black-box app.
    rows = [{**r, "retrieved_context_ids": None} for r in ROWS]
    report = build_report(make_run(x200, rows, response=BLACK_BOX))
    m = metrics(report)
    for name in ("recall@1", "recall@3", "recall@5", "mrr"):
        assert m[name].value is None and (m[name].scored, m[name].total) == (0, 0)
    assert report.by_tag["battery"]["recall@5"] is None
    assert m["missed_answer_rate"].value == pytest.approx((1 / 3 + 0) / 2)  # other metrics unchanged

    output = runner.invoke(app, ["report", str(x200 / "runs" / "2026-10-21_baseline")]).output
    assert "Retrieval  not measured" in output
    assert "recall@1" not in output and "mrr" not in output


def test_matrix_and_breakdowns(x200):
    report = build_report(make_run(x200))
    assert report.matrix == {"answer": {"answer": 2, "error": 1, "partial": 1},
                             "clarify": {"answer": 1}, "handoff": {"unknown": 1}}
    assert report.by_source["real_traffic"]["behavior_accuracy"] == pytest.approx((2 / 3 + 0) / 2)
    assert report.by_source["staff"]["behavior_accuracy"] == 0.0
    assert report.by_source["staff"]["questions"] == 2
    assert set(report.by_tag) == {"battery", "noise", "maintenance"}
    assert report.by_tag["battery"]["errors"] == pytest.approx(1 / 3)


def test_questions_to_look_at(x200):
    report = build_report(make_run(x200))
    assert report.to_check["Missed answers"] == [("q-0001", 1, 3)]
    assert report.to_check["Unsafe answers"] == [("q-0003", 1, 1)]
    assert report.to_check["Wrong behavior"] == [("q-0002", 1, 1), ("q-0003", 1, 1), ("q-0001", 1, 3)]


def test_warnings(x200):
    report = build_report(make_run(x200, status="incomplete", sha256="0" * 64))
    messages = " ".join(str(w) for w in report.warnings)
    assert "the dataset changed since this run" in messages
    assert "the run is incomplete (interrupted)" in messages


def test_report_command(x200):
    folder = make_run(x200)
    result = runner.invoke(app, ["report", str(folder)])
    assert result.exit_code == 0, result.output
    assert "behavior_accuracy" in result.output and "22.2%" in result.output
    assert "unsafe_answer_rate          100.0%     100.0-100.0%           1 of 2 questions  (lower is better)" in result.output
    assert "latency_p95_ms              480 ms       300-500 ms           5 of 6 calls" in result.output
    assert "behavior_accuracy            22.2%        0.0-66.7%  100.0%   3 of 4 questions" in result.output
    assert "Missed answers:  q-0001 (1/3)" in result.output
    assert "By source" in result.output and "By tag" in result.output

    saved = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    assert saved["metrics"]["behavior_accuracy"]["scored"] == 3
    assert saved["dataset_matches_run"] is True
    assert "q-0001" not in json.dumps(saved)  # aggregate numbers only


def test_group_tables_have_one_row_per_group_and_align_wide_characters(x200):
    from litmus.report import _width
    assert _width("空气净化器") == 10 and _width("abc") == 3
    records = [json.loads(line) for line in (x200 / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    records[0]["tags"] = ["空气净化器"]
    (x200 / "dataset.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                                        encoding="utf-8")
    output = runner.invoke(app, ["report", str(make_run(x200))]).output
    tag_rows = output.split("By tag\n")[1].split("\n\n")[0].splitlines()
    header, rows = tag_rows[0], tag_rows[1:]
    assert header.split() == ["questions", "behavior", "unsafe", "missed", "recall@5", "mrr", "errors"]
    assert [row.split()[0] for row in rows] == ["maintenance", "noise", "空气净化器"]
    assert len({_width(row) for row in tag_rows}) == 1  # every row lines up


def test_report_never_shows_question_text(x200):
    folder = make_run(x200)
    output = runner.invoke(app, ["report", str(folder)]).output
    assert "battery last" not in output and "HEPA" not in output


def test_not_a_run_folder(x200):
    result = runner.invoke(app, ["report", str(x200)])
    assert result.exit_code == 1
    assert "isn't a run folder" in result.output


def test_run_then_report(x200, fake_app):
    result = runner.invoke(app, ["run", str(x200), "--app", str(x200 / "app.yaml"), "--name", "e2e", "--yes"])
    assert result.exit_code == 0, result.output
    [folder] = (x200 / "runs").iterdir()
    result = runner.invoke(app, ["report", str(folder)])
    assert result.exit_code == 0, result.output
    # The fake app always says "ok" (answer) and retrieves p12.
    m = metrics(build_report(folder))
    assert m["behavior_accuracy"].value == pytest.approx(0.5)  # right on the 2 answer questions only
    assert m["unsafe_answer_rate"].value == 1.0
    assert m["recall@1"].value == pytest.approx((1 + 0) / 2)
