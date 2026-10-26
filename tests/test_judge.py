import json

import httpx
import pytest
import yaml
from typer.testing import CliRunner

import litmus.cli
from conftest import X200, Clock
from litmus.adapters import AppResponse
from litmus.cli import app
from litmus.dataset import Manifest, load_dataset
from litmus.judge import (JudgeClient, check_judge, default_prompt, judge_run, load_judge_config,
                          load_judgments, parse_verdict, plan_judgments)
from litmus.report import build_report

runner = CliRunner()
JUDGE_KEY = "judge-s3cret"


def row(qid, repeat, **fields):
    return {"id": qid, "repeat": repeat, **{**AppResponse().to_dict(), **fields}}


# q-0001 answers on repeat 1 and fails on repeat 2; q-0002 hands off instead of answering;
# q-0003's status isn't mapped, so its behavior is unknown; q-0004 hands off as it should.
ROWS = [
    row("q-0001", 1, actual_output="About 6 hours.", observed_behavior="answer", behavior_source="status",
        retrieved_context_ids=["x200-manual#p12"], retrieval_context=["Battery: about 6 h at normal speed."],
        latency_ms=100.0),
    row("q-0001", 2, error={"type": "timeout", "message": "no reply within 60 s"}),
    row("q-0002", 1, actual_output="Please contact support.", observed_behavior="handoff", latency_ms=100.0),
    row("q-0003", 1, actual_output="Which product do you mean?", status="odd", latency_ms=100.0),
    row("q-0004", 1, actual_output="Please contact support.", observed_behavior="handoff", latency_ms=100.0),
]


def make_run(x200, rows=ROWS, name="baseline"):
    folder = x200 / "runs" / f"2026-10-23_{name}"
    folder.mkdir(parents=True)
    (folder / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    run = {"name": name, "status": "complete", "stop_reason": None, "repeats": 2, "seed": 7,
           "dataset": {"sha256": load_dataset(x200).sha256, "version": 1},
           "app": {"file": "app.yaml", "config": {"name": "x200-support-bot", "response": {
               "actual_output": "reply.answer", "retrieved_context_ids": "debug.retrieved[*].id",
               "retrieval_context": "debug.retrieved[*].text"}}},
           "counts": {"written": len(rows)}}
    (folder / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
    return folder


def grader(request):
    """A fake judge: correctness passes when the answer has the 6-hour fact; behavior says clarify."""
    prompt = json.loads(request.content)["messages"][0]["content"]
    if prompt.startswith("Classify"):
        content = {"behavior": "clarify", "reason": "It asks which product."}
    elif "Reference answer" in prompt:
        passed = "Assistant's answer:\nAbout 6 hours." in prompt
        content = {"verdict": "pass" if passed else "fail", "reason": "Checked against the reference."}
    else:
        content = {"verdict": "pass", "reason": "Fine."}
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}],
                                     "usage": {"prompt_tokens": 120, "completion_tokens": 15}})


def make_client(config, handler=grader):
    clock = Clock()
    return JudgeClient(config, {"X200_JUDGE_KEY": JUDGE_KEY}, transport=httpx.MockTransport(handler),
                       clock=clock, sleep=clock.sleep)


@pytest.fixture
def fake_judge(monkeypatch):
    state = {"handler": grader, "calls": 0}

    def handler(request):
        state["calls"] += 1
        return state["handler"](request)

    monkeypatch.setattr(litmus.cli, "create_judge_client", lambda config, env: make_client(config, handler))
    monkeypatch.setenv("X200_JUDGE_KEY", JUDGE_KEY)
    return state


# --- config ---

def test_x200_judge_config_loads():
    config, issues = load_judge_config(X200 / "judge.yaml")
    assert issues == []
    assert (config.provider, config.model, config.temperature, config.json_mode) == (
        "example-provider", "judge-large", 0, True)
    assert config.metrics == ["correctness", "faithfulness", "rubric", "behavior"]
    assert config.prompts["correctness"] == default_prompt("correctness")
    assert set(config.prompt_sha256) == set(config.metrics)
    assert config.extra == {"enable_thinking": False}


@pytest.mark.parametrize("text, expected", [
    ("base_url: https://j.test/v1\nmodel: m\n", "'provider' is required"),
    ("provider: p\nbase_url: j.test\nmodel: m\n", "'base_url' must start with http"),
    ("provider: p\nbase_url: https://j.test/v1\n", "'model' is required"),
    ("provider: p\nbase_url: https://j.test/v1\nmodel: m\nmetrics: [accuracy]\n", "'metrics' must list some of"),
    ("provider: p\nbase_url: https://j.test/v1\nmodel: m\nmetrics: [rubric]\n", "the rubric metric needs 'rubric'"),
    ("provider: p\nbase_url: https://j.test/v1\nmodel: m\ntemprature: 0\n", "unknown setting 'temprature'"),
    ("provider: p\nbase_url: https://j.test/v1\nmodel: m\nprompts:\n  correctness: missing.txt\n", "not found"),
    ("provider: p\nbase_url: https://j.test/v1\nmodel: m\napi_key_env: not a name\n", "'api_key_env' must be"),
])
def test_judge_config_errors(tmp_path, text, expected):
    path = tmp_path / "judge.yaml"
    path.write_text(text, encoding="utf-8")
    config, issues = load_judge_config(path)
    assert config is None
    assert any(expected in str(issue) for issue in issues), [str(i) for i in issues]


def test_prompt_overrides_change_the_hash(tmp_path):
    (tmp_path / "correctness_zh.txt").write_text("用中文判断：$question / $reference / $answer", encoding="utf-8")
    path = tmp_path / "judge.yaml"
    path.write_text("provider: p\nbase_url: https://j.test/v1\nmodel: m\nmetrics: [correctness]\n"
                    "prompts:\n  correctness: correctness_zh.txt\n", encoding="utf-8")
    config, _ = load_judge_config(path)
    default, _ = load_judge_config(X200 / "judge.yaml")
    assert config.prompts["correctness"].startswith("用中文判断")
    assert config.prompt_sha256["correctness"] != default.prompt_sha256["correctness"]


def test_allowed_providers_and_missing_key():
    config, _ = load_judge_config(X200 / "judge.yaml")
    allowed = Manifest(name="d", version=1, allowed_providers=["example-provider"])
    blocked = Manifest(name="d", version=1, allowed_providers=["anthropic"])
    assert check_judge(allowed, config, {"X200_JUDGE_KEY": "k"}) == []
    issues = [str(i) for i in check_judge(blocked, config, {})]
    assert "judge.yaml: provider example-provider isn't in the dataset's allowed_providers" in issues
    assert "judge.yaml: environment variable X200_JUDGE_KEY is not set" in issues


# --- client and verdicts ---

def test_client_sends_an_openai_compatible_request():
    seen = {}

    def handler(request):
        seen["url"], seen["auth"] = str(request.url), request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return grader(request)

    config, _ = load_judge_config(X200 / "judge.yaml")
    reply = make_client(config, handler).ask("Classify this.")
    assert seen["url"] == "https://judge.example.com/v1/chat/completions"
    assert seen["auth"] == f"Bearer {JUDGE_KEY}"
    assert seen["body"] == {"model": "judge-large", "messages": [{"role": "user", "content": "Classify this."}],
                            "temperature": 0, "enable_thinking": False, "response_format": {"type": "json_object"}}
    assert (reply["input_tokens"], reply["output_tokens"], reply["error"]) == (120, 15, None)


@pytest.mark.parametrize("handler, kind", [
    (lambda r: httpx.Response(500), "http_status"),
    (lambda r: httpx.Response(200, json={"oops": 1}), "invalid_reply"),
])
def test_client_failures(handler, kind):
    config, _ = load_judge_config(X200 / "judge.yaml")
    assert make_client(config, handler).ask("x")["error"]["type"] == kind


@pytest.mark.parametrize("content, metric, expected", [
    ('{"verdict": "pass", "reason": "ok"}', "correctness", ("pass", "ok")),
    ('Sure! {"verdict": "FAIL", "reason": "wrong"} Hope that helps.', "correctness", ("fail", "wrong")),
    ('{"behavior": "Handoff", "reason": "r"}', "behavior", ("handoff", "r")),
])
def test_parse_verdict(content, metric, expected):
    assert parse_verdict(content, metric) == expected


@pytest.mark.parametrize("content, metric", [
    ("no json here", "correctness"),
    ('{"verdict": "maybe"}', "correctness"),
    ('{"behavior": "escalate"}', "behavior"),
])
def test_unreadable_verdicts(content, metric):
    with pytest.raises(ValueError):
        parse_verdict(content, metric)


# --- planning ---

def test_plan_judgments(x200):
    config, _ = load_judge_config(X200 / "judge.yaml")
    tasks = plan_judgments(load_dataset(x200), ROWS, config, glass_box=True)
    plan = {(t.id, t.repeat, t.metric): ("call" if t.prompt else t.verdict) for t in tasks}
    assert plan == {
        ("q-0001", 1, "correctness"): "call", ("q-0001", 1, "faithfulness"): "call", ("q-0001", 1, "rubric"): "call",
        ("q-0001", 2, "correctness"): "fail", ("q-0001", 2, "faithfulness"): "fail", ("q-0001", 2, "rubric"): "fail",
        ("q-0002", 1, "correctness"): "fail", ("q-0002", 1, "rubric"): "call",
        ("q-0003", 1, "rubric"): "call", ("q-0003", 1, "behavior"): "call",
        ("q-0004", 1, "rubric"): "call",
    }
    faithfulness = next(t for t in tasks if t.metric == "faithfulness" and t.prompt)
    assert "[1] (x200-manual#p12) Battery: about 6 h at normal speed." in faithfulness.prompt


def test_black_box_runs_skip_faithfulness(x200):
    config, _ = load_judge_config(X200 / "judge.yaml")
    tasks = plan_judgments(load_dataset(x200), ROWS, config, glass_box=False)
    assert not any(t.metric == "faithfulness" for t in tasks)


# --- judging and reports ---

def test_judge_run_writes_judgments_and_judging(x200):
    folder = make_run(x200)
    config, _ = load_judge_config(X200 / "judge.yaml")
    tasks = plan_judgments(load_dataset(x200), ROWS, config, glass_box=True)
    client = make_client(config)
    result = judge_run(folder, tasks, config, client)
    assert (result.planned_calls, result.automatic, result.judged, result.errors) == (7, 4, 7, 0)
    judging, lines = load_judgments(folder)
    assert judging["status"] == "complete" and judging["model"] == "judge-large"
    assert judging["prompt_sha256"] == config.prompt_sha256
    assert judging["counts"] == {"judge_calls": 7, "judged": 7, "errors": 0, "automatic": 4}
    assert len(lines) == 11
    for path in folder.iterdir():
        assert JUDGE_KEY not in path.read_text(encoding="utf-8")


def test_unreadable_verdicts_are_recorded_as_judge_errors(x200):
    folder = make_run(x200)
    config, _ = load_judge_config(X200 / "judge.yaml")
    bad = lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "I think it passes."}}]})
    tasks = plan_judgments(load_dataset(x200), ROWS, config, glass_box=True)
    result = judge_run(folder, tasks, config, make_client(config, bad))
    assert result.errors == 7
    _, lines = load_judgments(folder)
    assert {l["error"]["type"] for l in lines if l["source"] == "judge"} == {"unreadable_verdict"}


def test_report_with_judge_metrics(x200):
    folder = make_run(x200)
    config, _ = load_judge_config(X200 / "judge.yaml")
    judge_run(folder, plan_judgments(load_dataset(x200), ROWS, config, True), config, make_client(config))
    report = build_report(folder)
    m = {metric.name: metric for metric in report.metrics}
    # correctness: q-0001 passes repeat 1 and fails the timed-out repeat 2; q-0002 didn't answer.
    assert m["correctness"].value == pytest.approx((0.5 + 0) / 2)
    assert (m["correctness"].scored, m["correctness"].total) == (2, 2)
    assert m["faithfulness"].value == pytest.approx(0.5)
    assert m["rubric"].value == pytest.approx((0.5 + 1 + 1 + 1) / 4)
    # The judge fills in q-0003's unknown behavior (clarify, as expected).
    assert m["behavior_accuracy"].value == pytest.approx((0.5 + 0 + 1 + 1) / 4)
    assert report.to_check["Incorrect answers"] == [("q-0002", 1, 1), ("q-0001", 1, 2)]
    assert report.judge_tokens == (7 * 120, 7 * 15)


def test_judge_command_then_report(x200, fake_judge):
    folder = make_run(x200)
    result = runner.invoke(app, ["judge", str(folder), "--judge", str(X200 / "judge.yaml"), "--yes"])
    assert result.exit_code == 0, result.output
    assert "Calls      7 to the judge; 4 decided without a call" in result.output
    assert "Judged     7 by the judge, 4 without a call, 0 failed" in result.output
    assert fake_judge["calls"] == 7

    output = runner.invoke(app, ["report", str(folder)]).output
    assert "Judge      judge-large (example-provider): 7 judged, 0 failed, 4 decided without a call" in output
    assert "correctness" in output and "Incorrect answers:" in output
    saved = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    assert saved["judge"]["model"] == "judge-large" and "correctness" in saved["metrics"]


def test_judging_twice_needs_replace(x200, fake_judge):
    folder = make_run(x200)
    runner.invoke(app, ["judge", str(folder), "--judge", str(X200 / "judge.yaml"), "--yes"])
    result = runner.invoke(app, ["judge", str(folder), "--judge", str(X200 / "judge.yaml"), "--yes"])
    assert result.exit_code == 1 and "already judged; add --replace" in result.output
    result = runner.invoke(app, ["judge", str(folder), "--judge", str(X200 / "judge.yaml"), "--yes", "--replace"])
    assert result.exit_code == 0, result.output


def test_judge_respects_allowed_providers(x200, fake_judge):
    with open(x200 / "dataset.yaml", "a", encoding="utf-8") as manifest:
        manifest.write("allowed_providers: [anthropic]\n")
    folder = make_run(x200)
    result = runner.invoke(app, ["judge", str(folder), "--judge", str(X200 / "judge.yaml"), "--yes"])
    assert result.exit_code == 1
    assert "provider example-provider isn't in the dataset's allowed_providers" in result.output
    assert fake_judge["calls"] == 0


def test_judge_asks_before_calling(x200, fake_judge):
    folder = make_run(x200)
    result = runner.invoke(app, ["judge", str(folder), "--judge", str(X200 / "judge.yaml")], input="n\n")
    assert result.exit_code == 1 and "No judge calls were made." in result.output
    assert fake_judge["calls"] == 0


def test_run_with_judge(x200, fake_app, fake_judge):
    result = runner.invoke(app, ["run", str(x200), "--app", str(x200 / "app.yaml"), "--name", "judged",
                                 "--repeats", "1", "--judge", str(X200 / "judge.yaml"), "--yes"])
    assert result.exit_code == 0, result.output
    assert "Judge calls: at most 16, after the app calls" in result.output
    assert "Judged     " in result.output
    [folder] = (x200 / "runs").iterdir()
    judging, _ = load_judgments(folder)
    assert judging["status"] == "complete"
