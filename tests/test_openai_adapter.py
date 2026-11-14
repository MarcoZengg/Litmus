import json

import httpx
import pytest
import yaml
from typer.testing import CliRunner

import litmus.cli
from litmus.adapters import AdapterError, OpenAIAdapter, create_adapter
from litmus.appconfig import load_app_config
from litmus.cli import app
from litmus.dataset import Record, load_dataset
from litmus.passages import Passages, PassagesError, load_passages
from litmus.runner import is_glass_box

BASE = """\
name: qwen-base
adapter: openai
provider: dashscope
base_url: https://api.test/v1
model: qwen-test
"""
RECORD = Record(id="q-1", input="How long does the battery last?", metadata={"category": "purifier"})
ROWS = [
    {"id": "q-1", "repeat": 2, "retrieved_context_ids": ["p9"], "retrieval_context": ["Old"], "error": None},
    {"id": "q-1", "repeat": 1, "retrieved_context_ids": ["p12", "p7"],
     "retrieval_context": ["Battery: 6 h", "Noise: 24 dB"], "error": None},
    {"id": "q-2", "repeat": 1, "retrieved_context_ids": None, "retrieval_context": None,
     "error": {"type": "http_status", "message": "HTTP 502"}},
    {"id": "q-3", "repeat": 1, "retrieved_context_ids": [], "retrieval_context": [], "error": None},
]
runner = CliRunner()


class Clock:
    """A fake clock: sleeping just moves time forward."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def completion(content, usage=None):
    body = {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def make(tmp_path, handler, extra="", env=None):
    path = tmp_path / "app.yaml"
    path.write_text(BASE + extra, encoding="utf-8")
    config, issues = load_app_config(path)
    assert issues == [], issues
    clock = Clock()
    return OpenAIAdapter(config, env or {}, transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep)


def source_run(folder, rows, sha256="abc"):
    folder.mkdir(parents=True)
    (folder / "run.yaml").write_text(yaml.safe_dump({"name": "glass", "dataset": {"sha256": sha256}}),
                                     encoding="utf-8")
    (folder / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return folder


def test_a_full_request_and_reply(tmp_path):
    seen = {}

    def handler(request):
        seen["url"], seen["auth"] = str(request.url), request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return completion("About 6 hours.", {"prompt_tokens": 30, "completion_tokens": 4})

    extra = ('api_key_env: TEST_KEY\nsystem_prompt: "You support the {{metadata.category}}."\n'
             'user_template: "Q: {{input}}"\ntemperature: 0.2\nextra:\n  max_tokens: 100\n  enable_thinking: false\n')
    response = make(tmp_path, handler, extra, env={"TEST_KEY": "s3cret"}).call(RECORD)
    assert seen["url"] == "https://api.test/v1/chat/completions"
    assert seen["auth"] == "Bearer s3cret"
    assert seen["body"] == {"model": "qwen-test", "temperature": 0.2, "max_tokens": 100, "enable_thinking": False,
                            "messages": [{"role": "system", "content": "You support the purifier."},
                                         {"role": "user", "content": "Q: How long does the battery last?"}]}
    assert response.error is None
    assert response.actual_output == "About 6 hours."
    assert (response.input_tokens, response.output_tokens) == (30, 4)
    assert response.retrieved_context_ids is None and response.observed_behavior is None
    assert response.latency_ms is not None


def test_by_default_it_sends_just_the_question(tmp_path):
    seen = []

    def handler(request):
        seen.append((json.loads(request.content), request.headers.get("Authorization")))
        return completion("Yes.")

    response = make(tmp_path, handler).call(RECORD)
    assert seen == [({"model": "qwen-test", "temperature": 0,
                      "messages": [{"role": "user", "content": RECORD.input}]}, None)]
    assert response.actual_output == "Yes." and response.input_tokens is None


def test_temperature_can_be_left_to_the_api(tmp_path):
    bodies = []
    make(tmp_path, lambda request: bodies.append(json.loads(request.content)) or completion("ok"),
         "temperature: null\n").call(RECORD)
    assert "temperature" not in bodies[0]


def test_missing_key_stops_setup(tmp_path):
    with pytest.raises(AdapterError, match="TEST_KEY"):
        make(tmp_path, lambda request: completion("x"), "api_key_env: TEST_KEY\n")


def test_json_replies_are_read_with_response_paths(tmp_path):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return completion('```json\n{"answer": "About 6 hours.", "status": "supported"}\n```')

    extra = ("json_mode: true\nresponse:\n  actual_output: answer\n  status: status\n"
             "behavior_map:\n  supported: answer\n")
    response = make(tmp_path, handler, extra).call(RECORD)
    assert bodies[0]["response_format"] == {"type": "json_object"}
    assert response.error is None
    assert (response.actual_output, response.status, response.observed_behavior, response.behavior_source) == (
        "About 6 hours.", "supported", "answer", "status")


@pytest.mark.parametrize("reply, extra, kind, message", [
    (httpx.Response(500, json={"error": "boom"}), "", "http_status", "HTTP 500"),
    (httpx.Response(200, json={"id": "x"}), "", "invalid_reply", "not a chat completions reply"),
    (httpx.Response(200, text="<html>"), "", "invalid_reply", "not a chat completions reply"),
    (completion(None), "", "no_answer", "the model's reply has no text"),
    (completion("  "), "", "no_answer", "the model's reply has no text"),
    (completion("Six hours."), "response:\n  actual_output: answer\n", "invalid_json", "the model's reply isn't JSON"),
    (completion('{"text": "Six hours."}'), "response:\n  actual_output: answer\n", "no_answer",
     "the actual_output expression found nothing"),
])
def test_failures(tmp_path, reply, extra, kind, message):
    response = make(tmp_path, lambda request: reply, extra).call(RECORD)
    assert (response.error["type"], response.error["message"]) == (kind, message)
    assert response.actual_output is None


def test_timeouts_and_retries(tmp_path):
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    assert make(tmp_path, timeout).call(RECORD).error == {"type": "timeout", "message": "no reply within 60 s"}
    replies = iter([httpx.Response(429), completion("ok")])
    adapter = make(tmp_path, lambda request: next(replies), "retries: 1\n")
    assert adapter.call(RECORD).error is None
    assert adapter.calls_made == 2


def test_errors_never_include_bodies(tmp_path):
    adapter = make(tmp_path, lambda request: completion("SECRET ANSWER"), "response:\n  actual_output: answer\n")
    response = adapter.call(Record(id="q", input="SECRET QUESTION"))
    assert response.error is not None
    assert "SECRET" not in json.dumps(response.error)


def test_create_adapter_builds_it(tmp_path):
    path = tmp_path / "app.yaml"
    path.write_text(BASE, encoding="utf-8")
    config, _ = load_app_config(path)
    adapter = create_adapter(config, {})
    assert isinstance(adapter, OpenAIAdapter)
    adapter.close()


def test_load_passages_from_a_run(tmp_path):
    saved = load_passages(source_run(tmp_path / "runs" / "r", ROWS))
    assert saved.source == "run glass" and saved.dataset_sha256 == "abc"
    assert saved.by_id["q-1"] == Passages(["Battery: 6 h", "Noise: 24 dB"], ["p12", "p7"])  # the lowest repeat
    assert "q-2" not in saved.by_id  # a failed call saved no passages
    assert saved.by_id["q-3"] == Passages([], [])


def test_load_passages_from_a_jsonl_file(tmp_path):
    path = tmp_path / "frozen.jsonl"
    path.write_text(json.dumps({"id": "q-1", "retrieval_context": ["A", "B"], "retrieved_context_ids": ["a"]}) + "\n",
                    encoding="utf-8")
    saved = load_passages(path)
    assert saved.source == "frozen.jsonl" and saved.dataset_sha256 is None
    assert saved.by_id["q-1"] == Passages(["A", "B"], None)  # IDs that don't line up are dropped


def test_unreadable_passages(tmp_path):
    with pytest.raises(PassagesError, match="not found"):
        load_passages(tmp_path / "missing.jsonl")
    (tmp_path / "empty").mkdir()
    with pytest.raises(PassagesError, match="isn't a run folder"):
        load_passages(tmp_path / "empty")
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{oops\n", encoding="utf-8")
    with pytest.raises(PassagesError, match="line 1 of bad.jsonl isn't JSON"):
        load_passages(bad)
    with pytest.raises(AdapterError, match="not found"):
        make(tmp_path, lambda request: completion("x"), 'passages_from: nowhere.jsonl\nuser_template: "{{passages}}"\n')


def test_saved_passages_go_into_the_prompt_and_the_reply(tmp_path):
    source_run(tmp_path / "runs" / "glass", ROWS)
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return completion("About 6 hours [1].")

    extra = 'passages_from: runs/glass\nuser_template: "Passages:\\n{{passages}}\\n\\nQuestion: {{input}}"\n'
    adapter = make(tmp_path, handler, extra)
    response = adapter.call(RECORD)
    assert bodies[0]["messages"] == [{"role": "user", "content": "Passages:\n[1] (p12) Battery: 6 h\n"
                                                                 "[2] (p7) Noise: 24 dB\n\nQuestion: " + RECORD.input}]
    assert response.retrieved_context_ids == ["p12", "p7"]
    assert response.retrieval_context == ["Battery: 6 h", "Noise: 24 dB"]

    unknown = adapter.call(Record(id="q-2", input="Is it loud?"))
    assert bodies[1]["messages"][0]["content"] == "Passages:\n\n\nQuestion: Is it loud?"
    assert unknown.retrieved_context_ids is None and unknown.retrieval_context is None


@pytest.fixture
def fake_model(monkeypatch):
    """Route litmus run's model calls to a fake chat completions API; returns the request bodies."""
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return completion(f"answer {len(bodies)}", {"prompt_tokens": 10, "completion_tokens": 2})

    def create_adapter(config, env):
        clock = Clock()
        return OpenAIAdapter(config, env, transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep)

    monkeypatch.setattr(litmus.cli, "create_adapter", create_adapter)
    return bodies


def run(x200, name, *extra):
    return runner.invoke(app, ["run", str(x200), "--app", str(x200 / "model.yaml"), "--name", name,
                               "--repeats", "1", "--yes", *extra])


def test_validate_and_run_a_model(x200, fake_model):
    (x200 / "model.yaml").write_text(BASE, encoding="utf-8")
    result = runner.invoke(app, ["validate", str(x200), "--app", str(x200 / "model.yaml")])
    assert result.exit_code == 0, result.output
    assert "App        qwen-base (openai, qwen-test at api.test)" in result.output

    result = run(x200, "base")
    assert result.exit_code == 0, result.output
    [folder] = (x200 / "runs").iterdir()
    lines = [json.loads(line) for line in (folder / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == len(fake_model) == 4
    assert lines[0]["actual_output"] == "answer 1" and lines[0]["input_tokens"] == 10
    saved = yaml.safe_load((folder / "run.yaml").read_text(encoding="utf-8"))
    assert saved["app"]["config"]["model"] == "qwen-test"
    assert not is_glass_box(saved)


def test_a_run_can_replay_another_runs_passages(x200, fake_model):
    rows = [{"id": "q-0001", "repeat": 1, "retrieved_context_ids": ["x200-manual#p12"],
             "retrieval_context": ["Battery: about 6 hours."], "error": None}]
    source_run(x200 / "runs" / "2026-11-01_glass", rows, sha256=load_dataset(x200).sha256)
    (x200 / "model.yaml").write_text(BASE + 'passages_from: runs/2026-11-01_glass\n'
                                            'user_template: "{{passages}}\\n\\n{{input}}"\n', encoding="utf-8")
    result = run(x200, "replay")
    assert result.exit_code == 0, result.output
    assert "Passages   from run glass, for 1 of 4 records" in result.output
    assert "passages_from has no passages for 3 record(s)" in result.output
    assert fake_model[0]["messages"][0]["content"].startswith("[1] (x200-manual#p12) Battery: about 6 hours.\n\n")

    folder = next(f for f in (x200 / "runs").iterdir() if f.name.endswith("_replay"))
    assert is_glass_box(yaml.safe_load((folder / "run.yaml").read_text(encoding="utf-8")))
    first = json.loads((folder / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["retrieved_context_ids"] == ["x200-manual#p12"]


def test_passages_from_another_dataset_version_are_refused(x200, fake_model):
    source_run(x200 / "runs" / "2026-11-01_glass", ROWS, sha256="0" * 64)
    (x200 / "model.yaml").write_text(BASE + 'passages_from: runs/2026-11-01_glass\nuser_template: "{{passages}}"\n',
                                     encoding="utf-8")
    result = run(x200, "replay")
    assert result.exit_code == 1
    assert "used another version of the dataset" in result.output
    assert fake_model == []
