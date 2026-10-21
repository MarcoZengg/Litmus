import json
import threading

import httpx
import pytest

from litmus.adapters import AdapterError, BudgetExhausted, HttpAdapter, create_adapter
from litmus.appconfig import load_app_config
from litmus.dataset import Record

CONFIG = """\
name: x200-support-bot
adapter: http
request:
  url: https://bot.test/chat
  headers:
    Authorization: "Bearer ${BOT_TOKEN}"
  body:
    message: "{{input}}"
    category: "{{metadata.category?}}"
  timeout_s: 5
response:
  actual_output: reply.answer
  retrieved_context_ids: debug.retrieved[*].id
  retrieval_context: debug.retrieved[*].text
  status: reply.status
  cached: reply.from_cache
  input_tokens: usage.input_tokens
  output_tokens: usage.output_tokens
behavior_map:
  ok: answer
  escalate: handoff
"""
REPLY = {
    "reply": {"answer": "About 6 hours.", "status": "ok", "from_cache": False},
    "debug": {"retrieved": [{"id": "p12", "text": "Battery: 6 h"}, {"id": 7, "text": "Noise"}]},
    "usage": {"input_tokens": 120, "output_tokens": 9},
}
RECORD = Record(id="q-1", input="How long does the battery last?", metadata={"category": "purifier"})
ENV = {"BOT_TOKEN": "s3cret"}


class Clock:
    """A fake clock: sleeping just moves time forward."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make_adapter(tmp_path, handler, extra="", env=ENV, clock=None):
    path = tmp_path / "app.yaml"
    path.write_text(CONFIG + extra, encoding="utf-8")
    config, issues = load_app_config(path)
    assert issues == [], issues
    clock = clock or Clock()
    return HttpAdapter(config, env, transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep)


def reply_with(payload, status=200):
    return lambda request: httpx.Response(status, json=payload)


def test_a_full_reply_is_normalized(tmp_path):
    seen = {}

    def handler(request):
        seen["method"], seen["url"] = request.method, str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=REPLY)

    response = make_adapter(tmp_path, handler).call(RECORD)
    assert seen == {"method": "POST", "url": "https://bot.test/chat", "auth": "Bearer s3cret",
                    "body": {"message": "How long does the battery last?", "category": "purifier"}}
    assert response.error is None
    assert response.actual_output == "About 6 hours."
    assert response.retrieved_context_ids == ["p12", "7"]
    assert response.retrieval_context == ["Battery: 6 h", "Noise"]
    assert (response.status, response.observed_behavior, response.behavior_source) == ("ok", "answer", "status")
    assert response.cached is False
    assert (response.input_tokens, response.output_tokens) == (120, 9)
    assert response.latency_ms is not None


def test_optional_metadata_is_left_out(tmp_path):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=REPLY)

    make_adapter(tmp_path, handler).call(Record(id="q-2", input="What's the battery life?"))
    assert bodies == [{"message": "What's the battery life?"}]


def test_unmapped_status_leaves_behavior_for_the_judge(tmp_path):
    payload = {"reply": {"answer": "?", "status": "weird"}}
    response = make_adapter(tmp_path, reply_with(payload)).call(RECORD)
    assert response.status == "weird"
    assert response.observed_behavior is None and response.behavior_source is None


def test_black_box_reply_without_passages(tmp_path):
    response = make_adapter(tmp_path, reply_with({"reply": {"answer": "Yes."}})).call(RECORD)
    assert response.error is None
    assert response.retrieved_context_ids is None and response.status is None


@pytest.mark.parametrize("handler, kind, message", [
    (reply_with({"error": "boom"}, status=500), "http_status", "HTTP 500"),
    (reply_with({"reply": {}}), "no_answer", "the actual_output expression found nothing"),
    (lambda r: httpx.Response(200, text="<html>"), "invalid_json", "the reply isn't JSON"),
    (reply_with({"reply": {"answer": ["a"]}}), "bad_response_field", "actual_output isn't text"),
    (reply_with({"reply": {"answer": "a", "from_cache": "no"}}), "bad_response_field", "cached isn't true or false"),
    (reply_with({"reply": {"answer": "a"}, "usage": {"input_tokens": -1}}), "bad_response_field",
     "input_tokens isn't a whole number"),
    (lambda r: httpx.Response(302, headers={"Location": "https://elsewhere.test"}), "http_status", "HTTP 302"),
])
def test_failures(tmp_path, handler, kind, message):
    response = make_adapter(tmp_path, handler).call(RECORD)
    assert response.error["type"] == kind
    assert response.error["message"] == message
    assert response.actual_output is None


def test_timeouts_and_connection_errors(tmp_path):
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    assert make_adapter(tmp_path, timeout).call(RECORD).error == {"type": "timeout", "message": "no reply within 5 s"}
    assert make_adapter(tmp_path, refused).call(RECORD).error == {"type": "connection_error", "message": "ConnectError"}


def test_errors_never_include_bodies(tmp_path):
    secret = {"reply": {"answer": ["SECRET ANSWER"]}}
    response = make_adapter(tmp_path, reply_with(secret)).call(Record(id="q", input="SECRET QUESTION"))
    assert "SECRET" not in json.dumps(response.error)


def test_no_retries_by_default(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    adapter = make_adapter(tmp_path, handler)
    assert adapter.call(RECORD).error["message"] == "HTTP 503"
    assert len(calls) == 1


def test_retries_when_asked(tmp_path):
    replies = iter([httpx.Response(503), httpx.Response(429), httpx.Response(200, json=REPLY)])
    clock = Clock()
    adapter = make_adapter(tmp_path, lambda request: next(replies), extra="retries: 2\n", clock=clock)
    assert adapter.call(RECORD).error is None
    assert adapter.calls_made == 3
    assert clock.sleeps == [1, 2]


def test_client_errors_are_not_retried(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401)

    make_adapter(tmp_path, handler, extra="retries: 3\n").call(RECORD)
    assert len(calls) == 1


def test_max_requests_stops_calls(tmp_path):
    adapter = make_adapter(tmp_path, reply_with(REPLY), extra="limits:\n  max_requests: 2\n")
    adapter.call(RECORD)
    adapter.call(RECORD)
    with pytest.raises(BudgetExhausted, match="max_requests \\(2\\)"):
        adapter.call(RECORD)
    assert adapter.calls_made == 2


def test_requests_per_minute_spaces_out_calls(tmp_path):
    clock = Clock()
    adapter = make_adapter(tmp_path, reply_with(REPLY), extra="limits:\n  requests_per_minute: 30\n", clock=clock)
    for _ in range(3):
        adapter.call(RECORD)
    assert clock.sleeps == [2.0, 2.0]


def test_concurrency_limits_calls_in_flight(tmp_path):
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}
    release = threading.Event()

    def handler(request):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        release.wait(1)
        with lock:
            state["now"] -= 1
        return httpx.Response(200, json=REPLY)

    adapter = make_adapter(tmp_path, handler, extra="limits:\n  concurrency: 2\n")
    threads = [threading.Thread(target=adapter.call, args=(RECORD,)) for _ in range(5)]
    for thread in threads:
        thread.start()
    release.set()
    for thread in threads:
        thread.join()
    assert state["peak"] <= 2


def test_missing_secret_stops_setup(tmp_path):
    with pytest.raises(AdapterError, match="BOT_TOKEN"):
        make_adapter(tmp_path, reply_with(REPLY), env={})


def test_get_requests_send_the_body_as_query_parameters(tmp_path):
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=REPLY)

    path = tmp_path / "app.yaml"
    path.write_text("adapter: http\nrequest:\n  method: GET\n  url: https://bot.test/ask\n"
                    "  body:\n    q: \"{{input}}\"\nresponse:\n  actual_output: reply.answer\n", encoding="utf-8")
    config, _ = load_app_config(path)
    HttpAdapter(config, {}, transport=httpx.MockTransport(handler)).call(RECORD)
    assert seen["params"] == {"q": "How long does the battery last?"}


def test_other_adapters_are_not_implemented_yet(tmp_path):
    path = tmp_path / "app.yaml"
    path.write_text("adapter: python\nfunction: my_app:answer\n", encoding="utf-8")
    config, _ = load_app_config(path)
    with pytest.raises(AdapterError, match="python adapter isn't implemented yet"):
        create_adapter(config)
