import json
import shutil
from pathlib import Path

import httpx
import pytest

import litmus.cli
from litmus.adapters import HttpAdapter

X200 = Path(__file__).parent / "data" / "x200"
TOKEN = "s3cret-token-value"


@pytest.fixture
def x200(tmp_path: Path) -> Path:
    """A writable copy of the X200 example dataset and app config."""
    folder = tmp_path / "x200"
    shutil.copytree(X200, folder)
    return folder


def write_records(folder: Path, *lines: str) -> None:
    (folder / "dataset.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


class Clock:
    """A fake clock: sleeping just moves time forward."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def answer(request):
    body = json.loads(request.content)
    return httpx.Response(200, json={"reply": {"answer": f"echo: {len(body['message'])}", "status": "ok"},
                                     "debug": {"retrieved": [{"id": "x200-manual#p12"}]}})


@pytest.fixture
def fake_app(monkeypatch):
    """Route litmus run's HTTP calls to a handler, with a fake clock so rate limits don't sleep."""
    state = {"handler": answer, "calls": 0}

    def handler(request):
        state["calls"] += 1
        return state["handler"](request)

    def create_adapter(config, env):
        clock = Clock()
        return HttpAdapter(config, env, transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep)

    monkeypatch.setattr(litmus.cli, "create_adapter", create_adapter)
    monkeypatch.setenv("X200_BOT_TOKEN", TOKEN)
    return state
