import json

import httpx
import yaml
from typer.testing import CliRunner

from conftest import TOKEN
from litmus.cli import app

runner = CliRunner()


def run(x200, *extra, input=None):
    return runner.invoke(app, ["run", str(x200), "--app", str(x200 / "app.yaml"), "--name", "baseline", *extra],
                         input=input)


def run_dir(x200):
    [folder] = (x200 / "runs").iterdir()
    return folder


def records(folder):
    return [json.loads(line) for line in (folder / "records.jsonl").read_text(encoding="utf-8").splitlines()]


def test_a_complete_run(x200, fake_app):
    result = run(x200, "--yes")
    assert result.exit_code == 0, result.output
    assert "Calls      12 to the app (4 records x 3 repeats)" in result.output
    assert "Status     complete" in result.output
    assert "Records    12 of 12 written, 0 with an error" in result.output

    folder = run_dir(x200)
    assert folder.name.endswith("_baseline")
    lines = records(folder)
    assert len(lines) == 12
    assert [(r["id"], r["repeat"]) for r in lines[:4]] == [("q-0001", 1), ("q-0002", 1), ("q-0003", 1), ("q-0004", 1)]
    assert lines[0]["observed_behavior"] == "answer"
    assert lines[0]["retrieved_context_ids"] == ["x200-manual#p12"]

    saved = yaml.safe_load((folder / "run.yaml").read_text(encoding="utf-8"))
    assert saved["status"] == "complete" and saved["stop_reason"] is None
    assert saved["repeats"] == 3 and isinstance(saved["seed"], int)
    assert saved["dataset"]["name"] == "x200-examples"
    assert len(saved["dataset"]["sha256"]) == 64
    assert saved["app"]["config"]["request"]["headers"]["Authorization"] == "Bearer ${X200_BOT_TOKEN}"
    assert saved["counts"] == {"planned": 12, "written": 12, "errors": 0, "app_calls": 12}


def test_secrets_never_reach_the_run_folder(x200, fake_app):
    run(x200, "--yes")
    folder = run_dir(x200)
    for path in folder.iterdir():
        assert TOKEN not in path.read_text(encoding="utf-8")


def test_literal_secrets_are_redacted_with_a_warning(x200, fake_app):
    config = (x200 / "app.yaml").read_text(encoding="utf-8").replace("Bearer ${X200_BOT_TOKEN}", f"Bearer {TOKEN}")
    (x200 / "app.yaml").write_text(config, encoding="utf-8")
    result = run(x200, "--yes")
    assert "header Authorization has a literal secret" in result.output
    saved = yaml.safe_load((run_dir(x200) / "run.yaml").read_text(encoding="utf-8"))
    assert saved["app"]["config"]["request"]["headers"]["Authorization"] == "<redacted>"


def test_asks_before_calling(x200, fake_app):
    result = run(x200, input="n\n")
    assert result.exit_code == 1
    assert "Cancelled. No calls were made." in result.output
    assert fake_app["calls"] == 0
    assert not (x200 / "runs").exists()

    result = run(x200, input="y\n")
    assert result.exit_code == 0, result.output
    assert fake_app["calls"] == 12


def test_errors_are_recorded_and_the_run_finishes(x200, fake_app):
    fake_app["handler"] = lambda request: httpx.Response(500)
    result = run(x200, "--yes", "--repeats", "1")
    assert result.exit_code == 0
    assert "Records    4 of 4 written, 4 with an error" in result.output
    assert {r["error"]["type"] for r in records(run_dir(x200))} == {"http_status"}


def test_max_requests_stops_the_run_and_marks_it_incomplete(x200, fake_app):
    config = (x200 / "app.yaml").read_text(encoding="utf-8").replace("max_requests: 300", "max_requests: 6")
    (x200 / "app.yaml").write_text(config, encoding="utf-8")
    result = run(x200, "--yes")
    assert result.exit_code == 1
    assert "max_requests (6) is below the 12 calls" in result.output
    assert "Status     incomplete (max_requests reached)" in result.output
    lines = records(run_dir(x200))
    assert len(lines) == 6
    assert {r["repeat"] for r in lines[:4]} == {1}  # repeat 1 covers every question first
    saved = yaml.safe_load((run_dir(x200) / "run.yaml").read_text(encoding="utf-8"))
    assert saved["status"] == "incomplete" and saved["stop_reason"] == "max_requests reached"


def test_missing_secret_stops_before_any_call(x200, fake_app, monkeypatch):
    monkeypatch.delenv("X200_BOT_TOKEN")
    result = run(x200, "--yes")
    assert result.exit_code == 1
    assert "environment variable X200_BOT_TOKEN is not set" in result.output
    assert "No calls were made." in result.output
    assert fake_app["calls"] == 0


def test_allowed_hosts_stops_before_any_call(x200, fake_app):
    with open(x200 / "dataset.yaml", "a", encoding="utf-8") as manifest:
        manifest.write("allowed_hosts: []\n")
    result = run(x200, "--yes")
    assert result.exit_code == 1
    assert "host staging.example.com isn't in the dataset's allowed_hosts" in result.output
    assert fake_app["calls"] == 0


def test_an_existing_run_is_never_overwritten(x200, fake_app):
    run(x200, "--yes")
    result = run(x200, "--yes")
    assert result.exit_code == 1
    assert "that run folder already exists" in result.output


def test_run_names_are_checked(x200, fake_app):
    result = runner.invoke(app, ["run", str(x200), "--app", str(x200 / "app.yaml"), "--name", "../escape", "--yes"])
    assert result.exit_code == 2
    assert fake_app["calls"] == 0


def test_warnings_after_the_run(x200, fake_app):
    fake_app["handler"] = lambda request: httpx.Response(200, json={
        "reply": {"answer": "same", "status": "odd", "from_cache": True}})
    result = run(x200, "--yes", "--repeats", "2")
    assert "8 replies came from the app's cache" in result.output
    assert "4 of 4 questions gave identical answers on every repeat" in result.output
    assert "8 replies had a status missing from behavior_map" in result.output


def test_run_output_never_shows_question_text(x200, fake_app):
    result = run(x200, "--yes")
    assert "battery" not in result.output.lower()
