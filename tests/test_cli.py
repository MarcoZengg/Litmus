from typer.testing import CliRunner

from conftest import X200, write_records
from litmus import __version__
from litmus.cli import app

runner = CliRunner()


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("validate", "run", "report", "compare"):
        assert command in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_stubs_say_not_implemented(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    config = tmp_path / "app.yaml"
    config.write_text("name: example\n")
    calls = {
        "run": ["run", str(dataset), "--app", str(config), "--name", "baseline"],
        "report": ["report", str(dataset)],
        "compare": ["compare", str(dataset), str(dataset)],
    }
    for command, args in calls.items():
        result = runner.invoke(app, args)
        assert result.exit_code == 1, command
        assert f"litmus {command} is not implemented yet." in result.output


def test_missing_dataset_is_a_usage_error(tmp_path):
    result = runner.invoke(app, ["validate", str(tmp_path / "missing")])
    assert result.exit_code == 2


def test_validate_a_good_dataset():
    result = runner.invoke(app, ["validate", str(X200), "--app", str(X200 / "app.yaml")])
    assert result.exit_code == 0, result.output
    assert "Dataset    x200-examples (version 1)" in result.output
    assert "Records    4" in result.output
    assert "Behaviors  answer 2, clarify 1, handoff 1" in result.output
    assert "Sources    staff 2, real_traffic 2" in result.output
    assert "Hash       sha256:" in result.output
    assert result.output.rstrip().endswith("Valid.")


def test_validate_reports_errors_and_warnings(x200):
    write_records(x200, '{"id": "a", "input": "q", "expected_behavior": "answer"}',
                  '{"id": "b", "input": "q", "source": "customer"}')
    result = runner.invoke(app, ["validate", str(x200)])
    assert result.exit_code == 1
    assert "Warnings (1):\n  a: expects an answer" in result.output
    assert "Errors (1):\n  b: 'source' must be one of" in result.output
    assert "Not valid: 1 error(s)." in result.output


def test_validate_checks_templates_against_records(x200):
    config = (x200 / "app.yaml").read_text(encoding="utf-8").replace("{{metadata.category?}}", "{{metadata.category}}")
    (x200 / "app.yaml").write_text(config, encoding="utf-8")
    result = runner.invoke(app, ["validate", str(x200), "--app", str(x200 / "app.yaml")])
    assert result.exit_code == 1
    assert "metadata.category: the request template needs it, but 4 record(s)" in result.output


def test_validate_reports_app_config_errors(x200):
    (x200 / "app.yaml").write_text("adapter: http\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(x200), "--app", str(x200 / "app.yaml")])
    assert result.exit_code == 1
    assert "app.yaml: 'request' is required for the http adapter" in result.output


def test_validate_enforces_allowed_hosts(x200):
    with open(x200 / "dataset.yaml", "a", encoding="utf-8") as manifest:
        manifest.write("allowed_hosts: [bot.example.org]\n")
    result = runner.invoke(app, ["validate", str(x200), "--app", str(x200 / "app.yaml")])
    assert result.exit_code == 1
    assert "host staging.example.com isn't in the dataset's allowed_hosts" in result.output


def test_validate_warns_about_unset_secrets(monkeypatch):
    monkeypatch.delenv("X200_BOT_TOKEN", raising=False)
    result = runner.invoke(app, ["validate", str(X200), "--app", str(X200 / "app.yaml")])
    assert result.exit_code == 0
    assert "App        x200-support-bot (http, staging.example.com)" in result.output
    assert "app.yaml: environment variable X200_BOT_TOKEN is not set" in result.output


def test_validate_never_prints_question_text():
    result = runner.invoke(app, ["validate", str(X200)])
    assert "battery" not in result.output
