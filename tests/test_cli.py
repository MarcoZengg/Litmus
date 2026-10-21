from typer.testing import CliRunner

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
        "validate": ["validate", str(dataset)],
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
