"""The litmus command line. Commands are described in docs/design.md, section 3."""

import os
from collections import Counter
from pathlib import Path
from typing import Annotated, Optional
from urllib.parse import urlsplit

import typer

from litmus import __version__
from litmus.adapters import AdapterError, create_adapter
from litmus.appconfig import AppConfig, check_allowed, check_env, load_app_config, redacted_config
from litmus.dataset import BEHAVIORS, SOURCES, Dataset, Issue, load_dataset
from litmus.runner import RUN_NAME, execute_run, run_folder
from litmus.templates import check_request_templates

app = typer.Typer(
    help="Evaluate LLM and RAG apps, and tell real changes from noise.",
    no_args_is_help=True,
    add_completion=False,
)

ExistingPath = Annotated[Path, typer.Argument(exists=True)]


def not_implemented(command: str) -> None:
    typer.echo(f"litmus {command} is not implemented yet.", err=True)
    raise typer.Exit(code=1)


def show_version(value: bool) -> None:
    if value:
        typer.echo(f"litmus {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=show_version, is_eager=True, help="Show the version and exit.")
    ] = False,
) -> None:
    pass


def counts(values: list[str | None], order: tuple[str, ...]) -> str:
    tally = Counter(value or "unlabeled" for value in values)
    names = [name for name in order if name in tally] + sorted(set(tally) - set(order))
    return ", ".join(f"{name} {tally[name]}" for name in names)


def summary(dataset: Dataset) -> list[str]:
    lines = []
    if dataset.manifest:
        lines.append(f"Dataset    {dataset.manifest.name} (version {dataset.manifest.version})")
    lines.append(f"Records    {len(dataset.records)}")
    if dataset.records:
        lines.append(f"Behaviors  {counts([r.expected_behavior for r in dataset.records], BEHAVIORS)}")
        lines.append(f"Sources    {counts([r.source for r in dataset.records], SOURCES)}")
    if dataset.sha256:
        lines.append(f"Hash       sha256:{dataset.sha256[:12]}")
    if dataset.manifest:
        lines.append(f"Private    {'yes' if dataset.manifest.private else 'no'}")
    return lines


def check_app(dataset: Dataset, path: Path, secrets_required: bool
              ) -> tuple[AppConfig | None, list[str], list[Issue], list[Issue]]:
    """Load an app config and check it against a dataset: (config, summary lines, errors, warnings)."""
    config, errors = load_app_config(path)
    if config is None:
        return None, [], errors, []
    target = urlsplit(config.request.url).hostname if config.request else config.adapter
    lines = [f"App        {config.name} ({config.adapter}, {target})"]
    errors += check_request_templates(config, dataset.records)
    if dataset.manifest is not None:
        errors += check_allowed(dataset.manifest, config, os.environ)
    _, warnings = redacted_config(config)
    # Secrets may only be set where runs happen, so validate just warns.
    (errors if secrets_required else warnings).extend(check_env(config, os.environ))
    return config, lines, errors, warnings


def show(lines: list[str], warnings: list[Issue], errors: list[Issue]) -> None:
    for line in lines:
        typer.echo(line)
    for title, issues in (("Warnings", warnings), ("Errors", errors)):
        if issues:
            typer.echo(f"\n{title} ({len(issues)}):")
            for issue in issues:
                typer.echo(f"  {issue}")


@app.command()
def validate(
    dataset: ExistingPath,
    app_config: Annotated[
        Optional[Path],
        typer.Option("--app", exists=True, help="Also check this app config's request templates against every record."),
    ] = None,
) -> None:
    """Check a dataset, and an app config's templates, without making any calls."""
    loaded = load_dataset(dataset)
    errors = list(loaded.errors)
    warnings = list(loaded.warnings)
    lines = summary(loaded)
    if app_config is not None:
        _, app_lines, app_errors, app_warnings = check_app(loaded, app_config, secrets_required=False)
        lines += app_lines
        errors += app_errors
        warnings += app_warnings

    show(lines, warnings, errors)
    if errors:
        typer.echo(f"\nNot valid: {len(errors)} error(s).")
        raise typer.Exit(code=1)
    typer.echo("\nValid.")


def estimate(config: AppConfig, records: int, repeats: int) -> list[str]:
    planned = records * repeats
    lines = [f"Calls      {planned} to the app ({records} records x {repeats} repeats)"
             + (f", up to {planned * (1 + config.retries)} with retries" if config.retries else "")]
    lines.append("           No judge calls: judge metrics aren't available yet.")
    per_minute = config.limits.requests_per_minute
    if per_minute:
        lines.append(f"           At {per_minute} requests per minute, about {max(1, round(planned / per_minute))} minute(s).")
    return lines


@app.command()
def run(
    dataset: ExistingPath,
    app_config: Annotated[Path, typer.Option("--app", exists=True, help="App config (YAML).")],
    name: Annotated[str, typer.Option(help="Run name, used in the run folder's name.")],
    repeats: Annotated[int, typer.Option(min=1, help="Times to run each question.")] = 3,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation, e.g. in CI.")] = False,
) -> None:
    """Run a dataset against an app and save the results next to the dataset."""
    if not RUN_NAME.fullmatch(name):
        raise typer.BadParameter("use letters, digits, '.', '_' or '-', starting with a letter or digit",
                                 param_hint="--name")
    loaded = load_dataset(dataset)
    errors = list(loaded.errors)
    warnings = list(loaded.warnings)
    lines = summary(loaded)
    config, app_lines, app_errors, app_warnings = check_app(loaded, app_config, secrets_required=True)
    lines += app_lines
    errors += app_errors
    warnings += app_warnings
    folder = run_folder(loaded, name)
    if config is not None and loaded.records:
        lines += estimate(config, len(loaded.records), repeats)
        planned = len(loaded.records) * repeats
        limit = config.limits.max_requests
        if limit is not None and limit < planned:
            warnings.append(Issue("limits", f"max_requests ({limit}) is below the {planned} calls this run needs, "
                                            "so it will stop early and be marked incomplete"))
    lines.append(f"Run folder {folder}")
    if folder.exists():
        errors.append(Issue("run", "that run folder already exists; pick another --name"))

    show(lines, warnings, errors)
    if errors:
        typer.echo(f"\nNot started: {len(errors)} error(s). No calls were made.")
        raise typer.Exit(code=1)
    if not yes and not typer.confirm("\nStart the run?", default=False):
        typer.echo("Cancelled. No calls were made.")
        raise typer.Exit(code=1)
    try:
        adapter = create_adapter(config, os.environ)
    except AdapterError as error:
        typer.echo(f"Not started: {error}. No calls were made.")
        raise typer.Exit(code=1)

    saved_config, _ = redacted_config(config)
    step = max(1, len(loaded.records) * repeats // 10)

    def progress(done: int, total: int, failed: int) -> None:
        if done % step == 0 or done == total:
            typer.echo(f"  {done}/{total} done, {failed} error(s)")

    typer.echo("")
    try:
        result = execute_run(loaded, config, saved_config, adapter, folder, name, repeats, progress)
    finally:
        adapter.close()

    status = "complete" if result.complete else f"incomplete ({result.stop_reason or 'not every call finished'})"
    show([f"\nRun saved  {result.folder}",
          f"Status     {status}",
          f"Records    {result.written} of {result.planned} written, {result.errors} with an error",
          f"App calls  {result.app_calls}"], result.warnings, [])
    if not result.complete:
        raise typer.Exit(code=1)


@app.command()
def report(run_dir: Annotated[Path, typer.Argument(exists=True, metavar="RUN")]) -> None:
    """Show the metrics for one run."""
    not_implemented("report")


@app.command()
def compare(
    baseline: ExistingPath,
    candidate: ExistingPath,
    gates: Annotated[Optional[Path], typer.Option(exists=True, help="Gate rules (YAML); exit non-zero on failure.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Compare even if the runs aren't comparable.")] = False,
) -> None:
    """Compare two runs question by question, with a verdict per metric."""
    not_implemented("compare")
