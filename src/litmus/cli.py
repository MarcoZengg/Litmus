"""The litmus command line. Commands are described in docs/design.md, section 3."""

from collections import Counter
from pathlib import Path
from typing import Annotated, Optional

import typer

from litmus import __version__
from litmus.dataset import BEHAVIORS, SOURCES, Dataset, load_dataset
from litmus.templates import check_app_templates

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
    if app_config is not None:
        errors += check_app_templates(app_config, loaded.records)

    for line in summary(loaded):
        typer.echo(line)
    for title, issues in (("Warnings", loaded.warnings), ("Errors", errors)):
        if issues:
            typer.echo(f"\n{title} ({len(issues)}):")
            for issue in issues:
                typer.echo(f"  {issue}")
    if errors:
        typer.echo(f"\nNot valid: {len(errors)} error(s).")
        raise typer.Exit(code=1)
    typer.echo("\nValid.")


@app.command()
def run(
    dataset: ExistingPath,
    app_config: Annotated[Path, typer.Option("--app", exists=True, help="App config (YAML).")],
    name: Annotated[str, typer.Option(help="Run name, used in the run folder's name.")],
    repeats: Annotated[int, typer.Option(min=1, help="Times to run each question.")] = 3,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation, e.g. in CI.")] = False,
) -> None:
    """Run a dataset against an app and save the results next to the dataset."""
    not_implemented("run")


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
