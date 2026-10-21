"""The litmus command line. Commands are described in docs/design.md, section 3."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from litmus import __version__

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


@app.command()
def validate(dataset: ExistingPath) -> None:
    """Check a dataset, and an app config's templates, without making any calls."""
    not_implemented("validate")


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
