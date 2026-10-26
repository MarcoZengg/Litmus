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
from litmus.compare import check_comparable, check_gates, compare_reports, format_comparison, format_gates, load_gates
from litmus.dataset import BEHAVIORS, SOURCES, Dataset, Issue, load_dataset
from litmus.judge import (JUDGE_METRICS, JUDGMENTS_FILE, JudgeClient, JudgeConfig, JudgeError, check_judge,
                          judge_run, load_judge_config, plan_judgments)
from litmus.labels import SUGGESTED_LABELS, label_values, labels_file, pick_replies, write_sheet
from litmus.report import ReportError, build_report, format_report, save_metrics
from litmus.runner import RECORDS_FILE, RUN_FILE, RUN_NAME, execute_run, is_glass_box, load_rows, load_run, run_folder
from litmus.templates import check_request_templates

app = typer.Typer(
    help="Evaluate LLM and RAG apps, and tell real changes from noise.",
    no_args_is_help=True,
    add_completion=False,
)

ExistingPath = Annotated[Path, typer.Argument(exists=True)]


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


def estimate(config: AppConfig, records: int, repeats: int, judge: JudgeConfig | None = None) -> list[str]:
    planned = records * repeats
    lines = [f"Calls      {planned} to the app ({records} records x {repeats} repeats)"
             + (f", up to {planned * (1 + config.retries)} with retries" if config.retries else "")]
    if judge is None:
        lines.append("           No judge calls: add --judge to grade the run, or run litmus judge later.")
    else:
        lines.append(f"           Judge calls: at most {planned * len(judge.metrics)}, after the app calls "
                     "(failed calls and non-answers need none).")
    per_minute = config.limits.requests_per_minute
    if per_minute:
        lines.append(f"           At {per_minute} requests per minute, about {max(1, round(planned / per_minute))} minute(s).")
    return lines


def load_judge(path: Path, dataset: Dataset) -> tuple[JudgeConfig | None, list[str], list[Issue]]:
    config, errors = load_judge_config(path)
    if config is None:
        return None, [], errors
    errors += check_judge(dataset.manifest, config, os.environ)
    return config, [f"Judge      {config.model} ({config.provider}): {', '.join(config.metrics)}"], errors


def judge_and_show(folder: Path, config: JudgeConfig, tasks: list) -> bool:
    try:
        client = create_judge_client(config, os.environ)
    except JudgeError as error:
        typer.echo(f"Judge not started: {error}. No judge calls were made.")
        return False
    calls = sum(task.prompt is not None for task in tasks)
    step = max(1, calls // 10)

    def progress(done: int, total: int, failed: int) -> None:
        if done % step == 0 or done == total:
            typer.echo(f"  judged {done}/{total}, {failed} failed")

    try:
        result = judge_run(folder, tasks, config, client, progress)
    finally:
        client.close()
    status = "complete" if result.complete else f"incomplete ({result.stop_reason or 'not every call finished'})"
    show([f"\nJudged     {result.judged} by the judge, {result.automatic} without a call, {result.errors} failed",
          f"Status     {status}",
          f"Judge calls {client.calls_made}"], [], [])
    return result.complete


def create_judge_client(config: JudgeConfig, env) -> JudgeClient:
    return JudgeClient(config, env)


@app.command()
def run(
    dataset: ExistingPath,
    app_config: Annotated[Path, typer.Option("--app", exists=True, help="App config (YAML).")],
    name: Annotated[str, typer.Option(help="Run name, used in the run folder's name.")],
    repeats: Annotated[int, typer.Option(min=1, help="Times to run each question.")] = 3,
    judge_config: Annotated[Optional[Path], typer.Option(
        "--judge", exists=True, help="Judge config (YAML): grade the run when the app calls are done.")] = None,
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
    judge = None
    if judge_config is not None:
        judge, judge_lines, judge_errors = load_judge(judge_config, loaded)
        lines += judge_lines
        errors += judge_errors
    folder = run_folder(loaded, name)
    if config is not None and loaded.records:
        lines += estimate(config, len(loaded.records), repeats, judge)
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
    judged = True
    if judge is not None and result.written:
        typer.echo("")
        tasks = plan_judgments(loaded, load_rows(folder), judge, is_glass_box(load_run(folder)))
        judged = judge_and_show(folder, judge, tasks)
    if not result.complete or not judged:
        raise typer.Exit(code=1)


@app.command()
def judge(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, metavar="RUN")],
    judge_config: Annotated[Path, typer.Option("--judge", exists=True, help="Judge config (YAML).")],
    replace: Annotated[bool, typer.Option("--replace", help="Replace this run's earlier judgments.")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation, e.g. in CI.")] = False,
) -> None:
    """Grade a finished run with the judge model, without calling the app again."""
    if not (run_dir / RUN_FILE).is_file() or not (run_dir / RECORDS_FILE).is_file():
        typer.echo(f"Can't judge: {run_dir} isn't a run folder: it needs {RUN_FILE} and {RECORDS_FILE}.")
        raise typer.Exit(code=1)
    loaded = load_dataset(run_dir.parent.parent)  # runs live in <dataset>/runs/<run>/
    errors = list(loaded.errors)
    config, lines, judge_errors = load_judge(judge_config, loaded)
    errors += judge_errors
    if (run_dir / JUDGMENTS_FILE).exists() and not replace:
        errors.append(Issue("judge", "this run is already judged; add --replace to judge it again"))
    run_info = load_run(run_dir)
    if loaded.sha256 is not None and loaded.sha256 != run_info["dataset"]["sha256"]:
        errors.append(Issue("dataset", "the dataset changed since this run, so its labels no longer match it"))
    tasks = []
    if config is not None and not errors:
        tasks = plan_judgments(loaded, load_rows(run_dir), config, is_glass_box(run_info))
        calls = sum(task.prompt is not None for task in tasks)
        lines.append(f"Calls      {calls} to the judge; {len(tasks) - calls} decided without a call "
                     "(failed calls and non-answers)")
        if config.limits.max_requests is not None and config.limits.max_requests < calls:
            lines.append(f"           max_requests ({config.limits.max_requests}) will stop it early.")
    show([f"Run        {run_info['name']}"] + lines, [], errors)
    if errors:
        typer.echo(f"\nNot started: {len(errors)} error(s). No judge calls were made.")
        raise typer.Exit(code=1)
    if not yes and not typer.confirm("\nStart judging?", default=False):
        typer.echo("Cancelled. No judge calls were made.")
        raise typer.Exit(code=1)
    typer.echo("")
    if not judge_and_show(run_dir, config, tasks):
        raise typer.Exit(code=1)


@app.command()
def label(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, metavar="RUN")],
    metric: Annotated[str, typer.Option(
        help="Judge metric to label: correctness, faithfulness, rubric or behavior.")] = "correctness",
    sample: Annotated[int, typer.Option(min=1, help="How many replies to put in the sheet.")] = SUGGESTED_LABELS,
    replace: Annotated[bool, typer.Option("--replace", help="Overwrite this run's sheet for the metric.")] = False,
) -> None:
    """Write a sheet of replies to label by hand, so reports can check the judge against you."""
    if not (run_dir / RUN_FILE).is_file() or not (run_dir / RECORDS_FILE).is_file():
        typer.echo(f"Can't label: {run_dir} isn't a run folder: it needs {RUN_FILE} and {RECORDS_FILE}.")
        raise typer.Exit(code=1)
    loaded = load_dataset(run_dir.parent.parent)  # runs live in <dataset>/runs/<run>/
    errors = list(loaded.errors)
    run_info = load_run(run_dir)
    glass_box = is_glass_box(run_info)
    path = labels_file(run_dir, metric)
    if metric not in JUDGE_METRICS:
        errors.append(Issue("--metric", f"use one of {', '.join(JUDGE_METRICS)}"))
    elif metric == "faithfulness" and not glass_box:
        errors.append(Issue("--metric", "faithfulness needs a glass-box run, one whose app config reads "
                                        "retrieved passages"))
    if loaded.sha256 is not None and loaded.sha256 != run_info["dataset"]["sha256"]:
        errors.append(Issue("dataset", "the dataset changed since this run, so its labels no longer match it"))
    if path.exists() and not replace:
        errors.append(Issue("label", f"{path.name} already exists and may hold your labels; "
                                     "add --replace to write a new one"))
    if errors:
        show([f"Run        {run_info['name']}"], [], errors)
        typer.echo(f"\nNo sheet written: {len(errors)} error(s).")
        raise typer.Exit(code=1)

    replies, eligible = pick_replies(loaded, load_rows(run_dir), metric, glass_box, sample, run_info.get("seed") or 0)
    if not replies:
        typer.echo(f"Nothing to label: no reply in this run is graded by the judge for {metric}.")
        raise typer.Exit(code=1)
    write_sheet(path, loaded, replies, metric)
    values = label_values(metric)
    allowed = " or ".join(values) if len(values) == 2 else "one of " + ", ".join(values)
    lines = [f"Run        {run_info['name']}",
             f"Sheet      {path}",
             f"Replies    {len(replies)} of the {eligible} the judge grades for {metric}",
             "",
             f"Fill in the label column with {allowed} for each row,",
             "using the same rules as the judge (docs/labeling-guide.md). A blank label skips the row.",
             "Label before looking at the judge's verdicts, and save the sheet as CSV. Then litmus report",
             "compares the judge with your labels, for every run judged with the same model and prompt."]
    warnings = []
    if len(replies) < SUGGESTED_LABELS:
        warnings.append(Issue("sample", f"only {len(replies)} replies; agreement from fewer than "
                                        f"{SUGGESTED_LABELS} labels is rough"))
    if loaded.manifest is not None and loaded.manifest.private:
        warnings.append(Issue("private", "the sheet holds question and answer text; keep it in this folder"))
    show(lines, warnings, [])


@app.command()
def report(run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, metavar="RUN")]) -> None:
    """Show the metrics for one run, and save them as metrics.json in the run folder."""
    try:
        built = build_report(run_dir)
    except ReportError as error:
        typer.echo(f"Can't report: {error}.")
        raise typer.Exit(code=1)
    show(format_report(built), built.warnings, [])
    typer.echo(f"\nSaved      {save_metrics(built, run_dir)}")


@app.command()
def compare(
    baseline: Annotated[Path, typer.Argument(exists=True, file_okay=False, metavar="BASELINE")],
    candidate: Annotated[Path, typer.Argument(exists=True, file_okay=False, metavar="CANDIDATE")],
    gates: Annotated[Optional[Path], typer.Option(exists=True, help="Gate rules (YAML); exit non-zero on failure.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Compare even if the runs aren't comparable.")] = False,
) -> None:
    """Compare two runs question by question, with a verdict per metric."""
    reports = []
    for folder in (baseline, candidate):
        try:
            reports.append(build_report(folder))
        except ReportError as error:
            typer.echo(f"Can't compare: {error}.")
            raise typer.Exit(code=1)
    before, after = reports
    errors, warnings = check_comparable(before, after)
    gate_list = []
    if gates is not None:
        gate_list, gate_errors = load_gates(gates)
        if gate_errors:
            show([], [], gate_errors)
            typer.echo(f"\nNot compared: the gates file has {len(gate_errors)} error(s).")
            raise typer.Exit(code=1)
    if errors and not force:
        show([], warnings, errors)
        typer.echo("\nNot comparable. Add --force to compare anyway.")
        raise typer.Exit(code=1)
    warnings = [Issue(e.where, f"{e.message} (compared anyway, --force)") for e in errors] + warnings

    changes = compare_reports(before, after, {g.metric: g.min_effect for g in gate_list if g.min_effect is not None})
    lines = format_comparison(before, after, changes)
    results = check_gates(gate_list, changes, after) if gate_list else []
    if gate_list:
        lines += format_gates(results, gates)
    show(lines, warnings, [])
    if any(not result.passed for result in results):
        raise typer.Exit(code=1)
