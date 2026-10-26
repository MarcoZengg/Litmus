"""Compare two runs question by question, and check CI gates (docs/design.md, section 7).

Each metric's change is the candidate minus the baseline over the questions both
runs scored, with a 95% range from a paired bootstrap. Like reports, the output
holds numbers only: no question or answer text, and no question IDs.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from litmus.dataset import Issue
from litmus.labels import judge_key
from litmus.metrics import QUESTION_METRICS, call_metric_values
from litmus.report import JUDGE_SCORED, Report, format_value
from litmus.stats import paired_change

# Rates and scores between 0 and 1: the minimum change is in absolute points.
PROPORTIONS = {*QUESTION_METRICS, *JUDGE_SCORED, "errors"}
KNOWN_METRICS = PROPORTIONS | set(call_metric_values([]))
MIN_EFFECT = 0.02  # 2 percentage points
RELATIVE_MIN_EFFECT = 0.10  # latency, cost and tokens: 10% of the baseline
GATE_FIELDS = {"min_effect", "max", "min"}


@dataclass
class Change:
    metric: str
    baseline: float  # both values cover the questions scored in both runs
    candidate: float
    change: float
    low: float  # 95% range of the change
    high: float
    questions: int
    min_effect: float
    lower_is_better: bool

    @property
    def verdict(self) -> str:
        sign = -1 if self.lower_is_better else 1
        gain = sign * self.change
        low, high = sorted((sign * self.low, sign * self.high))
        if high < 0:  # worse, beyond noise
            return "regression" if -gain >= self.min_effect - 1e-12 else "small but real drop"
        if low > 0:
            return "improvement" if gain >= self.min_effect - 1e-12 else "small but real gain"
        return "no detectable change"

    @property
    def noise(self) -> float:
        """Half the range: changes smaller than this can't be told apart from noise."""
        return (self.high - self.low) / 2


def default_min_effect(metric: str, baseline: float) -> float:
    return MIN_EFFECT if metric in PROPORTIONS else RELATIVE_MIN_EFFECT * abs(baseline)


def check_comparable(baseline: Report, candidate: Report) -> tuple[list[Issue], list[Issue]]:
    """(reasons the runs can't be compared, warnings)."""
    errors, warnings = [], []
    before, after = baseline.run["dataset"], candidate.run["dataset"]
    if before["sha256"] != after["sha256"]:
        errors.append(Issue("dataset", "the runs used different versions of dataset.jsonl, so their scores "
                                       "measure different things"))
    if before.get("kb_version") != after.get("kb_version"):
        warnings.append(Issue("kb_version", f"the dataset's kb_version was {before.get('kb_version')} for the "
                                            f"baseline and {after.get('kb_version')} for the candidate"))
    for report, which in ((baseline, "baseline"), (candidate, "candidate")):
        if report.run["status"] != "complete":
            warnings.append(Issue(which, "the run is incomplete, so only the questions it finished are compared"))
    judged = [report.judging for report in (baseline, candidate)]
    if all(judged):
        for metric in sorted(set(judged[0]["metrics"]) & set(judged[1]["metrics"])):
            if judge_key(judged[0], metric) != judge_key(judged[1], metric):
                errors.append(Issue("judge", f"{metric} was judged with a different provider, model or prompt "
                                             "in each run"))
    elif any(judged):
        which = "baseline" if judged[0] else "candidate"
        warnings.append(Issue("judge", f"only the {which} run was judged, so judge metrics aren't compared"))
    return errors, warnings


def compare_reports(baseline: Report, candidate: Report, min_effects: dict[str, float] | None = None
                    ) -> list[Change]:
    """One change per metric measured in both runs, in report order. Both runs' seeds seed the bootstrap,
    so a comparison can be reproduced exactly."""
    seed = [baseline.run.get("seed") or 0, candidate.run.get("seed") or 0]
    min_effects = min_effects or {}
    both_judged = baseline.judging is not None and candidate.judging is not None
    changes = []
    for metric in baseline.metrics:
        name = metric.name
        if name not in baseline.values or name not in candidate.values:
            continue
        if name in JUDGE_SCORED and not both_judged:
            continue
        found = paired_change(baseline.values[name], candidate.values[name], seed)
        if found is None:
            continue
        before, after, change, low, high, questions = found
        changes.append(Change(name, before, after, change, low, high, questions,
                              min_effects.get(name, default_min_effect(name, before)), metric.lower_is_better))
    return changes


@dataclass
class Gate:
    metric: str
    min_effect: float | None = None  # regression check; None: the default minimum change
    max: float | None = None  # caps on the candidate's value
    min: float | None = None

    @property
    def checks_regression(self) -> bool:
        return self.min_effect is not None or (self.max is None and self.min is None)


@dataclass
class GateResult:
    metric: str
    passed: bool
    reason: str


def load_gates(path: Path) -> tuple[list[Gate], list[Issue]]:
    """A gates file maps metric names to {min_effect, max, min}. An empty entry checks for regressions
    with the default minimum change; an entry with only max or min checks just that cap."""
    where = path.name
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        return [], [Issue(where, f"isn't valid YAML ({error})")]
    if not isinstance(data, dict) or not isinstance(data.get("gates"), dict) or not data["gates"]:
        return [], [Issue(where, "needs a 'gates' mapping of metric names, e.g. gates: {correctness: {}}")]
    gates, issues = [], []
    for unknown in sorted(set(data) - {"gates"}):
        issues.append(Issue(where, f"unknown setting '{unknown}'"))
    for metric, settings in data["gates"].items():
        settings = settings or {}
        if metric not in KNOWN_METRICS:
            issues.append(Issue(where, f"'{metric}' isn't a metric Litmus reports"))
            continue
        if not isinstance(settings, dict):
            issues.append(Issue(where, f"{metric}: use a mapping such as {{min_effect: 0.02}} or {{max: 0.05}}"))
            continue
        for unknown in sorted(set(settings) - GATE_FIELDS):
            issues.append(Issue(where, f"{metric}: unknown setting '{unknown}'"))
        numbers = {}
        for key in GATE_FIELDS & set(settings):
            value = settings[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or (key == "min_effect" and value < 0):
                issues.append(Issue(where, f"{metric}: '{key}' must be a number"
                                           + (" of 0 or more" if key == "min_effect" else "")))
            else:
                numbers[key] = float(value)
        gates.append(Gate(metric, **numbers))
    return gates, issues


def check_gates(gates: list[Gate], changes: list[Change], candidate: Report) -> list[GateResult]:
    """Only overall numbers gate. A gate whose metric wasn't measured fails, so a misconfigured run
    can't pass CI by accident."""
    by_name = {change.metric: change for change in changes}
    values = {metric.name: metric.value for metric in candidate.metrics}
    results = []
    for gate in gates:
        problems, notes = [], []
        value = values.get(gate.metric)
        for cap, broken, word in ((gate.max, lambda v, c: v > c, "above"), (gate.min, lambda v, c: v < c, "below")):
            if cap is None:
                continue
            if value is None:
                problems.append("not measured in the candidate run")
            elif broken(value, cap):
                problems.append(f"{format_value(gate.metric, value)} is {word} the "
                                f"{'cap' if word == 'above' else 'floor'} of {format_value(gate.metric, cap)}")
            else:
                notes.append(f"{format_value(gate.metric, value)} is within the "
                             f"{'cap' if word == 'above' else 'floor'} of {format_value(gate.metric, cap)}")
        if gate.checks_regression:
            change = by_name.get(gate.metric)
            if change is None:
                problems.append("not measured in both runs, so a regression can't be ruled out")
            elif change.verdict == "regression":
                problems.append(f"regression of {format_change(change.metric, change.change)} "
                                f"(range {format_change(change.metric, change.low, False)} to "
                                f"{format_change(change.metric, change.high, False)})")
            else:
                notes.append(f"{change.verdict} (a regression needs a worsening of at least "
                             f"{format_size(change.metric, change.min_effect)})")
        results.append(GateResult(gate.metric, not problems, "; ".join(problems or notes)))
    return results


def format_change(metric: str, value: float, unit: bool = True) -> str:
    if metric == "mrr":
        return f"{value:+.2f}"
    if metric in PROPORTIONS:
        return f"{value * 100:+.1f}" + (" pts" if unit else "")
    if metric.startswith("latency"):
        return f"{value:+.0f}" + (" ms" if unit else "")
    if metric == "cost_per_1000":
        return f"{value:+.4g}"
    return f"{value:+.1f}"


def format_size(metric: str, value: float) -> str:
    return format_change(metric, value).lstrip("+")


def format_comparison(baseline: Report, candidate: Report, changes: list[Change]) -> list[str]:
    def describe(report: Report) -> str:
        run = report.run
        repeats = f"{run['repeats']} repeat{'s' if run['repeats'] != 1 else ''}"
        return f"{run['name']} ({run['status']}, {repeats}, {run['counts']['written']} calls)"

    manifest = candidate.dataset.manifest
    same = baseline.run["dataset"]["sha256"] == candidate.run["dataset"]["sha256"]
    lines = [f"Baseline   {describe(baseline)}",
             f"Candidate  {describe(candidate)}",
             f"Dataset    {manifest.name} (version {manifest.version})"
             + (", the same questions and labels in both runs" if same else ", DIFFERENT in each run"),
             "", f"{'Metric':<24}{'Baseline':>10}{'Candidate':>11}   {'Change (95% range)':<32}Verdict"]
    for change in changes:
        text = (f"{format_change(change.metric, change.change)} ({format_change(change.metric, change.low, False)} "
                f"to {format_change(change.metric, change.high, False)})")
        lines.append(f"{change.metric:<24}{format_value(change.metric, change.baseline):>10}"
                     f"{format_value(change.metric, change.candidate):>11}   {text:<32}{change.verdict}")
    if not changes:
        lines.append("  (no metric was measured in both runs)")
    example = next((c for c in changes if c.metric == "behavior_accuracy"), changes[0] if changes else None)
    if example is not None:
        lines += ["", f"Noise      changes smaller than about half the range can't be told apart from noise: "
                      f"here about ±{format_size(example.metric, example.noise)}",
                  f"           for {example.metric}, over {example.questions} questions. More questions narrow it."]
    return lines


def format_gates(results: list[GateResult], path: Path) -> list[str]:
    lines = ["", f"Gates      {path.name}"]
    width = max(24, *(len(result.metric) + 2 for result in results))
    for result in results:
        lines.append(f"  {result.metric:<{width}}{'pass' if result.passed else 'FAIL':<6}{result.reason}")
    failed = sum(not result.passed for result in results)
    lines.append(f"Result     {'FAIL' if failed else 'PASS'}: {failed} of {len(results)} gate(s) failed")
    return lines
