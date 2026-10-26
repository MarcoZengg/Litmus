import csv
import json

import pytest
import yaml
from typer.testing import CliRunner

from conftest import X200
from litmus.cli import app
from litmus.dataset import load_dataset
from litmus.judge import judge_run, load_judge_config, plan_judgments
from litmus.labels import cohen_kappa, labels_file, load_labels
from litmus.report import build_report
from litmus.runner import is_glass_box, load_rows, load_run
from test_judge import make_client, make_run, row

runner = CliRunner()


def label(folder, *args):
    return runner.invoke(app, ["label", str(folder), *args])


def read_sheet(path):
    with open(path, encoding="utf-8-sig", newline="") as sheet:
        return list(csv.DictReader(sheet))


def fill(path, labels):
    """Fill in the label column the way a person would, by (id, repeat)."""
    rows = read_sheet(path)
    for line in rows:
        line["label"] = labels.get((line["id"], int(line["repeat"])), "")
    with open(path, "w", encoding="utf-8-sig", newline="") as sheet:
        writer = csv.DictWriter(sheet, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def judge(folder, config_path=X200 / "judge.yaml"):
    config, issues = load_judge_config(config_path)
    assert issues == []
    tasks = plan_judgments(load_dataset(folder.parent.parent), load_rows(folder), config,
                           is_glass_box(load_run(folder)))
    judge_run(folder, tasks, config, make_client(config))


# --- the sheet ---

@pytest.mark.parametrize("metric, ids, extra", [
    ("correctness", ["q-0001"], ["reference"]),
    ("faithfulness", ["q-0001"], ["passages"]),
    ("rubric", ["q-0001", "q-0002", "q-0003", "q-0004"], []),
    ("behavior", ["q-0003"], []),
])
def test_sheet_holds_the_replies_the_judge_grades(x200, metric, ids, extra):
    # The test_judge rows: q-0001's repeat 2 failed, q-0002 handed off, q-0003's behavior is unknown.
    folder = make_run(x200)
    result = label(folder, "--metric", metric)
    assert result.exit_code == 0, result.output
    assert f"{len(ids)} of the {len(ids)} the judge grades for {metric}" in result.output
    path = labels_file(folder, metric)
    sheet = read_sheet(path)
    assert [line["id"] for line in sheet] == ids
    assert list(sheet[0]) == ["id", "repeat", "question", *extra, "answer", "label", "note"]
    assert all(line["label"] == "" for line in sheet)


def test_sheet_text(x200):
    folder = make_run(x200)
    label(folder, "--metric", "faithfulness")
    [line] = read_sheet(labels_file(folder, "faithfulness"))
    dataset = load_dataset(x200)
    assert line["question"] == dataset.records[0].input
    assert line["answer"] == "About 6 hours."
    assert line["passages"] == "[1] (x200-manual#p12) Battery: about 6 h at normal speed."


def test_sample_takes_one_repeat_of_each_question_first(x200):
    rows = ([row("q-0001", r, actual_output="About 6 hours.", observed_behavior="answer") for r in (1, 2, 3)]
            + [row("q-0002", r, actual_output="Please contact support.", observed_behavior="handoff") for r in (1, 2)])
    folder = make_run(x200, rows)
    result = label(folder, "--metric", "rubric", "--sample", "2")
    assert "2 of the 5 the judge grades for rubric" in result.output
    first = read_sheet(labels_file(folder, "rubric"))
    assert [line["id"] for line in first] == ["q-0001", "q-0002"]
    label(folder, "--metric", "rubric", "--sample", "2", "--replace")
    assert read_sheet(labels_file(folder, "rubric")) == first  # the run's seed makes it repeatable


def test_answers_that_look_like_formulas_stay_text(x200):
    folder = make_run(x200, [row("q-0001", 1, actual_output="- Charge it first.", observed_behavior="answer")])
    label(folder, "--metric", "rubric")
    assert read_sheet(labels_file(folder, "rubric"))[0]["answer"] == "'- Charge it first."


def test_label_never_overwrites_labels(x200):
    folder = make_run(x200)
    label(folder, "--metric", "rubric")
    fill(labels_file(folder, "rubric"), {("q-0001", 1): "pass"})
    result = label(folder, "--metric", "rubric")
    assert result.exit_code == 1 and "already exists and may hold your labels" in result.output
    assert read_sheet(labels_file(folder, "rubric"))[0]["label"] == "pass"


def test_label_errors(x200):
    folder = make_run(x200)
    result = label(folder, "--metric", "accuracy")
    assert result.exit_code == 1 and "use one of correctness, faithfulness, rubric, behavior" in result.output
    run = yaml.safe_load((folder / "run.yaml").read_text(encoding="utf-8"))
    run["app"]["config"]["response"] = {"actual_output": "reply.answer"}  # black-box
    (folder / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
    result = label(folder, "--metric", "faithfulness")
    assert result.exit_code == 1 and "faithfulness needs a glass-box run" in result.output
    assert not labels_file(folder, "faithfulness").exists()


def test_load_labels_is_forgiving(tmp_path):
    # Saved by a spreadsheet app with ';' and a non-UTF-8 encoding: only id, repeat and label are read.
    path = tmp_path / "labels-correctness.csv"
    path.write_bytes("id;repeat;question;label\n"
                     "q-0001;1;Wie lange hält der Akku?;PASS\n"
                     "q-0002;1;Größe?; fail \n"
                     "q-0003;1;x;maybe\n"
                     "q-0004;x;x;pass\n"
                     "q-0005;1;x;\n".encode("cp1252"))
    labels, issues = load_labels(path, "correctness")
    assert labels == {("q-0001", 1): "pass", ("q-0002", 1): "fail"}
    assert [str(i) for i in issues] == [
        f"{tmp_path.name}/labels-correctness.csv row 4: label 'maybe' isn't pass or fail; left out",
        f"{tmp_path.name}/labels-correctness.csv row 5: repeat isn't a number; left out",
    ]


def test_cohen_kappa():
    # 20 both pass, 5 you pass / judge fails, 10 you fail / judge passes, 15 both fail.
    pairs = [("pass", "pass")] * 20 + [("pass", "fail")] * 5 + [("fail", "pass")] * 10 + [("fail", "fail")] * 15
    assert cohen_kappa(pairs) == pytest.approx(0.4)
    assert cohen_kappa([("pass", "pass")] * 9 + [("fail", "pass")]) == pytest.approx(0)  # always-pass judge
    assert cohen_kappa([("pass", "pass")] * 5) is None


# --- the judge check in reports ---

def test_report_checks_the_judge_against_labels(x200):
    folder = make_run(x200)
    label(folder, "--metric", "rubric")
    # The fake judge passes every rubric reply; you fail q-0003.
    fill(labels_file(folder, "rubric"), {("q-0001", 1): "pass", ("q-0002", 1): "pass", ("q-0003", 1): "fail",
                                         ("q-0004", 1): "pass"})
    judge(folder)
    report = build_report(folder)
    check = report.judge_checks["rubric"]
    assert (check.labels, check.agreement, check.lenient, check.strict) == (4, 0.75, 1, 0)
    assert check.kappa == pytest.approx(0)
    assert (check.runs, check.dataset_versions) == (["baseline"], ["1"])
    assert {m for m, c in report.judge_checks.items() if c is None} == {"correctness", "faithfulness", "behavior"}
    warnings = [str(w) for w in report.warnings]
    assert ("judge: correctness, faithfulness, behavior: this judge model and prompt haven't been checked "
            "against hand labels; see litmus label") in warnings
    assert "judge: rubric: the judge agrees poorly with your labels (kappa 0.00), so treat rubric as rough" in warnings

    output = runner.invoke(app, ["report", str(folder)]).output
    assert ("  rubric                75.0% agreement, kappa 0.00, 4 labels (aim for 50); "
            "judge too lenient 1, too strict 0") in output
    assert "labels from baseline (dataset version 1)" in output
    assert "  correctness           not checked" in output
    assert "q-0003" not in output.split("Judge check")[1].split("Questions to look at")[0]
    saved = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    assert saved["judge_checks"]["rubric"]["labels"] == 4 and saved["judge_checks"]["correctness"] is None


def test_labels_check_every_run_judged_the_same_way(x200, tmp_path):
    labeled = make_run(x200, name="baseline")
    label(labeled, "--metric", "rubric")
    fill(labels_file(labeled, "rubric"), {("q-0001", 1): "pass", ("q-0002", 1): "pass"})
    judge(labeled)
    same = make_run(x200, name="candidate")
    judge(same)
    assert build_report(same).judge_checks["rubric"].runs == ["baseline"]

    (tmp_path / "rubric_v2.txt").write_text("Check these criteria: $criteria\n$question\n$answer", encoding="utf-8")
    config = yaml.safe_load((X200 / "judge.yaml").read_text(encoding="utf-8"))
    config["prompts"] = {"rubric": "rubric_v2.txt"}
    (tmp_path / "judge_v2.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    changed = make_run(x200, name="new-prompt")
    judge(changed, tmp_path / "judge_v2.yaml")
    report = build_report(changed)
    assert report.judge_checks["rubric"] is None
    assert any("rubric" in str(w) and "haven't been checked" in str(w) for w in report.warnings)
