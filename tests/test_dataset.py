import hashlib

import pytest

from conftest import X200, write_records
from litmus.dataset import load_dataset


def messages(issues):
    return [str(issue) for issue in issues]


def test_x200_examples_load():
    dataset = load_dataset(X200)
    assert dataset.ok, messages(dataset.errors)
    assert dataset.warnings == []
    assert dataset.manifest.name == "x200-examples"
    assert dataset.manifest.kb_version == "2026-10-15"
    assert [r.id for r in dataset.records] == ["q-0001", "q-0002", "q-0003", "q-0004"]
    assert dataset.sha256 == hashlib.sha256((X200 / "dataset.jsonl").read_bytes()).hexdigest()


def test_passage_groups_are_normalized():
    records = {r.id: r for r in load_dataset(X200).records}
    assert records["q-0001"].reference_context_ids == [["x200-manual#p12"]]
    assert records["q-0002"].reference_context_ids == [["x200-manual#p4", "x200-faq#7"], ["x200-manual#p15"]]
    assert records["q-0003"].reference_context_ids == []


def test_missing_passages_differ_from_empty(x200):
    write_records(x200, '{"id": "a", "input": "q", "reference_context_ids": []}', '{"id": "b", "input": "q"}')
    records = {r.id: r for r in load_dataset(x200).records}
    assert records["a"].reference_context_ids == []
    assert records["b"].reference_context_ids is None


def test_language_defaults_to_the_manifest():
    records = {r.id: r for r in load_dataset(X200).records}
    assert records["q-0002"].language == "en"


def test_a_file_inside_the_folder_works_too():
    assert load_dataset(X200 / "dataset.jsonl").ok


@pytest.mark.parametrize("line, expected", [
    ('{"input": "q"}', "line 1: 'id' is required"),
    ('{"id": "a", "input": "  "}', "a: 'input' is required and must be non-empty text"),
    ('{"id": "a", "input": "q", "expected_behavior": "anwser"}', "a: 'expected_behavior' must be one of"),
    ('{"id": "a", "input": "q", "source": "customer"}', "a: 'source' must be one of"),
    ('{"id": "a", "input": "q", "expected_behaviour": "answer"}', "a: unknown field 'expected_behaviour'"),
    ('{"id": "a", "input": "q", "turns": []}', "multi-turn records come after v1"),
    ('{"id": "a", "input": "q", "reference_context_ids": "p1"}', "a: 'reference_context_ids' must be"),
    ('{"id": "a", "input": "q", "reference_context_ids": [[]]}', "a: 'reference_context_ids' must be"),
    ('{"id": "a", "input": "q", "tags": "battery"}', "a: 'tags' must be a list of text"),
    ('{"id": "a", "input": "q", "metadata": {"category": {"nested": 1}}}', "a: 'metadata' must map names"),
    ('{"id": "a", "input": "q", "expected_output": 6}', "a: 'expected_output' must be text"),
    ('{"id": "a", "input": "q"', "line 1: not valid JSON"),
    ('["a", "q"]', "line 1: must be a JSON object"),
])
def test_record_errors(x200, line, expected):
    write_records(x200, line)
    dataset = load_dataset(x200)
    assert not dataset.ok
    assert any(expected in message for message in messages(dataset.errors)), messages(dataset.errors)


def test_duplicate_ids(x200):
    write_records(x200, '{"id": "a", "input": "q"}', '{"id": "a", "input": "q2"}')
    assert messages(load_dataset(x200).errors) == ["a: duplicate id (also on line 1)"]


def test_every_problem_is_reported_at_once(x200):
    write_records(x200, '{"id": "a", "input": "", "source": "x"}', "not json", '{"id": "b", "input": "q"}')
    dataset = load_dataset(x200)
    assert len(dataset.errors) == 3
    assert [r.id for r in dataset.records] == ["b"]


def test_blank_lines_and_byte_order_mark_are_fine(x200):
    (x200 / "dataset.jsonl").write_text('﻿{"id": "a", "input": "q"}\n\n{"id": "b", "input": "q"}\n',
                                        encoding="utf-8")
    assert [r.id for r in load_dataset(x200).records] == ["a", "b"]


def test_answer_with_nothing_to_score_against_warns(x200):
    write_records(x200, '{"id": "a", "input": "q", "expected_behavior": "answer"}')
    dataset = load_dataset(x200)
    assert dataset.ok
    assert messages(dataset.warnings) == [
        "a: expects an answer but has no expected_output or reference_context_ids, "
        "so only a rubric judge can score it"]


def test_errors_never_include_question_text(x200):
    write_records(x200, '{"id": "a", "input": "SECRET QUESTION", "source": "nope"}')
    assert not any("SECRET" in message for message in messages(load_dataset(x200).errors))


@pytest.mark.parametrize("manifest, expected", [
    ("version: 1\n", "'name' is required"),
    ("name: x\n", "'version' is required"),
    ("name: x\nversion: 1\nprivte: true\n", "unknown setting 'privte'"),
    ("name: x\nversion: 1\nprivate: yes please\n", "'private' must be true or false"),
    ("name: x\nversion: 1\nkb_version: 2026-10-15\n", "put dates in quotes"),
    ("name: x\nversion: 1\nallowed_providers: openai\n", "'allowed_providers' must be a list"),
    ("name: [x\n", "not valid YAML"),
])
def test_manifest_errors(x200, manifest, expected):
    (x200 / "dataset.yaml").write_text(manifest, encoding="utf-8")
    dataset = load_dataset(x200)
    assert dataset.manifest is None
    assert any(expected in message for message in messages(dataset.errors)), messages(dataset.errors)


def test_missing_files(tmp_path):
    assert messages(load_dataset(tmp_path).errors) == [
        "dataset.yaml: file not found", "dataset.jsonl: file not found"]


def test_private_manifest(x200):
    (x200 / "dataset.yaml").write_text(
        "name: x\nversion: 2\nprivate: true\nallowed_providers: [openai]\nallowed_hosts: []\n",
        encoding="utf-8")
    manifest = load_dataset(x200).manifest
    assert manifest.private is True
    assert manifest.allowed_providers == ["openai"]
    assert manifest.allowed_hosts == []
