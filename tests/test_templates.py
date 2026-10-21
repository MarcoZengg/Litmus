import pytest

from conftest import X200
from litmus.dataset import Record, load_dataset
from litmus.templates import Placeholder, check_app_templates, find_placeholders, parse_placeholder


@pytest.mark.parametrize("expression, expected", [
    ("input", Placeholder("input")),
    (" input ", Placeholder("input")),
    ("metadata.category", Placeholder("metadata", "category")),
    ("metadata.category?", Placeholder("metadata", "category", optional=True)),
    ("metadata.user_id", Placeholder("metadata", "user_id")),
    ("question", None),
    ("input?", None),
    ("metadata.", None),
    ("metadata.a.b", None),
])
def test_parse_placeholder(expression, expected):
    assert parse_placeholder(expression) == expected


def test_find_placeholders_in_nested_templates():
    template = {"body": {"message": "Q: {{input}}", "tags": ["{{metadata.tag?}}"]}, "url": "https://x"}
    found, issues = find_placeholders(template)
    assert found == [Placeholder("input"), Placeholder("metadata", "tag", optional=True)]
    assert issues == []


def test_malformed_placeholders_are_reported():
    _, issues = find_placeholders({"body": {"a": "{{question}}", "b": "{{input}", "c": "{{input}} }}"}})
    assert [str(i) for i in issues] == [
        "body.a: unknown placeholder {{question}}; use {{input}} or {{metadata.<key>}}",
        "body.b: unbalanced {{ or }}",
        "body.c: unbalanced {{ or }}",
    ]


def test_x200_app_config_matches_its_dataset():
    assert check_app_templates(X200 / "app.yaml", load_dataset(X200).records) == []


def test_required_metadata_missing_from_records(tmp_path):
    config = tmp_path / "app.yaml"
    config.write_text('request:\n  body:\n    category: "{{metadata.category}}"\n', encoding="utf-8")
    records = [Record(id=f"q-{n}", input="q") for n in range(12)]
    records[0].metadata = {"category": "air purifier"}
    [issue] = check_app_templates(config, records)
    assert issue.where == "metadata.category"
    assert "11 record(s)" in issue.message
    assert "q-1, q-2" in issue.message and "and 1 more" in issue.message
    assert "{{metadata.category?}}" in issue.message


def test_configs_without_request_templates_pass(tmp_path):
    config = tmp_path / "app.yaml"
    config.write_text("adapter: python\nfunction: my_app.eval:answer\n", encoding="utf-8")
    assert check_app_templates(config, []) == []


def test_invalid_config_yaml(tmp_path):
    config = tmp_path / "app.yaml"
    config.write_text("request: [", encoding="utf-8")
    assert [str(i) for i in check_app_templates(config, [])] == ["app.yaml: not valid YAML"]
