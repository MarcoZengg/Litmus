import pytest

from conftest import X200
from litmus.appconfig import load_app_config
from litmus.dataset import Record, load_dataset
from litmus.templates import Placeholder, check_request_templates, find_placeholders, parse_placeholder, render


def config_from(tmp_path, text):
    path = tmp_path / "app.yaml"
    path.write_text(text, encoding="utf-8")
    config, issues = load_app_config(path)
    assert issues == [], issues
    return config


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
    config, _ = load_app_config(X200 / "app.yaml")
    assert check_request_templates(config, load_dataset(X200).records) == []


def test_required_metadata_missing_from_records(tmp_path):
    config = config_from(tmp_path, 'adapter: http\nrequest:\n  url: https://x.test\n  body:\n'
                                   '    category: "{{metadata.category}}"\nresponse:\n  actual_output: answer\n')
    records = [Record(id=f"q-{n}", input="q") for n in range(12)]
    records[0].metadata = {"category": "air purifier"}
    [issue] = check_request_templates(config, records)
    assert issue.where == "metadata.category"
    assert "11 record(s)" in issue.message
    assert "q-1, q-2" in issue.message and "and 1 more" in issue.message
    assert "{{metadata.category?}}" in issue.message


def test_configs_without_request_templates_pass(tmp_path):
    config = config_from(tmp_path, "adapter: python\nfunction: my_app.eval:answer\n")
    assert check_request_templates(config, []) == []


def test_render_fills_input_and_metadata():
    record = Record(id="a", input="How loud is it?", metadata={"category": "purifier", "limit": 5})
    template = {"message": "{{input}}", "category": "{{metadata.category}}", "limit": "{{metadata.limit}}",
                "prompt": "Q ({{metadata.category}}): {{input}}", "fixed": 1}
    assert render(template, record) == {"message": "How loud is it?", "category": "purifier", "limit": 5,
                                        "prompt": "Q (purifier): How loud is it?", "fixed": 1}


def test_render_drops_missing_optional_keys():
    record = Record(id="a", input="q")
    template = {"message": "{{input}}", "category": "{{metadata.category?}}",
                "tags": ["x", "{{metadata.tag?}}"], "note": "cat={{metadata.category?}}"}
    assert render(template, record) == {"message": "q", "tags": ["x"], "note": "cat="}
    assert render("{{metadata.category?}}", record) is None


def test_render_refuses_missing_required_keys():
    with pytest.raises(ValueError, match="a has no metadata.category"):
        render({"category": "{{metadata.category}}"}, Record(id="a", input="q"))


def test_render_never_expands_environment_variables_in_questions():
    assert render({"message": "{{input}}"}, Record(id="a", input="${HOME}")) == {"message": "${HOME}"}
