import pytest

from conftest import X200
from litmus.appconfig import load_app_config
from litmus.dataset import Record, load_dataset
from litmus.templates import (Placeholder, check_request_templates, find_placeholders, format_passages,
                              parse_placeholder, render)


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
    ("passages", Placeholder("passages")),
    ("passages?", None),
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
        "body.a: unknown placeholder {{question}}; use {{input}}, {{metadata.<key>}} or {{passages}}",
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


def test_render_fills_passages():
    record = Record(id="a", input="q")
    assert render("P:\n{{passages}}\nQ: {{input}}", record, passages="[1] six hours") == "P:\n[1] six hours\nQ: q"
    with pytest.raises(ValueError, match="needs passages_from"):
        render("{{passages}}", record)


def test_format_passages():
    assert format_passages(["Battery: 6 h", "Noise: 24 dB"], ["p12", "p7"]) == "[1] (p12) Battery: 6 h\n[2] (p7) Noise: 24 dB"
    assert format_passages(["Battery: 6 h", "Noise: 24 dB"], ["p12"]) == "[1] Battery: 6 h\n[2] Noise: 24 dB"
    assert format_passages([]) == ""


OPENAI = "adapter: openai\nprovider: openai\nbase_url: https://api.test/v1\nmodel: m\n"


def test_openai_prompts_are_checked_like_request_templates(tmp_path):
    config = config_from(tmp_path, OPENAI + 'system_prompt: "You support {{metadata.product}}."\n'
                                            'user_template: "{{input}} {{metadata.extra?}} {{question}}"\n')
    issues = [str(issue) for issue in check_request_templates(config, [Record(id="q-1", input="q")])]
    assert any("user_template: unknown placeholder {{question}}" in issue for issue in issues), issues
    assert any(issue.startswith("metadata.product: the request template needs it, but 1 record(s)")
               for issue in issues), issues
    assert not any("metadata.extra" in issue for issue in issues)


def test_passages_placeholder_needs_passages_from(tmp_path):
    config = config_from(tmp_path, OPENAI + 'user_template: "{{passages}} {{input}}"\n')
    [issue] = check_request_templates(config, [])
    assert "{{passages}} needs passages_from" in issue.message
    http = config_from(tmp_path, 'adapter: http\nrequest:\n  url: https://x.test\n  body:\n'
                                 '    context: "{{passages}}"\nresponse:\n  actual_output: answer\n')
    [issue] = check_request_templates(http, [])
    assert "only the openai adapter has" in issue.message


def test_passages_from_needs_the_passages_placeholder(tmp_path):
    config = config_from(tmp_path, OPENAI + "passages_from: runs/r\n")
    [issue] = check_request_templates(config, [])
    assert "no prompt uses {{passages}}" in issue.message
