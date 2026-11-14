import pytest

from conftest import X200
from litmus.appconfig import check_allowed, check_env, load_app_config, resolve_env
from litmus.dataset import Manifest

MINIMAL = "adapter: http\nrequest:\n  url: https://bot.test/chat\nresponse:\n  actual_output: answer\n"


def load(tmp_path, text):
    path = tmp_path / "app.yaml"
    path.write_text(text, encoding="utf-8")
    return load_app_config(path)


def test_x200_config_loads():
    config, issues = load_app_config(X200 / "app.yaml")
    assert issues == []
    assert config.name == "x200-support-bot"
    assert config.request.method == "POST"
    assert config.request.timeout_s == 60
    assert config.response["retrieved_context_ids"] == "debug.retrieved[*].id"
    assert config.behavior_map["need_more_info"] == "clarify"
    assert (config.limits.concurrency, config.limits.requests_per_minute, config.limits.max_requests) == (1, 20, 300)
    assert config.retries == 0


def test_defaults(tmp_path):
    config, issues = load(tmp_path, MINIMAL)
    assert issues == []
    assert config.name == "app"
    assert config.request.headers == {} and config.request.body is None
    assert (config.limits.concurrency, config.limits.requests_per_minute, config.limits.max_requests) == (1, None, None)


@pytest.mark.parametrize("text, expected", [
    ("adapter: ftp\n", "'adapter' must be one of"),
    (MINIMAL + "retry: 2\n", "unknown setting 'retry'"),
    ("adapter: http\nresponse:\n  actual_output: a\n", "'request' is required"),
    ("adapter: http\nrequest:\n  url: bot.test\nresponse:\n  actual_output: a\n", "must start with http"),
    (MINIMAL.replace("url:", "method: DELETE\n  url:"), "'request.method' must be one of"),
    (MINIMAL.replace("url:", "headers:\n    X-Count: 3\n  url:"), "'request.headers' must map"),
    (MINIMAL.replace("url:", "timeout_s: 0\n  url:"), "'request.timeout_s' must be a positive"),
    (MINIMAL.replace("url:", "method: GET\n  body: [1]\n  url:"), "for GET, 'request.body'"),
    ("adapter: http\nrequest:\n  url: https://x.test\nresponse:\n  status: s\n", "'response.actual_output' is required"),
    (MINIMAL + "  answer_text: a\n", "unknown response field 'answer_text'"),
    (MINIMAL.replace("answer", "reply[?"), "not a valid JMESPath expression"),
    (MINIMAL + "behavior_map:\n  ok: answered\n", "'behavior_map.ok' must be one of"),
    (MINIMAL + "limits:\n  concurrency: 0\n", "'limits.concurrency' must be a whole number"),
    (MINIMAL + "limits:\n  max_calls: 5\n", "unknown limit 'max_calls'"),
    (MINIMAL + "retries: -1\n", "'retries' must be a whole number"),
])
def test_config_errors(tmp_path, text, expected):
    config, issues = load(tmp_path, text)
    assert config is None
    assert any(expected in str(issue) for issue in issues), [str(i) for i in issues]


def test_behavior_map_keys_are_compared_as_text(tmp_path):
    config, _ = load(tmp_path, MINIMAL + "behavior_map:\n  200: answer\n  true: refuse\n")
    assert config.behavior_map == {"200": "answer", "True": "refuse"}


def test_resolve_env():
    value = {"Authorization": "Bearer ${TOKEN}", "list": ["${A}", 3], "other": "${MISSING}"}
    resolved, missing = resolve_env(value, {"TOKEN": "t0k", "A": "a"})
    assert resolved == {"Authorization": "Bearer t0k", "list": ["a", 3], "other": "${MISSING}"}
    assert missing == ["MISSING"]


def test_check_env_names_missing_variables():
    config, _ = load_app_config(X200 / "app.yaml")
    assert [str(i) for i in check_env(config, {})] == ["app.yaml: environment variable X200_BOT_TOKEN is not set"]
    assert check_env(config, {"X200_BOT_TOKEN": "secret"}) == []


def test_allowed_hosts(tmp_path):
    config, _ = load(tmp_path, MINIMAL)
    assert check_allowed(Manifest(name="d", version=1), config, {}) == []
    assert check_allowed(Manifest(name="d", version=1, allowed_hosts=["bot.test"]), config, {}) == []
    [issue] = check_allowed(Manifest(name="d", version=1, allowed_hosts=[]), config, {})
    assert "host bot.test isn't in the dataset's allowed_hosts" in issue.message


def test_allowed_hosts_needs_a_written_out_host(tmp_path):
    config, _ = load(tmp_path, MINIMAL.replace("https://bot.test", "https://${HOST}"))
    [issue] = check_allowed(Manifest(name="d", version=1, allowed_hosts=["bot.test"]), config, {})
    assert "must be written out" in issue.message


OPENAI = "adapter: openai\nprovider: openai\nbase_url: https://api.test/v1\nmodel: m\n"


def test_openai_defaults(tmp_path):
    config, issues = load(tmp_path, OPENAI)
    assert issues == []
    settings = config.openai
    assert (settings.base_url, settings.model, settings.api_key_env) == ("https://api.test/v1", "m", None)
    assert settings.system_prompt is None and settings.user_template == "{{input}}"
    assert (settings.temperature, settings.timeout_s, settings.extra, settings.json_mode) == (0, 60, {}, False)
    assert settings.passages_from is None
    assert config.request is None and config.response == {} and config.provider == "openai"


def test_openai_settings(tmp_path):
    config, issues = load(tmp_path, OPENAI + "api_key_env: OPENAI_API_KEY\ntemperature: null\ntimeout_s: 90\n"
                                             "extra:\n  max_tokens: 500\njson_mode: true\n"
                                             "passages_from: runs/2026-11-20_glass\n"
                                             'user_template: "{{passages}} {{input}}"\n')
    assert issues == []
    settings = config.openai
    assert settings.api_key_env == "OPENAI_API_KEY" and settings.temperature is None and settings.timeout_s == 90
    assert settings.extra == {"max_tokens": 500} and settings.json_mode is True
    assert settings.passages_from == tmp_path / "runs" / "2026-11-20_glass"  # relative to the config file


@pytest.mark.parametrize("text, expected", [
    ("adapter: openai\nbase_url: https://api.test/v1\nmodel: m\n", "'provider' is required for the openai adapter"),
    (OPENAI.replace("https://api.test/v1", "api.test"), "'base_url' must start with http"),
    (OPENAI.replace("model: m\n", ""), "'model' is required"),
    (OPENAI + "api_key_env: MY-KEY\n", "'api_key_env' must be the name of an environment variable"),
    (OPENAI + "temperature: -1\n", "'temperature' must be a number"),
    (OPENAI + "timeout_s: 0\n", "'timeout_s' must be a positive number"),
    (OPENAI + "extra: [1]\n", "'extra' must be a list of settings"),
    (OPENAI + "extra:\n  model: other\n", "'extra' can't set model or messages"),
    (OPENAI + "json_mode: yes please\n", "'json_mode' must be true or false"),
    (OPENAI + "system_prompt: ''\n", "'system_prompt' must be text"),
    (OPENAI + "passages_from: 3\n", "'passages_from' must be a path"),
    (OPENAI + "request:\n  url: https://x.test\n", "'request' is only for the http adapter"),
    (OPENAI + "response:\n  status: s\n", "'response.actual_output' is required"),
    (MINIMAL + "model: m\n", "'model' is only for the openai adapter"),
])
def test_openai_config_errors(tmp_path, text, expected):
    config, issues = load(tmp_path, text)
    assert config is None
    assert any(expected in str(issue) for issue in issues), [str(i) for i in issues]


def test_check_env_names_the_openai_key(tmp_path):
    config, _ = load(tmp_path, OPENAI + "api_key_env: OPENAI_API_KEY\n")
    assert [issue.message for issue in check_env(config, {})] == ["environment variable OPENAI_API_KEY is not set"]
    assert check_env(config, {"OPENAI_API_KEY": "k"}) == []
    config, _ = load(tmp_path, OPENAI)  # local servers often need no key
    assert check_env(config, {}) == []


def test_allowed_providers(tmp_path):
    config, _ = load(tmp_path, "adapter: openai\nprovider: openai\nbase_url: https://api.test/v1\nmodel: m\n")
    manifest = Manifest(name="d", version=1, allowed_providers=["anthropic"])
    [issue] = check_allowed(manifest, config, {})
    assert "provider openai isn't in the dataset's allowed_providers" in issue.message
