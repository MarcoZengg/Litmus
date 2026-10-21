"""App configs: how Litmus reaches an app (docs/design.md, section 4).

Secrets are written as ${NAME} and read from environment variables, so a
config file never contains them and can be saved with every run as is.
"""

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import jmespath
import yaml
from jmespath.exceptions import JMESPathError

from litmus.dataset import BEHAVIORS, Issue, Manifest

ADAPTERS = ("http", "openai", "python")
METHODS = ("GET", "POST", "PUT", "PATCH")
RESPONSE_FIELDS = ("actual_output", "retrieved_context_ids", "retrieval_context", "status", "cached",
                   "input_tokens", "output_tokens", "cost", "kb_version")
TOP_LEVEL_FIELDS = {"name", "adapter", "request", "response", "behavior_map", "limits", "retries",
                    # Used by the OpenAI-compatible and Python adapters, which come later.
                    "provider", "base_url", "model", "api_key_env", "system_prompt", "temperature", "function"}
REQUEST_FIELDS = {"method", "url", "headers", "body", "timeout_s"}
LIMIT_FIELDS = {"concurrency", "requests_per_minute", "max_requests"}
ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
SENSITIVE_NAME = re.compile(r"auth|token|key|secret|cookie|password|signature", re.IGNORECASE)
REDACTED = "<redacted>"


@dataclass
class HttpRequest:
    url: str
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    body: object = None
    timeout_s: float = 60.0


@dataclass
class Limits:
    concurrency: int = 1
    requests_per_minute: int | None = None
    max_requests: int | None = None


@dataclass
class AppConfig:
    path: Path
    name: str
    adapter: str
    raw: dict  # the file as written, with ${NAME} unresolved: safe to save with a run
    request: HttpRequest | None = None
    response: dict[str, str] = field(default_factory=dict)  # field -> JMESPath expression
    behavior_map: dict[str, str] = field(default_factory=dict)
    limits: Limits = field(default_factory=Limits)
    retries: int = 0
    provider: str | None = None


def load_app_config(path: Path) -> tuple[AppConfig | None, list[Issue]]:
    where = path.name
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (yaml.YAMLError, UnicodeDecodeError):
        return None, [Issue(where, "not valid YAML")]
    if not isinstance(data, dict):
        return None, [Issue(where, "must be a list of settings, e.g. 'adapter: http'")]

    problems = [f"unknown setting '{key}'" for key in sorted(set(map(str, data)) - TOP_LEVEL_FIELDS)]
    adapter = data.get("adapter")
    if adapter not in ADAPTERS:
        problems.append(f"'adapter' must be one of: {', '.join(ADAPTERS)}")
    name = data.get("name", path.stem)
    if not isinstance(name, str) or not name.strip():
        problems.append("'name' must be text")
    if "provider" in data and not isinstance(data["provider"], str):
        problems.append("'provider' must be text")

    request = _request(data.get("request"), problems) if adapter == "http" else None
    response = _response(data.get("response"), problems, required=adapter == "http")
    behavior_map = _behavior_map(data.get("behavior_map", {}), problems)
    limits = _limits(data.get("limits", {}), problems)
    retries = data.get("retries", 0)
    if not _is_int(retries, minimum=0):
        problems.append("'retries' must be a whole number, 0 or more")

    if problems:
        return None, [Issue(where, problem) for problem in problems]
    return AppConfig(path=path, name=name, adapter=adapter, raw=data, request=request, response=response,
                     behavior_map=behavior_map, limits=limits, retries=retries,
                     provider=data.get("provider")), []


def _request(value: object, problems: list[str]) -> HttpRequest | None:
    if not isinstance(value, dict):
        problems.append("'request' is required for the http adapter")
        return None
    problems += [f"unknown request setting '{key}'" for key in sorted(set(map(str, value)) - REQUEST_FIELDS)]
    url = value.get("url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        problems.append("'request.url' must start with http:// or https://")
    method = str(value.get("method", "POST")).upper()
    if method not in METHODS:
        problems.append(f"'request.method' must be one of: {', '.join(METHODS)}")
    headers = value.get("headers", {})
    if not isinstance(headers, dict) or not all(isinstance(v, str) for v in headers.values()):
        problems.append("'request.headers' must map header names to text")
    body = value.get("body")
    if method == "GET" and body is not None and not isinstance(body, dict):
        problems.append("for GET, 'request.body' must map query parameter names to values")
    timeout = value.get("timeout_s", 60)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        problems.append("'request.timeout_s' must be a positive number of seconds")
    return HttpRequest(url=url, method=method, headers=headers, body=body, timeout_s=timeout)


def _response(value: object, problems: list[str], required: bool) -> dict[str, str]:
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        problems.append("'response' must map fields to JMESPath expressions")
        return {}
    problems += [f"unknown response field '{key}'; use one of: {', '.join(RESPONSE_FIELDS)}"
                 for key in sorted(set(map(str, value)) - set(RESPONSE_FIELDS))]
    if required and "actual_output" not in value:
        problems.append("'response.actual_output' is required: where the answer is in the reply")
    for key, expression in value.items():
        if not isinstance(expression, str):
            problems.append(f"'response.{key}' must be a JMESPath expression")
            continue
        try:
            jmespath.compile(expression)
        except JMESPathError:
            problems.append(f"'response.{key}' is not a valid JMESPath expression")
    return dict(value)


def _behavior_map(value: object, problems: list[str]) -> dict[str, str]:
    if not isinstance(value, dict):
        problems.append("'behavior_map' must map the app's status values to behaviors")
        return {}
    for status, behavior in value.items():
        if behavior not in BEHAVIORS:
            problems.append(f"'behavior_map.{status}' must be one of: {', '.join(BEHAVIORS)}")
    # YAML reads `true:` or `200:` as non-text keys; statuses are compared as text.
    return {str(status): behavior for status, behavior in value.items()}


def _limits(value: object, problems: list[str]) -> Limits:
    if not isinstance(value, dict):
        problems.append("'limits' must be a list of settings")
        return Limits()
    problems += [f"unknown limit '{key}'" for key in sorted(set(map(str, value)) - LIMIT_FIELDS)]
    for key in LIMIT_FIELDS & set(value):
        if not _is_int(value[key], minimum=1):
            problems.append(f"'limits.{key}' must be a whole number, 1 or more")
    return Limits(concurrency=value.get("concurrency", 1),
                  requests_per_minute=value.get("requests_per_minute"),
                  max_requests=value.get("max_requests"))


def _is_int(value: object, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def resolve_env(value: object, env: Mapping[str, str]) -> tuple[object, list[str]]:
    """Replace ${NAME} in every string with the environment variable's value. Returns missing names."""
    missing: list[str] = []

    def walk(item: object) -> object:
        if isinstance(item, str):
            def lookup(match: re.Match) -> str:
                if match.group(1) not in env:
                    missing.append(match.group(1))
                    return match.group(0)
                return env[match.group(1)]
            return ENV_VAR.sub(lookup, item)
        if isinstance(item, dict):
            return {key: walk(val) for key, val in item.items()}
        if isinstance(item, list):
            return [walk(val) for val in item]
        return item

    resolved = walk(value)
    return resolved, sorted(set(missing))


def redacted_config(config: AppConfig) -> tuple[dict, list[Issue]]:
    """The config as written, safe to save with a run. A secret-looking header or URL parameter
    written out literally, instead of as ${NAME}, is replaced with <redacted>."""
    saved = copy.deepcopy(config.raw)
    request = saved.get("request")
    if not isinstance(request, dict):
        return saved, []
    issues = []
    headers = request.get("headers")
    if isinstance(headers, dict):
        for name, value in headers.items():
            if SENSITIVE_NAME.search(str(name)) and "${" not in str(value):
                headers[name] = REDACTED
                issues.append(Issue(config.path.name, f"header {name} has a literal secret; "
                                                      "write it as ${NAME} and set it in your environment"))
    url = request.get("url")
    if isinstance(url, str):
        parts = urlsplit(url)
        query = parse_qsl(parts.query, keep_blank_values=True)
        if any(SENSITIVE_NAME.search(name) and "${" not in value for name, value in query):
            safe = [(name, REDACTED if SENSITIVE_NAME.search(name) and "${" not in value else value)
                    for name, value in query]
            request["url"] = urlunsplit(parts._replace(query=urlencode(safe, safe="${}")))
            issues.append(Issue(config.path.name, "the request URL has a literal secret; "
                                                  "write it as ${NAME} and set it in your environment"))
    return saved, issues


def check_env(config: AppConfig, env: Mapping[str, str]) -> list[Issue]:
    """Environment variables the config refers to but that aren't set."""
    _, missing = resolve_env(config.raw, env)
    return [Issue(config.path.name, f"environment variable {name} is not set") for name in missing]


def check_allowed(manifest: Manifest, config: AppConfig, env: Mapping[str, str]) -> list[Issue]:
    """Enforce the dataset's allowed_hosts and allowed_providers (docs/design.md, section 8)."""
    where = config.path.name
    issues = []
    if config.request is not None and manifest.allowed_hosts is not None:
        url, _ = resolve_env(config.request.url, env)
        host = urlsplit(url).hostname or ""
        if "${" in host or "{{" in host:
            issues.append(Issue(where, "the request URL's host must be written out, so allowed_hosts can be checked"))
        elif host not in manifest.allowed_hosts:
            issues.append(Issue(where, f"host {host} isn't in the dataset's allowed_hosts"))
    if config.provider is not None and manifest.allowed_providers is not None:
        if config.provider not in manifest.allowed_providers:
            issues.append(Issue(where, f"provider {config.provider} isn't in the dataset's allowed_providers"))
    return issues
