"""Request templates (docs/design.md, section 4).

A template may use {{input}} and {{metadata.<key>}}. A trailing ? makes a
metadata key optional: {{metadata.category?}} leaves the field out of the
request when a record has no category.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from litmus.dataset import Issue, Record

PLACEHOLDER = re.compile(r"\{\{(.*?)\}\}")
METADATA_KEY = re.compile(r"metadata\.([A-Za-z_][A-Za-z0-9_-]*)(\?)?")
SHOWN_IDS = 10


@dataclass(frozen=True)
class Placeholder:
    field: str  # "input" or "metadata"
    key: str | None = None
    optional: bool = False


def parse_placeholder(expression: str) -> Placeholder | None:
    expression = expression.strip()
    if expression == "input":
        return Placeholder("input")
    match = METADATA_KEY.fullmatch(expression)
    if match:
        return Placeholder("metadata", match.group(1), optional=match.group(2) == "?")
    return None


def find_placeholders(template: object, where: str = "") -> tuple[list[Placeholder], list[Issue]]:
    """Every placeholder in a nested template (strings, lists, mappings), plus any malformed ones."""
    found: list[Placeholder] = []
    issues: list[Issue] = []
    if isinstance(template, str):
        for match in PLACEHOLDER.finditer(template):
            placeholder = parse_placeholder(match.group(1))
            if placeholder is None:
                issues.append(Issue(where, f"unknown placeholder {match.group(0)}; "
                                           "use {{input}} or {{metadata.<key>}}"))
            else:
                found.append(placeholder)
        rest = PLACEHOLDER.sub("", template)
        if "{{" in rest or "}}" in rest:
            issues.append(Issue(where, "unbalanced {{ or }}"))
    elif isinstance(template, dict):
        for key, value in template.items():
            sub_found, sub_issues = find_placeholders(value, f"{where}.{key}" if where else str(key))
            found += sub_found
            issues += sub_issues
    elif isinstance(template, list):
        for index, value in enumerate(template):
            sub_found, sub_issues = find_placeholders(value, f"{where}[{index}]")
            found += sub_found
            issues += sub_issues
    return found, issues


def check_app_templates(config_path: Path, records: list[Record]) -> list[Issue]:
    """Check an app config's request templates, and that every record has the metadata they need."""
    name = config_path.name
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    except (yaml.YAMLError, UnicodeDecodeError):
        return [Issue(name, "not valid YAML")]
    if not isinstance(config, dict):
        return [Issue(name, "must be a list of settings, e.g. 'adapter: http'")]
    if "request" not in config:
        return []  # only the HTTP adapter has request templates

    found, issues = find_placeholders(config["request"], "request")
    issues = [Issue(f"{name} {issue.where}", issue.message) for issue in issues]
    required = sorted({p.key for p in found if p.field == "metadata" and not p.optional})
    for key in required:
        missing = [record.id for record in records if key not in record.metadata]
        if missing:
            shown = ", ".join(missing[:SHOWN_IDS])
            if len(missing) > SHOWN_IDS:
                shown += f" and {len(missing) - SHOWN_IDS} more"
            issues.append(Issue(
                f"metadata.{key}",
                f"the request template needs it, but {len(missing)} record(s) don't have it: {shown}. "
                f"Write {{{{metadata.{key}?}}}} if it's optional."))
    return issues
