"""Request templates (docs/design.md, section 4).

A template may use {{input}} and {{metadata.<key>}}. A trailing ? makes a
metadata key optional: {{metadata.category?}} leaves the field out of the
request when a record has no category. The openai adapter's prompts may also
use {{passages}}: the passages saved for the question (passages_from).
"""

import re
from dataclasses import dataclass

from litmus.appconfig import AppConfig
from litmus.dataset import Issue, Record

PLACEHOLDER = re.compile(r"\{\{(.*?)\}\}")
METADATA_KEY = re.compile(r"metadata\.([A-Za-z_][A-Za-z0-9_-]*)(\?)?")
SHOWN_IDS = 10
_MISSING = object()  # an optional metadata key the record doesn't have


@dataclass(frozen=True)
class Placeholder:
    field: str  # "input", "metadata" or "passages"
    key: str | None = None
    optional: bool = False


def parse_placeholder(expression: str) -> Placeholder | None:
    expression = expression.strip()
    if expression in ("input", "passages"):
        return Placeholder(expression)
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
                                           "use {{input}}, {{metadata.<key>}} or {{passages}}"))
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


def app_templates(config: AppConfig) -> dict[str, object]:
    """The parts of an app config that are templates, by name: the http adapter's request, and the
    openai adapter's prompts."""
    if config.openai is not None:
        return {"system_prompt": config.openai.system_prompt, "user_template": config.openai.user_template}
    return {"request": config.raw["request"]} if "request" in config.raw else {}


def check_request_templates(config: AppConfig, records: list[Record]) -> list[Issue]:
    """Check an app config's templates, and that every record has the metadata they need."""
    found, issues = [], []
    for name, template in app_templates(config).items():
        more_found, more_issues = find_placeholders(template, name)
        found += more_found
        issues += more_issues
    issues = [Issue(f"{config.path.name} {issue.where}", issue.message) for issue in issues]
    uses_passages = any(p.field == "passages" for p in found)
    has_passages = config.openai is not None and config.openai.passages_from is not None
    if uses_passages and not has_passages:
        issues.append(Issue(config.path.name, "{{passages}} needs passages_from, which only the openai adapter has"))
    if has_passages and not uses_passages:
        issues.append(Issue(config.path.name, "passages_from is set, but no prompt uses {{passages}}, "
                                              "so the model would never see them"))
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


def render(template: object, record: Record, passages: str | None = None) -> object:
    """Fill a template for one record. A string that is a single placeholder keeps the value's type
    (numbers stay numbers); an optional key the record lacks removes that field or list item.
    passages is what {{passages}} becomes, already formatted."""
    rendered = _render(template, record, passages)
    return None if rendered is _MISSING else rendered


def format_passages(texts: list[str], ids: list[str] | None = None) -> str:
    """Numbered passages for a prompt, one per line: "[1] (id) text", or "[1] text" without matching IDs."""
    ids = ids or []
    return "\n".join(f"[{n}] ({ids[n - 1]}) {text}" if len(ids) == len(texts) else f"[{n}] {text}"
                     for n, text in enumerate(texts, 1))


def _render(template: object, record: Record, passages: str | None) -> object:
    if isinstance(template, str):
        whole = PLACEHOLDER.fullmatch(template)
        if whole:
            return _value(whole.group(1), record, passages)

        def fill(match: re.Match) -> str:
            value = _value(match.group(1), record, passages)
            return "" if value is _MISSING else str(value)
        return PLACEHOLDER.sub(fill, template)
    if isinstance(template, dict):
        filled = {key: _render(value, record, passages) for key, value in template.items()}
        return {key: value for key, value in filled.items() if value is not _MISSING}
    if isinstance(template, list):
        return [value for value in (_render(item, record, passages) for item in template) if value is not _MISSING]
    return template


def _value(expression: str, record: Record, passages: str | None) -> object:
    placeholder = parse_placeholder(expression)
    if placeholder is None:
        raise ValueError(f"unknown placeholder {{{{{expression}}}}}")
    if placeholder.field == "input":
        return record.input
    if placeholder.field == "passages":
        if passages is None:
            raise ValueError("{{passages}} needs passages_from")
        return passages
    if placeholder.key in record.metadata:
        return record.metadata[placeholder.key]
    if placeholder.optional:
        return _MISSING
    raise ValueError(f"{record.id} has no metadata.{placeholder.key}, which the request template needs")
