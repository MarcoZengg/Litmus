"""Load and check datasets: dataset.yaml plus dataset.jsonl (docs/design.md, section 5).

Issues name a record ID, line or setting, never question text, so they are safe
to show for private datasets.
"""

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

BEHAVIORS = ("answer", "partial", "clarify", "refuse", "handoff")
SOURCES = ("staff", "real_traffic", "faq", "synthetic", "mixed")
MANIFEST_FILE = "dataset.yaml"
RECORDS_FILE = "dataset.jsonl"
MANIFEST_FIELDS = {"name", "version", "kb_version", "language", "private", "allowed_providers", "allowed_hosts"}
RECORD_FIELDS = {"id", "input", "expected_output", "reference_context_ids", "reference_contexts",
                 "expected_behavior", "tags", "language", "source", "metadata", "notes"}
MULTI_TURN_FIELDS = {"turns", "messages", "conversation"}


@dataclass(frozen=True)
class Issue:
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message}"


@dataclass
class Manifest:
    name: str
    version: str | int
    kb_version: str | None = None
    language: str | None = None
    private: bool = False
    allowed_providers: list[str] | None = None
    allowed_hosts: list[str] | None = None


@dataclass
class Record:
    id: str
    input: str
    expected_output: str | None = None
    # Groups of alternative passage IDs; any one ID in a group counts.
    # None means not labeled, [] means no passage should answer.
    reference_context_ids: list[list[str]] | None = None
    reference_contexts: list[str] | None = None
    expected_behavior: str | None = None
    tags: list[str] = field(default_factory=list)
    language: str | None = None
    source: str | None = None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    notes: str | None = None


@dataclass
class Dataset:
    folder: Path
    manifest: Manifest | None
    records: list[Record]
    sha256: str | None
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_dataset(path: Path) -> Dataset:
    """Load a dataset folder (or a file inside it), collecting every problem found."""
    folder = path.parent if path.is_file() else path
    errors: list[Issue] = []
    warnings: list[Issue] = []
    manifest = _load_manifest(folder / MANIFEST_FILE, errors)
    records, sha256 = _load_records(folder / RECORDS_FILE, manifest, errors, warnings)
    return Dataset(folder, manifest, records, sha256, errors, warnings)


def _load_manifest(path: Path, errors: list[Issue]) -> Manifest | None:
    where = MANIFEST_FILE
    if not path.is_file():
        errors.append(Issue(where, "file not found"))
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (yaml.YAMLError, UnicodeDecodeError):
        errors.append(Issue(where, "not valid YAML"))
        return None
    if not isinstance(data, dict):
        errors.append(Issue(where, "must be a list of settings, e.g. 'name: my-dataset'"))
        return None

    problems = [f"unknown setting '{key}'" for key in sorted(set(map(str, data)) - MANIFEST_FIELDS)]
    if not _is_text(data.get("name")):
        problems.append("'name' is required")
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, (str, int)) or str(version).strip() == "":
        problems.append("'version' is required (a number or text)")
    for key in ("kb_version", "language"):
        if key in data and isinstance(data[key], (datetime.date, datetime.datetime)):
            problems.append(f"'{key}' must be text; put dates in quotes")
        elif key in data and not isinstance(data[key], str):
            problems.append(f"'{key}' must be text")
    if "private" in data and not isinstance(data["private"], bool):
        problems.append("'private' must be true or false")
    for key in ("allowed_providers", "allowed_hosts"):
        if key in data and not _is_text_list(data[key]):
            problems.append(f"'{key}' must be a list of names")
    if problems:
        errors.extend(Issue(where, problem) for problem in problems)
        return None

    return Manifest(
        name=data["name"],
        version=version,
        kb_version=data.get("kb_version"),
        language=data.get("language"),
        private=data.get("private", False),
        allowed_providers=data.get("allowed_providers"),
        allowed_hosts=data.get("allowed_hosts"),
    )


def _load_records(path: Path, manifest: Manifest | None, errors: list[Issue],
                  warnings: list[Issue]) -> tuple[list[Record], str | None]:
    if not path.is_file():
        errors.append(Issue(RECORDS_FILE, "file not found"))
        return [], None
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        errors.append(Issue(RECORDS_FILE, "not UTF-8 text"))
        return [], sha256

    records: list[Record] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(Issue(f"line {number}", f"not valid JSON ({error.msg})"))
            continue
        if not isinstance(data, dict):
            errors.append(Issue(f"line {number}", "must be a JSON object"))
            continue
        record = _check_record(data, number, manifest, seen, errors, warnings)
        if record is not None:
            records.append(record)
    if not records and not errors:
        errors.append(Issue(RECORDS_FILE, "no records"))
    return records, sha256


def _check_record(data: dict, number: int, manifest: Manifest | None, seen: dict[str, int],
                  errors: list[Issue], warnings: list[Issue]) -> Record | None:
    record_id = data.get("id")
    where = record_id if _is_text(record_id) else f"line {number}"
    problems = []
    if not _is_text(record_id):
        problems.append("'id' is required")
    elif record_id in seen:
        problems.append(f"duplicate id (also on line {seen[record_id]})")
    else:
        seen[record_id] = number

    for key in sorted(set(map(str, data)) - RECORD_FIELDS):
        hint = " (multi-turn records come after v1)" if key in MULTI_TURN_FIELDS else ""
        problems.append(f"unknown field '{key}'{hint}")
    if not _is_text(data.get("input")):
        problems.append("'input' is required and must be non-empty text")
    for key in ("expected_output", "language", "notes"):
        if key in data and not isinstance(data[key], str):
            problems.append(f"'{key}' must be text")
    for key, allowed in (("expected_behavior", BEHAVIORS), ("source", SOURCES)):
        if key in data and data[key] not in allowed:
            problems.append(f"'{key}' must be one of: {', '.join(allowed)}")
    for key in ("tags", "reference_contexts"):
        if key in data and not _is_text_list(data[key]):
            problems.append(f"'{key}' must be a list of text")
    groups = None
    if "reference_context_ids" in data:
        groups = _groups(data["reference_context_ids"])
        if groups is None:
            problems.append("'reference_context_ids' must be a list of passage IDs, "
                            "or of lists of alternative IDs")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict) or not all(
            isinstance(value, (str, int, float, bool)) for value in metadata.values()):
        problems.append("'metadata' must map names to text, numbers or true/false")

    if problems:
        errors.extend(Issue(where, problem) for problem in problems)
        return None
    if data.get("expected_behavior") == "answer" and not data.get("expected_output") and not groups:
        warnings.append(Issue(where, "expects an answer but has no expected_output or "
                                     "reference_context_ids, so only a rubric judge can score it"))
    return Record(
        id=record_id,
        input=data["input"],
        expected_output=data.get("expected_output"),
        reference_context_ids=groups,
        reference_contexts=data.get("reference_contexts"),
        expected_behavior=data.get("expected_behavior"),
        tags=data.get("tags", []),
        language=data.get("language") or (manifest.language if manifest else None),
        source=data.get("source"),
        metadata=metadata,
        notes=data.get("notes"),
    )


def _groups(value: object) -> list[list[str]] | None:
    """Normalize passage IDs to groups: "p1" becomes ["p1"]. None if malformed."""
    if not isinstance(value, list):
        return None
    groups = []
    for item in value:
        if _is_text(item):
            groups.append([item])
        elif isinstance(item, list) and item and all(_is_text(i) for i in item):
            groups.append(list(item))
        else:
            return None
    return groups


def _is_text(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _is_text_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)
