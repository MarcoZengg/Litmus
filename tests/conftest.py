import shutil
from pathlib import Path

import pytest

X200 = Path(__file__).parent / "data" / "x200"


@pytest.fixture
def x200(tmp_path: Path) -> Path:
    """A writable copy of the X200 example dataset and app config."""
    folder = tmp_path / "x200"
    shutil.copytree(X200, folder)
    return folder


def write_records(folder: Path, *lines: str) -> None:
    (folder / "dataset.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
