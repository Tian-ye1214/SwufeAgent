from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from path_sandbox import resolve_readable_path, runtime_repo_root, work_database_root


def test_resolve_relative_under_work_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = tmp_path / "WorkDatabase"
    work.mkdir()
    repo = tmp_path / "repo"
    skills = repo / "src" / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setattr("path_sandbox.runtime_repo_root", lambda: repo)
    p = resolve_readable_path("notes.txt", work_base=work, repo_root=repo)
    assert p == (work / "notes.txt").resolve()


def test_work_database_root_uses_runtime_repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    monkeypatch.setattr("path_sandbox.runtime_repo_root", lambda: repo)

    assert work_database_root() == repo / "WorkDatabase"


def test_reject_outside_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = tmp_path / "WorkDatabase"
    work.mkdir()
    repo = tmp_path / "repo"
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    monkeypatch.setattr("path_sandbox.runtime_repo_root", lambda: repo)
    with pytest.raises(ValueError, match="not allowed"):
        resolve_readable_path(str(outside), work_base=work, repo_root=repo)
