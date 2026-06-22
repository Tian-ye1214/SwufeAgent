from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    """Project root: exe directory when frozen, else parent of src/."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def src_root() -> Path:
    return repo_root() / "src"


def work_database_root() -> Path:
    return repo_root() / "WorkDatabase"


def config_file() -> Path:
    """config.json: exe dir when frozen, else src/config.json."""
    if getattr(sys, "frozen", False):
        return repo_root() / "config.json"
    return src_root() / "config.json"


def dotenv_file() -> Path:
    """.env: exe dir when frozen, else project root."""
    return repo_root() / ".env"


def skills_dir() -> Path:
    return src_root() / "skills"
