from __future__ import annotations

import sys
from pathlib import Path


def runtime_repo_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def work_database_root() -> Path:
    return runtime_repo_root() / "WorkDatabase"


def readable_roots(*, work_base: Path) -> tuple[Path, ...]:
    repo = runtime_repo_root()
    return (
        work_base.resolve(),
        (repo / "src" / "skills").resolve(),
    )


def is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_readable_path(path: Path, *, work_base: Path) -> Path:
    """解析后的路径必须在 WorkDatabase 或 src/skills 下。"""
    resolved = path.resolve()
    for root in readable_roots(work_base=work_base):
        if is_under_root(resolved, root):
            return resolved
    roots = ", ".join(str(r) for r in readable_roots(work_base=work_base))
    raise ValueError(f"Path not allowed (must be under: {roots}): {resolved}")


def resolve_readable_path(
    name: str,
    *,
    work_base: Path,
    repo_root: Path | None = None,
) -> Path:
    """相对路径：WorkDatabase 或 src/ 锚定；绝对路径须落在可读根内。"""
    name = (name or "").strip()
    if not name:
        raise ValueError("Path name must not be empty")
    if name.replace("\\", "/").strip("/").rstrip("/") == "skills":
        name = "src/skills"

    repo = (repo_root if repo_root is not None else runtime_repo_root()).resolve()
    work = work_base.resolve()
    p_in = Path(name).expanduser()
    if p_in.is_absolute():
        return assert_readable_path(p_in, work_base=work)

    norm = name.replace("\\", "/")
    anchor = repo if norm == "src" or norm.startswith("src/") else work
    return assert_readable_path((anchor / name).resolve(), work_base=work)
