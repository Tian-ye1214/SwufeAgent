"""@文件 引用解析与读取。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from path_sandbox import is_under_root, runtime_repo_root

_FILE_REF_PATTERN = re.compile(r"@([^\s]+)")
_BINARY_SUFFIXES = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico",
        ".mp4", ".avi", ".mov", ".mkv", ".webm",
        ".pdf", ".exe", ".zip", ".tar", ".gz", ".7z", ".rar",
        ".dll", ".so", ".dylib", ".bin", ".pyc", ".woff", ".woff2",
    }
)
_TEXT_SUFFIXES = frozenset(
    {
        ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
        ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
        ".cpp", ".c", ".h", ".hpp", ".cs", ".rb", ".php",
        ".html", ".css", ".scss", ".sql", ".sh", ".bat", ".ps1",
        ".xml", ".csv", ".ini", ".cfg", ".env", ".log",
    }
)
_DEFAULT_MAX_CHARS = 20_000
_DEFAULT_TOTAL_MAX_CHARS = 50_000


@dataclass(frozen=True)
class FileRefResult:
    path: str
    ok: bool
    error: str | None = None
    content: str | None = None
    truncated: bool = False
    resolved: Path | None = None


def extract_file_refs(text: str) -> list[str]:
    return _FILE_REF_PATTERN.findall(text or "")


def _allowed_roots() -> tuple[Path, Path]:
    return Path.cwd().resolve(), runtime_repo_root().resolve()


def _assert_allowed_path(path: Path) -> Path:
    resolved = path.resolve()
    for root in _allowed_roots():
        if is_under_root(resolved, root):
            return resolved
    raise ValueError("路径不在当前目录或项目根目录下")


def _resolve_ref_path(ref: str) -> Path:
    p = Path(ref).expanduser()
    if p.is_absolute():
        return _assert_allowed_path(p)
    cwd = Path.cwd()
    candidate = (cwd / ref).resolve()
    if candidate.exists():
        return _assert_allowed_path(candidate)
    repo = runtime_repo_root()
    return _assert_allowed_path((repo / ref).resolve())


def _read_text_safely(path: Path) -> str:
    for encoding in ("utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文件编码")


def load_file_refs(
    text: str,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
    total_max_chars: int = _DEFAULT_TOTAL_MAX_CHARS,
) -> list[FileRefResult]:
    refs = extract_file_refs(text)
    if not refs:
        return []

    results: list[FileRefResult] = []
    total_used = 0
    seen: set[str] = set()

    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)

        try:
            path = _resolve_ref_path(ref)
        except ValueError as e:
            results.append(FileRefResult(path=ref, ok=False, error=str(e)))
            continue
        if not path.exists():
            results.append(FileRefResult(path=ref, ok=False, error="文件不存在"))
            continue
        if path.is_dir():
            results.append(FileRefResult(path=ref, ok=False, error="暂不支持直接引用目录"))
            continue

        suffix = path.suffix.lower()
        if suffix in _BINARY_SUFFIXES:
            results.append(FileRefResult(path=ref, ok=False, error="不支持引用二进制文件"))
            continue
        if suffix and suffix not in _TEXT_SUFFIXES:
            # 无后缀或未知后缀：尝试按文本读取，失败则报错
            pass

        try:
            content = _read_text_safely(path)
        except PermissionError:
            results.append(FileRefResult(path=ref, ok=False, error="无权限读取文件"))
            continue
        except OSError as e:
            results.append(FileRefResult(path=ref, ok=False, error=str(e)))
            continue
        except ValueError as e:
            results.append(FileRefResult(path=ref, ok=False, error=str(e)))
            continue

        truncated = False
        budget = min(max_chars, total_max_chars - total_used)
        if budget <= 0:
            results.append(FileRefResult(path=ref, ok=False, error="文件引用总字符数已达上限"))
            continue
        if len(content) > budget:
            content = content[:budget]
            truncated = True
        total_used += len(content)

        results.append(
            FileRefResult(
                path=str(path),
                ok=True,
                content=content,
                truncated=truncated,
                resolved=path,
            )
        )
    return results


def augment_text_with_file_refs(text: str, refs: list[FileRefResult]) -> str:
    """将成功读取的文件内容附加到用户输入后，供 Agent 使用。"""
    ok_refs = [r for r in refs if r.ok and r.content is not None]
    if not ok_refs:
        return text

    blocks = [text]
    for item in ok_refs:
        suffix = ""
        if item.truncated:
            suffix = "\n（内容已截断）"
        blocks.append(
            f"引用文件：{item.path}{suffix}\n\n```\n{item.content}\n```"
        )
    return "\n\n".join(blocks)
