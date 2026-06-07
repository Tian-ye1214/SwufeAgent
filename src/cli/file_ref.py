"""@文件 引用解析与读取。"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

from path_sandbox import runtime_repo_root

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


@dataclass(frozen=True)
class FileRefResult:
    path: str
    ok: bool
    error: str | None = None
    content: str | None = None
    truncated: bool = False
    resolved: Path | None = None


@dataclass(frozen=True)
class _ParsedFileRef:
    path: str
    locked: bool = False


def _parse_file_refs(text: str) -> list[_ParsedFileRef]:
    text = text or ""
    refs: list[_ParsedFileRef] = []
    index = 0
    length = len(text)

    while index < length:
        at_index = text.find("@", index)
        if at_index == -1:
            break
        start = at_index + 1
        if start >= length:
            break

        marker = text[start]
        if marker.isspace():
            index = start + 1
            continue

        if marker == "{":
            end = text.find("}", start + 1)
            if end == -1:
                index = start + 1
                continue
            ref = text[start + 1 : end].strip()
            if ref:
                refs.append(_ParsedFileRef(ref, locked=True))
            index = end + 1
            continue

        if marker in ("\"", "'"):
            end = text.find(marker, start + 1)
            if end == -1:
                index = start + 1
                continue
            ref = text[start + 1 : end].strip()
            if ref:
                refs.append(_ParsedFileRef(ref, locked=True))
            index = end + 1
            continue

        end = start
        while end < length and not text[end].isspace():
            end += 1
        ref = text[start:end]
        if ref:
            refs.append(_ParsedFileRef(ref))
        index = end

    return refs


def extract_file_refs(text: str) -> list[str]:
    return [ref.path for ref in _parse_file_refs(text)]


def _resolve_ref_path(ref: str) -> Path:
    p = Path(ref).expanduser()
    if p.is_absolute():
        return p.resolve()
    cwd = Path.cwd()
    candidate = (cwd / ref).resolve()
    if candidate.exists():
        return candidate
    repo = runtime_repo_root()
    repo_candidate = (repo / ref).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return candidate


def _looks_like_inline_text_suffix(suffix: str) -> bool:
    if not suffix:
        return False
    first = suffix[0]
    if first in ".-_/\\":
        return False
    codepoint = ord(first)
    if (
        0x4E00 <= codepoint <= 0x9FFF
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    ):
        return True
    return unicodedata.category(first).startswith("P")


def _resolve_existing_ref_prefix(ref: str) -> tuple[str, Path] | None:
    for end in range(len(ref) - 1, 0, -1):
        suffix = ref[end:]
        if not _looks_like_inline_text_suffix(suffix):
            continue
        prefix = ref[:end]
        try:
            path = _resolve_ref_path(prefix)
        except (OSError, RuntimeError, ValueError):
            continue
        if path.exists():
            return prefix, path
    return None


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
    max_chars: int = 20_000,
    total_max_chars: int = 50_000,
) -> list[FileRefResult]:
    refs = _parse_file_refs(text)
    if not refs:
        return []

    results: list[FileRefResult] = []
    total_used = 0
    seen: set[str] = set()

    for parsed_ref in refs:
        ref = parsed_ref.path
        if ref in seen:
            continue
        seen.add(ref)

        try:
            path = _resolve_ref_path(ref)
        except (OSError, RuntimeError, ValueError) as e:
            results.append(FileRefResult(path=ref, ok=False, error=str(e)))
            continue
        if not path.exists() and not parsed_ref.locked:
            recovered = _resolve_existing_ref_prefix(ref)
            if recovered is not None:
                ref, path = recovered
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
