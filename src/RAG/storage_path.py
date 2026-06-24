from __future__ import annotations

import os
import sys
from pathlib import Path

from infra import logger
from config.app_config import get_env
from infra.path_sandbox import runtime_repo_root

_UNSUPPORTED_FS = frozenset({"EXFAT", "FAT32", "FAT"})


def volume_filesystem(path: Path) -> str | None:
    """返回路径所在卷的 filesystem 名称（如 NTFS、exFAT）；无法检测时返回 None。"""
    resolved = path.resolve()
    if sys.platform == "win32":
        root = os.path.splitdrive(str(resolved))[0] + "\\"
        if len(root) < 3:
            return None
        import ctypes

        buf = ctypes.create_unicode_buffer(256)
        ok = ctypes.windll.kernel32.GetVolumeInformationW(
            ctypes.c_wchar_p(root),
            None,
            0,
            None,
            None,
            None,
            buf,
            len(buf),
        )
        return buf.value if ok else None
    return None


def _is_unsupported_fs(path: Path) -> bool:
    fs = volume_filesystem(path)
    return bool(fs and fs.upper() in _UNSUPPORTED_FS)


def resolve_lancedb_dir(configured_path: str) -> str:
    """将配置的 LanceDB 目录解析为绝对路径；exFAT/FAT 上自动改到 NTFS（RAG_DB_PATH 或 LOCALAPPDATA）。"""
    p = Path(configured_path)
    if not p.is_absolute():
        p = (runtime_repo_root() / p).resolve()
    else:
        p = p.resolve()

    if not _is_unsupported_fs(p):
        return str(p)

    override = get_env("RAG_DB_PATH", warn=False)
    if override:
        op = Path(override)
        if not op.is_absolute():
            op = (runtime_repo_root() / op).resolve()
        else:
            op = op.resolve()
        if not _is_unsupported_fs(op):
            logger.warning(
                "LanceDB: 配置路径 %s 位于 %s，不支持 Lance 事务；改用 RAG_DB_PATH=%s",
                p,
                volume_filesystem(p) or "?",
                op,
            )
            return str(op)

    local_root = os.environ.get("LOCALAPPDATA") or str(Path.home())
    fallback = Path(local_root) / "Agent" / "rag_lancedb" / p.name
    logger.warning(
        "LanceDB: 配置路径 %s 位于 %s（不支持硬链接）；已改用 %s。"
        "可在 config 的 short_term_memory.db_path 或环境变量 RAG_DB_PATH 中指定 NTFS 目录。",
        p,
        volume_filesystem(p) or "?",
        fallback,
    )
    return str(fallback.resolve())
