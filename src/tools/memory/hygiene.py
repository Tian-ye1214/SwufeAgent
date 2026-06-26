from __future__ import annotations

import hashlib
import re
import unicodedata

_NL_RUN = re.compile(r"\n{3,}")
_WS_ANY = re.compile(r"\s+")


def normalize_for_storage(text: str) -> str:
    """落盘文本的规范化：NFC + 逐行 rstrip + 折叠 3+ 空行。
    不折叠行内空格，避免破坏代码缩进。"""
    t = unicodedata.normalize("NFC", text or "")
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    t = _NL_RUN.sub("\n\n", t)
    return t.strip()


def normalize_for_key(text: str) -> str:
    """仅用于去重哈希的临时副本：在 storage 规范化基础上额外 NFKC
    （折叠全角↔半角数字，治中文 IME 漂移）并把所有空白折叠为单空格。
    不 lowercase、不剥标点（避免把不同命令/路径误并）。"""
    t = unicodedata.normalize("NFKC", normalize_for_storage(text))
    return _WS_ANY.sub(" ", t).strip()


def turn_key_hash(text: str) -> str:
    """turn 内容指纹：跨压缩/空白/IME 漂移稳定。"""
    return hashlib.sha1(normalize_for_key(text).encode("utf-8")).hexdigest()[:16]
