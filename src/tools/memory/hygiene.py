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


import math
from datetime import datetime


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def stm_is_forgotten(*, age_seconds: float, hit_count: int, max_age_seconds: float | None) -> bool:
    """命中越多 TTL 越长；max_age 为 None/<=0 时永不遗忘。"""
    if not max_age_seconds or max_age_seconds <= 0:
        return False
    effective_ttl = max_age_seconds * (1.0 + math.log1p(max(0, int(hit_count))))
    return age_seconds > effective_ttl


def apply_soft_forget(
    hits: list[dict],
    *,
    meta: dict,
    soft_forget_since: str | None,
    max_age_seconds: float | None,
    now_iso: str,
    grace_until_first_use: bool = True,
) -> list[dict]:
    """只过滤不重排：丢弃"曾被检索过、之后超过(命中延长的)TTL 未再检索"的行。
    历史行(created_at < soft_forget_since)与从未检索行(无 meta)永远保留。"""
    now = _parse_iso(now_iso)
    since = _parse_iso(soft_forget_since)
    kept: list[dict] = []
    for h in hits:
        src = h.get("source", "")
        created = _parse_iso(h.get("created_at"))
        if since is not None and created is not None and created < since:
            kept.append(h)
            continue
        m = meta.get(src)
        if grace_until_first_use and not m:
            kept.append(h)
            continue
        last = _parse_iso((m or {}).get("last_accessed")) or created
        if now is None or last is None:
            kept.append(h)
            continue
        age = (now - last).total_seconds()
        if stm_is_forgotten(
            age_seconds=age,
            hit_count=int((m or {}).get("hit", 0)),
            max_age_seconds=max_age_seconds,
        ):
            continue
        kept.append(h)
    return kept
