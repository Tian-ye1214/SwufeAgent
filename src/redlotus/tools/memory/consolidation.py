from __future__ import annotations

import difflib
from typing import Any

from redlotus.config.app_config import settings


def long_term_memory_consolidation_params(**kwargs: Any) -> dict[str, int | float]:
    cfg = settings()["long_term_memory"]["consolidation"]
    out = {
        "max_transcript_chars": int(cfg["max_transcript_chars"]),
        "max_output_tokens": int(cfg["max_output_tokens"]),
        "log_read_chunk_bytes": int(cfg["log_read_chunk_bytes"]),
        "merge_min_similarity": float(cfg["merge_min_similarity"]),
        "merge_max_len_ratio": float(cfg["merge_max_len_ratio"]),
        "merge_min_len_ratio": float(cfg["merge_min_len_ratio"]),
        "merge_len_check_min_old_chars": int(cfg["merge_len_check_min_old_chars"]),
    }
    out.update(kwargs)
    return out


def long_term_memory_debounce_params(**kwargs: Any) -> dict[str, float]:
    cfg = settings()["long_term_memory"]["consolidation"].get("debounce", {})
    out = {
        "min_new_turns": int(cfg.get("min_new_turns", 3)),
        "min_interval_sec": float(cfg.get("min_interval_sec", 120)),
    }
    out.update(kwargs)
    return out


def merge_looks_like_unrelated_rewrite(old_body: str, new_body: str, **kwargs: Any) -> bool:
    cfg = long_term_memory_consolidation_params(**kwargs)
    o = (old_body or "").strip()
    n = (new_body or "").strip()
    if not o:
        return False
    if not n:
        return True
    ol = len(o)
    nl = len(n)
    min_old = int(cfg["merge_len_check_min_old_chars"])
    if ol >= min_old:
        lo = max(1, int(ol * float(cfg["merge_min_len_ratio"])))
        hi = int(ol * float(cfg["merge_max_len_ratio"]))
        if nl < lo or nl > hi:
            return True
    ratio = difflib.SequenceMatcher(a=o, b=n).ratio()
    if ratio < float(cfg["merge_min_similarity"]):
        return True
    return False
