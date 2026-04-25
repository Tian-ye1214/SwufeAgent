from __future__ import annotations

import os
import sys
from pathlib import Path


def _set_tiktoken_cache_dir() -> None:
    # 在打包运行时，将 tiktoken 缓存目录固定到 exe 同级的打包资源目录。
    if not getattr(sys, "frozen", False):
        return
    exe_dir = Path(sys.executable).resolve().parent
    cache_dir = exe_dir / "tiktoken_cache"
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache_dir))


_set_tiktoken_cache_dir()
