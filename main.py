import os
import sys

if getattr(sys, "frozen", False):
    os.environ["LOGFIRE_PYDANTIC_RECORD"] = "off"

from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    src_root = str(repo_root / "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

_ensure_src_on_path()

from agent_app import main


if __name__ == "__main__":
    main()
