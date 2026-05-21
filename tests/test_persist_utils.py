from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from persist_utils import atomic_write_json


def test_atomic_write_json_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    data = {"version": 1, "sources": {"a": 1}}
    atomic_write_json(target, data)
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == data
    assert not (tmp_path / "state.json.tmp").exists()
