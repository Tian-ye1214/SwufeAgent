import threading
from pathlib import Path

from infra.persist_utils import update_locked_json


def test_update_creates_and_returns(tmp_path: Path):
    p = tmp_path / "s.json"

    def mut(s):
        s.setdefault("sources", {})["k"] = 1

    out = update_locked_json(p, 2, mut)
    assert out["version"] == 2 and out["sources"]["k"] == 1
    assert p.is_file()


def test_concurrent_rmw_no_lost_update(tmp_path: Path):
    p = tmp_path / "s.json"
    update_locked_json(p, 2, lambda s: s.setdefault("sources", {}))

    def writer(key):
        def mut(s):
            s.setdefault("sources", {})[key] = key
        for _ in range(50):
            update_locked_json(p, 2, mut)

    threads = [threading.Thread(target=writer, args=(f"k{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    out = update_locked_json(p, 2, lambda s: None)
    assert set(out["sources"]) == {f"k{i}" for i in range(8)}
