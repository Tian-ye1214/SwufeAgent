from tools.memory import hygiene

DAY = 86400.0


def test_is_forgotten_basic_and_hit_extends_ttl():
    # 31 天 > 30 天 -> 遗忘；但 hit 多则 TTL 延长，保住
    assert hygiene.stm_is_forgotten(age_seconds=31 * DAY, hit_count=0, max_age_seconds=30 * DAY)
    assert not hygiene.stm_is_forgotten(age_seconds=31 * DAY, hit_count=10, max_age_seconds=30 * DAY)


def test_is_forgotten_disabled_when_no_max_age():
    assert not hygiene.stm_is_forgotten(age_seconds=999 * DAY, hit_count=0, max_age_seconds=None)
    assert not hygiene.stm_is_forgotten(age_seconds=999 * DAY, hit_count=0, max_age_seconds=0)


def test_apply_preserves_order_and_drops_only_aged():
    hits = [
        {"source": "k#a", "created_at": "2026-06-01T00:00:00Z"},
        {"source": "k#b", "created_at": "2026-06-01T00:00:00Z"},
    ]
    meta = {
        "k#a": {"hit": 0, "last_accessed": "2026-01-01T00:00:00Z"},  # 很久前 -> 遗忘
        "k#b": {"hit": 0, "last_accessed": "2026-06-25T00:00:00Z"},  # 最近 -> 保留
    }
    out = hygiene.apply_soft_forget(
        hits, meta=meta, soft_forget_since="2026-05-01T00:00:00Z",
        max_age_seconds=30 * DAY, now_iso="2026-06-26T00:00:00Z",
    )
    assert [h["source"] for h in out] == ["k#b"]


def test_apply_exempts_historical_rows():
    hits = [{"source": "k#old", "created_at": "2026-01-01T00:00:00Z"}]
    meta = {"k#old": {"hit": 0, "last_accessed": "2026-01-01T00:00:00Z"}}
    out = hygiene.apply_soft_forget(
        hits, meta=meta, soft_forget_since="2026-05-01T00:00:00Z",
        max_age_seconds=30 * DAY, now_iso="2026-06-26T00:00:00Z",
    )
    assert len(out) == 1  # created_at < soft_forget_since -> 豁免


def test_apply_grace_for_never_retrieved():
    hits = [{"source": "k#new", "created_at": "2026-06-20T00:00:00Z"}]
    out = hygiene.apply_soft_forget(
        hits, meta={}, soft_forget_since="2026-05-01T00:00:00Z",
        max_age_seconds=30 * DAY, now_iso="2026-06-26T00:00:00Z",
    )
    assert len(out) == 1  # 无 meta 条目 -> 冷启动 grace
