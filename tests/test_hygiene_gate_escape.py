from tools.memory import hygiene

P = {"min_new_turns": 3, "min_interval_sec": 120}


def test_gate_first_run():
    ok, why = hygiene.should_consolidate(
        fingerprint="x", last_fingerprint=None, new_turn_count=0,
        seconds_since_last_run=None, params=P)
    assert ok and why == "first_run"


def test_gate_unchanged():
    ok, why = hygiene.should_consolidate(
        fingerprint="x", last_fingerprint="x", new_turn_count=99,
        seconds_since_last_run=9999, params=P)
    assert not ok and why == "unchanged"


def test_gate_too_few_turns():
    ok, why = hygiene.should_consolidate(
        fingerprint="y", last_fingerprint="x", new_turn_count=1,
        seconds_since_last_run=9999, params=P)
    assert not ok and why == "too_few_new_turns"


def test_gate_too_soon():
    ok, why = hygiene.should_consolidate(
        fingerprint="y", last_fingerprint="x", new_turn_count=5,
        seconds_since_last_run=10, params=P)
    assert not ok and why == "too_soon"


def test_gate_ok():
    ok, why = hygiene.should_consolidate(
        fingerprint="y", last_fingerprint="x", new_turn_count=5,
        seconds_since_last_run=9999, params=P)
    assert ok and why == "ok"


def test_like_escape_protects_underscore():
    pat, esc = hygiene.like_prefix_escaped("messages_231")
    assert esc == "\\"
    # 下划线被转义，故只能匹配字面 messages_231...，不会把 _ 当通配符
    assert pat == "messages\\_231%"


def test_like_escape_percent_and_escapechar():
    pat, _ = hygiene.like_prefix_escaped("a%b\\c")
    assert pat == "a\\%b\\\\c%"
