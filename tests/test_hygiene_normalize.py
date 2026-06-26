from tools.memory import hygiene


def test_storage_preserves_code_indentation():
    src = "def f():\n    x = 1\n        y = 2\n"
    out = hygiene.normalize_for_storage(src)
    # 行内空格（缩进）不被折叠
    assert "    x = 1" in out
    assert "        y = 2" in out


def test_storage_rstrips_lines_and_collapses_blank_runs():
    assert hygiene.normalize_for_storage("a   \n\n\n\nb") == "a\n\nb"


def test_key_folds_whitespace_and_fullwidth_digits():
    # 全角数字 ３ -> 半角 3（NFKC）；多空白/换行折叠为单空格
    a = hygiene.turn_key_hash("买入 ３ 手\n止损")
    b = hygiene.turn_key_hash("买入 3 手 止损")
    assert a == b


def test_key_keeps_punctuation_distinct():
    # 不剥标点：路径差异必须区分
    assert hygiene.turn_key_hash("rm -rf /data") != hygiene.turn_key_hash("rm -rf data")


def test_key_stable_length():
    h = hygiene.turn_key_hash("hello")
    assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)
