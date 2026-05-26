from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worker_result import parse_worker_result


def test_parse_worker_result_accepts_structured_success_json() -> None:
    parsed = parse_worker_result(
        '{"status": "success", "summary": "created report", "artifacts": ["report.md"], "needs_user_confirmation": false}'
    )

    assert parsed.success is True
    assert parsed.summary == "created report"
    assert parsed.artifacts == ["report.md"]
    assert parsed.needs_user_confirmation is False


def test_parse_worker_result_accepts_structured_failed_json() -> None:
    parsed = parse_worker_result('{"status": "failed", "summary": "network unavailable"}')

    assert parsed.success is False
    assert parsed.summary == "network unavailable"


def test_parse_worker_result_accepts_fenced_structured_json() -> None:
    parsed = parse_worker_result(
        '```json\n{"status": "completed", "summary": "done", "risks": ["none"]}\n```'
    )

    assert parsed.success is True
    assert parsed.summary == "done"
    assert parsed.risks == ["none"]


def test_parse_worker_result_accepts_status_prefixes() -> None:
    assert parse_worker_result("FAILED: no file").success is False
    assert parse_worker_result("SUCCESS: wrote file").success is True
    assert parse_worker_result("plain answer").success is True
