from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkerResult:
    success: bool
    artifacts: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    needs_user_confirmation: bool = False


def parse_worker_result(output: str) -> WorkerResult:
    text = (output or "").strip()
    structured = _parse_json_object(text)
    if structured is not None:
        status = str(structured.get("status", "")).strip().lower()
        if status in {"success", "succeeded", "ok", "completed"}:
            return _from_structured(True, structured)
        if status in {"failed", "failure", "error"}:
            return _from_structured(False, structured)

    first_line = text.splitlines()[0].upper() if text else ""
    if first_line.startswith("FAILED:"):
        return WorkerResult(False)
    if first_line.startswith("CONFIRM:"):
        return WorkerResult(True, needs_user_confirmation=True)
    if first_line.startswith("SUCCESS:"):
        return WorkerResult(True)
    if text.upper().startswith("ERROR:"):
        return WorkerResult(False)
    return WorkerResult(True)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    candidate = text
    if "```" in text:
        for part in text.split("```"):
            p = part.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                candidate = p
                break
    if not candidate.startswith("{"):
        return None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _from_structured(success: bool, data: dict[str, Any]) -> WorkerResult:
    return WorkerResult(
        success=success,
        artifacts=_string_list(data.get("artifacts")),
        risks=_string_list(data.get("risks")),
        needs_user_confirmation=bool(data.get("needs_user_confirmation", False)),
    )
