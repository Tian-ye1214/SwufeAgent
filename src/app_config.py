from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import logger
from dotenv import dotenv_values

_d = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
CONFIG_FILE = _d / "config.json"
DOTENV_FILE = _d / ".env" if getattr(sys, "frozen", False) else _d.parent / ".env"

_CONFIG: dict[str, Any] | None = None
_ROLES = frozenset({"coordinator", "manager", "worker"})


def load_config() -> dict[str, Any]:
    global _CONFIG
    with open(CONFIG_FILE, encoding="utf-8") as f:
        _CONFIG = json.load(f)
    return _CONFIG


def settings() -> dict[str, Any]:
    if _CONFIG is None:
        load_config()
    return _CONFIG  # type: ignore[return-value]


def get_env(key: str, *, warn: bool = True, default: str = "") -> str:
    env: dict[str, str] = {}
    if DOTENV_FILE.is_file():
        for k, v in (dotenv_values(DOTENV_FILE) or {}).items():
            if v and str(v).strip():
                env[str(k).strip()] = str(v).strip()
    if val := (env.get(key) or "").strip():
        return val
    raw = settings().get(key)
    if raw is not None and not isinstance(raw, (dict, list)):
        s = raw.strip() if isinstance(raw, str) else str(raw).strip()
        if s:
            return s
    if warn and not default:
        logger.warning("未配置 %r，请在 .env 或 config.json 根中填写。", key)
    return default


def save_config(cfg: dict[str, Any] | None = None) -> None:
    global _CONFIG
    cfg = settings() if cfg is None else cfg
    _CONFIG = cfg
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def get_model_and_params(role: str) -> tuple[str, dict[str, Any]]:
    m = settings()["models"][role]
    name = m["name"]
    params: dict[str, Any] = {
        "temperature": float(m["temperature"]),
        "max_tokens": int(m["max_tokens"]),
    }
    reff = m.get("reasoning_effort")
    if reff is not None:
        s = str(reff).strip().lower()
        if s in ("none", "off", "false"):
            params["thinking"] = False
        elif s in ("minimal", "low", "medium", "high", "xhigh"):
            params["thinking"] = s
        else:
            params["thinking"] = "medium"
    from ModelGateway.ModelChecker import merge_litellm_into_model_params

    return name, merge_litellm_into_model_params(name, params)


def http_chat_completions_thinking_extras(model_params: dict[str, Any]) -> dict[str, Any]:
    th = model_params.get("thinking")
    if th is None:
        return {}
    if th is False:
        return {"thinking": {"type": "disabled"}}
    if th is True:
        return {"thinking": {"type": "enabled"}, "reasoning_effort": "medium"}
    if isinstance(th, str):
        return {"thinking": {"type": "enabled"}, "reasoning_effort": th}
    return {}


def set_model_name(role: str, model_name: str) -> None:
    cfg = settings()
    cfg["models"][role]["name"] = model_name.strip()
    save_config(cfg)


def set_api(base_url: str | None = None, api_key: str | None = None) -> None:
    cfg = settings()
    if base_url is not None:
        cfg["BASE_URL"] = base_url.strip()
    if api_key is not None:
        cfg["API_KEY"] = api_key.strip()
    save_config(cfg)


def _merge_ctx(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if k in _ROLES:
            continue
        if k == "compressor" and isinstance(v, dict):
            prev = out.get("compressor")
            c = dict(prev) if isinstance(prev, dict) else {}
            c.update(v)
            out["compressor"] = c
        else:
            out[k] = v
    return out


def get_context_config(role: str) -> dict[str, Any]:
    raw = settings().get("context")
    if not isinstance(raw, dict):
        return {}
    if role not in _ROLES:
        role = "coordinator"
        logger.warning("警告，发现未知role，默认配置为coordinator")
    per_role = any(k in _ROLES and isinstance(raw.get(k), dict) for k in _ROLES)
    out: dict[str, Any] = {}
    if per_role:
        d = raw.get("defaults")
        if isinstance(d, dict):
            out = _merge_ctx(out, d)
        r = raw.get(role)
        if isinstance(r, dict):
            out = _merge_ctx(out, r)
    else:
        out = _merge_ctx(out, raw)
    return out
