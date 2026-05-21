from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from persist_utils import atomic_write_json, file_lock

if TYPE_CHECKING:
    from pydantic_ai.usage import UsageLimits as _UsageLimits

import logger
from dotenv import dotenv_values
from pydantic_ai.usage import UsageLimits
from runtime_state import AgentRunPolicy

_d = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
CONFIG_FILE = _d / "config.json"
DOTENV_FILE = _d / ".env" if getattr(sys, "frozen", False) else _d.parent / ".env"

_CONFIG: dict[str, Any] | None = None


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
    with file_lock(CONFIG_FILE):
        atomic_write_json(CONFIG_FILE, cfg)


def get_agent_usage_limits() -> "_UsageLimits":
    """单次 Agent 运行对模型请求次数上限"""
    cfg = settings()
    raw = cfg["request_limit"]
    if raw is None or raw.strip().lower() in ("none", "unlimited", "null"):
        return UsageLimits(request_limit=None)
    return UsageLimits(request_limit=int(raw))


def get_agent_run_policy() -> AgentRunPolicy:
    return AgentRunPolicy.from_config(settings())


def get_model_and_params(role: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
    from ModelGateway.ModelChecker import merge_litellm_into_model_params

    raw: dict[str, Any] = dict(settings()["models"][role])
    name = str(raw.pop("name")).strip()
    reff = raw.pop("reasoning_effort")
    s = str(reff).strip().lower()
    if s in ("none", "off", "false"):
        raw["reasoning_effort"] = False
    elif s in ("minimal", "low", "medium", "high", "xhigh", "max"):
        raw["reasoning_effort"] = s
    else:
        raw["reasoning_effort"] = "medium"
    raw["thinking"] = str(raw.pop("thinking")).strip().lower()

    out = merge_litellm_into_model_params(name, raw)
    out.update(kwargs)
    return name, out


def http_chat_completions_thinking_extras(model_params: dict[str, Any]) -> dict[str, Any]:
    reasoning_effort = model_params["reasoning_effort"]
    thinking_type = str(model_params["thinking"]).strip().lower()
    if thinking_type == "enabled" and reasoning_effort in ("minimal", "low", "medium", "high", "xhigh", "max"):
        return {"thinking": {"type": "enabled"}, "reasoning_effort": reasoning_effort}
    return {"thinking": {"type": "disabled"}}


def chat_completion_inference_request_fields(
    model_params: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """OpenAI 兼容 chat/completions 请求体中除 model、messages 外的推理相关字段；config 与运行时 kwargs 均可扩展。"""
    mp = dict(model_params)
    tex = http_chat_completions_thinking_extras(mp)
    mp.pop("thinking")
    mp.update(tex)
    mp.update(kwargs)
    return mp


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


def _merge_ctx(base: dict[str, Any], overlay: dict[str, Any], roles: tuple[str, ...]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if k in roles:
            continue
        out[k] = v
    return out


def get_agent_roles(**kwargs: Any) -> tuple[str, ...]:
    cfg = kwargs.pop("cfg", settings())
    models = cfg.get("models")
    roles: list[str] = []
    for role in models.keys():
        roles.append(role)
    return tuple(roles)


def get_context_profile_roles() -> tuple[str, ...]:
    """参与上下文配置（有独立 context 段）的角色，顺序与 config 中 context 键顺序一致。"""
    raw = settings().get("context")
    if not isinstance(raw, dict):
        return ()
    return tuple(k for k, v in raw.items() if k != "defaults" and isinstance(v, dict))


def get_context_config(role: str) -> dict[str, Any]:
    raw = settings().get("context")
    if not isinstance(raw, dict):
        return {}
    roles = get_agent_roles()
    if role not in roles:
        if not roles:
            return {}
        role = roles[0]
        logger.warning("警告，发现未知role，默认配置为%r", role)
    per_role = any(isinstance(raw.get(k), dict) for k in roles)
    out: dict[str, Any] = {}
    if per_role:
        d = raw.get("defaults")
        if isinstance(d, dict):
            out = _merge_ctx(out, d, roles)
        r = raw.get(role)
        if isinstance(r, dict):
            out = _merge_ctx(out, r, roles)
    else:
        out = _merge_ctx(out, raw, roles)
    return out
