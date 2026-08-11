from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from redlotus.infra.persist_utils import save_locked_json

if TYPE_CHECKING:
    from pydantic_ai.usage import UsageLimits as _UsageLimits

from redlotus.infra import logger
from dotenv import dotenv_values
from redlotus.infra.paths import config_file, default_config_file, dotenv_file
from pydantic_ai.usage import UsageLimits
from redlotus.runtime.runtime_state import AgentRunPolicy

CONFIG_FILE, DOTENV_FILE = config_file(), dotenv_file()

_CONFIG: dict[str, Any] | None = None
_DOTENV_CACHE: dict[str, str] | None = None
_API_CONFIG_KEYS = {"BASE_URL", "API_KEY", "SILICONFLOW_BASE", "SILICONFLOW_KEY"}
THINKING_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")


def _seed_config_if_missing() -> None:
    if CONFIG_FILE.exists():
        return
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    src = default_config_file()
    try:
        text = src.read_text(encoding="utf-8") if src.is_file() else "{}\n"
    except OSError:
        text = "{}\n"
    CONFIG_FILE.write_text(text, encoding="utf-8")


def load_config() -> dict[str, Any]:
    global _CONFIG
    _seed_config_if_missing()
    with open(CONFIG_FILE, encoding="utf-8") as f:
        _CONFIG = json.load(f)
    return _CONFIG


def reload_config() -> dict[str, Any]:
    global _CONFIG, _DOTENV_CACHE
    _CONFIG = None
    _DOTENV_CACHE = None
    return load_config()


def settings() -> dict[str, Any]:
    if _CONFIG is None:
        load_config()
    return _CONFIG  # type: ignore[return-value]


def _dotenv_files() -> list[Path]:
    """.env 来源：用户配置目录优先，当前工作目录（项目本地）覆盖之。"""
    out: list[Path] = []
    for p in (DOTENV_FILE, Path.cwd() / ".env"):
        if p not in out:
            out.append(p)
    return out


def _dotenv_values() -> dict[str, str]:
    """解析并缓存 .env（进程内静态）：键值均 strip，空值丢弃；cwd/.env 覆盖用户目录 .env。"""
    global _DOTENV_CACHE
    if _DOTENV_CACHE is None:
        env: dict[str, str] = {}
        for f in _dotenv_files():
            if f.is_file():
                for k, v in (dotenv_values(f) or {}).items():
                    if v and str(v).strip():
                        env[str(k).strip()] = str(v).strip()
        _DOTENV_CACHE = env
    return _DOTENV_CACHE


def _config_scalar(key: str) -> str:
    raw = settings().get(key)
    if raw is not None and not isinstance(raw, (dict, list)):
        return raw.strip() if isinstance(raw, str) else str(raw).strip()
    return ""


def get_env(key: str, *, warn: bool = True, default: str = "") -> str:
    """配置读取唯一入口；/api 管理的 key 让 config.json 优先于 .env。"""
    if env_val := (os.environ.get(key) or "").strip():
        return env_val
    if key in _API_CONFIG_KEYS:
        if val := _config_scalar(key):
            return val
        if val := (_dotenv_values().get(key) or "").strip():
            return val
        if warn and not default:
            logger.warning("未配置 %r，请在 .env 或 config.json 根中填写。", key)
        return default
    if val := (_dotenv_values().get(key) or "").strip():
        return val
    if val := _config_scalar(key):
        return val
    if warn and not default:
        logger.warning("未配置 %r，请在 .env 或 config.json 根中填写。", key)
    return default


def _missing_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(key for key in keys if not get_env(key, warn=False).strip())


def missing_main_api_keys() -> tuple[str, ...]:
    return _missing_keys(("BASE_URL", "API_KEY"))


def missing_rag_api_keys() -> tuple[str, ...]:
    return _missing_keys(("SILICONFLOW_BASE", "SILICONFLOW_KEY"))


def save_config(cfg: dict[str, Any] | None = None) -> None:
    global _CONFIG
    cfg = settings() if cfg is None else cfg
    _CONFIG = cfg
    save_locked_json(CONFIG_FILE, cfg)


def get_agent_usage_limits() -> "_UsageLimits":
    """单次 Agent 运行对模型请求次数上限"""
    cfg = settings()
    raw = cfg["request_limit"]
    if raw is None or str(raw).strip().lower() in ("none", "unlimited", "null", ""):
        return UsageLimits(request_limit=None)
    return UsageLimits(request_limit=int(raw))


def get_agent_run_policy() -> AgentRunPolicy:
    return AgentRunPolicy.from_config(settings())


def _lookup_openrouter_meta_for_model(model_name: str | None) -> dict[str, Any] | None:
    if not model_name:
        return None
    from redlotus.ModelGateway.ModelChecker import _lookup_openrouter_meta

    return _lookup_openrouter_meta(model_name)


def _supported_thinking_efforts_from_meta(meta: dict[str, Any] | None) -> tuple[str, ...]:
    raw = meta.get("supported_efforts") if meta else None
    if not isinstance(raw, list):
        return THINKING_EFFORTS
    supported = {str(e).strip().lower() for e in raw}
    return tuple(e for e in THINKING_EFFORTS if e in supported)


def supported_thinking_efforts(model_name: str | None) -> tuple[str, ...]:
    return _supported_thinking_efforts_from_meta(_lookup_openrouter_meta_for_model(model_name))


def role_supported_thinking_efforts(role: str) -> tuple[str, ...]:
    model_cfg = settings()["models"][role]
    return supported_thinking_efforts(str(model_cfg.get("name") or "").strip())


def _resolve_thinking_effort(effort: str, supported: tuple[str, ...]) -> str | None:
    effort = str(effort).strip().lower()
    return effort if effort in supported else (supported[-1] if supported else None)


def apply_thinking_config(
    model_params: dict[str, Any],
    *,
    model_name: str | None = None,
    chat_completions: bool = False,
    strict_effort: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    params = deepcopy(model_params)
    params.update(deepcopy(kwargs))
    thinking = str(params.pop("thinking", "")).strip().lower()
    effort = str(params.pop("reasoning_effort", "")).strip().lower()
    if thinking != "enabled":
        if not chat_completions and thinking in ("disabled", "off", "false"):
            extra_body = params.get("extra_body")
            params["extra_body"] = {
                **(extra_body if isinstance(extra_body, dict) else {}),
                "thinking": {"type": "disabled"},
            }
        elif chat_completions and thinking in ("disabled", "off", "false"):
            params["thinking"] = {"type": "disabled"}
        return params

    meta = _lookup_openrouter_meta_for_model(model_name)
    resolved_effort = _resolve_thinking_effort(
        effort,
        _supported_thinking_efforts_from_meta(meta),
    )
    if resolved_effort is None:
        return params

    raw_supported_params = meta.get("supported_parameters") if meta else None
    supported_params = {
        str(p)
        for p in raw_supported_params
    } if isinstance(raw_supported_params, list) else None
    if chat_completions:
        if supported_params is None:
            params.update({"thinking": {"type": "enabled"}, "reasoning_effort": resolved_effort})
        elif "reasoning" in supported_params:
            params["reasoning"] = {"effort": resolved_effort}
        elif "include_reasoning" in supported_params:
            params["include_reasoning"] = True
        elif "reasoning_effort" in supported_params:
            params["reasoning_effort"] = resolved_effort
        return params if supported_params is None else {k: v for k, v in params.items() if k in supported_params}

    params["thinking"] = resolved_effort
    return params


def get_model_and_params(role: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
    from redlotus.ModelGateway.ModelChecker import merge_openrouter_into_model_params

    raw: dict[str, Any] = deepcopy(settings()["models"][role])
    name = str(raw.pop("name")).strip()
    out = merge_openrouter_into_model_params(name, raw)
    out.update(deepcopy(kwargs))
    return name, out


def role_supports_input_modality(role: str, modality: str) -> bool:
    from redlotus.ModelGateway.ModelChecker import (
        _lookup_openrouter_meta,
        model_supports_input_modality,
    )

    model_cfg = settings().get("models", {}).get(role, {})
    model_name = str(model_cfg.get("name") or "").strip()
    if not model_name:
        return True
    if _lookup_openrouter_meta(model_name) is None:
        return True
    return model_supports_input_modality(model_name, modality)


def set_model_name(role: str, model_name: str) -> None:
    cfg = settings()
    cfg["models"][role]["name"] = model_name.strip()
    save_config(cfg)


def set_api(
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    embedding_url: str | None = None,
    embedding_key: str | None = None,
) -> None:
    values = {
        "BASE_URL": base_url,
        "API_KEY": api_key,
        "SILICONFLOW_BASE": embedding_url,
        "SILICONFLOW_KEY": embedding_key,
    }
    if all(v is None for v in values.values()):
        return
    cfg = settings()
    for key, value in values.items():
        if value is not None:
            cfg[key] = value.strip()
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
    return tuple(cfg.get("models").keys())


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
