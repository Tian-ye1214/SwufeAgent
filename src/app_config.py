"""从 config.json 加载/保存 API 与模型配置；运行时修改会写回文件。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import logger

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = None  # type: ignore[misc, assignment]

CONFIG_FILE = Path(__file__).resolve().parent / "config.json"
DOTENV_FILE = (Path(__file__).resolve().parent / ".." / ".env").resolve()

_CONFIG: dict[str, Any] | None = None


def _dotenv_map() -> dict[str, str]:
    if not dotenv_values or not DOTENV_FILE.is_file():
        return {}
    out: dict[str, str] = {}
    for k, v in (dotenv_values(DOTENV_FILE) or {}).items():
        if v is None or not str(v).strip():
            continue
        out[str(k).strip()] = str(v).strip()
    return out


def load_config() -> dict[str, Any]:
    global _CONFIG
    if not CONFIG_FILE.is_file():
        raise FileNotFoundError(
            f"缺少配置文件: {CONFIG_FILE}，请创建 config.json（含 BASE_URL、API_KEY、models 等）。"
        )
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        _CONFIG = json.load(f)
    return _CONFIG


def settings() -> dict[str, Any]:
    if _CONFIG is None:
        load_config()
    assert _CONFIG is not None
    return _CONFIG


def get_env(key: str, **kwargs: Any) -> str:
    w = bool(kwargs.pop("warn", True))
    default = str(kwargs.pop("default", ""))
    if kwargs:
        raise TypeError(f"get_env: 不支持的参数 {set(kwargs)}")

    v = (_dotenv_map().get(key) or "").strip()
    if v:
        return v

    raw = settings().get(key)
    if raw is not None and not isinstance(raw, (dict, list)):
        v = (raw or "").strip() if isinstance(raw, str) else str(raw).strip()
    else:
        v = ""
    if v:
        return v
    if w and not default:
        logger.warning("未配置 %r，请在 .env 或 config.json 根中填写。", key)
    return default


def save_config(cfg: dict[str, Any] | None = None) -> None:
    global _CONFIG
    cfg = cfg if cfg is not None else settings()
    _CONFIG = cfg
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _reasoning_effort_to_model_thinking(val: Any) -> Any | None:
    if val is None:
        return None

    s = str(val).strip().lower()
    if s in ("none", "off", "false"):
        return False
    for level in ("minimal", "low", "medium", "high", "xhigh"):
        if s == level:
            return level
    logger.warning("思考力度默认配置为medium")
    return "medium"


def get_model_and_params(role: str) -> tuple[str, dict[str, Any]]:
    if role not in ("manager", "worker", "coordinator"):
        raise ValueError(f"未知角色: {role}")
    m = settings()["models"][role]
    name = m["name"]
    params: dict[str, Any] = {
        "temperature": float(m["temperature"]),
        "max_tokens": int(m["max_tokens"]),
    }
    thinking = _reasoning_effort_to_model_thinking(m.get("reasoning_effort"))
    if thinking is not None:
        params["thinking"] = thinking
    from ModelGateway.ModelChecker import merge_litellm_into_model_params

    return name, merge_litellm_into_model_params(name, params)


def http_chat_completions_thinking_extras(model_params: dict[str, Any]) -> dict[str, Any]:
    """
    与 DeepSeek 官方 /chat/completions 体一致，供 httpx 直连时合并到 JSON（见文档 thinking + reasoning_effort）：
    - 启用：\"thinking\": {\"type\": \"enabled\"}，以及 \"reasoning_effort\" 档位字符串；
    - 关闭：\"thinking\": {\"type\": \"disabled\"}（不再带 reasoning_effort）；
    - 未配置 thinking（get_model_and_params 未写入）：返回 {}，与旧行为一致。
    """
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
    if role not in ("manager", "worker", "coordinator"):
        raise ValueError(f"未知角色: {role}")
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

CONTEXT_ROLE_KEYS = frozenset({"coordinator", "manager", "worker"})


def _merge_context_layer(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """将一层 overlay 合并进 base（处理 compressor 子 dict）。"""
    out = dict(base)
    for k, v in overlay.items():
        if k in CONTEXT_ROLE_KEYS:
            continue
        if k == "compressor" and isinstance(v, dict):
            prev = out.get("compressor")
            comp = dict(prev) if isinstance(prev, dict) else {}
            comp.update(v)
            out["compressor"] = comp
        else:
            out[k] = v
    return out


def _context_root_is_per_role(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    return any(
        k in CONTEXT_ROLE_KEYS and isinstance(raw.get(k), dict) for k in CONTEXT_ROLE_KEYS
    )


def get_context_config(role: str) -> dict[str, Any]:
    """
    返回指定角色合并后的 context 配置（不写回文件）。
    - 新格式：context.coordinator / context.manager / context.worker 各自独立；
      可选 context.defaults 先于角色层合并。
    - 旧格式：context 为单层扁平 dict（无角色子对象），则三角色共用该配置。
    """
    if role not in CONTEXT_ROLE_KEYS:
        role = "coordinator"
        logger.warning("警告，发现未知role，默认配置为coordinator")
    raw_root = settings().get("context")
    out: dict[str, Any] = {}
    if not isinstance(raw_root, dict):
        return out
    if _context_root_is_per_role(raw_root):
        defaults = raw_root.get("defaults")
        if isinstance(defaults, dict):
            out = _merge_context_layer(out, defaults)
        role_raw = raw_root.get(role)
        if isinstance(role_raw, dict):
            out = _merge_context_layer(out, role_raw)
    else:
        out = _merge_context_layer(out, raw_root)
    return out
