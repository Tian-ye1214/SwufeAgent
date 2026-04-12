"""从 config.json 加载/保存 API 与模型配置；运行时修改会写回文件。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

_CONFIG: dict[str, Any] | None = None


def load_config() -> dict[str, Any]:
    global _CONFIG
    if not CONFIG_FILE.is_file():
        raise FileNotFoundError(f"缺少配置文件: {CONFIG_FILE}，请创建 config.json（含 api_base、api_key、models）。")
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        _CONFIG = json.load(f)
    return _CONFIG


def get_config() -> dict[str, Any]:
    if _CONFIG is None:
        load_config()
    assert _CONFIG is not None
    return _CONFIG


def reload_config() -> dict[str, Any]:
    global _CONFIG
    _CONFIG = None
    return load_config()


def save_config(cfg: dict[str, Any] | None = None) -> None:
    global _CONFIG
    cfg = cfg if cfg is not None else get_config()
    _CONFIG = cfg
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def get_api_base() -> str:
    c = (get_config().get("api_base") or "").strip()
    if c:
        return c
    return (os.environ.get("BASE_URL") or "").strip()


def get_api_key() -> str:
    c = (get_config().get("api_key") or "").strip()
    if c:
        return c
    return (os.environ.get("API_KEY") or "").strip()


def apply_api_to_process_env() -> None:
    """将当前配置中的 api_base / api_key 同步到进程环境，供依赖环境变量的代码使用。"""
    cfg = get_config()
    base = (cfg.get("api_base") or "").strip()
    key = (cfg.get("api_key") or "").strip()
    if base:
        os.environ["BASE_URL"] = base
    if key:
        os.environ["API_KEY"] = key


def get_model_and_params(role: str) -> tuple[str, dict[str, Any]]:
    if role not in ("manager", "worker", "coordinator"):
        raise ValueError(f"未知角色: {role}")
    m = get_config()["models"][role]
    name = m["name"]
    params = {
        "temperature": float(m["temperature"]),
        "max_tokens": int(m["max_tokens"]),
    }
    return name, params


def set_model_name(role: str, model_name: str) -> None:
    if role not in ("manager", "worker", "coordinator"):
        raise ValueError(f"未知角色: {role}")
    cfg = get_config()
    cfg["models"][role]["name"] = model_name.strip()
    save_config(cfg)


def set_api(api_base: str | None, api_key: str | None) -> None:
    cfg = get_config()
    if api_base is not None:
        cfg["api_base"] = api_base.strip()
    if api_key is not None:
        cfg["api_key"] = api_key.strip()
    save_config(cfg)
    apply_api_to_process_env()
