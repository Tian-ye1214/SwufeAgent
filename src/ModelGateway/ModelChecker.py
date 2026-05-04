"""上下文窗口探测、Token 估算与各角色历史压缩（中间段摘录 + 结构化 Markdown 摘要）。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from app_config import (
    chat_completion_inference_request_fields,
    get_agent_roles,
    get_context_config,
    get_context_profile_roles,
    get_env,
    get_model_and_params,
)
from prompt import format_prompt_current_time, load_prompt
from tools.Memory import ChatHistory, _pydantic_messages_to_text

import tiktoken

from genai_prices.data_snapshot import get_snapshot as _genai_get_snapshot

import logger

from pydantic_ai.messages import (
    BaseToolCallPart,
    BaseToolReturnPart,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

_CONTEXT_LIMIT_CACHE: dict[str, int] = {}
_COMPRESS_PREFIX = "[CONTEXT_COMPRESSION_SUMMARY]"
_COMPRESS_MARKER = "<<COMPRESS_SUMMARY>>"


def _count_text_tokens(text: str, chars_per_token: float) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception as e:
        logger.error(f"tiktoken分词错误: {e}")
        return max(1, int(len(text) / max(chars_per_token, 0.25)))


def estimate_message_tokens(msg: Any, chars_per_token: float) -> int:
    """单条 pydantic-ai 消息的 token 估算。"""
    if isinstance(msg, ModelRequest):
        parts: list[str] = []
        for p in msg.parts:
            if isinstance(p, UserPromptPart):
                parts.append(str(p.content))
            elif isinstance(p, BaseToolReturnPart):
                parts.append(f"{p.tool_name}:{p.tool_call_id}:{p.model_response_str()}")
        return _count_text_tokens("\n".join(parts), chars_per_token)
    if isinstance(msg, ModelResponse):
        parts: list[str] = []
        for p in msg.parts:
            if isinstance(p, TextPart):
                parts.append(p.content)
            elif isinstance(p, BaseToolCallPart):
                args = p.args if isinstance(p.args, str) else json.dumps(p.args, ensure_ascii=False)
                parts.append(f"{p.tool_name}:{args}")
        return _count_text_tokens("\n".join(parts), chars_per_token)
    return _count_text_tokens(str(msg), chars_per_token)


def estimate_history_tokens(
    messages: list,
    *,
    chars_per_token: float | None = None,
    role
) -> int:
    if not messages:
        return 0
    ctx = get_context_config(role)
    cpt = float(chars_per_token if chars_per_token is not None else ctx["token_estimate_fallback_chars_per_token"])
    return sum(estimate_message_tokens(m, cpt) for m in messages)


def _normalize_model_name(name: str) -> str:
    """去掉供应商前缀和常见变体后缀，得到模型家族名。"""
    n = name.strip().lower()
    if "/" in n:
        n = n.rsplit("/", 1)[-1]
    n = re.sub(r"-\d{6,8}$", "", n)
    while True:
        _MODEL_VARIANT_SUFFIXES = re.compile(
            r"[-_](?:pro|flash|turbo|latest|preview|mini|lite|plus|max|ultra|fast|small|medium|large|long|free|instruct)$",
            re.IGNORECASE,
        )
        stripped = _MODEL_VARIANT_SUFFIXES.sub("", n)
        if stripped == n:
            break
        n = stripped
    return n


def _lookup_genai_prices(name: str) -> int | None:
    try:
        snap = _genai_get_snapshot()
    except Exception:
        return None
    for provider in snap.providers:
        for m in provider.models:
            if m.is_match(name) and isinstance(m.context_window, int) and m.context_window >= 4096:
                return m.context_window
    return None


_LITELLM_LOCK = threading.Lock()
_LITELLM_CONTEXT_MAP: dict[str, int] | None = None   # max_input_tokens（降级 max_tokens）
_LITELLM_OUTPUT_MAP: dict[str, int] | None = None    # max_output_tokens
_LITELLM_FAILED = False
_LITELLM_URL_USED: str | None = None


def _litellm_model_prices_json_url() -> str | None:
    for role in get_agent_roles():
        raw = get_context_config(role).get("litellm_model_prices_json_url")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _litellm_cache_file_path(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    d = logger.LOG_DIR / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"litellm_model_prices_{digest}.json"


def _read_litellm_cache(url: str) -> dict[str, Any] | None:
    path = _litellm_cache_file_path(url)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        env = json.load(f)
    if env.get("url") != url:
        return None
    return env["body"]


def _write_litellm_cache(url: str, raw: dict[str, Any]) -> None:
    path = _litellm_cache_file_path(url)
    envelope = {"url": url, "body": raw}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, ensure_ascii=False)


def _litellm_raw_to_maps(raw: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    ctx_map: dict[str, int] = {}
    out_map: dict[str, int] = {}
    for key, info in raw.items():
        if not isinstance(info, dict):
            continue
        low = key.lower()
        bare = key.rsplit("/", 1)[-1].lower() if "/" in key else None

        ctx = info.get("max_input_tokens") or info.get("max_tokens")
        if isinstance(ctx, int) and ctx >= 4096:
            ctx_map[low] = ctx
            if bare and bare not in ctx_map:
                ctx_map[bare] = ctx

        out = info.get("max_output_tokens")
        if isinstance(out, int) and out >= 1:
            out_map[low] = out
            if bare and bare not in out_map:
                out_map[bare] = out

    return ctx_map, out_map


def _ensure_litellm_maps() -> None:
    """拉取 litellm JSON 并填充 _LITELLM_CONTEXT_MAP 与 _LITELLM_OUTPUT_MAP，只拉一次；有缓存则直接读盘。"""
    global _LITELLM_CONTEXT_MAP, _LITELLM_OUTPUT_MAP, _LITELLM_FAILED, _LITELLM_URL_USED
    url = _litellm_model_prices_json_url()
    cache_key = url or "__no_url__"
    with _LITELLM_LOCK:
        if _LITELLM_CONTEXT_MAP is not None and _LITELLM_URL_USED == cache_key:
            return
        if _LITELLM_FAILED and _LITELLM_URL_USED == cache_key:
            return
    if not url:
        logger.warning(
            "未配置 litellm_model_prices_json_url（可在 context.defaults 或任一 context.coordinator|manager|worker 下设置），"
            "跳过 litellm 模型元数据拉取"
        )
        with _LITELLM_LOCK:
            _LITELLM_CONTEXT_MAP = {}
            _LITELLM_OUTPUT_MAP = {}
            _LITELLM_URL_USED = cache_key
            _LITELLM_FAILED = False
        return

    cached = _read_litellm_cache(url)
    if cached is not None:
        ctx_map, out_map = _litellm_raw_to_maps(cached)
        with _LITELLM_LOCK:
            _LITELLM_CONTEXT_MAP = ctx_map
            _LITELLM_OUTPUT_MAP = out_map
            _LITELLM_URL_USED = cache_key
            _LITELLM_FAILED = False
        logger.info("litellm 模型元数据已加载（%s），context=%d 条 output=%d 条", url, len(ctx_map), len(out_map))
        return

    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(url)
            r.raise_for_status()
            raw = r.json()
    except Exception as e:
        logger.warning("拉取 litellm 模型元数据失败 (%s): %s", url, e)
        with _LITELLM_LOCK:
            _LITELLM_FAILED = True
            _LITELLM_URL_USED = cache_key
        return

    _write_litellm_cache(url, raw)
    ctx_map, out_map = _litellm_raw_to_maps(raw)
    with _LITELLM_LOCK:
        _LITELLM_CONTEXT_MAP = ctx_map
        _LITELLM_OUTPUT_MAP = out_map
        _LITELLM_URL_USED = cache_key
        _LITELLM_FAILED = False
    logger.info("litellm 模型元数据已加载（%s），context=%d 条 output=%d 条", url, len(ctx_map), len(out_map))


def _litellm_lookup(name: str, m: dict[str, int] | None) -> int | None:
    if not m:
        return None
    low = name.lower()
    if low in m:
        return m[low]
    bare = low.rsplit("/", 1)[-1] if "/" in low else None
    if bare and bare in m:
        return m[bare]
    return None


def _lookup_litellm(name: str) -> int | None:
    _ensure_litellm_maps()
    return _litellm_lookup(name, _LITELLM_CONTEXT_MAP)


def _lookup_litellm_output(name: str) -> int | None:
    _ensure_litellm_maps()
    return _litellm_lookup(name, _LITELLM_OUTPUT_MAP)


def _multi_source_lookup(name: str, fns: tuple) -> int | None:
    """按顺序尝试多个查找函数，精确匹配优先，再用归一化名兜底。"""
    for fn in fns:
        val = fn(name)
        if val:
            return val
    normalized = _normalize_model_name(name)
    if normalized != name.lower():
        for fn in fns:
            val = fn(normalized)
            if val:
                return val
    return None


def lookup_model_context(model_name: str) -> int | None:
    """多源查找上下文窗口：内置字典 > genai-prices > litellm(max_input_tokens/max_tokens)。"""
    return _multi_source_lookup(model_name, (_lookup_genai_prices, _lookup_litellm))


def lookup_model_max_output_tokens(model_name: str) -> int | None:
    """多源查找最大输出 tokens：内置字典 > litellm(max_output_tokens)。"""
    return _multi_source_lookup(model_name, (_lookup_litellm_output,))


def merge_litellm_into_model_params(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    根据 litellm 元数据覆盖模型参数：
    - max_output_tokens 覆盖 params['max_tokens']
    """
    out = dict(params)
    max_out = lookup_model_max_output_tokens(model_name)
    if isinstance(max_out, int) and max_out > 0:
        out["max_tokens"] = max_out
    return out


def get_effective_max_context(
    model_name: str | None = None,
    *,
    role,
) -> int:
    """有效上下文上限：config 覆盖 > 缓存 > 多源查找 > default_context_tokens。"""
    r: str = role if role is not None else get_context_profile_roles()[0]
    ctx = get_context_config(r)
    fallback = int(ctx["default_context_tokens"])
    mid = model_name if model_name is not None else get_model_and_params(r)[0]

    raw_max = ctx.get("max_context_tokens")
    if isinstance(raw_max, int) and raw_max > 0:
        _CONTEXT_LIMIT_CACHE[mid] = raw_max
        return raw_max

    if mid in _CONTEXT_LIMIT_CACHE:
        return _CONTEXT_LIMIT_CACHE[mid]

    looked = lookup_model_context(mid)
    if looked:
        logger.info("模型 %s 上下文窗口已从模型元数据解析: %d", mid, looked)
        _CONTEXT_LIMIT_CACHE[mid] = looked
        return looked

    logger.debug("使用 default_context_tokens=%s（模型 %s）", fallback, mid)
    _CONTEXT_LIMIT_CACHE[mid] = fallback
    return fallback


def format_context_usage_line(
    used_tokens: int,
    max_tokens: int,
    *,
    width: int = 18,
) -> str:
    if max_tokens <= 0:
        return f"[ctx] {used_tokens} tok (max unknown)"
    pct = min(100.0, 100.0 * used_tokens / max_tokens)
    filled = int(round(width * pct / 100.0))
    filled = min(width, max(0, filled))
    bar = "=" * filled + "." * (width - filled)

    def _k(n: int) -> str:
        if n >= 1000:
            return f"{n / 1000:.1f}k"
        return str(n)

    return f"[ctx {bar}] {pct:.0f}% ({_k(used_tokens)}/{_k(max_tokens)} tok)"


def summarize_tool_result(tool_name: str, tool_args: Any, tool_content: str) -> str:
    """中间段里工具输出的单行摘要，供压缩模型阅读。"""
    name = (tool_name or "").strip() or "tool"
    if isinstance(tool_args, dict):
        arg_line = json.dumps(tool_args, ensure_ascii=False)[:240]
    else:
        arg_line = str(tool_args)
    return f"`{name}` {arg_line} → {len(tool_content)} chars"


def _user_spans(messages: list) -> list[tuple[int, int]]:
    """每个元素为 (span_start_msg_index, exclusive_end)。"""
    starts: list[int] = []
    for i, m in enumerate(messages):
        if isinstance(m, ModelRequest) and any(isinstance(p, UserPromptPart) for p in m.parts):
            starts.append(i)
    spans: list[tuple[int, int]] = []
    for j, s in enumerate(starts):
        end = starts[j + 1] if j + 1 < len(starts) else len(messages)
        spans.append((s, end))
    return spans


def _tokens_range(messages: list, i: int, j: int, cpt: float) -> int:
    if i >= j:
        return 0
    return sum(estimate_message_tokens(messages[k], cpt) for k in range(i, j))


def _compute_head_end(
    messages: list,
    spans: list[tuple[int, int]],
    *,
    head_turns: int,
    head_max_tokens: int,
    cpt: float,
) -> int:
    if not spans or head_turns <= 0:
        return 0
    n = min(head_turns, len(spans))
    end_idx = spans[n - 1][1]
    while n > 0 and _tokens_range(messages, 0, end_idx, cpt) > head_max_tokens:
        n -= 1
        if n <= 0:
            return 0
        end_idx = spans[n - 1][1]
    return end_idx


def _compute_tail_start(
    messages: list,
    spans: list[tuple[int, int]],
    *,
    tail_turns: int,
    tail_max_tokens: int,
    cpt: float,
) -> int:
    if not spans or tail_turns <= 0:
        return len(messages)
    n = min(tail_turns, len(spans))
    start_idx = spans[len(spans) - n][0]
    while n > 0 and _tokens_range(messages, start_idx, len(messages), cpt) > tail_max_tokens:
        n -= 1
        if n <= 0:
            return len(messages)
        start_idx = spans[len(spans) - n][0]
    return start_idx


def _tool_calls_up_to(messages: list, hi: int) -> dict[str, tuple[str, Any]]:
    out: dict[str, tuple[str, Any]] = {}
    for idx in range(0, hi):
        m = messages[idx]
        if isinstance(m, ModelResponse):
            for p in m.parts:
                if isinstance(p, BaseToolCallPart):
                    out[p.tool_call_id] = (p.tool_name, p.args)
    return out


def _middle_segment_markdown(messages: list, lo: int, hi: int) -> str:
    """提取中间对话：用户消息、助手文本、工具调用与工具返回（返回体为单行摘要）。"""
    by_id = _tool_calls_up_to(messages, hi)
    chunks: list[str] = []
    for idx in range(lo, hi):
        m = messages[idx]
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, UserPromptPart):
                    body = p.content if isinstance(p.content, str) else str(p.content)
                    chunks.append(f"### 用户\n{body}")
                elif isinstance(p, BaseToolReturnPart):
                    tname, args = by_id.get(p.tool_call_id, (p.tool_name, None))
                    text_content = p.model_response_str()
                    one_line = summarize_tool_result(tname, args, text_content)
                    chunks.append(f"### 工具返回 `{p.tool_name}`\n{one_line}")
        elif isinstance(m, ModelResponse):
            for p in m.parts:
                if isinstance(p, TextPart):
                    chunks.append(f"### 助手\n{p.content}")
                elif isinstance(p, BaseToolCallPart):
                    arg_txt = p.args if isinstance(p.args, str) else json.dumps(p.args, ensure_ascii=False)
                    chunks.append(f"### 工具调用 `{p.tool_name}`\n{arg_txt}")
    return "\n\n".join(chunks)


def _save_compress_debug_artifacts(
    *,
    role: str,
    system_prompt: str,
    user_content: str,
    summary_md: str,
    messages_before: list,
    new_messages: list,
    head_end: int,
    tail_start: int,
) -> None:
    root = logger.LOG_DIR / "context_compress_debug"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{int(time.time() * 1000)}_{role}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "before_messages.md").write_text(
        _pydantic_messages_to_text(messages_before), encoding="utf-8"
    )
    before_llm = f"## compressor system\n\n{system_prompt}\n\n## compressor user\n\n{user_content}\n"
    (run_dir / "before_compress.md").write_text(before_llm, encoding="utf-8")
    (run_dir / "compressor_output.md").write_text(summary_md, encoding="utf-8")
    (run_dir / "after_context.md").write_text(
        _pydantic_messages_to_text(new_messages), encoding="utf-8"
    )
    (run_dir / "slice_bounds.txt").write_text(
        f"role={role}\nhead_end={head_end}\ntail_start={tail_start}\n",
        encoding="utf-8",
    )
    logger.info("上下文压缩调试已保存: %s", run_dir)


def _build_compress_user_body(summary_md: str) -> str:
    body = summary_md.strip()
    return (
        f"{_COMPRESS_PREFIX}\n"
        "以下内容为此前对话的压缩摘要（Markdown）。请结合后续消息继续推理。\n"
        f"{_COMPRESS_MARKER}\n"
        f"{body}"
    )


def _call_compressor_llm(
    *,
    system_prompt: str,
    user_content: str,
    **kwargs: Any,
) -> str:
    model, comp_params = get_model_and_params("compressor")

    base = (get_env("BASE_URL", warn=False) or "").strip().rstrip("/")
    key = (get_env("API_KEY", warn=False) or "").strip()
    if not base:
        raise RuntimeError("BASE_URL 为空，无法调用压缩模型")

    payload: dict[str, Any] = {
        "model": model.strip(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        **chat_completion_inference_request_fields(comp_params, **kwargs),
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    with httpx.Client(timeout=120.0, http2=True) as client:
        r = client.post(f"{base}/v1/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def compress_history(history: ChatHistory, *, role: str, force: bool) -> bool:
    """
    压缩三步：1) 按阈值或 force 触发；2) 头尾保留，中间段展成 Markdown 摘录；
    3) 压缩模型输出固定结构的 Markdown，写入一条 User 摘要消息。
    """
    messages = list(history.messages)
    if len(messages) < 2:
        return False

    ctx = get_context_config(role)
    cpt = float(ctx["token_estimate_fallback_chars_per_token"])
    max_ctx = get_effective_max_context(role=role)
    used = estimate_history_tokens(messages, chars_per_token=cpt, role=role)
    threshold = max_ctx * float(ctx["auto_compress_ratio"])

    if not force and used < threshold:
        return False

    spans = _user_spans(messages)

    head_max_tokens = int(max_ctx * float(ctx["head_max_ratio"]))
    head_end = _compute_head_end(
        messages,
        spans,
        head_turns=int(ctx["head_turns"]),
        head_max_tokens=head_max_tokens,
        cpt=cpt,
    )
    tail_max_tokens = int(max_ctx * float(ctx["tail_max_ratio"]))
    tail_start = _compute_tail_start(
        messages,
        spans,
        tail_turns=int(ctx["tail_turns"]),
        tail_max_tokens=tail_max_tokens,
        cpt=cpt,
    )

    if head_end >= tail_start:
        logger.warning("上下文压缩跳过：头尾保护区重叠 (head_end=%s tail_start=%s)", head_end, tail_start)
        return False

    middle_lo, middle_hi = head_end, tail_start
    if middle_hi <= middle_lo:
        return False

    prev_summary = history.compress_summary_state
    excerpt = _middle_segment_markdown(messages, middle_lo, middle_hi)

    system_prompt = load_prompt("context_compress_structured_system.md").format(
        current_time=format_prompt_current_time()
    )
    user_parts: list[str] = []
    if prev_summary:
        user_parts.append("## 上轮压缩摘要（合并更新）\n\n" + prev_summary)
    user_parts.append("## 本轮待压缩中间段\n\n" + excerpt)
    user_content = "\n\n".join(user_parts)

    summary_md = _call_compressor_llm(
        system_prompt=system_prompt, user_content=user_content
    )
    new_body = _build_compress_user_body(summary_md)

    summary_msg = ModelRequest(parts=[UserPromptPart(content=new_body)])
    new_messages = messages[:head_end] + [summary_msg] + messages[tail_start:]
    _save_compress_debug_artifacts(
        role=role,
        system_prompt=system_prompt,
        user_content=user_content,
        summary_md=summary_md,
        messages_before=messages,
        new_messages=new_messages,
        head_end=head_end,
        tail_start=tail_start,
    )
    history.set_messages(new_messages)
    history.compress_summary_state = summary_md.strip()
    return True


async def get_effective_max_contexts_by_role_async(**kwargs: Any) -> dict[str, int]:
    configured_roles = kwargs.pop("roles", get_agent_roles())
    if kwargs:
        raise TypeError(
            f"get_effective_max_contexts_by_role_async() got unexpected keyword arguments: {tuple(kwargs.keys())}"
        )
    allowed = set(get_context_profile_roles())
    roles = tuple(
        role
        for role in configured_roles
        if isinstance(role, str) and role in allowed
    )

    async def one(r: str) -> tuple[str, int]:
        n = await asyncio.to_thread(get_effective_max_context, None, role=r)
        return r, n

    pairs = await asyncio.gather(*(one(r) for r in roles))
    return dict(pairs)


async def prewarm_effective_max_contexts_by_role_async(
    *, reason: str = "startup"
) -> dict[str, int]:
    """并行预取三角色有效上下文并写入缓存；在启动与切换模型后调用。返回各角色 max token。"""
    d = await get_effective_max_contexts_by_role_async()
    log_values = ", ".join(f"{role}={value}" for role, value in d.items())
    logger.info(
        "各角色有效上下文 token 上限（%s）: %s",
        reason,
        log_values,
    )
    return d


async def get_effective_max_context_async(
    model_name: str | None = None,
    *,
    role: str | None = None,
) -> int:
    return await asyncio.to_thread(lambda: get_effective_max_context(model_name, role=role))


async def estimate_history_tokens_async(
    messages: list,
    *,
    chars_per_token: float | None = None,
    role: str,
) -> int:
    return await asyncio.to_thread(
        estimate_history_tokens,
        messages,
        chars_per_token=chars_per_token,
        role=role,
    )


async def compress_history_async(
    history: ChatHistory,
    *,
    role: str,
    force: bool,
) -> bool:
    return await asyncio.to_thread(compress_history, history, role=role, force=force)


async def maybe_auto_compress_async(history: ChatHistory, *, role: str) -> bool:
    return await asyncio.to_thread(compress_history, history, role=role, force=False)
