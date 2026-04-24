"""上下文窗口探测、Token 估算与各角色历史压缩（工具输出修剪 + LLM Markdown 摘要）。"""
from __future__ import annotations

import asyncio
import json
import re
import threading
from copy import deepcopy
from typing import Any, Literal

import httpx
import json_repair

from app_config import (
    get_context_config,
    get_env,
    get_model_and_params,
    http_chat_completions_thinking_extras,
)
from prompt import format_prompt_current_time, load_prompt
from tools.Memory import ChatHistory, _pydantic_messages_to_text

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore

from genai_prices.data_snapshot import get_snapshot as _genai_get_snapshot

import logger

from pydantic_ai.messages import (
    BaseToolCallPart,
    BaseToolReturnPart,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)

AgentRole = Literal["coordinator", "manager", "worker"]

_CONTEXT_LIMIT_CACHE: dict[str, int] = {}

_COMPRESS_PREFIX = "[CONTEXT_COMPRESSION_SUMMARY]"
_COMPRESS_MARKER = "<<COMPRESS_SUMMARY>>"
_COMPRESS_MARKER_LEGACY = "<<COMPRESS_JSON>>"


def _count_text_tokens(text: str, chars_per_token: float) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    if enc and text:
        return len(enc.encode(text))
    return max(1, int(len(text) / max(chars_per_token, 0.25)))


def estimate_message_tokens(msg: Any, chars_per_token: float) -> int:
    """单条 pydantic-ai 消息的 token 估算。"""
    if isinstance(msg, ModelRequest):
        parts: list[str] = []
        for p in msg.parts:
            if isinstance(p, UserPromptPart):
                c = p.content
                if isinstance(c, str):
                    parts.append(c)
                else:
                    parts.append(str(c))
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
    role: AgentRole = "coordinator",
) -> int:
    if not messages:
        return 0
    ctx = get_context_config(role)
    cpt = float(chars_per_token if chars_per_token is not None else ctx["token_estimate_fallback_chars_per_token"])
    return sum(estimate_message_tokens(m, cpt) for m in messages)



_MODEL_VARIANT_SUFFIXES = re.compile(
    r"[-_](?:pro|flash|turbo|latest|preview|mini|lite|plus|max|ultra|fast|small|medium|large|long|free|instruct)$",
    re.IGNORECASE,
)


def _normalize_model_name(name: str) -> str:
    """去掉供应商前缀和常见变体后缀，得到模型家族名。"""
    n = name.strip().lower()
    if "/" in n:
        n = n.rsplit("/", 1)[-1]
    n = re.sub(r"-\d{6,8}$", "", n)
    while True:
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
    for role in ("coordinator", "manager", "worker"):
        raw = get_context_config(role).get("litellm_model_prices_json_url")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _ensure_litellm_maps() -> None:
    """拉取 litellm JSON 并填充 _LITELLM_CONTEXT_MAP 与 _LITELLM_OUTPUT_MAP，只拉一次。"""
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

    ctx_map: dict[str, int] = {}
    out_map: dict[str, int] = {}
    for key, info in raw.items():
        if not isinstance(info, dict):
            continue
        low = key.lower()
        bare = key.rsplit("/", 1)[-1].lower() if "/" in key else None

        # 上下文窗口：max_input_tokens 优先，降级 max_tokens（litellm legacy）
        ctx = info.get("max_input_tokens") or info.get("max_tokens")
        if isinstance(ctx, int) and ctx >= 4096:
            ctx_map[low] = ctx
            if bare and bare not in ctx_map:
                ctx_map[bare] = ctx

        # 最大输出 tokens
        out = info.get("max_output_tokens")
        if isinstance(out, int) and out >= 1:
            out_map[low] = out
            if bare and bare not in out_map:
                out_map[bare] = out

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
    role: AgentRole | None = None,
) -> int:
    """有效上下文上限：config 覆盖 > 缓存 > 多源查找 > default_context_tokens。"""
    r: AgentRole = role if role is not None else "coordinator"
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

    logger.warning("无法解析模型上下文窗口，使用 default_context_tokens=%s（模型 %s）", fallback, mid)
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


def _parse_tool_args(args: Any) -> dict[str, Any] | None:
    if args is None:
        return None
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            obj = json_repair.loads(args)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def summarize_tool_result(tool_name: str, tool_args: Any, tool_content: str) -> str:
    """将工具调用结果压成单行摘要（供中间区 LLM 输入）。"""
    name = (tool_name or "").strip() or "unknown"
    content = tool_content or ""
    line_count = content.count("\n") + (1 if content else 0)
    args = _parse_tool_args(tool_args)

    if name in ("run_command", "terminal", "execute_file"):
        cmd = ""
        if args:
            cmd = str(args.get("command") or args.get("name") or args.get("args") or "")[:120]
        m = re.search(r"Return code:\s*(-?\d+)", content, re.I)
        code = m.group(1) if m else "?"
        return f"[{name}] ran `{cmd}` -> exit {code}, {line_count} lines"

    if name == "read_file":
        path = str(args.get("name") or args.get("path") or "") if args else ""
        return f"[read_file] read {path or '?'} ({len(content):,} chars)"

    if name in ("write_file", "append_to_file"):
        path = str(args.get("name") or "") if args else ""
        return f"[{name}] {path or '?'} ({len(content):,} chars)"

    if name == "search_web":
        q = str(args.get("query") or "")[:80] if args else ""
        return f"[search_web] query={q!r} ({len(content):,} chars)"

    if name.startswith("browser_"):
        return f"[{name}] ({len(content):,} chars)"

    arg_preview = ""
    if args:
        arg_preview = str(args)[:80]
    return f"[{name}] args={arg_preview!r} ({len(content):,} chars)"


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
    tail_budget: int,
    cpt: float,
) -> int:
    if not spans:
        return len(messages)
    tail_start = spans[-1][0]
    total = _tokens_range(messages, tail_start, len(messages), cpt)
    if total > tail_budget:
        return tail_start
    for k in range(len(spans) - 2, -1, -1):
        s, _ = spans[k]
        new_total = _tokens_range(messages, s, len(messages), cpt)
        if new_total <= tail_budget:
            tail_start = s
        else:
            break
    return tail_start


def _tool_call_lookup(messages: list, lo: int, hi: int) -> dict[str, tuple[str, Any]]:
    """tool_call_id -> (tool_name, args) for ModelResponse in [lo, hi)."""
    out: dict[str, tuple[str, Any]] = {}
    for idx in range(lo, hi):
        m = messages[idx]
        if isinstance(m, ModelResponse):
            for p in m.parts:
                if isinstance(p, BaseToolCallPart):
                    out[p.tool_call_id] = (p.tool_name, p.args)
    return out


def _middle_with_summarized_tools(messages: list, lo: int, hi: int) -> list:
    """深拷贝 [lo:hi) 并将 ToolReturnPart 内容替换为单行摘要。"""
    lookup = _tool_call_lookup(messages, 0, hi)
    out: list = []
    for idx in range(lo, hi):
        m = messages[idx]
        if not isinstance(m, ModelRequest):
            out.append(deepcopy(m))
            continue
        new_parts = []
        changed = False
        for p in m.parts:
            if isinstance(p, BaseToolReturnPart):
                tname, args = lookup.get(p.tool_call_id, (p.tool_name, None))
                text_content = p.model_response_str() if hasattr(p, "model_response_str") else str(p.content)
                new_parts.append(
                    ToolReturnPart(
                        tool_name=p.tool_name,
                        content=summarize_tool_result(tname, args, text_content),
                        tool_call_id=p.tool_call_id,
                    )
                )
                changed = True
            else:
                new_parts.append(p)
        if changed:
            out.append(ModelRequest(parts=new_parts))
        else:
            out.append(deepcopy(m))
    return out


def _extract_previous_summary_payload(text: str) -> str | None:
    for marker in (_COMPRESS_MARKER, _COMPRESS_MARKER_LEGACY):
        if marker not in text:
            continue
        try:
            return text.split(marker, 1)[1].strip()
        except Exception:
            return None
    return None


def _strip_outer_markdown_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    first_nl = t.find("\n")
    if first_nl == -1:
        return t
    inner = t[first_nl + 1 :]
    end = inner.rfind("```")
    if end == -1:
        return t
    return inner[:end].strip()


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
    role: AgentRole,
) -> str:
    ctx = get_context_config(role)
    comp = ctx["compressor"]
    model = comp.get("model") if isinstance(comp.get("model"), str) else None
    if not (model and model.strip()):
        model = get_model_and_params(role)[0]
    max_tokens = int(comp.get("max_tokens") or 4096)
    temperature = float(comp.get("temperature") or 0.2)

    base = (get_env("BASE_URL", warn=False) or "").strip().rstrip("/")
    key = (get_env("API_KEY", warn=False) or "").strip()
    if not base:
        raise RuntimeError("BASE_URL 为空，无法调用压缩模型")

    _, role_params = get_model_and_params(role)
    payload: dict = {
        "model": model.strip(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    payload.update(http_chat_completions_thinking_extras(role_params))
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    with httpx.Client(timeout=120.0, http2=True) as client:
        r = client.post(f"{base}/v1/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    raw = data["choices"][0]["message"]["content"]
    text = (raw or "").strip()
    if text.startswith("```"):
        text = _strip_outer_markdown_fence(text)
    elif "```" in text:
        fence = re.search(r"```(?:\w+)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
    if not text:
        text = "（压缩模型返回为空；请根据后续对话继续。）"
    return text


def compress_history(history: ChatHistory, *, role: AgentRole, force: bool = False) -> bool:
    """
    压缩消息历史：头尾保留，中间替换为 Markdown 摘要 User 消息。
    使用对应 role 的模型解析上下文上限。返回是否执行了压缩。
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
    if len(spans) < 2:
        logger.info("上下文压缩跳过：用户轮次不足")
        return False

    head_max_tokens = int(max_ctx * float(ctx["head_max_ratio"]))
    head_end = _compute_head_end(
        messages,
        spans,
        head_turns=int(ctx["head_turns"]),
        head_max_tokens=head_max_tokens,
        cpt=cpt,
    )
    tail_budget = int(max_ctx * float(ctx["tail_reserve_ratio"]))
    tail_start = _compute_tail_start(messages, spans, tail_budget=tail_budget, cpt=cpt)

    if head_end >= tail_start:
        logger.warning("上下文压缩跳过：头尾保护区重叠 (head_end=%s tail_start=%s)", head_end, tail_start)
        return False

    middle_lo, middle_hi = head_end, tail_start
    if middle_hi <= middle_lo:
        return False

    prev_summary = history.compress_summary_state
    if not prev_summary:
        for m in messages[:head_end]:
            if isinstance(m, ModelRequest):
                for p in m.parts:
                    if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                        extracted = _extract_previous_summary_payload(p.content)
                        if extracted:
                            prev_summary = extracted
                            break

    middle_for_prompt = _middle_with_summarized_tools(messages, middle_lo, middle_hi)
    excerpt = _pydantic_messages_to_text(middle_for_prompt)

    system_prompt = load_prompt("context_compress_structured_system.md").format(
        current_time=format_prompt_current_time()
    )
    user_parts: list[str] = []
    if prev_summary and prev_summary.strip():
        user_parts.append(
            "## Previous summary (Markdown; update and merge; do not drop critical facts)\n\n"
            f"{prev_summary.strip()}"
        )
    user_parts.append(
        "## Conversation excerpt to merge (middle segment; tool outputs may be one-line summaries)\n\n"
        f"{excerpt}"
    )
    user_content = "\n\n".join(user_parts)

    summary_md = _call_compressor_llm(
        system_prompt=system_prompt, user_content=user_content, role=role
    )
    new_body = _build_compress_user_body(summary_md)

    summary_msg = ModelRequest(parts=[UserPromptPart(content=new_body)])
    new_messages = messages[:head_end] + [summary_msg] + messages[tail_start:]
    history.set_messages(new_messages)
    history.compress_summary_state = summary_md.strip()
    return True


_ALL_AGENT_ROLES: tuple[AgentRole, ...] = ("coordinator", "manager", "worker")


async def get_effective_max_contexts_by_role_async() -> dict[AgentRole, int]:
    async def one(r: AgentRole) -> tuple[AgentRole, int]:
        n = await asyncio.to_thread(get_effective_max_context, None, role=r)
        return r, n

    pairs = await asyncio.gather(*(one(r) for r in _ALL_AGENT_ROLES))
    return dict(pairs)


async def prewarm_effective_max_contexts_by_role_async(
    *, reason: str = "startup"
) -> dict[AgentRole, int]:
    """并行预取三角色有效上下文并写入缓存；在启动与切换模型后调用。返回各角色 max token。"""
    d = await get_effective_max_contexts_by_role_async()
    logger.info(
        "各角色有效上下文 token 上限（%s）: coordinator=%s, manager=%s, worker=%s",
        reason,
        d["coordinator"],
        d["manager"],
        d["worker"],
    )
    return d


async def get_effective_max_context_async(
    model_name: str | None = None,
    *,
    role: AgentRole | None = None,
) -> int:
    return await asyncio.to_thread(lambda: get_effective_max_context(model_name, role=role))


async def estimate_history_tokens_async(
    messages: list,
    *,
    chars_per_token: float | None = None,
    role: AgentRole = "coordinator",
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
    role: AgentRole,
    force: bool = False,
) -> bool:
    return await asyncio.to_thread(compress_history, history, role=role, force=force)


async def maybe_auto_compress_async(history: ChatHistory, *, role: AgentRole) -> bool:
    return await asyncio.to_thread(compress_history, history, role=role)
