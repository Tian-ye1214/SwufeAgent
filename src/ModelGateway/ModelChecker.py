"""上下文窗口探测与各角色历史压缩（中间段摘录 + 结构化 Markdown 摘要）。"""
from __future__ import annotations

import asyncio
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
    settings,
)
from prompt import (
    format_prompt_current_time,
    load_prompt,
)
from tools.memory import ChatHistory, pydantic_messages_to_text

from genai_prices.data_snapshot import get_snapshot as _genai_get_snapshot
from ModelGateway.usage_accounting import latest_usage_input_tokens

import logger

from pydantic_ai.messages import (
    BaseToolCallPart,
    BaseToolReturnPart,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

_CONTEXT_LIMIT_CACHE: dict[str, int] = {}
_COMPRESS_PREFIX = "[CONTEXT_COMPRESSION_SUMMARY]"
_COMPRESS_MARKER = "<<COMPRESS_SUMMARY>>"
_MODEL_VARIANT_SUFFIXES = re.compile(
    r"[-_](?:pro|flash|turbo|latest|preview|mini|lite|plus|max|ultra|fast|small|medium|large|long|free|instruct)$",
    re.IGNORECASE,
)
_COMPRESS_REQUIRED_HEADINGS = (
    "## 原始目标与当前目标",
    "## 已完成节点",
    "## 待完成节点",
    "## 工具调用与关键结果",
    "## 当前状态",
    "## 未解决问题与阻塞",
    "## 用户约束与已做决策",
    "## 恢复后下一步",
)


class CompressionValidationError(RuntimeError):
    """压缩摘要或写回消息不满足可恢复检查点契约。"""


def context_usage_breakdown(
    role: str,
    history_messages: list,
    *,
    skills_manager: Any,
    memory_injection: str,
) -> dict[str, Any]:
    """基于最近一次真实模型 usage 的上下文占用。没有真实 usage 时不回退估算。"""
    ctx = get_context_config(role)
    max_tokens = get_effective_max_context(role=role)
    used = latest_usage_input_tokens(history_messages)
    total = int(used or 0)
    threshold = int(max_tokens * float(ctx["auto_compress_ratio"]))
    percent = 0.0 if max_tokens <= 0 else min(100.0, total * 100.0 / max_tokens)
    return {
        "has_usage": used is not None,
        "input": total,
        "total": total,
        "max": max_tokens,
        "threshold": threshold,
        "percent": percent,
    }


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


_OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
_OPENROUTER_LOCK = threading.Lock()
_OPENROUTER_CONTEXT_MAP: dict[str, int] | None = None
_OPENROUTER_OUTPUT_MAP: dict[str, int] | None = None
_OPENROUTER_META_MAP: dict[str, dict[str, Any]] | None = None
_OPENROUTER_FAILED = False

def _openrouter_cache_path() -> Path:
    d = logger.LOG_DIR / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / "openrouter_models.json"


def _openrouter_raw_to_maps(
    data: dict[str, Any],
) -> tuple[dict[str, int], dict[str, int], dict[str, dict[str, Any]]]:
    ctx_map: dict[str, int] = {}
    out_map: dict[str, int] = {}
    meta_map: dict[str, dict[str, Any]] = {}
    for m in data.get("data", []):
        if not isinstance(m, dict):
            continue
        mid = m.get("id", "").lower()
        if not mid:
            continue
        bare = mid.rsplit("/", 1)[-1] if "/" in mid else None

        meta = {
            k: m[k]
            for k in (
                "id",
                "name",
                "canonical_slug",
                "context_length",
                "top_provider",
                "pricing",
                "architecture",
                "supported_parameters",
                "default_parameters",
                "knowledge_cutoff",
                "expiration_date",
            )
            if k in m
        }
        meta_map[mid] = meta
        if bare and bare not in meta_map:
            meta_map[bare] = meta

        ctx = None
        tp = m.get("top_provider")
        if isinstance(tp, dict):
            tp_ctx = tp.get("context_length")
            if isinstance(tp_ctx, int) and tp_ctx >= 4096:
                ctx = tp_ctx
        if ctx is None:
            raw_ctx = m.get("context_length")
            if isinstance(raw_ctx, int) and raw_ctx >= 4096:
                ctx = raw_ctx
        if isinstance(ctx, int) and ctx >= 4096:
            ctx_map[mid] = ctx
            if bare and bare not in ctx_map:
                ctx_map[bare] = ctx

        if isinstance(tp, dict):
            out = tp.get("max_completion_tokens")
            if isinstance(out, int) and out >= 1:
                out_map[mid] = out
                if bare and bare not in out_map:
                    out_map[bare] = out

    return ctx_map, out_map, meta_map


def _ensure_openrouter_maps() -> None:
    global _OPENROUTER_CONTEXT_MAP, _OPENROUTER_OUTPUT_MAP, _OPENROUTER_META_MAP, _OPENROUTER_FAILED
    with _OPENROUTER_LOCK:
        if (
            _OPENROUTER_CONTEXT_MAP is not None
            and _OPENROUTER_OUTPUT_MAP is not None
            and _OPENROUTER_META_MAP is not None
        ):
            return
        if _OPENROUTER_FAILED:
            return

    cache_path = _openrouter_cache_path()
    if cache_path.is_file():
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            ctx_map, out_map, meta_map = _openrouter_raw_to_maps(cached)
            with _OPENROUTER_LOCK:
                _OPENROUTER_CONTEXT_MAP = ctx_map
                _OPENROUTER_OUTPUT_MAP = out_map
                _OPENROUTER_META_MAP = meta_map
                _OPENROUTER_FAILED = False
            logger.info(
                "OpenRouter 模型元数据已从缓存加载，context=%d 条 output=%d 条 meta=%d 条",
                len(ctx_map),
                len(out_map),
                len(meta_map),
            )
            return
        except Exception as e:
            logger.warning("OpenRouter 缓存读取失败: %s，将重新拉取", e)

    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(_OPENROUTER_URL)
            r.raise_for_status()
            raw = r.json()
    except Exception as e:
        logger.warning("拉取 OpenRouter 模型元数据失败 (%s): %s", _OPENROUTER_URL, e)
        with _OPENROUTER_LOCK:
            _OPENROUTER_FAILED = True
        return

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)
    except Exception as e:
        logger.warning("OpenRouter 缓存写入失败: %s", e)

    ctx_map, out_map, meta_map = _openrouter_raw_to_maps(raw)
    with _OPENROUTER_LOCK:
        _OPENROUTER_CONTEXT_MAP = ctx_map
        _OPENROUTER_OUTPUT_MAP = out_map
        _OPENROUTER_META_MAP = meta_map
        _OPENROUTER_FAILED = False
    logger.info(
        "OpenRouter 模型元数据已加载（%s），context=%d 条 output=%d 条 meta=%d 条",
        _OPENROUTER_URL,
        len(ctx_map),
        len(out_map),
        len(meta_map),
    )


def _map_lookup(name: str, m: dict | None) -> Any:
    if not m:
        return None
    low = name.lower()
    if low in m:
        return m[low]
    bare = low.rsplit("/", 1)[-1] if "/" in low else None
    if bare and bare in m:
        return m[bare]
    return None


def _lookup_openrouter(name: str) -> int | None:
    _ensure_openrouter_maps()
    return _map_lookup(name, _OPENROUTER_CONTEXT_MAP)


def _lookup_openrouter_output(name: str) -> int | None:
    _ensure_openrouter_maps()
    return _map_lookup(name, _OPENROUTER_OUTPUT_MAP)


def _lookup_openrouter_meta(name: str) -> dict[str, Any] | None:
    _ensure_openrouter_maps()
    meta = _map_lookup(name, _OPENROUTER_META_MAP)
    if meta:
        return meta
    normalized = _normalize_model_name(name)
    if normalized != name.lower():
        return _map_lookup(normalized, _OPENROUTER_META_MAP)
    return None


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
    """多源查找上下文窗口：OpenRouter > genai-prices 兜底。"""
    return _multi_source_lookup(model_name, (_lookup_openrouter, _lookup_genai_prices))


def lookup_model_max_output_tokens(model_name: str) -> int | None:
    """多源查找最大输出 tokens：OpenRouter API。"""
    return _multi_source_lookup(model_name, (_lookup_openrouter_output,))


def merge_openrouter_into_model_params(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    根据 OpenRouter 元数据覆盖模型参数：
    - max_output_tokens 作为 params['max_tokens'] 的上限
    """
    out = dict(params)
    max_out = lookup_model_max_output_tokens(model_name)
    if isinstance(max_out, int) and max_out > 0:
        configured = out.get("max_tokens")
        if isinstance(configured, int) and configured > 0:
            out["max_tokens"] = min(configured, max_out)
        else:
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


_TOOL_IMPORTANT_LINE_RE = re.compile(
    r"(exit\s+code|return\s+code|status\s*code|error|failed|failure|exception|traceback|"
    r"stderr|command|cmd|path|file|artifact|output)",
    re.IGNORECASE,
)


def _compact_tool_content_for_compress(tool_content: str, *, max_chars: int = 480) -> str:
    text = (tool_content or "").strip()
    if not text:
        return "content=empty"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    important = [line for line in lines if _TOOL_IMPORTANT_LINE_RE.search(line)]
    selected = important[:6] if important else lines[:4]
    compact = " | ".join(selected).strip()
    if not compact:
        compact = text.replace("\n", " ")
    if len(compact) > max_chars:
        compact = compact[: max_chars - 24].rstrip() + " ...(tool output truncated)"
    return compact


def summarize_tool_result(tool_name: str, tool_args: Any, tool_content: str) -> str:
    """中间段里工具输出的单行摘要，供压缩模型阅读。"""
    name = (tool_name or "").strip() or "tool"
    if isinstance(tool_args, dict):
        arg_line = json.dumps(tool_args, ensure_ascii=False)[:240]
    else:
        arg_line = str(tool_args)
    result_line = _compact_tool_content_for_compress(tool_content)
    return f"`{name}` {arg_line} → {len(tool_content)} chars; {result_line}"


def _elide_tool_call_args_for_compress(arg_txt: str, max_chars: int) -> str:
    s = (arg_txt or "").strip()
    if len(s) <= max_chars:
        return s
    head = max_chars - 24
    if head < 80:
        head = 80
    return s[:head] + " …(tool args truncated)"


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


def _compute_head_end(
    spans: list[tuple[int, int]],
    *,
    head_turns: int,
) -> int:
    if not spans or head_turns <= 0:
        return 0
    n = min(head_turns, len(spans))
    return spans[n - 1][1]


def _compute_tail_start(
    spans: list[tuple[int, int]],
    *,
    tail_turns: int,
) -> int:
    if not spans or tail_turns <= 0:
        return spans[-1][1] if spans else 0
    n = min(tail_turns, len(spans))
    return spans[len(spans) - n][0]


def _tool_calls_up_to(messages: list, hi: int) -> dict[str, tuple[str, Any]]:
    out: dict[str, tuple[str, Any]] = {}
    for idx in range(0, hi):
        m = messages[idx]
        if isinstance(m, ModelResponse):
            for p in m.parts:
                if isinstance(p, BaseToolCallPart):
                    out[p.tool_call_id] = (p.tool_name, p.args)
    return out


def _middle_segment_markdown(messages: list, lo: int, hi: int, **kwargs: Any) -> str:
    """提取中间对话：用户消息、助手文本、工具调用与工具返回（返回体为单行摘要）。"""
    by_id = _tool_calls_up_to(messages, hi)
    chunks: list[str] = []
    seen_tool_return_keys: set[tuple[str, str]] = set()
    arg_cap = int(kwargs["middle_tool_args_max_chars"])
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
                    dedupe_key = (p.tool_name or "", one_line)
                    if dedupe_key in seen_tool_return_keys:
                        continue
                    seen_tool_return_keys.add(dedupe_key)
                    chunks.append(f"### 工具返回 `{p.tool_name}`\n{one_line}")
        elif isinstance(m, ModelResponse):
            for p in m.parts:
                if isinstance(p, TextPart):
                    chunks.append(f"### 助手\n{p.content}")
                elif isinstance(p, BaseToolCallPart):
                    arg_txt = p.args if isinstance(p.args, str) else json.dumps(p.args, ensure_ascii=False)
                    arg_txt = _elide_tool_call_args_for_compress(arg_txt, arg_cap)
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
        pydantic_messages_to_text(messages_before), encoding="utf-8"
    )
    before_llm = f"## compressor system\n\n{system_prompt}\n\n## compressor user\n\n{user_content}\n"
    (run_dir / "before_compress.md").write_text(before_llm, encoding="utf-8")
    (run_dir / "compressor_output.md").write_text(summary_md, encoding="utf-8")
    (run_dir / "after_context.md").write_text(
        pydantic_messages_to_text(new_messages), encoding="utf-8"
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


def _level2_sections(markdown: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line
            current_lines = []
        elif current_heading is not None:
            current_lines.append(raw_line)
    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return sections


def _nonempty_section(section_body: str) -> bool:
    return bool(section_body.strip())


def _lint_compression_summary(summary_md: str) -> str:
    body = (summary_md or "").strip()
    errors: list[str] = []
    if not body:
        errors.append("压缩摘要为空")
    if "```" in body:
        errors.append("压缩摘要不能包含 Markdown code fence")
    if body.startswith("{") or body.startswith("["):
        errors.append("压缩摘要必须是 Markdown，不得输出 JSON")

    sections = _level2_sections(body)
    headings = [heading for heading, _ in sections]
    required = list(_COMPRESS_REQUIRED_HEADINGS)
    missing = [h for h in required if h not in headings]
    if missing:
        errors.append(
            "压缩摘要缺少必需标题: "
            f"missing={missing!r} actual={headings!r}"
        )
    else:
        bodies = {heading: section_body for heading, section_body in sections}
        if not _nonempty_section(bodies["## 原始目标与当前目标"]):
            errors.append("`原始目标与当前目标` 不能为空")
        if not (
            _nonempty_section(bodies["## 已完成节点"])
            or _nonempty_section(bodies["## 待完成节点"])
        ):
            errors.append("`已完成节点` / `待完成节点` 至少一个不能为空")

    if errors:
        raise CompressionValidationError("; ".join(errors))
    return body


def _validate_compression_message(summary_msg: ModelRequest) -> None:
    parts = list(summary_msg.parts)
    if len(parts) != 1 or not isinstance(parts[0], UserPromptPart):
        raise CompressionValidationError("压缩摘要必须写入单个 UserPromptPart")
    content = parts[0].content
    if not content.startswith(f"{_COMPRESS_PREFIX}\n"):
        raise CompressionValidationError("压缩摘要缺少 CONTEXT_COMPRESSION_SUMMARY 前缀")
    if f"\n{_COMPRESS_MARKER}\n" not in content:
        raise CompressionValidationError("压缩摘要缺少 COMPRESS_SUMMARY marker")


def _validate_model_messages_round_trip(messages: list[Any]) -> None:
    try:
        raw = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
        ModelMessagesTypeAdapter.validate_python(raw)
    except Exception as e:
        raise CompressionValidationError(
            f"压缩后 model_messages dump/validate 失败: {e}"
        ) from e


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
        **chat_completion_inference_request_fields(comp_params, model_name=model, **kwargs),
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    with httpx.Client(timeout=120.0, http2=True) as client:
        r = client.post(f"{base}/v1/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def compress_history(
    history: ChatHistory,
    *,
    role: str,
    force: bool,
    task_state: str | None = None,
) -> bool:
    """
    压缩三步：1) 按阈值或 force 触发；2) 头尾保留，中间段展成 Markdown 摘录；
    3) 压缩模型输出固定结构的 Markdown，写入一条 User 摘要消息。
    """
    messages = list(history.messages)
    if len(messages) < 2:
        return False

    ctx = get_context_config(role)
    max_ctx = get_effective_max_context(role=role)
    used = latest_usage_input_tokens(messages)
    threshold = max_ctx * float(ctx["auto_compress_ratio"])

    if not force and (used is None or used < threshold):
        return False

    cfg = settings()["context"]["compression"]
    knobs = {
            "middle_tool_args_max_chars": int(cfg["middle_tool_args_max_chars"]),
    }

    spans = _user_spans(messages)

    head_end = _compute_head_end(
        spans,
        head_turns=int(ctx["head_turns"]),
    )
    tail_start = _compute_tail_start(
        spans,
        tail_turns=int(ctx["tail_turns"]),
    )

    if head_end >= tail_start:
        logger.warning("上下文压缩跳过：头尾保护区重叠 (head_end=%s tail_start=%s)", head_end, tail_start)
        return False

    middle_lo, middle_hi = head_end, tail_start

    prev_summary = history.compress_summary_state
    excerpt = _middle_segment_markdown(messages, middle_lo, middle_hi, **knobs)

    system_prompt = load_prompt("context_compress_structured_system.md").format(
        current_time=format_prompt_current_time()
    )
    user_parts: list[str] = []
    if prev_summary:
        user_parts.append(
            "## 上轮压缩摘要（必须合并更新，不能丢失仍有效信息）\n\n"
            + prev_summary
        )
    if task_state and task_state.strip():
        user_parts.append("## 当前结构化任务状态（权威）\n\n" + task_state.strip())
    user_parts.append("## 本轮待压缩中间段\n\n" + (excerpt or "unknown"))
    user_content = "\n\n".join(user_parts)

    summary_md = _call_compressor_llm(
        system_prompt=system_prompt, user_content=user_content
    )
    summary_md = _lint_compression_summary(summary_md)
    new_body = _build_compress_user_body(summary_md)

    summary_msg = ModelRequest(parts=[UserPromptPart(content=new_body)])
    new_messages = messages[:head_end] + [summary_msg] + messages[tail_start:]
    _validate_compression_message(summary_msg)
    _validate_model_messages_round_trip(new_messages)
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


async def compress_history_async(
    history: ChatHistory,
    *,
    role: str,
    force: bool,
    task_state: str | None = None,
) -> bool:
    return await asyncio.to_thread(
        compress_history,
        history,
        role=role,
        force=force,
        task_state=task_state,
    )


async def maybe_auto_compress_async(
    history: ChatHistory,
    *,
    role: str,
    task_state: str | None = None,
) -> bool:
    return await asyncio.to_thread(
        compress_history,
        history,
        role=role,
        force=False,
        task_state=task_state,
    )
