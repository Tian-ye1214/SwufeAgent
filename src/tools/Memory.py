from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
import os
import mimetypes
from typing import Any

import json_repair
import httpx
import logger
from pydantic_ai import BinaryContent
from pydantic_ai.messages import (
    BaseToolReturnPart,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from app_config import (
    chat_completion_inference_request_fields,
    get_env,
    get_model_and_params,
)
from RAG.RAG import RAG as RAGEngineCls
from tools.conversation_log import read_saved_model_messages_file


def _pack_messages_to_chunk_texts(
    messages: list[Any],
    chunk_max_tokens: int,
    chars_per_token: float,
) -> list[str]:
    from ModelGateway.ModelChecker import estimate_message_tokens

    if not messages:
        return []
    out: list[str] = []
    buf: list[Any] = []
    buf_tokens = 0
    for m in messages:
        t = estimate_message_tokens(m, chars_per_token)
        if buf and buf_tokens + t > chunk_max_tokens:
            out.append(_pydantic_messages_to_text(buf))
            buf = []
            buf_tokens = 0
        buf.append(m)
        buf_tokens += t
    if buf:
        out.append(_pydantic_messages_to_text(buf))
    return out


@dataclass
class UserMessage:
    """用户消息：文本 + 可选的多模态附件（图片/视频）。"""

    _MEDIA_EXT_PATTERN = re.compile(
        r'[a-zA-Z0-9_\-./\\:]+\.(?:jpg|jpeg|png|gif|webp|bmp|mp4|avi|mov|mkv|webm)',
        re.IGNORECASE,
    )

    text: str
    attachments: list = field(default_factory=list)

    def to_prompt(self):
        """转换为 pydantic-ai agent.run() 可接受的 prompt 格式。"""
        if not self.attachments:
            return self.text
        return [self.text, *self.attachments]

    def _read_file_to_binary(self, path: str) -> BinaryContent | None:
        """读取本地文件，返回 BinaryContent（pydantic-ai 会自动 base64 编码）。"""
        if not os.path.isfile(path):
            return None
        mime, _ = mimetypes.guess_type(path)
        if mime is None:
            mime = 'image/png'
        with open(path, 'rb') as f:
            data = f.read()
        return BinaryContent(data=data, media_type=mime)


def user_message_from_text(message: str | "UserMessage") -> "UserMessage":
    if isinstance(message, UserMessage):
        return message
    return UserMessage(text=str(message))


def user_message_from_cli_input(raw_input: str) -> "UserMessage":
    """解析命令行输入：用正则提取图片/视频文件路径，读取字节作为附件。"""
    attachments: list = []
    text = raw_input
    um = UserMessage(text="", attachments=[])  # 仅用于复用实例方法读取文件

    for match in UserMessage._MEDIA_EXT_PATTERN.finditer(raw_input):
        path = match.group()
        bc = um._read_file_to_binary(path)
        if bc:
            attachments.append(bc)
            text = text.replace(path, "")

    text = text.strip()
    if not text and attachments:
        text = "请分析这些内容。"
    return UserMessage(text=text, attachments=attachments)


class ChatHistory:
    """通用对话历史管理器，可被任意 Agent 组件复用。

    封装了 pydantic-ai 中反复出现的 message_history 读写模式：
        result = agent.run(prompt, message_history=history.messages)
        history.update(result)
    """

    __slots__ = ("_messages", "_compress_summary_state")

    def __init__(self):
        self._messages: list = []
        self._compress_summary_state: str | None = None

    def update(self, result) -> None:
        """从 RunResult / StreamedRunResult 提取完整消息列表并保存。"""
        self._messages = list(result.all_messages())

    def reset(self) -> None:
        self._messages = []
        self._compress_summary_state = None

    def set_messages(self, messages: list) -> None:
        """直接替换消息列表（供上下文压缩等使用）。"""
        self._messages = list(messages)

    @property
    def compress_summary_state(self) -> str | None:
        """上一轮压缩模型产出的 Markdown 摘要文本，供下次压缩合并。"""
        return self._compress_summary_state

    @compress_summary_state.setter
    def compress_summary_state(self, value: str | None) -> None:
        self._compress_summary_state = value

    @property
    def messages(self) -> list:
        """传入 agent.run(message_history=...) 的只读引用。"""
        return self._messages

    def __len__(self) -> int:
        return len(self._messages)

    def __bool__(self) -> bool:
        return bool(self._messages)


def _pydantic_messages_to_text(messages: list) -> str:
    """将 pydantic-ai 消息对象列表转为可读文本，供长期记忆合并等使用。"""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    content = part.content
                    if isinstance(content, str):
                        lines.append(f"[USER]: {content}")
                elif isinstance(part, BaseToolReturnPart):
                    c = part.model_response_str() if hasattr(part, "model_response_str") else str(part.content)
                    lines.append(f"[TOOL_RESULT:{part.tool_name}]: {c[:500]}")
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    lines.append(f"[ASSISTANT]: {part.content}")
                elif isinstance(part, ToolCallPart):
                    args_str = part.args if isinstance(part.args, str) else str(part.args)
                    lines.append(f"[TOOL_CALL:{part.tool_name}]: {args_str[:300]}")
    return "\n\n".join(lines)


class ShortTermMemory:
    """短期记忆：`logs/conversations/**/messages_*.model_messages.json` 按消息边界打包入向量库，供 Worker 检索。"""

    def __init__(
        self,
        chunk_max_tokens: int,
        chars_per_token: float,
        rag: Any | None = None,
        log_root: Path | None = None,
    ):
        self._chunk_max_tokens = chunk_max_tokens
        self._chars_per_token = chars_per_token
        self._rag = rag
        self._log_root = log_root
        self._ingest_lock = asyncio.Lock()

    def _get_rag(self) -> Any:
        if self._rag is None:
            self._rag = RAGEngineCls()
        return self._rag

    def _stm_state_path(self) -> Path:
        rag = self._get_rag()
        return Path(rag._db.db_path) / "conversation_stm_state.json"

    def _rel_log_key(self, path: Path, root: Path) -> str:
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def _load_stm_state_sync(self) -> dict[str, Any]:
        p = self._stm_state_path()
        if not p.is_file():
            return {"version": 1, "sources": {}}
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"version": 1, "sources": {}}
            src = data.get("sources")
            if not isinstance(src, dict):
                data["sources"] = {}
            return data
        except Exception:
            return {"version": 1, "sources": {}}

    def _save_stm_state_sync(self, state: dict[str, Any]) -> None:
        p = self._stm_state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _sync_scan_conversations_and_collect_pairs(
        self,
        root: Path,
    ) -> tuple[dict[str, Any], list[tuple[str, str]]]:
        state = self._load_stm_state_sync()
        sources: dict[str, Any] = state.setdefault("sources", {})
        pairs: list[tuple[str, str]] = []
        conv = root / "conversations"
        if not conv.is_dir():
            return state, pairs
        files = sorted(
            conv.rglob("messages_*.model_messages.json"),
            key=lambda p: str(p),
        )
        for fp in files:
            key = self._rel_log_key(fp, root)
            messages, _meta = read_saved_model_messages_file(fp)
            chunk_texts = _pack_messages_to_chunk_texts(
                messages,
                self._chunk_max_tokens,
                self._chars_per_token,
            )
            total = len(chunk_texts)
            done = 0
            ent = sources.get(key)
            if isinstance(ent, dict):
                try:
                    done = int(ent.get("chunks_done", 0))
                except (TypeError, ValueError):
                    done = 0
            if total < done:
                done = 0
            for i in range(done, total):
                pairs.append((f"{key}#c{i}", chunk_texts[i]))
            sources[key] = {"chunks_done": total}
        return state, pairs

    async def _ingest_conversation_delta_unlocked(self) -> None:
        root = (self._log_root if self._log_root is not None else logger.LOG_DIR).resolve()
        if not root.is_dir():
            return

        rag = self._get_rag()
        await rag.connect()
        await rag._db.ensure_connected()

        state, pairs = await asyncio.to_thread(
            self._sync_scan_conversations_and_collect_pairs,
            root,
        )
        if pairs:
            await rag.ingest_chunk_pairs(pairs)
        await asyncio.to_thread(self._save_stm_state_sync, state)

    async def query_short_term_memory(self, query: str) -> str:
        """
        在历史对话向量索引中检索与 query 相关的片段；结果为模型可用的短期记忆上下文。

        Parameters:
            query: 检索查询语句
        """
        q = query.strip()
        inner = ""
        if q:
            async with self._ingest_lock:
                await self._ingest_conversation_delta_unlocked()
                rag = self._get_rag()
                await rag.connect()
                hits = await rag.retrieve(q)

            if hits:
                inner = "\n\n".join(
                    f"[{i + 1}] (source: {h.get('source', '')})\n{h.get('text', '')}"
                    for i, h in enumerate(hits)
                )
        return f"<ShortTermMemory>\n{inner}\n</ShortTermMemory>"


class LongTermMemory:
    """跨会话长期记忆，持久化到 src/prompts/LongTermMemory/ 目录。

    - MEMORY_GUIDANCE.md：给模型的长期记忆**使用**说明（随 load 注入 get_injection，不是抽取提示词）
    - soul_user_consolidation.md：从对话或日志增量中**合并**出 SOUL/USER 整篇正文的**唯一**提示词
    - SOUL.md / USER.md：各一份 **Markdown** 文件（首行 `# SOUL` / `# USER`，正文为整体叙述）；旧版无标题的纯文本会在加载时整篇当正文；\\n---\\n 分隔的段落会合并
    - log_sources_state.json：各 .log 已读字节偏移，避免重复喂给模型；**不存在时首次加载会自动创建**（version + 空 sources）
    """
    _CHAR_LIMIT = 8000
    _PROMPT_BODY_ELIDE = 10_000
    _INJECTION_PATTERN = re.compile(
        r'(?i)(ignore\s+previous|disregard\s+instruction|system\s+prompt|jailbreak|prompt\s+injection)',
    )

    _MEMORY_DIR = Path(__file__).resolve().parent.parent / "prompts" / "LongTermMemory"
    _GUIDANCE_FILE = "MEMORY_GUIDANCE.md"
    _CONSOLIDATION_TEMPLATE_FILE = "soul_user_consolidation.md"
    _LOG_STATE_FILE = "log_sources_state.json"
    _TARGETS = {"soul": "SOUL.md", "user": "USER.md"}
    _MD_H1_SOUL = re.compile(r"^#\s*SOUL\s*$", re.IGNORECASE)
    _MD_H1_USER = re.compile(r"^#\s*USER\s*$", re.IGNORECASE)
    _CONSOLIDATION_MAX_TRANSCRIPT_CHARS = 28_000
    _CONSOLIDATION_MAX_OUTPUT_TOKENS = 4096
    _LOG_READ_CHUNK_BYTES = 72_000

    def __init__(self):
        self._guidance_text: str = ""
        self._soul_body: str = ""
        self._user_body: str = ""
        self._write_lock = asyncio.Lock()
        self._logs_consolidate_lock = asyncio.Lock()

    def _sync_read_guidance(self) -> str:
        path = self._MEMORY_DIR / self._GUIDANCE_FILE
        self._MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8").strip()

    async def load(self) -> None:
        """从磁盘加载 MEMORY_GUIDANCE.md、SOUL/USER 正文。应在程序启动时调用一次。"""
        await asyncio.to_thread(self._load_sync)

    def refresh_from_disk_sync(self) -> None:
        """同步从磁盘刷新指引与 SOUL/USER（供 AgentSystem 启动等非 async 场景）。"""
        self._load_sync()

    def _load_sync(self) -> None:
        self._MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_soul_user_files_sync()
        self._ensure_log_state_file_sync()
        self._guidance_text = self._sync_read_guidance()
        self._soul_body = self._read_whole_body(self._MEMORY_DIR / "SOUL.md")
        self._user_body = self._read_whole_body(self._MEMORY_DIR / "USER.md")

    def _ensure_soul_user_files_sync(self) -> None:
        """若 SOUL.md / USER.md 不存在，则创建带 Markdown 标题的空文件。"""
        for key, name in self._TARGETS.items():
            path = self._MEMORY_DIR / name
            if path.is_file():
                continue
            title = "SOUL" if key == "soul" else "USER"
            path.write_text(f"# {title}\n\n", encoding="utf-8")

    def _ensure_log_state_file_sync(self) -> None:
        """若 log_sources_state.json 不存在，则创建默认 `{\"version\": 1, \"sources\": {}}`。"""
        p = self._log_state_path()
        if p.is_file():
            return
        self._save_log_state_sync({"version": 1, "sources": {}})

    @staticmethod
    def _read_whole_body(path: Path) -> str:
        if not path.exists():
            return ""
        raw = path.read_text(encoding="utf-8")
        stripped = raw.strip()
        if not stripped:
            return ""
        lines = raw.splitlines()
        first = (lines[0] if lines else "").strip()
        if LongTermMemory._MD_H1_SOUL.match(first) or LongTermMemory._MD_H1_USER.match(
            first
        ):
            raw = "\n".join(lines[1:]).lstrip()
        else:
            raw = stripped
        if "\n---\n" in raw:
            parts = [p.strip() for p in raw.split("\n---\n") if p.strip()]
            return "\n\n".join(parts)
        return raw.strip()

    def _format_markdown_file(self, target: str, body: str) -> str:
        """将内存中的正文落盘为带一级标题的 Markdown。"""
        title = "SOUL" if target == "soul" else "USER"
        b = (body or "").rstrip()
        if b:
            return f"# {title}\n\n{b}\n"
        return f"# {title}\n\n"

    def _save_sync(self, target: str) -> None:
        path = self._MEMORY_DIR / self._TARGETS[target]
        path.parent.mkdir(parents=True, exist_ok=True)
        body = self._soul_body if target == "soul" else self._user_body
        path.write_text(self._format_markdown_file(target, body), encoding="utf-8")

    def _scan(self, content: str) -> str | None:
        """检测 prompt 注入等危险内容，返回错误信息或 None。"""
        if LongTermMemory._INJECTION_PATTERN.search(content):
            return "Content contains potentially unsafe injection patterns."
        return None

    def _clip_for_storage(self, text: str) -> str:
        t = (text or "").strip()
        cap = type(self)._CHAR_LIMIT
        if len(t) <= cap:
            return t
        return t[:cap].rstrip()

    @staticmethod
    def _elide_for_consolidation_prompt(text: str, max_chars: int) -> str:
        t = (text or "").strip()
        if len(t) <= max_chars:
            return t
        return f"…(前部已省略，共 {len(t)} 字；以下为尾部 {max_chars} 字)…\n\n" + t[-max_chars:]

    def _get_body(self, target: str) -> str:
        return self._soul_body if target == "soul" else self._user_body

    def _set_body(self, target: str, text: str) -> None:
        if target == "soul":
            self._soul_body = text
        else:
            self._user_body = text

    def get_injection(self) -> str:
        """返回注入到系统提示的记忆文本块，为空时返回空字符串。"""
        parts: list[str] = []
        g = (self._guidance_text or "").strip()
        if g:
            parts.append(g)
        s = (self._soul_body or "").strip()
        if s:
            parts.append(f"## AI Memory\n\n{s}")
        u = (self._user_body or "").strip()
        if u:
            parts.append(f"## User Preferences\n\n{u}")
        return "\n\n".join(parts)

    def _load_consolidation_template_sync(self) -> str:
        path = self._MEMORY_DIR / self._CONSOLIDATION_TEMPLATE_FILE
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")

    def _parse_consolidation_response(self, raw: str) -> dict[str, str | None]:
        """输出整篇正文的 JSON：soul / user 为字符串或 null。兼容旧版数组，将合并为一段。"""
        text = (raw or "").strip()
        if "```" in text:
            fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
            if fence:
                text = fence.group(1).strip()
        try:
            obj = json_repair.loads(text)
        except Exception:
            return {"soul": None, "user": None}
        if not isinstance(obj, dict):
            return {"soul": None, "user": None}
        out: dict[str, str | None] = {"soul": None, "user": None}
        for key in ("soul", "user"):
            val = obj.get(key)
            if val is None:
                out[key] = None
            elif isinstance(val, str):
                s = val.strip()
                out[key] = s if s else None
            elif isinstance(val, list):
                parts = [p.strip() for p in val if isinstance(p, str) and p.strip()]
                out[key] = "\n\n".join(parts) if parts else None
            else:
                out[key] = None
        return out

    async def _apply_consolidation_parsed(self, parsed: dict[str, str | None]) -> bool:
        """以模型给出的整篇正文**替换**对应 SOUL/USER；无更新则返回 False。"""
        changed = False
        async with self._write_lock:
            await asyncio.to_thread(self._load_sync)
            for key in ("soul", "user"):
                val = parsed.get(key)
                if not isinstance(val, str) or not val.strip():
                    continue
                text = self._clip_for_storage(val)
                if not text:
                    continue
                err = self._scan(text)
                if err:
                    logger.warning(f"长期记忆合并已跳过 {key}：{err}")
                    continue
                self._set_body(key, text)
                await asyncio.to_thread(self._save_sync, key)
                changed = True
        return changed

    def _log_state_path(self) -> Path:
        return self._MEMORY_DIR / self._LOG_STATE_FILE

    def _load_log_state_sync(self) -> dict[str, Any]:
        self._ensure_log_state_file_sync()
        p = self._log_state_path()
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"version": 1, "sources": {}}
            src = data.get("sources")
            if not isinstance(src, dict):
                data["sources"] = {}
            return data
        except Exception:
            return {"version": 1, "sources": {}}

    def _save_log_state_sync(self, state: dict[str, Any]) -> None:
        self._MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        with self._log_state_path().open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _rel_log_key(self, path: Path, root: Path) -> str:
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def _merge_log_state_updates_sync(self, updates: dict[str, int]) -> None:
        """合并已处理到的字节偏移（仅更新传入的键）。"""
        if not updates:
            return
        state = self._load_log_state_sync()
        sources = state.setdefault("sources", {})
        for k, br in updates.items():
            sources[k] = {"bytes_read": int(br)}
        self._save_log_state_sync(state)

    async def _consolidation_chat_and_parse(
        self,
        template: str,
        transcript: str,
        *,
        truncate: bool = True,
        include_existing_memory: bool = True,
        **kwargs: Any,
    ) -> dict[str, str | None]:
        """合并前会 _load_sync；提示中注入整篇 SOUL/USER 正文（过长则尾部摘要）供模型改写成**一份**自洽新正文。"""
        await asyncio.to_thread(self._load_sync)

        t = transcript.strip()
        if not t:
            return {"soul": None, "user": None}
        if truncate and len(t) > self._CONSOLIDATION_MAX_TRANSCRIPT_CHARS:
            t = t[-self._CONSOLIDATION_MAX_TRANSCRIPT_CHARS :]
            t = "[... earlier content truncated ...]\n\n" + t

        api_base = get_env("BASE_URL", warn=False).rstrip("/")
        api_key = get_env("API_KEY", warn=False)
        model_name, w_params = get_model_and_params("worker")
        ex = self._PROMPT_BODY_ELIDE
        existing = ""
        if include_existing_memory:
            s_block = self._elide_for_consolidation_prompt(self._soul_body, ex) or "（尚无）"
            u_block = self._elide_for_consolidation_prompt(self._user_body, ex) or "（尚无）"
            existing = (
                "## 当前 SOUL 整篇（须合并为**一份**自洽、无内部重复与冲突的新正文；无变更时填 null）\n"
                f"{s_block}\n\n"
                "## 当前 USER 整篇（规则同上；无变更时填 null）\n"
                f"{u_block}\n\n"
            )
        body = f"## 待分析的全文\n\n{t}"
        user_content = f"{template}\n\n{existing}{body}"

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": user_content}],
            **chat_completion_inference_request_fields(
                w_params,
                max_tokens=self._CONSOLIDATION_MAX_OUTPUT_TOKENS,
                temperature=0.2,
                **kwargs,
            ),
        }

        async with httpx.AsyncClient(http2=True, timeout=90.0) as client:
            resp = await client.post(
                f"{api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]

        return self._parse_consolidation_response(raw)

    async def consolidate_from_logs(
        self,
        log_root: Path | None = None,
        *,
        silent: bool = True,
    ) -> None:
        """
        后台：扫描 logs 下全部 .log 的增量内容，调用模型提取 SOUL/USER；用 log_sources_state.json 记录已读字节，避免重复。
        log_root 默认使用 logger.LOG_DIR（进程当前工作目录下的 logs）。
        """
        try:
            root = (log_root if log_root is not None else logger.LOG_DIR).resolve()
        except Exception:
            if not silent:
                raise
            logger.warning("长期记忆（日志）：无法解析日志目录。")
            return

        async with self._logs_consolidate_lock:
            try:
                template = await asyncio.to_thread(self._load_consolidation_template_sync)
                if not template.strip():
                    return

                state = await asyncio.to_thread(self._load_log_state_sync)
                sources: dict[str, Any] = state.setdefault("sources", {})

                log_files = sorted(
                    [p for p in root.rglob("*.log") if p.is_file()],
                    key=lambda p: str(p),
                )
                if not log_files:
                    return

                pending_state: dict[str, int] = {}

                for fp in log_files:
                    key = self._rel_log_key(fp, root)
                    try:
                        size = fp.stat().st_size
                    except OSError:
                        continue

                    prev = 0
                    ent = sources.get(key)
                    if isinstance(ent, dict):
                        try:
                            prev = int(ent.get("bytes_read", 0))
                        except (TypeError, ValueError):
                            prev = 0
                    if prev > size:
                        prev = 0
                    if prev >= size:
                        continue

                    pos = prev
                    last_committed = prev
                    try:
                        while pos < size:
                            to_read = min(self._LOG_READ_CHUNK_BYTES, size - pos)
                            with fp.open("rb") as f:
                                f.seek(pos)
                                raw = f.read(to_read)
                            if not raw:
                                last_committed = size
                                break
                            text = raw.decode("utf-8", errors="replace")
                            ci = 0
                            maxc = self._CONSOLIDATION_MAX_TRANSCRIPT_CHARS
                            while ci < len(text):
                                piece = text[ci : ci + maxc]
                                cj = ci + len(piece)
                                header = (
                                    f"=== FILE: {key} | file_bytes {pos}:{pos + len(raw)} / {size} "
                                    f"| chunk_chars {ci}-{cj} ===\n"
                                )
                                parsed = await self._consolidation_chat_and_parse(
                                    template,
                                    header + piece,
                                    truncate=False,
                                )
                                await self._apply_consolidation_parsed(parsed)
                                ci = cj
                            pos += len(raw)
                            last_committed = pos

                        pending_state[key] = last_committed
                        await asyncio.to_thread(self._merge_log_state_updates_sync, dict(pending_state))

                    except Exception:
                        if last_committed > prev:
                            pending_state[key] = last_committed
                            await asyncio.to_thread(self._merge_log_state_updates_sync, dict(pending_state))
                        raise

            except Exception as e:
                if not silent:
                    raise
                logger.warning(f"长期记忆（日志）合并失败（已忽略）: {e}")

    async def consolidate_from_messages(self, messages: list, *, silent: bool = True) -> None:
        """
        根据本轮消息转写，调用 worker 将 SOUL/USER 各**整篇**重写成自洽正文并落盘（见 soul_user_consolidation.md）。
        失败时静默（silent=True）或抛出异常。
        """
        try:
            template = await asyncio.to_thread(self._load_consolidation_template_sync)
            if not template.strip():
                return

            transcript = await asyncio.to_thread(_pydantic_messages_to_text, messages)
            transcript = transcript.strip()
            if not transcript:
                return

            parsed = await self._consolidation_chat_and_parse(template, transcript)
            if await self._apply_consolidation_parsed(parsed):
                logger.info("长期记忆已根据本轮对话更新（SOUL/USER 整篇）。")

        except Exception as e:
            if not silent:
                raise
            logger.warning(f"长期记忆合并失败（已忽略）: {e}")

    async def add(self, target: str, content: str) -> str:
        """
        向对应类别的**整篇**记忆正文末尾追加一段（自动合并为一块连续文本，非分条存储）。
        """
        target = target.lower().strip()
        if target not in self._TARGETS:
            return f"错误：target 须为 'soul' 或 'user'，收到 '{target}'。"

        content = content.strip()
        if not content:
            return "错误：内容不能为空。"

        cap = type(self)._CHAR_LIMIT
        if len(content) > cap:
            return f"错误：单段内容不得超过 {cap} 字符。"

        err = self._scan(content)
        if err:
            return f"安全检查拒绝：{err}"

        async with self._write_lock:
            await asyncio.to_thread(self._load_sync)
            body = (self._get_body(target) or "").strip()
            if content in body:
                return "正文中已包含该片段，未重复添加。"

            new = f"{body}\n\n{content}" if body else content
            new = new.strip()
            if len(new) > cap:
                return (
                    f"追加后总长度 {len(new)} 将超出单文件上限 {cap} 字符，"
                    "请先用 remove 删除一段，或等自动合并重写成更短正文。"
                )
            self._set_body(target, new)
            err2 = self._scan(new)
            if err2:
                return f"安全检查拒绝：{err2}"
            await asyncio.to_thread(self._save_sync, target)

        return f"已追加到 {target} 正文。"

    async def remove(self, target: str, content: str) -> str:
        """
        从正文中删除与 content **完全一致**的首次出现子串（用于手工删一段/一句）。
        """
        target = target.lower().strip()
        if target not in self._TARGETS:
            return f"错误：target 须为 'soul' 或 'user'，收到 '{target}'。"

        content = (content or "").strip()
        if not content:
            return "错误：要删除的片段不能为空。"

        async with self._write_lock:
            await asyncio.to_thread(self._load_sync)
            body = self._get_body(target)
            if content not in body:
                return f"正文中未找到该片段，无法删除：{content[:200]}{'…' if len(content) > 200 else ''}"

            new = body.replace(content, "", 1)
            new = re.sub(r"\n{3,}", "\n\n", new).strip()
            self._set_body(target, new)
            await asyncio.to_thread(self._save_sync, target)

        return f"已删除（{target}）中匹配片段。"

    async def list_memory(self, target: str) -> str:
        """
        返回对应类别的**整篇**正文，或 all 时与 get_injection 一致（含引导文本）。
        """
        target = target.lower().strip()
        if target == "all":
            await asyncio.to_thread(self._load_sync)
            injection = self.get_injection()
            return injection if injection else "记忆为空。"

        if target not in self._TARGETS:
            return f"错误：target 须为 'soul'、'user' 或 'all'，收到 '{target}'。"

        await asyncio.to_thread(self._load_sync)
        body = (self._get_body(target) or "").strip()
        if not body:
            return f"{target} 记忆为空。"
        return body
