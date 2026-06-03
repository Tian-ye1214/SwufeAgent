from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx
import json_repair
import logger
from persist_utils import atomic_write_json, atomic_write_text, file_lock
from app_config import (
    chat_completion_inference_request_fields,
    get_env,
    get_model_and_params,
)
from tools.memory.consolidation import (
    long_term_memory_consolidation_params,
    merge_looks_like_unrelated_rewrite,
)
from tools.memory.message_text import user_prompts_to_text


class LongTermMemory:
    """跨会话长期记忆，持久化到 src/prompts/LongTermMemory/ 目录。

    - MEMORY_GUIDANCE.md：给模型的长期记忆**使用**说明（随 load 注入 get_injection，不是抽取提示词）
    - soul_user_consolidation.md：从对话或日志增量中**合并**出 SOUL/USER 整篇正文的**唯一**提示词
    - SOUL.md / USER.md：各一份 **Markdown** 文件（首行 `# SOUL` / `# USER`，正文为整体叙述）；无标题时整篇当正文；`\\n---\\n` 分隔的段落会合并
    - log_sources_state.json：各 .log 已读字节偏移，避免重复喂给模型；**不存在时首次加载会自动创建**（version + 空 sources）
    """
    _CHAR_LIMIT = 8000
    _PROMPT_BODY_ELIDE = 10_000
    _INJECTION_PATTERN = re.compile(
        r'(?i)(ignore\s+previous|disregard\s+instruction|system\s+prompt|jailbreak|prompt\s+injection)',
    )

    _MEMORY_DIR = Path(__file__).resolve().parent.parent.parent / "prompts" / "LongTermMemory"
    _GUIDANCE_FILE = "MEMORY_GUIDANCE.md"
    _CONSOLIDATION_TEMPLATE_FILE = "soul_user_consolidation.md"
    _LOG_STATE_FILE = "log_sources_state.json"
    _TARGETS = {"soul": "SOUL.md", "user": "USER.md"}
    _MD_H1_SOUL = re.compile(r"^#\s*SOUL\s*$", re.IGNORECASE)
    _MD_H1_USER = re.compile(r"^#\s*USER\s*$", re.IGNORECASE)
    _GLOBAL_WRITE_LOCK = asyncio.Lock()

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
        with file_lock(path):
            atomic_write_text(path, self._format_markdown_file(target, body))

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
        """输出整篇正文的 JSON：soul / user 为字符串或 null。"""
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
            else:
                out[key] = None
        return out

    async def _apply_consolidation_parsed(self, parsed: dict[str, str | None]) -> bool:
        """以模型给出的整篇正文**替换**对应 SOUL/USER；无更新则返回 False。"""
        changed = False
        async with type(self)._GLOBAL_WRITE_LOCK:
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
                old = (self._get_body(key) or "").strip()
                if old and merge_looks_like_unrelated_rewrite(old, text):
                    logger.warning("长期记忆合并已跳过 %s：新正文与旧正文差异过大（疑似误抽取）", key)
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
        p = self._log_state_path()
        with file_lock(p):
            atomic_write_json(p, state)

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
        ltm = long_term_memory_consolidation_params()
        max_tc = int(ltm["max_transcript_chars"])
        if truncate and len(t) > max_tc:
            t = t[-max_tc:]
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
                max_tokens=int(ltm["max_output_tokens"]),
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
                            ltm = long_term_memory_consolidation_params()
                            log_chunk_b = int(ltm["log_read_chunk_bytes"])
                            max_tc = int(ltm["max_transcript_chars"])
                            to_read = min(log_chunk_b, size - pos)
                            with fp.open("rb") as f:
                                f.seek(pos)
                                raw = f.read(to_read)
                            if not raw:
                                last_committed = size
                                break
                            text = raw.decode("utf-8", errors="replace")
                            ci = 0
                            maxc = max_tc
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

            transcript = await asyncio.to_thread(user_prompts_to_text, messages)
            transcript = transcript.strip()
            if not transcript:
                return

            parsed = await self._consolidation_chat_and_parse(template, transcript)
            if await self._apply_consolidation_parsed(parsed):
                logger.warning("📝 长期记忆已根据你的输入更新（SOUL/USER）。")

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

        async with type(self)._GLOBAL_WRITE_LOCK:
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

        async with type(self)._GLOBAL_WRITE_LOCK:
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
