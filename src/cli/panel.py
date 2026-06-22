from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import logger
from ModelGateway.usage_accounting import (
    MODEL_MESSAGES_GLOB,
    UsageTotals,
    latest_usage_input_tokens,
    read_model_messages_file,
    summarize_messages,
)

Reader = Callable[[Path], tuple[list[Any], dict[str, Any]]]
RECENT_SESSION_LIMIT = 20


@dataclass
class TokenBucket:
    response_count: int = 0
    missing_usage_responses: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def add_totals(self, totals: UsageTotals) -> None:
        self.response_count += totals.responses
        self.missing_usage_responses += totals.missing_usage_responses
        self.input_tokens += totals.input_tokens
        self.output_tokens += totals.output_tokens
        self.reasoning_tokens += totals.reasoning_tokens

    def add_bucket(self, other: "TokenBucket") -> None:
        self.response_count += other.response_count
        self.missing_usage_responses += other.missing_usage_responses
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_tokens += other.reasoning_tokens


@dataclass
class PanelFileSummary:
    path: Path
    agent: str
    date: str
    topic: str
    saved_at: str
    totals: UsageTotals
    by_model: dict[str, TokenBucket]


@dataclass
class PanelFileResult:
    summary: PanelFileSummary | None = None
    error: str | None = None


@dataclass
class PanelSessionSummary:
    date: str
    topic: str
    agents: set[str] = field(default_factory=set)
    file_count: int = 0
    response_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    saved_at: str = ""

    def add_file(self, file_summary: PanelFileSummary) -> None:
        self.agents.add(file_summary.agent)
        self.file_count += 1
        self.response_count += file_summary.totals.responses
        self.input_tokens += file_summary.totals.input_tokens
        self.output_tokens += file_summary.totals.output_tokens
        self.reasoning_tokens += file_summary.totals.reasoning_tokens
        if file_summary.saved_at > self.saved_at:
            self.saved_at = file_summary.saved_at


@dataclass
class PanelHistoryStats:
    file_count: int = 0
    conversation_count: int = 0
    response_count: int = 0
    missing_usage_responses: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    by_agent: dict[str, TokenBucket] = field(default_factory=dict)
    by_model: dict[str, TokenBucket] = field(default_factory=dict)
    skipped_count: int = 0
    skipped_files: list[str] = field(default_factory=list)


@dataclass
class TaskPanelStats:
    total: int = 0
    completed: int = 0
    running: int = 0
    failed: int = 0
    pending: int = 0


@dataclass
class RuntimePanelStats:
    session_key: str = "-"
    active_invocations: int = 0
    context_input_tokens: dict[str, int] = field(default_factory=dict)
    tasks: TaskPanelStats = field(default_factory=TaskPanelStats)


@dataclass
class PanelSnapshot:
    runtime: RuntimePanelStats
    history: PanelHistoryStats
    visible_sessions: list[PanelSessionSummary]
    include_all: bool = False


@dataclass
class _CacheEntry:
    signature: tuple[int, int]
    result: PanelFileResult


class PanelSnapshotCache:
    def __init__(self, *, reader: Reader | None = None) -> None:
        self._reader = reader or read_model_messages_file
        self._cache: dict[Path, _CacheEntry] = {}

    def load(self, path: Path) -> PanelFileResult:
        path = Path(path)
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError as e:
            return PanelFileResult(error=f"{path}: {type(e).__name__}: {e}")
        cached = self._cache.get(path)
        if cached is not None and cached.signature == signature:
            return cached.result
        result = self._parse(path)
        self._cache[path] = _CacheEntry(signature=signature, result=result)
        return result

    def _parse(self, path: Path) -> PanelFileResult:
        try:
            messages, meta = self._reader(path)
            saved_at = _read_saved_at(path)
            usage_summary = summarize_messages(
                messages,
                meta=meta,
                path=path,
                price_resolver=lambda _model: None,
            )
            by_model = {
                model_name: _bucket_from_totals(model_summary.totals)
                for model_name, model_summary in usage_summary.by_model.items()
            }
            agent, date, topic = _file_identity(path, meta)
            return PanelFileResult(
                summary=PanelFileSummary(
                    path=path,
                    agent=agent,
                    date=date,
                    topic=topic,
                    saved_at=saved_at,
                    totals=usage_summary.totals,
                    by_model=by_model,
                )
            )
        except Exception as e:
            return PanelFileResult(error=f"{path}: {type(e).__name__}: {e}")


async def build_panel_snapshot(
    *,
    log_root: Path | None = None,
    system: Any = None,
    coordinator_history: Any = None,
    manager_history: Any = None,
    include_all: bool = False,
    cache: PanelSnapshotCache | None = None,
) -> PanelSnapshot:
    root = Path(log_root or logger.LOG_DIR)
    history, sessions = _collect_history(root, cache or PanelSnapshotCache())
    visible_sessions = sessions if include_all else sessions[:RECENT_SESSION_LIMIT]
    runtime = await _collect_runtime(system, coordinator_history, manager_history)
    return PanelSnapshot(
        runtime=runtime,
        history=history,
        visible_sessions=visible_sessions,
        include_all=include_all,
    )


def render_panel(snapshot: PanelSnapshot) -> Panel:
    history = snapshot.history
    runtime = snapshot.runtime
    # TODO(panel): 当前终端条形图视觉效果较弱且信息价值有限；后续应重设计为更有意义的趋势、占比、异常提示展示。
    parts = [
        _render_kpis(snapshot),
        _render_runtime(runtime),
        _render_token_chart(history),
        _render_task_chart(runtime.tasks),
        _render_distribution(history),
        _render_sessions(snapshot.visible_sessions, include_all=snapshot.include_all),
    ]
    if history.skipped_count:
        parts.append(_render_skipped(history))
    return Panel(Group(*parts), title="RedLotus Panel", border_style="cyan")


def _collect_history(
    log_root: Path,
    cache: PanelSnapshotCache,
) -> tuple[PanelHistoryStats, list[PanelSessionSummary]]:
    history = PanelHistoryStats()
    sessions: dict[tuple[str, str], PanelSessionSummary] = {}
    files = sorted((log_root / "conversations").rglob(MODEL_MESSAGES_GLOB), key=str)
    for path in files:
        result = cache.load(path)
        if result.error:
            history.skipped_count += 1
            if len(history.skipped_files) < 5:
                history.skipped_files.append(result.error)
            continue
        if result.summary is None:
            continue
        summary = result.summary
        history.file_count += 1
        history.response_count += summary.totals.responses
        history.missing_usage_responses += summary.totals.missing_usage_responses
        history.input_tokens += summary.totals.input_tokens
        history.output_tokens += summary.totals.output_tokens
        history.reasoning_tokens += summary.totals.reasoning_tokens

        agent_bucket = history.by_agent.setdefault(summary.agent, TokenBucket())
        agent_bucket.add_totals(summary.totals)
        for model_name, model_bucket in summary.by_model.items():
            history.by_model.setdefault(model_name, TokenBucket()).add_bucket(model_bucket)

        key = (summary.date, summary.topic)
        session = sessions.setdefault(
            key,
            PanelSessionSummary(date=summary.date, topic=summary.topic),
        )
        session.add_file(summary)

    ordered_sessions = sorted(sessions.values(), key=lambda s: (s.saved_at, s.date, s.topic), reverse=True)
    history.conversation_count = len(ordered_sessions)
    return history, ordered_sessions


async def _collect_runtime(
    system: Any,
    coordinator_history: Any,
    manager_history: Any,
) -> RuntimePanelStats:
    runtime = RuntimePanelStats()
    runtime.session_key = str(getattr(system, "session_key", "") or "-")
    runtime.context_input_tokens = {
        "Coordinator": latest_usage_input_tokens(getattr(coordinator_history, "messages", []) or []) or 0,
        "Manager": latest_usage_input_tokens(getattr(manager_history, "messages", []) or []) or 0,
    }
    runtime.tasks = _collect_task_stats(getattr(system, "_task_manager", None))
    registry = getattr(system, "registry", None)
    list_active = getattr(registry, "list_active_invocations", None)
    if callable(list_active):
        try:
            active = await list_active(getattr(system, "session_key", None))
            runtime.active_invocations = len(active)
        except Exception:
            runtime.active_invocations = 0
    return runtime


def _collect_task_stats(task_manager: Any) -> TaskPanelStats:
    tasks = getattr(task_manager, "tasks", {}) or {}
    stats = TaskPanelStats(total=len(tasks))
    for task in tasks.values():
        value = getattr(getattr(task, "status", None), "value", getattr(task, "status", ""))
        if value == "completed":
            stats.completed += 1
        elif value == "running":
            stats.running += 1
        elif value == "failed":
            stats.failed += 1
        else:
            stats.pending += 1
    return stats


def _read_saved_at(path: Path) -> str:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    return str(data.get("saved_at") or "")


def _file_identity(path: Path, meta: dict[str, Any]) -> tuple[str, str, str]:
    agent = str(meta.get("agent") or "")
    date = str(meta.get("date") or "")
    topic = str(meta.get("topic") or "")
    parts = path.parts
    if not agent or not date or not topic:
        try:
            idx = parts.index("conversations")
            agent = agent or parts[idx + 1]
            date = date or parts[idx + 2]
            topic = topic or parts[idx + 3]
        except Exception:
            pass
    return agent or "unknown", date or "unknown", topic or "unknown"


def _bucket_from_totals(totals: UsageTotals) -> TokenBucket:
    bucket = TokenBucket()
    bucket.add_totals(totals)
    return bucket


def _render_kpis(snapshot: PanelSnapshot) -> Table:
    history = snapshot.history
    runtime = snapshot.runtime
    table = Table.grid(expand=True)
    table.add_column(justify="left")
    table.add_column(justify="left")
    table.add_column(justify="left")
    table.add_column(justify="left")
    table.add_row(
        f"历史对话 {history.conversation_count}",
        f"model_messages {history.file_count}",
        f"responses {history.response_count}",
        f"active {runtime.active_invocations}",
    )
    table.add_row(
        f"session {runtime.session_key}",
        f"missing usage {history.missing_usage_responses}",
        f"skipped {history.skipped_count}",
        f"tasks {runtime.tasks.completed}/{runtime.tasks.total}",
    )
    return table


def _render_runtime(runtime: RuntimePanelStats) -> Table:
    table = Table(title="当前运行态", expand=True)
    table.add_column("项目")
    table.add_column("值", justify="right")
    table.add_row("Session", runtime.session_key)
    table.add_row("Active invocations", str(runtime.active_invocations))
    for label, tokens in runtime.context_input_tokens.items():
        table.add_row(f"{label} context input", _fmt_int(tokens))
    return table


def _render_token_chart(history: Any) -> Table:
    table = Table(title="Token 用量", expand=True)
    table.add_column("类型")
    table.add_column("数量", justify="right")
    table.add_column("图表")
    values = [
        ("Input", int(getattr(history, "input_tokens", 0) or 0)),
        ("Output", int(getattr(history, "output_tokens", 0) or 0)),
        ("Reasoning", int(getattr(history, "reasoning_tokens", 0) or 0)),
    ]
    max_value = max([v for _, v in values] + [1])
    for label, value in values:
        table.add_row(label, _fmt_int(value), _bar(value, max_value))
    return table


def _render_task_chart(tasks: Any) -> Table:
    total = int(getattr(tasks, "total", 0) or 0)
    completed = int(getattr(tasks, "completed", 0) or 0)
    running = int(getattr(tasks, "running", 0) or 0)
    failed = int(getattr(tasks, "failed", 0) or 0)
    pending = int(getattr(tasks, "pending", 0) or 0)
    table = Table(title="任务", expand=True)
    table.add_column("状态")
    table.add_column("数量", justify="right")
    table.add_column("进度")
    table.add_row("Completed", f"{completed}/{total}", _bar(completed, total or 1))
    table.add_row("Running", str(running), _bar(running, total or 1))
    table.add_row("Failed", str(failed), _bar(failed, total or 1))
    table.add_row("Pending", str(pending), _bar(pending, total or 1))
    return table


def _render_distribution(history: PanelHistoryStats) -> Table:
    table = Table(title="分布", expand=True)
    table.add_column("类型")
    table.add_column("名称")
    table.add_column("Responses", justify="right")
    table.add_column("Tokens", justify="right")
    rows: list[tuple[str, str, TokenBucket]] = []
    rows.extend(("Agent", name, bucket) for name, bucket in history.by_agent.items())
    rows.extend(("Model", name, bucket) for name, bucket in history.by_model.items())
    rows.sort(
        key=lambda row: (
            row[2].input_tokens + row[2].output_tokens + row[2].reasoning_tokens,
            row[2].response_count,
            row[1],
        ),
        reverse=True,
    )
    if not rows:
        table.add_row("-", "暂无分布数据", "0", "0")
        return table
    for kind, name, bucket in rows[:12]:
        tokens = bucket.input_tokens + bucket.output_tokens + bucket.reasoning_tokens
        table.add_row(kind, name, str(bucket.response_count), _fmt_int(tokens))
    return table


def _render_sessions(sessions: list[PanelSessionSummary], *, include_all: bool) -> Table:
    title = "历史对话（全部）" if include_all else f"历史对话（最近 {RECENT_SESSION_LIMIT}）"
    table = Table(title=title, expand=True)
    table.add_column("时间")
    table.add_column("Topic")
    table.add_column("Agents")
    table.add_column("Responses", justify="right")
    table.add_column("Tokens", justify="right")
    if not sessions:
        table.add_row("-", "暂无历史对话", "-", "0", "0")
        return table
    for session in sessions:
        agents = ",".join(sorted(session.agents))
        tokens = session.input_tokens + session.output_tokens
        table.add_row(
            session.saved_at or session.date,
            session.topic,
            agents,
            str(session.response_count),
            _fmt_int(tokens),
        )
    return table


def _render_skipped(history: PanelHistoryStats) -> Text:
    text = Text()
    text.append(f"Skipped corrupt model_messages: {history.skipped_count}\n", style="yellow")
    for item in history.skipped_files:
        text.append(f"- {item}\n", style="dim yellow")
    return text


def _bar(value: int, max_value: int, *, width: int = 20) -> str:
    if max_value <= 0:
        filled = 0
    else:
        filled = round(width * max(0, value) / max_value)
    filled = max(0, min(width, filled))
    return "[" + "=" * filled + "." * (width - filled) + "]"


def _fmt_int(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)
