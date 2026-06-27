"""Output routing for legacy console and Textual UI modes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import sys
from typing import Any, Protocol

from rich.console import Console
from rich.text import Text


@dataclass(frozen=True)
class ContextUsageItem:
    role_label: str
    used_tokens: int
    max_tokens: int
    percent: float


class OutputSink(Protocol):
    @property
    def supports_model_stream(self) -> bool:
        ...

    def emit(self, renderable: Any) -> None:
        ...

    def rule(self, title: str) -> None:
        ...

    def set_status(self, message: str) -> None:
        ...

    def clear_status(self) -> None:
        ...

    def set_context_usage(self, items: list[ContextUsageItem]) -> None:
        ...

    def clear_context_usage(self) -> None:
        ...

    def begin_model_stream(self, title: str) -> None:
        ...

    def append_model_stream_delta(self, text: str) -> None:
        ...

    def clear_model_stream(self) -> None:
        ...


class LegacyOutputSink:
    def __init__(self, console: Console) -> None:
        self.console = console

    @property
    def supports_model_stream(self) -> bool:
        return False

    def emit(self, renderable: Any) -> None:
        self.console.print(renderable)

    def rule(self, title: str) -> None:
        self.console.rule(title)

    def set_status(self, message: str) -> None:
        self.console.print(Text(message, style="dim"))

    def clear_status(self) -> None:
        return

    def set_context_usage(self, items: list[ContextUsageItem]) -> None:
        return

    def clear_context_usage(self) -> None:
        return

    def begin_model_stream(self, title: str) -> None:
        return

    def append_model_stream_delta(self, text: str) -> None:
        return

    def clear_model_stream(self) -> None:
        return


_console = Console(highlight=False, legacy_windows=sys.platform == "win32")
_sink: OutputSink = LegacyOutputSink(_console)


def set_output_sink(sink: OutputSink | None) -> None:
    global _sink
    _sink = sink if sink is not None else LegacyOutputSink(_console)


def supports_model_stream() -> bool:
    value = getattr(_sink, "supports_model_stream", False)
    if callable(value):
        return bool(value())
    return bool(value)


def emit_renderable(renderable: Any) -> None:
    _sink.emit(renderable)


def emit_rule(title: str) -> None:
    _sink.rule(title)


def set_status(message: str) -> None:
    _sink.set_status(message)


def clear_status() -> None:
    _sink.clear_status()


def set_context_usage(items: list[ContextUsageItem]) -> None:
    _sink.set_context_usage(items)


def clear_context_usage() -> None:
    _sink.clear_context_usage()


def begin_model_stream(title: str) -> None:
    _sink.begin_model_stream(title)


def append_model_stream_delta(text: str) -> None:
    _sink.append_model_stream_delta(text)


def clear_model_stream() -> None:
    _sink.clear_model_stream()


@contextmanager
def status_message(message: str):
    set_status(message)
    try:
        yield
    finally:
        clear_status()
