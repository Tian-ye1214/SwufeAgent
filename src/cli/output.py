"""Output routing for legacy console and Textual UI modes."""

from __future__ import annotations

from contextlib import contextmanager
import sys
from typing import Any, Protocol

from rich.console import Console
from rich.text import Text


class OutputSink(Protocol):
    def emit(self, renderable: Any) -> None:
        ...

    def rule(self, title: str) -> None:
        ...

    def set_status(self, message: str) -> None:
        ...

    def clear_status(self) -> None:
        ...


class LegacyOutputSink:
    def __init__(self, console: Console) -> None:
        self.console = console

    def emit(self, renderable: Any) -> None:
        self.console.print(renderable)

    def rule(self, title: str) -> None:
        self.console.rule(title)

    def set_status(self, message: str) -> None:
        self.console.print(Text(message, style="dim"))

    def clear_status(self) -> None:
        return


_console = Console(highlight=False, legacy_windows=sys.platform == "win32")
_sink: OutputSink = LegacyOutputSink(_console)


def set_output_sink(sink: OutputSink | None) -> None:
    global _sink
    _sink = sink if sink is not None else LegacyOutputSink(_console)


def current_output_sink() -> OutputSink:
    return _sink


def emit_renderable(renderable: Any) -> None:
    _sink.emit(renderable)


def emit_text(text: str) -> None:
    _sink.emit(text)


def emit_rule(title: str) -> None:
    _sink.rule(title)


def set_status(message: str) -> None:
    _sink.set_status(message)


def clear_status() -> None:
    _sink.clear_status()


@contextmanager
def status_message(message: str):
    set_status(message)
    try:
        yield
    finally:
        clear_status()
