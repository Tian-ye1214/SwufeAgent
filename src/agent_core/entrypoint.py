from __future__ import annotations

import asyncio
import signal

from agent_core.system import AgentSystem


def install_stop_handlers(stop_event: asyncio.Event) -> None:
    """Map process signals to the interactive runner's stop event."""
    loop = asyncio.get_running_loop()

    def request_stop(*_args: object) -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except (NotImplementedError, ValueError):
            signal.signal(sig, request_stop)


async def run_cli() -> None:
    """Run the interactive RedLotus CLI/TUI."""
    system = AgentSystem()
    stop_event = asyncio.Event()
    install_stop_handlers(stop_event)
    try:
        await system.run_interactive(stop_event=stop_event)
    finally:
        await system.shutdown()


def main() -> None:
    """CLI entrypoint used by root ``main.py``."""
    asyncio.run(run_cli())
