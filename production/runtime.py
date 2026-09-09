from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Callable, Mapping, TextIO


Event = dict[str, object]
Emitter = Callable[[Event], None]


@dataclass(frozen=True)
class CommandSpec:
    stage: str
    argv: tuple[str, ...]
    cwd: Path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit_json(event: Event, stream: TextIO = sys.stdout) -> None:
    payload = {"timestamp": utc_timestamp(), **event}
    print(json.dumps(payload, sort_keys=True), file=stream, flush=True)


def run_checked(
    spec: CommandSpec,
    environment: Mapping[str, str] | None,
    emit: Emitter = emit_json,
) -> None:
    argv = tuple(str(value) for value in spec.argv)
    event_base: Event = {
        "stage": spec.stage,
        "command": list(argv),
        "cwd": str(spec.cwd),
    }
    emit({**event_base, "status": "started", "timestamp": utc_timestamp()})
    started = time.monotonic()
    process = subprocess.Popen(
        argv,
        cwd=spec.cwd,
        env=dict(environment) if environment is not None else os.environ.copy(),
    )
    previous_handlers: dict[signal.Signals, object] = {}

    def forward_signal(signum, _frame) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, forward_signal)
            except ValueError:
                previous_handlers.clear()
                break
        return_code = process.wait()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    duration = round(time.monotonic() - started, 3)
    if return_code != 0:
        emit(
            {
                **event_base,
                "status": "failed",
                "timestamp": utc_timestamp(),
                "duration_seconds": duration,
                "return_code": return_code,
            }
        )
        raise subprocess.CalledProcessError(return_code, argv)
    emit(
        {
            **event_base,
            "status": "completed",
            "timestamp": utc_timestamp(),
            "duration_seconds": duration,
        }
    )
