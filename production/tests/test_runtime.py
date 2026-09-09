from pathlib import Path
import subprocess
import sys

import pytest

from production.runtime import CommandSpec, run_checked


def test_run_checked_raises_and_emits_failure(tmp_path: Path) -> None:
    events = []
    spec = CommandSpec(
        stage="bad",
        argv=(sys.executable, "-c", "raise SystemExit(7)"),
        cwd=tmp_path,
    )

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_checked(spec, environment=None, emit=events.append)

    assert error.value.returncode == 7
    assert [event["status"] for event in events] == ["started", "failed"]
    assert events[-1]["stage"] == "bad"


def test_run_checked_emits_duration_on_success(tmp_path: Path) -> None:
    events = []
    spec = CommandSpec(
        stage="ok",
        argv=(sys.executable, "-c", "print('ok')"),
        cwd=tmp_path,
    )

    run_checked(spec, environment=None, emit=events.append)

    assert [event["status"] for event in events] == ["started", "completed"]
    assert events[-1]["duration_seconds"] >= 0
