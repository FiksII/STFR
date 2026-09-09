from pathlib import Path
import subprocess

import pytest

from reconstruction.run_colmap import build_colmap_commands, run_colmap


def command_for(commands: list[tuple[str, ...]], name: str) -> tuple[str, ...]:
    return next(command for command in commands if command[1] == name)


def test_colmap_commands_use_312_sift_option_names(tmp_path: Path) -> None:
    commands = build_colmap_commands(tmp_path / "capture with spaces")

    feature = command_for(commands, "feature_extractor")
    matching = command_for(commands, "exhaustive_matcher")

    assert "--SiftExtraction.use_gpu" in feature
    assert "--FeatureExtraction.use_gpu" not in feature
    assert "--SiftMatching.use_gpu" in matching
    assert "--SiftMatching.guided_matching" in matching
    assert "--FeatureMatching.use_gpu" not in matching
    assert "--FeatureMatching.guided_matching" not in matching
    assert str(tmp_path / "capture with spaces" / "images") in feature


def test_colmap_runner_propagates_the_first_failed_command(tmp_path: Path) -> None:
    calls: list[tuple[tuple[str, ...], bool]] = []

    def failing_runner(command: tuple[str, ...], check: bool) -> None:
        calls.append((command, check))
        raise subprocess.CalledProcessError(2, command)

    with pytest.raises(subprocess.CalledProcessError):
        run_colmap(tmp_path, runner=failing_runner)

    assert len(calls) == 1
    assert calls[0][0][1] == "feature_extractor"
    assert calls[0][1] is True
