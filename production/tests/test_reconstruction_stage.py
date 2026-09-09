from pathlib import Path

from production.reconstruction_stage import (
    ReconstructionConfig,
    build_reconstruction_commands,
)


def command_by_stage(commands, stage: str):
    return next(command for command in commands if command.stage == stage)


def test_reconstruction_uses_full_checkpoint_and_one_cluster(tmp_path: Path) -> None:
    config = ReconstructionConfig(
        code_root=tmp_path / "code",
        video_path=tmp_path / "capture.mov",
        workspace_root=tmp_path / "workspace",
        python="python",
        iterations=30000,
        mesh_res=1024,
    )

    commands = build_reconstruction_commands(config)
    train = command_by_stage(commands, "train_2dgs")
    render = command_by_stage(commands, "extract_mesh")

    assert "--iterations" in train.argv
    assert train.argv[train.argv.index("--iterations") + 1] == "30000"
    assert "--iteration" in render.argv
    assert render.argv[render.argv.index("--iteration") + 1] == "30000"
    assert "--mesh_res" in render.argv
    assert render.argv[render.argv.index("--mesh_res") + 1] == "1024"
    assert "--num_cluster" in render.argv
    assert render.argv[render.argv.index("--num_cluster") + 1] == "1"


def test_reconstruction_command_order_is_worker_stable(tmp_path: Path) -> None:
    config = ReconstructionConfig(
        code_root=tmp_path / "code",
        video_path=tmp_path / "capture.mov",
        workspace_root=tmp_path / "workspace",
    )

    assert [command.stage for command in build_reconstruction_commands(config)] == [
        "extract_frames",
        "matting",
        "prepare_2dgs",
        "colmap",
        "train_2dgs",
        "extract_mesh",
        "convert_mesh",
        "refinement",
        "registration",
    ]
