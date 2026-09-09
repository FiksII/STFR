from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

from production.runtime import CommandSpec, Emitter, emit_json, run_checked


PRODUCTION_ITERATIONS = 30000


@dataclass(frozen=True)
class ReconstructionConfig:
    code_root: Path
    video_path: Path
    workspace_root: Path
    python: str = "python"
    ffmpeg: str = "ffmpeg"
    iterations: int = PRODUCTION_ITERATIONS
    mesh_res: int = 1024
    video_step_size: int = 10
    video_ds_ratio: float = 0.5
    close_eye: int = 0

    def validate(self, require_input: bool = True) -> None:
        if self.iterations != PRODUCTION_ITERATIONS:
            raise ValueError("Production STFR requires exactly 30000 2DGS iterations")
        if self.mesh_res < 256:
            raise ValueError("Mesh resolution must be at least 256")
        if self.video_step_size < 1:
            raise ValueError("Video step size must be positive")
        if not 0 < self.video_ds_ratio <= 1:
            raise ValueError("Video downscale ratio must be in (0, 1]")
        if require_input and not Path(self.video_path).is_file():
            raise FileNotFoundError(self.video_path)


def build_reconstruction_commands(config: ReconstructionConfig) -> list[CommandSpec]:
    config.validate(require_input=False)
    code_root = Path(config.code_root).resolve()
    workspace = Path(config.workspace_root).resolve()
    raw_frames = workspace / "raw_frames"
    masks = workspace / "mask"
    reconstruction = code_root / "reconstruction"
    gaussian = reconstruction / "2d-gaussian-splatting"
    recon_output = workspace / "recon"
    frame_filter = (
        f"select=not(mod(n\\,{config.video_step_size})),"
        f"scale=iw*{config.video_ds_ratio:g}:ih*{config.video_ds_ratio:g},setsar=1:1"
    )
    return [
        CommandSpec(
            "extract_frames",
            (
                config.ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(Path(config.video_path).resolve()),
                "-vf",
                frame_filter,
                "-fps_mode",
                "vfr",
                "-q:v",
                "1",
                str(raw_frames / "%05d.png"),
            ),
            code_root,
        ),
        CommandSpec(
            "matting",
            (
                config.python,
                str(code_root / "matting" / "run_matting.py"),
                "--input_root",
                str(raw_frames),
                "--output_root",
                str(masks),
            ),
            code_root / "matting",
        ),
        CommandSpec(
            "prepare_2dgs",
            (
                config.python,
                str(reconstruction / "to_2dgs_format.py"),
                "--data_root",
                str(workspace),
            ),
            reconstruction,
        ),
        CommandSpec(
            "colmap",
            (
                config.python,
                str(reconstruction / "run_colmap.py"),
                "--data_root",
                str(workspace),
            ),
            reconstruction,
        ),
        CommandSpec(
            "train_2dgs",
            (
                config.python,
                str(gaussian / "train.py"),
                "-s",
                str(workspace),
                "-m",
                str(recon_output),
                "--iterations",
                str(config.iterations),
                "--save_iterations",
                str(config.iterations),
                "--test_iterations",
                str(config.iterations),
            ),
            gaussian,
        ),
        CommandSpec(
            "extract_mesh",
            (
                config.python,
                str(gaussian / "render.py"),
                "-s",
                str(workspace),
                "-m",
                str(recon_output),
                "--iteration",
                str(config.iterations),
                "--mesh_res",
                str(config.mesh_res),
                "--num_cluster",
                "1",
                "--skip_test",
            ),
            gaussian,
        ),
        CommandSpec(
            "convert_mesh",
            (
                config.python,
                str(reconstruction / "to_my_format.py"),
                "--data_root",
                str(workspace),
            ),
            reconstruction,
        ),
        CommandSpec(
            "refinement",
            (
                config.python,
                str(code_root / "refinement" / "run_refinement.py"),
                "--data_root",
                str(workspace),
            ),
            code_root / "refinement",
        ),
        CommandSpec(
            "registration",
            (
                config.python,
                str(code_root / "registration" / "run_registration.py"),
                "--data_root",
                str(workspace),
                "--close_eye",
                str(config.close_eye),
            ),
            code_root / "registration",
        ),
    ]


def required_reconstruction_outputs(config: ReconstructionConfig) -> list[Path]:
    workspace = Path(config.workspace_root)
    return [
        workspace
        / "recon"
        / "point_cloud"
        / f"iteration_{config.iterations}"
        / "point_cloud.ply",
        workspace / "2dgs_recon.obj",
        workspace / "transforms.json",
        workspace / "register" / "fine_align" / "align_canonical.obj",
        workspace / "register" / "wrap" / "final_hack.obj",
    ]


def validate_reconstruction_outputs(config: ReconstructionConfig) -> dict:
    missing = [path for path in required_reconstruction_outputs(config) if not path.is_file()]
    selected = sorted((config.workspace_root / "refinement" / "sample" / "image").glob("*.png"))
    if missing:
        raise FileNotFoundError("Missing reconstruction outputs: " + ", ".join(map(str, missing)))
    if not selected:
        raise FileNotFoundError("Refinement did not select any texture frames")
    return {
        "checkpoint_iteration": config.iterations,
        "mesh_res": config.mesh_res,
        "selected_texture_frames": len(selected),
        "mesh": str((config.workspace_root / "2dgs_recon.obj").resolve()),
    }


def run_reconstruction_stage(
    config: ReconstructionConfig,
    environment: Mapping[str, str] | None = None,
    emit: Emitter = emit_json,
) -> dict:
    config.validate(require_input=True)
    config.workspace_root.mkdir(parents=True, exist_ok=True)
    (config.workspace_root / "raw_frames").mkdir(parents=True, exist_ok=True)
    (config.workspace_root / "mask").mkdir(parents=True, exist_ok=True)
    child_environment = os.environ.copy()
    if environment is not None:
        child_environment.update(environment)
    for command in build_reconstruction_commands(config):
        run_checked(command, child_environment, emit)
    return validate_reconstruction_outputs(config)
