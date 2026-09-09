from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Callable, Iterable

import numpy as np

from production.clean_face_mesh import CleanupConfig, clean_face_mesh
from production.export_glb import export_canonical_asset
from production.face_crop_stage import FaceCropConfig, crop_face_mesh
from production.reconstruction_stage import (
    ReconstructionConfig,
    build_reconstruction_commands,
    required_reconstruction_outputs,
    run_reconstruction_stage,
)
from production.runtime import emit_json
from production.state import PipelineState, json_value
from production.texture_stage import TextureConfig, run_texture_stage
from production.validate_asset import validate_asset


PIPELINE_STAGES = (
    "reconstruction",
    "face_crop",
    "clean_geometry",
    "texture",
    "asset_export",
    "validate_publish",
)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_texture_resume_config(config: TextureConfig) -> dict:
    payload = asdict(config)
    code_root = Path(config.code_root)
    payload["source_mesh_sha256"] = file_sha256(config.source_mesh)
    payload["renderer_code_sha256"] = file_sha256(
        code_root / "texture" / "mesh_renderer.py"
    )
    payload["texture_code_sha256"] = file_sha256(
        code_root / "texture" / "build_texture.py"
    )
    payload["unwrap_code_sha256"] = file_sha256(
        code_root / "production" / "unwrap_2dgs_uv.py"
    )
    return payload


def build_face_crop_resume_config(
    config: FaceCropConfig,
    source_mesh: Path,
    transforms_path: Path,
    selected_frames_root: Path,
) -> dict:
    selected_frames = sorted(Path(selected_frames_root).glob("*.png"))
    code_root = Path(__file__).resolve().parents[1]
    return {
        **asdict(config),
        "uv_input": "visible_2dgs_faces",
        "crop_code_sha256": file_sha256(
            code_root / "production" / "face_crop_stage.py"
        ),
        "renderer_code_sha256": file_sha256(
            code_root / "texture" / "mesh_renderer.py"
        ),
        "source_mesh_sha256": file_sha256(Path(source_mesh)),
        "transforms_sha256": file_sha256(Path(transforms_path)),
        "selected_frames": [
            {"name": frame.name, "sha256": file_sha256(frame)}
            for frame in selected_frames
        ],
    }


def build_asset_export_resume_config(
    texture_report: dict,
    matrix: np.ndarray,
) -> dict:
    return {
        "matrix": np.asarray(matrix, dtype=np.float64).tolist(),
        "stem": "face",
        "source_obj_sha256": file_sha256(Path(texture_report["obj"])),
        "source_mtl_sha256": file_sha256(Path(texture_report["mtl"])),
        "texture_sha256": file_sha256(Path(texture_report["texture"])),
    }


def build_validation_resume_config(
    clean_geometry: Path,
    staged_glb: Path,
    atlas_size: int,
    uv_method: str,
) -> dict:
    return {
        "texture_size": [atlas_size, atlas_size],
        "uv_method": uv_method,
        "clean_geometry_sha256": file_sha256(clean_geometry),
        "staged_glb_sha256": file_sha256(staged_glb),
    }


def publish_validated_glb(staged_glb: Path, output: Path, quality: dict) -> dict:
    staged_glb = Path(staged_glb)
    output = Path(output)
    if not staged_glb.is_file() or staged_glb.stat().st_size == 0:
        raise FileNotFoundError(staged_glb)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    shutil.copyfile(staged_glb, temporary)
    os.replace(temporary, output)
    manifest = {
        "status": "complete",
        "output": str(output.resolve()),
        "bytes": output.stat().st_size,
        "sha256": file_sha256(output),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "quality": quality,
    }
    write_json_atomic(output.parent / "result.json", manifest)
    return manifest


def execute_stage(
    stage: str,
    config: object,
    outputs: Iterable[Path],
    report_path: Path,
    action: Callable[[], dict],
    state: PipelineState,
    resume: bool,
) -> dict:
    output_paths = [Path(path) for path in outputs]
    resumable_outputs = [*output_paths, report_path]
    if resume and state.can_resume(stage, config, resumable_outputs):
        emit_json({"stage": stage, "status": "skipped", "reason": "validated_resume"})
        return json.loads(report_path.read_text(encoding="utf-8"))
    state.start(stage, config)
    emit_json({"stage": stage, "status": "started"})
    try:
        report = action()
        write_json_atomic(report_path, report)
        state.complete(stage, config, resumable_outputs, report)
    except Exception as error:
        state.fail(stage, config, str(error))
        emit_json(
            {"stage": stage, "status": "failed", "error": str(error)},
            stream=sys.stderr,
        )
        raise
    emit_json({"stage": stage, "status": "completed"})
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one validated textured GLB face from one video."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--video-step-size", type=int, default=10)
    parser.add_argument("--video-ds-ratio", type=float, default=0.5)
    parser.add_argument("--mesh-res", type=int, default=1024)
    parser.add_argument("--face-oval-scale", type=float, default=1.0)
    parser.add_argument("--face-adjacency-rings", type=int, default=0)
    parser.add_argument("--face-maximum-hole-faces", type=int, default=1000)
    parser.add_argument("--smooth-iterations", type=int, default=3)
    parser.add_argument("--texture-iterations", type=int, default=301)
    parser.add_argument("--lpips-max-size", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    code_root = Path(__file__).resolve().parents[1]
    video = args.video.resolve()
    job_root = args.job_root.resolve()
    output = args.output.resolve()
    workspace = job_root / "workspace"
    artifacts = job_root / "artifacts"
    reconstruction_config = ReconstructionConfig(
        code_root=code_root,
        video_path=video,
        workspace_root=workspace,
        python=args.python,
        ffmpeg=args.ffmpeg,
        mesh_res=args.mesh_res,
        video_step_size=args.video_step_size,
        video_ds_ratio=args.video_ds_ratio,
    )
    cleanup_config = CleanupConfig(
        smooth_iterations=args.smooth_iterations,
    )
    face_crop_config = FaceCropConfig(
        oval_scale=args.face_oval_scale,
        adjacency_rings=args.face_adjacency_rings,
        maximum_hole_faces=args.face_maximum_hole_faces,
    )

    if args.dry_run:
        reconstruction_config.validate(require_input=False)
        for stage in PIPELINE_STAGES:
            details = {}
            if stage == "reconstruction":
                details["commands"] = [
                    {"stage": command.stage, "argv": list(command.argv), "cwd": str(command.cwd)}
                    for command in build_reconstruction_commands(reconstruction_config)
                ]
            emit_json({"stage": stage, "status": "planned", **details})
        return 0

    try:
        if job_root.exists() and any(job_root.iterdir()) and not args.resume:
            raise ValueError("Job root is not empty; pass --resume to reuse it")
        if output.exists() and not args.resume:
            raise ValueError("Output already exists; pass --resume to replace it")
        reconstruction_config.validate(require_input=True)
        if args.physical_gpu < 0:
            raise ValueError("Physical GPU index cannot be negative")
        job_root.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)
        state = PipelineState(job_root / "pipeline-state.json")
        environment = {
            "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
            "PYTHONPATH": str(code_root),
        }
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)

        reconstruction_report_path = artifacts / "reconstruction-report.json"
        execute_stage(
            "reconstruction",
            asdict(reconstruction_config),
            required_reconstruction_outputs(reconstruction_config),
            reconstruction_report_path,
            lambda: run_reconstruction_stage(reconstruction_config, environment),
            state,
            args.resume,
        )

        face_crop_geometry = artifacts / "face_crop.ply"
        face_crop_report_path = artifacts / "face-crop-report.json"
        face_crop_resume_config = build_face_crop_resume_config(
            face_crop_config,
            workspace / "2dgs_recon.obj",
            workspace / "transforms.json",
            workspace / "refinement" / "sample" / "image",
        )
        execute_stage(
            "face_crop",
            face_crop_resume_config,
            (face_crop_geometry,),
            face_crop_report_path,
            lambda: crop_face_mesh(
                source_path=workspace / "2dgs_recon.obj",
                selected_frames_root=workspace / "refinement" / "sample" / "image",
                transforms_path=workspace / "transforms.json",
                output_path=face_crop_geometry,
                config=face_crop_config,
                device="cuda:0",
            ),
            state,
            args.resume,
        )

        clean_geometry = artifacts / "face_geometry.ply"
        clean_report_path = artifacts / "geometry-report.json"
        clean_resume_config = {
            **asdict(cleanup_config),
            "source_mesh_sha256": file_sha256(face_crop_geometry),
        }
        clean_report = execute_stage(
            "clean_geometry",
            clean_resume_config,
            (clean_geometry,),
            clean_report_path,
            lambda: clean_face_mesh(
                face_crop_geometry,
                clean_geometry,
                cleanup_config,
            ),
            state,
            args.resume,
        )

        texture_config = TextureConfig(
            code_root=code_root,
            source_root=workspace,
            source_mesh=clean_geometry,
            output_root=artifacts,
            python=args.python,
            physical_gpu=args.physical_gpu,
            iterations=args.texture_iterations,
            lpips_max_size=args.lpips_max_size,
        )
        texture_report_path = artifacts / "texture-report.json"
        texture_report = execute_stage(
            "texture",
            build_texture_resume_config(texture_config),
            (
                artifacts / "face_uv_source.obj",
                artifacts / "face_uv_source.mtl",
                artifacts / "uv.png",
            ),
            texture_report_path,
            lambda: run_texture_stage(texture_config, environment),
            state,
            args.resume,
        )

        export_root = artifacts / "export"
        export_report_path = export_root / "export-report.json"
        export_config = build_asset_export_resume_config(
            texture_report,
            np.asarray(clean_report["source_to_output_row_matrix"], dtype=np.float64),
        )
        export_report = execute_stage(
            "asset_export",
            export_config,
            (
                export_root / "face.obj",
                export_root / "face.mtl",
                export_root / "uv.png",
                export_root / "face.glb",
                export_root / "face_vertex_colors.ply",
            ),
            export_report_path,
            lambda: export_canonical_asset(
                Path(texture_report["obj"]),
                Path(texture_report["mtl"]),
                Path(texture_report["texture"]),
                np.asarray(clean_report["source_to_output_row_matrix"], dtype=np.float64),
                export_root,
            ),
            state,
            args.resume,
        )

        validation_report_path = artifacts / "validation-report.json"
        validation_config = build_validation_resume_config(
            clean_geometry,
            Path(export_report["glb"]),
            texture_config.atlas_size,
            texture_config.uv_method,
        )

        def validate_and_publish() -> dict:
            quality = validate_asset(
                clean_geometry,
                Path(export_report["obj"]),
                Path(export_report["texture"]),
                Path(export_report["glb"]),
                expected_texture_size=(texture_config.atlas_size,) * 2,
                uv_method=texture_config.uv_method,
                source_to_output_matrix=np.asarray(
                    clean_report["source_to_output_row_matrix"],
                    dtype=np.float64,
                ),
            )
            return publish_validated_glb(Path(export_report["glb"]), output, quality)

        execute_stage(
            "validate_publish",
            validation_config,
            (output, output.parent / "result.json"),
            validation_report_path,
            validate_and_publish,
            state,
            args.resume,
        )
        emit_json(
            {
                "stage": "pipeline",
                "status": "completed",
                "output": str(output),
                "manifest": str(output.parent / "result.json"),
            }
        )
        return 0
    except Exception as error:
        emit_json(
            {"stage": "pipeline", "status": "failed", "error": str(error)},
            stream=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
