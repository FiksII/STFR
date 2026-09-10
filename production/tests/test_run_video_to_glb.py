import json
from pathlib import Path

import numpy as np

import production.run_video_to_glb as pipeline
from production.run_video_to_glb import (
    PIPELINE_STAGES,
    build_clean_resume_config,
    build_head_crop_resume_config,
    build_texture_resume_config,
    main,
    parse_args,
    publish_validated_glb,
)
from production.head_crop_stage import HeadCropConfig
from production.clean_face_mesh import CleanupConfig
from production.texture_stage import TextureConfig


def test_pipeline_crops_observed_head_from_direct_2dgs_mesh() -> None:
    assert PIPELINE_STAGES == (
        "reconstruction",
        "head_crop",
        "clean_geometry",
        "texture",
        "asset_export",
        "validate_publish",
    )


def test_dry_run_lists_all_stages_without_touching_job_root(
    tmp_path: Path, capsys
) -> None:
    video = tmp_path / "input.mov"
    job_root = tmp_path / "job"
    output = job_root / "result" / "face.glb"

    result = main(
        [
            "--video",
            str(video),
            "--job-root",
            str(job_root),
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert result == 0
    assert [event["stage"] for event in events if event["status"] == "planned"] == list(
        PIPELINE_STAGES
    )
    assert not job_root.exists()


def test_nonempty_job_root_requires_resume(tmp_path: Path, capsys) -> None:
    job_root = tmp_path / "job"
    job_root.mkdir()
    (job_root / "foreign.txt").write_text("occupied", encoding="utf-8")

    result = main(
        [
            "--video",
            str(tmp_path / "input.mov"),
            "--job-root",
            str(job_root),
            "--output",
            str(job_root / "result" / "face.glb"),
        ]
    )

    error = json.loads(capsys.readouterr().err)
    assert result == 2
    assert error["status"] == "failed"
    assert "--resume" in error["error"]


def test_publish_uses_atomic_result_and_manifest(tmp_path: Path) -> None:
    staged = tmp_path / "artifacts" / "face.glb"
    staged.parent.mkdir()
    staged.write_bytes(b"valid-glb")
    output = tmp_path / "result" / "face.glb"

    manifest = publish_validated_glb(staged, output, {"glb_faces": 123})

    assert output.read_bytes() == b"valid-glb"
    stored = json.loads((output.parent / "result.json").read_text(encoding="utf-8"))
    assert stored == manifest
    assert stored["status"] == "complete"
    assert stored["quality"]["glb_faces"] == 123
    assert len(stored["sha256"]) == 64


def test_texture_resume_fingerprint_changes_with_same_size_mesh_content(
    tmp_path: Path,
) -> None:
    mesh = tmp_path / "face.ply"
    mesh.write_bytes(b"aaaa")
    code_root = Path(__file__).resolve().parents[2]
    config = TextureConfig(code_root, tmp_path, mesh, tmp_path / "artifacts")
    first = build_texture_resume_config(config)

    mesh.write_bytes(b"bbbb")
    second = build_texture_resume_config(config)

    assert first["uv_method"] == second["uv_method"] == "xatlas"
    assert first["atlas_size"] == second["atlas_size"] == 1024
    assert first["lpips_max_size"] == second["lpips_max_size"] == 512
    assert len(first["renderer_code_sha256"]) == 64
    assert len(first["texture_code_sha256"]) == 64
    assert len(first["unwrap_code_sha256"]) == 64
    assert first["source_mesh_sha256"] != second["source_mesh_sha256"]


def test_clean_resume_fingerprint_tracks_mesh_cameras_and_code(tmp_path: Path) -> None:
    mesh = tmp_path / "head-crop.ply"
    anchor = tmp_path / "face-anchor.ply"
    transforms = tmp_path / "transforms.json"
    mesh.write_bytes(b"mesh-a")
    anchor.write_bytes(b"anchor-a")
    transforms.write_bytes(b"cameras-a")

    first = build_clean_resume_config(CleanupConfig(), mesh, transforms, anchor)
    anchor.write_bytes(b"anchor-b")
    second = build_clean_resume_config(CleanupConfig(), mesh, transforms, anchor)
    transforms.write_bytes(b"cameras-b")
    third = build_clean_resume_config(CleanupConfig(), mesh, transforms, anchor)

    assert len(first["cleanup_code_sha256"]) == 64
    assert first["target_face_height"] == 1.35
    assert first["orientation_mesh_sha256"] != second["orientation_mesh_sha256"]
    assert second["transforms_sha256"] != third["transforms_sha256"]


def test_head_crop_resume_fingerprint_tracks_model_mesh_cameras_and_frames(
    tmp_path: Path,
) -> None:
    mesh = tmp_path / "2dgs_recon.obj"
    transforms = tmp_path / "transforms.json"
    model = tmp_path / "face-parser.onnx"
    frames = tmp_path / "frames"
    frames.mkdir()
    frame = frames / "00001.png"
    mesh.write_bytes(b"mesh-a")
    transforms.write_bytes(b"cameras-a")
    model.write_bytes(b"model-a")
    frame.write_bytes(b"frame-a")
    config = HeadCropConfig()

    first = build_head_crop_resume_config(config, mesh, transforms, frames, model)
    frame.write_bytes(b"frame-b")
    second = build_head_crop_resume_config(config, mesh, transforms, frames, model)
    transforms.write_bytes(b"cameras-b")
    third = build_head_crop_resume_config(config, mesh, transforms, frames, model)
    model.write_bytes(b"model-b")
    fourth = build_head_crop_resume_config(config, mesh, transforms, frames, model)

    assert first["uv_input"] == second["uv_input"] == "visible_2dgs_faces"
    assert len(first["crop_code_sha256"]) == 64
    assert len(first["segmentation_code_sha256"]) == 64
    assert len(first["renderer_code_sha256"]) == 64
    assert first["selected_frames"] != second["selected_frames"]
    assert second["transforms_sha256"] != third["transforms_sha256"]
    assert third["model_sha256"] != fourth["model_sha256"]


def test_head_crop_cli_defaults_are_production_values(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--video",
            str(tmp_path / "input.mov"),
            "--job-root",
            str(tmp_path / "job"),
            "--output",
            str(tmp_path / "head.glb"),
        ]
    )

    assert args.head_neck_height_ratio == 0.45
    assert args.head_maximum_hole_faces == 1000
    assert args.head_opening_rings == 3
    assert args.head_parsing_model.name == "face-parsing-resnet18.onnx"


def test_asset_export_fingerprint_tracks_same_size_texture_changes(
    tmp_path: Path,
) -> None:
    obj = tmp_path / "face.obj"
    mtl = tmp_path / "face.mtl"
    texture = tmp_path / "uv.png"
    obj.write_bytes(b"obj1")
    mtl.write_bytes(b"mtl1")
    texture.write_bytes(b"aaaa")
    report = {"obj": str(obj), "mtl": str(mtl), "texture": str(texture)}

    first = pipeline.build_asset_export_resume_config(report, np.eye(4))
    texture.write_bytes(b"bbbb")
    second = pipeline.build_asset_export_resume_config(report, np.eye(4))

    assert first["stem"] == second["stem"] == "head"
    assert first["texture_sha256"] != second["texture_sha256"]


def test_validation_fingerprint_tracks_same_size_staged_glb_changes(
    tmp_path: Path,
) -> None:
    clean_mesh = tmp_path / "face.ply"
    staged_glb = tmp_path / "face.glb"
    clean_mesh.write_bytes(b"mesh")
    staged_glb.write_bytes(b"aaaa")

    first = pipeline.build_validation_resume_config(
        clean_mesh, staged_glb, 1024, "xatlas"
    )
    staged_glb.write_bytes(b"bbbb")
    second = pipeline.build_validation_resume_config(
        clean_mesh, staged_glb, 1024, "xatlas"
    )

    assert first["staged_glb_sha256"] != second["staged_glb_sha256"]
    assert first["uv_method"] == second["uv_method"] == "xatlas"
