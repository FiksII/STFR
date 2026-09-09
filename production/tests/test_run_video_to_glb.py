import json
from pathlib import Path

import numpy as np

import production.run_video_to_glb as pipeline
from production.run_video_to_glb import (
    PIPELINE_STAGES,
    build_texture_resume_config,
    main,
    publish_validated_glb,
)
from production.texture_stage import TextureConfig


def test_pipeline_exports_direct_2dgs_mesh_without_canonical_registration() -> None:
    assert PIPELINE_STAGES == (
        "reconstruction",
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
    config = TextureConfig(tmp_path, tmp_path, mesh, tmp_path / "artifacts")
    first = build_texture_resume_config(config)

    mesh.write_bytes(b"bbbb")
    second = build_texture_resume_config(config)

    assert first["uv_method"] == second["uv_method"] == "cube"
    assert first["atlas_size"] == second["atlas_size"] == 1024
    assert first["uv_padding_pixels"] == second["uv_padding_pixels"] == 2
    assert first["lpips_max_size"] == second["lpips_max_size"] == 512
    assert first["source_mesh_sha256"] != second["source_mesh_sha256"]


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

    assert first["texture_sha256"] != second["texture_sha256"]


def test_validation_fingerprint_tracks_same_size_staged_glb_changes(
    tmp_path: Path,
) -> None:
    clean_mesh = tmp_path / "face.ply"
    staged_glb = tmp_path / "face.glb"
    clean_mesh.write_bytes(b"mesh")
    staged_glb.write_bytes(b"aaaa")

    first = pipeline.build_validation_resume_config(clean_mesh, staged_glb, 1024, 2)
    staged_glb.write_bytes(b"bbbb")
    second = pipeline.build_validation_resume_config(clean_mesh, staged_glb, 1024, 2)

    assert first["staged_glb_sha256"] != second["staged_glb_sha256"]
