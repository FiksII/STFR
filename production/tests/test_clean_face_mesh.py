from pathlib import Path

import numpy as np
import pytest
import trimesh
from trimesh.smoothing import filter_taubin

from production.clean_face_mesh import (
    CleanupConfig,
    canonical_face_transform,
    clean_face_mesh,
    transform_points,
)


def test_cleanup_defaults_to_three_taubin_passes() -> None:
    assert CleanupConfig().smooth_iterations == 3
    assert CleanupConfig().output_orientation == "gltf_y_up"
    assert CleanupConfig().target_face_height == 1.35


def test_camera_canonical_transform_centers_scales_and_orients_face() -> None:
    canonical = np.array(
        [
            [x, y, z]
            for x in (-1.0, 1.0)
            for y in (-2.0, 2.0)
            for z in (-0.25, 0.25)
        ]
    )
    rotation = trimesh.transformations.euler_matrix(0.2, -0.4, 0.3)[:3, :3]
    translation = np.array([4.0, -3.0, 7.0])
    vertices = canonical @ rotation.T + translation
    world_up = rotation[:, 1]
    world_forward = rotation[:, 2]
    camera_to_world = []
    for offset in (-0.5, 0.5):
        transform = np.eye(4)
        transform[:3, 1] = -world_up
        transform[:3, 3] = translation + world_forward * 10.0 + rotation[:, 0] * offset
        camera_to_world.append(transform)

    matrix = canonical_face_transform(
        vertices,
        np.asarray(camera_to_world),
        target_height=1.35,
    )
    transformed = transform_points(vertices, matrix)
    transformed_cameras = transform_points(
        np.asarray(camera_to_world)[:, :3, 3],
        matrix,
    )
    transformed_up = world_up @ matrix[:3, :3]

    assert np.allclose((transformed.min(axis=0) + transformed.max(axis=0)) / 2, 0)
    assert np.ptp(transformed[:, 1]) == pytest.approx(1.35)
    assert transformed_cameras[:, 2].mean() > 0
    assert transformed_up[1] > 0
    assert np.linalg.det(matrix[:3, :3]) > 0


def test_cleanup_keeps_largest_2dgs_component_in_source_coordinates(
    tmp_path: Path,
) -> None:
    face = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    noise = trimesh.Trimesh(
        vertices=np.array([[5, 5, 5], [5.1, 5, 5], [5, 5.1, 5]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    source = trimesh.util.concatenate((face, noise))
    source_path = tmp_path / "source.ply"
    output_path = tmp_path / "clean.ply"
    source.export(source_path)

    report = clean_face_mesh(
        source_path,
        output_path,
        CleanupConfig(
            smooth_iterations=0,
            minimum_faces=10,
            maximum_roughness_p90_degrees=180.0,
        ),
    )

    cleaned = trimesh.load_mesh(output_path, process=False)
    assert len(cleaned.faces) == len(face.faces)
    assert report["component_face_counts"] == [len(face.faces), len(noise.faces)]
    assert np.allclose(cleaned.bounds, face.bounds, atol=1e-6)
    assert report["source_to_output_row_matrix"] == np.diag(
        [-1.0, -1.0, 1.0, 1.0]
    ).tolist()
    assert "mask_faces_before_closing" not in report


def test_cleanup_does_not_materialize_every_split_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = trimesh.util.concatenate(
        (trimesh.creation.icosphere(subdivisions=1), trimesh.creation.box())
    )
    source_path = tmp_path / "source.ply"
    output_path = tmp_path / "clean.ply"
    source.export(source_path)

    def reject_split(*args, **kwargs):
        raise AssertionError("Trimesh.split duplicates every component")

    monkeypatch.setattr(trimesh.Trimesh, "split", reject_split)

    report = clean_face_mesh(
        source_path,
        output_path,
        CleanupConfig(
            smooth_iterations=0,
            minimum_faces=1,
            minimum_largest_component_fraction=0.1,
            maximum_roughness_p90_degrees=180.0,
        ),
    )

    assert report["output_faces"] == max(80, 12)


def test_cleanup_does_not_repair_holes_or_add_triangles(tmp_path: Path) -> None:
    source = trimesh.creation.box()
    source.update_faces(np.arange(len(source.faces)) != 0)
    source_path = tmp_path / "source.ply"
    output_path = tmp_path / "clean.ply"
    source.export(source_path)

    report = clean_face_mesh(
        source_path,
        output_path,
        CleanupConfig(
            smooth_iterations=0,
            minimum_faces=1,
            maximum_roughness_p90_degrees=180.0,
        ),
    )

    assert report["output_faces"] == len(source.faces)


def test_cleanup_applies_volume_preserving_taubin_smoothing(
    tmp_path: Path,
) -> None:
    source = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    source.vertices[0] *= 1.75
    source_path = tmp_path / "source.ply"
    output_path = tmp_path / "clean.ply"
    source.export(source_path)
    expected = trimesh.load_mesh(source_path, process=False)
    filter_taubin(expected, lamb=0.2, nu=0.21, iterations=2)

    clean_face_mesh(
        source_path,
        output_path,
        CleanupConfig(
            smooth_iterations=2,
            minimum_faces=10,
            maximum_roughness_p90_degrees=180.0,
        ),
    )

    actual = trimesh.load_mesh(output_path, process=False)
    assert np.allclose(actual.vertices, expected.vertices, atol=1e-6)
