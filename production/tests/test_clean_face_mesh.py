from pathlib import Path

import numpy as np
import trimesh
from trimesh.smoothing import filter_taubin

from production.clean_face_mesh import CleanupConfig, clean_face_mesh


def test_cleanup_defaults_to_three_taubin_passes() -> None:
    assert CleanupConfig().smooth_iterations == 3
    assert CleanupConfig().output_orientation == "gltf_y_up"


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
