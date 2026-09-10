from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh
from trimesh.smoothing import filter_taubin

from production.unwrap_2dgs_uv import load_triangle_mesh


SOURCE_TO_GLTF_Y_UP_ROW_MATRIX = np.diag([-1.0, -1.0, 1.0, 1.0])


@dataclass(frozen=True)
class CleanupConfig:
    smooth_iterations: int = 3
    target_face_height: float = 1.35
    maximum_boundary_hole_extent: float = 0.18
    minimum_faces: int = 1000
    minimum_largest_component_fraction: float = 0.5
    maximum_roughness_p90_degrees: float = 30.0
    maximum_front_yaw_degrees: float = 20.0
    minimum_side_yaw_degrees: float = 30.0
    output_orientation: str = "gltf_y_up"

    def validate(self) -> None:
        if self.smooth_iterations < 0:
            raise ValueError("Smooth iterations cannot be negative")
        if self.target_face_height <= 0:
            raise ValueError("Target face height must be positive")
        if self.maximum_boundary_hole_extent < 0:
            raise ValueError("Maximum boundary hole extent cannot be negative")
        if self.minimum_faces < 1:
            raise ValueError("Minimum face count must be positive")
        if not 0 < self.minimum_largest_component_fraction <= 1:
            raise ValueError("Largest component fraction must be in (0, 1]")
        if not 0 <= self.maximum_front_yaw_degrees < 90:
            raise ValueError("Maximum front yaw must be in [0, 90)")
        if not 0 <= self.minimum_side_yaw_degrees < 90:
            raise ValueError("Minimum side yaw must be in [0, 90)")
        if self.output_orientation != "gltf_y_up":
            raise ValueError("Production STFR output orientation must be gltf_y_up")


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("Transform matrix must be 4x4")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ matrix)[:, :3]


def load_camera_to_world_matrices(
    path: Path,
    frame_names: Iterable[str] | None = None,
) -> np.ndarray:
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = metadata.get("frames", [])
    if frame_names is not None:
        frames_by_name = {
            Path(frame["file_path"]).name: frame
            for frame in frames
        }
        names = list(frame_names)
        missing = [name for name in names if name not in frames_by_name]
        if missing:
            raise ValueError(
                "Camera transforms are missing selected frames: "
                + ", ".join(missing)
            )
        frames = [frames_by_name[name] for name in names]
    matrices = np.asarray(
        [frame["transform_matrix"] for frame in frames],
        dtype=np.float64,
    )
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not len(matrices):
        raise ValueError("Camera transforms must contain at least one 4x4 matrix")
    if not np.isfinite(matrices).all():
        raise ValueError("Camera transforms contain non-finite values")
    return matrices


def camera_view_coverage(
    camera_to_world_matrices: np.ndarray,
    source_to_output: np.ndarray,
    maximum_front_yaw_degrees: float,
    minimum_side_yaw_degrees: float,
) -> dict:
    cameras = np.asarray(camera_to_world_matrices, dtype=np.float64)
    if cameras.ndim != 3 or cameras.shape[1:] != (4, 4) or not len(cameras):
        raise ValueError("Camera transforms must have shape [N, 4, 4]")
    positions = transform_points(cameras[:, :3, 3], source_to_output)
    yaw = np.rad2deg(np.arctan2(positions[:, 0], positions[:, 2]))
    if not np.isfinite(yaw).all():
        raise ValueError("Camera view yaw contains non-finite values")
    front_covered = bool(np.min(np.abs(yaw)) <= maximum_front_yaw_degrees)
    left_covered = bool(np.min(yaw) <= -minimum_side_yaw_degrees)
    right_covered = bool(np.max(yaw) >= minimum_side_yaw_degrees)
    missing = []
    if not front_covered:
        missing.append("front view")
    if not left_covered:
        missing.append("left side")
    if not right_covered:
        missing.append("right side")
    if missing:
        raise ValueError("Camera coverage is missing " + ", ".join(missing))
    return {
        "yaw_degrees": sorted(float(value) for value in yaw),
        "minimum_yaw_degrees": float(np.min(yaw)),
        "maximum_yaw_degrees": float(np.max(yaw)),
        "minimum_absolute_yaw_degrees": float(np.min(np.abs(yaw))),
        "front_covered": front_covered,
        "left_covered": left_covered,
        "right_covered": right_covered,
    }


def canonical_face_transform(
    vertices: np.ndarray,
    camera_to_world_matrices: np.ndarray,
    target_height: float,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    cameras = np.asarray(camera_to_world_matrices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("Face vertices must have shape [N, 3] with N >= 3")
    if cameras.ndim != 3 or cameras.shape[1:] != (4, 4) or not len(cameras):
        raise ValueError("Camera transforms must have shape [N, 4, 4]")
    if not np.isfinite(vertices).all() or not np.isfinite(cameras).all():
        raise ValueError("Canonicalization inputs contain non-finite values")
    if target_height <= 0:
        raise ValueError("Target face height must be positive")

    bounds_center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    camera_vectors = cameras[:, :3, 3] - bounds_center
    camera_distances = np.linalg.norm(camera_vectors, axis=1, keepdims=True)
    if np.any(camera_distances <= 1e-8):
        raise ValueError("A camera is too close to the face center")
    camera_facing = (camera_vectors / camera_distances).mean(axis=0)
    facing_length = np.linalg.norm(camera_facing)
    if facing_length <= 1e-8:
        raise ValueError("Camera positions do not define a stable face direction")
    camera_facing /= facing_length

    covariance = np.cov(vertices - vertices.mean(axis=0), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    forward = eigenvectors[:, int(np.argmin(eigenvalues))]
    if np.dot(forward, camera_facing) < 0:
        forward = -forward

    camera_up = (-cameras[:, :3, 1]).mean(axis=0)
    up = camera_up - forward * np.dot(camera_up, forward)
    up_length = np.linalg.norm(up)
    if up_length <= 1e-8:
        raise ValueError("Camera orientations do not define a stable up direction")
    up /= up_length
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    basis = np.column_stack((right, up, forward))
    projected = vertices @ basis
    projected_min = projected.min(axis=0)
    projected_max = projected.max(axis=0)
    height = projected_max[1] - projected_min[1]
    if height <= 1e-8:
        raise ValueError("Face mesh has no measurable height")
    scale = target_height / height

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = basis * scale
    matrix[3, :3] = -(projected_min + projected_max) * 0.5 * scale
    return matrix


def roughness_percentiles(mesh: trimesh.Trimesh) -> dict[str, float]:
    angles = np.rad2deg(np.asarray(mesh.face_adjacency_angles, dtype=np.float64))
    if len(angles) == 0:
        return {"median": 0.0, "p90": 0.0, "p99": 0.0}
    median, p90, p99 = np.quantile(angles, (0.5, 0.9, 0.99))
    return {"median": float(median), "p90": float(p90), "p99": float(p99)}


def largest_face_component(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, list[int]]:
    face_count = len(mesh.faces)
    if face_count == 0:
        raise ValueError("2DGS mesh has no triangle faces")
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency):
        rows = np.concatenate((adjacency[:, 0], adjacency[:, 1]))
        columns = np.concatenate((adjacency[:, 1], adjacency[:, 0]))
        graph = coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
            shape=(face_count, face_count),
        ).tocsr()
        _, labels = connected_components(graph, directed=False)
    else:
        labels = np.arange(face_count, dtype=np.int64)
    counts = np.bincount(labels)
    largest_label = int(np.argmax(counts))
    largest_indices = np.flatnonzero(labels == largest_label)
    if len(largest_indices) == face_count:
        largest = mesh
    else:
        largest = mesh.submesh([largest_indices], append=True, repair=False)
    return largest, sorted(counts.astype(int).tolist(), reverse=True)


def fill_small_boundary_loops(
    mesh: trimesh.Trimesh,
    source_to_output: np.ndarray,
    maximum_extent: float,
) -> tuple[trimesh.Trimesh, dict[str, int | float]]:
    if maximum_extent < 0:
        raise ValueError("Maximum boundary hole extent cannot be negative")
    matrix = np.asarray(source_to_output, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("Source-to-output transform must be 4x4")

    faces = np.asarray(mesh.faces, dtype=np.int64)
    directed_edges = np.vstack(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        )
    )
    undirected_edges = np.sort(directed_edges, axis=1)
    _, inverse, counts = np.unique(
        undirected_edges,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    boundary_edges = directed_edges[counts[inverse] == 1]
    empty_report = {
        "maximum_extent": float(maximum_extent),
        "boundary_components": 0,
        "closed_components": 0,
        "filled_components": 0,
        "filled_faces": 0,
        "skipped_nonplanar_components": 0,
    }
    if not len(boundary_edges):
        return mesh.copy(), empty_report

    graph = coo_matrix(
        (
            np.ones(len(boundary_edges) * 2, dtype=np.uint8),
            (
                np.concatenate((boundary_edges[:, 0], boundary_edges[:, 1])),
                np.concatenate((boundary_edges[:, 1], boundary_edges[:, 0])),
            ),
        ),
        shape=(len(mesh.vertices), len(mesh.vertices)),
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    boundary_vertices = np.unique(boundary_edges)
    component_labels = np.unique(labels[boundary_vertices])
    canonical_vertices = transform_points(np.asarray(mesh.vertices), matrix)

    added_vertices: list[np.ndarray] = []
    added_colors: list[np.ndarray] = []
    added_faces: list[np.ndarray] = []
    source_colors = np.asarray(mesh.visual.vertex_colors)
    preserve_colors = len(source_colors) == len(mesh.vertices)
    closed_components = 0
    filled_components = 0
    filled_faces = 0
    skipped_nonplanar_components = 0
    for label in component_labels:
        component_edges = boundary_edges[labels[boundary_edges[:, 0]] == label]
        component_vertices, degrees = np.unique(component_edges, return_counts=True)
        if len(component_edges) != len(component_vertices) or not np.all(degrees == 2):
            continue
        closed_components += 1
        extent = np.ptp(canonical_vertices[component_vertices], axis=0)
        if float(extent.max()) > maximum_extent:
            continue
        centered = (
            canonical_vertices[component_vertices]
            - canonical_vertices[component_vertices].mean(axis=0)
        )
        singular_values = np.linalg.svd(centered, compute_uv=False)
        if (
            len(singular_values) >= 3
            and singular_values[0] > 1e-12
            and singular_values[-1] / singular_values[0] > 0.1
        ):
            skipped_nonplanar_components += 1
            continue

        center_id = len(mesh.vertices) + len(added_vertices)
        added_vertices.append(np.asarray(mesh.vertices)[component_vertices].mean(axis=0))
        if preserve_colors:
            added_colors.append(source_colors[component_vertices].mean(axis=0))
        added_faces.append(
            np.column_stack(
                (
                    component_edges[:, 1],
                    component_edges[:, 0],
                    np.full(len(component_edges), center_id, dtype=np.int64),
                )
            )
        )
        filled_components += 1
        filled_faces += len(component_edges)

    if not added_vertices:
        output = mesh.copy()
    else:
        output_vertices = np.vstack(
            (np.asarray(mesh.vertices), np.asarray(added_vertices))
        )
        output_faces = np.vstack([faces, *added_faces])
        output_colors = None
        if preserve_colors:
            output_colors = np.vstack(
                (source_colors, np.asarray(added_colors))
            ).astype(source_colors.dtype)
        output = trimesh.Trimesh(
            vertices=output_vertices,
            faces=output_faces,
            vertex_colors=output_colors,
            process=False,
            maintain_order=True,
        )

    return output, {
        "maximum_extent": float(maximum_extent),
        "boundary_components": int(len(component_labels)),
        "closed_components": int(closed_components),
        "filled_components": int(filled_components),
        "filled_faces": int(filled_faces),
        "skipped_nonplanar_components": int(skipped_nonplanar_components),
    }


def clean_face_mesh(
    source_path: Path,
    output_path: Path,
    config: CleanupConfig = CleanupConfig(),
    camera_to_world_matrices: np.ndarray | None = None,
    orientation_path: Path | None = None,
) -> dict:
    config.validate()
    source = load_triangle_mesh(source_path)
    cleaned_input = source.copy()
    cleaned_input.update_faces(cleaned_input.nondegenerate_faces())
    cleaned_input.update_faces(cleaned_input.unique_faces())
    cleaned_input.remove_unreferenced_vertices()

    cleaned, component_face_counts = largest_face_component(cleaned_input)
    if len(cleaned.faces) < config.minimum_faces:
        raise ValueError(
            f"Largest component has {len(cleaned.faces)} triangles, fewer than "
            f"the required {config.minimum_faces}"
        )
    component_fraction = len(cleaned.faces) / len(cleaned_input.faces)
    if component_fraction < config.minimum_largest_component_fraction:
        raise ValueError(
            f"Largest component covers only {component_fraction:.3f} of the mesh"
        )

    before_smoothing = np.asarray(cleaned.vertices).copy()
    if config.smooth_iterations:
        filter_taubin(
            cleaned,
            lamb=0.2,
            nu=0.21,
            iterations=config.smooth_iterations,
        )

    displacement = np.linalg.norm(cleaned.vertices - before_smoothing, axis=1)
    orientation_vertices = np.asarray(cleaned.vertices)
    if camera_to_world_matrices is None:
        source_to_output = SOURCE_TO_GLTF_Y_UP_ROW_MATRIX
        canonicalization = "fixed_gltf_y_up"
        view_coverage = None
    else:
        canonicalization = "camera_pca"
        if orientation_path is not None:
            orientation = load_triangle_mesh(Path(orientation_path))
            orientation_vertices = np.asarray(orientation.vertices)
            if not len(orientation_vertices) or not np.isfinite(
                orientation_vertices
            ).all():
                raise ValueError("Orientation mesh must contain finite vertices")
            canonicalization = "camera_pca_orientation_mesh"
        source_to_output = canonical_face_transform(
            orientation_vertices,
            camera_to_world_matrices,
            config.target_face_height,
        )
        view_coverage = camera_view_coverage(
            camera_to_world_matrices,
            source_to_output,
            config.maximum_front_yaw_degrees,
            config.minimum_side_yaw_degrees,
        )

    cleaned, boundary_hole_report = fill_small_boundary_loops(
        cleaned,
        source_to_output,
        config.maximum_boundary_hole_extent,
    )
    roughness = roughness_percentiles(cleaned)
    if roughness["p90"] > config.maximum_roughness_p90_degrees:
        raise ValueError(
            f"Mesh roughness p90 {roughness['p90']:.2f} degrees exceeds "
            f"{config.maximum_roughness_p90_degrees:.2f}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.export(output_path)

    return {
        "config": asdict(config),
        "source_vertices": int(len(source.vertices)),
        "source_faces": int(len(source.faces)),
        "sanitized_vertices": int(len(cleaned_input.vertices)),
        "sanitized_faces": int(len(cleaned_input.faces)),
        "component_face_counts": component_face_counts[:20],
        "largest_component_fraction": float(component_fraction),
        "output_vertices": int(len(cleaned.vertices)),
        "output_faces": int(len(cleaned.faces)),
        "canonicalization": canonicalization,
        "orientation_vertices": int(len(orientation_vertices)),
        "boundary_holes": boundary_hole_report,
        "source_to_output_row_matrix": source_to_output.tolist(),
        "view_coverage": view_coverage,
        "smooth_displacement_mean": float(displacement.mean()),
        "smooth_displacement_p95": float(np.quantile(displacement, 0.95)),
        "smooth_displacement_max": float(displacement.max()),
        "roughness_degrees": roughness,
        "bounds": cleaned.bounds.astype(float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean a detailed STFR 2DGS mesh without template registration."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--smooth-iterations", type=int, default=3)
    parser.add_argument("--target-face-height", type=float, default=1.35)
    parser.add_argument("--transforms", type=Path)
    args = parser.parse_args()
    report = clean_face_mesh(
        args.source,
        args.output,
        CleanupConfig(
            smooth_iterations=args.smooth_iterations,
            target_face_height=args.target_face_height,
        ),
        camera_to_world_matrices=(
            load_camera_to_world_matrices(args.transforms)
            if args.transforms is not None
            else None
        ),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
