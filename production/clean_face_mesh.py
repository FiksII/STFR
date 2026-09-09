from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pymeshlab
import trimesh
from trimesh.smoothing import filter_taubin

from production.unwrap_2dgs_uv import load_triangle_mesh


@dataclass(frozen=True)
class CleanupConfig:
    distance: float = 0.05
    smooth_iterations: int = 3
    mask_close_holes: int = 50
    minimum_faces: int = 1000
    minimum_largest_component_fraction: float = 0.8
    maximum_correspondence_residual: float = 1e-5
    maximum_roughness_p90_degrees: float = 30.0

    def validate(self) -> None:
        if self.distance <= 0:
            raise ValueError("Cleanup distance must be positive")
        if self.smooth_iterations < 0:
            raise ValueError("Smooth iterations cannot be negative")
        if self.mask_close_holes < 0:
            raise ValueError("Mask hole size cannot be negative")
        if self.minimum_faces < 1:
            raise ValueError("Minimum face count must be positive")
        if not 0 < self.minimum_largest_component_fraction <= 1:
            raise ValueError("Largest component fraction must be in (0, 1]")


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("Transform matrix must be 4x4")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ matrix)[:, :3]


def fit_correspondence_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Source and canonical vertices must have matching Nx3 shapes")
    sample = np.linspace(0, len(source) - 1, min(20000, len(source)), dtype=np.int64)
    design = np.column_stack((source[sample], np.ones(len(sample))))
    affine = np.linalg.lstsq(design, target[sample], rcond=None)[0]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:, :3] = affine
    return matrix


def load_closed_reference(path: Path, max_hole_size: int) -> trimesh.Trimesh:
    if max_hole_size <= 0:
        return load_triangle_mesh(path)
    mesh_set = pymeshlab.MeshSet()
    mesh_set.load_new_mesh(str(path))
    mesh_set.apply_filter(
        "meshing_close_holes",
        maxholesize=max_hole_size,
        newfaceselected=True,
    )
    mesh = mesh_set.current_mesh()
    return trimesh.Trimesh(
        vertices=mesh.vertex_matrix(),
        faces=mesh.face_matrix(),
        process=False,
    )


def distance_to_mesh(points: np.ndarray, reference: trimesh.Trimesh) -> np.ndarray:
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(reference.vertices),
        o3d.utility.Vector3iVector(reference.faces),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene.compute_distance(
        o3d.core.Tensor(np.asarray(points, dtype=np.float32))
    ).numpy()


def roughness_percentiles(mesh: trimesh.Trimesh) -> dict[str, float]:
    angles = np.rad2deg(np.asarray(mesh.face_adjacency_angles, dtype=np.float64))
    if len(angles) == 0:
        return {"median": 0.0, "p90": 0.0, "p99": 0.0}
    median, p90, p99 = np.quantile(angles, (0.5, 0.9, 0.99))
    return {"median": float(median), "p90": float(p90), "p99": float(p99)}


def clean_face_mesh(
    source_path: Path,
    canonical_path: Path,
    reference_path: Path,
    output_source: Path,
    output_canonical: Path,
    config: CleanupConfig = CleanupConfig(),
) -> dict:
    config.validate()
    source = load_triangle_mesh(source_path)
    canonical = load_triangle_mesh(canonical_path)
    if source.vertices.shape != canonical.vertices.shape:
        raise ValueError("Source and canonical meshes have different vertex counts")
    if source.faces.shape != canonical.faces.shape or not np.array_equal(
        source.faces, canonical.faces
    ):
        raise ValueError("Source and canonical meshes do not share topology")

    original_reference = load_triangle_mesh(reference_path)
    reference = load_closed_reference(reference_path, config.mask_close_holes)
    matrix = fit_correspondence_transform(source.vertices, canonical.vertices)
    canonical_points = transform_points(source.vertices, matrix)
    residual = np.linalg.norm(canonical_points - canonical.vertices, axis=1)
    residual_max = float(residual.max())
    if residual_max > config.maximum_correspondence_residual:
        raise ValueError(
            f"Canonical correspondence residual {residual_max:.6g} exceeds "
            f"{config.maximum_correspondence_residual:.6g}"
        )

    distances = distance_to_mesh(canonical_points, reference)
    selected_vertices = distances < config.distance
    selected_faces = np.all(selected_vertices[np.asarray(source.faces)], axis=1)
    selected_face_indices = np.flatnonzero(selected_faces)
    if len(selected_face_indices) < config.minimum_faces:
        raise ValueError(
            f"Face crop selected {len(selected_face_indices)} triangles, fewer than "
            f"the required {config.minimum_faces}"
        )

    cropped = source.submesh([selected_face_indices], append=True, repair=False)
    components = list(cropped.split(only_watertight=False))
    component_face_counts = sorted(
        (len(component.faces) for component in components), reverse=True
    )
    cropped = max(components, key=lambda component: len(component.faces))
    component_fraction = len(cropped.faces) / len(selected_face_indices)
    if component_fraction < config.minimum_largest_component_fraction:
        raise ValueError(
            f"Largest face component covers only {component_fraction:.3f} of crop"
        )

    canonical_cropped = cropped.copy()
    canonical_cropped.vertices = transform_points(cropped.vertices, matrix)
    before_smoothing = np.asarray(canonical_cropped.vertices).copy()
    if config.smooth_iterations:
        filter_taubin(
            canonical_cropped,
            lamb=0.2,
            nu=0.21,
            iterations=config.smooth_iterations,
        )

    roughness = roughness_percentiles(canonical_cropped)
    if roughness["p90"] > config.maximum_roughness_p90_degrees:
        raise ValueError(
            f"Mesh roughness p90 {roughness['p90']:.2f} degrees exceeds "
            f"{config.maximum_roughness_p90_degrees:.2f}"
        )

    source_cropped = canonical_cropped.copy()
    source_cropped.vertices = transform_points(
        canonical_cropped.vertices, np.linalg.inv(matrix)
    )
    for path, mesh in (
        (Path(output_source), source_cropped),
        (Path(output_canonical), canonical_cropped),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(path)

    displacement = np.linalg.norm(canonical_cropped.vertices - before_smoothing, axis=1)
    return {
        "config": asdict(config),
        "source_vertices": int(len(source.vertices)),
        "source_faces": int(len(source.faces)),
        "selected_vertices": int(selected_vertices.sum()),
        "selected_faces": int(len(selected_face_indices)),
        "component_face_counts": component_face_counts[:20],
        "largest_component_fraction": float(component_fraction),
        "output_vertices": int(len(canonical_cropped.vertices)),
        "output_faces": int(len(canonical_cropped.faces)),
        "mask_faces_before_closing": int(len(original_reference.faces)),
        "mask_faces_after_closing": int(len(reference.faces)),
        "correspondence_residual_max": residual_max,
        "source_to_canonical_row_matrix": matrix.tolist(),
        "smooth_displacement_mean": float(displacement.mean()),
        "smooth_displacement_p95": float(np.quantile(displacement, 0.95)),
        "smooth_displacement_max": float(displacement.max()),
        "roughness_degrees": roughness,
        "canonical_bounds": canonical_cropped.bounds.astype(float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and canonicalize an STFR 2DGS face mesh.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--face-reference", type=Path, required=True)
    parser.add_argument("--output-source", type=Path, required=True)
    parser.add_argument("--output-canonical", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--distance", type=float, default=0.05)
    parser.add_argument("--smooth-iterations", type=int, default=3)
    parser.add_argument("--mask-close-holes", type=int, default=50)
    args = parser.parse_args()
    report = clean_face_mesh(
        args.source,
        args.canonical,
        args.face_reference,
        args.output_source,
        args.output_canonical,
        CleanupConfig(
            distance=args.distance,
            smooth_iterations=args.smooth_iterations,
            mask_close_holes=args.mask_close_holes,
        ),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
