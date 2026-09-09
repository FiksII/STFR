from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


CHART_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
ATLAS_COLUMNS = 3
ATLAS_ROWS = 2


@dataclass(frozen=True)
class CubeAtlas:
    vertex_mapping: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    face_charts: np.ndarray


def load_triangle_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Mesh scene is empty: {path}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh, got {type(loaded).__name__}")
    if loaded.faces.ndim != 2 or loaded.faces.shape[1] != 3:
        raise ValueError("Only triangular meshes are supported")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError("Mesh must contain vertices and faces")
    if not np.isfinite(loaded.vertices).all():
        raise ValueError("Mesh contains non-finite vertex coordinates")
    return loaded


def _normalized_positions(vertices: np.ndarray) -> np.ndarray:
    bounds_min = vertices.min(axis=0)
    extent = vertices.max(axis=0) - bounds_min
    scale = max(float(extent.max()), 1.0)
    normalized = np.full(vertices.shape, 0.5, dtype=np.float64)
    usable = extent > scale * 1e-12
    normalized[:, usable] = (
        vertices[:, usable] - bounds_min[usable]
    ) / extent[usable]
    return normalized


def _local_chart_uv(normalized: np.ndarray, charts: np.ndarray) -> np.ndarray:
    x, y, z = normalized.T
    local = np.empty((len(normalized), 2), dtype=np.float64)
    projections = (
        (1.0 - z, y),
        (z, y),
        (x, 1.0 - z),
        (x, z),
        (x, y),
        (1.0 - x, y),
    )
    for chart_id, (u, v) in enumerate(projections):
        selected = charts == chart_id
        local[selected, 0] = u[selected]
        local[selected, 1] = v[selected]
    return local


def validate_cube_uv_padding(
    uvs: np.ndarray,
    atlas_size: int,
    padding_pixels: int,
) -> None:
    uvs = np.asarray(uvs, dtype=np.float64)
    if atlas_size < 1 or padding_pixels < 0:
        raise ValueError("Atlas size must be positive and padding non-negative")
    pad = padding_pixels / atlas_size
    if 2.0 * pad >= 1.0 / ATLAS_COLUMNS:
        raise ValueError("Padding leaves no usable cube-atlas tile area")
    if uvs.ndim != 2 or uvs.shape[1] != 2 or not len(uvs):
        raise ValueError("UV coordinates must have shape [N, 2]")
    if not np.isfinite(uvs).all():
        raise ValueError("UV coordinates contain non-finite values")
    tolerance = 1e-9
    if uvs.min() < -tolerance or uvs.max() > 1.0 + tolerance:
        raise ValueError("UV coordinates are outside [0, 1]")

    columns = np.minimum(
        (uvs[:, 0] * ATLAS_COLUMNS).astype(np.int64),
        ATLAS_COLUMNS - 1,
    )
    rows = np.minimum(
        (uvs[:, 1] * ATLAS_ROWS).astype(np.int64),
        ATLAS_ROWS - 1,
    )
    lower_u = columns / ATLAS_COLUMNS + pad
    upper_u = (columns + 1) / ATLAS_COLUMNS - pad
    lower_v = rows / ATLAS_ROWS + pad
    upper_v = (rows + 1) / ATLAS_ROWS - pad
    valid = (
        (uvs[:, 0] >= lower_u - tolerance)
        & (uvs[:, 0] <= upper_u + tolerance)
        & (uvs[:, 1] >= lower_v - tolerance)
        & (uvs[:, 1] <= upper_v + tolerance)
    )
    if not np.all(valid):
        raise ValueError("UV coordinates violate cube-atlas tile padding")


def build_cube_atlas(
    vertices: np.ndarray,
    faces: np.ndarray,
    atlas_size: int = 1024,
    padding_pixels: int = 2,
) -> CubeAtlas:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("Faces must have shape [F, 3]")
    if not np.isfinite(vertices).all():
        raise ValueError("Mesh contains non-finite vertex coordinates")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Mesh contains invalid face indices")
    if atlas_size < 1 or padding_pixels < 0:
        raise ValueError("Atlas size must be positive and padding non-negative")

    pad = padding_pixels / atlas_size
    tile_width = 1.0 / ATLAS_COLUMNS
    tile_height = 1.0 / ATLAS_ROWS
    if 2.0 * pad >= min(tile_width, tile_height):
        raise ValueError("Padding leaves no usable cube-atlas tile area")

    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(face_normals, axis=1)
    mesh_scale = max(float(np.ptp(vertices, axis=0).max()), 1.0)
    if np.any(lengths <= mesh_scale * mesh_scale * 1e-14):
        raise ValueError("Mesh contains a degenerate triangle")
    axes = np.argmax(np.abs(face_normals), axis=1)
    signs = face_normals[np.arange(len(faces)), axes] < 0.0
    face_charts = axes * 2 + signs.astype(np.int64)

    corner_keys = (face_charts[:, None] * len(vertices) + faces).reshape(-1)
    unique_keys, inverse = np.unique(corner_keys, return_inverse=True)
    vertex_mapping = unique_keys % len(vertices)
    vertex_charts = unique_keys // len(vertices)
    remapped_faces = inverse.reshape(faces.shape)

    normalized = _normalized_positions(vertices)[vertex_mapping]
    local = _local_chart_uv(normalized, vertex_charts)
    columns = vertex_charts % ATLAS_COLUMNS
    rows = vertex_charts // ATLAS_COLUMNS
    uvs = np.empty_like(local)
    uvs[:, 0] = (
        columns * tile_width
        + pad
        + local[:, 0] * (tile_width - 2.0 * pad)
    )
    uvs[:, 1] = (
        rows * tile_height
        + pad
        + local[:, 1] * (tile_height - 2.0 * pad)
    )
    validate_cube_uv_padding(uvs, atlas_size, padding_pixels)
    return CubeAtlas(vertex_mapping, remapped_faces, uvs, face_charts)


def write_mtl(path: Path, material_name: str, texture_name: str) -> None:
    content = (
        f"newmtl {material_name}\n"
        "Ka 1.000000 1.000000 1.000000\n"
        "Kd 1.000000 1.000000 1.000000\n"
        "Ks 0.020000 0.020000 0.020000\n"
        "Ns 8.000000\n"
        "illum 2\n"
        f"map_Kd {texture_name}\n"
    )
    path.write_text(content, encoding="ascii")


def write_obj(
    path: Path,
    vertices: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    material_name: str,
) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"mtllib {path.with_suffix('.mtl').name}\n")
        handle.write("o 2dgs_recon_uv\n")
        for x, y, z in vertices:
            handle.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for u, v in uvs:
            handle.write(f"vt {u:.9g} {v:.9g}\n")
        for nx, ny, nz in normals:
            handle.write(f"vn {nx:.9g} {ny:.9g} {nz:.9g}\n")
        handle.write(f"usemtl {material_name}\n")
        for face in faces + 1:
            a, b, c = (int(index) for index in face)
            handle.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")


def _parametrize_with_xatlas(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import xatlas

    return xatlas.parametrize(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.uint32),
    )


def unwrap_mesh(
    input_path: Path,
    output_obj: Path,
    material_name: str = "face",
    texture_name: str = "uv.png",
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_obj = Path(output_obj)
    source = load_triangle_mesh(input_path)
    source_vertices = np.asarray(source.vertices, dtype=np.float64)
    source_faces = np.asarray(source.faces, dtype=np.int64)
    source_normals = np.asarray(source.vertex_normals, dtype=np.float64)
    if not np.isfinite(source_normals).all():
        raise ValueError("Mesh contains non-finite vertex normals")

    vertex_mapping, atlas_faces, atlas_uvs = _parametrize_with_xatlas(
        source_vertices,
        source_faces,
    )
    vertex_mapping = np.asarray(vertex_mapping, dtype=np.int64)
    atlas_faces = np.asarray(atlas_faces, dtype=np.int64)
    atlas_uvs = np.asarray(atlas_uvs, dtype=np.float32)
    if vertex_mapping.ndim != 1 or not len(vertex_mapping):
        raise ValueError("xatlas produced an invalid vertex mapping")
    if vertex_mapping.min() < 0 or vertex_mapping.max() >= len(source_vertices):
        raise ValueError("xatlas produced invalid source vertex indices")
    if atlas_faces.shape != source_faces.shape:
        raise ValueError("xatlas changed the triangle count or topology shape")
    if atlas_faces.min() < 0 or atlas_faces.max() >= len(vertex_mapping):
        raise ValueError("xatlas produced invalid face indices")
    if atlas_uvs.shape != (len(vertex_mapping), 2):
        raise ValueError("xatlas produced an invalid UV array")
    if not np.isfinite(atlas_uvs).all():
        raise ValueError("xatlas produced non-finite UV coordinates")
    tolerance = 1e-6
    if atlas_uvs.min() < -tolerance or atlas_uvs.max() > 1.0 + tolerance:
        raise ValueError("xatlas produced UV coordinates outside [0, 1]")
    atlas_uvs = np.clip(atlas_uvs, 0.0, 1.0)

    output_obj.parent.mkdir(parents=True, exist_ok=True)
    write_obj(
        output_obj,
        source_vertices[vertex_mapping],
        source_normals[vertex_mapping],
        atlas_uvs,
        atlas_faces,
        material_name,
    )
    write_mtl(output_obj.with_suffix(".mtl"), material_name, texture_name)
    referenced_source_vertices = len(np.unique(source_faces))
    return {
        "method": "xatlas",
        "source": str(input_path.resolve()),
        "output": str(output_obj.resolve()),
        "source_vertices": int(len(source_vertices)),
        "referenced_source_vertices": int(referenced_source_vertices),
        "output_vertices": int(len(vertex_mapping)),
        "duplicated_seam_vertices": int(
            len(vertex_mapping) - referenced_source_vertices
        ),
        "source_faces": int(len(source_faces)),
        "output_faces": int(len(atlas_faces)),
        "uv_min": atlas_uvs.min(axis=0).astype(float).tolist(),
        "uv_max": atlas_uvs.max(axis=0).astype(float).tolist(),
        "bounds_min": source_vertices.min(axis=0).astype(float).tolist(),
        "bounds_max": source_vertices.max(axis=0).astype(float).tolist(),
        "material": material_name,
        "texture": texture_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an xatlas UV layout for a cleaned STFR 2DGS mesh."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--material", default="face")
    parser.add_argument("--texture", default="uv.png")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = unwrap_mesh(
        args.input,
        args.output,
        args.material,
        args.texture,
    )
    report_path = args.report or args.output.with_suffix(".uv-report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
