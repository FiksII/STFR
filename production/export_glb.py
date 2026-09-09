from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import trimesh


def transform_obj(
    input_path: Path,
    output_path: Path,
    matrix: np.ndarray,
    mtl_name: str,
) -> None:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("Canonical transform matrix must be 4x4")
    linear = matrix[:3, :3]
    normal_matrix = np.linalg.inv(linear).T
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(input_path).open("r", encoding="ascii") as source, output_path.open(
        "w", encoding="ascii", newline="\n"
    ) as target:
        for line in source:
            if line.startswith("mtllib "):
                target.write(f"mtllib {mtl_name}\n")
            elif line.startswith("v "):
                values = np.fromstring(line[2:], sep=" ")[:3]
                transformed = np.append(values, 1.0) @ matrix
                target.write(
                    f"v {transformed[0]:.10g} {transformed[1]:.10g} "
                    f"{transformed[2]:.10g}\n"
                )
            elif line.startswith("vn "):
                normal = np.fromstring(line[3:], sep=" ")[:3] @ normal_matrix
                normal /= max(np.linalg.norm(normal), 1e-12)
                target.write(
                    f"vn {normal[0]:.10g} {normal[1]:.10g} {normal[2]:.10g}\n"
                )
            else:
                target.write(line)


def write_rebound_mtl(input_mtl: Path, output_mtl: Path, texture_name: str) -> None:
    lines = Path(input_mtl).read_text(encoding="ascii").splitlines()
    replaced = False
    output_lines = []
    for line in lines:
        if line.startswith("map_Kd "):
            output_lines.append(f"map_Kd {texture_name}")
            replaced = True
        else:
            output_lines.append(line)
    if not replaced:
        output_lines.append(f"map_Kd {texture_name}")
    output_mtl.write_text("\n".join(output_lines) + "\n", encoding="ascii")


def load_output_transform(report_path: Path) -> np.ndarray:
    cleanup = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return np.asarray(cleanup["source_to_output_row_matrix"], dtype=np.float64)


def export_canonical_asset(
    input_obj: Path,
    input_mtl: Path,
    texture: Path,
    matrix: np.ndarray,
    output_dir: Path,
    stem: str = "face",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_obj = output_dir / f"{stem}.obj"
    output_mtl = output_dir / f"{stem}.mtl"
    output_texture = output_dir / "uv.png"
    output_glb = output_dir / f"{stem}.glb"
    output_ply = output_dir / f"{stem}_vertex_colors.ply"

    transform_obj(input_obj, output_obj, matrix, output_mtl.name)
    write_rebound_mtl(input_mtl, output_mtl, output_texture.name)
    shutil.copy2(texture, output_texture)
    scene = trimesh.load(output_obj, process=False, force="scene")
    if not scene.geometry:
        raise ValueError("Output OBJ contains no geometry")
    scene.export(output_glb)
    combined = trimesh.util.concatenate(tuple(scene.geometry.values()))
    combined.visual = combined.visual.to_color()
    combined.export(output_ply)

    report = {
        "obj": str(output_obj.resolve()),
        "mtl": str(output_mtl.resolve()),
        "texture": str(output_texture.resolve()),
        "glb": str(output_glb.resolve()),
        "vertex_color_ply": str(output_ply.resolve()),
        "vertices": int(sum(len(mesh.vertices) for mesh in scene.geometry.values())),
        "faces": int(sum(len(mesh.faces) for mesh in scene.geometry.values())),
        "bounds": scene.bounds.astype(float).tolist(),
    }
    (output_dir / "export-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a direct textured STFR face.")
    parser.add_argument("--input-obj", type=Path, required=True)
    parser.add_argument("--input-mtl", type=Path, required=True)
    parser.add_argument("--texture", type=Path, required=True)
    parser.add_argument("--clean-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="face")
    args = parser.parse_args()
    report = export_canonical_asset(
        args.input_obj,
        args.input_mtl,
        args.texture,
        load_output_transform(args.clean_report),
        args.output_dir,
        args.stem,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
