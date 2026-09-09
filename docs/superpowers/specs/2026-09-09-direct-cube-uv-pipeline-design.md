# Direct 2DGS Cube-UV Production Pipeline

## Status

This document supersedes the production geometry, UV, export, and dependency
sections of `2026-09-09-video-to-glb-pipeline-design.md`. The earlier document is
design history only.

## Goal

Accept one face video and produce one validated, textured GLB through the existing
STFR worker entry point. Preserve the full cleaned `2dgs_recon.obj` surface and keep
the production path independent of Faceform Wrap, FLAME registration, xatlas, and
mesh decimation.

The intended quality/runtime balance is the approved 20-minute class pipeline. The
2DGS training and extraction configuration remains responsible for geometry quality;
UV generation must add only linear-time processing and must not damage that geometry.

## Public Contract

The worker invokes:

```bash
python -m production.run_video_to_glb \
  --video /input/capture.mov \
  --job-root /jobs/123 \
  --output /jobs/123/result/face.glb \
  --physical-gpu 1
```

Input is a readable video. Successful output is an embedded-texture GLB plus the
existing JSON reports and resumable stage state. The process exits with zero only
after reloading and validating the final GLB.

## Pipeline

### 1. Preprocess and reconstruct

Keep the existing checked subprocess flow for frame extraction, foreground matting,
COLMAP calibration, 2DGS training, mesh extraction, and STFR format conversion. The
geometry source for production is `workspace/2dgs_recon.obj`.

The production entry point does not run refinement registration or Faceform Wrap.
Legacy research commands may remain in the repository, but they are outside the
worker path and are not production dependencies.

### 2. Clean geometry directly

Load `2dgs_recon.obj`, remove non-finite vertices, degenerate faces, duplicate faces,
and disconnected debris, then keep the largest connected triangle component. Apply
the currently validated low-strength Laplacian smoothing to reduce reconstruction
noise.

Do not crop against a reference head, fit FLAME, repair holes, decimate triangles, or
transform the mesh into another coordinate system. The cleaner emits one source-space
PLY and an identity transform report. All surviving triangle indices and coordinates
become the input contract for UV generation.

### 3. Generate deterministic cube UVs

Assign every triangle to one of six charts from the dominant absolute component of
its face normal: `+X`, `-X`, `+Y`, `-Y`, `+Z`, or `-Z`. Ties use a fixed `X`, then `Y`,
then `Z` priority so repeated runs are byte-stable apart from serializer formatting.

Each chart projects onto its two orthogonal object-space axes. The negative and
positive directions use mirrored orientations so adjacent sides are consistently
oriented. Projection coordinates are normalized from the cleaned mesh bounds. When a
single projection axis has near-zero extent, place that coordinate at the tile center;
reject only a wholly degenerate mesh or non-finite data.

Split vertices only where UV charts meet. A generated vertex is identified by the
pair `(source_vertex_index, chart_id)`. Use the integer key
`chart_id * source_vertex_count + source_vertex_index` to build the remap with
vectorized NumPy operations. Positions and normals are copied from the source vertex;
faces are remapped without removal, reordering, or winding changes.

Pack the six charts into a deterministic 3-by-2 atlas. Each chart owns one rectangular
tile. UV coordinates stay inside a two-pixel inset for the default 1024-pixel texture,
and all final UV values are finite and within `[0, 1]`. Atlas size and padding are
explicit parameters so the padding remains pixel-correct at other texture sizes.

The UV section of `texture-prepare-report.json` records method `cube`, atlas size,
padding, per-chart face counts, source and output vertex counts, duplicated seam
vertices, and face count.

### 4. Train the texture

Pass the cube-UV OBJ and MTL to the existing STFR texture stage. Render visibility and
geometry buffers from the selected sharp frames, then train the neural texture with
the configured iteration count on the requested physical GPU. The stage writes
`uv.png` and preserves the OBJ material binding.

Texture generation may expose seams at chart boundaries, but it must not smooth,
resample, decimate, or otherwise modify mesh positions or faces. No generative fill,
beauty filter, or skin retouching is part of the production pipeline.

### 5. Export and validate

Export the textured OBJ directly to GLB in reconstructed coordinates. Embed the
texture, reload the GLB with trimesh, and validate it before atomically publishing the
requested output.

Validation requires:

- unchanged triangle count between cleaned PLY, cube-UV OBJ, and GLB;
- unchanged geometry bounds within serializer tolerance;
- finite positions, normals, and UVs;
- UVs within `[0, 1]` and inside their chart padding;
- valid MTL-to-`uv.png` binding and an embedded GLB material texture;
- non-empty, non-uniform texture pixels;
- no dependency on registration, Wrap, or FLAME artifacts.

## Artifacts

```text
job-root/
  workspace/
    2dgs_recon.obj
  artifacts/
    face_geometry.ply
    geometry-report.json
    face_uv_source.obj
    face_uv_source.mtl
    uv.png
    texture-prepare-report.json
    texture-report.json
    export/
      face.obj
      face.mtl
      uv.png
      face.glb
      face_vertex_colors.ply
      export-report.json
    validation-report.json
  pipeline-state.json
  result/
    face.glb
    result.json
```

Cleanup owns geometry, UV owns only seams and coordinates, texture owns pixels, and
export owns the final container.

## Failure and Resume Behavior

Reject empty meshes, wholly degenerate bounds, non-finite arrays, invalid face
indices, missing material assets, and topology changes. Stage errors include the
stage and failed artifact without logging source imagery or texture contents.

Resume compatibility includes cube-UV method, atlas size, padding, texture settings,
and the cleaned mesh fingerprint. An xatlas result or output from a different mesh
must not satisfy the cube-UV stage.

## Performance and Memory

Cube projection is `O(V + F)`. The implementation uses NumPy arrays over corners and
integer pair keys, not per-face Python chart assignment. Peak memory must fit the
current 15 GiB server GPU host while processing the measured 1.7-million-vertex,
3.3-million-face sample. Texture training remains the dominant GPU operation; OBJ
serialization may be the dominant CPU portion of UV generation and is benchmarked on
the real sample before release.

No quality claim is accepted from a decimated benchmark. The integration benchmark
uses the full cleaned production mesh.

## Dependencies

The production Python dependency set removes `xatlas` after the cube implementation
is active. Faceform Wrap, FLAME registration weights, `chumpy`, face-alignment stacks,
and `pymeshlab` are not installed for production. COLMAP, FFmpeg, CUDA, PyTorch, the
STFR renderer stack, trimesh, Open3D, NumPy, and Pillow remain required as documented
by their owning stages.

## Test Strategy

Unit tests cover all six chart assignments with a box mesh, deterministic tie
handling, seam splitting, face order and winding preservation, exact position reuse,
normal reuse, pixel padding, finite normalized UVs, and invalid/degenerate inputs.

Integration tests cover OBJ/MTL round-tripping, texture-stage command construction,
GLB texture embedding, resume fingerprints, and the complete dry-run stage graph. A
real server benchmark generates cube UVs for the full cleaned 3.3-million-face sample,
records elapsed time and peak memory, and validates the resulting OBJ before a full
video-to-GLB production run.

## Non-Goals

- Canonical FLAME topology or expression rigging.
- Watertight head completion or hair reconstruction.
- Automatic facial cropping based on landmarks or a reference mesh.
- Invisible seams under every camera and lighting condition.
- Reducing 2DGS training time by changing `mesh_res` or decimating its result.
