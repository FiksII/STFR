# STFR Video-to-GLB Production Pipeline

## Goal

Provide worker-callable scripts that accept one face video and publish one validated,
canonically oriented, textured GLB. The worker itself is outside this repository and
is not changed by this work.

## Public Contract

The worker invokes one module:

```bash
python -m production.run_video_to_glb \
  --video /input/capture.mov \
  --job-root /jobs/123 \
  --output /jobs/123/result/face.glb \
  --physical-gpu 1
```

Defaults match the validated quality run:

- 30,000 2DGS training iterations;
- mesh extraction resolution 1024;
- face-reference distance 0.05 canonical units;
- three low-strength Taubin smoothing iterations;
- 301 neural-texture iterations;
- physical GPU 1 (the second GPU).

The process exits with zero only after the final GLB has been reloaded and validated.
Progress is emitted as one JSON object per stdout line. Fatal diagnostics go to stderr
and produce a non-zero exit code.

## Pipeline

### 1. Capture preprocessing

Extract frames with ffmpeg using the configured frame step and downscale ratio. Run
the existing foreground matting module. Existing non-empty outputs may be reused only
when `--resume` is supplied.

### 2. Full 2DGS reconstruction

Run the existing conversion, COLMAP calibration, 2DGS training, mesh extraction, and
STFR format conversion as checked subprocesses. Production requires the requested
30,000-iteration checkpoint; a fast 4,000-iteration checkpoint must never satisfy the
stage validation. Run the existing refinement and registration stages because their
selected source frames and canonical correspondence are needed by texturing and mesh
cleanup.

### 3. Face mesh cleanup

Use `2dgs_recon.obj` as the geometry source. Derive an affine correspondence from that
mesh to `register/fine_align/align_canonical.obj`, which has identical topology and
vertex ordering. Use `register/wrap/final_hack.obj` only as a spatial face mask, never
as output geometry. Close its small eye openings for mask-distance evaluation, keep
the largest connected 2DGS face surface within the configured distance, and apply
low-strength Taubin smoothing. Preserve both source-coordinate and canonical versions:
source coordinates are required for camera-based texturing; canonical coordinates are
used for final export.

Reject the stage when correspondence residual is too high, the mesh is empty, the
largest component is unexpectedly small, coordinates are non-finite, or geometric
roughness exceeds the configured production threshold.

### 4. UV and texture

Unwrap the cleaned source-coordinate mesh with xatlas. Render visibility and geometry
buffers from the sharp frames selected by STFR refinement. Train the existing STFR
neural texture for 301 iterations on the selected physical GPU. No generative fill,
beauty filtering, or skin retouching is permitted.

The texture modules receive the mesh path explicitly and respect an externally supplied
`CUDA_VISIBLE_DEVICES`. They must not silently select another physical GPU.

### 5. Canonical export and validation

Apply the exact source-to-canonical transform to OBJ positions and the inverse-transpose
transform to normals while preserving UV indices and material binding. Export an
embedded-texture GLB to a temporary path, reload it with trimesh, validate topology,
bounds, finite UVs, non-uniform texture data, and material presence, then atomically
replace the requested output path.

## Files and Artifacts

The implementation adds a `production` package with focused modules:

- `run_video_to_glb.py`: public CLI and stage orchestration;
- `reconstruction_stage.py`: checked calls into ffmpeg and existing STFR modules;
- `clean_face_mesh.py`: correspondence, mask crop, smoothing, and geometry metrics;
- `unwrap_2dgs_uv.py`: deterministic xatlas OBJ/MTL export;
- `texture_stage.py`: selected-frame dataset and STFR texture execution;
- `export_glb.py`: canonical OBJ/GLB/PLY export;
- `validate_asset.py`: machine-readable validation.

The job directory contains:

```text
job-root/
  workspace/              # native STFR stage outputs
  artifacts/              # cleaned PLY, UV OBJ/MTL, uv.png, reports
  pipeline-state.json      # completed stages and timings
  result/
    face.glb               # or the explicit --output path
    result.json            # final manifest and quality metrics
```

Only `face.glb` is the required product artifact. Intermediate files remain available
for diagnosis and can be deleted by the worker after successful upload.

## Resuming and Failure Behavior

Each stage records its status, duration, configuration, and validated outputs. With
`--resume`, a stage is skipped only when its recorded configuration still matches and
all required outputs pass validation. Without `--resume`, an existing non-empty job
root is rejected to avoid mixing patients or configurations.

The orchestrator handles termination by forwarding it to the active subprocess and
never publishing a partial GLB. Python exceptions include the failed stage and command
but do not expose texture or source-image contents in logs.

## Tests

Unit tests use synthetic meshes and temporary files. They cover command construction,
frame filtering, correspondence transforms, eye-safe mask cropping, UV preservation,
normal transformation, GLB reload, stage-state compatibility, and failure propagation.
A dry-run CLI test verifies the complete video-to-GLB stage graph without invoking GPU
code. The existing server sample is the integration fixture for a real 30,000-step run.

## Dependencies

Add `xatlas` and `pymeshlab` to the documented STFR environment. Existing dependencies
already provide numpy, trimesh, Open3D, Pillow, PyTorch, and the STFR rendering stack.
