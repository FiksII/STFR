# MediaPipe Face Crop and xatlas Design

## Goal

Improve the production texture quality while preserving the detailed 2DGS
geometry and avoiding Faceform Wrap. The pipeline continues to accept one video
and publish one validated GLB. A completed 30,000-iteration reconstruction can be
resumed from the geometry stage without retraining.

## Pipeline

1. Reconstruct `2dgs_recon.obj` and select 16 sharp registered frames as today.
2. Run MediaPipe Face Mesh on each selected frame and build a face-oval mask for
   every successful detection. Optional padding remains configurable but defaults
   to no expansion because the measured 5% expansion retained non-face fragments.
3. Rasterize the original 2DGS mesh with the matching COLMAP camera. Collect only
   the visible triangle IDs whose pixels fall inside the face mask.
4. Union the triangle IDs across views, fill unselected topology components of at
   most 1000 faces that touch the selection, and keep the largest connected
   component. This closes sub-pixel rasterization gaps while leaving the large
   background component outside the crop. Optional face-adjacency expansion
   remains configurable and defaults to zero.
5. Apply three low-displacement Taubin smoothing iterations to the cropped mesh.
   Estimate a right-handed canonical frame from the facial surface plane and
   COLMAP camera poses, then center the face and scale its height to 1.35 units.
6. Generate non-overlapping UV islands with `xatlas`, optimize the 1024x1024
   texture for 301 iterations, export the canonical Y-up GLB, and run the existing
   asset checks.

MediaPipe landmarks are used only to locate the face in images. They do not
replace or deform the 2DGS surface.

## Components

`production/face_crop_stage.py` owns MediaPipe detection, face-mask construction,
camera-matched rasterization, face selection, bounded topology-gap filling,
adjacency expansion, and the crop report. Detection and selection helpers remain
separable so geometry logic can be unit tested without loading MediaPipe or CUDA.

`production/clean_face_mesh.py` remains responsible for mesh sanitization,
largest-component selection, bounded smoothing, and geometry quality metrics.
It consumes the selected face IDs rather than a proprietary template.

`production/unwrap_2dgs_uv.py` uses `xatlas==0.0.11` for the production unwrap.
Cube projection is not used by the production path because distinct surfaces can
overlap in UV space.

`production/run_video_to_glb.py` invokes the face crop after reconstruction and
before texture training. Resume fingerprints include the crop configuration,
selected-frame metadata, source mesh hash, and UV method.

## Failure Handling

The crop fails with a clear report instead of publishing a weak asset when fewer
than three frames contain a face, too few triangles are selected, or the selected
component is implausibly small. Per-frame detector misses are allowed and recorded.
The original mesh is never silently substituted for a failed face crop.

## Verification

Unit tests cover oval-mask padding, visible-face aggregation, adjacency expansion,
crop thresholds, xatlas geometry preservation, dependency pinning, resume
fingerprints, and generic UV validation. The existing `ljf.MOV` reconstruction is
then resumed from the crop stage so only crop, unwrap, texture, export, and
validation run. Front and side renders are compared with the previous cube-UV GLB
and the known-good `face_30000_uv.glb` before the result is published.
