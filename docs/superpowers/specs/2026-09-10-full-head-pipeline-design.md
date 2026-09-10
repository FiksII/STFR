# Full-Head Production Pipeline Design

## Goal

Replace the face-oval-only geometry crop with a production crop that retains the
observed head, hair, both ears, and a short section of neck while excluding the
shirt, shoulders, background, and face accessories. The external contract stays
unchanged: one input video produces one validated textured GLB. The result is
optimized for front and side views; it does not synthesize an unobserved rear
surface.

Development happens on `full-head-pipeline`, created from the current `pipeline`
HEAD. The existing `pipeline` branch remains unchanged.

## Selected Approach

Use the ResNet18 BiSeNet face-parsing model trained on CelebAMask-HQ through the
repository's existing ONNX Runtime CPU provider. Its 19 semantic classes distinguish
skin, facial features, ears, hair, neck, clothing, and accessories. MediaPipe
Face Mesh remains in the pipeline as a face anchor: it selects the correct
semantic component, bounds the neck crop, and provides stable geometry for
canonical orientation.

The production head mask includes labels for skin, nose, eyes, eyebrows, ears,
mouth, lips, hair, and neck. It excludes background, eyeglasses, hats, earrings,
necklaces, and clothing. A small morphological close joins narrow gaps at the
hairline and ears. Components are retained only when they intersect the
MediaPipe-anchored head region. The neck label is limited to the jaw width and
to 0.45 face heights below the chin, leaving a short neck without shoulders.

The default model is the MIT-licensed ResNet18 ONNX checkpoint from
`yakhyo/face-parsing`. It is installed separately into the repository model
cache, verified by SHA-256, and addressed by an explicit pipeline option. Model
weights are not committed to Git. The model hash is part of the resume
fingerprint.

## Pipeline

1. Reconstruct `2dgs_recon.obj` and select the 16 sharp registered frames using
   the existing reconstruction stage.
2. Run MediaPipe Face Mesh and BiSeNet parsing on each selected frame. Build and
   save a binary head mask plus per-class pixel counts for diagnostics.
3. Rasterize the original 2DGS mesh with each matching COLMAP camera. Aggregate
   visible triangle IDs whose rendered pixels lie inside the corresponding head
   mask.
4. Fill only small enclosed topology gaps, apply a three-ring topology opening
   to remove narrow reconstruction protrusions, and reject crops below the
   configured minimum size. Export both `head_crop.ply` and the face-anchor
   selection used for orientation.
5. Keep the largest usable head component, apply the existing conservative
   Taubin cleanup, and calculate the canonical transform from the face anchor
   rather than from the full head bounds. Center and scale the resulting head in
   the same right-handed glTF Y-up coordinate system as the existing output.
6. Generate non-overlapping xatlas UV islands, optimize the 1024 by 1024 texture
   against the selected views, export the GLB, and run asset validation.

The stage sequence becomes:

`reconstruction -> head_crop -> clean_geometry -> texture -> asset_export -> validate_publish`

Reconstruction remains resumable. Changing the parser model, included labels,
mask morphology, neck limit, crop code, selected frames, cameras, or source mesh
invalidates `head_crop` and all downstream stages.

## Components

`production/head_segmentation.py` owns ONNX preprocessing, semantic-label
selection, MediaPipe anchoring, neck clipping, mask post-processing, and mask
diagnostics. Its inference adapter is injectable so mask behavior can be unit
tested without loading a model or GPU runtime.

`production/head_crop_stage.py` owns camera-matched rasterization, visible-face
aggregation, bounded topology repair, crop validation, and the crop report. The
generic adjacency helpers currently in `face_crop_stage.py` move here without
changing their tested behavior.

`production/clean_face_mesh.py` gains an optional orientation mesh. The head is
cleaned as before, but the canonical basis is estimated from the face-anchor
mesh so hair and the neck cannot tilt the result.

`production/run_video_to_glb.py` exposes the parser model path and head-crop
parameters, uses head-oriented artifact names, and includes all new code and
model hashes in resume state.

`production/download_models.py` downloads the pinned ONNX checkpoint to the
model cache and verifies its expected SHA-256 before it can be used. `AGENTS.md`
documents model installation for a worker host. No new Python runtime dependency
is required because ONNX Runtime is already pinned. The CPU provider is
intentional: the 16 production masks take about 1.1 seconds on the verified
server, while avoiding a cuDNN 9 dependency that conflicts with the pipeline's
PyTorch 2.3.1 and cuDNN 8 stack. Reconstruction, mesh rasterization, and texture
optimization still use physical GPU 1.

## Failure Handling

The pipeline stops without publishing when the model file is absent or has the
wrong hash, MediaPipe cannot anchor the minimum number of frames, semantic masks
are empty, the retained head is implausibly small, or clothing dominates the
candidate region. Individual frame misses are recorded and allowed when the
minimum detected-frame threshold is still met. It never silently falls back to
the old face-only crop or the uncropped bust.

Each successful run records detected frames, semantic pixel counts, neck-clipped
pixels, selected face counts before and after topology cleanup, bounds, model
hash, stage timing, and paths to diagnostic masks.

## Verification

Unit tests cover semantic-label inclusion, accessory and clothing exclusion,
MediaPipe component anchoring, the 0.45-height neck limit, mask morphology,
multi-view triangle aggregation, crop thresholds, model-hash validation,
orientation by the face anchor, CLI wiring, and resume invalidation.

The full production test suite must pass on the server. The existing `ljf.MOV`
job is then resumed from its saved 30,000-iteration reconstruction on physical
GPU 1. The acceptance artifacts are the final GLB, UV texture, binary-mask
contact sheet, and front/left/right renders. Visual acceptance requires intact
hair and ears, a short neck, no shirt or shoulders, no narrow floating spikes,
and no texture cuts across the visible face. The measured post-reconstruction
runtime is reported rather than assumed.
