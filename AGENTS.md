# STFR production environment

This repository targets Ubuntu 24.04 x86_64 with NVIDIA GPUs. The verified
server uses two RTX 3090 cards, NVIDIA driver 595.84, Ubuntu's CUDA 12.0
compiler for PyTorch extensions, CUDA Toolkit 12.4 for COLMAP, PyTorch
2.3.1+cu121, and COLMAP 3.12.6 with CUDA enabled.

Python packages are managed by uv. NVIDIA drivers, the CUDA Toolkit, COLMAP,
Faceform Wrap, FFmpeg, Xvfb, model weights, and licensed assets are system or
operator-managed dependencies and must not be added to the uv environment.

Agents must inspect the existing installation before changing system packages.
Do not reinstall a working NVIDIA driver, reboot the host, or activate/change a
Faceform license without explicit operator approval.

## Python dependencies with uv

Install uv and the pinned Python interpreter from the repository root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.10
```

The native CUDA packages use the compiler from `CUDA_HOME`. Before the first
sync, make the NVIDIA toolkit explicit:

```bash
export CUDA_HOME=/usr
export PATH="/usr/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"
```

Use `/usr/bin/nvcc` from Ubuntu's CUDA 12.0 package for the PyTorch 2.3.1
extensions. On the verified Ubuntu 24.04 server, building `simple-knn` against
CUDA 12.4 and GCC 13 fails inside the PyTorch headers.

Create the locked environment after exporting the compiler settings:

```bash
uv sync --frozen
```

For a production image without test dependencies, use:

```bash
uv sync --frozen --no-dev
```

Verify the environment after synchronization:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
uv run python -c "import pytorch3d, tinycudann, diff_surfel_rasterization, simple_knn"
uv run python -m pytest -q production/tests
```

Do not run `uv lock --upgrade` casually. The direct wheel URLs and Git commit
SHAs in `pyproject.toml` reproduce the environment used for the production
pipeline.

## NVIDIA driver and CUDA Toolkit

These commands are for Ubuntu 24.04 x86_64. Install matching kernel headers,
enable NVIDIA's signed APT repository, then install the driver and the 12.4
development toolkit. Also install Ubuntu's CUDA 12.0 toolkit for the PyTorch
extensions:

```bash
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r) wget
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-drivers cuda-toolkit-12-4 nvidia-cuda-toolkit
sudo reboot
```

After reconnecting, verify both the kernel driver and compiler. Do not continue
if either command fails:

```bash
nvidia-smi
/usr/bin/nvcc --version
/usr/local/cuda-12.4/bin/nvcc --version
```

The PyTorch wheels carry their CUDA 12.1 runtime. The system CUDA Toolkit is
still required to compile tiny-cuda-nn and the 2DGS rasterization extensions.
Keep `CUDA_HOME=/usr` while running `uv sync`. Use CUDA 12.4 only for the
separate COLMAP build below.

## CUDA-enabled COLMAP

STFR explicitly requests GPU SIFT extraction and matching. Ubuntu's prebuilt
COLMAP package does not include CUDA support, so build COLMAP from source.
Install the dependencies listed by the COLMAP project:

```bash
sudo apt-get update
sudo apt-get install -y \
  git cmake ninja-build build-essential ccache \
  libboost-program-options-dev libboost-graph-dev libboost-system-dev \
  libeigen3-dev libopenimageio-dev openimageio-tools libmetis-dev \
  libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev libglew-dev \
  qt6-base-dev libqt6opengl6-dev libqt6openglwidgets6 qt6-svg-dev \
  libcgal-dev libceres-dev libsuitesparse-dev libcurl4-openssl-dev \
  libssl-dev libmkl-full-dev
sudo mkdir -p /usr/include/opencv4
```

Build the tested COLMAP release for the local GPU architecture. RTX 3090 uses
compute capability 8.6:

```bash
git clone --branch 3.12.6 --depth 1 https://github.com/colmap/colmap.git /tmp/colmap
cmake -S /tmp/colmap -B /tmp/colmap/build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCUDA_ENABLED=ON \
  -DGUI_ENABLED=OFF \
  -DTESTS_ENABLED=OFF
cmake --build /tmp/colmap/build
sudo cmake --install /tmp/colmap/build
sudo ldconfig
```

Verify the installed binary and confirm it links to CUDA:

```bash
colmap -h
ldd "$(command -v colmap)" | grep -E 'cuda|cudart'
```

## Faceform Wrap

Wrap is proprietary and requires an operator with Faceform download and license
access. Download the Linux Portable build of Wrap 2025.11.14 from the official
Faceform customer portal. Extract it so these files exist:

```text
registration/wrap/Wrap
registration/wrap/WrapCmd
registration/wrap/Gallery/
registration/wrap/lib/
registration/wrap/plugins/
```

Install the headless display helper and ensure the executables can run:

```bash
sudo apt-get install -y xvfb ffmpeg
cd registration/wrap
chmod +x Wrap WrapCmd
./WrapCmd --version
./WrapCmd --license
```

Activate a nodelocked or floating license according to the Faceform account.
For example:

```bash
./WrapCmd activateNodelocked /secure/path/Faceform_Node_License.lic
# Or: ./WrapCmd activateFloating LICENSE_SERVER 7308
```

The production registration stage invokes Wrap headlessly as
`xvfb-run --auto-servernum ./WrapCmd compute <project.wrap>`. A nonzero exit
code, especially exit code 3, must fail the worker job rather than publish a
partial asset.

## External assets

The following licensed or large files are intentionally excluded from Git and
must be provisioned separately:

```text
matting/model/foreground-segmentation-model-vitl16_384.onnx
registration/pretrained/
registration/align/AlbedoMMFitting/data/FLAME2020/generic_model.pkl
registration/wrap/
```

Official references:

- https://docs.astral.sh/uv/concepts/projects/sync/
- https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html
- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- https://colmap.github.io/install.html
- https://faceform.com/download-wrap/
- https://docs.faceform.com/Wrap/CommandLineInterface/CommandLineInterface.html
