from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]


def load_pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_uv_manifest_pins_the_tested_linux_cuda_environment() -> None:
    config = load_pyproject()
    project = config["project"]
    dependencies = set(project["dependencies"])

    assert project["requires-python"] == ">=3.10,<3.11"
    assert config["tool"]["uv"]["package"] is False
    assert config["tool"]["uv"]["required-environments"] == [
        "sys_platform == 'linux' and platform_machine == 'x86_64'"
    ]
    assert (
        "torch @ https://download.pytorch.org/whl/cu121/"
        "torch-2.3.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
    ) in dependencies
    assert (
        "pytorch3d @ https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/"
        "py310_cu121_pyt231/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl"
    ) in dependencies
    assert "numpy==1.26.4" in dependencies
    assert "pip==25.3" in dependencies
    assert "pymeshlab==2025.7" in dependencies
    assert "xatlas==0.0.11" in dependencies
    assert "wheel==0.48.0" in dependencies
    assert config["tool"]["uv"]["override-dependencies"] == [
        "protobuf==3.20.3"
    ]


def test_uv_manifest_pins_native_git_dependencies_and_build_mode() -> None:
    config = load_pyproject()
    sources = config["tool"]["uv"]["sources"]
    no_isolation = set(config["tool"]["uv"]["no-build-isolation-package"])
    extra_build = config["tool"]["uv"]["extra-build-dependencies"]

    assert sources["tinycudann"]["rev"] == (
        "749dd70c5afc5a9dadb85e5652ed65d55e0ba187"
    )
    assert sources["diff-surfel-rasterization"]["rev"] == (
        "e0ed0207b3e0669960cfad70852200a4a5847f61"
    )
    assert sources["simple-knn"]["rev"] == (
        "f155ec04131cb579f53443a06879d37115f4612f"
    )
    assert sources["ibug-face-detection"]["lfs"] is True
    assert {
        "chumpy",
        "tinycudann",
        "diff-surfel-rasterization",
        "simple-knn",
    } <= no_isolation
    assert extra_build["chumpy"] == ["pip==25.3", "wheel==0.48.0"]
    assert "pytest>=8,<9" in config["dependency-groups"]["dev"]


def test_agents_documents_non_python_production_dependencies() -> None:
    instructions = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    instruction_lines = instructions.splitlines()

    for heading in (
        "## Python dependencies with uv",
        "## NVIDIA driver and CUDA Toolkit",
        "## CUDA-enabled COLMAP",
        "## Faceform Wrap",
    ):
        assert heading in instructions
    for verification_command in (
        "uv sync --frozen",
        "nvidia-smi",
        "/usr/bin/nvcc --version",
        "/usr/local/cuda-12.4/bin/nvcc --version",
        "colmap -h",
        "./WrapCmd --version",
        "./WrapCmd --license",
    ):
        assert verification_command in instructions
    assert "export CUDA_HOME=/usr" in instruction_lines
    assert instructions.index("export CUDA_HOME=/usr") < instructions.index(
        "uv sync --frozen"
    )


def test_uv_environment_is_ignored_by_git() -> None:
    ignored_paths = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".venv/" in ignored_paths
