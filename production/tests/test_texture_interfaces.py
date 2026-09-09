import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def argument_default(script: Path, option: str):
    tree = ast.parse(script.read_text(encoding="utf-8"))
    flag = f"--{option}"
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant) and node.args[0].value == flag:
            for keyword in node.keywords:
                if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                    return keyword.value.value
            return None
    raise AssertionError(f"{flag} is not declared in {script}")


def test_texture_scripts_accept_explicit_mesh_and_iterations() -> None:
    assert argument_default(PROJECT_ROOT / "texture/render_gbuffer.py", "mesh_path") is None
    assert argument_default(PROJECT_ROOT / "texture/build_texture.py", "mesh_path") is None
    assert argument_default(PROJECT_ROOT / "texture/build_texture.py", "iterations") == 301


def test_texture_builder_preserves_parent_gpu_mapping() -> None:
    script = (PROJECT_ROOT / "texture/build_texture.py").read_text(encoding="utf-8")
    assert 'os.environ.setdefault("CUDA_VISIBLE_DEVICES", opt.device)' in script
