"""The vLLM version a vllm-kind can declares must be the vLLM of the image its wheels come from (issue 507).

config/can.qwen35.toml ships a plugin wheel, a ttnn wheel and a models wheel that scripts/qwen35/build_wheels.py cuts from one container
image. `vllm_version` in the spec goes to package-thin's `--vllm-version` and selects the boot check's quirk notes. When the vLLM
release moves, the version and the wheels have to move together: changing only the version string would label a 0.26.0 plugin as 0.31.0.
build_wheels.py records the vLLM version of its default image in IMAGE_VLLM_VERSION, and this test compares the two.
"""
import importlib.util
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("q35_build_wheels_pin", ROOT / "scripts" / "qwen35" / "build_wheels.py")
bw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bw)


def _can_vllm_version() -> str:
    return tomllib.loads((ROOT / "config" / "can.qwen35.toml").read_text(encoding="utf-8"))["can"]["vllm_version"]


def test_the_qwen35_can_names_the_vllm_version_of_the_image_its_wheels_come_from():
    assert _can_vllm_version() == bw.IMAGE_VLLM_VERSION


def test_the_image_vllm_version_is_a_release_number():
    parts = bw.IMAGE_VLLM_VERSION.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)
