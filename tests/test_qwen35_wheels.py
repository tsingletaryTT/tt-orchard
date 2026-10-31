"""scripts/qwen35/build_wheels.py: the wheel writer and the loader patch step. Nothing here needs docker or the image."""
import base64
import difflib
import hashlib
import importlib.util
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("q35_build_wheels", ROOT / "scripts" / "qwen35" / "build_wheels.py")
bw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bw)


def test_a_wheel_lists_every_member_in_record_with_its_hash_and_is_reproducible(tmp_path):
    files = {"models/__init__.py": b"", "models/a.py": b"x = 1\n", "models/__pycache__/a.cpython-312.pyc": b"junk", "models/data.yaml": b"k: v\n"}
    w1 = bw.pack_wheel("tt-metal-models", "1.0+toy", files, tmp_path / "one")
    w2 = bw.pack_wheel("tt-metal-models", "1.0+toy", files, tmp_path / "two")
    assert w1.read_bytes() == w2.read_bytes()                      # fixed timestamps and sorted members
    z = zipfile.ZipFile(w1)
    names = z.namelist()
    assert not any("__pycache__" in n for n in names)
    info = "tt_metal_models-1.0+toy.dist-info"
    assert {f"{info}/METADATA", f"{info}/WHEEL", f"{info}/RECORD", f"{info}/top_level.txt"} <= set(names)
    record = dict((line.split(",")[0], line.split(",")[1:]) for line in z.read(f"{info}/RECORD").decode().splitlines())
    for n in names:
        if n.endswith("/RECORD"):
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(z.read(n)).digest()).rstrip(b"=").decode()
        assert record[n] == [f"sha256={digest}", str(len(z.read(n)))], n
    assert z.read(f"{info}/top_level.txt").decode().split() == ["models"]
    assert b"Name: tt-metal-models" in z.read(f"{info}/METADATA") and b"Version: 1.0+toy" in z.read(f"{info}/METADATA")


def test_a_plugin_wheel_carries_its_entry_points(tmp_path):
    w = bw.pack_wheel("vllm_tt_plugin", "0.1.0", {"vllm_tt_plugin/__init__.py": b""}, tmp_path,
                      entry_points="[vllm.platform_plugins]\ntt = vllm_tt_plugin.entrypoints:platform_plugin\n")
    z = zipfile.ZipFile(w)
    assert z.read("vllm_tt_plugin-0.1.0.dist-info/entry_points.txt").decode().startswith("[vllm.platform_plugins]")


def toy_patch(orig: str, new: str) -> bytes:
    return "".join(difflib.unified_diff(orig.splitlines(True), new.splitlines(True), "a/tt/weights.py", "b/tt/weights.py")).encode()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


ORIG = "line one\nline two\nline three\n"
NEW = "line one\nline 2\nline three\n"


def test_the_loader_patch_is_applied_when_both_hashes_match(tmp_path):
    patch = tmp_path / "p.patch"
    patch.write_bytes(toy_patch(ORIG, NEW))
    files = {bw.LOADER: ORIG.encode()}
    bw.apply_loader_patch(files, patch, sha(ORIG), sha(NEW))
    assert files[bw.LOADER] == NEW.encode()


def test_a_loader_that_is_not_the_one_the_patch_was_made_for_is_refused(tmp_path):
    patch = tmp_path / "p.patch"
    patch.write_bytes(toy_patch(ORIG, NEW))
    with pytest.raises(bw.WheelError, match="patch was made for"):
        bw.apply_loader_patch({bw.LOADER: b"another file\n"}, patch, sha(ORIG), sha(NEW))


def test_a_patched_loader_with_the_wrong_hash_is_refused(tmp_path):
    patch = tmp_path / "p.patch"
    patch.write_bytes(toy_patch(ORIG, NEW))
    with pytest.raises(bw.WheelError, match="pins.env says"):
        bw.apply_loader_patch({bw.LOADER: ORIG.encode()}, patch, sha(ORIG), sha("something else"))


def test_the_ttnn_build_bundles_the_images_own_sfpi_compiler():
    # the image's kernels do not compile with the newer SFPI a host may have; run.sh prefers the wheel's runtime/sfpi
    assert "BUNDLE_SFPI = True" in bw.TTNN_SCRIPT and "setup.py bdist_wheel" in bw.TTNN_SCRIPT


def test_the_recipes_pins_file_names_both_loader_hashes():
    pins = bw.pins(ROOT / "package_src" / "pins.env")
    assert len(pins["ORIG_SHA"]) == 64 and len(pins["PATCHED_SHA"]) == 64 and pins["ORIG_SHA"] != pins["PATCHED_SHA"]
