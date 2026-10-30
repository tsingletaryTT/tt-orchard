#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Build the three wheels the Qwen3.5-35B-A3B can (config/can.qwen35.toml) ships, from the image the recipe package already pins.

The recipe package (package_src/run.sh) runs the published image `tt-model/qwen3-coder-next:0c170421ca1b` and mounts a patched weight
loader over the image's own. A v6 thin bundle has no image, so the same three things come out of the image as wheels:

  * ttnn:            the tt-metal build in the image. It holds the MoE ops the model needs (`ttnn.experimental.moe_sorted_dispatch` and
                     `moe_gather_add_rows`). The PyPI wheel of ttnn 0.80.0 and the 27B bundles' wheel (0.79.0.dev20260929) do not have them.
  * tt_metal_models: the `models.*` tree of the image, with `weights_qwen35_moe.patch` applied to the autoport's `tt/weights.py`
                     (the recipe applies the same patch at run time). The file hashes are the ones pinned in package_src/pins.env.
  * vllm_tt_plugin:  the vLLM TT plugin installed in the image, with its entry points.

    python3 scripts/qwen35/build_wheels.py wheelhouse            # needs docker and the image; opens no device

The wheels are written with fixed timestamps, so a second run produces the same bytes. Nothing here is uploaded.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
IMAGE = "tt-model/qwen3-coder-next:0c170421ca1b"
VERSION = "0.79.0.dev20261030+qwen35moe"
AUTOPORT = "models/autoports/qwen_qwen3_coder_next"
LOADER = f"{AUTOPORT}/tt/weights.py"
FIXED_TIME = (2026, 10, 30, 0, 0, 0)
SKIP_PARTS = ("__pycache__",)
SKIP_SUFFIXES = (".pyc", ".pyo")


class WheelError(Exception):
    """A pin did not match or a step failed; the message says which."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pins(path: Path) -> dict:
    """KEY=VALUE lines of package_src/pins.env."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z_]+)=(\S+)$", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def pack_wheel(name: str, version: str, files: dict[str, bytes], out_dir: Path, *, requires_python: str = ">=3.10",
               metadata_extra: str = "", entry_points: str | None = None) -> Path:
    """Write a pure-Python wheel. `files` maps a path inside the wheel to its bytes. Returns the wheel's path.

    The archive has fixed timestamps and sorted members, so the same inputs give the same bytes. RECORD lists the sha256 of every member."""
    dist = f"{name.replace('-', '_')}-{version}"
    info = f"{dist}.dist-info"
    members = {p: b for p, b in files.items() if not any(s in p.split("/") for s in SKIP_PARTS) and not p.endswith(SKIP_SUFFIXES)}
    members[f"{info}/METADATA"] = (f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nRequires-Python: {requires_python}\n"
                                   f"{metadata_extra}").encode()
    members[f"{info}/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: tt-orchard qwen35 build_wheels\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    members[f"{info}/top_level.txt"] = ("\n".join(sorted({p.split("/")[0] for p in files if "/" in p})) + "\n").encode()
    if entry_points:
        members[f"{info}/entry_points.txt"] = entry_points.encode()
    record = []
    for path in sorted(members):
        digest = base64.urlsafe_b64encode(hashlib.sha256(members[path]).digest()).rstrip(b"=").decode()
        record.append(f"{path},sha256={digest},{len(members[path])}")
    record.append(f"{info}/RECORD,,")
    members[f"{info}/RECORD"] = ("\n".join(record) + "\n").encode()
    out_dir.mkdir(parents=True, exist_ok=True)
    wheel = out_dir / f"{dist}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(members):
            zi = zipfile.ZipInfo(path, FIXED_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, members[path])
    return wheel


def tree_files(root: Path, prefix: str) -> dict[str, bytes]:
    """Every regular file under `root` as {prefix/relative path: bytes}."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            out[f"{prefix}/{p.relative_to(root).as_posix()}"] = p.read_bytes()
    return out


def apply_loader_patch(files: dict[str, bytes], patch: Path, want_orig: str, want_patched: str) -> None:
    """Apply the recipe's patch to the autoport's loader inside `files`. The file before and after must have the pinned hashes."""
    orig = files.get(LOADER)
    if orig is None:
        raise WheelError(f"the image has no {LOADER}")
    if sha256_bytes(orig) != want_orig:
        raise WheelError(f"the image's loader has sha256 {sha256_bytes(orig)}; the patch was made for {want_orig}")
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "tt").mkdir()
        (t / "tt" / "weights.py").write_bytes(orig)
        subprocess.run(["patch", "-s", "-p1", "-i", str(patch.resolve()), "--no-backup-if-mismatch"], check=True, cwd=t)
        patched = (t / "tt" / "weights.py").read_bytes()
    if sha256_bytes(patched) != want_patched:
        raise WheelError(f"the patched loader has sha256 {sha256_bytes(patched)}; pins.env says {want_patched}")
    files[LOADER] = patched


def docker(*args, check=True):
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True)


def extract_from_image(image: str, dest: Path) -> None:
    """Copy the models tree, the plugin package and its dist-info out of the image's filesystem (the image is not started)."""
    name = f"can-q35-extract-{hashlib.sha256(str(dest).encode()).hexdigest()[:8]}"
    docker("rm", "-f", name, check=False)
    docker("create", "--name", name, image)
    try:
        for src, rel in (("/opt/tt-metal/models", "models"), ("/opt/tt-venv/lib/python3.12/site-packages/vllm_tt_plugin", "vllm_tt_plugin"),
                         ("/opt/tt-venv/lib/python3.12/site-packages/vllm_tt_plugin-0.1.0.dist-info", "plugin-dist-info")):
            docker("cp", f"{name}:{src}", str(dest / rel))
    finally:
        docker("rm", "-f", name, check=False)


TTNN_SCRIPT = """set -e
cd /opt/tt-metal
echo "ttnn built from the tt-metal tree in the qwen3-coder-next image" > README.md
export TT_FROM_PRECOMPILED_DIR=/opt/tt-metal SETUPTOOLS_SCM_PRETEND_VERSION=%s CXX=clang++-20
python setup.py bdist_wheel --dist-dir /out
"""


def build_ttnn(image: str, out_dir: Path) -> Path:
    """Run the image's own setup.py against its prebuilt libraries (TT_FROM_PRECOMPILED_DIR), as root inside a throwaway container."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "build.sh"
        script.write_text(TTNN_SCRIPT % VERSION, encoding="utf-8")
        stage = Path(tmp) / "out"
        stage.mkdir()
        stage.chmod(0o777)
        subprocess.run(["docker", "run", "--rm", "--user", "root", "-v", f"{tmp}:/w:ro", "-v", f"{stage}:/out", "--entrypoint", "/bin/bash",
                        image, "/w/build.sh"], check=True, stdout=subprocess.DEVNULL)
        made = sorted(stage.glob("ttnn-*.whl"))
        if len(made) != 1:
            raise WheelError(f"the ttnn build wrote {[m.name for m in made]}")
        target = out_dir / made[0].name
        shutil.copyfile(made[0], target)
    return target


def build_models_and_plugin(image: str, out_dir: Path, patch: Path, pinned: dict) -> tuple[Path, Path]:
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        extract_from_image(image, dest)
        files = tree_files(dest / "models", "models")
        apply_loader_patch(files, patch, pinned["ORIG_SHA"], pinned["PATCHED_SHA"])
        models = pack_wheel("tt-metal-models", VERSION, files, out_dir,
                            metadata_extra="Summary: the models.* tree of the tt-metal build in the qwen3-coder-next image, with the Qwen3.5-35B-A3B loader patch\n"
                                           "License-Expression: Apache-2.0\n")
        dist = dest / "plugin-dist-info"
        plugin_files = tree_files(dest / "vllm_tt_plugin", "vllm_tt_plugin")
        extra = (dist / "METADATA").read_text(encoding="utf-8")
        # keep the plugin's own metadata (its Requires-Dist lines); the name and version lines are written by pack_wheel
        body = "".join(line for line in extra.splitlines(keepends=True)
                       if not re.match(r"(Metadata-Version|Name|Version|Requires-Python):", line))
        plugin = pack_wheel("vllm_tt_plugin", "0.1.0", plugin_files, out_dir, metadata_extra=body,
                            entry_points=(dist / "entry_points.txt").read_text(encoding="utf-8"))
    return models, plugin


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="directory for the wheels (created)")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--patch", default=str(ROOT / "package_src" / "weights_qwen35_moe.patch"))
    ap.add_argument("--pins", default=str(ROOT / "package_src" / "pins.env"))
    ap.add_argument("--skip-ttnn", action="store_true", help="skip the ttnn build (the slow step)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    pinned = pins(Path(a.pins))
    models, plugin = build_models_and_plugin(a.image, out, Path(a.patch), pinned)
    print(models, plugin)
    if not a.skip_ttnn:
        print(build_ttnn(a.image, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
