#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Build a second reference interpreter with another transformers, for a model type the installed one lacks.

    python3 scripts/reference_venv.py 'transformers==5.23.0' [--arch Qwen4ForCausalLM] [--venv-dir DIR] [--dry-run]

Stage 1 runs the CPU reference with `reference_python` (config/bringup.toml), and agents may not install packages. When
`tt-orchard drop-check` says a release needs a newer transformers (a new model_type such as `qwen4`), the operator runs this:
it builds a new venv beside the existing one (default ~/.local/share/tt-orchard/venvs/reference-<version>) with the same steps
as `tt-orchard setup` (CPU torch from the PyTorch CPU index, then transformers, tokenizers, safetensors, pillow and numpy),
checks that the named architecture class imports, and prints the line to put in bringup.toml. It never touches an existing
directory, so the current reference interpreter stays as it is.

The transformers spec must be a version pin (`transformers==X`, `transformers>=X`) or a git commit of
github.com/huggingface/transformers; anything else is refused before a command runs.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"      # the same index orchard/setup_machine.py uses
EXTRA_PACKAGES = ("tokenizers", "safetensors", "pillow", "numpy")
VERSION_SPEC = re.compile(r"transformers(==|>=)[0-9]+(\.[0-9]+){0,3}([a-z]+[0-9]*)?")
GIT_SPEC = re.compile(r"transformers @ git\+https://github\.com/huggingface/transformers@[0-9a-f]{7,40}")
ARCH_NAME = re.compile(r"[A-Z][A-Za-z0-9]{1,80}")


def spec_problem(spec) -> str | None:
    if not isinstance(spec, str) or not (VERSION_SPEC.fullmatch(spec) or GIT_SPEC.fullmatch(spec)):
        return ("the spec must be transformers==X, transformers>=X, or "
                "'transformers @ git+https://github.com/huggingface/transformers@<commit>'")
    return None


def default_venv_dir(spec: str, home: Path | None = None) -> Path:
    tail = re.sub(r"[^0-9A-Za-z.]+", "-", spec.split("@")[-1] if "@" in spec else spec.split("=")[-1]).strip("-")[:40]
    return (home or Path.home()) / ".local" / "share" / "tt-orchard" / "venvs" / f"reference-{tail or 'new'}"


def commands(spec: str, venv: Path, uv: str | None, arch: str | None) -> list[tuple[str, list[str]]]:
    py = str(venv / "bin" / "python")
    if uv:
        steps = [("create the venv", [uv, "venv", str(venv), "--python", "3.12"]),
                 ("install CPU torch", [uv, "pip", "install", "--python", py, "--index-url", TORCH_CPU_INDEX, "torch"]),
                 ("install transformers", [uv, "pip", "install", "--python", py, spec, *EXTRA_PACKAGES])]
    else:
        steps = [("create the venv", [sys.executable, "-m", "venv", str(venv)]),
                 ("install CPU torch", [py, "-m", "pip", "install", "--index-url", TORCH_CPU_INDEX, "torch"]),
                 ("install transformers", [py, "-m", "pip", "install", spec, *EXTRA_PACKAGES])]
    steps.append(("check the imports", [py, "-c", "import torch, transformers, tokenizers, safetensors; "
                                                  "print(transformers.__version__)"]))
    if arch:
        steps.append((f"check the class {arch}", [py, "-c", f"import transformers; getattr(transformers, {arch!r})"]))
    return steps


def real_run(argv) -> tuple[int, str, str]:
    done = subprocess.run(argv, capture_output=True, text=True)
    return done.returncode, done.stdout, done.stderr


def main(argv=None, run=real_run) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("spec", help="transformers==X, transformers>=X, or 'transformers @ git+https://github.com/huggingface/transformers@<commit>'")
    p.add_argument("--arch", help="an architecture class the new transformers must have, e.g. Qwen4ForCausalLM")
    p.add_argument("--venv-dir", type=Path, help="where to build it (default ~/.local/share/tt-orchard/venvs/reference-<version>)")
    p.add_argument("--uv", default=shutil.which("uv"), help="the uv command (default: found on PATH; without uv, venv and pip)")
    p.add_argument("--dry-run", action="store_true", help="print the commands and run nothing")
    args = p.parse_args(argv)
    problem = spec_problem(args.spec)
    if problem:
        print(f"refused: {problem}", file=sys.stderr)
        return 2
    if args.arch and not ARCH_NAME.fullmatch(args.arch):
        print("refused: --arch must be a class name", file=sys.stderr)
        return 2
    venv = args.venv_dir or default_venv_dir(args.spec)
    steps = commands(args.spec, venv, args.uv, args.arch)
    if args.dry_run:
        for what, argv_ in steps:
            print(f"# {what}\n" + " ".join(argv_))
        return 0
    if venv.exists() or venv.is_symlink():
        print(f"refused: {venv} exists; this script never changes an existing directory", file=sys.stderr)
        return 2
    for what, argv_ in steps:
        code, out, err = run(argv_)
        if code != 0:
            print(f"failed at '{what}' (exit {code}): {err.strip()[-2000:]}", file=sys.stderr)
            return 1
        if what == "check the imports":
            print(f"transformers {out.strip()}")
    print("built. Put this in config/bringup.toml, then run tt-orchard drop-check again:")
    print(f'reference_python = "{venv}/bin/python"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
