# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fetch a model snapshot into the operator's Hugging Face cache before a run starts.

The supervisor never downloads weights, and agents run with HF_HUB_OFFLINE=1, so `tt-orchard bringup` makes
sure the snapshot is on disk first. The download is `hf download` run as a subprocess, pinned to the
revision the preflight looked at, and it resumes where an earlier attempt stopped.

The harness never uses the operator's Hugging Face token. The subprocess gets an environment with every
token variable removed, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, and a token path that does not exist (the default
token file lives inside HF_HOME, which may be the operator's own cache). A gated or private model therefore
fails here, and the preflight has already refused it with a clear reason.

A file in the model repo is only downloaded. Nothing in this module imports or runs it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

NO_TOKEN_PATH = "/nonexistent/tt-orchard-no-token"
STRIPPED_VARIABLES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH", "HF_HUB_OFFLINE")
ERROR_TAIL_LINES = 12


class FetchError(RuntimeError):
    """The snapshot could not be fetched or is incomplete."""


def snapshot_dir(hf_home, model_id: str, revision: str) -> Path:
    org, name = model_id.split("/", 1)
    return Path(hf_home) / "hub" / f"models--{org}--{name}" / "snapshots" / revision


def clean_env(env: dict, hf_home) -> dict:
    out = {k: v for k, v in env.items() if k not in STRIPPED_VARIABLES}
    out.update(HF_HOME=str(hf_home), HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_TOKEN_PATH=NO_TOKEN_PATH)
    return out


def _missing(snap: Path, files: list[str]) -> list[str]:
    return [f for f in files if not (snap / f).exists()]


def fetch_snapshot(model_id: str, revision: str, hf_home, *, files: list[str], run=subprocess.run,
                   which=shutil.which, env: dict | None = None) -> Path:
    """Make `files` of `model_id` at `revision` exist under `hf_home` and return the snapshot directory."""
    snap = snapshot_dir(hf_home, model_id, revision)
    if not _missing(snap, files):
        return snap
    exe = which("hf")
    if not exe:
        raise FetchError("the `hf` command is not installed (pip install huggingface_hub), so the model "
                         "cannot be downloaded. Download it yourself, then run bringup again")
    argv = [exe, "download", model_id, "--revision", revision, "--cache-dir", str(Path(hf_home) / "hub"),
            "--quiet"]
    done = run(argv, env=clean_env(os.environ if env is None else env, hf_home), capture_output=True,
               text=True)
    if done.returncode != 0:
        tail = "\n".join((done.stderr or "").strip().splitlines()[-ERROR_TAIL_LINES:])
        raise FetchError(f"`hf download {model_id}` exited {done.returncode}:\n{tail}")
    missing = _missing(snap, files)
    if missing:
        raise FetchError(f"the download finished but {len(missing)} files are missing from {snap}: "
                         + ", ".join(missing[:5]))
    return snap
