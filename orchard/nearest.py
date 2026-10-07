# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Find the model a new model is compared with.

Stage 0 compares the new model with a model that already runs on Tenstorrent hardware, and later stages
serve the new weights with that model's installed tt-model bundle. Both must exist on the machine before a
run starts, or an agent spends hours looking for them (the humanizer run, a Gemma 4 fine-tune, ended that
way on a machine that only had Qwen bundles).

This module holds the parts that touch the outside world: reading the installed bundles' manifests, asking
the official `tt` CLI to search for bundles (`tt model search`, falling back to `tt-model search`) and
installing one with `tt-model pull`. The decision itself is `preflight.check_base`.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

SEARCH_TIMEOUT_S = 90
PULL_TIMEOUT_S = 3600
ERROR_TAIL_LINES = 8


class Installed(NamedTuple):
    name: str            # org/name of the bundle, from its directory
    weights_repo: str    # the Hugging Face repo its manifest serves
    schema: str          # "6" for a thin bundle, "5.1" for a container package
    chips: int


def family_query(model_id: str) -> str:
    """The word to search bundles for: the letters at the start of the repo name (gemma-4-12B -> gemma)."""
    name = model_id.split("/")[-1].lower()
    m = re.match(r"[a-z]+", name)
    return m.group(0) if m else name


def read_installed(models_root) -> list[Installed]:
    """Every installed bundle under `models_root` (<org>/<name>/tt_kernel_manifest.json). A manifest that
    cannot be read is skipped."""
    out = []
    for mf in sorted(Path(models_root).expanduser().glob("*/*/tt_kernel_manifest.json")):
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            repo = m["weights"]["repo_id"]
            out.append(Installed(f"{mf.parent.parent.name}/{mf.parent.name}", str(repo),
                                 str(m.get("schema_version", "")), int(m.get("device_count") or 0)))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return out


def parse_search(text: str) -> list[dict] | None:
    """[{"name", "installed"}] from `tt model search --json` ({"bundles": [...]}) or `tt-model search
    --json` ([{"id": ...}]). None when the text is neither."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if isinstance(data, dict):
        items = data.get("bundles")
        key = "name"
    else:
        items, key = data, "id"
    if not isinstance(items, list):
        return None
    out = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get(key), str):
            inst = it.get("installed")
            out.append({"name": it[key], "installed": inst if isinstance(inst, bool) else None})
    return out


def search(query: str, *, run=subprocess.run, which=shutil.which) -> list[dict] | None:
    """Published bundles matching `query`, or None when no search could be made. The official `tt` CLI is
    asked first, because it owns this question; `tt-model search` is the fallback."""
    attempts = []
    if which("tt"):
        attempts.append([which("tt"), "model", "search", query, "--json"])
    if which("tt-model"):
        attempts.append([which("tt-model"), "search", query, "--json"])
    for argv in attempts:
        try:
            done = run(argv, capture_output=True, text=True, timeout=SEARCH_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0:
            got = parse_search(done.stdout)
            if got is not None:
                return got
    return None


def pull(bundle: str, *, run=subprocess.run, which=shutil.which) -> tuple[bool, str]:
    """Install `bundle` with `tt-model pull`. Returns (ok, the end of the error text when not ok)."""
    exe = which("tt-model")
    if not exe:
        return False, "the `tt-model` command is not installed"
    try:
        done = run([exe, "pull", bundle], capture_output=True, text=True, timeout=PULL_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if done.returncode == 0:
        return True, ""
    tail = "\n".join((done.stderr or done.stdout or "").strip().splitlines()[-ERROR_TAIL_LINES:])
    return False, tail or f"exit {done.returncode}"
