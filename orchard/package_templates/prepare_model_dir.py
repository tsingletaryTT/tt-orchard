#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Build model-dir/ for this bundle before vLLM starts. run.sh runs it with the bundle's python.

Written by tt-orchard (stage 7) for a model whose weights are a fine-tune of a supported model.
The model code in this bundle is registered for the base model's architecture, so vLLM must see
the base model's config files. The tokenizer and weights must be the fine-tune's. This script
builds model-dir/ next to itself from both:

- COPIED from base_config/ (shipped in the bundle): every file there, for example config.json.
- LINKED from the fine-tune's Hugging Face snapshot (absolute links to the resolved files):
  tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json,
  model.safetensors.index.json and every *.safetensors file. tokenizer.json and at least one
  *.safetensors file are required.

The snapshot is the manifest's `weights.repo_id` at `weights.revision` (which must be a pinned
40-character commit) under $HF_HUB_CACHE or $HF_HOME/hub (run.sh sets HF_HOME). When it is missing
and HF_HUB_OFFLINE is not set, it is downloaded with huggingface_hub from the bundle's venv. When it
is missing and HF_HUB_OFFLINE is set, the script exits 2 and names it.

model-dir/.weights records "<repo>@<revision>". A model-dir without that file was not built here,
and the script exits 2 without touching it. Exit 0 means model-dir is ready.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINKED = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
          "model.safetensors.index.json")
MARKER = ".weights"


def fail(message: str) -> None:
    print(f"prepare_model_dir: {message}", file=sys.stderr)
    sys.exit(2)


def snapshot(repo: str, rev: str) -> Path:
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    org, name = repo.split("/", 1)
    snap = Path(hub) / f"models--{org}--{name}" / "snapshots" / rev
    if snap.is_dir():
        return snap
    if os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on"):
        fail(f"the weights snapshot {repo}@{rev} is not in {hub} and HF_HUB_OFFLINE is set. "
             f"Download it first: hf download {repo} --revision {rev}")
    from huggingface_hub import snapshot_download      # in the bundle's venv
    return Path(snapshot_download(repo_id=repo, revision=rev))


def main() -> int:
    weights = json.loads((HERE / "tt_kernel_manifest.json").read_text(encoding="utf-8"))["weights"]
    repo, rev = weights["repo_id"], weights.get("revision") or ""
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        fail(f"the manifest's weights revision {rev!r} is not a 40-character commit")
    md = HERE / "model-dir"
    if md.is_symlink() or (md.exists() and not (md / MARKER).is_file()):
        fail(f"{md} exists and was not built by this script; move it aside")
    snap = snapshot(repo, rev)
    names = list(LINKED) + sorted(p.name for p in snap.glob("*.safetensors"))
    present = [n for n in names if (snap / n).exists()]
    if "tokenizer.json" not in present or not any(n.endswith(".safetensors") for n in present):
        fail(f"{snap} needs tokenizer.json and at least one *.safetensors file")
    if md.exists():
        shutil.rmtree(md)
    md.mkdir()
    for f in sorted((HERE / "base_config").iterdir()):
        shutil.copyfile(f, md / f.name)
    for n in present:
        os.symlink(os.path.realpath(snap / n), md / n)
    (md / MARKER).write_text(f"{repo}@{rev}", encoding="utf-8")
    print(f"prepare_model_dir: {md} holds the base config and {len(present)} files of {repo}@{rev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
