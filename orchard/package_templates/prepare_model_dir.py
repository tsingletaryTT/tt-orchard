#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Build model-dir/ for this bundle before vLLM starts. run.sh runs it with the bundle's python.

Written by tt-orchard (stage 7) for a model whose weights are a fine-tune of a supported model.
The model code in this bundle is registered for the base model's architecture, so vLLM must see
the base model's config files. The tokenizer and weights must be the fine-tune's. This script
builds model-dir/ next to itself from both:

- COPIED from base_config/ (shipped in the bundle): every file there, for example config.json. When the
  fine-tune ships no generation_config.json, base_config/ holds the base model's, and the server needs
  its eos_token_id.
- LINKED from the fine-tune's Hugging Face snapshot (absolute links to the resolved files):
  tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json,
  model.safetensors.index.json and every *.safetensors file. tokenizer.json and at least one
  *.safetensors file are required.

The snapshot is the manifest's `weights.repo_id` at `weights.revision` (which must be a pinned
40-character commit) under $HF_HUB_CACHE or $HF_HOME/hub (run.sh sets HF_HOME). When it is missing
and HF_HUB_OFFLINE is not set, it is downloaded with huggingface_hub from the bundle's venv. When it
is missing and HF_HUB_OFFLINE is set, the script exits 2 and names it.

When the package has a `graft.json`, the model has no mtp.* tensors and the package serves it with the speculative
drafter on. This script then also adds the nearest model's mtp.* tensors (the drafter's MTP head) to model-dir/ as
one more weight file, named by graft.json's `shard`. They are read from the nearest model's pinned snapshot
(`repo` at `revision`, only the file `file`). When that file is missing and HF_HUB_OFFLINE is not set, it is
downloaded alone with huggingface_hub; when HF_HUB_OFFLINE is set, the script exits 2 and names it. The shard is
built from the file's bytes as they are, in tensor-name order, and its sha256 must equal graft.json's: the
package was tested with that shard. A listed tensor that is missing or not BF16, or a different sha256, exits 2
and removes model-dir/. The model's own weight index is replaced in model-dir/ by a copy that also lists the
grafted tensors; the cached index it was linked to is never written through. The package ships no weights.

model-dir/.weights records "<repo>@<revision>". A model-dir without that file was not built here,
and the script exits 2 without touching it. Exit 0 means model-dir is ready.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINKED = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
          "model.safetensors.index.json")
MARKER = ".weights"
GRAFT = "graft.json"


def fail(message: str) -> None:
    print(f"prepare_model_dir: {message}", file=sys.stderr)
    sys.exit(2)


def hub_dir() -> str:
    return os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")


def offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on")


def snapshot(repo: str, rev: str) -> Path:
    hub = hub_dir()
    org, name = repo.split("/", 1)
    snap = Path(hub) / f"models--{org}--{name}" / "snapshots" / rev
    if snap.is_dir():
        return snap
    if offline():
        fail(f"the weights snapshot {repo}@{rev} is not in {hub} and HF_HUB_OFFLINE is set. "
             f"Download it first: hf download {repo} --revision {rev}")
    from huggingface_hub import snapshot_download      # in the bundle's venv
    return Path(snapshot_download(repo_id=repo, revision=rev))


def read_shard_header(path: Path) -> tuple[dict, int]:
    """(the safetensors header, the file offset where tensor data starts). Exits 2 when it cannot be read."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            (n,) = struct.unpack("<Q", raw)
            if not 0 < n <= 100 * 1024 * 1024:
                raise ValueError("header length")
            header = json.loads(f.read(n).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, struct.error) as exc:
        fail(f"cannot read the safetensors header of {path.name}: {exc}")
    if not isinstance(header, dict):
        fail(f"the safetensors header of {path.name} is not an object")
    return header, 8 + n


def graft_source(g: dict) -> Path:
    """The nearest model's shard that holds the mtp.* tensors, from its pinned snapshot, or downloaded alone."""
    hub = hub_dir()
    org, name = g["repo"].split("/", 1)
    path = Path(hub) / f"models--{org}--{name}" / "snapshots" / g["revision"] / g["file"]
    if path.exists():
        return path
    if offline():
        fail(f"the file {g['file']} of {g['repo']}@{g['revision']} (the drafter's MTP head) is not in {hub} and "
             f"HF_HUB_OFFLINE is set. Download it first: hf download {g['repo']} {g['file']} --revision {g['revision']}")
    from huggingface_hub import hf_hub_download      # in the bundle's venv; this file only
    return Path(hf_hub_download(repo_id=g["repo"], filename=g["file"], revision=g["revision"]))


def build_graft_shard(g: dict, md: Path) -> None:
    """Write md/<shard> from the nearest model's mtp.* tensors, copying their bytes as they are."""
    src = graft_source(g)
    header, start = read_shard_header(src)
    size = src.stat().st_size
    entries = {}
    for name in sorted(g["tensors"]):
        e = header.get(name)
        if not isinstance(e, dict):
            fail(f"{name} is not in {g['file']} of {g['repo']}@{g['revision']}")
        if e.get("dtype") != "BF16":
            fail(f"{name} is {e.get('dtype')}; the graft copies BF16 tensors only")
        try:
            a, b = e["data_offsets"]
            elements = 1
            for d in e["shape"]:
                elements *= d
        except (KeyError, TypeError, ValueError):
            fail(f"{name} has a header entry the graft cannot read")
        if not (0 <= a <= b and start + b <= size and b - a == 2 * elements):
            fail(f"{name}'s header entry does not match the bytes in {g['file']}")
        entries[name] = e
    out_header, off = {"__metadata__": {"format": "pt"}}, 0
    for name, e in entries.items():
        n = e["data_offsets"][1] - e["data_offsets"][0]
        out_header[name] = {"dtype": "BF16", "shape": e["shape"], "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(out_header).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    digest, dest = hashlib.sha256(), md / g["shard"]
    # The destination must be a new file. model-dir holds links into the model's cached blobs, and opening one of
    # them for writing would overwrite the blob. O_EXCL refuses an existing file or link; this script never unlinks
    # one to make room, because it cannot tell a link it made from a file a person left there.
    try:
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    except FileExistsError:
        fail(f"{dest} already exists; the graft shard is only ever written to a new file, never through a link")
    with open(src, "rb") as f, os.fdopen(fd, "wb") as out:
        for chunk in (struct.pack("<Q", len(raw)), raw):
            out.write(chunk)
            digest.update(chunk)
        for name, e in entries.items():
            a, b = e["data_offsets"]
            f.seek(start + a)
            left = b - a
            while left:
                chunk = f.read(min(left, 1 << 20))
                if not chunk:
                    fail(f"{g['file']} ended while copying {name}")
                out.write(chunk)
                digest.update(chunk)
                left -= len(chunk)
    if digest.hexdigest() != g["sha256"]:
        fail(f"the shard built from {g['file']} of {g['repo']}@{g['revision']} has sha256 {digest.hexdigest()}, but "
             f"graft.json says {g['sha256']}: that is not the file this package was tested with")
    index = md / "model.safetensors.index.json"
    if os.path.lexists(index):
        try:
            data = json.loads(index.read_text(encoding="utf-8"))          # follows the link
            weight_map = data["weight_map"]
        except (OSError, ValueError, KeyError, TypeError):
            fail("the model's model.safetensors.index.json cannot be read, so the graft cannot add to it")
        if not isinstance(weight_map, dict):
            fail("the model's model.safetensors.index.json has no weight_map object")
        weight_map.update({n: g["shard"] for n in entries})
        meta = data.setdefault("metadata", {})
        if isinstance(meta, dict) and isinstance(meta.get("total_size"), int):
            meta["total_size"] += off
        os.unlink(index)               # the link, not the cached file it points to
        index.write_text(json.dumps(data, indent=1), encoding="utf-8")


def read_graft() -> dict | None:
    path = HERE / GRAFT
    if not path.is_file():
        return None
    try:
        g = json.loads(path.read_text(encoding="utf-8"))
        ok = (isinstance(g["repo"], str) and re.fullmatch(r"[0-9a-f]{40}", g["revision"]) and isinstance(g["file"], str)
              and g["file"] and isinstance(g["shard"], str) and g["shard"] and isinstance(g["sha256"], str)
              and isinstance(g["tensors"], list) and g["tensors"] and "/" in g["repo"])
    except (OSError, ValueError, KeyError, TypeError):
        ok = False
    if not ok:
        fail(f"{path} is not a graft record (repo, revision, file, shard, tensors, sha256)")
    return g


def main() -> int:
    weights = json.loads((HERE / "tt_kernel_manifest.json").read_text(encoding="utf-8"))["weights"]
    repo, rev = weights["repo_id"], weights.get("revision") or ""
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        fail(f"the manifest's weights revision {rev!r} is not a 40-character commit")
    md = HERE / "model-dir"
    if md.is_symlink() or (md.exists() and not (md / MARKER).is_file()):
        fail(f"{md} exists and was not built by this script; move it aside")
    graft = read_graft()
    snap = snapshot(repo, rev)
    if graft and os.path.lexists(snap / graft["shard"]):
        fail(f"the model {repo}@{rev} ships a file named {graft['shard']}, the name this package uses for the "
             "grafted shard; the package cannot be started with that model's files")
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
        if os.path.lexists(md / n):         # the fine-tune's own file replaces a base copy of it
            os.unlink(md / n)
        os.symlink(os.path.realpath(snap / n), md / n)
    (md / MARKER).write_text(f"{repo}@{rev}", encoding="utf-8")
    if graft:
        try:
            build_graft_shard(graft, md)
        except SystemExit:
            shutil.rmtree(md, ignore_errors=True)         # a model-dir without its head would not start
            raise
    print(f"prepare_model_dir: {md} holds the base config and {len(present)} files of {repo}@{rev}")
    if graft:
        print(f"prepare_model_dir: added {len(graft['tensors'])} mtp.* tensors of {graft['repo']}@{graft['revision']} "
              f"as {graft['shard']} (sha256 {graft['sha256'][:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
