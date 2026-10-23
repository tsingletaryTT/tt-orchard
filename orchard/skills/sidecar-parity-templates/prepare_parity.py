#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Build the launcher and the model directory for the sidecar parity check.

The sidecar-parity skill copies this file into the stage directory and runs it. It reads
`parity_config.json` from its own directory and writes two things next to itself:

1. `parity-model-dir/` (not `model-dir/`: prepare_swap.py builds that name in the same stage directory, and the
   two directories differ: the swap server needs the nearest model's processor files and this one does not),
   a clean directory for the Qwen3.8 TT model code to load weights from. It holds
   - a COPY of the new model's own config.json (the model is loaded directly, not through vLLM, so the
     nearest model's config is not needed), and
   - SYMLINKS (absolute, fully resolved) to the tokenizer files, model.safetensors.index.json and every
     model-*.safetensors shard.
   The sidecar files (the head's weights and config) and the model repo's code and README are left out,
   so the TT loader cannot read them by accident. `parity-model-dir-sources.json` beside it lists what was
   copied, linked and absent. A snapshot without a generation_config.json is a note: hidden_parity.py loads
   weights and runs the layers and never generates, so nothing here reads that file.
2. `parity-run.sh`, a copy of `<bundle_dir>/run.sh` with these edits:
   - the `HERE=...` line becomes `HERE="<bundle_dir>"` (exactly once), so the copy still finds the
     bundle's venv and model code;
   - the final `exec "${CMD[@]}"` line becomes `exec "$PYBIN" "<this directory>/hidden_parity.py" "$@"`
     (exactly once), so the script runs in the bundle's python with the environment the server uses;
   - `export QWEN36_DRAFTER=...` becomes `export QWEN36_DRAFTER=""` (at most once): no drafter is loaded;
   - just before the exec line these are exported, so they come after the bundle's own lines and win:
     `QWEN36_DRAFTER=""`, `HF_MODEL` and `MODEL_WEIGHTS_DIR` set to `<parity-model-dir>` (the TT runtime takes
     its weights directory from MODEL_WEIGHTS_DIR, then HF_MODEL; the bundle sets HF_MODEL to the nearest
     model's id), `TT_CACHE_PATH` and `TT_CACHE_HOME` set to `tt_cache`, and `HF_HUB_OFFLINE=1`.
   - `export QWEN36_DRAFTER=...` in the bundle's own lines becomes empty too (at most once), so a reader
     of the launcher does not see a drafter named.
   Everything else in the script, including its other environment lines, stays as it is.

An expected edit that does not happen exactly once exits 2 with a message that names it, and nothing is
written. Config keys read here: bundle_dir, model_snapshot, tt_cache. The others in parity_config.json are
for hidden_parity.py.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE_DIR = Path(__file__).resolve().parent
COPIED = ("config.json",)
LINKED = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
          "model.safetensors.index.json")
REQUIRED = ("config.json", "tokenizer.json")
SHARD_GLOB = "model-*.safetensors"        # the backbone's shards; joint_head.safetensors does not match
EXEC_LINE = re.compile(r'^exec[ \t]+"\$\{CMD\[@\]\}"[ \t]*$', re.MULTILINE)


def fail(message: str) -> None:
    print(f"prepare_parity: {message}", file=sys.stderr)
    sys.exit(2)


def load_config() -> dict:
    path = HERE_DIR / "parity_config.json"
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"cannot read {path}: {exc}")
    missing = [k for k in ("bundle_dir", "model_snapshot", "tt_cache") if not cfg.get(k)]
    if missing:
        fail(f"parity_config.json is missing {missing}")
    return cfg


def check_snapshot(snap: Path) -> None:
    for name in REQUIRED:
        if not (snap / name).exists():
            fail(f"{name} is not in model_snapshot {snap}")
    if not list(snap.glob(SHARD_GLOB)):
        fail(f"model_snapshot {snap} has no {SHARD_GLOB} file")


def build_model_dir(snap: Path) -> tuple[Path, list[str], list[str], list[str]]:
    """Make parity-model-dir from scratch. Returns (path, copied names, linked names, absent names): the absent ones
    are the optional files of LINKED that the snapshot does not have."""
    md = HERE_DIR / "parity-model-dir"
    if md.is_symlink():
        fail(f"{md} is a symlink; remove it so this script can build a real directory")
    if md.exists():
        shutil.rmtree(md)         # only ever a directory this script built: a copy and symlinks
    md.mkdir()
    copied, linked, absent = [], [], []
    for name in COPIED:
        shutil.copyfile(snap / name, md / name)          # follows the snapshot link to the blob
        copied.append(name)
    for name in list(LINKED) + sorted(p.name for p in snap.glob(SHARD_GLOB)):
        if (snap / name).exists():
            os.symlink(os.path.realpath(snap / name), md / name)
            linked.append(name)
        else:
            absent.append(name)
    return md, copied, linked, absent


def write_sources(copied: list[str], linked: list[str], absent: list[str]) -> Path:
    """parity-model-dir-sources.json, beside the model dir (a file inside it would sit where the TT loader looks).
    It says which files the check loaded. hidden_parity.py loads weights and runs the layers and never generates, so
    nothing reads a generation_config.json here and a missing one is a note, not an error."""
    rec = {"schema": 1, "copied": copied, "linked": linked, "absent": absent,
           "generation_config": "model" if "generation_config.json" in linked else "absent"}
    path = HERE_DIR / "parity-model-dir-sources.json"
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return path


def edit_run_script(text: str, bundle: Path, script: Path, model_dir: Path, cache: str) -> tuple[str, list[str]]:
    """Apply the edits to run.sh's text. Returns (new text, notes about optional edits)."""
    text, n = re.subn(r"^HERE=.*$", lambda m: f'HERE="{bundle}"', text, flags=re.MULTILINE)
    if n != 1:
        fail(f"expected exactly one HERE= line in {bundle / 'run.sh'}, found {n}")
    notes = []
    text, n = re.subn(r"^([ \t]*)export[ \t]+QWEN36_DRAFTER=.*$",
                      lambda m: f'{m.group(1)}export QWEN36_DRAFTER=""', text, flags=re.MULTILINE)
    if n > 1:
        fail(f"expected at most one export QWEN36_DRAFTER= line in {bundle / 'run.sh'}, found {n}")
    if n == 0:
        notes.append("export QWEN36_DRAFTER=... was absent; the launcher sets it empty before the exec")
    found = len(EXEC_LINE.findall(text))
    if found != 1:
        fail(f'expected exactly one exec "${{CMD[@]}}" line in {bundle / "run.sh"}, found {found}')
    before_exec = (f'export QWEN36_DRAFTER=""\n'
                   f'export HF_MODEL="{model_dir}"\n'
                   f'export MODEL_WEIGHTS_DIR="{model_dir}"\n'
                   f'export TT_CACHE_PATH="{cache}"\n'
                   f'export TT_CACHE_HOME="{cache}"\n'
                   f"export HF_HUB_OFFLINE=1\n"
                   f'exec "$PYBIN" "{script}" "$@"')
    text = EXEC_LINE.sub(lambda m: before_exec, text)
    return text, notes


def main() -> int:
    cfg = load_config()
    bundle, snap = Path(cfg["bundle_dir"]).resolve(), Path(cfg["model_snapshot"])
    for label, d in (("bundle_dir", bundle), ("model_snapshot", snap)):
        if not d.is_dir():
            fail(f"{label} {d} is not a directory")
    src_run = bundle / "run.sh"
    if not src_run.is_file():
        fail(f"{src_run} does not exist")
    model_dir = HERE_DIR / "parity-model-dir"
    # Edit in memory and check the snapshot first, so a failure leaves no launcher and no parity-model-dir.
    text, notes = edit_run_script(src_run.read_text(encoding="utf-8"), bundle, HERE_DIR / "hidden_parity.py",
                                  model_dir, cfg["tt_cache"])
    check_snapshot(snap)
    model_dir, copied, linked, absent = build_model_dir(snap)
    write_sources(copied, linked, absent)
    out = HERE_DIR / "parity-run.sh"
    out.write_text(text, encoding="utf-8")
    out.chmod(out.stat().st_mode | 0o111)
    print(f"parity-model-dir: {model_dir}")
    print(f"  copied: {', '.join(copied)}")
    print(f"  linked: {len(linked)} files ({', '.join(linked)})")
    print(f"parity-run.sh: {out}")
    print(f'  HERE="{bundle}"; exec hidden_parity.py; HF_MODEL and MODEL_WEIGHTS_DIR={model_dir}')
    for note in notes:
        print(f"  {note}")
    if absent:
        print(f"  absent from the model snapshot: {', '.join(absent)} (recorded in parity-model-dir-sources.json)")
    if "generation_config.json" in absent:
        print("  generation_config.json is absent from the model snapshot. hidden_parity.py loads weights and runs "
              "the layers; it never generates, so this check needs no stop token or sampling defaults.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
