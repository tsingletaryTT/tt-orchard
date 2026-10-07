#!/usr/bin/env python3
"""Build the model directory and the run script for a weights-only swap (stage 2).

The weights-swap-check skill copies this file into the stage directory and runs it. It reads
`swap_config.json` from its own directory and writes two things next to itself:

1. `model-dir/`, a directory that looks like the nearest model's but holds the new weights.
   - COPIED from base_snapshot: config.json, preprocessor_config.json and
     video_preprocessor_config.json. The bundle's TT model class is registered for the nearest
     model's architecture name, and that model is a vision-language model, so its config must be
     the one vLLM sees. A file absent from base_snapshot is skipped without a message.
   - SYMLINKED from new_snapshot (absolute, fully resolved targets): tokenizer.json,
     tokenizer_config.json, chat_template.jinja, generation_config.json,
     model.safetensors.index.json and every *.safetensors file. tokenizer.json and at least one
     *.safetensors file are required (exit 2 without them); the other names are linked when present
     and reported when absent.
2. `run.sh`, a copy of `<bundle_dir>/run.sh` with these edits:
   - the `HERE=...` line becomes `HERE="<bundle_dir>"` (exactly once), so the copy still finds
     the bundle's venv and model code;
   - `--model "<nearest_model_id>"` (quoted or unquoted) becomes `--model <model-dir>` (exactly once);
   - ` --revision <40 hex>` and ` --tokenizer-revision <40 hex>` are deleted, because a local
     directory has no revision. Each may appear at most once; an absent one is reported.
   - every `export HF_MODEL=...` line becomes `export HF_MODEL="<model-dir>"`. The TT runtime
     takes its weights directory from MODEL_WEIGHTS_DIR, then HF_MODEL, then the config path.
     The bundle sets HF_MODEL to the nearest model's id, which resolves to that model's HF cache,
     so `--model <model-dir>` alone serves the BASE weights. An absent line is reported; it is not
     an error, because serve_and_compare.py also sets both variables in the server's environment.

   - when the new model has no `mtp.*` tensor, `export QWEN36_DRAFTER=...` becomes
     `export QWEN36_DRAFTER=""` (at most one such line). The nearest bundle may serve with a speculative
     drafter that needs the model's MTP head, and a model without those tensors dies at engine start with
     "model has no MTP head" (Cloudflare/clef, 2026-10-06). With the drafter off the bundle serves plain
     decoding, which gives the same greedy tokens, so the check measures the same weights. The check reads
     the new model's `model.safetensors.index.json`, or the shard headers when there is no index. When it
     cannot tell, the line is left alone and the script says so.

An expected edit that does not happen exactly once exits 2 with a message that names it, and no
run.sh is written. Everything else in the script, including its environment lines, stays as it is.

For a container package (stage 4's container configuration) swap_config.json names `package`
and no `bundle_dir`. Then only model-dir/ is built: serve_and_compare_container.py edits the
container's docker command and needs no run.sh.

Config keys read here: nearest_model_id, base_snapshot, new_snapshot, and bundle_dir or package.
The other keys in swap_config.json are for serve_and_compare.py and serve_and_compare_container.py.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import struct
import sys
from pathlib import Path

HERE_DIR = Path(__file__).resolve().parent
COPIED = ("config.json", "preprocessor_config.json", "video_preprocessor_config.json")
LINKED = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
          "model.safetensors.index.json")
REQUIRED_LINKS = ("tokenizer.json",)


def fail(message: str) -> None:
    print(f"prepare_swap: {message}", file=sys.stderr)
    sys.exit(2)


def load_config() -> dict:
    path = HERE_DIR / "swap_config.json"
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"cannot read {path}: {exc}")
    missing = [k for k in ("nearest_model_id", "base_snapshot", "new_snapshot") if not cfg.get(k)]
    if not cfg.get("bundle_dir") and not cfg.get("package"):
        missing.append("bundle_dir (or package, for a container package)")
    if missing:
        fail(f"swap_config.json is missing {missing}")
    return cfg


def build_model_dir(base: Path, new: Path) -> tuple[Path, list[str], list[str], list[str]]:
    """Make model-dir from scratch. Returns (path, copied, linked, absent link names)."""
    md = HERE_DIR / "model-dir"
    if md.is_symlink():
        fail(f"{md} is a symlink; remove it so this script can build a real directory")
    if md.exists():
        shutil.rmtree(md)      # only ever a directory this script built: copies and symlinks
    md.mkdir()
    copied = []
    for name in COPIED:
        src = base / name
        if src.is_file():
            shutil.copyfile(src, md / name)        # follows the snapshot link to the blob
            copied.append(name)
    names = list(LINKED) + sorted(p.name for p in new.glob("*.safetensors"))
    linked, absent = [], []
    for name in names:
        src = new / name
        if not src.exists():
            absent.append(name)
            continue
        os.symlink(os.path.realpath(src), md / name)
        linked.append(name)
    for name in REQUIRED_LINKS:
        if name not in linked:
            fail(f"{name} is not in new_snapshot {new}")
    if not any(n.endswith(".safetensors") for n in linked):
        fail(f"new_snapshot {new} has no *.safetensors file")
    return md, copied, linked, absent


MTP_NAME = re.compile(r"(^|\.)mtp\.")


def shard_tensor_names(path: Path) -> list[str] | None:
    """Tensor names from a safetensors header, or None when the file is not readable as one."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return None
            (n,) = struct.unpack("<Q", raw)
            if not 0 < n <= 100 * 1024 * 1024:
                return None
            header = json.loads(f.read(n).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return [k for k in header if k != "__metadata__"] if isinstance(header, dict) else None


def has_mtp_tensors(new: Path) -> bool | None:
    """Does the new model hold `mtp.*` tensors? Read from its weight index, else from the shard headers.
    None when neither can be read, so the caller can leave things alone."""
    try:
        weight_map = json.loads((new / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        if isinstance(weight_map, dict):
            return any(MTP_NAME.search(k) for k in weight_map)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    shards = sorted(new.glob("*.safetensors"))
    if not shards:
        return None
    names = []
    for shard in shards:
        got = shard_tensor_names(shard)
        if got is None:
            return None
        names += got
    return any(MTP_NAME.search(k) for k in names)


def edit_run_script(text: str, bundle: Path, nearest: str, model_dir: Path,
                    drafter_off: bool = False) -> tuple[str, list[str]]:
    """Apply the edits. Returns (new text, notes about optional edits that did not apply)."""
    text, n = re.subn(r"^HERE=.*$", lambda m: f'HERE="{bundle}"', text, flags=re.MULTILINE)
    if n != 1:
        fail(f"expected exactly one HERE= line in {bundle / 'run.sh'}, found {n}")
    model_re = r'--model\s+(?:"' + re.escape(nearest) + r'"|' + re.escape(nearest) + r')(?=\s|$)'
    text, n = re.subn(model_re, lambda m: f"--model {shlex.quote(str(model_dir))}", text)
    if n != 1:
        fail(f'expected exactly one --model "{nearest}" in {bundle / "run.sh"}, found {n}')
    notes = []
    text, n = re.subn(r"^([ \t]*)export[ \t]+HF_MODEL=.*$",
                      lambda m: f'{m.group(1)}export HF_MODEL="{model_dir}"', text, flags=re.MULTILINE)
    if n == 0:
        notes.append("export HF_MODEL=... was absent; nothing to rewrite (serve_and_compare.py sets "
                     "HF_MODEL and MODEL_WEIGHTS_DIR itself)")
    else:
        notes.append(f'export HF_MODEL="{model_dir}" ({n} line(s) rewritten)')
    if drafter_off:
        text, n = re.subn(r"^([ \t]*)export[ \t]+QWEN36_DRAFTER=.*$",
                          lambda m: f'{m.group(1)}export QWEN36_DRAFTER=""', text, flags=re.MULTILINE)
        if n > 1:
            fail(f"expected at most one export QWEN36_DRAFTER= line in {bundle / 'run.sh'}, found {n}")
        notes.append('export QWEN36_DRAFTER="" (the new model has no mtp.* tensors, so the speculative drafter '
                     "that needs the MTP head is switched off)" if n else
                     "export QWEN36_DRAFTER=... was absent; nothing to clear")
    for flag in ("--revision", "--tokenizer-revision"):
        # The leading space keeps " --revision" from matching inside "--tokenizer-revision".
        text, n = re.subn(r" " + flag + r'\s+"?[0-9a-f]{40}"?(?=\s|$)', "", text)
        if n > 1:
            fail(f"expected at most one {flag} <40 hex> in {bundle / 'run.sh'}, found {n}")
        if n == 0:
            notes.append(f"{flag} <40 hex> was absent; nothing to delete")
    return text, notes


def main() -> int:
    cfg = load_config()
    base, new = Path(cfg["base_snapshot"]), Path(cfg["new_snapshot"])
    bundle = Path(cfg["bundle_dir"]).resolve() if cfg.get("bundle_dir") else None
    for label, d in (("bundle_dir", bundle), ("base_snapshot", base), ("new_snapshot", new)):
        if d is not None and not d.is_dir():
            fail(f"{label} {d} is not a directory")
    model_dir = HERE_DIR / "model-dir"
    if bundle is not None:
        src_run = bundle / "run.sh"
        if not src_run.is_file():
            fail(f"{src_run} does not exist")
        # Edit in memory first, so a failed edit leaves no run.sh and no half-built model-dir.
        mtp = has_mtp_tensors(new)
        text, notes = edit_run_script(src_run.read_text(encoding="utf-8"), bundle,
                                      cfg["nearest_model_id"], model_dir, drafter_off=(mtp is False))
        if mtp is None:
            notes.append("could not tell whether the new model has mtp.* tensors (no readable weight index or "
                         "shard headers), so the drafter setting is unchanged")
    model_dir, copied, linked, absent = build_model_dir(base, new)
    print(f"model-dir: {model_dir}")
    print(f"  copied from base_snapshot: {', '.join(copied) or 'nothing'}")
    print(f"  linked from new_snapshot: {len(linked)} files ({', '.join(linked)})")
    if absent:
        print(f"  absent in new_snapshot, not linked: {', '.join(absent)}")
    if bundle is None:
        print(f"run.sh: not built; {cfg['package']} is a container package, and "
              "serve_and_compare_container.py starts it")
        return 0
    run_out = HERE_DIR / "run.sh"
    run_out.write_text(text, encoding="utf-8")
    run_out.chmod(run_out.stat().st_mode | 0o111)
    print(f"run.sh: {run_out}")
    print(f'  HERE="{bundle}"; --model {model_dir}')
    for note in notes:
        print(f"  {note}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
