#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Build the launcher and the model directory for the restricted-head parity check.

It reuses the sidecar parity builder (orchard/skills/sidecar-parity-templates/prepare_parity.py), which this directory's skill copies next to
this file, and changes only which script the launcher runs. It reads `restricted_head_config.json` from its own directory
(keys: bundle_dir, model_snapshot, tt_cache) and writes two things next to itself:

* `parity-model-dir/`: a copy of the model's config.json and symlinks to its tokenizer files, `model.safetensors.index.json` and every
  `model-*.safetensors` shard. The readout and the model's code are left out, so the TT loader cannot read them by accident
  (restricted_head_parity.py reads the readout from the snapshot itself).
* `parity-run.sh`: a copy of `<bundle_dir>/run.sh` that runs `restricted_head_parity.py` in the bundle's python with the server's
  environment, the drafter off, the weights directory set to `parity-model-dir` and the tensor cache set to `tt_cache`.

An expected edit that does not happen exactly once exits 2 with a message that names it, and nothing is written.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE_DIR = Path(__file__).resolve().parent


def fail(message: str) -> None:
    print(f"prepare_restricted_head: {message}", file=sys.stderr)
    sys.exit(2)


def load_builder():
    """prepare_parity.py from this directory, or from the skills tree when this file has not been copied."""
    for d in (HERE_DIR, HERE_DIR.parent / "sidecar-parity-templates"):
        path = d / "prepare_parity.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("prepare_parity_for_restricted_head", str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.HERE_DIR = HERE_DIR                      # build_model_dir writes next to THIS script, wherever the builder was found
            return mod
    fail("prepare_parity.py is not next to this script or in the sidecar-parity-templates directory")


def main() -> int:
    try:
        cfg = json.loads((HERE_DIR / "restricted_head_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"cannot read restricted_head_config.json: {exc}")
    missing = [k for k in ("bundle_dir", "model_snapshot", "tt_cache") if not cfg.get(k)]
    if missing:
        fail(f"restricted_head_config.json is missing {missing}")
    pp = load_builder()
    bundle, snap = Path(cfg["bundle_dir"]).resolve(), Path(cfg["model_snapshot"])
    for label, d in (("bundle_dir", bundle), ("model_snapshot", snap)):
        if not d.is_dir():
            fail(f"{label} {d} is not a directory")
    src_run = bundle / "run.sh"
    if not src_run.is_file():
        fail(f"{src_run} does not exist")
    # Edit in memory and check the snapshot first, so a failure leaves no launcher and no parity-model-dir.
    text, notes = pp.edit_run_script(src_run.read_text(encoding="utf-8"), bundle, HERE_DIR / "restricted_head_parity.py",
                                     HERE_DIR / "parity-model-dir", cfg["tt_cache"])
    pp.check_snapshot(snap)
    model_dir, copied, linked = pp.build_model_dir(snap)
    out = HERE_DIR / "parity-run.sh"
    out.write_text(text, encoding="utf-8")
    out.chmod(out.stat().st_mode | 0o111)
    print(f"parity-model-dir: {model_dir}")
    print(f"  copied: {', '.join(copied)}")
    print(f"  linked: {len(linked)} files")
    print(f"parity-run.sh: {out}")
    for note in notes:
        print(f"  {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
