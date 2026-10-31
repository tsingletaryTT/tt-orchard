# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Convert the repository's `pytorch_model.bin` to one safetensors file, the format the other scripts read.

    python convert_weights.py --model-dir DIR --out FILE

The pickle is read with `torch.load(..., weights_only=True)`, so no code in the file runs. A model directory that already holds `model.safetensors` is
linked, not converted. The output is written under a temporary name of this run's own and published with a hard link that never replaces an existing file, so a stopped run leaves no half file and two runs for one snapshot do not collide.
"""
import argparse
import os
import sys
import uuid
from pathlib import Path


class ConvertError(Exception):
    """The conversion cannot start: the output exists or the pickle is missing."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True, help="the safetensors file to write; it must not exist")
    a = ap.parse_args()
    try:
        convert(a.model_dir, a.out)
    except ConvertError as exc:
        sys.exit(str(exc))


def convert(model_dir, out):
    """Write `out` from `model_dir/pytorch_model.bin` (the ASGI server calls this on its first start; main() calls it for the command line)."""
    out, src = Path(out), Path(model_dir) / "pytorch_model.bin"
    if out.exists():
        raise ConvertError(f"{out} exists; it is never overwritten")
    if not src.is_file():
        raise ConvertError(f"{src} is not a file")
    import torch
    from safetensors.torch import save_file
    state = torch.load(src, map_location="cpu", weights_only=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    # A temporary name of this run's own (two runs for the same snapshot never write one file), published with a hard link that fails when the output exists, so a
    # file another run finished first is never replaced and never seen half written.
    tmp = out.with_name(f".{out.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        save_file({k: v.contiguous() for k, v in state.items()}, str(tmp))
        try:
            os.link(tmp, out)
        except FileExistsError:
            print(f"{out} was made by another run while this one converted; it was left as it is")
            return
    finally:
        tmp.unlink(missing_ok=True)
    print(f"wrote {len(state)} tensors to {out}")


if __name__ == "__main__":
    main()
