# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Convert the repository's `pytorch_model.bin` to one safetensors file, the format the other scripts read.

    python convert_weights.py --model-dir DIR --out FILE

The pickle is read with `torch.load(..., weights_only=True)`, so no code in the file runs. A model directory that already holds `model.safetensors` is
linked, not converted. The output file is written under a temporary name and renamed when complete, so a stopped run leaves no half file.
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True, help="the safetensors file to write; it must not exist")
    a = ap.parse_args()
    out, src = Path(a.out), Path(a.model_dir) / "pytorch_model.bin"
    if out.exists():
        sys.exit(f"{out} exists; it is never overwritten")
    if not src.is_file():
        sys.exit(f"{src} is not a file")
    import torch
    from safetensors.torch import save_file
    state = torch.load(src, map_location="cpu", weights_only=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    save_file({k: v.contiguous() for k, v in state.items()}, str(tmp))
    os.replace(tmp, out)
    print(f"wrote {len(state)} tensors to {out}")


if __name__ == "__main__":
    main()
