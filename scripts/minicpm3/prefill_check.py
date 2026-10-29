# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Batched prefill on the chip against a recorded float32 reference: the last-prompt-token logits, then teacher-forced decode logits from the filled cache.

    python prefill_check.py --ref REF/float32 [--prompt-file TEXT] [--max-len N]    (under a gozer lease; the reference must be of the same prompt)
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ttnn  # noqa: E402
from chip_model import ChipMiniCPM3  # noqa: E402
import make_reference  # noqa: E402
from orchard import audio_reference as ar  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--prompt-file")
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--model-dir", default=os.environ.get("MINICPM3_DIR"), required="MINICPM3_DIR" not in os.environ)
ap.add_argument("--weights", default=os.environ.get("MINICPM3_WEIGHTS"), required="MINICPM3_WEIGHTS" not in os.environ)
a = ap.parse_args()
if a.prompt_file:
    make_reference.PROMPT = Path(a.prompt_file).read_text(encoding="utf-8").strip()
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(a.model_dir)
ids = make_reference.prompt_ids(tok)[0].tolist()
ref = ar.load_run(a.ref)
toks = [int(v) for v in ref.array("other", "tokens")]


def cmp(x, r):
    x, r = np.asarray(x, dtype=np.float64), np.asarray(r, dtype=np.float64)
    return round(float(np.corrcoef(x, r)[0, 1]), 5), round(float(np.linalg.norm(x - r) / np.linalg.norm(r)), 4)


dev = ttnn.open_device(device_id=0, trace_region_size=256 * 1024 * 1024)
try:
    m = ChipMiniCPM3(dev, a.model_dir, a.weights, attn="mla", max_len=a.max_len, upd="paged", opt=3)
    m.enable_trace()
    for rep in range(3):
        t0 = time.time()
        lg = m.prefill(ids)
        dt = time.time() - t0
        print(f"prefill {len(ids)} tokens: {dt * 1e3:.0f} ms ({len(ids) / dt:.0f} tok/s){'  (first call includes compile)' if rep == 0 else ''}", flush=True)
    r = np.asarray(ref.array("other", "prefill_logits"))
    print("prefill last-token logits: pcc, relL2", cmp(lg.numpy(), r), "argmax chip", int(lg.argmax()), "ref", int(r.argmax()), flush=True)
    flips = 0
    for k, t in enumerate(toks):
        lg = m.traced_step(t, len(ids) + k)
        rr = np.asarray(ref.array("other", f"dec_logits_{k}"))
        flips += int(int(lg.argmax()) != int(rr.argmax()))
        if k in (0, 7, 15):
            print(f"decode step {k} logits: pcc, relL2", cmp(lg.numpy(), rr), flush=True)
    print(f"decode argmax flips after batched prefill: {flips} of {len(toks)}", flush=True)
finally:
    ttnn.close_device(dev)
