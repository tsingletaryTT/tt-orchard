# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Decode speed against context length: one traced step loop runs positions 0 to MAX-1 with the token ids of a real text, and the time of 32-step windows at chosen
positions is reported. Long positions cost more in the flash MLA op (it reads every cached row up to the position). Under a gozer lease."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ttnn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default=os.environ.get("MINICPM3_DIR"), required="MINICPM3_DIR" not in os.environ)
ap.add_argument("--weights", default=os.environ.get("MINICPM3_WEIGHTS"), required="MINICPM3_WEIGHTS" not in os.environ)
ap.add_argument("--text-file", required=True, help="text whose token ids are fed (repeated if too short)")
ap.add_argument("--max", type=int, default=4096)
ap.add_argument("--windows", default="32,512,1024,2048,4000")
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--out")
a = ap.parse_args()
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(a.model_dir)
base = tok(Path(a.text_file).read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
ids = (base * (a.max // len(base) + 1))[: a.max]
starts = [int(x) for x in a.windows.split(",")]
if a.tp == 2:
    from chip_model_tp import ChipMiniCPM3TP
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), trace_region_size=256 * 1024 * 1024)
else:
    from chip_model import ChipMiniCPM3
    dev = ttnn.open_device(device_id=0, trace_region_size=256 * 1024 * 1024)
try:
    if a.tp == 2:
        m = ChipMiniCPM3TP(dev, a.model_dir, a.weights, max_len=a.max)
    else:
        m = ChipMiniCPM3(dev, a.model_dir, a.weights, attn="mla", max_len=a.max, upd="paged", opt=3)
    m.enable_trace()
    res = []
    for p in range(a.max):
        if p in starts:
            t0 = time.time()
        m.traced_step(ids[p], p)
        if any(p == s + 31 for s in starts):
            dt = time.time() - t0
            res.append({"start_position": p - 31, "steps": 32, "seconds": dt, "tok_s": 32 / dt})
            print(json.dumps(res[-1]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"tp": a.tp, "windows": res}, indent=1) + "\n")
finally:
    if a.tp == 2:
        ttnn.close_mesh_device(dev)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    else:
        ttnn.close_device(dev)
