# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Compare recorded chip runs with a reference run through orchard.first_divergence and print one line per run: error at chosen blocks, logits agreement and
the teacher-forced token flips. Usage: summarize.py REF_DIR FLOOR_A FLOOR_B CHIP_RUN [CHIP_RUN ...]   (REF_DIR holds float32, bfloat16 and native)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from orchard import audio_reference as ar  # noqa: E402
from orchard import first_divergence as fd  # noqa: E402

ref_dir = Path(sys.argv[1])
ref = ar.load_run(ref_dir / "float32")
floor = (ar.load_run(ref_dir / "bfloat16"), ar.load_run(ref_dir / "native"))
toks = [int(v) for v in ref.array("other", "tokens")]
for path in sys.argv[2:]:
    cand = ar.load_run(path)
    rep = fd.find_first_divergence(ref, cand, floor=floor, tolerance=fd.Tolerance(rel_l2=0.10))
    by = {b["name"]: b for b in rep["blocks"]}
    flips = 0
    margins = []
    for k in range(len(toks)):
        r = np.asarray(ref.array("other", f"dec_logits_{k}"))
        c = np.asarray(cand.array("other", f"dec_logits_{k}"))
        flips += int(c.argmax() != r.argmax())
    t = {x["name"]: x for x in rep["tensors"]} if "tensors" in rep else {}
    def g(n, key):
        return round(t[n][key], 4) if n in t else None
    print(f"{Path(path).name:28s} layer0 {by['layer_00']['rel_l2']:.4f} layer30 {by['layer_30']['rel_l2']:.4f} layer61 {by['layer_61']['rel_l2']:.4f} "
          f"(floor {by['layer_61']['floor']['rel_l2']:.4f}) | prefill logits rel {g('prefill_logits', 'rel_l2')} pcc {g('prefill_logits', 'pcc')} | "
          f"decode0 rel {g('dec_logits_0', 'rel_l2')} pcc {g('dec_logits_0', 'pcc')} | decode15 rel {g('dec_logits_15', 'rel_l2')} | flips {flips}/{len(toks)}")
