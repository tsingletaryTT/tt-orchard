# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Teacher-forced perplexity on a text window: CPU float32 reference against the chip. The text is the start of the WikiText-2 test split (raw), tokenized
without a chat template, with the model's beginning-of-sequence token first.

    python ppl_check.py ref  --parquet TEST.parquet --tokens 512 --out ref_ppl.json
    python ppl_check.py chip --parquet TEST.parquet --tokens 512 --out chip_ppl.json [--tp 2] [--opt 3] [--mlp-dtype bfp8 --mlp-fidelity lofi]

The chip runs the window one token per step (positions 0 to N-1) and reads the logits after every step, so it measures the decode path, not a separate prefill.
Reports mean negative log likelihood, perplexity, and, for the chip, the mean absolute difference of the per-token log probabilities against the reference file.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def window(tok, parquet, n):
    import pandas as pd
    text = "".join(t for t in pd.read_parquet(parquet)["text"].tolist()[:400])
    ids = tok(text, add_special_tokens=False)["input_ids"][: n - 1]
    return [tok.bos_token_id] + ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "chip"])
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--model-dir", default=os.environ.get("MINICPM3_DIR"), required="MINICPM3_DIR" not in os.environ)
    ap.add_argument("--weights", default=os.environ.get("MINICPM3_WEIGHTS"), required="MINICPM3_WEIGHTS" not in os.environ)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--opt", type=int, default=3)
    ap.add_argument("--mlp-dtype", default="bf16")
    ap.add_argument("--mlp-fidelity", default="hifi2")
    a = ap.parse_args()
    if a.tp == 2:
        from chip_model_tp import refuse_unsupported_tp_args
        refuse_unsupported_tp_args(ap, a)                        # before any device is opened or the fabric is configured
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    ids = window(tok, a.parquet, a.tokens)
    n = len(ids)
    lps = []
    if a.mode == "ref":
        from ref_model import RefModel
        torch.set_num_threads(8)
        m = RefModel(a.model_dir, a.weights, dtype=torch.float32, max_len=n + 8)
        lg = m.forward(torch.tensor([ids]))[0]
        lp = torch.log_softmax(lg, dim=-1)
        lps = [lp[i, ids[i + 1]].item() for i in range(n - 1)]
    else:
        import ttnn
        if a.tp == 2:
            from chip_model_tp import ChipMiniCPM3TP
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), trace_region_size=256 * 1024 * 1024)
        else:
            from chip_model import ChipMiniCPM3
            dev = ttnn.open_device(device_id=0, trace_region_size=256 * 1024 * 1024)
        try:
            if a.tp == 2:
                m = ChipMiniCPM3TP(dev, a.model_dir, a.weights, max_len=n + 8, mlp_dtype=a.mlp_dtype, mlp_fidelity=a.mlp_fidelity)
            else:
                m = ChipMiniCPM3(dev, a.model_dir, a.weights, attn="mla", max_len=n + 8, upd="paged", opt=a.opt, mlp_dtype=a.mlp_dtype,
                                 mlp_fidelity=a.mlp_fidelity)
            m.enable_trace()
            t0 = time.time()
            for i in range(n - 1):
                lg = m.traced_step(ids[i], i)
                lps.append(torch.log_softmax(lg.float(), dim=-1)[ids[i + 1]].item())
            print(f"{n - 1} steps in {time.time() - t0:.1f}s", flush=True)
        finally:
            if a.tp == 2:
                ttnn.close_mesh_device(dev)
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            else:
                ttnn.close_device(dev)
    nll = -float(np.mean(lps))
    rec = {"mode": a.mode, "tokens": n - 1, "mean_nll": nll, "perplexity": math.exp(nll), "logprobs": lps}
    if a.ref:
        ref = json.load(open(a.ref))
        d = np.abs(np.array(lps) - np.array(ref["logprobs"]))
        rec["vs_reference"] = {"mean_abs_logprob_difference": float(d.mean()), "max_abs": float(d.max()), "reference_ppl": ref["perplexity"]}
    print({k: v for k, v in rec.items() if k != "logprobs"}, flush=True)
    Path(a.out).write_text(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
