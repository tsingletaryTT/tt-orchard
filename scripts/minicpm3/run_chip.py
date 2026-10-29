# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run MiniCPM3-4B on a chip (under a gozer lease) and record the run for the first-divergence probe, or time decode.

    python run_chip.py agree --attn mla --ref REFDIR/float32 --out OUTDIR      # teacher-forced agreement run, recorded
    python run_chip.py bench --attn mla --steps 64 --repeats 3                 # free-running greedy decode timing
"""
import argparse
import json
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
from make_reference import PROMPT, make_recorder, prompt_ids  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["agree", "bench"])
    ap.add_argument("--model-dir", default=os.environ.get("MINICPM3_DIR"), required="MINICPM3_DIR" not in os.environ, help="model directory with config.json and the tokenizer (or set MINICPM3_DIR)")
    ap.add_argument("--weights", default=os.environ.get("MINICPM3_WEIGHTS"), required="MINICPM3_WEIGHTS" not in os.environ, help="the weights as one safetensors file (or set MINICPM3_WEIGHTS)")
    ap.add_argument("--attn", default="mla", choices=["mla", "host"])
    ap.add_argument("--ref", help="a recorded reference run directory (float32) for the token path")
    ap.add_argument("--out", help="new directory for the recorded chip run")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--fidelity", default="default", choices=["default", "hifi4"])
    ap.add_argument("--upd", default="int", choices=["int", "paged", "onehot"], help="cache write; a trace needs onehot or paged")
    ap.add_argument("--trace", action="store_true", help="bench: capture and replay the step as a trace")
    ap.add_argument("--skip", default="", help="timing only: comma list of mlp, lm")
    ap.add_argument("--opt", type=int, default=1, choices=[1, 2, 3], help="attention composition: 1 first version, 2 fewer ops, 3 also a sharded residual stream and a fused projection")
    ap.add_argument("--prompt-file", help="the prompt text the reference run used")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--shard", action="store_true", help="DRAM-sharded weights for the MLP and o_proj")
    ap.add_argument("--mlp-dtype", default="bf16", choices=["bf16", "bfp8", "bfp4"])
    ap.add_argument("--mlp-fidelity", default="hifi2", choices=["hifi2", "lofi", "hifi4"])
    ap.add_argument("--tp", type=int, default=1, choices=[1, 2], help="2: both chips of the board, tensor parallel (chip_model_tp.py)")
    ap.add_argument("--layers", type=int, default=0, help="debug: truncate the stack (0 = all)")
    a = ap.parse_args()
    if a.prompt_file:
        import make_reference
        make_reference.PROMPT = open(a.prompt_file, encoding="utf-8").read().strip()
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    ids = prompt_ids(tok)[0].tolist()
    n = len(ids)
    if a.tp == 2:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), trace_region_size=256 * 1024 * 1024)
    else:
        dev = ttnn.open_device(device_id=0, trace_region_size=256 * 1024 * 1024)
    try:
        t0 = time.time()
        if a.tp == 2:
            from chip_model_tp import ChipMiniCPM3TP
            m = ChipMiniCPM3TP(dev, a.model_dir, a.weights, max_len=a.max_len, mlp_dtype=a.mlp_dtype, mlp_fidelity=a.mlp_fidelity)
        else:
            m = ChipMiniCPM3(dev, a.model_dir, a.weights, attn=a.attn, max_len=a.max_len, fidelity=a.fidelity, upd=a.upd, opt=a.opt, shard=a.shard,
                             mlp_dtype=a.mlp_dtype, mlp_fidelity=a.mlp_fidelity)
        m.skip = set(x for x in a.skip.split(",") if x)
        print(f"model loaded on chip in {time.time() - t0:.1f}s", flush=True)
        if a.mode == "agree":
            agree(a, m, ids, n)
        else:
            bench(a, m, ids, n, tok)
    finally:
        if a.tp == 2:
            ttnn.close_mesh_device(dev)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        else:
            ttnn.close_device(dev)


def agree(a, m, ids, n):
    from orchard import audio_reference as ar
    ref = ar.load_run(a.ref)
    toks = [int(v) for v in ref.array("other", "tokens")]
    rec = make_recorder(Path(a.out), "chip", "bfloat16", n, len(toks), f"{a.model_dir}/config.json", Path(a.weights))
    rec.settings["device"] = "blackhole p300 chip 0, " + a.attn + " attention"
    streams = {}
    state = {"phase": "prefill", "k": -1}
    t0 = time.time()

    def tap(name, t):
        if name == "embed":
            return
        streams.setdefault((state["phase"], state["k"], name), []).append(t.numpy())

    logits = None
    for p, t in enumerate(ids):
        logits = m.run(t, p, tap)
    print(f"prefill {n} tokens in {time.time() - t0:.1f}s (first call includes kernel compile)", flush=True)
    for name in sorted({k[2] for k in streams}):
        rec.record("layer", name, np.stack(streams[("prefill", -1, name)]))
    rec.record("other", "prefill_logits", logits.numpy())
    matches = int(logits.argmax().item() == toks[0])
    free_ok = True
    nxt_free = int(logits.argmax().item())
    for k, t in enumerate(toks):
        state["phase"], state["k"] = "dec", k
        lg = m.run(t, n + k, tap if k == 0 else None)
        rec.record("other", f"dec_logits_{k}", lg.numpy())
        if k + 1 < len(toks):
            matches += int(lg.argmax().item() == toks[k + 1])
        if k == 0:
            for name in sorted({kk[2] for kk in streams if kk[0] == "dec"}):
                if "." not in name:
                    rec.record("other", "dec0_" + name, streams[("dec", 0, name)][0])
    rec.record("other", "tokens", np.array(toks, dtype=np.float32))
    rec.finish()
    print(f"teacher-forced next-token matches: {matches}/{len(toks)}", flush=True)


def bench(a, m, ids, n, tok):
    """Free-running greedy decode: 1 warm-up run, then `repeats` measured runs of `steps` tokens each on the loaded model (the contract in
    docs/benchmark-rules.md). Writes a record with spread, label, host load and the share of the DRAM-bandwidth floor."""
    import os
    from orchard import decode_floor as df, resident
    from orchard import model_config as mc
    results, timings = [], []
    step = m.run
    if a.trace:
        t0 = time.time()
        m.enable_trace()
        print(f"trace captured in {time.time() - t0:.1f}s", flush=True)
        step = lambda t, p, tap=None: m.traced_step(t, p)  # noqa: E731
    for r in range(a.repeats + resident.WARMUP_RUNS):
        if not a.trace:
            m.reset()          # a trace is bound to its cache buffers; stale rows past the position are masked by the op
        logits = None
        t_pre = time.time()
        for p, t in enumerate(ids):
            logits = step(t, p)
        prefill_s = time.time() - t_pre
        nxt = int(logits.argmax().item())
        out = []
        t0 = time.time()
        for k in range(a.steps):
            out.append(nxt)
            lg = step(nxt, n + k)
            nxt = int(lg.argmax().item())
        dt = time.time() - t0
        rec = {"repeat": r, "warmup": r < resident.WARMUP_RUNS, "steps": a.steps, "decode_s": dt, "decode_tok_s": a.steps / dt,
               "prefill_tokens": n, "prefill_s": prefill_s, "host_loadavg_1m": os.getloadavg()[0]}
        results.append(rec)
        print(json.dumps(rec), flush=True)
        if not rec["warmup"]:
            timings.append({k: v for k, v in rec.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("repeat",)})
    summary = resident.summarise_runs(timings, warmup=resident.WARMUP_RUNS)
    cfg = mc.read_model_config(a.model_dir + "/config.json")
    floor = df.floor_for_board(cfg, {"default": "bf16"}, "p300", dies=1)
    med = summary["parts"]["decode_tok_s"]["median"]
    record = {"label": summary["label"], "model": "openbmb/MiniCPM3-4B", "config": {"attn": a.attn, "opt": a.opt, "shard": a.shard, "trace": a.trace, "upd": a.upd,
              "fidelity": a.fidelity, "weights": "bfloat16 (MLP " + a.mlp_dtype + ", " + a.mlp_fidelity + ")", "chips": a.tp, "prompt_tokens": n, "steps": a.steps},
              "runs": results, "summary": summary, "floor": floor.as_dict(), "share_of_floor_median": df.share_of_floor(med, floor),
              "share_text": df.share_text(med, floor), "text": tok.decode(out)}
    print(record["share_text"], flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(record, indent=1) + "\n")
    print(tok.decode(out))


if __name__ == "__main__":
    main()
