#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Time or compare the step-loop prefill and the chunked prefill of GraniteChip on one chip, under the guards in orchard/prefill_bench.py.

    timeout 570 python3 scripts/granite_prefill_bench.py time  --model-dir DIR --tokens 512 --path step --trace off --log L.jsonl --out R.json
    timeout 570 python3 scripts/granite_prefill_bench.py agree --model-dir DIR --tokens 512 --trace off --log L.jsonl --out R.json

Run it under a gozer lease (it opens device 0 of the leased chips). Every iteration, the model construction included, logs a line before and
after and has its own time limit; a hang writes a `hang` line and ends the process with status 97. The prompt is the first N tokens of a text
file (default docs/project-log.md); its sha256 is recorded so two runs can be shown to use the same prompt.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pkgsrc" / "runtime"))

from orchard import prefill_bench as pb  # noqa: E402


def prompt_ids(model_dir, text_file, n):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    text = Path(text_file).read_text(encoding="utf-8")
    ids = tok.encode(text).ids[:n]
    if len(ids) != n:
        sys.exit(f"the text has only {len(ids)} tokens, {n} wanted")
    return ids, hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["time", "agree"])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--path", choices=["step", "chunked"], default="step", help="time mode: which prefill")
    ap.add_argument("--trace", choices=["on", "off"], default="off", help="on: the model captures the decode trace (the server's setting)")
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--text", default=str(ROOT / "docs" / "project-log.md"))
    ap.add_argument("--construct-timeout", type=float, default=240)
    ap.add_argument("--warmup-timeout", type=float, default=180)
    ap.add_argument("--timeout", type=float, default=120, help="per timed iteration, seconds")
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    trace = a.trace == "on"
    log = pb.BenchLog(a.log)
    wd = pb.Watchdog(log)
    log.line("start", mode=a.mode, path=a.path, tokens=a.tokens, trace=trace, warmups=a.warmups, runs=a.runs, chunk=a.chunk, max_len=a.max_len)
    if trace and a.mode == "time" and a.warmups < 1:
        raise pb.TraceWithoutWarmup("a trace-on run needs at least one warm-up iteration")
    ids, sha = prompt_ids(a.model_dir, a.text, a.tokens)
    log.line("prompt", tokens=len(ids), sha256=sha)

    import torch
    import ttnn  # opens nothing by itself under a lease, but the device is opened below, so this must run under `gozer`
    from granite_chip import GraniteChip

    def open_and_build():
        d = ttnn.open_device(device_id=0, trace_region_size=400 * 1024 * 1024)
        return d, GraniteChip(d, a.model_dir, max_len=a.max_len, trace=trace)

    (dev, m), _ = pb.run_iteration(log, wd, "construct", 0, a.construct_timeout, open_and_build, phase="setup")
    result = {"mode": a.mode, "tokens": a.tokens, "trace": trace, "max_len": a.max_len, "prompt_sha256": sha, "chunk": a.chunk}
    try:
        if a.mode == "time":
            def once():
                m.reset()
                if a.path == "step":
                    m.prefill(ids)
                else:
                    m.prefill_chunked(ids, chunk=a.chunk)
                ttnn.synchronize_device(dev)
            secs = pb.run_plan(log, wd, a.path, once, a.warmups, a.runs, trace, a.warmup_timeout, a.timeout)
            result.update(path=a.path, **pb.summarize(a.tokens, secs))
        else:
            def greedy(lg):
                out, t = [], int(lg.argmax())
                for _ in range(4):
                    out.append(t)
                    t = int(m.step(t).argmax())
                return out

            def states():
                return [ttnn.to_torch(m.ss[i]).float().clone() for i in m.ss]

            def ref():
                m.reset()
                lg = m.prefill(ids)
                return lg, states(), greedy(lg)

            def cand():
                m.reset()
                lg = m.prefill_chunked(ids, chunk=a.chunk)
                return lg, states(), greedy(lg)

            (lg_r, st_r, nx_r), s_step = pb.run_iteration(log, wd, "step", 1, a.warmup_timeout, ref, phase="agree")
            (lg_c, st_c, nx_c), s_ch = pb.run_iteration(log, wd, "chunked", 1, a.warmup_timeout, cand, phase="agree")
            d = (lg_c - lg_r).abs()
            t5r, t5c = set(lg_r.topk(5).indices.tolist()), set(lg_c.topk(5).indices.tolist())
            result.update(
                logit_cosine=round(torch.nn.functional.cosine_similarity(lg_c, lg_r, dim=0).item(), 7), logit_max_abs_diff=round(d.max().item(), 4),
                logit_max_abs=round(lg_r.abs().max().item(), 3), top1_equal=int(lg_c.argmax()) == int(lg_r.argmax()), top5_overlap=len(t5r & t5c),
                next4_step=nx_r, next4_chunked=nx_c, next4_equal=nx_r == nx_c,
                max_state_rel_diff=max(((x - y).abs().max() / (y.abs().max() + 1e-9)).item() for x, y in zip(st_c, st_r)),
                seconds_step_including_readout=round(s_step, 3), seconds_chunked_including_readout=round(s_ch, 3))
        Path(a.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        log.line("result", **{k: v for k, v in result.items() if k in ("tok_s_median", "path", "tokens")})
    finally:
        # a process ended by the watchdog never reaches this line
        pb.run_iteration(log, wd, "close", 0, 60, lambda: (m.close(), ttnn.close_device(dev)), phase="setup")


if __name__ == "__main__":
    main()
