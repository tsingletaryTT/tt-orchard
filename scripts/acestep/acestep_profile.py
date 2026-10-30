# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-stage wall time of one ACE-Step 1.5 turbo request on the chip (issue 254; the CPU pieces of a request).

    python acestep_profile.py --bundle-code CODE --snapshot SNAP --root WORK --durations 30,120 --out profile.json   # run under a gozer lease

It loads a `Session` (acestep_run.py), makes one unrecorded call per duration so kernels are compiled, then one timed call with a `StageClock` wrapped around
each stage. The clock records inclusive time (a stage with the stages it calls) and exclusive time (the stage's own code), so a table of exclusive times adds up to the
call's total. Nothing here changes what the request computes.

`StageClock` and `table` need only the standard library, so they are tested without a chip.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path


class StageClock:
    """Wraps callables with a wall clock. Each thread has its own stack of open stages, so a stage running in another thread (ACE-Step runs the diffusion in a
    service thread while the caller waits) is not counted as a child of whatever the caller happens to be running."""

    def __init__(self):
        self._rows = {}
        self._local = threading.local()
        self._lock = threading.Lock()

    def _stack(self):
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def depth(self) -> int:
        return len(self._stack())

    def wrap(self, obj, attr: str, name: str):
        """Replaces obj.attr (a method or function) with a timed version that returns the same value. Returns the original."""
        original = getattr(obj, attr)
        clock = self

        def timed(*a, **kw):
            stack = clock._stack()
            frame = [0.0]                     # time spent in stages called from this one
            stack.append(frame)
            t0 = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                dt = time.perf_counter() - t0
                stack.pop()
                if stack:
                    stack[-1][0] += dt
                with clock._lock:
                    row = clock._rows.setdefault(name, {"calls": 0, "inclusive_s": 0.0, "exclusive_s": 0.0})
                    row["calls"] += 1
                    row["inclusive_s"] += dt
                    row["exclusive_s"] += dt - frame[0]

        timed.__dict__.update(getattr(original, "__dict__", {}))      # attributes of the wrapped function (acestep_fast's `reset`) stay visible
        timed.__wrapped__ = original
        setattr(obj, attr, timed)
        return original

    def rows(self) -> dict:
        with self._lock:
            return {k: dict(v) for k, v in self._rows.items()}

    def reset(self):
        with self._lock:
            self._rows.clear()


def table(rows: dict, total_s: float) -> str:
    """A text table sorted by exclusive time, with each stage's share of `total_s`."""
    lines = [f"{'stage':44s} {'calls':>5s} {'excl s':>8s} {'share %':>8s} {'incl s':>8s}"]
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["exclusive_s"]):
        lines.append(f"{name:44s} {r['calls']:5d} {r['exclusive_s']:8.3f} {100 * r['exclusive_s'] / total_s:8.1f} {r['inclusive_s']:8.3f}")
    return "\n".join(lines)


def instrument(sess, clock: StageClock):
    """Wraps the stages of one loaded `Session`. Names say what the stage is; where a stage contains another the inclusive column shows it."""
    import gc
    import acestep.inference as inf
    h, model = sess.handler, sess.handler.model
    for attr, name in (("_prepare_batch", "cond: _prepare_batch (tokenise, masks, silence latents)"),
                       ("infer_text_embeddings", "cond: text encoder (Qwen3-Embedding, CPU)"),
                       ("infer_lyric_embeddings", "cond: lyric token embedding"),
                       ("infer_refer_latent", "cond: reference-audio latent"),
                       ("service_generate", "service_generate (all of the diffusion call)"),
                       ("_decode_generate_music_pred_latents", "decode wrapper (VAE call and its clean-up)")):
        if hasattr(h, attr):
            clock.wrap(h, attr, name)
    clock.wrap(model, "generate_audio", "generate_audio (encoder, sampler loop)")
    clock.wrap(model, "prepare_condition", "cond: prepare_condition (condition encoder)")
    enc = model.encoder
    for sub, name in (("text_projector", "cond: text projector"), ("lyric_encoder", "cond: lyric encoder (CPU transformer)"),
                      ("timbre_encoder", "cond: timbre encoder (reference audio)")):
        if hasattr(enc, sub):
            clock.wrap(getattr(enc, sub), "forward", name)
    inner = sess.tap.inner
    clock.wrap(inner, "forward", "dit step: RemoteDecoder.forward (copies in, chip, copy out)")
    worker = getattr(sess, "worker", None)
    if worker is not None:
        clock.wrap(worker, "call", "worker.call (host to chip round trip, all ops)")
        dit = worker.model.dit
        clock.wrap(dit, "prepare", "chip prepare (cross K/V on host, tables, upload)")
        clock.wrap(dit.ref, "encode", "  prepare: cross K/V projection on the host CPU")
        clock.wrap(dit, "forward", "chip forward (one DiT step incl. patchify and copies)")
        import tt_acestep.acestep_host as host
        clock.wrap(host, "patchify", "  forward: patchify on the host")
        clock.wrap(host, "unpatchify", "  forward: unpatchify on the host")
        clock.wrap(worker.vae, "decode", "chip VAE decode (one window)")
    if hasattr(sess.vtap.inner, "_decode"):
        clock.wrap(sess.vtap.inner, "_decode", "vae: one window (wire encode, chip, wire decode)")
    clock.wrap(gc, "collect", "gc.collect")
    if hasattr(inf, "normalize_audio"):
        clock.wrap(inf, "normalize_audio", "post: normalize_audio (peak)")
    import acestep.audio_utils as au
    clock.wrap(au.AudioSaver, "save_audio", "post: save_audio (WAV write)")
    for modname, mod in list(sys.modules.items()):
        if modname.endswith("modeling_acestep_v15_turbo") and hasattr(mod, "DCWCorrector"):
            clock.wrap(mod.DCWCorrector, "apply", "sampler: DCW correction (wavelet)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bundle-code", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--durations", default="30,120")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--prompt", default="warm lo-fi hip hop, mellow rhodes piano, soft vinyl crackle, relaxed drums")
    p.add_argument("--lyrics", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast", default="none", help="steps of acestep_fast.py to switch on (all, none or a list)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import acestep_run
    import acestep_fast
    sess = acestep_run.Session(device="chip", bundle_code=args.bundle_code, snapshot=args.snapshot, root=args.root, fast=acestep_fast.parse_steps(args.fast))
    clock = StageClock()
    tmp = Path(args.root) / "profile"
    result = {"prompt": args.prompt, "seed": args.seed, "fast": list(sess.fast), "runs": []}
    try:
        for d in (float(x) for x in args.durations.split(",")):
            sess.one(tmp / f"warm{int(d)}", d, args.prompt, args.lyrics, args.seed)       # compile the kernels for this length
            if not getattr(sess, "_instrumented", False):
                instrument(sess, clock)
                sess._instrumented = True
            for r in range(args.repeats):
                clock.reset()
                _, t = sess.one(tmp / f"d{int(d)}r{r}", d, args.prompt, args.lyrics, args.seed)
                rows = clock.rows()
                rows["vae: RemoteVae total (windows, copies, chip)"] = {"calls": len(sess.vtap.seconds), "inclusive_s": sum(sess.vtap.seconds), "exclusive_s": 0.0}
                total = t["generate_and_decode_s"]
                chip = {"dit_chip_s": t.get("dit_s"), "vae_chip_s": t.get("vae_s")}
                print(f"--- {d:g} s, repeat {r}: total {total:.3f} s, {chip}", flush=True)
                print(table(rows, total), flush=True)
                result["runs"].append({"duration_s": d, "repeat": r, "total_s": total, "chip": chip, "stages": rows})
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    finally:
        sess.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
