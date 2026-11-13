#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""How many device ops the generated Qwen4 port issues per layer, counted on the CPU with the fake ttnn (no chip, no lease).

The fake counts every ttnn call (tests/fake_ttnn_compute.py `COUNTS`). This script generates the tiny Qwen4 port (the fixtures of tests/test_port_template.py), loads it
through the fake, runs the embedding and the prelude once and then each layer variant on its own, and prints the ops of each variant by name. A layer's op count does
not depend on the sequence length or the width, so the counts carry over to Flash-Next, where the pattern is 12 groups of three gated-delta-net layers and one sparse
attention layer. The tiny checkpoint's MLP is dense: for the MoE block the script runs the expert-parallel block of tests/test_port_template_moe_parallel.py and reports its
fixed ops and its ops per hit slot.

The projection multiplies the op count by a per-op cost taken from probe 16 (8 to 14 microseconds for a small elementwise op on one chip, 12 to 37 for a matmul at 32 rows);
the default is 10. It is a floor on launch and queue time for tiny tensors and does not predict the speed of the real model.

Usage: python3 scripts/port_op_count.py [--us-per-op 10] [--json]   (ORCHARD_ROOT=<checkout> counts that checkout's generator; the default is this checkout)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.environ.get("ORCHARD_ROOT") or Path(__file__).resolve().parents[1])          # another checkout can be counted: ORCHARD_ROOT=<path>
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]


def _weights(torch, load):
    g = torch.Generator().manual_seed(1)
    weights = {}
    for t in load("tiny-qwen4-tensors.json"):
        if t["dtype"] == "BF16":
            weights[t["name"]] = (torch.randn(*t["shape"], generator=g) * 0.1).to(torch.bfloat16)
        else:
            weights[t["name"]] = torch.arange(int(torch.tensor(t["shape"]).prod()), dtype=torch.int64).reshape(t["shape"])
        if t["name"].endswith("ple_embedding.ngram_heads_vocab_sizes"):
            weights[t["name"]] = torch.tensor([1009, 1013, 1019, 1021])
        elif t["name"].endswith("ple_embedding.ngram_heads_offsets"):
            weights[t["name"]] = torch.tensor([0, 1009, 2022, 3041])
        elif t["name"].endswith("ple_embedding.layer_multipliers"):
            weights[t["name"]] = torch.tensor([3, 5, 7])
    return weights


def count_layers() -> dict:
    """{variant: Counter of ops} for the layer functions of the tiny Qwen4 port, plus the ops of everything outside the layers."""
    import torch
    from _pytest.monkeypatch import MonkeyPatch
    from safetensors.torch import save_file

    import fake_ttnn_compute as fake
    from test_port_template import _load, import_port, tiny_qwen4

    mp = MonkeyPatch()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        text, _plan = tiny_qwen4()
        mod = import_port(mp, text, tmp)
        snap = tmp / "snap"
        snap.mkdir()
        save_file(_weights(torch, _load), str(snap / "model.safetensors"))
        model = mod.load(str(snap), fake.MeshDevice(1))
        ids = list(range(1, 33))
        model["ids"] = ids
        fake.COUNTS.clear()
        x = mod._embedding(model, "model.embed_tokens", ids)
        x = mod._model_prelude(model, x, ids)
        before = collections.Counter(fake.COUNTS)
        out = {"embedding_and_prelude": before}
        first = {}
        for i in range(mod.N_LAYERS):
            variant = mod.LAYER_VARIANT[i]
            if variant in first:
                continue
            first[variant] = i
        for variant, i in sorted(first.items()):
            fake.COUNTS.clear()
            mod.LAYERS[variant](model, i, x, ids, None, None)
            out[f"layer_{variant}"] = collections.Counter(fake.COUNTS)
        fake.COUNTS.clear()
        y = mod._model_hyper_connection_mixer(model, x)
        mod._linear(model, "lm_head", y)
        out["mixer_and_head"] = collections.Counter(fake.COUNTS)
        out["layer_variants"] = {v: i for v, i in first.items()}
        out["variant_of_layer"] = [mod.LAYER_VARIANT[i] for i in range(mod.N_LAYERS)]
        if hasattr(mod, "_forward_step"):
            out["decode"] = _count_decode_step(mod, model, fake, ids)
    mp.undo()
    return out


def _count_decode_step(mod, model, fake, ids) -> dict:
    """The ops of one cached decode step (issue 759): run `prefill` on the prompt, then `decode_step` once with each layer function wrapped so its calls are counted
    per variant. Returns {"layer_<variant>": Counter, "outside_layers": Counter (embedding, prelude, mixer and head), "whole_step": Counter}."""
    mod.prefill(model, ids)
    per_variant = {}
    originals = dict(mod.LAYERS)

    def wrap(variant, fn):
        def counted(*args, **kw):
            before = collections.Counter(fake.COUNTS)
            res = fn(*args, **kw)
            delta = collections.Counter(fake.COUNTS)
            delta.subtract(before)
            per_variant.setdefault(variant, collections.Counter()).update({k: v for k, v in delta.items() if v})
            return res
        return counted

    for variant, fn in originals.items():
        mod.LAYERS[variant] = wrap(variant, fn)
    try:
        fake.COUNTS.clear()
        mod.decode_step(model, 5, len(ids))
        whole = collections.Counter(fake.COUNTS)
    finally:
        mod.LAYERS.update(originals)
    # a variant is used by more than one layer of the tiny port (A twice): report one layer's share
    uses = collections.Counter(mod.LAYER_VARIANT[i] for i in range(mod.N_LAYERS))
    res = {}
    layers_total = collections.Counter()
    for variant, c in per_variant.items():
        layers_total.update(c)
        res[f"layer_{variant}"] = collections.Counter({k: v // uses[variant] for k, v in c.items()})
    outside = collections.Counter(whole)
    outside.subtract(layers_total)
    res["outside_layers"] = collections.Counter({k: v for k, v in outside.items() if v})
    res["whole_step"] = whole
    return res


def count_moe_block() -> dict:
    """The expert-parallel MoE block on 4 chips of the fake, forward only (load is not counted): ops for the tiny block and per hit slot."""
    import tempfile

    import torch
    from _pytest.monkeypatch import MonkeyPatch

    import fake_ttnn_compute as fake
    import test_port_template_moe_parallel as t

    mp = MonkeyPatch()
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        text = t.parallel_text(mp, t.TOTAL // 2)
        mod = t.import_port(mp, text, tmp, name="op_count_moe_port")
        _config, block = t._hf_block(11)
        model = mod.load(str(t._checkpoint(tmp, block)), fake.MeshDevice(4))
        x_host = (torch.randn(1, t.SEQ, 64, generator=torch.Generator().manual_seed(5)) * 0.8).to(torch.bfloat16)
        x = fake.Tensor(x_host.reshape(1, 1, t.SEQ, 64).float(), replicated=True, layout="TILE")
        fake.COUNTS.clear()
        mod._A_mlp(model, 0, x, list(range(t.SEQ)))
        res["whole_block"] = collections.Counter(fake.COUNTS)
        res["matmuls"] = res["whole_block"]["matmul"]
    mp.undo()
    res.pop("matmuls")
    return res


def total(c: collections.Counter, skip=("from_torch", "to_torch")) -> int:
    """Device ops: every counted call except the host transfers. The fake's matmul calls its linear, so each matmul is counted twice and is taken off once."""
    return sum(v for k, v in c.items() if k not in skip) - c.get("matmul", 0)


PER_SLOT_OPS = 6          # the emitted per-slot code of `_moe_experts_parallel`: slice of the gathered rows, gate-up matmul, two slices, the silu-multiply, the down matmul
MLP_OPS = 5               # the dense MLP of the tiny layers (three linears, silu, multiply), which the MoE block replaces
TINY_SLOTS = 2           # 8 experts over 4 chips in the tiny MoE test


def project(layer_ops: dict, moe_fixed: int, slots_hit: int, us_per_op: float, groups: int = 12, gdn_per_group: int = 3) -> dict:
    """Ops and floor time of one forward of Flash-Next: `groups` x (`gdn_per_group` gated-delta-net layers + 1 sparse attention layer), the first GDN layer being the
    PLE variant (layer_B), every MLP replaced by the MoE block (`moe_fixed` ops plus PER_SLOT_OPS per hit slot), plus the embedding, prelude, mixer and head."""
    gdn_layers = groups * gdn_per_group
    attn_layers = groups
    mixers = (gdn_layers - 1) * layer_ops["layer_A"] + layer_ops["layer_B"] + attn_layers * layer_ops["layer_C"]
    moe = (gdn_layers + attn_layers) * (moe_fixed + PER_SLOT_OPS * slots_hit)
    outside = layer_ops["embedding_and_prelude"] + layer_ops["mixer_and_head"]
    ops = mixers - (gdn_layers + attn_layers) * MLP_OPS + moe + outside
    return {"ops": ops, "mixer_ops": mixers - (gdn_layers + attn_layers) * MLP_OPS, "moe_ops": moe, "floor_ms": round(ops * us_per_op / 1000, 1), "slots_hit": slots_hit}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--us-per-op", type=float, default=10.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    layers = count_layers()
    moe = count_moe_block()
    result = {"layers": {k: dict(v) for k, v in layers.items() if isinstance(v, collections.Counter)}, "moe_block_4chips_tiny": dict(moe["whole_block"]),
              "layer_variants": layers["layer_variants"], "variant_of_layer": layers["variant_of_layer"]}
    if "decode" in layers:
        result["decode_step"] = {k: dict(v) for k, v in layers["decode"].items()}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    for name, c in layers.items():
        if isinstance(c, collections.Counter):
            print(f"{name:24} {total(c):4} ops  " + "  ".join(f"{k}={v}" for k, v in c.most_common(9)))
    print(f"{'moe_block (4 chips)':24} {total(moe['whole_block']):4} ops  " + "  ".join(f"{k}={v}" for k, v in moe["whole_block"].most_common(9)))
    ops = {k: total(v) for k, v in layers.items() if isinstance(v, collections.Counter)}
    if "decode" in layers:
        d = layers["decode"]
        print(f"\ncached decode step (one row through every layer, state kept): {total(d['whole_step'])} ops in the tiny port")
        for name, c in d.items():
            if name != "whole_step":
                print(f"  {'decode_' + name:30} {total(c):4} ops  " + "  ".join(f"{k}={v}" for k, v in c.most_common(9)))
    if not moe["whole_block"].get("concat"):
        print("\nthis checkout's generator writes the older per-slot MoE form (one embedding and one scatter per slot); the projection below assumes the batched form, so it is skipped")
        return 0
    fixed = total(moe["whole_block"]) - PER_SLOT_OPS * TINY_SLOTS
    print(f"\nthe MoE block has about {fixed} fixed ops and {PER_SLOT_OPS} per hit slot (tiny block: {total(moe['whole_block'])} ops with {TINY_SLOTS} slots)")
    for slots, what in ((10, "one token on 4 chips: about 10 slots hit"), (117, "32 tokens on 4 chips: about 117 slots hit")):
        r = project(ops, fixed, slots, args.us_per_op)
        print(f"Flash-Next forward, {what}: {r['ops']} ops ({r['mixer_ops']} in the mixers and gated residuals, {r['moe_ops']} in the MoE), "
              f"{r['floor_ms']} ms at {args.us_per_op:g} us per op")
    if "decode" in layers:
        d = layers["decode"]
        dec = {k: total(v) for k, v in d.items() if k.startswith("layer_")}
        if set(dec) >= {"layer_A", "layer_B", "layer_C"}:
            dec["embedding_and_prelude"] = total(d["outside_layers"])           # the outside ops of a step are reported as one figure
            dec["mixer_and_head"] = 0
            for slots, what in ((10, "one token on 4 chips: about 10 slots hit"),):
                r = project(dec, fixed, slots, args.us_per_op)
                print(f"Flash-Next cached decode step, {what}: {r['ops']} ops ({r['mixer_ops']} in the mixers and gated residuals, {r['moe_ops']} in the MoE), "
                      f"{r['floor_ms']} ms at {args.us_per_op:g} us per op")
    return 0


if __name__ == "__main__":
    sys.exit(main())
