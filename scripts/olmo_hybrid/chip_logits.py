#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Teacher-forced logits of the Olmo-Hybrid-7B port on a 1x4 mesh, for the prompts in a reference-ids file. Run it under a four-chip lease.

    python3 scripts/olmo_hybrid/chip_logits.py SNAPSHOT_DIR REFERENCE_IDS.json OUT.json

Same shape as ref_logits.py: for each prompt the reference's continuation is fed in one token at a time and every position keeps its top five
logits. Uses the port module in pkgsrc/runtime and the tokenizer in the snapshot.
"""
import importlib.util
import json
import sys
from pathlib import Path


def main():
    import torch
    import ttnn
    from tokenizers import Tokenizer
    snap, ref_file, out = sys.argv[1:4]
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("olmo_hybrid_port", here.parent.parent / "pkgsrc" / "runtime" / "olmo_hybrid_port.py")
    port = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(port)
    tk = Tokenizer.from_file(str(Path(snap) / "tokenizer.json"))
    ref = json.load(open(ref_file))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    items = []
    try:
        model = port.load(snap, dev)
        for it in ref["items"]:
            ids = tk.encode(it["prompt"]).ids
            pos = []
            logits = port.prefill(model, ids)
            for k, forced in enumerate(it["ids"]):
                x = logits.reshape(-1).float()
                v, i = x.topk(5)
                pos.append({"k": k, "top": [{"id": int(a), "text": tk.decode([int(a)]), "logit": round(float(b), 4)} for a, b in zip(i, v)],
                            "gap_top1_top2": round(float(v[0] - v[1]), 4), "reference_choice": forced,
                            "logit_of_reference_choice": round(float(x[forced]), 4)})
                if k + 1 < len(it["ids"]):
                    logits = port.decode_step(model, forced, len(ids) + k)
            items.append({"prompt": it["prompt"], "prompt_tokens": len(ids), "positions": pos})
            print("done", it["prompt"][:30], flush=True)
        port.close(model)
    finally:
        ttnn.close_mesh_device(dev)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    json.dump({"model": "allenai/Olmo-Hybrid-7B", "device": "1x4 mesh", "items": items}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
