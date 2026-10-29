# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Split a traced decode step into input copy, trace execution and logits read back."""
import sys
import time
from pathlib import Path

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chip_model import ChipMiniCPM3  # noqa: E402

opt, upd = int(sys.argv[1]), sys.argv[2]
dev = ttnn.open_device(device_id=0, trace_region_size=256 * 1024 * 1024)
try:
    m = ChipMiniCPM3(dev, "/home/ttuser/.cache/orchard-overnight2/minicpm3", "/home/ttuser/.cache/orchard-overnight2/minicpm3.safetensors",
                     attn="mla", max_len=256, upd=upd, opt=opt)
    variants = (sys.argv[3] if len(sys.argv) > 3 else "").split(";")      # skip sets separated by ;  (an empty one is the full step)
    for var in variants:
        m.skip = set(x for x in var.split(",") if x)
        m.enable_trace()
        for rep in range(2):
            tc = te = tr = 0.0
            n = 30
            for pos in range(n):
                t0 = time.time()
                h = m._host_inputs(5, pos)
                for k, v in h.items():
                    ttnn.copy_host_to_device_tensor(v, m.t_in[k])
                t1 = time.time()
                ttnn.execute_trace(m.dev, m.trace_id, cq_id=0, blocking=True)
                t2 = time.time()
                lg = m._host(m.t_out)
                t3 = time.time()
                tc += t1 - t0
                te += t2 - t1
                tr += t3 - t2
            if rep:
                print(f"opt {opt} upd {upd} skip {sorted(m.skip)}: copy {tc / n * 1e3:.2f} ms, execute {te / n * 1e3:.2f} ms, read {tr / n * 1e3:.2f} ms", flush=True)
        ttnn.release_trace(m.dev, m.trace_id)
finally:
    ttnn.close_device(dev)
