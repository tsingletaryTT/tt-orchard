# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run a few steps of a truncated stack under the device profiler (python -m tracy -r -p ... prof_step.py). The ops report has one row per op."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ttnn  # noqa: E402
from chip_model import ChipMiniCPM3  # noqa: E402

dev = ttnn.open_device(device_id=0)
try:
    m = ChipMiniCPM3(dev, "/home/ttuser/.cache/orchard-overnight2/minicpm3", "/home/ttuser/.cache/orchard-overnight2/minicpm3.safetensors",
                     attn="mla", max_len=256, n_layers=2, upd=sys.argv[2] if len(sys.argv) > 2 else "onehot",
                     opt=int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    for p, t in enumerate([1, 5, 9, 12]):
        m.run(t, p)
    ttnn.ReadDeviceProfiler(dev)
finally:
    ttnn.close_device(dev)
