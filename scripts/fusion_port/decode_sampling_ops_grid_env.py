# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-device top-k of the decode logits rows for the on-device sampler (models/common/sampling/tt_sampling.py).

The sampler's per-device step on the (1, 4) mesh masked the vocab tail, padded the [32, V / 4] logits to a power of two
and ran ttnn.topk over all 32 tile rows: 333 us per decode step (279 for the top-k) on Ornith (V / 4 = 62080), with
one live row at batch 1. RowsTopK reads only the live rows: `ncores` scan cores each keep the K best of their tiles
per row (kernels/topk_scan.cpp), one merge core merges the sorted lists (kernels/topk_merge.cpp) and writes what
ttnn.topk returns (values BF16 [1, 1, 32, K] TILE, local indices UINT16 [1, 1, 32, K] TILE, sorted descending, ties to
the lower index; rows >= B copies of row 0). The sampler's all-gathers, index offsets and draw are unchanged.
Widths above 65536 columns (Ornith / 122B on 2 dies: 124160) return UINT32 indices, as ttnn.topk's large-index route
does: the scan cores key on their own column range (always < 65536), the merge adds each core's first column.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
import os
GRID = tuple(int(v) for v in os.environ.get("ORCHARD_SAMPLER_GRID", "10,10").split(","))  # scan cores (env override: fusion port ablation)
MERGE_CORE = (10, 0)
MAX_ROWS = 8
MAX_WIDTH = 1 << 20  # columns per device (indices UINT32 above 65536)


class RowsTopK:
    def __init__(self, mesh_device, width, k=32):
        assert width % 32 == 0 and k == 32
        assert width <= MAX_WIDTH, width
        self.wide = width > 0xFFFF + 1  # UINT32 indices
        self.mesh_device = mesh_device
        self.tiles = width // 32
        self.k = k
        cores = [ttnn.CoreCoord(x, y) for y in range(GRID[1]) for x in range(GRID[0])]
        per = -(-self.tiles // len(cores))
        self.work = []
        t = 0
        for c in cores:
            n = min(per, self.tiles - t)
            if n <= 0:
                break
            self.work.append((c, t, n))
            t += n
        self.per = per
        self.scan_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in self.work])
        self.merge = ttnn.CoreCoord(*MERGE_CORE)
        self.merge_set = ttnn.CoreRangeSet([ttnn.CoreRange(self.merge, self.merge)])
        phys = mesh_device.worker_core_from_logical_core(self.merge)
        self.merge_xy = (phys.x, phys.y)

    def __call__(self, logits, rows):
        """logits: [1, 1, 32, width] BF16 TILE (rows 0 .. rows-1 live). Returns (values, indices) as ttnn.topk(k=32)."""
        B, K, nc = rows, self.k, len(self.work)
        assert 1 <= B <= MAX_ROWS
        mk = lambda dt: ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, K]), dt, ttnn.TILE_LAYOUT, self.mesh_device,
                                                       ttnn.L1_MEMORY_CONFIG)
        values, indices = mk(ttnn.bfloat16), mk(ttnn.uint32 if self.wide else ttnn.uint16)
        mbytes = nc * B * K * 4
        pages = -(-mbytes // 2048)
        merge_buf = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 32, 32 * pages]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh_device,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                              ttnn.ShardSpec(self.merge_set, (32, 32 * pages), ttnn.ShardOrientation.ROW_MAJOR)),
        )
        cb_buf, cb_list, cb_out = 0, 1, 2
        rows_bytes = -(-B * 32 // 64) * 64
        fmt = lambda i, size, cores: ttnn.CBDescriptor(
            total_size=size, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=size)])
        cbs = [fmt(cb_buf, self.per * 2 * rows_bytes, self.scan_set), fmt(cb_list, B * K * 4, self.scan_set),
               fmt(cb_out, 2048 + (4096 if self.wide else 2048), self.merge_set)]
        scan_ct = [B, K, cb_buf, cb_list, self.merge_xy[0], self.merge_xy[1]]
        scan_ct += ttnn.TensorAccessorArgs(logits).get_compile_time_args()
        scan_rt = ttnn.RuntimeArgs()
        for i, (c, t0, n) in enumerate(self.work):
            scan_rt[c.x][c.y] = [logits.buffer_address(), t0, n, merge_buf.buffer_address(), i]
        merge_ct = [B, K, nc, cb_out, int(self.wide)]
        merge_ct += ttnn.TensorAccessorArgs(values).get_compile_time_args()
        merge_ct += ttnn.TensorAccessorArgs(indices).get_compile_time_args()
        merge_rt = ttnn.RuntimeArgs()
        merge_rt[self.merge.x][self.merge.y] = [merge_buf.buffer_address(), values.buffer_address(),
                                                indices.buffer_address()] + [t0 * 32 for _, t0, _ in self.work]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "topk_scan.cpp"), core_ranges=self.scan_set,
                                  compile_time_args=scan_ct, runtime_args=scan_rt,
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
            ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "topk_merge.cpp"), core_ranges=self.merge_set,
                                  compile_time_args=merge_ct, runtime_args=merge_rt,
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
        ]
        sems = [ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, self.merge_set, 0)]
        ttnn.generic_op([logits, merge_buf, values, indices],
                        ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        ttnn.deallocate(merge_buf)
        return values, indices
