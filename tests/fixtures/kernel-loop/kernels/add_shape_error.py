# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# TTLANG_HARDWARE_CI: skip-compiler
# type: ignore
#
# Test fixture for tt-orchard's kernel loop: upstream tt-lang examples/errors/eltwise_add_error.py with the
# function renamed OPERATION and the host code removed. It copies one tile into a 2x2 block, so the
# check must end shape_mismatch.

import ttl
import ttnn


@ttl.operation(grid=(1, 1))
def OPERATION(a, b, y):
    row_tiles = 2
    col_tiles = 2

    rows = 2
    cols = 2

    a_dfb = ttl.make_dataflow_buffer_like(
        a, shape=(row_tiles, col_tiles), block_count=2
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b, shape=(row_tiles, col_tiles), block_count=2
    )
    y_dfb = ttl.make_dataflow_buffer_like(
        y, shape=(row_tiles, col_tiles), block_count=2
    )

    @ttl.compute()
    def add_compute():
        for _ in range(rows):
            for _ in range(cols):
                with a_dfb.wait() as a, b_dfb.wait() as b, y_dfb.reserve() as y:
                    y.store(a + b)

    @ttl.datamovement()
    def add_read():
        for r in range(rows):
            for c in range(cols):
                with a_dfb.reserve() as a_block, b_dfb.reserve() as b_block:
                    tx_a = ttl.copy(a[r, c], a_block)
                    tx_b = ttl.copy(b[r, c], b_block)

                    tx_a.wait()
                    tx_b.wait()

    @ttl.datamovement()
    def add_write():
        for r in range(rows):
            for c in range(cols):
                with y_dfb.wait() as y_block:
                    tx = ttl.copy(y_dfb, y[r, c])
                    tx.wait()

