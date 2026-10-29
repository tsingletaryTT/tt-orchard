# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Worked tt-lang kernel for ssd_state_update: out = decay * s + dtx * b
# Reference: eltwise_add.py and the pattern from library_silu_mul.py
import ttl
import ttnn

TILE_SIZE = 32
GRANULARITY = 2


@ttl.operation(grid="full")
def ssd_state_update(s_in: ttnn.Tensor, decay_in: ttnn.Tensor, dtx_in: ttnn.Tensor, b_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = s_in.shape[0] // TILE_SIZE // GRANULARITY
    col_tiles = s_in.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)

    s_dfb = ttl.make_dataflow_buffer_like(s_in, shape=(GRANULARITY, 1), block_count=2)
    decay_dfb = ttl.make_dataflow_buffer_like(decay_in, shape=(GRANULARITY, 1), block_count=2)
    dtx_dfb = ttl.make_dataflow_buffer_like(dtx_in, shape=(GRANULARITY, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b_in, shape=(GRANULARITY, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(GRANULARITY, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with (
                            s_dfb.wait() as s_blk,
                            decay_dfb.wait() as decay_blk,
                            dtx_dfb.wait() as dtx_blk,
                            b_dfb.wait() as b_blk,
                            out_dfb.reserve() as out_blk,
                        ):
                            out_blk.store(decay_blk * s_blk + dtx_blk * b_blk)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with s_dfb.reserve() as s_blk, decay_dfb.reserve() as decay_blk, dtx_dfb.reserve() as dtx_blk, b_dfb.reserve() as b_blk:
                            tx_s = ttl.copy(s_in[r0:r1, col : col + 1], s_blk)
                            tx_decay = ttl.copy(decay_in[r0:r1, col : col + 1], decay_blk)
                            tx_dtx = ttl.copy(dtx_in[r0:r1, col : col + 1], dtx_blk)
                            tx_b = ttl.copy(b_in[r0:r1, col : col + 1], b_blk)
                            tx_s.wait()
                            tx_decay.wait()
                            tx_dtx.wait()
                            tx_b.wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[r0:r1, col : col + 1])
                            tx.wait()
