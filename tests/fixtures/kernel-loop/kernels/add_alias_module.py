# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Test fixture for tt-orchard's kernel loop: add_good (upstream eltwise_add.py, renamed OPERATION, host code
# removed) with ttl imported as tl. The gate must accept @tl.operation and the check must pass.
import ttl as tl
import ttnn

TILE_SIZE = 32
GRANULARITY = 2


@tl.operation(grid="full")
def OPERATION(a_in: ttnn.Tensor, b_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = a_in.shape[0] // TILE_SIZE // GRANULARITY
    col_tiles = a_in.shape[1] // TILE_SIZE

    grid_cols, grid_rows = tl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)

    a_dfb = tl.make_dataflow_buffer_like(a_in, shape=(GRANULARITY, 1), block_count=2)
    b_dfb = tl.make_dataflow_buffer_like(b_in, shape=(GRANULARITY, 1), block_count=2)
    out_dfb = tl.make_dataflow_buffer_like(out, shape=(GRANULARITY, 1), block_count=2)

    @tl.compute()
    def compute():
        node_col, node_row = tl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with (
                            a_dfb.wait() as a_blk,
                            b_dfb.wait() as b_blk,
                            out_dfb.reserve() as out_blk,
                        ):
                            out_blk.store(a_blk + b_blk)

    @tl.datamovement()
    def read():
        node_col, node_row = tl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                            tx_a = tl.copy(a_in[r0:r1, col : col + 1], a_blk)
                            tx_b = tl.copy(b_in[r0:r1, col : col + 1], b_blk)
                            tx_a.wait()
                            tx_b.wait()

    @tl.datamovement()
    def write():
        node_col, node_row = tl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = tl.copy(out_blk, out[r0:r1, col : col + 1])
                            tx.wait()

