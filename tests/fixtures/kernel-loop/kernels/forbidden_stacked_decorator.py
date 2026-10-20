# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Test fixture for tt-orchard's kernel loop: add_good (upstream eltwise_add.py, renamed OPERATION, host code
# removed) with a second decorator stacked over @ttl.operation. The gate must
# refuse it (forbidden_code, rule operation).
import ttl
import ttnn

TILE_SIZE = 32
GRANULARITY = 2


@staticmethod
@ttl.operation(grid="full")
def OPERATION(a_in: ttnn.Tensor, b_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = a_in.shape[0] // TILE_SIZE // GRANULARITY
    col_tiles = a_in.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)

    a_dfb = ttl.make_dataflow_buffer_like(a_in, shape=(GRANULARITY, 1), block_count=2)
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
                            a_dfb.wait() as a_blk,
                            b_dfb.wait() as b_blk,
                            out_dfb.reserve() as out_blk,
                        ):
                            out_blk.store(a_blk + b_blk)

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
                        with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                            tx_a = ttl.copy(a_in[r0:r1, col : col + 1], a_blk)
                            tx_b = ttl.copy(b_in[r0:r1, col : col + 1], b_blk)
                            tx_a.wait()
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

