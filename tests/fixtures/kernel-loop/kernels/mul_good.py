# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# Test fixture for tt-orchard's kernel loop (ladder rung `mul`, ops/mul.json): out = a * b.
# Each node takes whole rows of tiles, in the layout of the center and rmsnorm kernels. Verified on tt-lang-sim.
import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid="full")
def OPERATION(a_in: ttnn.Tensor, b_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = a_in.shape[0] // TILE_SIZE
    col_tiles = a_in.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    nodes = grid_cols * grid_rows
    rows_per_node = -(-row_tiles // nodes)

    a_dfb = ttl.make_dataflow_buffer_like(a_in, shape=(1, col_tiles), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b_in, shape=(1, col_tiles), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, col_tiles), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_id * rows_per_node + local_row
            if row < row_tiles:
                with (
                    a_dfb.wait() as a_blk,
                    b_dfb.wait() as b_blk,
                    out_dfb.reserve() as out_blk,
                ):
                    out_blk.store(a_blk * b_blk)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_id * rows_per_node + local_row
            if row < row_tiles:
                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                    tx_a = ttl.copy(a_in[row : row + 1, 0:col_tiles], a_blk)
                    tx_b = ttl.copy(b_in[row : row + 1, 0:col_tiles], b_blk)
                    tx_a.wait()
                    tx_b.wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_id * rows_per_node + local_row
            if row < row_tiles:
                with out_dfb.wait() as out_blk:
                    tx = ttl.copy(out_blk, out[row : row + 1, 0:col_tiles])
                    tx.wait()
