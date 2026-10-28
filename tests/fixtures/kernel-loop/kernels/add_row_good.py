# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# Test fixture for tt-orchard's kernel loop (ladder rung `add_row`, ops/add_row.json): out = x + r, r (32 rows) repeated down the rows of x; r is read again for every row of tiles.
# Each node takes whole rows of tiles, in the layout of the center and rmsnorm kernels. Verified on tt-lang-sim.
import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid="full")
def OPERATION(x_in: ttnn.Tensor, r_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = x_in.shape[0] // TILE_SIZE
    col_tiles = x_in.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    nodes = grid_cols * grid_rows
    rows_per_node = -(-row_tiles // nodes)

    x_dfb = ttl.make_dataflow_buffer_like(x_in, shape=(1, col_tiles), block_count=2)
    r_dfb = ttl.make_dataflow_buffer_like(r_in, shape=(1, col_tiles), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, col_tiles), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_id * rows_per_node + local_row
            if row < row_tiles:
                with (
                    x_dfb.wait() as x_blk,
                    r_dfb.wait() as r_blk,
                    out_dfb.reserve() as out_blk,
                ):
                    out_blk.store(x_blk + r_blk)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_id * rows_per_node + local_row
            if row < row_tiles:
                with x_dfb.reserve() as x_blk, r_dfb.reserve() as r_blk:
                    tx_x = ttl.copy(x_in[row : row + 1, 0:col_tiles], x_blk)
                    tx_r = ttl.copy(r_in[0:1, 0:col_tiles], r_blk)
                    tx_x.wait()
                    tx_r.wait()

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
