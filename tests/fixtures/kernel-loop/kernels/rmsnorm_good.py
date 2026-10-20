# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# Test fixture for tt-orchard's kernel loop: out = x * rsqrt(mean(x^2, last axis) + 1e-6) * w, with w given at
# full size (ops/rmsnorm.json). Each node takes whole rows of tiles: a row of tiles holds 32 rows of x across
# the full last axis, so the mean over the last axis needs nothing from any other node.
# Verified on tt-lang-sim at PCC >= 0.999 on every input set of ops/rmsnorm.json.
import ttl
import ttnn

TILE_SIZE = 32
EPS = 1e-6


@ttl.operation(grid="full")
def OPERATION(x_in: ttnn.Tensor, w_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = x_in.shape[0] // TILE_SIZE
    col_tiles = x_in.shape[1] // TILE_SIZE
    inv_cols = 1.0 / x_in.shape[1]

    # Spread the rows of tiles over every node of the grid, one contiguous run of rows per node.
    grid_cols, grid_rows = ttl.grid_size(dims=2)
    nodes = grid_cols * grid_rows
    rows_per_node = -(-row_tiles // nodes)

    # One block is one row of tiles: (1, col_tiles).
    x_dfb = ttl.make_dataflow_buffer_like(x_in, shape=(1, col_tiles), block_count=2)
    w_dfb = ttl.make_dataflow_buffer_like(w_in, shape=(1, col_tiles), block_count=2)
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
                    w_dfb.wait() as w_blk,
                    out_dfb.reserve() as out_blk,
                ):
                    # Sum of squares over the last axis: one value per row, held in column 0 of each tile.
                    sum_sq = ttl.math.reduce_sum(x_blk * x_blk, dims=[-1], shape=(1, 1))
                    mean_sq = sum_sq * ttl.block.fill(inv_cols, shape=(1, 1))
                    inv_rms = ttl.math.rsqrt(mean_sq + ttl.block.fill(EPS, shape=(1, 1)))
                    # Spread column 0 across each tile and then across the row of tiles.
                    inv_rms_row = ttl.block.broadcast(inv_rms, dims=[-1], shape=(1, col_tiles))
                    out_blk.store(x_blk * inv_rms_row * w_blk)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_id * rows_per_node + local_row
            if row < row_tiles:
                with x_dfb.reserve() as x_blk, w_dfb.reserve() as w_blk:
                    tx_x = ttl.copy(x_in[row : row + 1, 0:col_tiles], x_blk)
                    tx_w = ttl.copy(w_in[row : row + 1, 0:col_tiles], w_blk)
                    tx_x.wait()
                    tx_w.wait()

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
