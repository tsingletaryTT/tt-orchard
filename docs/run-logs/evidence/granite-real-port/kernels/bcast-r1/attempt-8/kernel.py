# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# TT-Lang kernel for ssd_state_update_bcast: out = decay * s + dtx * b
# decay and dtx are [64, 32] with only column 0 used (value decay[r,0] and dtx[r,0])
# b is [32, 128] with only row 0 used (value b[0,j])
# Result: out[r,j] = decay[r,0] * s[r,j] + dtx[r,0] * b[0,j]
import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid="full")
def ssd_state_update_bcast(
    s_in: ttnn.Tensor,
    decay_in: ttnn.Tensor,
    dtx_in: ttnn.Tensor,
    b_in: ttnn.Tensor,
    out: ttnn.Tensor,
) -> None:
    row_tiles = s_in.shape[0] // TILE_SIZE
    col_tiles = s_in.shape[1] // TILE_SIZE

    # Spread the rows of tiles over every node of the grid
    grid_cols, grid_rows = ttl.grid_size(dims=2)
    nodes = grid_cols * grid_rows
    rows_per_node = -(-row_tiles // nodes)

    # Dataflow buffers
    # s: [64, 128] - 2x4 tiles, each node handles rows_per_node rows
    # decay: [64, 32] - 2x1 tiles, each node handles rows_per_node rows of column 0
    # dtx: [64, 32] - 2x1 tiles, each node handles rows_per_node rows of column 0
    # b: [32, 128] - 1x4 tiles, full row 0 shared by all nodes
    
    # For s, each node handles rows_per_node rows, each row has col_tiles tiles
    s_dfb = ttl.make_dataflow_buffer_like(s_in, shape=(rows_per_node, col_tiles), block_count=2)
    
    # For decay and dtx, each node handles rows_per_node rows of column 0 only
    decay_dfb = ttl.make_dataflow_buffer_like(decay_in, shape=(rows_per_node, 1), block_count=2)
    dtx_dfb = ttl.make_dataflow_buffer_like(dtx_in, shape=(rows_per_node, 1), block_count=2)
    
    # For b, all nodes share the same row 0 (which has col_tiles tiles)
    b_dfb = ttl.make_dataflow_buffer_like(b_in, shape=(1, col_tiles), block_count=2)
    
    # Output buffer: same as s
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(rows_per_node, col_tiles), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_id * rows_per_node + local_row
            if row < row_tiles:
                with (
                    s_dfb.wait() as s_blk,
                    decay_dfb.wait() as decay_blk,
                    dtx_dfb.wait() as dtx_blk,
                    b_dfb.wait() as b_blk,
                    out_dfb.reserve() as out_blk,
                ):
                    # s_blk: (rows_per_node, col_tiles) - one row of s per node
                    # decay_blk: (rows_per_node, 1) - column 0 of decay per node
                    # dtx_blk: (rows_per_node, 1) - column 0 of dtx per node
                    # b_blk: (1, col_tiles) - row 0 of b (shared)

                    # Broadcast decay_blk across columns to match s_blk shape
                    decay_row = ttl.block.broadcast(decay_blk, dims=[-1], shape=(rows_per_node, col_tiles))
                    
                    # Broadcast dtx_blk across columns to match s_blk shape
                    dtx_row = ttl.block.broadcast(dtx_blk, dims=[-1], shape=(rows_per_node, col_tiles))
                    
                    # Broadcast b_blk across rows to match s_blk shape
                    b_row = ttl.block.broadcast(b_blk, dims=[0], shape=(rows_per_node, col_tiles))
                    
                    # Compute: out = decay * s + dtx * b
                    out_blk.store(decay_row * s_blk + dtx_row * b_row)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_id * rows_per_node + local_row
            if row < row_tiles:
                with (
                    s_dfb.reserve() as s_blk,
                    decay_dfb.reserve() as decay_blk,
                    dtx_dfb.reserve() as dtx_blk,
                    b_dfb.reserve() as b_blk,
                ):
                    # Copy full row s[row, 0:col_tiles]
                    tx_s = ttl.copy(s_in[row : row + 1, 0:col_tiles], s_blk)
                    
                    # Copy decay[row, 0:1] (just column 0)
                    tx_decay = ttl.copy(decay_in[row : row + 1, 0:1], decay_blk)
                    
                    # Copy dtx[row, 0:1] (just column 0)
                    tx_dtx = ttl.copy(dtx_in[row : row + 1, 0:1], dtx_blk)
                    
                    # Copy b[0, 0:col_tiles] (full row 0)
                    tx_b = ttl.copy(b_in[0:1, 0:col_tiles], b_blk)

                    tx_s.wait()
                    tx_decay.wait()
                    tx_dtx.wait()
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
