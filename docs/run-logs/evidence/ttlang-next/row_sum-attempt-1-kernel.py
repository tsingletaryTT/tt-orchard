# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid="full")
def row_sum(x: ttnn.Tensor, out: ttnn.Tensor) -> None:
    # Input x has shape [64, 256] in elements
    # That's 64 tiles in row dimension, 8 tiles in column dimension (256/32 = 8)
    # Output has shape [64, 32] in elements
    # That's 64 tiles in row dimension, 1 tile in column dimension
    
    # Number of tiles
    row_tiles = x.shape[0] // TILE_SIZE  # 64
    col_tiles = x.shape[1] // TILE_SIZE  # 8
    
    # Get grid size
    grid_cols, grid_rows = ttl.grid_size(dims=2)
    
    # Calculate how many tiles each node should process
    rows_per_node = -(-row_tiles // grid_rows)  # ceiling division
    
    # Create dataflow buffers
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, col_tiles), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                with (
                    x_dfb.wait() as x_blk,
                    out_dfb.reserve() as out_blk,
                ):
                    # x_blk is a block of shape (1, col_tiles) in tiles
                    # We need to sum across the column dimension (dim -1 or dim 1)
                    # This reduces the 1x8 tile block to 1x1
                    result = ttl.math.reduce_sum(x_blk, dims=[-1], shape=(1, 1))
                    
                    # Store the result in out_blk
                    out_blk.store(result)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                with x_dfb.reserve() as x_blk:
                    # Copy the row slice from x to x_blk
                    # x has shape (64, 8) in tiles, so we copy x[row, 0:8] to x_blk
                    tx = ttl.copy(x[row:row+1, 0:col_tiles], x_blk)
                    tx.wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                with out_dfb.wait() as out_blk:
                    # Copy the result from out_blk to out[row, 0:1]
                    tx = ttl.copy(out_blk, out[row:row+1, 0:1])
                    tx.wait()
