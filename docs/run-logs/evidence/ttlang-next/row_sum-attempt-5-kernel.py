# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid="full")
def row_sum(x: ttnn.Tensor, out: ttnn.Tensor) -> None:
    # Input x shape: [64, 256] elements = [2, 8] tiles
    # Output out shape: [64, 32] elements = [2, 1] tiles
    # We need to sum across columns (dim=1) for each row

    # Shape in tiles
    row_tiles = x.shape[0] // TILE_SIZE  # 2
    col_tiles = x.shape[1] // TILE_SIZE  # 8

    # Create dataflow buffers
    # x_dfb shape is in tiles: (1, col_tiles) means each block holds 1 row of tiles with col_tiles columns
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, col_tiles), block_count=2)
    # out_dfb shape is in tiles: (1, 1) means each block holds 1x1 tile
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        
        # Each node reads its portion of rows
        # grid_rows is the number of rows in the grid
        grid_rows = ttl.grid_size(dims=2)[1]
        rows_per_node = -(-row_tiles // grid_rows)
        
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                # Copy the entire row (all col_tiles columns) from x to x_dfb
                # x_dfb shape is (1, col_tiles) tiles, so we need to copy row 'row' from x
                # which is x[row:row+1, 0:col_tiles] in tile indices
                with x_dfb.reserve() as x_blk:
                    tx = ttl.copy(x[row : row + 1, 0 : col_tiles], x_blk)
                    tx.wait()

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        
        grid_rows = ttl.grid_size(dims=2)[1]
        rows_per_node = -(-row_tiles // grid_rows)
        
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                # Wait for x block and reserve output block
                with x_dfb.wait() as x_blk, out_dfb.reserve() as out_blk:
                    # x_blk has shape (1, col_tiles) tiles
                    # We want to reduce over the last dimension (columns) to get a single value per row
                    # The result should have shape (1, 1) tiles
                    sum_result = ttl.math.reduce_sum(x_blk, dims=[-1], shape=(1, 1))
                    
                    # Store the result
                    out_blk.store(sum_result)

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        
        grid_rows = ttl.grid_size(dims=2)[1]
        rows_per_node = -(-row_tiles // grid_rows)
        
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                # Wait for the output block and copy to actual output tensor
                with out_dfb.wait() as out_blk:
                    # out_dfb shape is (1, 1) tiles, so we copy to out[row:row+1, 0:1] in tile indices
                    tx = ttl.copy(out_blk, out[row : row + 1, 0 : 1])
                    tx.wait()
