# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid="full")
def row_sum(x: ttnn.Tensor, out: ttnn.Tensor) -> None:
    # x shape: [64, 256] elements = [2, 8] tiles
    # out shape: [64, 32] elements = [2, 1] tiles
    
    row_tiles = x.shape[0] // TILE_SIZE
    col_tiles = x.shape[1] // TILE_SIZE
    
    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    
    # Create dataflow buffers with shape (1, 1) - each buffer holds one tile
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    
    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                # Accumulate sum across all columns
                total = None
                for local_col in range(col_tiles):
                    with x_dfb.wait() as x_blk:
                        summed = ttl.math.reduce_sum(x_blk, dims=[-1], shape=(1, 1))
                        if total is None:
                            total = summed
                        else:
                            total += summed
                
                # Store the accumulated result
                if total is not None:
                    with out_dfb.reserve() as out_blk:
                        out_blk.store(total)
    
    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                for local_col in range(col_tiles):
                    with x_dfb.reserve() as x_blk:
                        tx = ttl.copy(x[row : row + 1, local_col : local_col + 1], x_blk)
                        tx.wait()
    
    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                with out_dfb.wait() as out_blk:
                    tx = ttl.copy(out_blk, out[row : row + 1, 0:1])
                    tx.wait()
