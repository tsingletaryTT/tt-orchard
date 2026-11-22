# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttl
import ttnn


TILE_SIZE = 32


@ttl.operation(grid="full")
def row_sum(x: ttnn.Tensor, out: ttnn.Tensor) -> None:
    # Input x: [64, 256] elements = [2, 8] tiles
    # Output: [64, 32] elements = [2, 1] tiles
    # We need to sum over the last dimension (8 column tiles) to get 1 column tile
    
    row_tiles = x.shape[0] // TILE_SIZE  # 2
    col_tiles = x.shape[1] // TILE_SIZE  # 8
    
    # DFB shape: (1, 8) to hold 1 row x 8 columns of tiles = 8 tiles
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, col_tiles), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        
        rows_per_node = -(-row_tiles // grid_rows)
        cols_per_node = -(-col_tiles // grid_cols)

        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                with (
                    x_dfb.wait() as x_blk,
                    out_dfb.reserve() as out_blk,
                ):
                    # Reduce over the last dimension (columns)
                    # x_blk has shape (1, 8) tiles
                    # reduced should have shape (1, 1) tiles
                    reduced = ttl.math.reduce_sum(
                        x_blk, dims=[-1], shape=(1, 1)
                    )
                    out_blk.store(reduced)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        
        rows_per_node = -(-row_tiles // grid_rows)
        cols_per_node = -(-col_tiles // grid_cols)

        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * 1, (row + 1) * 1
                # Copy all 8 columns at once
                c0, c1 = 0, col_tiles
                with x_dfb.reserve() as x_blk:
                    tx = ttl.copy(x[r0:r1, c0:c1], x_blk)
                    tx.wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        
        rows_per_node = -(-row_tiles // grid_rows)
        cols_per_node = -(-col_tiles // grid_cols)

        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * 1, (row + 1) * 1
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[r0:r1, 0:1])
                            tx.wait()
