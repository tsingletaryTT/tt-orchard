# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid="full")
def center(x_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    """Center operation: out = x - mean(x, last axis).
    
    The mean is taken over the 256 elements of each row (last axis).
    Input shape: (64, 256), Output shape: (64, 256)
    """
    row_tiles = x_in.shape[0] // TILE_SIZE  # 64 // 32 = 2
    col_tiles = x_in.shape[1] // TILE_SIZE  # 256 // 32 = 8
    
    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)
    
    # DFB shape is (1, col_tiles) to process all columns for a row together
    x_dfb = ttl.make_dataflow_buffer_like(x_in, shape=(1, col_tiles), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, col_tiles), block_count=2)
    
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
                            x_dfb.wait() as x_blk,
                            out_dfb.reserve() as out_blk,
                        ):
                            # x_blk shape in tiles: (1, 8)
                            # Compute row sum over last axis (dim -1)
                            # reduce_sum over dims=[-1] gives (1, 1)
                            row_sum = ttl.math.reduce_sum(
                                x_blk, dims=[-1], shape=(1, 1)
                            )
                            
                            # Broadcast row sum across all columns to get (1, 8)
                            mean_full = ttl.block.broadcast(
                                row_sum, dims=[-1], shape=(1, col_tiles)
                            )
                            
                            # Create a block with 1/256 constant
                            one_over_256 = ttl.block.fill(
                                1.0 / 256.0, shape=(1, 1)
                            )
                            
                            # Broadcast the scalar to match mean_full shape
                            one_over_256_bcast = ttl.block.broadcast(
                                one_over_256, dims=[-1], shape=(1, col_tiles)
                            )
                            
                            # Scale by 1/256 to get mean
                            scaled_mean = mean_full * one_over_256_bcast
                            
                            # Center: x - mean (both should be (1, 8))
                            centered = x_blk - scaled_mean
                            
                            out_blk.store(centered)
    
    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with x_dfb.reserve() as x_blk:
                            tx = ttl.copy(x_in[row:row+1, 0:col_tiles], x_blk)
                            tx.wait()
    
    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[row:row+1, 0:col_tiles])
                            tx.wait()
