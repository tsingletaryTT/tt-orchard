# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid="full")
def ssd_state_update_bcast(s, decay, dtx, b, out):
    """
    out = decay * s + dtx * b
    
    decay and dtx are [64, 32] and only their column 0 is read (the value of row r is decay[r, 0] and dtx[r, 0])
    b is [32, 128] and only its row 0 is read (the value of column j is b[0, j])
    """
    # Shape in tiles
    s_row_tiles = s.shape[0] // TILE_SIZE
    s_col_tiles = s.shape[1] // TILE_SIZE
    
    # Create dataflow buffers - shape counts tiles
    s_dfb = ttl.make_dataflow_buffer_like(s, shape=(1, 1), block_count=2)
    decay_dfb = ttl.make_dataflow_buffer_like(decay, shape=(1, 1), block_count=2)
    dtx_dfb = ttl.make_dataflow_buffer_like(dtx, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    
    @ttl.datamovement()
    def read():
        """Reader thread: load all inputs once"""
        node_col, node_row = ttl.node(dims=2)
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        
        rows_per_node = -(-s_row_tiles // grid_rows)
        cols_per_node = -(-s_col_tiles // grid_cols)
        
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < s_row_tiles:
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < s_col_tiles:
                        with (
                            s_dfb.reserve() as s_blk,
                            decay_dfb.reserve() as decay_blk,
                            dtx_dfb.reserve() as dtx_blk,
                            b_dfb.reserve() as b_blk,
                        ):
                            # Slices count tiles, not elements
                            tx_s = ttl.copy(s[row:row+1, col:col+1], s_blk)
                            tx_decay = ttl.copy(decay[row:row+1, 0:1], decay_blk)
                            tx_dtx = ttl.copy(dtx[row:row+1, 0:1], dtx_blk)
                            tx_b = ttl.copy(b[0:1, col:col+1], b_blk)
                            tx_s.wait()
                            tx_decay.wait()
                            tx_dtx.wait()
                            tx_b.wait()
    
    @ttl.compute()
    def compute():
        """Compute thread: out[r, c] = decay[r, 0] * s[r, c] + dtx[r, 0] * b[0, c]"""
        node_col, node_row = ttl.node(dims=2)
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        
        rows_per_node = -(-s_row_tiles // grid_rows)
        cols_per_node = -(-s_col_tiles // grid_cols)
        
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < s_row_tiles:
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < s_col_tiles:
                        with (
                            s_dfb.wait() as s_blk,
                            decay_dfb.wait() as decay_blk,
                            dtx_dfb.wait() as dtx_blk,
                            b_dfb.wait() as b_blk,
                            out_dfb.reserve() as out_blk,
                        ):
                            # Broadcast decay and dtx across columns (dim 1)
                            decay_bcast = ttl.block.broadcast(decay_blk, dims=[1], shape=(1, 1))
                            dtx_bcast = ttl.block.broadcast(dtx_blk, dims=[1], shape=(1, 1))
                            
                            # Broadcast b across rows (dim 0)
                            b_bcast = ttl.block.broadcast(b_blk, dims=[0], shape=(1, 1))
                            
                            # Compute: out = decay * s + dtx * b
                            result = decay_bcast * s_blk + dtx_bcast * b_bcast
                            out_blk.store(result)
    
    @ttl.datamovement()
    def write():
        """Writer thread: write results to output tensor"""
        node_col, node_row = ttl.node(dims=2)
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        
        rows_per_node = -(-s_row_tiles // grid_rows)
        cols_per_node = -(-s_col_tiles // grid_cols)
        
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < s_row_tiles:
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < s_col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[row:row+1, col:col+1])
                            tx.wait()
