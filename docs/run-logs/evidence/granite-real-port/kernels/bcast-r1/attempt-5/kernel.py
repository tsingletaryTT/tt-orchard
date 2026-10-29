import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid="full")
def ssd_state_update_bcast(s, decay, dtx, b, out):
    # Shape in tiles
    s_row_tiles = s.shape[0] // TILE_SIZE  # 64/32 = 2 tiles
    s_col_tiles = s.shape[1] // TILE_SIZE  # 128/32 = 4 tiles
    
    # Dataflow buffer shapes in tiles
    # We use 1x1 blocks to process one tile at a time
    s_dfb = ttl.make_dataflow_buffer_like(s, shape=(1, 1), block_count=2)
    decay_dfb = ttl.make_dataflow_buffer_like(decay, shape=(1, 1), block_count=2)
    dtx_dfb = ttl.make_dataflow_buffer_like(dtx, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    
    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        
        # Calculate which tiles this node processes
        rows_per_node = -(-s_row_tiles // 1)  # grid="full" means 1 tile per node
        cols_per_node = -(-s_col_tiles // 1)
        
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
                            # decay is [64, 32] = 2x1 tiles, so decay_blk is 1x1
                            # We need decay[row, 0] which is in decay_blk[0, 0]
                            # But we can't index blocks, so we use broadcast
                            
                            # Broadcast decay[row,0] across columns for this row
                            # decay_blk is 1x1, so we broadcast it to 1x1 (no-op) then to 1x4
                            decay_scalar = decay_blk
                            decay_bcast = ttl.block.broadcast(
                                decay_scalar, dims=[-1], shape=(1, 1)
                            )
                            
                            # Broadcast dtx[row,0] across columns for this row
                            dtx_scalar = dtx_blk
                            dtx_bcast = ttl.block.broadcast(
                                dtx_scalar, dims=[-1], shape=(1, 1)
                            )
                            
                            # Broadcast b[0,col] across rows for this column
                            # b is [32, 128] = 1x4 tiles, b_blk is 1x1
                            b_scalar = b_blk
                            b_bcast = ttl.block.broadcast(
                                b_scalar, dims=[0], shape=(1, 1)
                            )
                            
                            # Compute: out = decay * s + dtx * b
                            # First compute decay * s
                            decay_times_s = decay_bcast * s_blk
                            # Then compute dtx * b
                            dtx_times_b = dtx_bcast * b_scalar
                            # Add them together
                            result = decay_times_s + dtx_times_b
                            out_blk.store(result)
    
    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        
        rows_per_node = -(-s_row_tiles // 1)
        cols_per_node = -(-s_col_tiles // 1)
        
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < s_row_tiles:
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < s_col_tiles:
                        # Slices count tiles, not elements
                        with (
                            s_dfb.reserve() as s_blk,
                            decay_dfb.reserve() as decay_blk,
                            dtx_dfb.reserve() as dtx_blk,
                            b_dfb.reserve() as b_blk,
                        ):
                            # Copy s[row:row+1, col:col+1] (1 tile)
                            tx_s = ttl.copy(s[row:row+1, col:col+1], s_blk)
                            # Copy decay[row:row+1, 0:1] (1 tile, column 0)
                            tx_decay = ttl.copy(decay[row:row+1, 0:1], decay_blk)
                            # Copy dtx[row:row+1, 0:1] (1 tile, column 0)
                            tx_dtx = ttl.copy(dtx[row:row+1, 0:1], dtx_blk)
                            # Copy b[0:1, col:col+1] (1 tile, row 0)
                            tx_b = ttl.copy(b[0:1, col:col+1], b_blk)
                            tx_s.wait()
                            tx_decay.wait()
                            tx_dtx.wait()
                            tx_b.wait()
    
    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        
        rows_per_node = -(-s_row_tiles // 1)
        cols_per_node = -(-s_col_tiles // 1)
        
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < s_row_tiles:
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < s_col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[row:row+1, col:col+1])
                            tx.wait()
