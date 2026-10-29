import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid="full")
def ssd_state_update_bcast(s, decay, dtx, b, out):
    # Shape in tiles
    s_row_tiles = s.shape[0] // TILE_SIZE  # 2
    s_col_tiles = s.shape[1] // TILE_SIZE  # 4
    
    # Dataflow buffers - hold the whole column/row vectors
    s_dfb = ttl.make_dataflow_buffer_like(s, shape=(1, 1), block_count=2)
    decay_dfb = ttl.make_dataflow_buffer_like(decay, shape=(s_row_tiles, 1), block_count=2)
    dtx_dfb = ttl.make_dataflow_buffer_like(dtx, shape=(s_row_tiles, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, s_col_tiles), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        
        # Process each tile of s
        for local_row in range(s_row_tiles):
            row = node_row * s_row_tiles + local_row
            if row < s_row_tiles:
                for local_col in range(s_col_tiles):
                    col = node_col * s_col_tiles + local_col
                    if col < s_col_tiles:
                        with (
                            s_dfb.wait() as s_blk,
                            decay_dfb.wait() as decay_blk,
                            dtx_dfb.wait() as dtx_blk,
                            b_dfb.wait() as b_blk,
                            out_dfb.reserve() as out_blk,
                        ):
                            # Broadcast decay[:, 0] across columns to match s shape
                            decay_bcast = ttl.block.broadcast(decay_blk, dims=[-1], shape=(s_row_tiles, s_col_tiles))
                            # Broadcast dtx[:, 0] across columns to match s shape
                            dtx_bcast = ttl.block.broadcast(dtx_blk, dims=[-1], shape=(s_row_tiles, s_col_tiles))
                            # Broadcast b[0, :] across rows to match s shape
                            b_bcast = ttl.block.broadcast(b_blk, dims=[0], shape=(s_row_tiles, s_col_tiles))
                            
                            # Compute: out = decay * s + dtx * b
                            result = decay_bcast * s_blk + dtx_bcast * b_bcast
                            out_blk.store(result)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        
        for local_row in range(s_row_tiles):
            row = node_row * s_row_tiles + local_row
            if row < s_row_tiles:
                for local_col in range(s_col_tiles):
                    col = node_col * s_col_tiles + local_col
                    if col < s_col_tiles:
                        with (
                            s_dfb.reserve() as s_blk,
                            decay_dfb.reserve() as decay_blk,
                            dtx_dfb.reserve() as dtx_blk,
                            b_dfb.reserve() as b_blk,
                        ):
                            # Copy s[row, col]
                            tx_s = ttl.copy(s[row:row+1, col:col+1], s_blk)
                            # Copy decay[row, 0] - only column 0 (tile 0)
                            tx_decay = ttl.copy(decay[row:row+1, 0:1], decay_blk)
                            # Copy dtx[row, 0] - only column 0 (tile 0)
                            tx_dtx = ttl.copy(dtx[row:row+1, 0:1], dtx_blk)
                            # Copy b[0, col] - only row 0 (tile 0)
                            tx_b = ttl.copy(b[0:1, col:col+1], b_blk)
                            
                            tx_s.wait()
                            tx_decay.wait()
                            tx_dtx.wait()
                            tx_b.wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        
        for local_row in range(s_row_tiles):
            row = node_row * s_row_tiles + local_row
            if row < s_row_tiles:
                for local_col in range(s_col_tiles):
                    col = node_col * s_col_tiles + local_col
                    if col < s_col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[row:row+1, col:col+1])
                            tx.wait()
