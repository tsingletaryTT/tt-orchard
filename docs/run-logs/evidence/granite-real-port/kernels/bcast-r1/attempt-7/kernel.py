import ttl
import ttnn

TILE_SIZE = 32
GRANULARITY = 2


@ttl.operation(grid="full")
def ssd_state_update_bcast(s_in: ttnn.Tensor, decay_in: ttnn.Tensor, dtx_in: ttnn.Tensor, b_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = s_in.shape[0] // TILE_SIZE // GRANULARITY
    col_tiles = s_in.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)

    s_dfb = ttl.make_dataflow_buffer_like(s_in, shape=(GRANULARITY, 1), block_count=2)
    decay_dfb = ttl.make_dataflow_buffer_like(decay_in, shape=(GRANULARITY, 1), block_count=2)
    dtx_dfb = ttl.make_dataflow_buffer_like(dtx_in, shape=(GRANULARITY, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b_in, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(GRANULARITY, 1), block_count=2)

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
                            s_dfb.wait() as s_blk,
                            decay_dfb.wait() as decay_blk,
                            dtx_dfb.wait() as dtx_blk,
                            b_dfb.wait() as b_blk,
                            out_dfb.reserve() as out_blk,
                        ):
                            # decay_blk is [GRANULARITY, 1] = [2, 1] tiles = 64x32 elements
                            # dtx_blk is [GRANULARITY, 1] = [2, 1] tiles = 64x32 elements
                            # b_blk is [1, 1] = 1 tile = 32x32 elements
                            # s_blk is [GRANULARITY, 1] = [2, 1] tiles = 64x32 elements
                            
                            # We need to broadcast:
                            # - decay_blk across columns (dim=-1) to get [2, 2] tiles? No, we need [2, 1] tiles
                            # - dtx_blk across columns (dim=-1) to get [2, 2] tiles? No, we need [2, 1] tiles
                            # - b_blk across rows (dim=0) to get [2, 1] tiles
                            
                            # Actually, let's think about this more carefully.
                            # For output tile at (row, col):
                            # - We have decay_blk holding decay[row*2:(row+1)*2, 0] = 64x1 elements
                            # - We have dtx_blk holding dtx[row*2:(row+1)*2, 0] = 64x1 elements
                            # - We have b_blk holding b[0, col] = 1x32 elements
                            # - We have s_blk holding s[row*2:(row+1)*2, col] = 64x32 elements
                            
                            # We want:
                            # - decay_blk * s_blk: decay_blk is [2, 1], s_blk is [2, 1], result is [2, 1]
                            # - dtx_blk * b_blk: dtx_blk is [2, 1], b_blk is [1, 1], need to broadcast b_blk to [2, 1]
                            
                            # Broadcast b_blk across rows (dim=0) to get [2, 1]
                            b_bcast = ttl.block.broadcast(b_blk, dims=[0], shape=(GRANULARITY, 1))
                            
                            # Compute: out = decay * s + dtx * b
                            result = decay_blk * s_blk + dtx_blk * b_bcast
                            out_blk.store(result)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with (
                            s_dfb.reserve() as s_blk,
                            decay_dfb.reserve() as decay_blk,
                            dtx_dfb.reserve() as dtx_blk,
                            b_dfb.reserve() as b_blk,
                        ):
                            tx_s = ttl.copy(s_in[r0:r1, col : col + 1], s_blk)
                            tx_decay = ttl.copy(decay_in[r0:r1, 0:1], decay_blk)
                            tx_dtx = ttl.copy(dtx_in[r0:r1, 0:1], dtx_blk)
                            tx_b = ttl.copy(b_in[0:1, col : col + 1], b_blk)
                            tx_s.wait()
                            tx_decay.wait()
                            tx_dtx.wait()
                            tx_b.wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[r0:r1, col : col + 1])
                            tx.wait()
