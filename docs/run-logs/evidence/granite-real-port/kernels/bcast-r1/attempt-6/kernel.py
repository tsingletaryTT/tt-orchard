import ttl
import ttnn

TILE_SIZE = 32
GRANULARITY = 2


@ttl.operation(grid="full")
def ssd_state_update_bcast(s_in: ttnn.Tensor, decay_in: ttnn.Tensor, dtx_in: ttnn.Tensor, b_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = s_in.shape[0] // TILE_SIZE
    col_tiles = s_in.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)

    # Dataflow buffers
    s_dfb = ttl.make_dataflow_buffer_like(s_in, shape=(1, 1), block_count=2)
    decay_dfb = ttl.make_dataflow_buffer_like(decay_in, shape=(1, 1), block_count=2)
    dtx_dfb = ttl.make_dataflow_buffer_like(dtx_in, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b_in, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

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
                            # Broadcast decay over columns: (1, 1) -> (1, col_tiles)
                            decay_bcast = ttl.block.broadcast(decay_blk, dims=[-1], shape=(1, col_tiles))
                            
                            # Broadcast dtx over columns: (1, 1) -> (1, col_tiles)
                            dtx_bcast = ttl.block.broadcast(dtx_blk, dims=[-1], shape=(1, col_tiles))
                            
                            # Broadcast b over rows: (1, 1) -> (row_tiles, 1)
                            b_bcast = ttl.block.broadcast(b_blk, dims=[0], shape=(row_tiles, 1))
                            
                            # Broadcast b_bcast over columns: (row_tiles, 1) -> (row_tiles, col_tiles)
                            b_full_bcast = ttl.block.broadcast(b_bcast, dims=[-1], shape=(row_tiles, col_tiles))
                            
                            # Broadcast decay_bcast over rows: (1, col_tiles) -> (row_tiles, col_tiles)
                            decay_full_bcast = ttl.block.broadcast(decay_bcast, dims=[0], shape=(row_tiles, col_tiles))
                            
                            # Broadcast dtx_bcast over rows: (1, col_tiles) -> (row_tiles, col_tiles)
                            dtx_full_bcast = ttl.block.broadcast(dtx_bcast, dims=[0], shape=(row_tiles, col_tiles))
                            
                            # Now I have:
                            # - s_blk: (1, 1) tiles = (32, 32) elements
                            # - decay_full_bcast: (row_tiles, col_tiles) tiles = (64, 128) elements
                            # - dtx_full_bcast: (row_tiles, col_tiles) tiles = (64, 128) elements
                            # - b_full_bcast: (row_tiles, col_tiles) tiles = (64, 128) elements
                            
                            # But s_blk is only (1, 1), so I need to broadcast it too
                            s_full_bcast = ttl.block.broadcast(s_blk, dims=[0, -1], shape=(row_tiles, col_tiles))
                            
                            # Compute: out = decay_full_bcast * s_full_bcast + dtx_full_bcast * b_full_bcast
                            result = decay_full_bcast * s_full_bcast + dtx_full_bcast * b_full_bcast
                            
                            # Store only the current tile
                            out_blk.store(result)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * 1, (row + 1) * 1
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with s_dfb.reserve() as s_blk, decay_dfb.reserve() as decay_blk, dtx_dfb.reserve() as dtx_blk, b_dfb.reserve() as b_blk:
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
                r0, r1 = row * 1, (row + 1) * 1
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[r0:r1, col : col + 1])
                            tx.wait()
