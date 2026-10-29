import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid="full")
def ssd_readout(c_in: ttnn.Tensor, s_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = c_in.shape[0] // TILE_SIZE
    col_tiles = c_in.shape[1] // TILE_SIZE

    # Spread the rows of tiles over every node of the grid, one contiguous run of rows per node.
    grid_cols, grid_rows = ttl.grid_size(dims=2)
    nodes = grid_cols * grid_rows
    rows_per_node = -(-row_tiles // nodes)

    # One block is one row of tiles: (1, col_tiles).
    c_dfb = ttl.make_dataflow_buffer_like(c_in, shape=(1, col_tiles), block_count=2)
    s_dfb = ttl.make_dataflow_buffer_like(s_in, shape=(1, col_tiles), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, col_tiles), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_id * rows_per_node + local_row
            if row < row_tiles:
                with (
                    c_dfb.wait() as c_blk,
                    s_dfb.wait() as s_blk,
                    out_dfb.reserve() as out_blk,
                ):
                    # Element-wise multiplication
                    prod = c_blk * s_blk
                    # Sum over the last axis: one value per row, held in column 0 of each tile.
                    sum_val = ttl.math.reduce_sum(prod, dims=[-1], shape=(1, 1))
                    # Broadcast the sum across all columns in this row
                    sum_row = ttl.block.broadcast(sum_val, dims=[-1], shape=(1, col_tiles))
                    out_blk.store(sum_row)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_id * rows_per_node + local_row
            if row < row_tiles:
                with c_dfb.reserve() as c_blk, s_dfb.reserve() as s_blk:
                    tx_c = ttl.copy(c_in[row : row + 1, 0:col_tiles], c_blk)
                    tx_s = ttl.copy(s_in[row : row + 1, 0:col_tiles], s_blk)
                    tx_c.wait()
                    tx_s.wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_id * rows_per_node + local_row
            if row < row_tiles:
                with out_dfb.wait() as out_blk:
                    tx = ttl.copy(out_blk, out[row : row + 1, 0:col_tiles])
                    tx.wait()
