import ttl
import ttnn


TILE_SIZE = 32


@ttl.operation(grid="full")
def row_sum(x: ttnn.Tensor, out: ttnn.Tensor) -> None:
    # x has shape [64, 256] = [2 tile rows, 8 tile cols]
    # out has shape [64, 32] = [2 tile rows, 1 tile col]
    row_tiles = x.shape[0] // TILE_SIZE  # 2
    col_tiles = x.shape[1] // TILE_SIZE  # 8
    
    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)
    
    # DFB shapes: each block holds 1 row of tiles (1 row, 8 cols of tiles)
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, col_tiles), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    
    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                with (
                    x_dfb.wait() as x_blk,
                    out_dfb.reserve() as out_blk,
                ):
                    # x_blk has shape (1, col_tiles) = (1, 8) tiles
                    # Reduce over the last dimension (columns) to get (1, 1)
                    reduced = ttl.math.reduce_sum(
                        x_blk, dims=[-1], shape=(1, 1)
                    )
                    out_blk.store(reduced)
    
    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * 1, (row + 1) * 1
                with x_dfb.reserve() as x_blk:
                    tx = ttl.copy(x[r0:r1, 0:col_tiles], x_blk)
                    tx.wait()
    
    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * 1, (row + 1) * 1
                with out_dfb.wait() as out_blk:
                    tx = ttl.copy(out_blk, out[r0:r1, 0:1])
                    tx.wait()
