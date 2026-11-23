import ttl
import ttnn


TILE_SIZE = 32


@ttl.operation(grid="full")
def row_sum(x: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = x.shape[0] // TILE_SIZE
    col_tiles = x.shape[1] // TILE_SIZE

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, col_tiles), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)

        for local_row in range(row_tiles):
            row = node_row * row_tiles + local_row
            if row < row_tiles:
                with (
                    x_dfb.wait() as x_blk,
                    out_dfb.reserve() as out_blk,
                ):
                    # x_blk has shape (1, col_tiles) tiles = (32, 256) elements
                    # Reduce over dimension -1 (columns) to get shape (1, 1) tiles
                    sum_result = ttl.math.reduce_sum(x_blk, dims=[-1], shape=(1, 1))
                    out_blk.store(sum_result)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(row_tiles):
            row = node_row * row_tiles + local_row
            if row < row_tiles:
                with x_dfb.reserve() as x_blk:
                    tx = ttl.copy(x[row : row + 1, 0 : col_tiles], x_blk)
                    tx.wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(row_tiles):
            row = node_row * row_tiles + local_row
            if row < row_tiles:
                with out_dfb.wait() as out_blk:
                    tx = ttl.copy(out_blk, out[row : row + 1, 0 : 1])
                    tx.wait()
