import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid="full")
def ssd_readout(c: ttnn.Tensor, s: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = c.shape[0] // TILE_SIZE
    col_tiles = c.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)

    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), block_count=2)
    s_dfb = ttl.make_dataflow_buffer_like(s, shape=(1, 1), block_count=2)
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
                            c_dfb.wait() as c_blk,
                            s_dfb.wait() as s_blk,
                            out_dfb.reserve() as out_blk,
                        ):
                            prod = c_blk * s_blk
                            reduced = ttl.math.reduce_sum(
                                prod, dims=[-1], shape=(1, 1)
                            )
                            out_blk.store(reduced)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row, row + 1
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        c0, c1 = col, col + 1
                        with c_dfb.reserve() as c_blk, s_dfb.reserve() as s_blk:
                            tx_c = ttl.copy(c[r0:r1, c0:c1], c_blk)
                            tx_s = ttl.copy(s[r0:r1, c0:c1], s_blk)
                            tx_c.wait()
                            tx_s.wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row, row + 1
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        c0, c1 = col, col + 1
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[r0:r1, c0:c1])
                            tx.wait()
