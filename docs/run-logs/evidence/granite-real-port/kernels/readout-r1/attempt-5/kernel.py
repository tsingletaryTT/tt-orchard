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

    # Dataflow buffers for c and s: each holds a full row of tiles
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, col_tiles), block_count=2)
    s_dfb = ttl.make_dataflow_buffer_like(s, shape=(1, col_tiles), block_count=2)
    # Output buffer holds 1 tile
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)

        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                # Accumulator for this row's reduction
                acc = ttl.block.fill(0, shape=(1, col_tiles))
                
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with (
                            c_dfb.wait() as c_blk,
                            s_dfb.wait() as s_blk,
                        ):
                            # Multiply c and s tiles
                            prod = c_blk * s_blk
                            # Accumulate into row accumulator
                            acc += prod

                # Now reduce the accumulated row across columns
                # acc has shape (1, col_tiles), reduce over dim -1 (columns) to get (1, 1)
                reduced = ttl.math.reduce_sum(acc, dims=[-1], shape=(1, 1))
                
                # Store the reduced value to each column of the output row
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with out_dfb.reserve() as out_blk:
                            out_blk.store(reduced)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)

        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                with c_dfb.reserve() as c_blk, s_dfb.reserve() as s_blk:
                    tx_c = ttl.copy(c[row : row + 1, 0:col_tiles], c_blk)
                    tx_s = ttl.copy(s[row : row + 1, 0:col_tiles], s_blk)
                    tx_c.wait()
                    tx_s.wait()

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
                            tx = ttl.copy(out_blk, out[row : row + 1, col : col + 1])
                            tx.wait()
