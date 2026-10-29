import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid="full")
def ssd_readout(c: ttnn.Tensor, s: ttnn.Tensor, out: ttnn.Tensor) -> None:
    # Shape in tiles
    row_tiles = c.shape[0] // TILE_SIZE
    col_tiles = c.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)

    # Dataflow buffers for c, s, and intermediate results
    # Each buffer holds 1 tile
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), block_count=2)
    s_dfb = ttl.make_dataflow_buffer_like(s, shape=(1, 1), block_count=2)
    prod_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), block_count=2)
    sum_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), block_count=2)

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

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row, row + 1
                # Accumulate sum across column tiles
                # Initialize accumulator to zero
                sum_acc = ttl.block.fill(0, shape=(1, 1))
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        c0, c1 = col, col + 1
                        with (
                            c_dfb.wait() as c_blk,
                            s_dfb.wait() as s_blk,
                            prod_dfb.reserve() as prod_blk,
                        ):
                            # Element-wise multiply
                            prod_blk.store(c_blk * s_blk)
                            # Reduce over columns (last dimension) within this tile
                            # The reduce_sum over dim=-1 gives us a (1, 1) result
                            sum_val = ttl.math.reduce_sum(
                                prod_blk, dims=[-1], shape=(1, 1)
                            )
                            sum_acc += sum_val
                # Store the accumulated sum
                sum_dfb.reserve().store(sum_acc)

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
                        with sum_dfb.wait() as sum_blk:
                            # Broadcast the sum across all columns
                            broadcasted = ttl.block.broadcast(
                                sum_blk, dims=[-1], shape=(1, 1)
                            )
                            tx = ttl.copy(broadcasted, out[r0:r1, c0:c1])
                            tx.wait()
