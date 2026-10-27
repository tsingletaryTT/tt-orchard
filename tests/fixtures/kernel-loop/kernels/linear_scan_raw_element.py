import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid=(1, 1))
def linear_scan(a, b, out):
    row_tiles = a.shape[0] // TILE_SIZE
    col_tiles = a.shape[1] // TILE_SIZE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    scan_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def read():
        # Scan column tile by column tile, top to bottom, carrying 32 per-column states.
        for c in range(col_tiles):
            carry = [0.0] * TILE_SIZE
            for r in range(row_tiles):
                with (
                    a_dfb.reserve() as a_blk,
                    b_dfb.reserve() as b_blk,
                    scan_dfb.reserve() as s_blk,
                ):
                    tx_a = ttl.copy(a[r : r + 1, c : c + 1], a_blk)
                    tx_b = ttl.copy(b[r : r + 1, c : c + 1], b_blk)
                    tx_a.wait()
                    tx_b.wait()
                    for i in range(TILE_SIZE):
                        for j in range(TILE_SIZE):
                            av = ttl.raw_element_read(a_blk, i, j)
                            bv = ttl.raw_element_read(b_blk, i, j)
                            carry[j] = av * carry[j] + bv
                            ttl.raw_element_write(s_blk, i, j, value=carry[j])

    @ttl.compute()
    def compute():
        zero = ttl.block.fill(0.0, shape=(1, 1))
        for c in range(col_tiles):
            for r in range(row_tiles):
                with (
                    a_dfb.wait() as a_blk,
                    b_dfb.wait() as b_blk,
                    scan_dfb.wait() as s_blk,
                    out_dfb.reserve() as o_blk,
                ):
                    o_blk.store(s_blk + (a_blk + b_blk) * zero)

    @ttl.datamovement()
    def write():
        for c in range(col_tiles):
            for r in range(row_tiles):
                with out_dfb.wait() as o_blk:
                    tx = ttl.copy(o_blk, out[r : r + 1, c : c + 1])
                    tx.wait()
