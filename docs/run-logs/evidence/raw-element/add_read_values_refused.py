import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid=(1, 1))
def linear_scan(a, b, out):
    row_tiles = a.shape[0] // TILE_SIZE
    col_tiles = a.shape[1] // TILE_SIZE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(row_tiles, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(row_tiles, 1), block_count=2)
    scan_dfb = ttl.make_dataflow_buffer_like(out, shape=(row_tiles, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(row_tiles, 1), block_count=2)

    @ttl.datamovement()
    def read():
        for c in range(col_tiles):
            with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                tx_a = ttl.copy(a[0:row_tiles, c : c + 1], a_blk)
                tx_b = ttl.copy(b[0:row_tiles, c : c + 1], b_blk)
                tx_a.wait()
                tx_b.wait()
            with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk, scan_dfb.reserve() as s_blk:
                for i in range(64):
                    for j in range(32):
                        ttl.raw_element_write(s_blk, i, j, ttl.raw_element_read(a_blk, i, j) + ttl.raw_element_read(b_blk, i, j))

    @ttl.compute()
    def compute():
        for c in range(col_tiles):
            with scan_dfb.wait() as s_blk, out_dfb.reserve() as o_blk:
                o_blk.store(s_blk)

    @ttl.datamovement()
    def write():
        for c in range(col_tiles):
            with out_dfb.wait() as o_blk:
                tx = ttl.copy(o_blk, out[0:row_tiles, c : c + 1])
                tx.wait()
