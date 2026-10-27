import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid=(1, 1))
def rms_op(x, w, out):
    row_tiles = x.shape[0] // TILE_SIZE
    col_tiles = x.shape[1] // TILE_SIZE
    inv_n = 1.0 / x.shape[1]

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, col_tiles), block_count=2)
    w_dfb = ttl.make_dataflow_buffer_like(w, shape=(1, col_tiles), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, col_tiles), block_count=2)

    @ttl.datamovement()
    def read():
        for r in range(row_tiles):
            with x_dfb.reserve() as x_blk, w_dfb.reserve() as w_blk:
                tx = ttl.copy(x[r : r + 1, 0:col_tiles], x_blk)
                tw = ttl.copy(w[r : r + 1, 0:col_tiles], w_blk)
                tx.wait()
                tw.wait()

    @ttl.compute()
    def compute():
        for r in range(row_tiles):
            with x_dfb.wait() as x_blk, w_dfb.wait() as w_blk, out_dfb.reserve() as o_blk:
                total = ttl.math.reduce_sum(x_blk * x_blk, dims=[-1], shape=(1, 1))
                inv_blk = ttl.block.fill(inv_n, shape=(1, 1))
                eps_blk = ttl.block.fill(1e-6, shape=(1, 1))
                scale = ttl.math.rsqrt(total * inv_blk + eps_blk)
                scale_b = ttl.block.broadcast(scale, dims=[-1], shape=(1, col_tiles))
                o_blk.store(x_blk * scale_b * w_blk)

    @ttl.datamovement()
    def write():
        for r in range(row_tiles):
            with out_dfb.wait() as o_blk:
                t = ttl.copy(o_blk, out[r : r + 1, 0:col_tiles])
                t.wait()
