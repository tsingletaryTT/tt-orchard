# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid="full")
def ssd_state_update_bcast(s: ttnn.Tensor, decay: ttnn.Tensor, dtx: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = s.shape[0] // TILE_SIZE
    col_tiles = s.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    nodes = grid_cols * grid_rows
    rows_per_node = -(-row_tiles // nodes)

    # DFBs: s is full [64, 128], decay/dtx are [64, 1], b is [1, 128]
    s_dfb = ttl.make_dataflow_buffer_like(s, shape=(1, 1), block_count=2)
    decay_dfb = ttl.make_dataflow_buffer_like(decay, shape=(1, 1), block_count=2)
    dtx_dfb = ttl.make_dataflow_buffer_like(dtx, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_id * rows_per_node + local_row
            if row < row_tiles:
                for local_col in range(col_tiles):
                    col = local_col
                    if col < col_tiles:
                        with (
                            s_dfb.wait() as s_blk,
                            decay_dfb.wait() as decay_blk,
                            dtx_dfb.wait() as dtx_blk,
                            b_dfb.wait() as b_blk,
                            out_dfb.reserve() as out_blk,
                        ):
                            # decay_blk is [1,1] containing decay[row, 0]
                            # dtx_blk is [1,1] containing dtx[row, 0]
                            # b_blk is [1,1] containing b[0, col]
                            # s_blk is [1,1] containing s[row, col]
                            
                            # Since decay_blk, dtx_blk, b_blk are all [1,1], they will broadcast
                            # when multiplied with s_blk
                            result = decay_blk * s_blk + dtx_blk * b_blk
                            out_blk.store(result)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_id * rows_per_node + local_row
            if row < row_tiles:
                for local_col in range(col_tiles):
                    col = local_col
                    if col < col_tiles:
                        with (
                            s_dfb.reserve() as s_blk,
                            decay_dfb.reserve() as decay_blk,
                            dtx_dfb.reserve() as dtx_blk,
                            b_dfb.reserve() as b_blk,
                        ):
                            # Load s[row, col]
                            tx_s = ttl.copy(s[row : row + 1, col : col + 1], s_blk)
                            # Load decay[row, 0]
                            tx_decay = ttl.copy(decay[row : row + 1, 0 : 1], decay_blk)
                            # Load dtx[row, 0]
                            tx_dtx = ttl.copy(dtx[row : row + 1, 0 : 1], dtx_blk)
                            # Load b[0, col]
                            tx_b = ttl.copy(b[0 : 1, col : col + 1], b_blk)
                            tx_s.wait()
                            tx_decay.wait()
                            tx_dtx.wait()
                            tx_b.wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col
        for local_row in range(rows_per_node):
            row = node_id * rows_per_node + local_row
            if row < row_tiles:
                for local_col in range(col_tiles):
                    col = local_col
                    if col < col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[row : row + 1, col : col + 1])
                            tx.wait()
