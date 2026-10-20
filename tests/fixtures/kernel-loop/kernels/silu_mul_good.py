# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Test fixture for tt-orchard's kernel loop: out = silu(gate) * up, elementwise. Written from upstream
# tt-lang examples/eltwise_add.py: the same tiling and threads, with the compute line changed.
# Verified on tt-lang-sim at PCC >= 0.999 on every input set of ops/silu_mul.json.
import ttl
import ttnn

TILE_SIZE = 32
GRANULARITY = 2


@ttl.operation(grid="full")
def OPERATION(gate_in: ttnn.Tensor, up_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = gate_in.shape[0] // TILE_SIZE // GRANULARITY
    col_tiles = gate_in.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)

    gate_dfb = ttl.make_dataflow_buffer_like(gate_in, shape=(GRANULARITY, 1), block_count=2)
    up_dfb = ttl.make_dataflow_buffer_like(up_in, shape=(GRANULARITY, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(GRANULARITY, 1), block_count=2)

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
                            gate_dfb.wait() as gate_blk,
                            up_dfb.wait() as up_blk,
                            out_dfb.reserve() as out_blk,
                        ):
                            out_blk.store(ttl.math.silu(gate_blk) * up_blk)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with gate_dfb.reserve() as gate_blk, up_dfb.reserve() as up_blk:
                            tx_gate = ttl.copy(gate_in[r0:r1, col : col + 1], gate_blk)
                            tx_up = ttl.copy(up_in[r0:r1, col : col + 1], up_blk)
                            tx_gate.wait()
                            tx_up.wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[r0:r1, col : col + 1])
                            tx.wait()

