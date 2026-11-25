# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The probe kernels behind the TT-Lang cheat sheet's chip claims (issue 801).

Each probe is a small kernel that uses one call of the sheet inside a compute or data-movement thread, so that compiling it with a chip
package's own compiler (no device, orchard/ttlang_hwcompile.py) shows whether that compiler takes the call. scripts/verify_cheat_compiles.py
compiles every probe with every chip package this machine has and writes the records to ttlang_cheat_compiles.json; orchard/cheat_compiles.py
judges the sheet's claims by them.

Two base kernels carry every probe: an elementwise kernel (read two tensors, store one expression, write the result: the same shape as the
control recipe `ctl-mul-tiles`) and a row reduction (the recipe `row-reduce-spread`'s skeleton). A probe changes one line of a base. A change
that did not apply is an error in this module, never a probe equal to its base.

`compiled` probes list the sheet entries they `covers`. `refused` probes use a form that the chip compilers refuse (`ttl.math.fill` and
`ttl.math.broadcast`: both exist as Python functions in the chip package, and the compiler resolves a call in a thread by its own table) and
list it in `refuses`. The refused probes drop `shape=` from the reduction so that both packages reach the refused call: tt-lang 1.1.6
refuses `shape=` on `reduce_sum` before it gets that far.
"""
from __future__ import annotations

ELEMENTWISE = '''import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid=(1, 1))
def probe(x, y, out):
    row_tiles = x.shape[0] // TILE_SIZE
    col_tiles = x.shape[1] // TILE_SIZE

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def read():
        for row in range(row_tiles):
            for col in range(col_tiles):
                with x_dfb.reserve() as x_blk, y_dfb.reserve() as y_blk:
                    tx = ttl.copy(x[row : row + 1, col : col + 1], x_blk)
                    ty = ttl.copy(y[row : row + 1, col : col + 1], y_blk)
                    tx.wait()
                    ty.wait()

    @ttl.compute()
    def compute():
        for _ in range(row_tiles * col_tiles):
            with x_dfb.wait() as x_blk, y_dfb.wait() as y_blk, out_dfb.reserve() as out_blk:
                out_blk.store(x_blk * y_blk)

    @ttl.datamovement()
    def write():
        for row in range(row_tiles):
            for col in range(col_tiles):
                with out_dfb.wait() as out_blk:
                    tx = ttl.copy(out_blk, out[row : row + 1, col : col + 1])
                    tx.wait()
'''

REDUCTION = '''import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid=(1, 1))
def probe(x, out):
    row_tiles = x.shape[0] // TILE_SIZE
    col_tiles = x.shape[1] // TILE_SIZE

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, col_tiles), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, col_tiles), block_count=2)

    @ttl.datamovement()
    def read():
        for row in range(row_tiles):
            with x_dfb.reserve() as x_blk:
                tx = ttl.copy(x[row : row + 1, 0:col_tiles], x_blk)
                tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(row_tiles):
            with x_dfb.wait() as x_blk, out_dfb.reserve() as o_blk:
                half = ttl.block.fill(0.5, shape=(1, col_tiles))
                s = ttl.math.reduce_sum(x_blk * x_blk * half, dims=[-1], shape=(1, 1))
                r = ttl.block.broadcast(s, dims=[-1], shape=(1, col_tiles))
                o_blk.store(x_blk * r)

    @ttl.datamovement()
    def write():
        for row in range(row_tiles):
            with out_dfb.wait() as o_blk:
                tx = ttl.copy(o_blk, out[row : row + 1, 0:col_tiles])
                tx.wait()
'''

STORE = "out_blk.store(x_blk * y_blk)"
COMPUTE_LOOP = "        for _ in range(row_tiles * col_tiles):"
REDUCE_CALL = "ttl.math.reduce_sum(x_blk * x_blk * half, dims=[-1], shape=(1, 1))"
NO_SHAPE_REDUCE = "ttl.math.reduce_sum(x_blk * x_blk * half, dims=[-1])"

ELEMENTWISE_SHAPES = [[64, 256], [64, 256], [64, 256]]
REDUCTION_SHAPES = [[64, 256], [64, 256]]
ELEMENTWISE_ENTRIES = ["ttl.operation", "ttl.compute", "ttl.datamovement", "ttl.make_dataflow_buffer_like", "DataflowBuffer.wait",
                       "DataflowBuffer.reserve", "TensorBlock.store", "ttl.copy"]


class ProbeError(Exception):
    """A change to a base kernel that did not apply exactly once."""


def _change(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ProbeError(f"the text {old[:60]!r} must occur once in the base kernel and occurs {text.count(old)} times")
    return text.replace(old, new)


def _unary(expr: str) -> str:
    return _change(ELEMENTWISE, STORE, f"out_blk.store({expr})")


def _in_compute(statement: str) -> str:
    return _change(ELEMENTWISE, COMPUTE_LOOP, f"        {statement}\n{COMPUTE_LOOP}")


def _torch(text: str) -> str:
    return _change(text, "import ttl\nimport ttnn", "import torch\nimport ttl\nimport ttnn")


def definitions() -> list[dict]:
    """The probes in the order they are compiled: name, expect, the entries covered or the forms refused, the tensor shapes, the kernel text."""
    def compiled(name, text, covers, shapes=ELEMENTWISE_SHAPES):
        return {"name": name, "expect": "compiled", "covers": list(covers), "shapes": [list(s) for s in shapes], "text": text}

    def refused(name, text, forms, shapes):
        return {"name": name, "expect": "refused", "refuses": list(forms), "shapes": [list(s) for s in shapes], "text": text}

    no_shape = _change(REDUCTION, REDUCE_CALL, NO_SHAPE_REDUCE)
    return [
        compiled("elementwise", ELEMENTWISE, ELEMENTWISE_ENTRIES),
        compiled("make-dfb", _change(ELEMENTWISE, "ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)",
                                     "ttl.make_dfb('bf16', shape=(1, 1), block_count=2)"), ["ttl.make_dfb"]),
        compiled("grid-size", _change(ELEMENTWISE, "    row_tiles = x.shape[0] // TILE_SIZE", "    cols, rows = ttl.grid_size(dims=2)\n"
                                      "    row_tiles = x.shape[0] // TILE_SIZE"), ["ttl.grid_size"]),
        compiled("node-in-compute", _in_compute("node_col, node_row = ttl.node(dims=2)"), ["ttl.node"]),
        compiled("reduce-sum-broadcast-fill", REDUCTION, ["ttl.math.reduce_sum", "ttl.block.broadcast", "ttl.block.fill"], REDUCTION_SHAPES),
        compiled("reduce-max", _change(REDUCTION, "ttl.math.reduce_sum", "ttl.math.reduce_max"), ["ttl.math.reduce_max"], REDUCTION_SHAPES),
        compiled("exp", _unary("ttl.math.exp(x_blk)"), ["ttl.math.exp"]),
        compiled("rsqrt", _unary("ttl.math.rsqrt(x_blk * x_blk + y_blk)"), ["ttl.math.rsqrt"]),
        compiled("silu", _unary("ttl.math.silu(x_blk) * y_blk"), ["ttl.math.silu"]),
        compiled("transpose", _unary("ttl.math.transpose(x_blk)"), ["ttl.math.transpose"]),
        compiled("matmul", _unary("ttl.matmul(x_blk, y_blk)"), ["ttl.matmul"]),
        compiled("signpost", _in_compute("ttl.signpost('probe')"), ["ttl.signpost"]),
        compiled("typecast-torch-dtype", _torch(_unary("ttl.math.typecast(x_blk, torch.bfloat16)")), ["ttl.math.typecast"]),
        refused("math-fill", _unary("x_blk * ttl.math.fill(0.5, shape=(1, 1))"), ["ttl.math.fill"], ELEMENTWISE_SHAPES),
        refused("math-broadcast", _change(no_shape, "ttl.block.broadcast", "ttl.math.broadcast"), ["ttl.math.broadcast"], REDUCTION_SHAPES),
    ]
