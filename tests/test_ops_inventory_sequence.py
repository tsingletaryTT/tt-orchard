# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The ordered op list (`sequence`) the trace template writes for the subgraph matcher (orchard/oppatterns.py)."""
import importlib.util
import json
import operator
import types

import pytest

import test_ops_inventory_template as t

TEMPLATE = t.TEMPLATE


def load():
    spec = importlib.util.spec_from_file_location("ops_inventory_seq_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Node:
    def __init__(self, name, op="call_function", target=None, args=(), kwargs=None):
        self.name, self.op, self.target, self.args, self.kwargs = name, op, target, tuple(args), kwargs or {}
        self.meta = {}
        self.all_input_nodes = [a for a in self.args if isinstance(a, Node)]


@pytest.mark.parametrize("args,kwargs,expected", [
    ((2,), {}, [2]), ((1e-6,), {}, [1e-6]), (([-1], True), {}, [[-1], True]), ((True,), {}, [True]),
    (((1, 2, 3),), {}, [[1, 2, 3]]), (([1, 2, 3, 4, 5],), {}, []), (([],), {}, []), (([1, "a"],), {}, []),
    (([True, 2],), {}, []), (("x", None), {}, []), ((), {"dim": -1}, [-1]), ((1, 2, 3, 4, 5, 6, 7, 8), {}, [1, 2, 3, 4, 5, 6]),
])
def test_consts_keep_numbers_and_short_integer_lists_only(args, kwargs, expected):
    assert load()._consts(Node("n", args=args, kwargs=kwargs)) == expected


def test_a_node_argument_is_not_a_constant():
    assert load()._consts(Node("n", args=(Node("a"), 3))) == [3]


def test_source_follows_getitem_to_the_node_that_made_the_tuple():
    m = load()
    root = Node("wrap", target=object())
    first = Node("g1", target=operator.getitem, args=(root, 0))
    second = Node("g2", target=operator.getitem, args=(first, 1))
    assert m._source(second) is root
    assert m._source(root) is root
    plain = Node("p", target=len, args=(root,))
    assert m._source(plain) is plain


@pytest.fixture(scope="module")
def traced(tmp_path_factory):
    if t.PY is None:
        pytest.skip("no interpreter with torch and transformers")
    done, raw = t.run_template(tmp_path_factory.mktemp("seq"), t.QWEN3_TINY)
    assert done.returncode == 0, done.stderr
    return raw


@t.needs_stack
def test_the_trace_carries_an_ordered_op_list_with_data_flow(traced):
    seq = traced["sequence"]
    assert len(seq) > 50 and "sequence_note" in traced
    ids = [s["id"] for s in seq]
    assert len(set(ids)) == len(ids)
    assert all(set(s) == {"id", "op", "in", "module", "shape", "const"} for s in seq)
    assert all(s["op"].startswith("aten.") for s in seq)
    by = {s["id"]: s for s in seq}
    rsqrt = next(s for s in seq if s["op"] == "aten.rsqrt.default")
    add = by[rsqrt["in"][0]]
    assert add["op"] == "aten.add.Tensor" and any(isinstance(c, float) and c < 1e-3 for c in add["const"])
    mean = by[add["in"][0]]
    assert mean["op"] == "aten.mean.dim" and [-1] in mean["const"]
    assert by[mean["in"][0]]["op"] == "aten.pow.Tensor_Scalar" and 2 in by[mean["in"][0]]["const"]


@t.needs_stack
def test_only_the_first_layer_and_the_nodes_outside_the_layers_are_listed(traced):
    mods = {s["module"] for s in traced["sequence"]}
    assert any(".layers.0." in m or m.endswith(".layers.0") for m in mods)
    assert not any(".layers.1" in m or ".layers.2" in m or ".layers.3" in m for m in mods)
    assert any(m in ("", "model", "model.norm", "lm_head") for m in mods)
    assert traced["layers_in_config"] == 4 and traced["layers_traced"] == 2


@t.needs_stack
def test_a_read_through_a_higher_order_wrapper_does_not_name_a_getitem_node(traced):
    assert not any("getitem" in i for s in traced["sequence"] for i in s["in"] + [s["id"]])


@t.needs_stack
def test_the_list_is_capped(tmp_path):
    m = load()
    m.SEQUENCE_MAX = 3
    import torch

    class Tiny(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x) + torch.sigmoid(x) * 2 + torch.exp(x)

    ep = torch.export.export(Tiny(), (torch.zeros(2, 2),), strict=False)
    got = m.sequence(m.graph_modules(ep))
    assert len(got) == 3
    m.SEQUENCE_MAX = 6000
    assert len(m.sequence(m.graph_modules(ep))) >= 5


@t.needs_stack
def test_the_note_says_what_the_list_holds_and_how_large_it_can_be(traced):
    note = traced["sequence_note"]
    assert "first layer" in note and "6000" in note and "getitem" in note


@t.needs_stack
def test_a_skipped_model_has_no_sequence(tmp_path):
    done, raw = t.run_template(tmp_path, {**t.QWEN3_TINY, "auto_map": {"A": "m.X"}})
    assert done.returncode == 0 and "sequence" not in raw
    assert json.dumps(raw)
