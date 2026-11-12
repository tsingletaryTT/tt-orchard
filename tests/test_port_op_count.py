# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU checks of scripts/port_op_count.py: the op counts of the generated tiny Qwen4 layers, the counting rule (host transfers are not device ops and the fake's matmul
is counted once), and the arithmetic of the Flash-Next projection."""
import collections
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "port_op_count.py"


def load():
    spec = importlib.util.spec_from_file_location("port_op_count_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["port_op_count_script"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_host_transfers_are_not_device_ops_and_a_matmul_is_counted_once():
    m = load()
    c = collections.Counter({"linear": 5, "matmul": 2, "from_torch": 9, "to_torch": 3, "add": 4})
    # five linear calls already include the two matmuls the fake routes through linear, so the device ran 5 linears or matmuls and 4 adds
    assert m.total(c) == 5 + 4


def test_the_projection_adds_the_mixers_the_moe_and_the_outside_ops():
    m = load()
    layer = {"layer_A": 100, "layer_B": 150, "layer_C": 120, "embedding_and_prelude": 3, "mixer_and_head": 20}
    r = m.project(layer, moe_fixed=10, slots_hit=4, us_per_op=10)
    mixers = 35 * 100 + 150 + 12 * 120 - 48 * m.MLP_OPS
    moe = 48 * (10 + m.PER_SLOT_OPS * 4)
    assert r["ops"] == mixers + moe + 23 and r["floor_ms"] == round(r["ops"] * 10 / 1000, 1)
    assert m.project(layer, 10, 8, 10)["moe_ops"] - r["moe_ops"] == 48 * m.PER_SLOT_OPS * 4


def test_the_tiny_qwen4_layers_have_the_op_counts_the_generated_text_implies():
    m = load()
    layers = m.count_layers()
    a, b, c = (m.total(layers[f"layer_{v}"]) for v in "ABC")
    assert 50 < a < b and c < b, "the PLE layer is the largest; the gated-delta-net layer is the smallest"
    assert layers["variant_of_layer"] == ["A", "B", "A", "C"]
    assert m.total(layers["mixer_and_head"]) < a
