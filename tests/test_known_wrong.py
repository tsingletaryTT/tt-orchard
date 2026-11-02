# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 338: ttnn.conv1d with dilation above 1 and groups above 1 returns a wrong result and no error on the probed build. The
inventory flags a traced convolution that matches, and the probe's conv1d wrapper refuses the call unless told to reproduce it."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import knownwrong
from orchard import opinventory as oi

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "qwen4_ops_probe"

# What the trace template records for aten.convolution(x, w, None, [1], [0], [3], False, [0], 64): lists, the bool, the groups int.
DEPTHWISE_DILATED = [[1], [0], [3], False, [0], 64]


@pytest.mark.parametrize("dilation,groups,wrong", [(3, 64, True), (2, 2, True), (1, 64, False), (3, 1, False), (1, 1, False)])
def test_only_dilation_and_groups_both_above_one_match(dilation, groups, wrong):
    assert (knownwrong.conv1d_known_wrong(dilation, groups) is not None) is wrong


def test_the_entry_carries_the_issue_number_and_the_build():
    e = knownwrong.conv1d_known_wrong(3, 64)
    assert e["issue"] == 338 and "2026-10-30" in e["build"]


def test_check_conv1d_raises_with_the_issue_number():
    with pytest.raises(knownwrong.KnownWrong, match="issue 338"):
        knownwrong.check_conv1d(3, 64)
    knownwrong.check_conv1d(1, 64)             # no error


def test_conv_args_reads_aten_convolution_and_aten_conv1d():
    assert knownwrong.conv_args("aten.convolution.default", DEPTHWISE_DILATED) == (3, 64)
    assert knownwrong.conv_args("aten.conv1d.default", [[1], [0], [3], 64]) == (3, 64)
    assert knownwrong.conv_args("aten.mm.default", DEPTHWISE_DILATED) is None
    # 2D and transposed convolutions are other ops (review of PR 346)
    assert knownwrong.conv_args("aten.convolution.default", [[1, 1], [0, 0], [3, 3], False, [0, 0], 64]) is None
    assert knownwrong.conv_args("aten.convolution.default", [[1], [0], [3], True, [0], 64]) is None
    assert knownwrong.conv_args("aten.convolution.default", [[1], [0], [3], [0], 64]) is None      # no transposed flag recorded
    assert knownwrong.conv_args("aten.convolution.default", [[1], [0]]) is None
    assert knownwrong.conv_args("aten.convolution.default", None) is None


def seq(consts, op="aten.convolution.default", module="model.layers.0.conv"):
    return [{"id": "g0:c", "op": op, "in": [], "module": module, "shape": [1, 8, 64], "const": consts}]


def raw_with(sequence):
    return {"schema": 1, "model_dir": "/m", "architectures": ["X"], "traced_class": "X", "layers_in_config": 2, "layers_traced": 2,
            "seq_len": 8, "failures": [], "ops": [], "sequence": sequence}


def build(raw):
    return oi.build(raw, mapping={}, decomps={}, dump=None)


def test_the_inventory_flags_a_traced_dilated_depthwise_convolution():
    inv = build(raw_with(seq(DEPTHWISE_DILATED)))
    assert [(f["issue"], f["dilation"], f["groups"], f["module"]) for f in inv["known_wrong"]] == [(338, 3, 64, "model.layers.0.conv")]
    md = oi.render_md(inv)
    assert "Known wrong" in md and "| 338 |" in md


def test_the_inventory_flags_nothing_for_a_dilation_one_depthwise_or_a_grouped_one_without_dilation():
    for consts in ([[1], [0], [1], False, [0], 64], [[1], [0], [3], False, [0], 1]):
        inv = build(raw_with(seq(consts)))
        assert inv["known_wrong"] == [] and "Known wrong" not in oi.render_md(inv)


def test_the_flag_does_not_depend_on_a_ttnn_dump():
    inv = oi.build(raw_with(seq(DEPTHWISE_DILATED)), mapping={"aten.convolution.default": {"ttnn": "conv1d"}}, decomps={},
                   dump={"ttnn_version": "1", "names": ["conv1d"]}, installed_ttnn="1")
    assert len(inv["known_wrong"]) == 1


def test_the_stage_0_record_lists_the_known_wrong_calls_and_omits_the_key_when_there_are_none(tmp_path):
    import json
    for name, consts, expected in (("hit", DEPTHWISE_DILATED, [{"op": "aten.convolution.default", "ttnn": "conv1d", "dilation": 3,
                                                                  "groups": 64, "issue": 338}]),
                                   ("miss", [[1], [0], [1], False, [0], 64], None)):
        sd = tmp_path / name / "stages" / "0"
        sd.mkdir(parents=True)
        (sd / "delta.json").write_text("{}")
        (sd / "ops-inventory.json").write_text(json.dumps(build(raw_with(seq(consts)))))
        assert oi.record_in_delta(sd) is True
        rec = json.loads((sd / "delta.json").read_text())["inventory"]
        assert rec.get("known_wrong") == expected


# The probe: its conv1d wrapper refuses the known-wrong call before ttnn is reached.

def load_probe():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import p1_conv1d
        return p1_conv1d
    finally:
        sys.path.pop(0)


def test_the_probe_wrapper_refuses_a_dilated_depthwise_call_before_it_reaches_ttnn():
    torch = pytest.importorskip("torch")
    p1 = load_probe()
    x = torch.zeros(1, 8, 16)
    w = torch.zeros(16, 1, 4)
    boom = SimpleNamespace()                              # any attribute access on ttnn would raise AttributeError
    with pytest.raises(knownwrong.KnownWrong, match="issue 338"):
        p1.run_conv1d(boom, None, x, w, 4, 3, False)
    with pytest.raises(AttributeError):                   # allow_known_wrong lets the probe reproduce the result: it reaches ttnn
        p1.run_conv1d(boom, None, x, w, 4, 3, False, allow_known_wrong=True)
    with pytest.raises(AttributeError):                   # dilation 1 is not flagged
        p1.run_conv1d(boom, None, x, w, 4, 1, False)
