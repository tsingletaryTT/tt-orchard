# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU-only checks of probe 16 (scripts/qwen4_ops_probe/p16_moe_loop_cost.py, issue 723): the per-slot loop adds up to the reference, the slot inputs are
well formed, the selftest can fail, and the device mode refuses to run without ttnn."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
PROBES = Path(__file__).resolve().parents[1] / "scripts" / "qwen4_ops_probe"


def load():
    sys.path.insert(0, str(PROBES))
    spec = importlib.util.spec_from_file_location("p16_loop", PROBES / "p16_moe_loop_cost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_selftest_passes_and_hits_some_slots():
    r = load().selftest()
    assert r["ok"] and r["slots_hit"] > 0 and r["stats"]["max_abs"] < 1e-4


def test_slot_inputs_are_padded_to_32_rows_and_the_scatter_matrix_is_one_hot():
    p16 = load()
    idx, w = p16.make_routing(40, 64, 4, 1)
    items = p16.slot_inputs(idx, w, 16, 40)
    assert items
    for j, ids, wt, back in items:
        assert ids.shape[0] % 32 == 0 and ids.shape[0] >= 32 and wt.shape == ids.shape
        n = int((idx == j).sum())
        assert int(back.sum()) == n and float(wt.sum()) == pytest.approx(float(w[idx == j].sum()), rel=1e-5)
        assert (back.sum(0)[n:] == 0).all()                # padding rows scatter nowhere


def test_only_slots_some_token_picked_are_listed():
    p16 = load()
    idx, w = p16.make_routing(1, 512, 10, 0)
    hit = {j for j, *_ in p16.slot_inputs(idx, w, 128, 1)}
    assert hit == {int(e) for e in idx[0] if int(e) < 128}


def test_the_selftest_goes_red_when_a_slot_is_skipped(monkeypatch):
    p16 = load()
    real = p16.slot_inputs
    monkeypatch.setattr(p16, "slot_inputs", lambda *a: real(*a)[1:])
    assert not p16.selftest()["ok"]


def test_the_device_mode_stops_when_ttnn_cannot_be_imported():
    code = ("import sys; sys.modules['ttnn'] = None; import runpy; sys.argv = ['x']; "
            f"runpy.run_path(r'{PROBES / 'p16_moe_loop_cost.py'}', run_name='__main__')")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0 and "PROBE_RESULT" not in proc.stdout
