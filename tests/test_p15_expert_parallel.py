# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU-only checks of probe 15 (scripts/qwen4_ops_probe/p15_moe_expert_parallel.py, issue 723): the per-chip partial outputs add up to the unsplit
experts layer, the selftest can fail, and the device mode refuses to run without ttnn."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
PROBES = Path(__file__).resolve().parents[1] / "scripts" / "qwen4_ops_probe"


def load():
    sys.path.insert(0, str(PROBES))
    spec = importlib.util.spec_from_file_location("p15", PROBES / "p15_moe_expert_parallel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_selftest_passes_and_the_partials_add_up_to_the_reference():
    p15 = load()
    r = p15.selftest()
    assert r["ok"] and r["stats"]["max_abs"] < 1e-4


def test_each_chip_partial_holds_only_its_own_experts():
    p15 = load()
    import common
    from orchard import expert_placement as ep
    t = common.moe_inputs(tokens=32, hidden=64, experts=8, top_k=2, inter=64, shared=64)
    _, w, idx = common.moe_router_ref(t["hidden"], t["router"], 2)
    route = p15.routing_matrix(idx, w, 8)
    placement = ep.place_experts(8, 2, 1, 100)
    parts = p15.chip_partials(t, route, placement)
    # zero the routing of chip 0's experts: chip 0's partial must vanish and chip 1's must not change
    route0 = route.clone()
    route0[:, placement.experts_of(0)] = 0
    parts0 = p15.chip_partials(t, route0, placement)
    assert parts0[0].abs().max() == 0 and torch.equal(parts0[1], parts[1])


def test_the_selftest_goes_red_when_a_chips_experts_are_dropped(monkeypatch):
    p15 = load()
    real = p15.chip_partials
    monkeypatch.setattr(p15, "chip_partials", lambda t, r, pl: real(t, r, pl)[:1].repeat(pl.chips, 1, 1) * 0)
    assert not p15.selftest()["ok"]


def test_the_device_mode_stops_when_ttnn_cannot_be_imported():
    code = ("import sys; sys.modules['ttnn'] = None; import runpy; sys.argv = ['x']; "
            f"runpy.run_path(r'{PROBES / 'p15_moe_expert_parallel.py'}', run_name='__main__')")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0 and "PROBE_RESULT" not in proc.stdout
