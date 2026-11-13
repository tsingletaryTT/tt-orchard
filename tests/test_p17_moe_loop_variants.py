# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU-only checks of probe 17 (scripts/qwen4_ops_probe/p17_moe_loop_variants.py, issue 723): every variant of the slot loop adds up to the baseline loop on the CPU,
the variant list covers each flag alone and combined, the selftest can fail, and the device mode refuses to run without ttnn."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
PROBES = Path(__file__).resolve().parents[1] / "scripts" / "qwen4_ops_probe"


def load():
    sys.path.insert(0, str(PROBES))
    spec = importlib.util.spec_from_file_location("p17_variants", PROBES / "p17_moe_loop_variants.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_selftest_passes_for_every_variant():
    p17 = load()
    r = p17.selftest()
    assert r["ok"] and set(r["max_abs_by_variant"]) == {n for n, _ in p17.variants()}


def test_every_flag_has_a_variant_alone_and_one_in_the_combined_chain():
    p17 = load()
    names = dict(p17.variants())
    for f in p17.FLAGS:
        assert names[f"only_{f}"] == frozenset([f])
    assert any(fl == frozenset(p17.FLAGS) for fl in names.values())
    assert names["baseline"] == frozenset()


def test_the_selftest_goes_red_when_folding_the_weight_drops_it(monkeypatch):
    p17 = load()
    real = p17.cpu_variant
    monkeypatch.setattr(p17, "cpu_variant", lambda x, g, u, d, items, flags: real(x, g, u, d, items, flags - {"fold_weight"} if "fold_weight" in flags else flags) * (0.5 if "fold_weight" in flags else 1))
    assert not p17.selftest()["ok"]


def test_the_device_mode_stops_when_ttnn_cannot_be_imported():
    code = ("import sys; sys.modules['ttnn'] = None; import runpy; sys.argv = ['x']; "
            f"runpy.run_path(r'{PROBES / 'p17_moe_loop_variants.py'}', run_name='__main__')")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0 and "PROBE_RESULT" not in proc.stdout
