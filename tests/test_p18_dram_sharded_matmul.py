# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU-only checks of probe 18 (scripts/qwen4_ops_probe/p18_dram_sharded_matmul.py, issue 723): the layout arithmetic of the DRAM-sharded decode matmul, the selftest can fail,
and the device mode refuses to run without ttnn."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
PROBES = Path(__file__).resolve().parents[1] / "scripts" / "qwen4_ops_probe"


def load():
    sys.path.insert(0, str(PROBES))
    spec = importlib.util.spec_from_file_location("p18_dram", PROBES / "p18_dram_sharded_matmul.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_selftest_passes_for_every_flash_next_projection():
    assert load().selftest()["ok"]


def test_the_activation_cores_split_k_into_whole_tiles_over_one_row_or_whole_columns_of_eight():
    p18 = load()
    for k in (640, 1536, 2560, 4096):
        c = p18.activation_cores(k)
        assert (k // 32) % c == 0 and (c <= 8 or c % 8 == 0) and c <= 64
    assert p18.activation_cores(640) == 5 and p18.activation_cores(2560) == 40


def test_the_bank_shard_holds_whole_tiles_and_covers_n():
    p18 = load()
    assert p18.bank_tiles(1280, 8) == 5 and p18.bank_tiles(4160, 8) == 17 and p18.bank_tiles(4160, 7) == 19
    for n in (32, 1280, 2560, 4160, 8192):
        for banks in (7, 8):
            assert p18.bank_tiles(n, banks) * banks * 32 >= n


def test_the_block_width_divides_one_cores_share_of_k_and_is_at_most_eight_tiles():
    p18 = load()
    assert p18.in0_block_width(2560, 40) == 2 and p18.in0_block_width(640, 5) == 4
    assert all(8 >= p18.in0_block_width(k, p18.activation_cores(k)) >= 1 for k in (640, 1536, 2560))


def test_the_selftest_goes_red_when_the_bank_tiles_do_not_cover_n(monkeypatch):
    p18 = load()
    monkeypatch.setattr(p18, "bank_tiles", lambda n, banks: 1)
    assert not p18.selftest()["ok"]


def test_the_device_mode_stops_when_ttnn_cannot_be_imported():
    code = ("import sys; sys.modules['ttnn'] = None; import runpy; sys.argv = ['x']; "
            f"runpy.run_path(r'{PROBES / 'p18_dram_sharded_matmul.py'}', run_name='__main__')")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0 and "PROBE_RESULT" not in proc.stdout
