# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU-only checks of scripts/qwen4_ops_probe: each probe's --selftest passes and prints one JSON result line without
importing ttnn, the shared reference code matches the transformers Qwen4Exp code when an interpreter that has it exists,
and the selftests can fail (a broken reference makes them go red)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "qwen4_ops_probe"
PROBES = ["p1_conv1d", "p2_chunk_gdn", "p3_recurrent_gdn", "p4_gated_residual", "p5_update_padded_kv_cache"]


def run_selftest(name, env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    proc = subprocess.run([sys.executable, str(SCRIPTS / f"{name}.py"), "--selftest"], capture_output=True, text=True,
                          env=env, timeout=300)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE_RESULT ")]
    return proc, lines


@pytest.mark.parametrize("name", PROBES)
def test_selftest_passes_with_one_result_line(name):
    proc, lines = run_selftest(name)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert len(lines) == 1
    res = json.loads(lines[0][len("PROBE_RESULT "):])
    assert res["ok"] is True and res["mode"] == "selftest" and res["probe"] == name


@pytest.mark.parametrize("name", PROBES)
def test_selftest_does_not_import_ttnn(name):
    """The wiring check: a ttnn module that raises on import must not stop the selftest."""
    code = ("import sys, runpy; sys.modules['ttnn'] = None; sys.argv = ['x', '--selftest']; "
            f"runpy.run_path(r'{SCRIPTS / (name + '.py')}', run_name='__main__')")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert "PROBE_RESULT" in proc.stdout, proc.stdout + proc.stderr


def test_selftest_goes_red_when_the_shifted_conv_is_wrong(monkeypatch):
    sys.path.insert(0, str(SCRIPTS))
    try:
        import common
        import p1_conv1d
        good = p1_conv1d.selftest()
        assert good["ok"]
        monkeypatch.setattr(common, "short_conv_shifted", lambda x, w, kernel=4, dilation=3: common.short_conv_ref(x, w, kernel, 1))
        assert p1_conv1d.selftest()["ok"] is False
    finally:
        sys.path.remove(str(SCRIPTS))


def test_selftest_goes_red_when_the_chunk_reference_is_wrong(monkeypatch):
    sys.path.insert(0, str(SCRIPTS))
    try:
        import common
        import p2_chunk_gdn
        assert p2_chunk_gdn.selftest()["ok"]
        real = common.torch_chunk_gated_delta_rule

        def wrong(*a, **k):
            o, s = real(*a, **k)
            return o * 1.01, s

        monkeypatch.setattr(common, "torch_chunk_gated_delta_rule", wrong)
        assert p2_chunk_gdn.selftest()["ok"] is False
    finally:
        sys.path.remove(str(SCRIPTS))


def _reference_python():
    cand = os.environ.get("ORCHARD_REFERENCE_PYTHON")
    for py in ([cand] if cand else []):
        if py and Path(py).exists():
            # The interpreter must have transformers with qwen4_exp, not only exist (older transformers lack it).
            probe = subprocess.run([py, "-c", "from transformers.models.qwen4_exp import modeling_qwen4_exp"],
                                   capture_output=True, text=True, timeout=300)
            if probe.returncode == 0:
                return py
    return None


def test_reference_python_without_qwen4_exp_is_skipped(monkeypatch):
    """An interpreter that exists but has no transformers qwen4_exp must not be chosen (the test then skips)."""
    monkeypatch.setenv("ORCHARD_REFERENCE_PYTHON", sys.executable)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "no qwen4_exp"))
    assert _reference_python() is None


def test_copies_match_transformers_qwen4_exp():
    """Our torch copies give the same numbers as transformers' Qwen4Exp functions (needs transformers >= 5.19)."""
    py = _reference_python()
    if py is None:
        pytest.skip("set ORCHARD_REFERENCE_PYTHON to an interpreter with transformers that has qwen4_exp")
    code = f"""
import sys, torch
sys.path.insert(0, r'{SCRIPTS}')
import common
from transformers.models.qwen4_exp import modeling_qwen4_exp as m
q, k, v, g, b = common.make_gdn_inputs(1, 70, 2, 4, 16, 16)
qe, ke = common.expand_heads(q, 4), common.expand_heads(k, 4)
a, sa = common.torch_chunk_gated_delta_rule(qe, ke, v, g, b, output_final_state=True, use_qk_l2norm_in_kernel=True)
r, sr = m.torch_chunk_gated_delta_rule(qe, ke, v, g, b, output_final_state=True, use_qk_l2norm_in_kernel=True)
a2, s2 = common.torch_recurrent_gated_delta_rule(qe, ke, v, g, b, output_final_state=True, use_qk_l2norm_in_kernel=True)
r2, t2 = m.torch_recurrent_gated_delta_rule(qe, ke, v, g, b, output_final_state=True, use_qk_l2norm_in_kernel=True)
assert (a - r).abs().max() < 1e-5 and (sa - sr).abs().max() < 1e-5
assert (a2 - r2).abs().max() < 1e-5 and (s2 - t2).abs().max() < 1e-5
print('MATCH')
"""
    proc = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=300)
    assert "MATCH" in proc.stdout, proc.stdout + proc.stderr


def test_result_line_is_strict_json_when_an_op_returns_nan_and_inf(capsys):
    """An op result with NaN and inf must still give a result line a strict JSON parser accepts, with the count kept."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        import common
        got = torch.tensor([1.0, float("nan"), float("inf"), 2.0])
        ref = torch.tensor([1.0, 2.0, 3.0, 4.0])
        common.emit({"probe": "x", "o": common.error_stats(got, ref)})
        line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("PROBE_RESULT ")][0]

        def refuse(token):
            raise ValueError(f"non-standard JSON token {token}")

        res = json.loads(line[len("PROBE_RESULT "):], parse_constant=refuse)
        assert res["o"]["nonfinite"] == 2
        assert res["o"]["max_abs"] is None
    finally:
        sys.path.remove(str(SCRIPTS))
