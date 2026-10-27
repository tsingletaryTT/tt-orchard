# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The `demo-env` preflight row (issue 92): it appears for a stock-runtime run or when demo_python is configured, and it blocks
when the interpreter fails `tt-orchard demo-env --check`. A tag difference between the host tree and the venv is a warning."""
from pathlib import Path

import pytest

from orchard import demo_env as de
from orchard import preflight as pf
from test_preflight import cfg, signals
from test_research_preflight import report
from test_stock_preflight import ROW_PARAMS, ROW_README

PY = Path("/v/demo/bin/python")
CMD = "tt-orchard demo-env --venv-dir /v/demo --tt-metal-dir /host/tt-metal"


def ok(name, detail="fine"):
    return de.Result(name, de.OK, detail)


HEALTHY = [ok("venv"), ok("numpy", "numpy 1.26.4, below 2"), ok("cv2"), ok("datasets"), ok("torch"), ok("ttnn"), ok("ttml"),
           ok("ttnn-version"), ok("host-tree"), ok("record"), ok("record-tag")]


def row(results, command=CMD, python=PY):
    return pf.check_demo_env(python, results, command)


# ---- the row itself ---------------------------------------------------------------------------------------------------

def test_a_venv_that_passes_every_rule_is_ok_and_says_which_python():
    c = row(HEALTHY)
    assert c.name == "demo-env" and c.status == pf.OK and str(PY) in c.detail


@pytest.mark.parametrize("rule", ["numpy", "cv2", "datasets", "torch", "ttnn", "ttml", "host-tree", "venv", "venv-dir"])
def test_a_failed_rule_blocks_names_the_rule_and_the_exact_command(rule):
    results = [de.Result(r.name, de.FAIL, "numpy 2.5.3 is too new") if r.name == rule else r for r in HEALTHY]
    if rule == "venv-dir":
        results = [de.Result("venv-dir", de.FAIL, "it is the shared venv")]
    c = row(results)
    assert c.status == pf.BLOCK and c.reason == "config-invalid"
    assert rule in c.detail and CMD in c.detail and str(PY) in c.detail


def test_every_failing_rule_is_listed():
    results = [de.Result(r.name, de.FAIL, f"{r.name} is broken") if r.name in ("numpy", "cv2") else r for r in HEALTHY]
    c = row(results)
    assert "numpy is broken" in c.detail and "cv2 is broken" in c.detail


@pytest.mark.parametrize("rule", ["ttnn-version", "record-tag"])
def test_a_tag_that_differs_from_the_host_tree_is_a_warning_and_never_a_block(rule):
    results = [de.Result(r.name, de.FAIL, "built for v0.79.0; the tree is at v0.80.0") if r.name == rule else r for r in HEALTHY]
    c = row(results)
    assert c.status == pf.WARN and c.reason is None
    assert "v0.79.0" in c.detail and CMD in c.detail


def test_a_missing_record_is_a_warning():
    results = [de.Result("record", de.TODO, "missing") if r.name == "record" else r for r in HEALTHY]
    assert row(results).status == pf.WARN


def test_a_block_wins_over_a_tag_warning():
    results = [de.Result(r.name, de.FAIL, "x") if r.name in ("ttnn-version", "cv2") else r for r in HEALTHY]
    c = row(results)
    assert c.status == pf.BLOCK and "cv2" in c.detail


def test_no_demo_python_is_a_warning_that_names_the_command_and_the_key():
    c = pf.check_demo_env(None, None, CMD)
    assert c.status == pf.WARN and CMD in c.detail and "demo_python" in c.detail


def test_a_python_that_is_not_in_a_venv_bin_directory_is_a_warning_that_could_not_check():
    c = pf.check_demo_env(Path("/usr/bin/python3"), None, CMD)
    assert c.status == pf.WARN and "could not check" in c.detail


# ---- when the row appears -----------------------------------------------------------------------------------------------

def stock_signals(**over):
    rep = report(tt_transformers=[ROW_PARAMS, ROW_README])
    base = dict(research=lambda *a: rep, stock_params=lambda name: Path("/host/params") / name,
                demo_env=lambda python: (HEALTHY if python else None, CMD))
    base.update(over)
    return signals(**base)


def names(out):
    return [c.name for c in out]


def test_a_stock_runtime_run_gets_the_row(tmp_path):
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=stock_signals())
    assert names(out).count("demo-env") == 1
    assert next(c for c in out if c.name == "demo-env").status == pf.WARN           # none configured


def test_a_stock_runtime_run_with_a_healthy_demo_python_passes(tmp_path):
    c = cfg(tmp_path)
    c.demo_python = PY
    out = pf.run_preflight(c, "Cloudflare/clef", accept_credentials=False, signals=stock_signals())
    assert next(x for x in out if x.name == "demo-env").status == pf.OK and not pf.blocked(out)


def test_a_broken_demo_python_blocks_the_run_before_anything_starts(tmp_path):
    c = cfg(tmp_path)
    c.demo_python = PY
    bad = [de.Result("numpy", de.FAIL, "numpy 2.5.3") if r.name == "numpy" else r for r in HEALTHY]
    out = pf.run_preflight(c, "Cloudflare/clef", accept_credentials=False, signals=stock_signals(demo_env=lambda p: (bad, CMD)))
    assert [x.name for x in pf.blocked(out)] == ["demo-env"]


def test_a_run_that_is_not_stock_runtime_has_no_row_unless_demo_python_is_configured(tmp_path):
    called = []
    sig = signals(demo_env=lambda p: called.append(p) or (HEALTHY, CMD))
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=sig)
    assert "demo-env" not in names(out) and called == []
    c = cfg(tmp_path)
    c.demo_python = PY
    out = pf.run_preflight(c, "Cloudflare/clef", accept_credentials=False, signals=sig)
    assert "demo-env" in names(out) and called == [PY]


def test_without_the_signal_there_is_no_row_even_for_a_stock_run(tmp_path):
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=stock_signals(demo_env=None))
    assert "demo-env" not in names(out)


def test_a_lab_run_has_no_demo_row_because_its_tests_do_not_run_here(tmp_path):
    c = cfg(tmp_path)
    c.demo_python = PY
    c.mode = "lab"
    out = pf.run_preflight(c, "Cloudflare/clef", accept_credentials=False, signals=signals(demo_env=lambda p: (HEALTHY, CMD)))
    assert "demo-env" not in names(out)


def test_a_failing_signal_blocks_instead_of_raising(tmp_path):
    c = cfg(tmp_path)
    c.demo_python = PY

    def boom(p):
        raise OSError("no such file")
    out = pf.run_preflight(c, "Cloudflare/clef", accept_credentials=False, signals=signals(demo_env=boom))
    row_ = next(x for x in out if x.name == "demo-env")
    assert row_.status == pf.BLOCK and "could not check" in row_.detail and row_.reason == "config-invalid"


def test_a_resumed_stock_run_keeps_the_row(tmp_path):
    """The base row of a resumed stock run comes from the ledger (stock_recorded); the demo row follows the same route."""
    params = tmp_path / "tt-metal" / "models" / "tt_transformers" / "model_params" / "X-1B"
    params.mkdir(parents=True)
    (params / "config.json").write_text("{}")
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=signals(demo_env=lambda p: (None, CMD)),
                           stock_recorded=str(params))
    assert "demo-env" in names(out)


# ---- the real signal ----------------------------------------------------------------------------------------------------

def test_default_signals_build_the_environment_from_the_operator_home_and_the_tree_variable(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(de, "ensure", lambda env, check_only, say=None: seen.append((env, check_only)) or HEALTHY)
    monkeypatch.setenv("ORCHARD_TT_METAL_DIR", str(tmp_path / "tree"))
    c = cfg(tmp_path)
    c.operator_home = tmp_path / "op"
    results, command = pf.default_signals(c).demo_env(tmp_path / "venvs" / "demo" / "bin" / "python")
    env, check_only = seen[0]
    assert results is HEALTHY and check_only is True
    assert env.venv_dir == tmp_path / "venvs" / "demo" and env.tt_metal_dir == tmp_path / "tree"
    assert env.shared_venv == tmp_path / "op" / ".tenstorrent-venv"
    assert command == f"tt-orchard demo-env --venv-dir {tmp_path / 'venvs' / 'demo'} --tt-metal-dir {tmp_path / 'tree'}"


def test_default_signal_with_no_python_runs_no_check_and_names_the_default_venv(tmp_path, monkeypatch):
    monkeypatch.setattr(de, "ensure", lambda *a, **k: pytest.fail("no check without an interpreter"))
    monkeypatch.delenv("ORCHARD_TT_METAL_DIR", raising=False)
    monkeypatch.delenv("ORCHARD_DEMO_VENV", raising=False)
    c = cfg(tmp_path)
    c.operator_home = tmp_path / "op"
    results, command = pf.default_signals(c).demo_env(None)
    assert results is None and f"--venv-dir {tmp_path / 'op' / 'orchard-venvs' / 'tt-demo'}" in command


def test_default_signal_with_a_python_outside_a_bin_directory_runs_no_check(tmp_path, monkeypatch):
    monkeypatch.setattr(de, "ensure", lambda *a, **k: pytest.fail("no check for an unknown layout"))
    results, _ = pf.default_signals(cfg(tmp_path)).demo_env(Path("/usr/bin/python3"))
    assert results is None
