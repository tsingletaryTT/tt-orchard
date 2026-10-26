# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Third Copilot review of PR 80 (and the operator's instruction): a control must have the run's token limit, and exit 4 means one thing."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_prompt_suite_template import control_file, run, stage_for_suite, swap  # noqa: E402,F401
from test_prompt_suite_verdict import control_of, make_record, make_report  # noqa: E402
from test_suite_control import _prep, prep  # noqa: E402,F401
from test_suite_control_launch import BDFS, GRANT, Gozer, control_dir, fakes, run_cmd  # noqa: E402,F401

from orchard import prompt_suite as ps  # noqa: E402
from orchard import suite_control as sc  # noqa: E402


# ---- 1: the control's token limit ---------------------------------------------------------------------------------------------

def gate_reasons(run_limit, control_limit):
    rec = make_record(max_tokens=run_limit)
    ctl = control_of(rec, differ=False)
    if control_limit is None:
        ctl["record"].pop("max_tokens")
    else:
        ctl["record"]["max_tokens"] = control_limit
    rec["control"] = ctl
    return ps.evaluate(make_report(rec)).reasons


def test_a_control_with_another_token_limit_is_refused_by_the_gate():
    reasons = gate_reasons(512, 64)
    assert any("token limit" in r and "64" in r and "512" in r and "regenerate" in r for r in reasons)


def test_a_control_without_a_recorded_token_limit_is_refused_by_the_gate():
    assert any("token limit" in r for r in gate_reasons(512, None))


def test_a_control_and_a_run_that_both_lack_a_token_limit_are_refused():
    rec = make_record()
    rec.pop("max_tokens")
    rec["control"] = control_of(rec, differ=False)
    assert any("token limit" in r for r in ps.evaluate(make_report(rec)).reasons)


def test_a_control_with_the_same_token_limit_is_not_refused_for_it():
    assert not any("token limit" in r for r in gate_reasons(512, 512))


def test_a_refused_control_does_not_explain_a_finding():
    """The old 64-token control would have matched a truncated reply; it must not count as a control at all."""
    reasons = gate_reasons(512, 64)
    assert reasons and all("identical" not in r for r in reasons)


def test_a_control_with_another_limit_cannot_trigger_the_identical_replies_rule_as_a_pass():
    rec = make_record(max_tokens=512)
    rec["control"] = control_of(rec, differ=False)          # identical replies, but the control ran 64 tokens
    rec["control"]["record"]["max_tokens"] = 64
    v = ps.evaluate(make_report(rec))
    assert any("token limit" in r for r in v.reasons)


def test_the_template_refuses_a_control_with_another_token_limit_before_any_server_starts(swap, tmp_path):
    path = control_file(swap, tmp_path, max_tokens=64)
    swap["pid_file"].unlink(missing_ok=True)             # making the control started a server; this run must not
    r = run(swap, tmp_path, suite_control=str(path))
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "token limit" in out and "regenerate" in out and not swap["pid_file"].exists()


def test_the_template_refuses_a_control_without_a_token_limit(swap, tmp_path):
    import json
    path = control_file(swap, tmp_path)
    rep = json.loads(path.read_text())
    rep["prompt_suite"].pop("max_tokens")
    path.write_text(json.dumps(rep))
    assert run(swap, tmp_path, suite_control=str(path)).returncode == 2


def test_the_template_accepts_a_control_with_the_runs_own_limit(swap, tmp_path):
    path = control_file(swap, tmp_path)
    assert run(swap, tmp_path, suite_control=str(path)).returncode == 0


def test_the_limit_the_run_is_configured_with_is_the_one_compared(swap, tmp_path):
    path = control_file(swap, tmp_path, max_tokens=64)
    assert run(swap, tmp_path, suite_control=str(path), suite_max_tokens=64).returncode == 0


# ---- 3: exit 4 from the serve script is not an unconfirmed cleanup ----------------------------------------------------------------------

def launch_with_serve_exit(tmp_path, fakes, code):
    d = control_dir(tmp_path, serve_body=f"import sys\nsys.exit({code})\n")
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}", TT_VISIBLE_DEVICES=f"{BDFS[0]},{BDFS[1]}")
    env.pop(sc.LOCK_FD_VAR, None)
    return d, env


def test_a_serve_script_that_exits_4_is_reported_as_a_serve_failure_after_a_confirmed_cleanup(tmp_path, fakes):
    d, env = launch_with_serve_exit(tmp_path, fakes, 4)
    assert sc.launch(d, env=env) == sc.EXIT_SERVE_FAILED != sc.EXIT_UNCLEAN


@pytest.mark.parametrize("code", [0, 2, 3, 5, 6])
def test_other_serve_exit_codes_pass_through(tmp_path, fakes, code):
    d, env = launch_with_serve_exit(tmp_path, fakes, code)
    assert sc.launch(d, env=env) == code


def test_an_unconfirmed_cleanup_still_gives_4_even_when_the_serve_script_exited_4(tmp_path, fakes):
    d, env = launch_with_serve_exit(tmp_path, fakes, 4)
    assert sc.launch(d, env=env, sweep=lambda *a, **k: [{"pid": 5, "cmdline": "x", "stopped": False}]) == sc.EXIT_UNCLEAN


def test_run_releases_the_lease_after_a_serve_failure_that_was_cleaned_up(prep):
    g = Gozer(acquire=GRANT)
    code, _, _ = run_cmd(prep, g, spawn=lambda argv, env, cwd: sc.EXIT_SERVE_FAILED)
    assert code == sc.EXIT_SERVE_FAILED and [c[1] for c in g.calls] == ["acquire", "release"]
