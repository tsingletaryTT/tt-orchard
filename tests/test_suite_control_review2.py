# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Second Copilot review of PR 80 on orchard/suite_control.py. Each test group names its finding."""
import fcntl
import io
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_suite_control import CHECKOUT, _prep, ok_run, prep, stage_cfg  # noqa: E402,F401
from test_suite_control_launch import BDFS, GRANT, Gozer, control_dir, fakes, run_cmd  # noqa: E402,F401

from orchard import suite_control as sc  # noqa: E402


# ---- 1: a lease is not released while the launcher's cleanup is unconfirmed -------------------------------------------------

def run_with_spawn(prep, gozer, spawn, *args):
    out = io.StringIO()
    code = sc.main(["run", "--stage-dir", str(prep["stage"]), "--orchard-dir", str(CHECKOUT), *args], out=out, run=ok_run, gozer_call=gozer, spawn=spawn)
    return code, out.getvalue()


def test_an_unconfirmed_cleanup_keeps_the_lease_and_reports_its_id(prep, capsys):
    g = Gozer(acquire=GRANT)
    code, _ = run_with_spawn(prep, g, lambda argv, env, cwd: sc.EXIT_UNCLEAN)
    assert code == sc.EXIT_UNCLEAN
    assert [c[1] for c in g.calls] == ["acquire"]
    err = capsys.readouterr().err
    assert "L7" in err and "not released" in err


@pytest.mark.parametrize("launcher_code", [-9, -15])
def test_a_launcher_killed_by_a_signal_keeps_the_lease(prep, capsys, launcher_code):
    g = Gozer(acquire=GRANT)
    code, _ = run_with_spawn(prep, g, lambda argv, env, cwd: launcher_code)
    assert code == sc.EXIT_UNCLEAN and [c[1] for c in g.calls] == ["acquire"]


def test_a_terminated_run_whose_launcher_confirmed_its_cleanup_releases(prep):
    g = Gozer(acquire=GRANT)

    def spawn(argv, env, cwd):
        raise sc.Terminated(sc.EXIT_TERMINATED)
    code, _ = run_with_spawn(prep, g, spawn)
    assert code == sc.EXIT_TERMINATED and [c[1] for c in g.calls] == ["acquire", "release"]


@pytest.mark.parametrize("code_seen", [None, -9, sc.EXIT_UNCLEAN, 1])
def test_a_terminated_run_whose_cleanup_is_not_confirmed_keeps_the_lease(prep, code_seen):
    g = Gozer(acquire=GRANT)

    def spawn(argv, env, cwd):
        raise sc.Terminated(code_seen)
    code, _ = run_with_spawn(prep, g, spawn)
    assert code == sc.EXIT_UNCLEAN and [c[1] for c in g.calls] == ["acquire"]


def test_other_launcher_exit_codes_still_release(prep):
    for launcher_code in (0, 2, 5, sc.EXIT_TERMINATED):
        g = Gozer(acquire=GRANT)
        run_with_spawn(prep, g, lambda argv, env, cwd, c=launcher_code: c)
        assert [c[1] for c in g.calls] == ["acquire", "release"], launcher_code


def child_script(tmp_path, body):
    f = tmp_path / "child.py"
    f.write_text(body)
    return [sys.executable, str(f)]


def terminate_self_when(ready: Path):
    def term():
        deadline = time.monotonic() + 20
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGTERM)
    t = threading.Thread(target=term)
    t.start()
    return t


def test_spawn_reports_the_launchers_exit_code_when_it_stops_cleanly(tmp_path):
    ready = tmp_path / "ready"
    argv = child_script(tmp_path, f"import signal, sys, time\nsignal.signal(signal.SIGTERM, lambda *a: sys.exit(143))\n"
                                  f"open({str(ready)!r}, 'w').write('1')\ntime.sleep(60)\n")
    t = terminate_self_when(ready)
    try:
        with pytest.raises(sc.Terminated) as info:
            sc._spawn(argv, dict(os.environ), str(tmp_path))
    finally:
        t.join()
    assert info.value.code == 143


def test_spawn_says_unknown_when_the_launcher_had_to_be_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "STOP_GRACE_S", 0.2)
    ready = tmp_path / "ready"
    argv = child_script(tmp_path, f"import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                                  f"open({str(ready)!r}, 'w').write('1')\ntime.sleep(60)\n")
    t = terminate_self_when(ready)
    try:
        with pytest.raises(sc.Terminated) as info:
            sc._spawn(argv, dict(os.environ), str(tmp_path))
    finally:
        t.join()
    assert info.value.code != sc.EXIT_TERMINATED and info.value.code in (None, -9)


def test_terminated_carries_a_code_and_defaults_to_unknown():
    assert sc.Terminated().code is None and sc.Terminated(143).code == 143


# ---- 2: one control at a time per stage, from prepare to release ------------------------------------------------------------

def test_the_lock_is_outside_the_directory_prepare_deletes(prep):
    assert sc.lock_path(prep["stage"] / sc.CONTROL_DIR) == prep["stage"] / "suite-control.lock"
    assert sc.lock_path(prep["stage"] / sc.CONTROL_DIR).parent == prep["stage"]       # not inside the directory prepare deletes


def test_a_second_holder_of_the_lock_is_refused(tmp_path):
    with sc.stage_lock(tmp_path):
        with pytest.raises(sc.ControlError, match="already in use"):
            with sc.stage_lock(tmp_path):
                pass
    with sc.stage_lock(tmp_path):        # free again after the first exit
        pass


def test_prepare_while_a_control_runs_is_refused_and_touches_nothing(prep, capsys):
    ctl = prep["stage"] / sc.CONTROL_DIR
    ctl.mkdir()
    (ctl / "evidence").mkdir()
    (ctl / "evidence" / "swap-check.json").write_text("live")
    with sc.stage_lock(prep["stage"]):
        out = io.StringIO()
        code = sc.main(["prepare", "--stage-dir", str(prep["stage"]), "--orchard-dir", str(CHECKOUT)], out=out, run=ok_run)
    assert code == 2 and "already in use" in capsys.readouterr().err
    assert (ctl / "evidence" / "swap-check.json").read_text() == "live"


def test_run_while_a_control_runs_is_refused_before_it_asks_gozer(prep):
    g = Gozer(acquire=GRANT)
    with sc.stage_lock(prep["stage"]):
        code, _, spawned = run_cmd(prep, g)
    assert code == 2 and g.calls == [] and spawned == []


def test_a_queued_run_leaves_the_lock_free(prep):
    queued = (10, json.dumps({"granted": False, "queued": True, "ticket": "t1", "position": 1}))
    run_cmd(prep, Gozer(acquire=queued))
    with sc.stage_lock(prep["stage"]):
        pass


def test_the_lock_is_held_while_the_launcher_runs_and_handed_to_it(prep):
    seen = {}

    def spawn(argv, env, cwd):
        seen["fd"] = env.get(sc.LOCK_FD_VAR)
        try:
            with sc.stage_lock(prep["stage"]):
                seen["second"] = "got it"
        except sc.ControlError:
            seen["second"] = "refused"
        return 0
    run_cmd(prep, Gozer(acquire=GRANT), spawn=spawn)
    assert seen["second"] == "refused" and seen["fd"] and seen["fd"].isdigit()
    with sc.stage_lock(prep["stage"]):       # and free afterwards
        pass


def test_the_launcher_started_by_run_may_reuse_the_inherited_lock(tmp_path):
    fd = os.open(tmp_path / sc.LOCK_NAME, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    with sc.stage_lock(tmp_path, inherited_fd=fd) as got:
        assert got == fd
    os.close(fd)


def test_launch_refuses_while_another_launch_holds_the_lock(tmp_path, fakes, capsys):
    d = control_dir(tmp_path)
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}", TT_VISIBLE_DEVICES=f"{BDFS[0]},{BDFS[1]}")
    env.pop(sc.LOCK_FD_VAR, None)
    with sc.stage_lock(d.parent):
        assert sc.launch(d, env=env) == 2
    assert "already in use" in capsys.readouterr().err


def test_launch_takes_and_frees_its_own_lock_when_started_directly(tmp_path, fakes):
    d = control_dir(tmp_path)
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}", TT_VISIBLE_DEVICES=f"{BDFS[0]},{BDFS[1]}")
    env.pop(sc.LOCK_FD_VAR, None)
    assert sc.launch(d, env=env) == 0
    with sc.stage_lock(d.parent):
        pass


def test_two_stages_do_not_block_each_other(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    with sc.stage_lock(a):
        with sc.stage_lock(b):
            pass


# ---- 3: gozer grants whole boards ------------------------------------------------------------------------------------------------

def one_chip_manifest(prep):
    (prep["bundle"] / "tt_kernel_manifest.json").write_text(json.dumps({"device_count": 1}))


def test_a_larger_grant_than_asked_is_accepted_and_released_afterwards(prep):
    one_chip_manifest(prep)
    g = Gozer(acquire=GRANT)             # a two-chip board for a one-chip request
    code, _, spawned = run_cmd(prep, g)
    assert code == 0 and spawned and [c[1] for c in g.calls] == ["acquire", "release"]
    assert spawned[0]["env"]["TT_VISIBLE_DEVICES"] == f"{BDFS[0]},{BDFS[1]}"       # the whole lease is kept; the launcher narrows it


def test_a_grant_smaller_than_needed_is_refused_and_released(prep):
    one = (0, json.dumps({"granted": True, "lease_id": "L7", "chips": [BDFS[0]]}))
    g = Gozer(acquire=one)
    code, _, spawned = run_cmd(prep, g)         # the bundle needs 2
    assert code == 2 and spawned == [] and [c[1] for c in g.calls] == ["acquire", "release"]


def test_the_launcher_narrows_a_whole_board_lease_to_the_required_chips(tmp_path, fakes):
    seen = tmp_path / "seen.json"
    d = control_dir(tmp_path, cfg={"chips": 1, "run_dir": str(tmp_path), "bundle_dir": "b"},
                    serve_body=f"import json, os\njson.dump({{k: os.environ.get(k) for k in "
                               f"('ORCHARD_DEVICE_IDS', 'TT_VISIBLE_DEVICES')}}, open({str(seen)!r}, 'w'))\n")
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}", TT_VISIBLE_DEVICES=f"{BDFS[2]},{BDFS[3]}")
    env.pop(sc.LOCK_FD_VAR, None)
    assert sc.launch(d, env=env) == 0
    assert json.loads(seen.read_text()) == {"ORCHARD_DEVICE_IDS": "2", "TT_VISIBLE_DEVICES": BDFS[2]}


def test_the_launcher_still_refuses_fewer_chips_than_needed(tmp_path, fakes, capsys):
    d = control_dir(tmp_path)           # needs 2
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}", TT_VISIBLE_DEVICES=BDFS[0])
    env.pop(sc.LOCK_FD_VAR, None)
    assert sc.launch(d, env=env) == 2 and "at least 2 chips" in capsys.readouterr().err


def test_narrow_keeps_the_leased_order():
    assert sc.narrow("a,b,c,d", 2) == "a,b" and sc.narrow("a,b", 2) == "a,b" and sc.narrow(" a , b ", 1) == "a"


# ---- 4: the chosen gozer executable reaches the launcher ------------------------------------------------------------------------

def test_a_custom_gozer_is_forwarded_to_the_launcher(prep):
    g = Gozer(acquire=GRANT)
    code, _, spawned = run_cmd(prep, g, "--gozer", "/opt/gozer-x/bin/gozer")
    argv = spawned[0]["argv"]
    assert argv[argv.index("--gozer") + 1] == "/opt/gozer-x/bin/gozer"
    assert g.calls[0][0] == "/opt/gozer-x/bin/gozer"


def test_a_relative_gozer_path_is_made_absolute(prep, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    g = Gozer(acquire=GRANT)
    code, _, spawned = run_cmd(prep, g, "--gozer", "./tools/gozer")
    argv = spawned[0]["argv"]
    assert argv[argv.index("--gozer") + 1] == str(tmp_path / "tools" / "gozer")
    assert g.calls[0][0] == str(tmp_path / "tools" / "gozer")


def test_the_default_gozer_is_not_added_to_the_launcher_command(prep):
    g = Gozer(acquire=GRANT)
    _, _, spawned = run_cmd(prep, g)
    assert "--gozer" not in spawned[0]["argv"]


def test_the_printed_launcher_command_carries_the_custom_executable_too(prep):
    out = io.StringIO()
    sc.main(["prepare", "--stage-dir", str(prep["stage"]), "--orchard-dir", str(CHECKOUT), "--gozer", "/opt/g/gozer"], out=out, run=ok_run)
    printed = json.loads(out.getvalue())["launch"]
    assert printed[printed.index("--gozer") + 1] == "/opt/g/gozer"


def test_launch_uses_the_gozer_it_is_given(tmp_path, fakes):
    custom = tmp_path / "bin" / "mygozer"
    custom.write_text((fakes["bin"] / "gozer").read_text())
    custom.chmod(0o755)
    (fakes["bin"] / "gozer").write_text("#!/bin/sh\nexit 9\n")          # the default one is broken on purpose
    d = control_dir(tmp_path)
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}", TT_VISIBLE_DEVICES=f"{BDFS[0]},{BDFS[1]}")
    env.pop(sc.LOCK_FD_VAR, None)
    assert sc.launch(d, env=env, gozer=str(custom)) == 0
    assert sc.launch(d, env=env) == 2


def test_the_launch_action_passes_the_gozer_option_on(tmp_path, fakes, monkeypatch):
    seen = {}
    monkeypatch.setattr(sc, "launch", lambda control_dir, **kw: seen.update(kw) or 0)
    sc.main(["launch", "--control-dir", str(tmp_path), "--gozer", "/x/g"])
    assert seen["gozer"] == "/x/g"
