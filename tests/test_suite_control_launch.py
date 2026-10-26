# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 80, items 3, 4 and 7: the launcher that runs inside the lease (device ids and label for the container template, cleanup of
servers and containers on any exit, SIGTERM included) and the acquire / wait / release flow of `suite-control run`, which never starts a
second `gozer run` behind a lease it already holds."""
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_suite_control import CHECKOUT, _prep, ok_run, prep, stage_cfg  # noqa: E402,F401

from orchard import suite_control as sc  # noqa: E402

BDFS = ["0000:01:00.0", "0000:02:00.0", "0000:03:00.0", "0000:04:00.0"]


def status_json(held=()):
    return json.dumps({"chips": [{"bdf": b, "dev_index": i, "state": "HELD" if b in held else "FREE"} for i, b in enumerate(BDFS)]})


@pytest.fixture
def fakes(tmp_path):
    """A fake `gozer` (status --json only) and a fake `docker` on one bin directory, with a state file for the containers."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gozer").write_text(f"#!/bin/sh\ncat <<'EOF'\n{status_json()}\nEOF\n")
    state = tmp_path / "docker-state"
    state.write_text("")
    (bin_dir / "docker").write_text(f'''#!{sys.executable}
import sys
state = {str(state)!r}
a = sys.argv[1:]
ids = [l for l in open(state).read().split() if l]
if a[:2] == ["ps", "-aq"]:
    print("\\n".join(ids))
elif a[:2] == ["rm", "-f"]:
    import os
    if not os.path.exists(state + ".stubborn"):
        open(state, "w").write("")
else:
    sys.exit(3)
''')
    for n in ("gozer", "docker"):
        (bin_dir / n).chmod(0o755)
    return {"bin": bin_dir, "state": state}


def control_dir(tmp_path, cfg=None, serve_body="pass"):
    d = tmp_path / "ctl"
    d.mkdir()
    (d / "swap_config.json").write_text(json.dumps(cfg or {"chips": 2, "run_dir": str(tmp_path), "bundle_dir": "b"}))
    (d / "serve_and_compare.py").write_text(serve_body)
    return d


# ---- 3: device ids and the label ------------------------------------------------------------------------------------

def test_device_ids_follow_the_granted_chips_in_order():
    status = json.loads(status_json())
    assert sc.device_ids("0000:03:00.0,0000:04:00.0", status) == [2, 3]
    assert sc.device_ids("0000:02:00.0,0000:01:00.0", status) == [1, 0]


@pytest.mark.parametrize("visible", ["", "0000:09:00.0", "0000:01:00.0,0000:09:00.0", " "])
def test_device_ids_refuse_what_gozer_does_not_list(visible):
    with pytest.raises(sc.ControlError):
        sc.device_ids(visible, json.loads(status_json()))


def test_the_label_is_specific_to_the_control_directory(tmp_path):
    a, b = sc.test_label(tmp_path / "a"), sc.test_label(tmp_path / "b")
    assert a != b and a.startswith("orchard.suite-control=") and a == sc.test_label(tmp_path / "a")
    assert not a.startswith("orchard.test=")        # never the supervisor's own sweep label


def test_the_child_environment_has_ids_label_and_marker(tmp_path):
    d = control_dir(tmp_path)
    env = sc.child_env({"TT_VISIBLE_DEVICES": "0000:03:00.0,0000:04:00.0", "PATH": "/x"}, d, json.loads(status_json()))
    assert env["ORCHARD_DEVICE_IDS"] == "2,3" and env["ORCHARD_TEST_LABEL"] == sc.test_label(d)
    assert env[sc.CONTROL_VAR] == str(d.resolve()) and env["TT_VISIBLE_DEVICES"] == "0000:03:00.0,0000:04:00.0"


def test_the_launcher_refuses_a_lease_of_the_wrong_size(tmp_path, fakes, capsys):
    d = control_dir(tmp_path)         # needs 2 chips
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}", TT_VISIBLE_DEVICES=BDFS[0])
    assert sc.launch(d, env=env) == 2
    assert "2 chips" in capsys.readouterr().err


def test_the_launcher_refuses_without_visible_devices(tmp_path, fakes, capsys):
    d = control_dir(tmp_path)
    env = {k: v for k, v in os.environ.items() if k != "TT_VISIBLE_DEVICES"}
    env["PATH"] = f"{fakes['bin']}:{env['PATH']}"
    assert sc.launch(d, env=env) == 2 and "TT_VISIBLE_DEVICES" in capsys.readouterr().err


def test_the_serve_script_runs_with_the_variables_the_container_template_reads(tmp_path, fakes):
    out = tmp_path / "seen.json"
    d = control_dir(tmp_path, serve_body=f"import json, os\njson.dump({{k: os.environ.get(k) for k in "
                                          f"('ORCHARD_DEVICE_IDS', 'ORCHARD_TEST_LABEL', 'TT_VISIBLE_DEVICES')}}, open({str(out)!r}, 'w'))\n")
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}", TT_VISIBLE_DEVICES=f"{BDFS[2]},{BDFS[3]}")
    assert sc.launch(d, env=env) == 0
    seen = json.loads(out.read_text())
    assert seen == {"ORCHARD_DEVICE_IDS": "2,3", "ORCHARD_TEST_LABEL": sc.test_label(d), "TT_VISIBLE_DEVICES": f"{BDFS[2]},{BDFS[3]}"}


def test_the_serve_scripts_exit_code_is_returned(tmp_path, fakes):
    d = control_dir(tmp_path, serve_body="import sys\nsys.exit(5)\n")
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}", TT_VISIBLE_DEVICES=f"{BDFS[0]},{BDFS[1]}")
    assert sc.launch(d, env=env) == 5


def test_a_container_config_names_the_container_script(tmp_path, fakes):
    d = tmp_path / "ctl"
    d.mkdir()
    (d / "swap_config.json").write_text(json.dumps({"chips": 2, "run_dir": str(tmp_path), "package": "p"}))
    (d / "serve_and_compare_container.py").write_text("import sys\nsys.exit(7)\n")
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}", TT_VISIBLE_DEVICES=f"{BDFS[0]},{BDFS[1]}")
    assert sc.launch(d, env=env) == 7


# ---- 4: cleanup on every exit, SIGTERM included ---------------------------------------------------------------------------

LEAK = ("import os, subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], start_new_session=True)\n"
        "open(os.environ['LEAK_PID_FILE'], 'w').write(str(p.pid))\n")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def test_a_server_left_behind_by_a_finished_script_is_stopped(tmp_path, fakes):
    pid_file = tmp_path / "pid"
    d = control_dir(tmp_path, serve_body=LEAK)
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}", TT_VISIBLE_DEVICES=f"{BDFS[0]},{BDFS[1]}", LEAK_PID_FILE=str(pid_file))
    assert sc.launch(d, env=env) == 0
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not alive(pid)


def test_sigterm_to_the_launcher_stops_the_script_and_its_server(tmp_path, fakes):
    pid_file = tmp_path / "pid"
    body = LEAK + "time.sleep(120)\n"
    d = control_dir(tmp_path, serve_body=body)
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}", TT_VISIBLE_DEVICES=f"{BDFS[0]},{BDFS[1]}", LEAK_PID_FILE=str(pid_file),
               PYTHONPATH=str(CHECKOUT))
    proc = subprocess.Popen([sys.executable, "-m", "orchard.suite_control", "launch", "--control-dir", str(d)], env=env, cwd=str(CHECKOUT),
                            start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert pid_file.exists()
        server = int(pid_file.read_text())
        assert alive(server)
        os.kill(proc.pid, signal.SIGTERM)
        code = proc.wait(timeout=60)
        assert code == sc.EXIT_TERMINATED
        deadline = time.monotonic() + 10
        while alive(server) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not alive(server)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)


def test_labelled_containers_are_removed_and_checked(tmp_path, fakes):
    fakes["state"].write_text("abc123\n")
    d = tmp_path / "ctl"
    d.mkdir()
    (d / "swap_config.json").write_text(json.dumps({"chips": 2, "run_dir": str(tmp_path), "package": "p"}))
    (d / "serve_and_compare_container.py").write_text("pass\n")
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}", TT_VISIBLE_DEVICES=f"{BDFS[0]},{BDFS[1]}")
    assert sc.launch(d, env=env) == 0
    assert fakes["state"].read_text().strip() == ""


def test_a_container_that_survives_removal_gives_exit_4(tmp_path, fakes, capsys):
    fakes["state"].write_text("abc123\n")
    Path(str(fakes["state"]) + ".stubborn").write_text("")
    d = tmp_path / "ctl"
    d.mkdir()
    (d / "swap_config.json").write_text(json.dumps({"chips": 2, "run_dir": str(tmp_path), "package": "p"}))
    (d / "serve_and_compare_container.py").write_text("pass\n")
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}", TT_VISIBLE_DEVICES=f"{BDFS[0]},{BDFS[1]}")
    assert sc.launch(d, env=env) == sc.EXIT_UNCLEAN
    assert "abc123" in capsys.readouterr().err


def test_a_bundle_config_never_asks_docker(tmp_path, fakes):
    (fakes["bin"] / "docker").write_text("#!/bin/sh\nexit 99\n")
    d = control_dir(tmp_path)
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}", TT_VISIBLE_DEVICES=f"{BDFS[0]},{BDFS[1]}")
    assert sc.launch(d, env=env) == 0


def test_cleanup_reports_a_stray_that_would_not_stop(tmp_path):
    class Rec(dict):
        pass
    problems = sc.cleanup(tmp_path, "cfg", sweep=lambda *a, **k: [{"pid": 5, "cmdline": "x", "stopped": False}],
                          docker=lambda argv: (0, ""), uses_docker=False)
    assert problems and "5" in problems[0]
    assert sc.cleanup(tmp_path, "cfg", sweep=lambda *a, **k: [{"pid": 5, "cmdline": "x", "stopped": True}],
                      docker=lambda argv: (0, ""), uses_docker=False) == []


def test_cleanup_sweeps_by_the_marker_variable(tmp_path):
    seen = {}
    sc.cleanup(tmp_path, "cfg", sweep=lambda run_dir, **k: seen.update(run_dir=run_dir, **k) or [], docker=lambda a: (0, ""), uses_docker=False)
    assert seen["var"] == sc.CONTROL_VAR and seen["run_dir"] == tmp_path


def test_cleanup_says_when_docker_cannot_confirm(tmp_path):
    problems = sc.cleanup(tmp_path, "lbl", sweep=lambda *a, **k: [], docker=lambda argv: (1, "no daemon"), uses_docker=True)
    assert problems and "docker" in problems[0]


# ---- 7: acquire / wait / release ---------------------------------------------------------------------------------------

class Gozer:
    def __init__(self, acquire=None, wait=None, release=(0, "")):
        self.calls, self._acquire, self._wait, self._release = [], acquire, wait, release

    def __call__(self, argv):
        self.calls.append(list(argv))
        verb = argv[1]
        if verb == "acquire":
            return self._acquire
        if verb == "wait":
            return self._wait
        if verb == "release":
            return self._release
        pytest.fail(f"unexpected gozer call {argv}")


GRANT = (0, json.dumps({"granted": True, "lease_id": "L7", "chips": [BDFS[0], BDFS[1]], "dev_indices": [0, 1]}))
QUEUED = (10, json.dumps({"granted": False, "queued": True, "ticket": "t42", "position": 3, "ahead": ["a", "b"]}))


def run_cmd(prep, gozer, *args, spawn=None, tmp=None):
    out = io.StringIO()
    spawned = []

    def fake_spawn(argv, env, cwd):
        spawned.append({"argv": argv, "env": env})
        return 0
    code = sc.main(["run", "--stage-dir", str(prep["stage"]), "--orchard-dir", str(CHECKOUT), *args], out=out, run=ok_run,
                   gozer_call=gozer, spawn=spawn or fake_spawn)
    return code, out.getvalue(), spawned


def test_run_acquires_launches_under_the_lease_and_releases(prep):
    g = Gozer(acquire=GRANT)
    code, text, spawned = run_cmd(prep, g)
    assert code == 0
    assert [c[1] for c in g.calls] == ["acquire", "release"] and g.calls[1][-1] == "L7"
    assert spawned[0]["env"]["TT_VISIBLE_DEVICES"] == f"{BDFS[0]},{BDFS[1]}"
    assert spawned[0]["argv"][-4:-2] == ["launch", "--control-dir"] or "launch" in spawned[0]["argv"]


def test_run_never_starts_a_gozer_run(prep):
    g = Gozer(acquire=GRANT)
    run_cmd(prep, g)
    assert all(c[1] != "run" for c in g.calls)


def test_a_queued_request_gives_the_ticket_and_exit_3_without_launching(prep, capsys):
    g = Gozer(acquire=QUEUED)
    code, text, spawned = run_cmd(prep, g)
    assert code == 3 and spawned == []
    err = capsys.readouterr().err
    assert "t42" in err and "--ticket t42" in err
    assert [c[1] for c in g.calls] == ["acquire"]


def test_a_ticket_waits_then_launches_and_releases(prep):
    g = Gozer(wait=GRANT)
    code, _, spawned = run_cmd(prep, g, "--ticket", "t42")
    assert code == 0 and spawned
    assert [c[1] for c in g.calls] == ["wait", "release"] and "t42" in g.calls[0]
    assert "acquire" not in [c[1] for c in g.calls]


def test_a_ticket_that_is_not_granted_yet_exits_3_with_the_position(prep, capsys):
    g = Gozer(wait=(124, json.dumps({"granted": False, "ticket": "t42", "position": 2})))
    code, _, spawned = run_cmd(prep, g, "--ticket", "t42")
    assert code == 3 and spawned == []
    assert "position 2" in capsys.readouterr().err
    assert [c[1] for c in g.calls] == ["wait"]


def test_a_ticket_that_is_gone_exits_2(prep, capsys):
    g = Gozer(wait=(1, json.dumps({"granted": False, "ticket": "t42", "reason": "gone"})))
    code, _, spawned = run_cmd(prep, g, "--ticket", "t42")
    assert code == 2 and spawned == []


def test_the_lease_is_released_when_the_launcher_fails(prep):
    g = Gozer(acquire=GRANT)
    out = io.StringIO()
    code = sc.main(["run", "--stage-dir", str(prep["stage"]), "--orchard-dir", str(CHECKOUT)], out=out, run=ok_run, gozer_call=g,
                   spawn=lambda argv, env, cwd: 9)
    assert code == 9 and [c[1] for c in g.calls] == ["acquire", "release"]


def test_the_lease_is_released_when_the_run_is_terminated(prep):
    g = Gozer(acquire=GRANT)

    def spawn(argv, env, cwd):
        raise sc.Terminated()
    out = io.StringIO()
    code = sc.main(["run", "--stage-dir", str(prep["stage"]), "--orchard-dir", str(CHECKOUT)], out=out, run=ok_run, gozer_call=g, spawn=spawn)
    assert code == sc.EXIT_TERMINATED and [c[1] for c in g.calls] == ["acquire", "release"]


def test_a_failed_release_is_reported(prep, capsys):
    g = Gozer(acquire=GRANT, release=(1, "boom"))
    code, _, _ = run_cmd(prep, g)
    assert "release" in capsys.readouterr().err


def test_a_grant_without_chips_is_refused_and_released(prep):
    g = Gozer(acquire=(0, json.dumps({"granted": True, "lease_id": "L7", "chips": []})))
    code, _, spawned = run_cmd(prep, g)
    assert code == 2 and spawned == [] and [c[1] for c in g.calls] == ["acquire", "release"]


def test_the_acquire_request_names_the_chip_count_and_the_run(prep):
    g = Gozer(acquire=GRANT)
    run_cmd(prep, g, "--who", "claude:x", "--reason", "why")
    a = g.calls[0]
    assert a[a.index("--chips") + 1] == "2" and a[a.index("--who") + 1] == "claude:x" and a[a.index("--reason") + 1] == "why" and "--json" in a


# ---- the printed commands ------------------------------------------------------------------------------------------------

def test_prepare_prints_the_launcher_under_gozer_run_and_the_acquire_flow(prep):
    out = io.StringIO()
    assert sc.main(["prepare", "--stage-dir", str(prep["stage"]), "--orchard-dir", str(CHECKOUT)], out=out, run=ok_run) == 0
    printed = json.loads(out.getvalue())
    g = printed["gozer"]
    tail = g[g.index("--") + 1:]
    assert "orchard.suite_control" in tail and "launch" in tail and "--control-dir" in tail
    assert any(a.startswith("PYTHONPATH=") for a in tail)
    assert printed["run"].startswith("tt-orchard suite-control run")
