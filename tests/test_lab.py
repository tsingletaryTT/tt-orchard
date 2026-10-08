# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The lab helper (orchard/lab.py) and its client (orchard/labclient.py).

The helper runs on the lab box; here it runs as a local process through the same JSON-lines
protocol (the transport is `bash -c` instead of `ssh <host>`). A fake gozer stands in for the real
one. The tests that matter most prove what must never happen: a hardware test outliving its deadline
or the brain's connection, chips staying leased after the brain goes away, and gozer seeing a lease
owner that is not alive on the lab.
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

from orchard import labclient, lab
from orchard.adapters import Lease

REPO = Path(__file__).resolve().parent.parent

FAKE_GOZER = r'''#!/usr/bin/env python3
"""A tiny gozer: acquire/release/status with leases in a JSON file. It records every call."""
import json, os, sys
state = os.environ["FAKE_GOZER_STATE"]
log = state + ".log"
with open(log, "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\n")
try:
    leases = json.load(open(state))
except (OSError, ValueError):
    leases = {}
cmd = sys.argv[1]
if cmd == "acquire":
    owner = int(sys.argv[sys.argv.index("--owner-pid") + 1])
    n = int(sys.argv[sys.argv.index("--chips") + 1])
    lid = f"L{len(leases) + 1}"
    chips = [f"0000:0{i + 1}:00.0" for i in range(n)]
    leases[lid] = {"owner": owner, "chips": chips}
    json.dump(leases, open(state, "w"))
    print(json.dumps({"granted": True, "lease_id": lid, "chips": chips, "dev_indices": list(range(n)),
                      "owner_pid": owner, "units": chips, "env": {"TT_VISIBLE_DEVICES": ",".join(chips)},
                      "requested": n, "expanded": False}))
elif cmd == "release":
    lid = sys.argv[2]
    leases.pop(lid, None)
    json.dump(leases, open(state, "w"))
    print(json.dumps({"released": lid}))
elif cmd == "status":
    print(json.dumps({"chips": []}))
'''


@pytest.fixture
def labroot(tmp_path):
    root = tmp_path / "srv-orchard"
    root.mkdir()
    gz = tmp_path / "bin" / "gozer"
    gz.parent.mkdir()
    gz.write_text(FAKE_GOZER)
    gz.chmod(0o755)
    return root, gz


def connect(root, gz, tmp_path, **kw):
    env = {"FAKE_GOZER_STATE": str(tmp_path / "gozer-state.json"),
           "PYTHONPATH": str(REPO)}
    return labclient.Lab.local(root=root, gozer=str(gz), env=env, **kw)


def gozer_log(tmp_path) -> list[str]:
    p = tmp_path / "gozer-state.json.log"
    return p.read_text().splitlines() if p.exists() else []


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][:1] != "Z"
    except OSError:
        return True


# ---- the helper ---------------------------------------------------------------------------------

def test_hello_names_the_helper_process_and_the_lab(labroot, tmp_path):
    root, gz = labroot
    with connect(root, gz, tmp_path) as lab_:
        info = lab_.hello
        assert info["pid"] > 1 and pid_alive(info["pid"])
        assert info["root"] == str(root) and info["hostname"]


def test_a_command_runs_on_the_lab_and_its_output_streams_back(labroot, tmp_path):
    root, gz = labroot
    out = tmp_path / "out.txt"
    with connect(root, gz, tmp_path) as lab_, open(out, "wb") as f:
        code, timed_out = lab_.shell("echo hello; echo there 1>&2; exit 3", cwd=tmp_path, env={}, timeout=10,
                                     stdout=f)
    assert (code, timed_out) == (3, False)
    assert out.read_text() == "hello\nthere\n"


def test_a_command_is_killed_on_the_lab_at_its_deadline(labroot, tmp_path):
    root, gz = labroot
    pidfile = tmp_path / "child.pid"
    with connect(root, gz, tmp_path) as lab_, open(tmp_path / "o", "wb") as f:
        t0 = time.time()
        code, timed_out = lab_.shell(f"sleep 30 & echo $! > {pidfile}; wait", cwd=tmp_path, env={}, timeout=1,
                                     stdout=f)
        assert time.time() - t0 < 10
    assert timed_out and code is None
    assert not pid_alive(int(pidfile.read_text()))           # the whole group went, not just bash


def test_the_brain_going_away_kills_running_tests_and_releases_leases(labroot, tmp_path):
    root, gz = labroot
    pidfile = tmp_path / "child.pid"
    lab_ = connect(root, gz, tmp_path)
    adapter = labclient.LabGozerAdapter(lab_, gozer=str(gz))
    lease = adapter.acquire(2, "orchard:supervisor", "test")
    assert list(lease.chips) == ["0000:01:00.0", "0000:02:00.0"]
    lab_.start_shell(f"sleep 60 & echo $! > {pidfile}; wait", cwd=tmp_path, env={}, timeout=120)
    deadline = time.time() + 5
    while not pidfile.exists() and time.time() < deadline:
        time.sleep(0.05)
    child = int(pidfile.read_text())
    lab_.drop()                                             # the ssh connection breaks
    deadline = time.time() + 10
    while time.time() < deadline and (pid_alive(child) or "release L1" not in " ".join(gozer_log(tmp_path))):
        time.sleep(0.1)
    assert not pid_alive(child)
    assert "release L1" in gozer_log(tmp_path)[-1]


def test_a_released_lease_is_not_released_again_when_the_brain_leaves(labroot, tmp_path):
    root, gz = labroot
    lab_ = connect(root, gz, tmp_path)
    adapter = labclient.LabGozerAdapter(lab_, gozer=str(gz))
    lease = adapter.acquire(1, "orchard:supervisor", "test")
    adapter.release(lease)
    lab_.close()
    time.sleep(0.5)
    assert [line.split()[0] for line in gozer_log(tmp_path)] == ["acquire", "release"]


def test_gozer_sees_the_helper_as_the_lease_owner(labroot, tmp_path):
    root, gz = labroot
    with connect(root, gz, tmp_path) as lab_:
        adapter = labclient.LabGozerAdapter(lab_, gozer=str(gz))
        assert adapter.owner_pid == lab_.hello["pid"]
        adapter.acquire(1, "orchard:supervisor", "test")
        call = gozer_log(tmp_path)[0].split()
        assert call[call.index("--owner-pid") + 1] == str(lab_.hello["pid"])


def test_lab_file_operations(labroot, tmp_path):
    root, gz = labroot
    cache = root / "cache" / "org--m" / "tt_cache"
    cache.mkdir(parents=True)
    (cache / ".orchard-model").write_text("org/m@abc\n")
    (cache / "w.bin").write_bytes(b"x" * 10)
    with connect(root, gz, tmp_path) as lab_:
        assert lab_.disk_free_gb(root) > 0
        audit = lab_.cache_audit([str(cache), str(root / "cache" / "missing")])
        assert audit[str(cache)] == {"exists": True, "marker": "org/m@abc", "files": 1}   # the marker is not counted
        assert audit[str(root / "cache" / "missing")] == {"exists": False, "marker": None, "files": 0}
        aside = lab_.move_aside(str(cache))
        assert aside and not cache.exists() and Path(aside).exists()
        assert lab_.move_aside(str(cache)) is None


def test_file_operations_stay_under_the_lab_root(labroot, tmp_path):
    root, gz = labroot
    with connect(root, gz, tmp_path) as lab_:
        with pytest.raises(labclient.LabError, match="outside the lab root"):
            lab_.move_aside("/etc")
        with pytest.raises(labclient.LabError, match="outside the lab root"):
            lab_.cache_audit([str(root / ".." / "x")])


def test_an_unknown_request_is_an_error_not_a_crash(labroot, tmp_path):
    root, gz = labroot
    with connect(root, gz, tmp_path) as lab_:
        with pytest.raises(labclient.LabError, match="unknown"):
            lab_.call("format_the_disk")
        assert lab_.call("hello")["pid"] == lab_.hello["pid"]


def test_gozer_commands_run_through_the_helper_as_its_children(labroot, tmp_path):
    root, gz = labroot
    with connect(root, gz, tmp_path) as lab_:
        res = lab_.run_argv([str(gz), "status", "--json"], 30)
        assert res.returncode == 0 and json.loads(res.stdout) == {"chips": []}
        res = lab_.run_argv(["false"], 30)
        assert res.returncode == 1


# ---- the client's data movement -----------------------------------------------------------------

def test_sync_uses_rsync_over_ssh_with_the_same_path_on_both_sides():
    calls = []

    def run(argv, timeout, **kw):
        calls.append(argv)
        from orchard.commands import CommandResult
        return CommandResult(argv, 0, "", "")

    sync = labclient.Sync(host="node4", ssh=["ssh", "-T", "-o", "BatchMode=yes"], run=run)
    sync.up("/srv/orchard/runs/r/stages/2")
    sync.down("/srv/orchard/runs/r/stages/2")
    up, down = calls
    assert up[0] == "rsync" and "-a" in up and "--mkpath" in up
    assert up[-2:] == ["/srv/orchard/runs/r/stages/2/", "node4:/srv/orchard/runs/r/stages/2/"]
    assert down[-2:] == ["node4:/srv/orchard/runs/r/stages/2/", "/srv/orchard/runs/r/stages/2/"]
    assert up[up.index("-e") + 1] == "ssh -T -o BatchMode=yes"
    assert "--delete" not in up and "--delete" not in down      # never removes anything on either side


def test_a_failed_sync_is_an_error():
    def run(argv, timeout, **kw):
        from orchard.commands import CommandResult
        return CommandResult(argv, 23, "", "rsync: some files could not be transferred")

    with pytest.raises(labclient.LabError, match="could not be transferred"):
        labclient.Sync(host="node4", ssh=["ssh"], run=run).up("/srv/orchard/x")


def test_the_lab_helper_is_launched_from_the_synced_checkout():
    argv = labclient.ssh_launch_argv(host="node4", root="/srv/orchard", gozer="gozer",
                                     path=["~/.local/bin", "~/.tenstorrent-venv/bin"], python="python3")
    assert argv[:4] == ["ssh", "-T", "-o", "BatchMode=yes"] and argv[-2] == "node4"
    assert "ServerAliveInterval=15" in argv                    # a dead connection is noticed
    remote = argv[-1]
    assert "cd /srv/orchard/orchard" in remote and "-m orchard.lab serve" in remote
    assert "--root /srv/orchard" in remote and "--gozer gozer" in remote
