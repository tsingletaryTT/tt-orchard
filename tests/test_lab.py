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
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from orchard import labclient, lab
from waits import PATIENCE_S, wait_until
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
elif cmd == "queue":
    try:
        print(json.dumps({"queue": json.load(open(state + ".queue"))}))
    except (OSError, ValueError):
        print(json.dumps({"queue": []}))
elif cmd == "cancel":
    print(json.dumps({"cancelled": sys.argv[2]}))
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


LABS: list = []                 # every connection this module has opened, so a test and its teardown can look at them


def connect(root, gz, tmp_path, **kw):
    env = {"FAKE_GOZER_STATE": str(tmp_path / "gozer-state.json"),
           "PYTHONPATH": str(REPO)}
    lab_ = labclient.Lab.local(root=root, gozer=str(gz), env=env, **kw)
    LABS.append(lab_)
    return lab_


def close_every_lab() -> None:
    """Close every connection this module opened and forget them. A teardown must not hide the failure of the test itself, so a close that
    raises does not stop the others from being closed."""
    for lab_ in LABS:
        try:
            lab_.close()
        except Exception:
            pass
    LABS.clear()


@pytest.fixture(autouse=True)
def _close_every_lab():
    """A test that opens a connection without `with` and fails before it closes it would leave the helper, and whatever the helper runs, up
    until its own deadline. Whatever a test opened is closed when the test ends (review of PR 447, round 2)."""
    yield
    close_every_lab()


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
        t0 = time.monotonic()
        # The child sleeps for 120 s, so a command that is not killed at its deadline returns after 120 s with exit code 0 (and the
        # check below fails on that). The stopwatch only has to tell 1 s from 120 s: 60 s holds on a box where everything is 30 times slower.
        # The helper counts the deadline from the start of the shell, so the shell has to write its child's pid inside it: 5 s leaves room
        # for a starved process (1 s did not).
        code, timed_out = lab_.shell(f"sleep 120 & echo $! > {pidfile}; wait", cwd=tmp_path, env={}, timeout=5,
                                     stdout=f)
        assert time.monotonic() - t0 < 60
    assert timed_out and code is None
    assert pidfile.exists() and pidfile.read_text().strip().isdigit(), \
        "the shell was killed before it wrote the pid of its child, so the group kill could not be checked: the box was too slow for the 5 s deadline"
    assert not pid_alive(int(pidfile.read_text()))           # the whole group went, not just bash


def test_the_brain_going_away_kills_running_tests_and_releases_leases(labroot, tmp_path):
    root, gz = labroot
    pidfile = tmp_path / "child.pid"
    lab_ = connect(root, gz, tmp_path)
    child = None
    try:
        adapter = labclient.LabGozerAdapter(lab_, gozer=str(gz))
        lease = adapter.acquire(2, "orchard:supervisor", "test")
        assert list(lease.chips) == ["0000:01:00.0", "0000:02:00.0"]
        # Two things must outlast both polling phases below (the wait for the pid and the wait for the cleanup, each `PATIENCE_S`). The command's
        # own deadline: if it expired inside the second phase, the ordinary command timeout could kill the child and a helper that never kills it
        # on a dropped connection would pass (review of PR 447). And the child itself, which must outlive the deadline so that nothing but the
        # helper can end it inside the test.
        command_deadline_s = 600
        assert command_deadline_s > 2 * PATIENCE_S + 30
        lab_.start_shell(f"sleep {command_deadline_s + 300} & echo $! > {pidfile}; wait", cwd=tmp_path, env={}, timeout=command_deadline_s)
        # `echo $! > file` creates the file before it writes the pid, so the file can exist and be empty: wait for the digits.
        wait_until(lambda: pidfile.exists() and pidfile.read_text().strip().isdigit(), "the test's shell to write its child's pid")
        child = int(pidfile.read_text())
        lab_.drop()                                         # the ssh connection breaks
        wait_until(lambda: not pid_alive(child) and "release L1 --json" in gozer_log(tmp_path),
                   "the helper to kill the running test and release the lease after the connection broke")
        assert not pid_alive(child)
        assert "release L1 --json" in gozer_log(tmp_path)
    finally:
        # A run that fails anywhere above, the wait for the pid included, leaves nothing behind: closing the connection makes the helper stop what
        # it runs and release what it holds (review of PR 447, round 2), and a child that is still there is killed by its pid, read from the file
        # when the test never got that far.
        lab_.close()
        if child is None and pidfile.exists() and pidfile.read_text().strip().isdigit():
            child = int(pidfile.read_text())
        if child is not None and pid_alive(child):
            os.kill(child, signal.SIGKILL)


def test_a_released_lease_is_not_released_again_when_the_brain_leaves(labroot, tmp_path):
    root, gz = labroot
    lab_ = connect(root, gz, tmp_path)
    adapter = labclient.LabGozerAdapter(lab_, gozer=str(gz))
    lease = adapter.acquire(1, "orchard:supervisor", "test")
    adapter.release(lease)
    helper = int(lab_.hello["pid"])
    lab_.close()
    # Nothing can be proved about a call that never comes by waiting a fixed time: the helper may not have started its clean-up yet.
    # Its exit is the point after which it makes no more calls.
    wait_until(lambda: not pid_alive(helper), "the helper to exit after the connection closed")
    assert [line.split()[0] for line in gozer_log(tmp_path) if line.split()[0] != "queue"] == ["acquire", "release"]


def test_when_the_brain_leaves_the_helper_cancels_its_own_queue_tickets(labroot, tmp_path):
    """The supervisor was killed while its 4-chip request waited in node4's queue. The helper killed the
    waiting gozer process, but the ticket stayed, and gozer opened claim windows for it, holding freed
    chips back from later requests. The helper now cancels the tickets its own pid owns."""
    root, gz = labroot
    lab_ = connect(root, gz, tmp_path)
    mine = int(lab_.hello["pid"])
    (tmp_path / "gozer-state.json.queue").write_text(json.dumps([
        {"ticket": "T1", "pid": mine, "who": "orchard:supervisor", "min_chips": 4},
        {"ticket": "T2", "pid": mine + 1, "who": "someone-else", "min_chips": 1}]))
    lab_.close()
    wait_until(lambda: "cancel T1" in gozer_log(tmp_path), "the helper to cancel the queue ticket its pid owned")
    log = gozer_log(tmp_path)
    assert "cancel T1" in log and "cancel T2" not in log


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


def test_mirror_up_makes_the_lab_copy_of_one_directory_exactly_the_brains():
    """Only a stage directory is mirrored, and only towards the lab: a failed attempt's files on the
    lab must not come back into the next attempt's directory with the next sync down."""
    calls = []

    def run(argv, timeout, **kw):
        calls.append(argv)
        from orchard.commands import CommandResult
        return CommandResult(argv, 0, "", "")

    labclient.Sync(host="node4", ssh=["ssh"], run=run).mirror_up("/srv/orchard/runs/r/stages/2")
    (argv,) = calls
    assert "--delete" in argv
    assert argv[-2:] == ["/srv/orchard/runs/r/stages/2/", "node4:/srv/orchard/runs/r/stages/2/"]


def test_mirror_up_refuses_anything_but_a_stage_directory():
    def run(argv, timeout, **kw):
        raise AssertionError("nothing may run")
    sync = labclient.Sync(host="node4", ssh=["ssh"], run=run)
    for path in ("/srv/orchard", "/srv/orchard/runs/r", "/srv/orchard/hf", "/srv/orchard/runs/r/stages",
                 "/srv/orchard/runs/r/stages/2/evidence", "/srv/orchard/runs/r/stages/../stages/2",
                 "/srv/orchard/runs/r/stages/2.partial-1"):
        with pytest.raises(labclient.LabError, match="stage directory"):
            sync.mirror_up(path)


def test_sync_down_leaves_out_the_files_it_is_told_to():
    """The test's output streams into a file on the brain; the lab's copy of that file is the empty one
    the sync up carried. Without the exclusion the sync down overwrote every streamed output with it."""
    calls = []

    def run(argv, timeout, **kw):
        calls.append(argv)
        from orchard.commands import CommandResult
        return CommandResult(argv, 0, "", "")

    labclient.Sync(host="node4", ssh=["ssh"], run=run).down("/srv/orchard/runs/r/stages/4",
                                                             exclude=["tests/1/output.txt"])
    (argv,) = calls
    assert argv[argv.index("--exclude") + 1] == "/tests/1/output.txt"      # anchored at the stage directory
    with pytest.raises(labclient.LabError):
        labclient.Sync(host="node4", ssh=["ssh"], run=run).down("/x", exclude=["../outside"])


def test_the_lab_side_counts_the_labs_chips_from_gozer():
    from orchard.commands import CommandResult

    class Helper:
        hello = {"pid": 4242, "hostname": "node4"}

        def run_argv(self, argv, timeout, **kw):
            text = ("grain: chip   (2 boards, 2 chips)\nboard A  (p150a)\n  chip 0  0000:01:00.0  FREE\n"
                    "board B  (p150a)\n  chip 1  0000:03:00.0  FREE\n")
            return CommandResult(list(argv), 0, text, "")
    side = labclient.LabSide(Helper(), labclient.Sync(host="node4"), host="node4", root="/srv/orchard")
    assert side.chips == 2 and side.info["chips"] == 2


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



# ---- what a test needs from the HF cache, and a directory on the lab -------------------------------

def _hub(tmp_path):
    """An HF cache with the new model, the base model and a drafter, the way the hub lays them out."""
    hub = tmp_path / "hf" / "hub"
    for repo, rev in (("models--Altworld--Hemmingway-1", "def"), ("models--Qwen--Qwen3.8-27B", "abc"),
                      ("models--incoai--Qwen3.8-27B-DFlash2", "d1")):
        (hub / repo / "snapshots" / rev).mkdir(parents=True)
        (hub / repo / "blobs").mkdir()
        (hub / repo / "blobs" / "b1").write_text("w")
        (hub / repo / "snapshots" / rev / "model.safetensors").symlink_to(hub / repo / "blobs" / "b1")
    return hub


def test_a_test_needs_the_new_model_and_its_drafter_and_never_the_base(tmp_path):
    hub = _hub(tmp_path)
    stage = tmp_path / "run" / "stages" / "2"
    (stage / "model-dir").mkdir(parents=True)
    (stage / "model-dir" / "model.safetensors").symlink_to(hub / "models--Altworld--Hemmingway-1" / "blobs" / "b1")
    (stage / "model-dir" / "config.json").write_text("{}")         # copied from the base: no link
    (stage / "run.sh").write_text('export DFLASH_WEIGHTS="incoai/Qwen3.8-27B-DFlash2@d1"\n')
    inputs = {"model": str(hub / "models--Altworld--Hemmingway-1" / "snapshots" / "def"),
              "base": str(hub / "models--Qwen--Qwen3.8-27B" / "snapshots" / "abc")}
    got = labclient.lab_hf_repos(stage, tmp_path / "hf", inputs)
    assert [g.name for g in got] == ["models--Altworld--Hemmingway-1", "models--incoai--Qwen3.8-27B-DFlash2"]


def test_a_drafter_that_is_not_in_the_cache_is_left_out_and_a_link_outside_the_hub_is_ignored(tmp_path):
    hub = _hub(tmp_path)
    stage = tmp_path / "run" / "stages" / "4" / "configs" / "1"
    stage.mkdir(parents=True)
    (stage / "run.sh").write_text('DFLASH_WEIGHTS=someone/Missing@x\n')
    (stage / "elsewhere").symlink_to(tmp_path)
    got = labclient.lab_hf_repos(stage.parent.parent, tmp_path / "hf", {})
    assert got == []


def test_the_helper_makes_a_directory_only_under_its_root(labroot, tmp_path):
    root, gz = labroot
    lab_ = connect(root, gz, tmp_path)
    try:
        lab_.ensure_dir(root / "cache" / "kernels")
        assert (root / "cache" / "kernels").is_dir()
        with pytest.raises(labclient.LabError):
            lab_.ensure_dir(tmp_path / "outside")
        assert not (tmp_path / "outside").exists()
    finally:
        lab_.close()


def test_a_failed_wait_for_the_childs_pid_still_closes_the_connection(labroot, tmp_path, monkeypatch):
    """Review of PR 447, round 2: when the wait for the child's pid timed out, the brain-going-away test left before its cleanup, with the lab
    connection still open, so the helper and the child it ran stayed up until the command's own deadline (600 s) and held the fake lease.
    Here the pid wait is made to fail at once, and the test checks that the helper has exited anyway and that nothing it ran is left."""
    real_wait = wait_until

    def failing_wait(condition, what, **kw):
        if "child's pid" in what:
            raise AssertionError("simulated: the wait for the child's pid timed out")
        return real_wait(condition, what, **kw)

    monkeypatch.setattr(sys.modules[__name__], "wait_until", failing_wait)
    before = len(LABS)
    pidfile = tmp_path / "child.pid"
    try:
        with pytest.raises(AssertionError, match="simulated"):
            test_the_brain_going_away_kills_running_tests_and_releases_leases(labroot, tmp_path)
        opened = LABS[before:]
        assert opened, "the scenario opened no connection"
        for lab_ in opened:
            real_wait(lambda: lab_.proc.poll() is not None, "the helper to exit after the pid wait failed", timeout=20)
        if pidfile.exists() and pidfile.read_text().strip().isdigit():
            real_wait(lambda: not pid_alive(int(pidfile.read_text())), "the test's child to be gone after the pid wait failed", timeout=20)
    finally:                                                # this test leaves nothing behind either, whatever it found
        for lab_ in LABS[before:]:
            if lab_.proc.poll() is None:
                lab_.proc.kill()
        if pidfile.exists() and pidfile.read_text().strip().isdigit() and pid_alive(int(pidfile.read_text())):
            os.kill(int(pidfile.read_text()), signal.SIGKILL)


def test_a_failed_pid_wait_kills_the_child_by_the_pid_in_the_file_even_when_closing_the_connection_does_not(labroot, tmp_path, monkeypatch):
    """The cleanup of the brain-going-away test has two parts: closing the connection (the helper then stops what it runs) and killing the child
    by the pid its shell wrote, which covers a helper that does not stop it. Here closing is made to do nothing and the pid wait fails after
    the pid was written, so only the second part can end the child."""
    real_wait = wait_until
    pidfile = tmp_path / "child.pid"

    def failing_wait(condition, what, **kw):
        value = real_wait(condition, what, **kw)
        if "child's pid" in what:
            raise AssertionError("simulated: the test failed right after its child wrote the pid")
        return value

    monkeypatch.setattr(sys.modules[__name__], "wait_until", failing_wait)
    monkeypatch.setattr(labclient.Lab, "close", lambda self: None)          # closing the connection ends nothing here
    before = len(LABS)
    child = None
    try:
        with pytest.raises(AssertionError, match="simulated"):
            test_the_brain_going_away_kills_running_tests_and_releases_leases(labroot, tmp_path)
        child = int(pidfile.read_text())
        real_wait(lambda: not pid_alive(child), "the child to be killed by the pid in the file", timeout=20)
    finally:                                                # the helpers this test left running are ended here
        for lab_ in LABS[before:]:
            if lab_.proc.poll() is None:
                lab_.proc.kill()
        if child is not None and pid_alive(child):
            os.kill(child, signal.SIGKILL)


def test_closing_every_lab_closes_the_helpers_and_forgets_them(labroot, tmp_path):
    lab_ = connect(labroot[0], labroot[1], tmp_path)
    assert lab_.proc.poll() is None and lab_ in LABS
    close_every_lab()
    assert lab_.proc.poll() is not None and LABS == []


def test_a_test_that_fails_after_opening_a_connection_has_it_closed_by_the_module_teardown(tmp_path):
    """The wiring, not the helper: a separate pytest run whose test opens a connection, writes the helper's pid and fails. The teardown fixture
    of this module must have closed the connection by the time that run is over."""
    inner = tmp_path / "test_inner.py"
    inner.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
        "from test_lab import _close_every_lab, labroot, connect  # noqa: F401  (the autouse fixture applies to this module too)\n"
        "def test_opens_and_fails(labroot, tmp_path):\n"
        "    lab_ = connect(labroot[0], labroot[1], tmp_path)\n"
        f"    open({str(tmp_path / 'helper.pid')!r}, 'w').write(str(lab_.proc.pid))\n"
        "    assert False, 'this test fails on purpose'\n")
    done = subprocess.run([sys.executable, "-m", "pytest", str(inner), "-q", "-p", "no:cacheprovider", f"--basetemp={tmp_path / 'inner-bt'}"],
                          capture_output=True, text=True, timeout=120, cwd=REPO)
    assert done.returncode == 1 and "this test fails on purpose" in done.stdout, done.stdout[-400:]
    helper = int((tmp_path / "helper.pid").read_text())
    try:
        wait_until(lambda: not pid_alive(helper), "the helper to be gone once the failed inner run was over", timeout=20)
    finally:
        if pid_alive(helper):
            os.kill(helper, signal.SIGKILL)
