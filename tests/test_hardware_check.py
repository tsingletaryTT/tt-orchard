"""Tests for the hardware-check driver, run against fake gozer roots.

Nothing here touches a real device. gozer runs from the tt-gozer branch checkout with a fake
sysfs, a fake /proc and a fake reset command. The child is a small Python script that writes
the files gozer reads from /proc. The docker and tt-smi binaries are stubs. The driver is called
in-process, so the driver's pid is the pytest pid, and that pid gets a directory in the fake
/proc so gozer sees the owner alive.
"""
import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from orchard import hardware_check as hc
from orchard.ledger import Ledger

REAL_GOZER = Path("/home/ttuser/code/tt-gozer-orchard/bin/gozer")
pytestmark = pytest.mark.skipif(
    not REAL_GOZER.exists(), reason=f"{REAL_GOZER} (the gozer branch with `reset`) is not present")

REPO = Path(__file__).resolve().parent.parent
BOARD = "0000:03:00.0"          # first chip of the second board in QUIETBOX
OTHER_BOARD = "0000:01:00.0"

QUIETBOX = [
    {"dev_index": 0, "bdf": "0000:01:00.0", "serial": "0000000000000002",
     "asic_id": "1111111111111111", "card": "p300c"},
    {"dev_index": 1, "bdf": "0000:02:00.0", "serial": "0000000000000002",
     "asic_id": "2222222222222222", "card": "p300c"},
    {"dev_index": 2, "bdf": "0000:03:00.0", "serial": "0000000000000001",
     "asic_id": "3333333333333333", "card": "p300c"},
    {"dev_index": 3, "bdf": "0000:04:00.0", "serial": "0000000000000001",
     "asic_id": "4444444444444444", "card": "p300c"},
]


def build_sysfs(root, chips):
    """A fake /sys/class/tenstorrent tree, copied from the gozer test conftest."""
    for c in chips:
        base = os.path.join(root, "class", "tenstorrent", f"tenstorrent!{c['dev_index']}")
        os.makedirs(base, exist_ok=True)
        for name, val in (("tt_serial", c["serial"]), ("tt_asic_id", c["asic_id"]),
                          ("tt_card_type", c["card"]), ("tt_heartbeat", "12345")):
            with open(os.path.join(base, name), "w") as f:
                f.write(val)
        pci = os.path.join(root, "bus", "pci", "devices", c["bdf"])
        os.makedirs(pci, exist_ok=True)
        link = os.path.join(base, "device")
        if not os.path.lexists(link):
            os.symlink(pci, link)
    return os.path.join(root, "class", "tenstorrent")


def make_proc_dir(proc_root, pid, ppid=1):
    """What gozer reads for a live process: status (PPid), comm, and an fd directory."""
    d = Path(proc_root) / str(pid)
    (d / "fd").mkdir(parents=True, exist_ok=True)
    (d / "comm").write_text("python\n")
    (d / "status").write_text(f"Name:\tpython\nPPid:\t{ppid}\n")


def script(path, body):
    path.write_text("#!/bin/bash\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


STUB_CHILD = '''
import os, shutil, signal, sys, time
dev = sys.argv[1]
opts = sys.argv[2:]
proc = os.environ["GOZER_PROC_ROOT"]
d = os.path.join(proc, str(os.getpid()))
ppid = os.getppid()
if "--fake-ppid" in opts:
    ppid = int(opts[opts.index("--fake-ppid") + 1])

def cleanup():
    shutil.rmtree(d, ignore_errors=True)

def on_term(*a):
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGTERM, on_term)
if "--ignore-term" in opts:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if "--no-open" in opts:
    time.sleep(60)
    sys.exit(0)
os.makedirs(os.path.join(d, "fd"))
open(os.path.join(d, "comm"), "w").write("python\\n")
open(os.path.join(d, "status"), "w").write("Name:\\tpython\\nPPid:\\t%d\\n" % ppid)
os.symlink("/dev/tenstorrent/" + dev, os.path.join(d, "fd", "3"))
if "--extra-dev" in opts:
    os.symlink("/dev/tenstorrent/" + opts[opts.index("--extra-dev") + 1], os.path.join(d, "fd", "4"))
print("ENV " + os.environ.get("TT_VISIBLE_DEVICES", ""), flush=True)
print("OPENED", flush=True)
if "--stall" in opts:
    # Ignore quit for a long time, so a test can signal the driver while it waits.
    time.sleep(60)
sys.stdin.readline()
cleanup()
print("CLOSED", flush=True)
'''


class Env:
    """Fake roots, stub binaries and helpers for one test."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp = tmp_path
        self.proc = tmp_path / "proc"
        self.proc.mkdir()
        self.marker = tmp_path / "resets"
        self.log = tmp_path / "gozer-calls.log"
        self.state = tmp_path / "state"
        monkeypatch.setenv("GOZER_ROOT", str(self.state))
        monkeypatch.setenv("GOZER_SYSFS_ROOT", build_sysfs(str(tmp_path / "sys"), QUIETBOX))
        monkeypatch.setenv("GOZER_PROC_ROOT", str(self.proc))
        monkeypatch.setenv("GOZER_HISTORY_ROOT", str(tmp_path / "history"))
        # The fake chip reset. It logs a line, then optionally sleeps or fails when the number
        # of logged lines equals the number in the file slow-at or fail-at.
        reset = script(tmp_path / "reset.sh", f'''
            echo "$@" >> {self.marker}
            n=$(wc -l < {self.marker})
            if [ -f {tmp_path}/slow-at ] && [ "$(cat {tmp_path}/slow-at)" = "$n" ]; then
                sleep 4
                echo done >> {tmp_path}/reset-finished
            fi
            if [ -f {tmp_path}/fail-at ] && [ "$(cat {tmp_path}/fail-at)" = "$n" ]; then exit 1; fi
            exit "${{FAKE_RESET_EXIT:-0}}"
        ''')
        monkeypatch.setenv("GOZER_RESET_CMD", str(reset))
        # A wrapper that logs every gozer call with its exit code and the number of resets
        # that had run when it returned.
        self.gozer = script(tmp_path / "gozer-wrap", f'''
            {sys.executable} {REAL_GOZER} "$@"
            rc=$?
            blk=$(awk '/SigBlk/ {{print $2}}' /proc/$$/status)
            sid=$(ps -o sid= -p $$ | tr -d ' ')
            echo "$*|rc=$rc|resets=$(cat {self.marker} 2>/dev/null | wc -l)|blk=$blk|sid=$sid" >> {self.log}
            exit $rc
        ''')
        self.tt_smi = script(tmp_path / "fake-tt-smi",
                             'echo \'{"device_info": [{"board": 1}, {"board": 2}]}\'\n')
        self.docker = script(tmp_path / "fake-docker", "exit 0\n")
        self.child_py = tmp_path / "child.py"
        self.child_py.write_text(STUB_CHILD)
        make_proc_dir(self.proc, os.getpid())     # the driver's own pid, so gozer sees the owner alive
        self.clock_now = 1000.0
        self.sleeps = []

    # fake clock and sleep for the driver's idle wait
    def clock(self):
        return self.clock_now

    def sleep(self, n):
        self.sleeps.append(n)
        self.clock_now += n

    def child_cmd(self, *extra):
        return json.dumps([sys.executable, str(self.child_py), "{dev}", *extra])

    def argv(self, out, *extra, child_extra=(), idle="1", board=BOARD, gozer=None):
        a = ["--gozer", str(gozer or self.gozer), "--board", board, "--out-dir", str(out),
             "--child-cmd", self.child_cmd(*child_extra), "--idle-seconds", idle,
             "--tt-smi", str(self.tt_smi), "--docker", str(self.docker),
             "--opened-timeout", "10", "--stop-timeout", "5", "--cmd-timeout", "60",
             "--allow-gozer-env"]
        return a + list(extra)

    def run(self, out, *extra, hook=None, **kw):
        return hc.main(self.argv(out, *extra, **kw), clock=self.clock, sleep=self.sleep, hook=hook)

    # direct gozer access, bypassing the wrapper so the call log stays the driver's
    def raw(self, *args):
        return subprocess.run([sys.executable, str(REAL_GOZER), *args], capture_output=True, text=True)

    def status(self):
        return json.loads(self.raw("status", "--json").stdout)

    def board_states(self, bdfs):
        return {c["bdf"]: c["state"] for c in self.status()["chips"] if c["bdf"] in bdfs}

    def resets(self):
        return len(self.marker.read_text().splitlines()) if self.marker.exists() else 0

    def calls(self):
        return [ln.split("|") for ln in self.log.read_text().splitlines()] if self.log.exists() else []

    def real_resets(self):
        """`gozer reset` calls, without the `reset --help` capability probe."""
        return [c for c in self.calls() if c[0].startswith("reset") and "--help" not in c[0]]

    def wrapper(self, rules=(), rewrite=None):
        """A gozer that answers some calls itself and passes the rest to the logging wrapper.

        rules: (verb, first_n, last_n, exit_code, stdout). Calls to `verb` numbered first_n
        to last_n (last_n None means for ever) get that answer. `reset --help` is not counted.
        rewrite: (verb, python statements) filters the real stdout of `verb` through python.
        """
        lines = ["#!/bin/bash", 'verb="$1"']
        for i, (verb, lo, hi, code, out) in enumerate(rules):
            f = self.tmp / f"count-{i}-{verb}"
            hi_test = "true" if hi is None else f"[ $c -le {hi} ]"
            lines += [
                f'if [ "$1" = {verb} ] && [ "$2" != --help ]; then',
                f'  c=$(( $(cat {f} 2>/dev/null || echo 0) + 1 )); echo $c > {f}',
                f"  if [ $c -ge {lo} ] && {hi_test}; then",
                f"    echo '{out}'", f"    exit {code}", "  fi", "fi"]
        if rewrite:
            verb, code = rewrite
            lines += [f'if [ "$1" = {verb} ]; then',
                      f'  out=$({self.gozer} "$@"); rc=$?',
                      f'  printf "%s" "$out" | {sys.executable} -c {shlex.quote(textwrap.dedent(code))}',
                      '  exit $rc', 'fi']
        lines.append(f'exec {self.gozer} "$@"')
        return script(self.tmp / f"gozer-rules-{len(list(self.tmp.glob('gozer-rules-*')))}",
                      "\n".join(lines[1:]) + "\n")

    def lease_other_board(self):
        """Another agent's lease on the board we do not use."""
        r = self.raw("acquire", "--exact", OTHER_BOARD, "--chips", "1", "--who", "other:agent",
                     "--owner-pid", str(os.getpid()), "--json")
        assert r.returncode == 0, r.stderr
        return json.loads(r.stdout)["lease_id"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    return Env(tmp_path, monkeypatch)


def ledger_entries(out):
    # Read the file directly. The driver has closed its ledger by now.
    with Ledger(Path(out) / "ledger.jsonl") as led:
        return led.read()


def notices(out):
    return [e["data"] for e in ledger_entries(out) if e["event"] == "notice"]


def check_ids(out):
    return [n["check"] for n in notices(out)]


def by_check(out, check):
    return [n for n in notices(out) if n["check"] == check]


# ---------------------------------------------------------------------------------------------

def test_happy_path_end_to_end(env, tmp_path, capsys):
    out = tmp_path / "out"
    code = env.run(out)
    assert code == 0, (out / "summary.md").read_text()
    entries = ledger_entries(out)
    assert entries[0]["event"] == "run_start"
    opts = entries[0]["data"]
    assert opts["options"]["board"] == BOARD and opts["gozer_version"]["exit_code"] == 0
    assert "gozer" in opts["gozer_version"]["stdout"]
    ids = check_ids(out)
    order = ["P", "A", "H2", "H3.refuse", "H3.stop", "H3.gone", "H3.reset", "H3.smi",
             "H4", "H4.stop", "H4.gone", "H4.reset", "H1", "H1.reconcile", "H6.release", "H6.free"]
    assert [i for i in ids if i in order] == order
    assert all(n["ok"] for n in notices(out))
    measured = [e["data"] for e in entries if e["event"] == "measurement"]
    assert all(m["label"] == "measured" for m in measured)
    assert {"reset_seconds", "reset2_seconds"} <= {m["name"] for m in measured}
    # the smi snapshot counted its devices
    assert by_check(out, "H3.smi")[0]["evidence"]["devices"] == 2
    assert (out / "summary.md").exists()
    assert "PASS" in capsys.readouterr().out
    # the lease is gone and the board is FREE
    assert set(env.board_states([BOARD, "0000:04:00.0"]).values()) == {"FREE"}
    # Three `gozer reset` calls: the refused probe, the reset after H3 and the reset after H4.
    # Two of them ran the chip reset, and the release ran it a third time.
    assert env.resets() == 3
    resets = env.real_resets()
    assert [c[1:3] for c in resets] == [["rc=15", "resets=0"], ["rc=0", "resets=1"],
                                        ["rc=0", "resets=2"]]
    assert [c for c in env.calls() if c[0].startswith("release")][0][2] == "resets=3"
    # the refusal probe used its own reset command, which never ran
    assert not (out / "refusal-probe-marker").exists()
    # the stub child removed its fake proc dir
    assert [p.name for p in env.proc.iterdir()] == [str(os.getpid())]
    # the child ran in its own session, with the board's first chip as its only visible device
    h2 = by_check(out, "H2")[0]["evidence"]
    assert h2["child_pgid"] == h2["child_pid"] != os.getpgid(0)
    assert any(str(h2["child_pid"]) in row for row in h2["ps_rows"])
    assert "ENV " + BOARD in by_check(out, "H4")[0]["evidence"]["output"]
    # the release in H6 was the only one: the cleanup had nothing left to release
    assert len([c for c in env.calls() if c[0].startswith("release")]) == 1
    assert not [n for n in notices(out) if n["check"].startswith("cleanup")]
    # the exact command is in the evidence
    acq = by_check(out, "A")[0]["evidence"]["acquire"]["command"]
    assert "--owner-pid" in acq and str(os.getpid()) in acq and "--no-queue" in acq


def test_preflight_refuses_a_board_that_is_leased(env, tmp_path):
    other = env.raw("acquire", "--exact", BOARD, "--chips", "1", "--who", "other:agent",
                    "--owner-pid", str(os.getpid()), "--json")
    assert other.returncode == 0
    out = tmp_path / "out"
    code = env.run(out)
    assert code == 2
    assert not [c for c in env.calls() if c[0].startswith(("acquire", "release"))]
    assert not env.real_resets()
    assert env.resets() == 0
    assert by_check(out, "P")[0]["ok"] is False
    assert "A" not in check_ids(out)
    # the other agent's lease is untouched
    assert env.board_states([BOARD]) == {BOARD: "CLAIMED"}


def test_reset_refused_while_the_child_holds_the_device(env, tmp_path):
    out = tmp_path / "out"
    assert env.run(out) == 0
    refuse = by_check(out, "H3.refuse")[0]
    assert refuse["ok"] and refuse["evidence"]["exit_code"] == 15
    resets = env.real_resets()
    # the first reset call returned with no reset command run; the second ran one
    assert resets[0][1] == "rc=15" and resets[0][2] == "resets=0"
    assert resets[1][1] == "rc=0" and resets[1][2] == "resets=1"
    # the refusal came before the child was stopped, in the ledger
    ids = check_ids(out)
    assert ids.index("H3.refuse") < ids.index("H3.stop") < ids.index("H3.reset")


def test_failed_chip_reset_exit_17_is_a_stop_condition(env, tmp_path, monkeypatch):
    # With FAKE_RESET_EXIT the chip reset command fails, which gozer reports as exit 17: a stop.
    monkeypatch.setenv("FAKE_RESET_EXIT", "1")
    out = tmp_path / "out"
    code = env.run(out)
    assert code == 1
    reasons = [n["evidence"]["reason"] for n in by_check(out, "ABORT")]
    assert any("17" in r for r in reasons)
    assert set(env.board_states([BOARD, "0000:04:00.0"]).values()) <= {"FREE", "CLAIMED"}


def test_acquire_failure_exits_3(env, tmp_path):
    def hook(name, **info):
        if name == "before_acquire":
            # another agent takes our board between preflight and acquire
            r = env.raw("acquire", "--exact", BOARD, "--chips", "1", "--who", "other:agent",
                        "--owner-pid", str(os.getpid()), "--json")
            assert r.returncode == 0
    out = tmp_path / "out"
    assert env.run(out, hook=hook) == 3
    assert by_check(out, "A")[0]["ok"] is False
    assert env.resets() == 0
    assert not [c for c in env.calls() if c[0].startswith("release")]


def test_child_never_opens_aborts_and_cleans_up(env, tmp_path):
    out = tmp_path / "out"
    code = env.run(out, "--opened-timeout", "1", child_extra=("--no-open",))
    assert code == 1
    reasons = [n["evidence"]["reason"] for n in by_check(out, "ABORT")]
    assert any("OPENED" in r for r in reasons)
    assert set(env.board_states([BOARD, "0000:04:00.0"]).values()) == {"FREE"}
    assert by_check(out, "cleanup.release")[0]["ok"]
    assert by_check(out, "cleanup.stop")           # the silent child was stopped
    assert not any("--force" in c[0] for c in env.calls())    # cleanup never forces a release
    assert "H3.refuse" not in check_ids(out)


def test_held_foreign_child_fails_h2(env, tmp_path):
    out = tmp_path / "out"
    code = env.run(out, child_extra=("--fake-ppid", "1"))
    assert code == 1
    h2 = by_check(out, "H2")[0]
    assert h2["ok"] is False and h2["evidence"]["chip_state"] == "HELD-FOREIGN"
    assert set(env.board_states([BOARD, "0000:04:00.0"]).values()) == {"FREE"}


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_signal_mid_run_stops_the_child_and_releases(env, tmp_path, sig):
    out = tmp_path / "out"
    # The driver runs in a subprocess so the signal does not hit pytest. The wrapper makes the
    # fake /proc entry for the driver's own pid before main() takes the lease.
    code = textwrap.dedent(f'''
        import os, sys
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        from test_hardware_check import make_proc_dir
        make_proc_dir(os.environ["GOZER_PROC_ROOT"], os.getpid())
        from orchard.hardware_check import main
        sys.exit(main(sys.argv[1:]))
    ''')
    argv = env.argv(out, "--stop-timeout", "4", child_extra=("--stall",), idle="1")
    p = subprocess.Popen([sys.executable, "-c", code, *argv], cwd=REPO,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # Wait until the driver has seen the refusal, i.e. it is waiting on the stalled child.
        deadline = time.time() + 40
        while time.time() < deadline:
            lp = out / "ledger.jsonl"
            if lp.exists() and "H3.refuse" in lp.read_text():
                break
            time.sleep(0.2)
        else:
            pytest.fail("driver never reached H3: " + p.stderr.read())
        time.sleep(0.5)
        p.send_signal(sig)
        stdout, stderr = p.communicate(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == 1, stdout + stderr
    reasons = [n["evidence"]["reason"] for n in by_check(out, "ABORT")]
    assert any(sig.name in r for r in reasons)
    assert set(env.board_states([BOARD, "0000:04:00.0"]).values()) == {"FREE"}
    assert by_check(out, "cleanup.release")[0]["ok"]
    # the child had to be signalled, so the board may need a reset and the notice says so
    stop = by_check(out, "cleanup.stop")[0]
    assert stop["ok"] is False and "board needs a reset" in stop["evidence"]["summary"]
    leftovers = [d.name for d in env.proc.iterdir() if d.name != str(p.pid)]
    assert str(p.pid) not in leftovers
    assert [d for d in leftovers if d != str(os.getpid())] == []     # no stub child dir left


def test_lease_vanishing_is_a_stop_condition(env, tmp_path):
    def hook(name, driver=None, **info):
        if name == "after_h2_open":
            # Someone removes our lease record while the driver is alive.
            r = env.raw("release", driver.lease_id, "--no-reset", "--force", "--json")
            assert r.returncode == 0, r.stdout + r.stderr
    out = tmp_path / "out"
    code = env.run(out, hook=hook)
    assert code == 1
    reasons = [n["evidence"]["reason"] for n in by_check(out, "ABORT")]
    assert any("disappeared" in r for r in reasons)
    # no second release was attempted: the cleanup saw the lease gone and recorded no failure
    assert all(n["ok"] for n in notices(out) if n["check"].startswith("cleanup"))
    # the child was stopped
    assert [d.name for d in env.proc.iterdir()] == [str(os.getpid())]


def test_unexpected_lease_on_our_board_is_a_stop_condition(env, tmp_path):
    def hook(name, driver=None, **info):
        if name == "before_idle_status":
            # Replace our lease with another agent's on the same board. The child is gone by
            # now, so the board is free to take.
            env.raw("release", driver.lease_id, "--no-reset", "--force", "--json")
            r = env.raw("acquire", "--exact", BOARD, "--chips", "1", "--who", "intruder",
                        "--owner-pid", str(os.getpid()), "--json")
            assert r.returncode == 0, r.stdout
    out = tmp_path / "out"
    assert env.run(out, hook=hook) == 1
    reasons = [n["evidence"]["reason"] for n in by_check(out, "ABORT")]
    assert any("unexpected lease" in r for r in reasons)


def test_idle_check_waits_until_the_lease_is_old_enough(env, tmp_path):
    seen = {}

    def hook(name, driver=None, **info):
        if name == "after_acquire":
            seen["acquired"] = env.clock()
        if name == "before_idle_status":
            seen["checked"] = env.clock()
    out = tmp_path / "out"
    assert env.run(out, hook=hook, idle="3") == 0
    assert seen["checked"] - seen["acquired"] >= 3
    assert sum(env.sleeps) >= 3
    h1 = by_check(out, "H1")[0]
    assert h1["evidence"]["lease_age_seconds"] >= 3


def test_reconcile_runs_when_no_other_lease_is_on_the_box(env, tmp_path):
    out = tmp_path / "out"
    assert env.run(out) == 0
    assert any(c[0].startswith("reconcile") for c in env.calls())
    assert "H1.reconcile" in check_ids(out)


def test_reconcile_skipped_when_another_lease_is_present(env, tmp_path):
    env.lease_other_board()
    out = tmp_path / "out"
    assert env.run(out) == 0, (out / "summary.md").read_text()
    assert not any(c[0].startswith("reconcile") for c in env.calls())
    assert "H1.reconcile" not in check_ids(out)
    skip = [e["data"] for e in ledger_entries(out)
            if e["event"] == "decision" and "reconcile skipped" in e["data"]["decision"]]
    assert skip and skip[0]["leases"][0]["who"] == "other:agent"
    # the other lease was not touched
    assert env.board_states([OTHER_BOARD]) == {OTHER_BOARD: "CLAIMED"}


def test_default_child_argv(tmp_path):
    argv = hc.default_child_argv(BOARD, cache_dir="/x/cache", logs_dir="/x/logs")
    assert argv[:2] == ["bash", "-c"]
    script_text = argv[2]
    assert BOARD in script_text
    assert f"source {hc.DEFAULT_ENV_SCRIPT}" in script_text
    assert "exec " in script_text
    assert f"exec {hc.DEFAULT_PYTHON} -c" in script_text
    assert "open_mesh_device" in script_text and "OPENED" in script_text
    # the board is exported after the env script, which sets its own value
    assert script_text.index("source") < script_text.index(f"TT_VISIBLE_DEVICES={BOARD}")
    # the JIT cache and logs are per run, set after the env script so two drivers never share them
    assert script_text.index("source") < script_text.index("TT_METAL_CACHE=/x/cache")
    assert script_text.index("source") < script_text.index("TT_METAL_LOGS_PATH=/x/logs")
    # with no --child-cmd the driver builds exactly that argv, with its cache under the out dir
    args = hc.parse_args(["--board", BOARD])
    d = hc.Driver(args, ledger=None, clock=None, sleep=None, hook=None, out_dir=tmp_path / "o")
    d.dev_index = 2
    built = d.child_argv()
    assert built[:2] == ["bash", "-c"]
    assert f"TT_METAL_CACHE={tmp_path / 'o' / 'cache'}" in built[2]
    assert f"TT_METAL_LOGS_PATH={tmp_path / 'o' / 'logs'}" in built[2]
    assert "exec " in built[2] and BOARD in built[2]


def test_every_gozer_call_is_a_known_verb(env, tmp_path):
    out = tmp_path / "out"
    assert env.run(out) == 0
    allowed = {"status", "queue", "acquire", "reset", "release", "reconcile", "--version"}
    verbs = {c[0].split()[0] for c in env.calls()}
    assert verbs <= allowed
    # and the driver only used the commands it was given besides ps and pgrep
    for n in notices(out):
        cmd = n["evidence"].get("command")
        if cmd:
            assert cmd[0] in {str(env.gozer), str(env.docker), str(env.tt_smi), "ps", "pgrep"}


def test_reset_is_never_forced(env, tmp_path):
    out = tmp_path / "out"
    assert env.run(out) == 0
    for c in env.calls():
        assert "--force" not in c[0]


def test_default_out_dir_has_timestamp_and_board_and_never_collides(tmp_path):
    now = time.gmtime(1_790_000_000)
    a = hc.make_out_dir(BOARD, tmp_path / "runs", now)
    b = hc.make_out_dir(OTHER_BOARD, tmp_path / "runs", now)     # same second, other board
    c = hc.make_out_dir(BOARD, tmp_path / "runs", now)           # same second, same board
    assert len({a, b, c}) == 3
    assert "0000-03-00.0" in a.name and "0000-01-00.0" in b.name
    assert a.name.startswith(time.strftime("%Y%m%dT%H%M%SZ", now))
    assert all(p.is_dir() for p in (a, b, c))


def test_main_without_out_dir_uses_the_default_location(env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    argv = env.argv(tmp_path / "unused")
    i = argv.index("--out-dir")
    del argv[i:i + 2]
    assert hc.main(argv, clock=env.clock, sleep=env.sleep) == 0
    dirs = list((tmp_path / "runs" / "hardware-check").iterdir())
    assert len(dirs) == 1 and "0000-03-00.0" in dirs[0].name
    assert (dirs[0] / "ledger.jsonl").exists() and (dirs[0] / "summary.md").exists()


@pytest.mark.parametrize("ours, theirs", [(BOARD, OTHER_BOARD), (OTHER_BOARD, BOARD)])
def test_another_drivers_lease_on_the_other_board_is_not_unexpected(env, tmp_path, ours, theirs):
    # A second driver holds a lease on the other board under the same `who`, with its own owner
    # pid (its own process in a real run). It is not ours, but it is not on our board either.
    make_proc_dir(env.proc, 5_000_100)
    r = env.raw("acquire", "--exact", theirs, "--chips", "1", "--who", hc.WHO,
                "--owner-pid", "5000100", "--json")
    assert r.returncode == 0, r.stdout
    theirs_chips = [c["bdf"] for c in env.status()["chips"] if c.get("who") == hc.WHO]
    out = tmp_path / "out"
    code = env.run(out, board=ours)
    assert code == 0, (out / "summary.md").read_text()
    # the other driver's lease is untouched, and our board is free again
    after = env.status()["chips"]
    assert [c["bdf"] for c in after if c.get("who") == hc.WHO] == theirs_chips
    ours_chips = [c for c in after if c["bdf"] not in theirs_chips]
    assert {c["state"] for c in ours_chips} == {"FREE"}
    # reconcile was skipped, and the reason names the other lease's `who`
    assert not any(c[0].startswith("reconcile") for c in env.calls())
    skip = [e["data"]["decision"] for e in ledger_entries(out)
            if e["event"] == "decision" and "reconcile skipped" in e["data"]["decision"]]
    assert skip and hc.WHO in skip[0]
    assert not [n for n in notices(out) if not n["ok"]]


def test_child_that_ignores_quit_and_sigterm_is_killed(env, tmp_path):
    out = tmp_path / "out"
    def hook(name, driver=None, how=None, **info):
        # A real kernel closes a killed process's descriptors. The fake /proc cannot, so the
        # test removes the dead child's directory by hand.
        if name == "after_stop" and how == "SIGKILL":
            shutil.rmtree(env.proc / str(driver.child.pid), ignore_errors=True)
    code = env.run(out, "--stop-timeout", "1", hook=hook,
                   child_extra=("--stall", "--ignore-term"))
    assert code == 1                       # a forced stop is reported as a failed check
    stop = by_check(out, "H3.stop")[0]
    assert stop["ok"] is False and stop["evidence"]["how"] == "SIGKILL"
    decisions = [e["data"]["decision"] for e in ledger_entries(out) if e["event"] == "decision"]
    assert any("SIGTERM" in d for d in decisions) and any("SIGKILL" in d for d in decisions)
    assert by_check(out, "H3.gone")[0]["ok"]          # and it is confirmed gone
    assert set(env.board_states([BOARD, "0000:04:00.0"]).values()) == {"FREE"}


def test_stub_that_leaves_a_process_in_the_group_is_not_gone(env, tmp_path):
    # A grandchild that outlives the child keeps the process group alive.
    cmd = json.dumps(["bash", "-c",
                      f"sleep 30 & {sys.executable} {env.child_py} {{dev}}"])
    out = tmp_path / "out"
    code = hc.main(env.argv(out) + ["--child-cmd", cmd], clock=env.clock, sleep=env.sleep)
    assert code == 1
    gone = by_check(out, "H3.gone")[0]
    assert gone["ok"] is False and gone["evidence"]["pgrep"]["exit_code"] == 0
    # the cleanup killed what was left in the group, so nothing of ours is left running
    pgid = by_check(out, "H2")[0]["evidence"]["child_pgid"]
    assert subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True).returncode == 1


def add_foreign_holder(env, pid=5_000_300, dev=2):
    """A process outside the driver's tree with a device of our board open."""
    make_proc_dir(env.proc, pid, ppid=1)
    os.symlink(f"/dev/tenstorrent/{dev}", env.proc / str(pid) / "fd" / "3")


def test_idle_check_fails_when_a_foreign_process_holds_the_device(env, tmp_path):
    def hook(name, **info):
        if name == "before_idle_status":
            add_foreign_holder(env)
    out = tmp_path / "out"
    assert env.run(out, hook=hook) == 1
    h1 = by_check(out, "H1")[0]
    assert h1["ok"] is False and "HELD-FOREIGN" in h1["evidence"]["summary"]


def test_child_gone_check_fails_when_a_foreign_holder_remains(env, tmp_path):
    def hook(name, check=None, **info):
        if name == "after_stop" and check == "H3.stop":
            add_foreign_holder(env)
    out = tmp_path / "out"
    assert env.run(out, hook=hook) == 1
    gone = by_check(out, "H3.gone")[0]
    assert gone["ok"] is False and gone["evidence"]["chip_state"] == "HELD-FOREIGN"


def test_confirm_gone_needs_ps_to_show_no_row(env, tmp_path):
    # A live process whose group id differs from the recorded group: pgrep finds nothing, the chip is
    # CLAIMED, and only `ps -p` can show that the child is still there.
    sleeper = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        args = hc.parse_args(["--board", BOARD, "--gozer", str(env.gozer)])
        with Ledger(tmp_path / "ledger.jsonl") as led:
            d = hc.Driver(args, led, None, None, None)
            d.pid = os.getpid()
            d.grant = {"chips": [BOARD]}
            d.child = type("C", (), {"pid": sleeper.pid, "pgid": 4_000_000})()
            claimed = {"bdf": BOARD, "state": "CLAIMED", "who": hc.WHO, "pid": os.getpid()}
            d.wait_state = lambda want, **kw: ({}, claimed, [claimed])
            assert d.confirm_gone("X") is False
            ev = led.read()[-1]["data"]["evidence"]
            assert ev["ps"]["stdout"].strip() == str(sleeper.pid)
            assert ev["pgrep"]["exit_code"] == 1
    finally:
        sleeper.kill()
        sleeper.wait()


def test_child_that_exits_before_opening_is_reported_as_such(env, tmp_path):
    out = tmp_path / "out"
    code = hc.main(env.argv(out) + ["--child-cmd", json.dumps(["true"])],
                   clock=env.clock, sleep=env.sleep)
    assert code == 1
    reasons = [n["evidence"]["reason"] for n in by_check(out, "ABORT")]
    assert any("exited before" in r for r in reasons)
    assert set(env.board_states([BOARD, "0000:04:00.0"]).values()) == {"FREE"}


def test_a_command_that_times_out_is_a_stop_condition(env, tmp_path):
    slow = script(tmp_path / "slow-docker", "exec sleep 20\n")
    out = tmp_path / "out"
    t0 = time.time()
    code = hc.main(env.argv(out) + ["--docker", str(slow), "--cmd-timeout", "1"],
                   clock=env.clock, sleep=env.sleep)
    assert code == 1 and time.time() - t0 < 15
    reasons = [n["evidence"]["reason"] for n in by_check(out, "ABORT")]
    assert any("timed out" in r for r in reasons)
    assert "A" not in check_ids(out)          # it stopped in preflight, before any lease


def test_acquire_failing_its_claimed_check_exits_3_and_releases(env, tmp_path):
    def hook(name, **info):
        if name == "after_acquire":
            add_foreign_holder(env)
    out = tmp_path / "out"
    assert env.run(out, hook=hook) == 3
    assert by_check(out, "A")[0]["ok"] is False
    assert by_check(out, "cleanup.release")      # the lease that was granted is released


def test_signals_are_blocked_while_acquire_runs_and_not_afterwards(env, tmp_path):
    out = tmp_path / "out"
    assert env.run(out) == 0
    blk = {c[0].split()[0]: int(c[3].split("=")[1], 16) for c in env.calls()}
    both = ((1 << (signal.SIGINT - 1)) | (1 << (signal.SIGTERM - 1))
            | (1 << (signal.SIGHUP - 1)))
    # The harness may start pytest with a signal already blocked, so compare with a call made
    # before the acquire (`--version`) and not with zero.
    assert blk["acquire"] & both == both
    assert blk["release"] == blk["--version"]


# ---------------------------------------------------------------------------------------------
# Fix round 1: safety review findings.

OWNER_BOOT = textwrap.dedent('''
    import os, sys
    sys.path.insert(0, {tests!r})
    from test_hardware_check import make_proc_dir
    make_proc_dir(os.environ["GOZER_PROC_ROOT"], os.getpid())
    from orchard.hardware_check import main
    sys.exit(main(sys.argv[1:]))
''')


def spawn_driver(env, argv):
    """The driver in a subprocess, with its own pid registered in the fake /proc."""
    code = OWNER_BOOT.format(tests=str(Path(__file__).parent))
    return subprocess.Popen([sys.executable, "-c", code, *argv], cwd=REPO,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def wait_for(pred, seconds=60):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.1)
    return False


def board_free(env):
    return set(env.board_states([BOARD, "0000:04:00.0"]).values()) == {"FREE"}


def test_failed_gone_check_stops_before_any_reset(env, tmp_path):
    # A grandchild stays in the child's group, so "child is gone" fails. No chip reset may run.
    cmd = json.dumps(["bash", "-c", f"sleep 31 & {sys.executable} {env.child_py} {{dev}}"])
    out = tmp_path / "out"
    code = hc.main(env.argv(out) + ["--child-cmd", cmd], clock=env.clock, sleep=env.sleep)
    assert code == 1
    assert by_check(out, "H3.gone")[0]["ok"] is False
    assert len(env.real_resets()) == 1          # only the refused probe
    assert "H3.reset" not in check_ids(out) and "H4" not in check_ids(out)
    assert by_check(out, "ABORT")
    assert board_free(env)
    pgid = by_check(out, "H2")[0]["evidence"]["child_pgid"]
    assert subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True).returncode == 1


def test_unrefused_probe_is_a_stop_and_never_runs_the_real_reset(env, tmp_path):
    # The child holds a device index that is on no board, so gozer has no reason to refuse the
    # probe. The probe runs with its own reset command, so the real one must never run.
    cmd = json.dumps([sys.executable, str(env.child_py), "9"])
    out = tmp_path / "out"
    code = hc.main(env.argv(out) + ["--child-cmd", cmd], clock=env.clock, sleep=env.sleep)
    assert code == 1
    # The probe call returned with the fake chip reset never run. The one marker line comes
    # later, from the cleanup release.
    assert env.real_resets()[0][2] == "resets=0"
    assert env.resets() == 1
    assert (out / "refusal-probe-marker").exists()    # the probe's own stand-in ran instead
    refuse = by_check(out, "H3.refuse")[0]
    assert refuse["ok"] is False
    reasons = [n["evidence"]["reason"] for n in by_check(out, "ABORT")]
    assert any("refusal failed" in r for r in reasons)
    assert "H4" not in check_ids(out)
    assert board_free(env)


def test_probe_needs_still_open_text_with_exit_15(env, tmp_path):
    # A 15 for another reason is not a refusal because a device is open.
    w = env.wrapper([("reset", 1, 1, 15,
                      '{"reset": false, "status": "refused", "message": "no longer locked"}')])
    out = tmp_path / "out"
    code = hc.main(env.argv(out, gozer=w), clock=env.clock, sleep=env.sleep)
    assert code == 1
    reasons = [n["evidence"]["reason"] for n in by_check(out, "ABORT")]
    assert any("refusal failed" in r for r in reasons)
    assert "H4" not in check_ids(out) and board_free(env)


@pytest.mark.parametrize("code_", [13, 15])
def test_a_reset_exit_other_than_0_after_the_child_is_gone_stops_the_run(env, tmp_path, code_):
    # Reset call 1 is the probe (real). Call 2 is the reset after H3.
    w = env.wrapper([("reset", 2, 2, code_,
                      '{"reset": false, "status": "refused", "message": "refusing to reset: x"}')])
    out = tmp_path / "out"
    code = hc.main(env.argv(out, gozer=w), clock=env.clock, sleep=env.sleep)
    assert code == 1
    assert "H4" not in check_ids(out)           # the device was never opened again
    assert by_check(out, "ABORT") and board_free(env)
    assert [p.name for p in env.proc.iterdir()] == [str(os.getpid())]


def test_the_lease_is_checked_before_every_device_open(env, tmp_path):
    def hook(name, check=None, **info):
        if name == "before_start_child" and check == "H4":
            add_foreign_holder(env)
    out = tmp_path / "out"
    assert env.run(out, hook=hook) == 1
    assert "H4" not in check_ids(out)
    reasons = [n["evidence"]["reason"] for n in by_check(out, "ABORT")]
    assert any("not ready" in r for r in reasons)


def test_every_granted_chip_must_be_claimed_after_the_stop(env, tmp_path):
    def hook(name, check=None, **info):
        if name == "after_stop" and check == "H3.stop":
            add_foreign_holder(env, pid=5_000_301, dev=3)    # the board's second chip
    out = tmp_path / "out"
    assert env.run(out, hook=hook) == 1
    assert by_check(out, "H3.gone")[0]["ok"] is False
    assert len(env.real_resets()) == 1 and "H3.reset" not in check_ids(out)


@pytest.mark.parametrize("edit", [
    "d['chips'] = d['chips'] + ['0000:01:00.0']",
    "d['owner_pid'] = 1",
    "d['neighbours'] = ['0000:01:00.0']",
    "d['units'] = d['units'] + ['extra']",
    "d['dev_indices'] = [2]",
    "d['dev_indices'] = [2, 9]",
    "d['chips'] = d['chips'][:1]",
])
def test_a_grant_that_is_not_exactly_our_board_aborts_and_releases(env, tmp_path, edit):
    w = env.wrapper(rewrite=("acquire", f"""
        import json, sys
        d = json.load(sys.stdin)
        {edit}
        print(json.dumps(d))
    """))
    out = tmp_path / "out"
    code = hc.main(env.argv(out, gozer=w), clock=env.clock, sleep=env.sleep)
    assert code == 3
    assert "H2" not in check_ids(out)
    assert board_free(env)                       # the real lease was released
    assert not env.proc.joinpath("never").exists()
    assert [p.name for p in env.proc.iterdir()] == [str(os.getpid())]


def test_child_touching_the_other_board_is_a_stop_condition(env, tmp_path):
    out = tmp_path / "out"
    code = env.run(out, child_extra=("--extra-dev", "0"))
    assert code == 1
    reasons = [n["evidence"]["reason"] for n in by_check(out, "ABORT")]
    assert any("other board" in r for r in reasons)
    assert board_free(env)


def test_preflight_refuses_a_gozer_without_reset(env, tmp_path):
    w = env.wrapper(rules=[("reset", 1, None, 2, "usage: gozer: invalid choice: 'reset'")])
    # a --help call is not counted by the rules, so make this one fail on its own
    w = script(tmp_path / "gozer-noreset", f'''
        if [ "$1" = reset ]; then echo "invalid choice: reset" >&2; exit 2; fi
        exec {env.gozer} "$@"
    ''')
    out = tmp_path / "out"
    code = hc.main(env.argv(out, gozer=w), clock=env.clock, sleep=env.sleep)
    assert code == 2
    assert not [c for c in env.calls() if c[0].startswith("acquire")]
    assert by_check(out, "P.gozer")[0]["ok"] is False


def test_default_gozer_is_the_branch_binary_and_tt_smi_is_off():
    args = hc.parse_args(["--board", BOARD])
    assert args.gozer == "/home/ttuser/code/tt-gozer-orchard/bin/gozer"
    assert args.tt_smi == "none"


def test_tt_smi_is_not_run_unless_asked(env, tmp_path):
    argv = env.argv(tmp_path / "out")
    i = argv.index("--tt-smi")
    del argv[i:i + 2]
    assert hc.main(argv, clock=env.clock, sleep=env.sleep) == 0
    assert "H3.smi" not in check_ids(tmp_path / "out")


def test_h6_fails_when_the_reset_inside_release_failed(env, tmp_path):
    (tmp_path / "fail-at").write_text("3")       # the third chip reset is the release's
    out = tmp_path / "out"
    code = env.run(out)
    assert code == 1
    rel = by_check(out, "H6.release")[0]
    assert rel["ok"] is False and "NOT marked clean" in rel["evidence"]["stdout"]
    assert rel["evidence"]["exit_code"] == 0     # gozer itself says 0


def test_h6_fails_on_the_message_even_if_history_looks_fine(env, tmp_path):
    w = env.wrapper(rewrite=("release", """
        import json, sys
        d = json.load(sys.stdin)
        d["message"] += "; reset failed -- unit released but NOT marked clean"
        print(json.dumps(d))
    """))
    out = tmp_path / "out"
    assert hc.main(env.argv(out, gozer=w), clock=env.clock, sleep=env.sleep) == 1
    assert by_check(out, "H6.release")[0]["ok"] is False


def test_h6_checks_for_stray_files_under_gozer_root(env, tmp_path):
    def hook(name, driver=None, **info):
        if name == "after_release":
            (env.state / "leases" / f"{driver.lease_id}.json").write_text("{}")
    out = tmp_path / "out"
    assert env.run(out, hook=hook) == 1
    files = by_check(out, "H6.files")[0]
    assert files["ok"] is False


def test_release_reset_confirmation_reads_the_history_log(tmp_path):
    hist = tmp_path / "history.jsonl"
    rows = [{"event": "released", "lease_id": "a", "reset_ran": True, "reset_ok": True},
            {"event": "released", "lease_id": "b", "reset_ran": True, "reset_ok": False},
            {"event": "released", "lease_id": "c", "reset_ran": False, "reset_ok": None}]
    hist.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    assert hc.release_reset_confirmed(str(tmp_path), "a") is True
    assert hc.release_reset_confirmed(str(tmp_path), "b") is False
    assert hc.release_reset_confirmed(str(tmp_path), "c") is False
    assert hc.release_reset_confirmed(str(tmp_path), "zzz") is None
    assert hc.release_reset_confirmed(str(tmp_path / "missing"), "a") is None


def test_summary_lists_what_was_and_was_not_checked(env, tmp_path):
    out = tmp_path / "out"
    assert env.run(out) == 0
    text = (out / "summary.md").read_text()
    assert "H5 (container server) is not part of this driver" in text
    assert str(env.gozer) in text and "gozer " in text
    assert "Reconcile: ran" in text
    assert "docker ps at start" in text and "docker ps at end" in text
    assert "gozer status at start" in text and "gozer status at end" in text
    assert "detached" in text
    assert "SIGKILL" in text


def test_summary_names_the_checks_skipped_by_a_stop(env, tmp_path):
    out = tmp_path / "out"
    assert env.run(out, "--opened-timeout", "1", child_extra=("--no-open",)) == 1
    text = (out / "summary.md").read_text()
    assert "Not run because of the stop:" in text
    line = [ln for ln in text.splitlines() if ln.startswith("Not run because of the stop:")][0]
    assert "H3.refuse" in line and "H6.release" in line and "H2" not in line


def test_summary_says_why_reconcile_was_skipped(env, tmp_path):
    env.lease_other_board()
    out = tmp_path / "out"
    assert env.run(out) == 0
    assert "Reconcile: skipped" in (out / "summary.md").read_text()


def test_signal_during_release_lets_gozer_finish(env, tmp_path):
    (tmp_path / "slow-at").write_text("3")      # the release's chip reset takes 4 s
    out = tmp_path / "out"
    p = spawn_driver(env, env.argv(out, "--reset-timeout", "60"))
    assert wait_for(lambda: env.resets() >= 3), "release never started"
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=60)
    assert p.returncode == 1
    assert env.resets() == 3                     # no second reset from cleanup
    assert (tmp_path / "reset-finished").exists()    # the slow reset ran to its end
    releases = [c for c in env.calls() if c[0].startswith("release")]
    assert len(releases) == 1 and releases[0][1] == "rc=0"
    assert board_free(env)
    assert not by_check(out, "cleanup.release")


def test_second_signal_during_cleanup_does_not_start_another_release(env, tmp_path):
    (tmp_path / "slow-at").write_text("1")      # the cleanup release is the first chip reset
    out = tmp_path / "out"
    p = spawn_driver(env, env.argv(out, "--reset-timeout", "60", "--stop-timeout", "2",
                                   child_extra=("--stall",)))
    assert wait_for(lambda: "H3.refuse" in (out / "ledger.jsonl").read_text()
                    if (out / "ledger.jsonl").exists() else False)
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    time.sleep(0.8)
    p.send_signal(signal.SIGHUP)                 # during cleanup's wait for the child to stop
    assert wait_for(lambda: env.resets() >= 1), "cleanup release never started"
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)                # during the cleanup release
    p.communicate(timeout=60)
    assert p.returncode == 1
    stop = by_check(out, "cleanup.stop")[0]
    assert stop["evidence"]["how"] == "SIGTERM"  # the stop ran through to its end
    assert env.resets() == 1
    assert (tmp_path / "reset-finished").exists()
    assert len([c for c in env.calls() if c[0].startswith("release")]) == 1
    assert board_free(env)


def test_a_reset_that_outlives_its_timeout_is_not_killed_or_repeated(env, tmp_path):
    (tmp_path / "slow-at").write_text("1")      # the reset after H3 takes 4 s
    out = tmp_path / "out"
    code = hc.main(env.argv(out, "--reset-timeout", "1"), clock=env.clock, sleep=env.sleep)
    assert code == 1
    assert wait_for(lambda: (tmp_path / "reset-finished").exists(), 20)   # gozer was left to finish
    assert env.resets() == 1                     # no second reset, no release
    assert not [c for c in env.calls() if c[0].startswith("release")]
    assert by_check(out, "ABORT")


def test_cleanup_retries_a_release_refused_with_15_and_never_forces(env, tmp_path):
    refused = '{"released": false, "message": "chips [2] still open by [1]"}'
    w = env.wrapper([("release", 1, 2, 15, refused)])
    out = tmp_path / "out"
    code = hc.main(env.argv(out, "--opened-timeout", "1", gozer=w, child_extra=("--no-open",)),
                   clock=env.clock, sleep=env.sleep)
    assert code == 1
    assert by_check(out, "cleanup.release")[-1]["ok"] is True
    assert len([c for c in env.calls() if c[0].startswith("release")]) == 1   # only the third reached gozer
    assert len(env.sleeps) >= 2                  # it waited between attempts, with the injected sleep
    assert not any("--force" in c[0] for c in env.calls())
    assert board_free(env)


def test_cleanup_that_cannot_release_prints_recovery_steps(env, tmp_path, capsys):
    refused = '{"released": false, "message": "chips [2] still open by [1]"}'
    w = env.wrapper([("release", 1, None, 15, refused)])
    out = tmp_path / "out"
    code = hc.main(env.argv(out, "--opened-timeout", "1", gozer=w, child_extra=("--no-open",)),
                   clock=env.clock, sleep=env.sleep)
    assert code == 1
    lease = by_check(out, "A")[0]["evidence"]["grant"]["lease_id"]
    err = capsys.readouterr().err
    summary = (out / "summary.md").read_text()
    for text in (err, summary):
        assert lease in text
        assert f"gozer release {lease}" in text
        assert "never --force" in text and "tt-smi -r" in text
    assert by_check(out, "cleanup.release")[-1]["ok"] is False


def test_a_failing_ledger_write_in_cleanup_cannot_skip_the_release(env, tmp_path, monkeypatch):
    real = hc.Driver.notice

    def flaky(self, check, *a, **k):
        if check == "cleanup.stop":
            raise OSError("disk full")
        return real(self, check, *a, **k)
    monkeypatch.setattr(hc.Driver, "notice", flaky)
    out = tmp_path / "out"
    code = env.run(out, "--opened-timeout", "1", child_extra=("--no-open",))
    assert code == 1
    assert board_free(env)


def test_cleanup_kills_a_group_left_behind_by_an_exited_child(env, tmp_path):
    # The stub exits at once but leaves a grandchild in its group. Cleanup must kill the group.
    cmd = json.dumps(["bash", "-c", "sleep 33 & echo OPENED; exit 0"])
    out = tmp_path / "out"
    code = hc.main(env.argv(out) + ["--child-cmd", cmd], clock=env.clock, sleep=env.sleep)
    assert code == 1
    notes = by_check(out, "cleanup.group")
    assert notes and notes[0]["ok"] is False
    pgid = notes[0]["evidence"]["pgid"]
    assert wait_for(lambda: subprocess.run(["pgrep", "-g", str(pgid)],
                                           capture_output=True).returncode == 1, 5)


def test_acquire_with_unreadable_json_says_a_lease_may_exist(env, tmp_path, capsys):
    w = env.wrapper(rewrite=("acquire", 'import sys; sys.stdin.read(); print("garbage")'))
    out = tmp_path / "out"
    assert hc.main(env.argv(out, gozer=w), clock=env.clock, sleep=env.sleep) == 3
    err = capsys.readouterr().err
    assert "lease owned by the driver pid" in err and "reaped" in err
    assert by_check(out, "A.malformed")


def test_module_documents_what_a_sigkill_leaves_behind():
    doc = hc.__doc__
    assert "SIGKILL" in doc and "STALE" in doc and "HELD-FOREIGN" in doc


def test_every_gozer_call_runs_in_its_own_session(env, tmp_path):
    out = tmp_path / "out"
    assert env.run(out) == 0
    sids = {int(c[4].split("=")[1]) for c in env.calls()}
    assert os.getsid(0) not in sids


def test_reset_exit_0_with_a_status_other_than_reset_stops_the_run(env, tmp_path):
    w = env.wrapper([("reset", 2, 2, 0, '{"reset": true, "status": "weird", "message": "x"}')])
    out = tmp_path / "out"
    assert hc.main(env.argv(out, gozer=w), clock=env.clock, sleep=env.sleep) == 1
    assert "H4" not in check_ids(out)


def test_failed_gone_check_after_h4_stops_before_the_second_reset(env, tmp_path):
    flag = tmp_path / "starts"
    cmd = json.dumps(["bash", "-c", f"""
        n=$(cat {flag} 2>/dev/null || echo 0); echo $((n+1)) > {flag}
        if [ $n -ge 1 ]; then sleep 34 & fi
        exec {sys.executable} {env.child_py} {{dev}}"""])
    out = tmp_path / "out"
    code = hc.main(env.argv(out) + ["--child-cmd", cmd], clock=env.clock, sleep=env.sleep)
    assert code == 1
    assert by_check(out, "H4.gone")[0]["ok"] is False
    assert "H4.reset" not in check_ids(out)
    assert len(env.real_resets()) == 2           # the probe and the reset after H3
    assert board_free(env)


def test_h6_fails_on_the_history_even_if_the_message_looks_clean(env, tmp_path):
    (tmp_path / "fail-at").write_text("3")
    w = env.wrapper(rewrite=("release", """
        import json, sys
        d = json.load(sys.stdin)
        d["message"] = "reset ok"
        print(json.dumps(d))
    """))
    out = tmp_path / "out"
    assert hc.main(env.argv(out, gozer=w), clock=env.clock, sleep=env.sleep) == 1
    rel = by_check(out, "H6.release")[0]
    assert rel["ok"] is False and rel["evidence"]["history_reset_ok"] is False


def test_chips_must_be_claimed_after_the_reset(env, tmp_path):
    def hook(name, check=None, **info):
        if name == "before_reset_status" and check == "H3.reset":
            add_foreign_holder(env)
    out = tmp_path / "out"
    assert env.run(out, hook=hook) == 1
    assert "H4" not in check_ids(out)
    reasons = [n["evidence"]["reason"] for n in by_check(out, "ABORT")]
    assert any("not CLAIMED after H3.reset" in r for r in reasons)


def test_signal_during_the_h3_reset_lets_it_finish(env, tmp_path):
    (tmp_path / "slow-at").write_text("1")      # the reset after H3 takes 4 s
    out = tmp_path / "out"
    p = spawn_driver(env, env.argv(out, "--reset-timeout", "60"))
    assert wait_for(lambda: env.resets() >= 1), "the H3 reset never started"
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=60)
    assert p.returncode == 1
    assert (tmp_path / "reset-finished").exists()
    resets = env.real_resets()
    assert [c[1] for c in resets] == ["rc=15", "rc=0"]    # the reset reported success
    assert env.resets() == 2                     # that reset, then the cleanup release
    assert board_free(env)


def test_a_reset_slower_than_one_timeout_but_not_two_is_waited_for(env, tmp_path):
    (tmp_path / "slow-at").write_text("1")      # takes 4 s
    out = tmp_path / "out"
    assert hc.main(env.argv(out, "--reset-timeout", "3"), clock=env.clock, sleep=env.sleep) == 0
    assert by_check(out, "H3.reset")[0]["ok"]


def test_probe_needs_exit_15_even_with_the_right_text(env, tmp_path):
    w = env.wrapper([("reset", 1, 1, 17, '{"message": "chips [2] still open by [1]"}')])
    out = tmp_path / "out"
    assert hc.main(env.argv(out, gozer=w), clock=env.clock, sleep=env.sleep) == 1
    assert by_check(out, "H3.refuse")[0]["ok"] is False


def test_probe_fails_if_its_stand_in_reset_ran_even_with_exit_15(env, tmp_path):
    out = tmp_path / "out"
    w = script(tmp_path / "gozer-ran", f'''
        if [ "$1" = reset ] && [ "$2" != --help ]; then
            echo ran >> {out}/refusal-probe-marker
            echo '{{"message": "chips [2] still open by [1]"}}'
            exit 15
        fi
        exec {env.gozer} "$@"
    ''')
    assert hc.main(env.argv(out, gozer=w), clock=env.clock, sleep=env.sleep) == 1
    refuse = by_check(out, "H3.refuse")[0]
    assert refuse["ok"] is False and refuse["evidence"]["stand_in_ran"] is True


def test_signal_during_the_refusal_probe_lets_it_finish(env, tmp_path):
    w = script(tmp_path / "gozer-slow-probe", f'''
        if [ "$1" = reset ] && [ "$2" != --help ] && [ ! -e {tmp_path}/probed ]; then
            touch {tmp_path}/probed
            sleep 3
        fi
        exec {env.gozer} "$@"
    ''')
    out = tmp_path / "out"
    p = spawn_driver(env, env.argv(out, gozer=w))
    assert wait_for(lambda: (tmp_path / "probed").exists())
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=60)
    assert p.returncode == 1
    first = env.real_resets()[0]
    assert first[1] == "rc=15"                   # the probe ran to the end and was logged
    assert board_free(env)


# ---------------------------------------------------------------------------------------------
# Fix round 2.

def test_a_signal_during_popen_of_a_protected_call_is_deferred(env, tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    sent = []

    def popen(argv, *a, **k):
        # The signal lands after the driver decided to run the reset and before the process exists.
        if "reset" in argv and "--help" not in argv and not sent:
            sent.append(1)
            os.kill(os.getpid(), signal.SIGTERM)
        return real_popen(argv, *a, **k)
    monkeypatch.setattr(hc.subprocess, "Popen", popen)
    out = tmp_path / "out"
    code = env.run(out)
    assert code == 1
    first = env.real_resets()[0]
    assert first[1] == "rc=15"                   # the probe ran to the end and was logged
    assert env.resets() == 1                     # only the cleanup release reset the chips
    reasons = [n["evidence"]["reason"] for n in by_check(out, "ABORT")]
    assert any("SIGTERM" in r for r in reasons)
    assert board_free(env)


def test_gozer_env_overrides_refuse_the_run(env, tmp_path):
    argv = env.argv(tmp_path / "out")
    argv.remove("--allow-gozer-env")
    out = tmp_path / "out"
    assert hc.main(argv, clock=env.clock, sleep=env.sleep) == 2
    assert not [c for c in env.calls() if c[0].startswith("acquire")]
    refused = by_check(out, "P.env")[0]
    assert refused["ok"] is False and "GOZER_RESET_CMD" in refused["evidence"]["summary"]
    start = ledger_entries(out)[0]["data"]
    assert "GOZER_RESET_CMD" in start["gozer_env"]


def test_gozer_env_overrides_are_allowed_with_the_flag(env, tmp_path):
    out = tmp_path / "out"
    assert env.run(out) == 0
    assert "GOZER_ROOT" in ledger_entries(out)[0]["data"]["gozer_env"]


def test_clean_environment_passes_the_env_check(env, tmp_path, monkeypatch):
    # The fake roots are gone from the driver's environment, so the check passes. gozer then
    # looks at the real root, so stop the run in preflight with a missing reset command.
    for k in list(os.environ):
        if k.startswith("GOZER_"):
            monkeypatch.delenv(k)
    d = hc.Driver(hc.parse_args(["--board", BOARD]), None, None, None, None)
    assert d.gozer_env() == {}


def test_history_event_older_than_the_acquire_does_not_count(tmp_path):
    hist = tmp_path / "history.jsonl"
    hist.write_text(json.dumps({"event": "released", "lease_id": "a", "reset_ran": True,
                                "reset_ok": True, "ts": "2026-10-02T10:00:00Z"}) + "\n")
    assert hc.release_reset_confirmed(str(tmp_path), "a", since="2026-10-02T09:00:00Z") is True
    assert hc.release_reset_confirmed(str(tmp_path), "a", since="2026-10-02T10:00:01Z") is None


def test_h6_ignores_an_old_released_event_for_the_same_lease(env, tmp_path):
    (tmp_path / "fail-at").write_text("3")       # the real release's reset fails
    w = env.wrapper(rewrite=("release", """
        import json, sys
        d = json.load(sys.stdin)
        d["message"] = "reset ok"
        print(json.dumps(d))
    """))

    def hook(name, driver=None, **info):
        if name == "after_release":
            hist = Path(os.environ["GOZER_HISTORY_ROOT"]) / "history.jsonl"
            with open(hist, "a") as f:
                f.write(json.dumps({"event": "released", "lease_id": driver.lease_id,
                                    "reset_ran": True, "reset_ok": True,
                                    "ts": "2000-01-01T00:00:00Z"}) + "\n")
    out = tmp_path / "out"
    assert hc.main(env.argv(out, gozer=w), clock=env.clock, sleep=env.sleep, hook=hook) == 1
    assert by_check(out, "H6.release")[0]["ok"] is False


@pytest.mark.parametrize("name, use_dot", [("out dir $x`y`\"q", False), ("dot", True)])
def test_probe_stand_in_survives_odd_out_dirs(env, tmp_path, monkeypatch, name, use_dot):
    # The child holds an index on no board, so the probe is not refused and the stand-in runs.
    cmd = json.dumps([sys.executable, str(env.child_py), "9"])
    base = tmp_path / name
    base.mkdir()
    argv = env.argv(base, board=BOARD) + ["--child-cmd", cmd]
    if use_dot:
        monkeypatch.chdir(base)
        argv[argv.index("--out-dir") + 1] = "."
    assert hc.main(argv, clock=env.clock, sleep=env.sleep) == 1
    assert by_check(base, "H3.refuse")[0]["evidence"]["stand_in_ran"] is True
    assert env.real_resets()[0][2] == "resets=0"


def test_a_baseexception_while_waiting_does_not_kill_a_protected_call(env, tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    procs = []

    class Wrapped:
        def __init__(self, *a, **k):
            self.real = real_popen(*a, **k)
            procs.append(self.real)
            self.pid = self.real.pid

        def communicate(self, timeout=None):
            raise KeyboardInterrupt

        def __getattr__(self, name):
            return getattr(self.real, name)
    monkeypatch.setattr(hc.subprocess, "Popen", Wrapped)
    d = hc.Driver(hc.parse_args(["--board", BOARD]), None, None, None, None)
    try:
        with pytest.raises(KeyboardInterrupt):
            d.run_cmd(["sleep", "20"], 30, protect=True)
        assert d.abandoned == ["sleep", "20"]
        assert procs[0].poll() is None           # gozer was left running
        # an unprotected call is still stopped
        with pytest.raises(KeyboardInterrupt):
            d.run_cmd(["sleep", "20"], 30)
        procs[1].wait(timeout=5)
        assert procs[1].returncode != 0
    finally:
        for pr in procs:
            if pr.poll() is None:
                pr.kill()
            pr.wait()


def test_pause_after_open_waits_before_the_probe_and_says_what_to_check(env, tmp_path, capsys):
    seen = {}

    def hook(name, **info):
        if name == "after_pause":
            seen["slept"] = sum(env.sleeps)
            seen["resets"] = len(env.real_resets())
    out = tmp_path / "out"
    assert env.run(out, "--pause-after-open", "5", hook=hook) == 0
    assert seen["slept"] >= 5 and seen["resets"] == 0     # the probe had not run yet
    assert max(env.sleeps[:5]) <= 1.0                       # in small slices
    err = capsys.readouterr().err
    pid = by_check(out, "H2")[0]["evidence"]["child_pid"]
    assert f"ls -l /proc/{pid}/fd" in err


def test_pause_after_open_defaults_to_zero():
    assert hc.parse_args(["--board", BOARD]).pause_after_open == 0
