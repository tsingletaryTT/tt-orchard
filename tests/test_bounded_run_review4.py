# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/bounded_run.py after the independent review of PR 422 (one Medium and two Low findings).

1. MEDIUM, a hang: the child ends by itself but a descendant in its process group keeps the output pipes open and writes nothing. The old
   `run` skipped the group stop (the child had ended), left its reader threads blocked in `read1`, and then `BufferedReader.close()` waited for
   the lock the blocked reader held: `run` returned only when the descendant closed the pipe, and with a long-lived one it never returned.
2. LOW: a child that writes more than the cap and exits before a reader has crossed it was reported as success with cut-short output.
3. LOW: the group was signalled after the child had been reaped, so a recycled pid could have named somebody else's group.

Every scenario runs in the second, limited child of `test_bounded_run.limited` (2 GiB of address space, a 60 s deadline, a session of its own),
so a mutation that brings the hang back is bounded by the scenario's own short deadline and cannot take the machine down."""
import os
import time

import pytest

from test_bounded_run import gone, limited

# a long sleep that a mutation which brings the hang back would sit on for this long (the scenario deadline below is shorter than the sleep)
HOLD_S = 25


def test_a_descendant_that_holds_the_pipes_cannot_hold_run_past_its_return(tmp_path):
    """The Medium finding, in its reproduced form: `sh -c "sleep N & echo hi; exit 0"`."""
    pid_file = tmp_path / "grandchild.pid"
    got = limited(f"""
        import time
        br.GRACE_S = 1.0
        t0 = time.time()
        res = br.run(["sh", "-c", "sleep {HOLD_S} & echo $! > {pid_file}; echo hi; exit 0"], "", timeout_s=10, out_max=1000, err_max=1000)
        print(json.dumps({{"elapsed": time.time() - t0, "problem": res.problem, "returncode": res.returncode, "stdout": res.stdout}}), flush=True)
    """, deadline_s=HOLD_S + 15)
    assert got["elapsed"] < 8, f"run() was held for {got['elapsed']:.1f} s by a descendant that holds the pipes"
    assert (got["problem"], got["returncode"], got["stdout"].strip()) == (None, 0, "hi")
    assert gone(int(pid_file.read_text())), "the descendant was left running after the child ended by itself"


def test_a_descendant_that_ignores_the_first_signal_is_killed_when_the_child_ended_by_itself(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    got = limited(f"""
        import time
        br.GRACE_S = 1.0
        t0 = time.time()
        script = "(trap '' TERM; exec sleep {HOLD_S}) & echo $! > {pid_file}; exit 0"
        res = br.run(["sh", "-c", script], "", timeout_s=10, out_max=1000, err_max=1000)
        print(json.dumps({{"elapsed": time.time() - t0, "problem": res.problem, "returncode": res.returncode}}), flush=True)
    """, deadline_s=HOLD_S + 15)
    assert got["elapsed"] < 8 and got["problem"] is None and got["returncode"] == 0
    assert gone(int(pid_file.read_text()))


def test_a_flood_with_a_silent_descendant_is_still_bounded_by_the_deadline(tmp_path):
    """Every way out is bounded: output over the cap while a descendant that never writes keeps its own copy of the pipes."""
    pid_file = tmp_path / "grandchild.pid"
    got = limited(f"""
        import time
        br.GRACE_S = 1.0
        t0 = time.time()
        script = "sleep {HOLD_S} & echo $! > {pid_file}; while true; do head -c 65536 /dev/zero | tr '\\\\0' x; done"
        res = br.run(["bash", "-c", script], "", timeout_s=5, out_max=100000, err_max=1000)
        print(json.dumps({{"elapsed": time.time() - t0, "problem": res.problem}}), flush=True)
    """, deadline_s=HOLD_S + 15)
    assert got["problem"] == "stdout" and got["elapsed"] < 10
    assert gone(int(pid_file.read_text()))


def test_the_verdict_is_read_after_the_readers_are_done_so_an_over_cap_stream_is_never_reported_as_success():
    """The first Low finding. The child ends by itself and the supervisor sees that before the reader has crossed the cap; the reader crosses it
    a moment later. The race is too narrow to hit on demand, so the test fixes its outcome: `_wait_for_end` reports 'ended by itself' only after
    the child has written 140 kB (cap 100 kB) and the reader has had time to cross. The old code kept the verdict it had been given (success
    with cut-short output); the verdict must come from what the readers saw."""
    got = limited("""
        import time
        real = br._wait_for_end
        def late(proc, deadline, over_out, over_err):
            real(proc, deadline, over_out, over_err)       # returns when the child has ended or a cap was crossed
            time.sleep(0.5)
            return None                                    # 'ended by itself', as the race would have reported
        br._wait_for_end = late
        res = br.run([sys.executable, "-I", "-c", "import sys; sys.stdout.write('x' * 140000)"], "", timeout_s=30, out_max=100000, err_max=1000)
        print(json.dumps({"problem": res.problem, "stdout_len": len(res.stdout)}), flush=True)
    """, deadline_s=50)
    assert got["problem"] == "stdout", got
    assert got["stdout_len"] <= 100_000 + 65536


def test_a_small_output_from_a_child_that_ends_at_once_is_still_complete():
    """The guard against the guard: the drain after the child's end must not cut a small answer."""
    got = limited("""
        res = br.run([sys.executable, "-I", "-c", "import sys; sys.stdout.write('y' * 5000); sys.stderr.write('e' * 70)"], "", timeout_s=20,
                     out_max=100000, err_max=1000)
        print(json.dumps({"problem": res.problem, "returncode": res.returncode, "stdout_len": len(res.stdout), "stderr_len": len(res.stderr)}), flush=True)
    """, deadline_s=40)
    assert (got["problem"], got["returncode"], got["stdout_len"], got["stderr_len"]) == (None, 0, 5000, 70)


def test_the_group_is_only_signalled_while_the_leader_is_not_yet_reaped():
    """The second Low finding: a signal sent to the group after the leader had been reaped names a pid that may have been reused. The leader
    stays a zombie (or alive) until the last signal has gone."""
    got = limited("""
        seen = []
        real = br._signal_group
        def watch(proc, sig):
            seen.append(proc.returncode is None)          # Popen sets returncode when it reaps
            real(proc, sig)
        br._signal_group = watch
        br.GRACE_S = 1.0
        # sh dies on SIGTERM, its background child ignores it: the SIGKILL that follows used to go to a group whose leader was already reaped
        res = br.run(["sh", "-c", "(trap '' TERM; exec sleep 20) & wait"], "", timeout_s=1.0, out_max=1000, err_max=1000)
        print(json.dumps({"problem": res.problem, "unreaped_at_each_signal": seen}), flush=True)
    """, deadline_s=60)
    assert got["problem"] == "timeout"
    assert got["unreaped_at_each_signal"] and all(got["unreaped_at_each_signal"]), got


def test_the_leader_is_reaped_at_the_end_so_no_zombie_is_left_behind():
    got = limited("""
        import os
        procs = []
        real_popen = br.subprocess.Popen
        class Spy(real_popen):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                procs.append(self)
        br.subprocess.Popen = Spy
        res = br.run(["sh", "-c", "sleep 20 & exit 0"], "", timeout_s=10, out_max=100, err_max=100)
        print(json.dumps({"returncode": res.returncode, "reaped": procs[0].returncode is not None}), flush=True)
    """, deadline_s=60)
    assert got["returncode"] == 0 and got["reaped"] is True


def test_the_scan_of_a_group_sees_a_live_member_and_ignores_a_zombie():
    """`_live_members` is what lets the end-of-run wait return as soon as the group is gone and not after a fixed grace period."""
    from orchard import bounded_run as br
    import signal
    import subprocess
    import sys
    proc = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        end = time.time() + 5
        while time.time() < end and proc.pid not in (br._live_members(proc.pid) or []):
            time.sleep(0.02)
        assert proc.pid in br._live_members(proc.pid)
        os.killpg(proc.pid, signal.SIGKILL)
        end = time.time() + 5
        while time.time() < end and br._live_members(proc.pid):
            time.sleep(0.02)
        assert br._live_members(proc.pid) == []          # a zombie leader (not reaped yet) is not a live member
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def test_a_child_that_ended_by_itself_with_nothing_left_returns_at_once_and_does_not_wait_out_the_grace_period():
    """The end-of-run wait returns when the group has no live member, not after a fixed grace period (a long grace here would show)."""
    got = limited("""
        import time
        br.GRACE_S = 3.0
        t0 = time.time()
        res = br.run(["sh", "-c", "echo done; exit 0"], "", timeout_s=10, out_max=1000, err_max=1000)
        print(json.dumps({"elapsed": time.time() - t0, "problem": res.problem, "returncode": res.returncode, "stdout": res.stdout.strip()}), flush=True)
    """, deadline_s=40)
    assert got["elapsed"] < 1.5, got
    assert (got["problem"], got["returncode"], got["stdout"]) == (None, 0, "done")


def test_a_descendant_that_left_the_group_cannot_hold_run_and_run_does_not_pretend_to_reach_it(tmp_path):
    """The group is what `run` can stop. A process that started its own session keeps the pipes open and is outside it: `run` still returns
    within the bound (the readers poll a stop flag; nothing blocks on the pipe) and says nothing false about it."""
    pid_file = tmp_path / "escaped.pid"
    got = limited(f"""
        import time
        br.GRACE_S = 1.0
        t0 = time.time()
        res = br.run(["sh", "-c", "setsid sleep 20 & echo $! > {pid_file}; echo hi; exit 0"], "", timeout_s=10, out_max=1000, err_max=1000)
        print(json.dumps({{"elapsed": time.time() - t0, "problem": res.problem, "stdout": res.stdout.strip()}}), flush=True)
    """, deadline_s=60)
    try:
        assert got["elapsed"] < 4.5, got                 # JOIN_S is 2 s in all
        assert (got["problem"], got["stdout"]) == (None, "hi")
    finally:
        try:
            os.kill(int(pid_file.read_text()), 9)          # the test's own cleanup: the escaped process is outside the group
        except (OSError, ValueError):
            pass


def test_no_signal_is_sent_to_the_group_of_a_child_that_has_been_reaped(monkeypatch):
    """`_signal_group` is the last line against a recycled pid: it does nothing once Popen has reaped the leader."""
    import subprocess
    import sys
    from orchard import bounded_run as br
    sent = []
    monkeypatch.setattr(br.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    proc = subprocess.Popen([sys.executable, "-I", "-c", "pass"], start_new_session=True)
    proc.wait()
    assert proc.returncode is not None
    br._signal_group(proc, 15)
    assert sent == []
    live = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        br._signal_group(live, 15)
        assert sent == [(live.pid, 15)]
    finally:
        monkeypatch.undo()
        os.killpg(live.pid, 9)
        live.wait()


def test_a_descriptor_is_not_closed_while_a_reader_thread_may_still_use_it():
    """If a reader has not ended when the join runs out, its stream is left open: closing a descriptor another thread may be reading from
    could close a reused number. (The readers poll a stop flag, so this does not happen with the real reader; the test uses one that ignores it.)"""
    got = limited("""
        import time
        def stubborn(fd, cap, sink, over, stop):
            time.sleep(3.0)                               # ignores the stop flag and the end of the stream
        br._read = stubborn
        br.JOIN_S = 0.3
        procs = []
        real_popen = br.subprocess.Popen
        class Spy(real_popen):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                procs.append(self)
        br.subprocess.Popen = Spy
        t0 = time.time()
        res = br.run(["sh", "-c", "echo hi"], "", timeout_s=10, out_max=100, err_max=100)
        print(json.dumps({"elapsed": time.time() - t0, "stdout_closed": procs[0].stdout.closed, "stderr_closed": procs[0].stderr.closed}), flush=True)
    """, deadline_s=40)
    assert got["elapsed"] < 2.5, got                       # the join was bounded by JOIN_S
    assert got["stdout_closed"] is False and got["stderr_closed"] is False
