# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""scripts/can_cad shell and serve scripts against fakes (review of PR 629): the lease wrapper claims a queued ticket with an owner pid and
stops on errors that are not a queue answer; serve_staged only suppresses tt-model's device check for chips of a live lease; boot_check.sh
fails when the health check or a benchmark fails and always removes the container it started; the 2-chip serve default is the working one.
Every external program (gozer, curl, docker, python scripts) is a small fake in a temporary directory."""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "can_cad"
sys.path.insert(0, str(SCRIPTS))


def write_exec(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


# ---- lease_run.sh ---------------------------------------------------------------------------------------------------------------------

GRANT = {"granted": True, "lease_id": "abc123", "owner_pid": 4242, "dev_indices": [0, 1], "chips": ["0000:01:00.0", "0000:02:00.0"],
         "env": {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}, "units": ["u"]}


def fake_gozer(tmp: Path, script: list):
    """A gozer whose `acquire` calls answer from `script`, one (exit code, json) per call; every call line goes to calls.log."""
    (tmp / "script.json").write_text(json.dumps(script))
    write_exec(tmp / "gozer", f"""#!/usr/bin/env python3
import json, sys
log = open({str(tmp / 'calls.log')!r}, 'a'); log.write(' '.join(sys.argv[1:]) + '\\n'); log.close()
if sys.argv[1] == 'release':
    sys.exit(0)
state = {str(tmp / 'n.txt')!r}
try: n = int(open(state).read())
except FileNotFoundError: n = 0
open(state, 'w').write(str(n + 1))
code, payload = json.load(open({str(tmp / 'script.json')!r}))[min(n, len(json.load(open({str(tmp / 'script.json')!r}))) - 1)]
print(json.dumps(payload) if isinstance(payload, dict) else payload)
sys.exit(code)
""")


def run_lease(tmp: Path, *cmd, env_extra=None):
    env = dict(os.environ, PATH=f"{tmp}:{os.environ['PATH']}", LEASE_POLL_S="0", LEASE_SERIAL_WAIT_S="0", **(env_extra or {}))
    return subprocess.run(["bash", str(SCRIPTS / "lease_run.sh"), "test reason", *cmd], env=env, capture_output=True, text=True, timeout=60)


def calls(tmp: Path):
    return (tmp / "calls.log").read_text().splitlines() if (tmp / "calls.log").exists() else []


def test_a_direct_grant_runs_the_command_with_the_lease_exported_and_releases(tmp_path):
    fake_gozer(tmp_path, [[0, GRANT]])
    r = run_lease(tmp_path, "bash", "-c", 'echo "$TT_LEASE_ID $TT_LEASE_DEV_INDICES $TT_VISIBLE_DEVICES $TT_LEASE_OWNER_PID"; exit 3')
    assert r.returncode == 3 and r.stdout.split() == ["abc123", "0,1", "0000:01:00.0,0000:02:00.0", "4242"]
    assert calls(tmp_path)[-1] == "release abc123"
    assert "--owner-pid" in calls(tmp_path)[0]


def fake_gozer_release_status(tmp: Path, rc: int):
    """Make the fake gozer's `release` exit with rc."""
    g = (tmp / "gozer").read_text().replace("if sys.argv[1] == 'release':\n    sys.exit(0)", f"if sys.argv[1] == 'release':\n    print('release refused', file=sys.stderr)\n    sys.exit({rc})")
    (tmp / "gozer").write_text(g)


def test_a_failed_release_after_a_successful_command_is_the_wrappers_exit_status(tmp_path):
    fake_gozer(tmp_path, [[0, GRANT]])
    fake_gozer_release_status(tmp_path, 15)
    r = run_lease(tmp_path, "true")
    assert r.returncode == 15 and "release refused" in r.stderr


def test_a_failed_command_status_wins_over_a_failed_release(tmp_path):
    fake_gozer(tmp_path, [[0, GRANT]])
    fake_gozer_release_status(tmp_path, 19)
    assert run_lease(tmp_path, "bash", "-c", "exit 3").returncode == 3


def test_a_queued_ticket_is_claimed_with_acquire_ticket_and_the_owner_pid_never_wait(tmp_path):
    fake_gozer(tmp_path, [[10, {"queued": True, "ticket": "t77", "position": 1}], [10, {"queued": True, "ticket": "t77", "position": 1}], [0, GRANT]])
    r = run_lease(tmp_path, "true")
    assert r.returncode == 0, r.stderr
    log = calls(tmp_path)
    assert not any(line.startswith("wait") for line in log)
    claims = [line for line in log if "--ticket t77" in line]
    assert len(claims) == 2 and all("--owner-pid" in line for line in claims)
    assert log[-1] == "release abc123"


def test_exit_14_unreadable_serial_is_retried_a_few_times_then_passed_on(tmp_path):
    fake_gozer(tmp_path, [[14, "gozer: unreadable tt_serial on a board"]])
    r = run_lease(tmp_path, "true")
    assert r.returncode == 14
    assert len([c for c in calls(tmp_path) if c.startswith("acquire")]) == 4          # the first call and three retries, not a loop


def test_exit_14_that_clears_goes_on(tmp_path):
    fake_gozer(tmp_path, [[14, "gozer: unreadable tt_serial on a board"], [0, GRANT]])
    assert run_lease(tmp_path, "true").returncode == 0


@pytest.mark.parametrize("code", [12, 13, 15, 2])
def test_other_failures_are_passed_on_at_once_with_no_ticket_loop(tmp_path, code):
    fake_gozer(tmp_path, [[code, "boom"]])
    r = run_lease(tmp_path, "true")
    assert r.returncode == code and len(calls(tmp_path)) == 1


def test_queued_without_a_ticket_is_an_error_not_a_loop(tmp_path):
    fake_gozer(tmp_path, [[10, {"queued": True}]])
    r = run_lease(tmp_path, "true")
    assert r.returncode == 10 and len(calls(tmp_path)) == 1


# ---- serve_staged.lease_covers --------------------------------------------------------------------------------------------------------

def status(*chips):
    return {"chips": [{"bdf": b, "dev_index": i, "state": s, "pid": p} for (b, i, s, p) in chips]}


LEASE_ENV = {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0", "TT_LEASE_OWNER_PID": "4242", "TT_LEASE_DEV_INDICES": "0,1"}
LIVE = status(("0000:01:00.0", 0, "HELD", 4242), ("0000:02:00.0", 1, "CLAIMED", 4242), ("0000:03:00.0", 2, "FREE", None), ("0000:04:00.0", 3, "FREE", None))


def test_a_pin_inside_a_live_lease_is_covered():
    import serve_staged
    ok, why = serve_staged.lease_covers([0, 1], LEASE_ENV, LIVE)
    assert ok, why


def test_a_pin_outside_the_lease_is_not_covered():
    import serve_staged
    ok, why = serve_staged.lease_covers([2, 3], LEASE_ENV, LIVE)
    assert not ok and "not in the lease" in why
    assert not serve_staged.lease_covers([0, 2], LEASE_ENV, LIVE)[0]


def test_a_lease_that_gozer_no_longer_shows_is_not_covered():
    import serve_staged
    gone = status(("0000:01:00.0", 0, "FREE", None), ("0000:02:00.0", 1, "FREE", None))
    ok, why = serve_staged.lease_covers([0, 1], LEASE_ENV, gone)
    assert not ok and "not held" in why
    other = status(("0000:01:00.0", 0, "HELD", 999), ("0000:02:00.0", 1, "HELD", 999))
    assert not serve_staged.lease_covers([0, 1], LEASE_ENV, other)[0]


def test_a_chip_gozer_shows_free_is_not_covered_even_with_the_owner_pid():
    import serve_staged
    stale = status(("0000:01:00.0", 0, "FREE", 4242), ("0000:02:00.0", 1, "HELD", 4242))
    assert not serve_staged.lease_covers([0, 1], LEASE_ENV, stale)[0]


def test_no_lease_environment_means_not_covered():
    import serve_staged
    assert not serve_staged.lease_covers([0, 1], {}, LIVE)[0]


# ---- boot_check.sh --------------------------------------------------------------------------------------------------------------------

def boot_rig(tmp: Path, *, health_ok=True, bench_rc=0, serve_rc=0, container_stays=True, preexisting=False, serve_creates=True):
    work = tmp / "work"; (work / "logs").mkdir(parents=True)
    bin_ = tmp / "bin"; bin_.mkdir()
    exists = tmp / "container-exists"
    if preexisting:
        exists.write_text("1")
    (work / "bench-package-p300.json").write_text("STALE")                  # a stale result from an earlier run must not survive a failed one
    create = f"open({str(exists)!r}, 'w').write('1')\n" if serve_creates else ""
    write_exec(work / "serve_staged.py", f"import sys\n{create}sys.exit({serve_rc})\n")
    write_exec(work / "bench.py", f"import sys, json\nopen('bench-package-p300.json','w').write('{{}}')\nsys.exit({bench_rc})\n")
    write_exec(work / "bench_prefill.py", "import sys\nopen('prefill-package-p300.json','w').write('[]')\n")
    write_exec(bin_ / "curl", f"#!/bin/bash\n[[ \"$*\" == *--fail* ]] || {{ echo 'curl without --fail' >> {tmp}/curl-bad.log; }}\nexit {0 if health_ok else 22}\n")
    name = "tt-model-berkelium-qwen3-32b-cadquery-p300-p300"
    running = f"[ -e {exists} ] && echo {name}" if container_stays else "true"
    write_exec(bin_ / "docker", f"""#!/bin/bash
echo "$@" >> {tmp}/docker.log
case "$*" in
  "ps -a --format {{{{.Names}}}}") [ -e {exists} ] && echo {name};;
  "ps -a -q "*) [ -e {exists} ] && echo cid123;;
  "ps --format {{{{.Names}}}}") {running};;
esac
exit 0
""")
    return work, bin_


def run_boot(tmp: Path, work: Path, bin_: Path):
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", CAN_CAD_WORK=str(work), HF_HOME=str(tmp / "hf"), TT_MODEL_PYTHON=sys.executable,
               BOOT_TIMEOUT_S="2", BOOT_POLL_S="0.2", BENCH_PYTHON=sys.executable)
    return subprocess.run(["bash", str(SCRIPTS / "boot_check.sh")], env=env, capture_output=True, text=True, timeout=120)


def test_a_clean_boot_check_exits_zero_and_leaves_results(tmp_path):
    work, bin_ = boot_rig(tmp_path)
    r = run_boot(tmp_path, work, bin_)
    assert r.returncode == 0, r.stderr
    assert (work / "logs" / "boot-status.txt").read_text().strip() == "OK" and (work / "bench-package-p300.json").read_text() == "{}"
    assert not (tmp_path / "curl-bad.log").exists(), "curl was called without --fail"


def test_a_health_check_that_never_passes_fails_and_removes_stale_results_and_the_container(tmp_path):
    work, bin_ = boot_rig(tmp_path, health_ok=False)
    r = run_boot(tmp_path, work, bin_)
    assert r.returncode != 0
    assert not (work / "bench-package-p300.json").exists() and not (work / "logs" / "boot-ready.txt").exists()
    log = (tmp_path / "docker.log").read_text()
    assert "stop cid123" in log and "rm -f cid123" in log


def test_a_failed_benchmark_fails_the_check(tmp_path):
    work, bin_ = boot_rig(tmp_path, bench_rc=1)
    assert run_boot(tmp_path, work, bin_).returncode != 0


def test_a_failed_start_that_created_a_container_cleans_up_that_container_by_id(tmp_path):
    work, bin_ = boot_rig(tmp_path, serve_rc=3)
    assert run_boot(tmp_path, work, bin_).returncode != 0
    log = (tmp_path / "docker.log").read_text()
    assert "stop cid123" in log and "rm -f cid123" in log


def test_a_failed_start_that_created_nothing_stops_and_removes_nothing(tmp_path):
    work, bin_ = boot_rig(tmp_path, serve_rc=3, serve_creates=False)
    assert run_boot(tmp_path, work, bin_).returncode != 0
    log = (tmp_path / "docker.log").read_text()
    assert " stop " not in f" {log}" and "rm -f" not in log


def test_a_pre_existing_container_with_the_name_is_refused_and_left_alone(tmp_path):
    work, bin_ = boot_rig(tmp_path, preexisting=True)
    r = run_boot(tmp_path, work, bin_)
    assert r.returncode != 0 and "already exists" in (work / "logs" / "boot-status.txt").read_text()
    log = (tmp_path / "docker.log").read_text()
    assert "stop" not in log and "rm -f" not in log and (tmp_path / "container-exists").exists()


def test_a_container_that_disappears_fails_the_check(tmp_path):
    work, bin_ = boot_rig(tmp_path, health_ok=False, container_stays=False)
    r = run_boot(tmp_path, work, bin_)
    assert r.returncode != 0 and "gone" in (work / "logs" / "boot-status.txt").read_text()


# ---- serve-2chip.py -------------------------------------------------------------------------------------------------------------------

def test_the_two_chip_serve_default_is_the_working_32_sequences_and_the_usage_example_says_so():
    src = (SCRIPTS / "serve-2chip.py").read_text()
    assert 'ap.add_argument("--max-num-seqs", type=int, default=32)' in src
    assert "--max-num-seqs 8" not in src and "--max-num-seqs 32" in src
