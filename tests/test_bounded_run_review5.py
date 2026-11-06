# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 422, Copilot round 5, for orchard/bounded_run.py and its test harness.

1. The request writer was a blocking `write` on a daemon thread. A wrapper that exits after starting a descendant in another session, which
   inherits stdin and never reads it, left that thread blocked in the write for good, holding the pipe and the request bytes. Repeated calls
   accumulate them. The writer now polls a non-blocking descriptor, looks at the stop flag, and `run` joins it with the readers.
2. `limited()` killed only the scenario's process group. A helper that `bounded_run` had started in a session of its own survived a failed
   scenario. `limited()` now also kills every process that carries the scenario's environment tag."""
import os
import signal
import subprocess
import sys
import time

import pytest

from test_bounded_run import TAG_VAR, gone, limited, reap_tagged


def test_a_blocked_request_writer_does_not_outlive_run(tmp_path):
    """The wrapper ends at once. Its descendant (a session of its own) holds stdin and never reads, so a request larger than the pipe blocks."""
    pid_file = tmp_path / "descendant.pid"
    got = limited(f"""
        import threading
        br.GRACE_S = 1.0
        br.POLL_S = 0.5                     # a writer that is not joined is still alive when run returns
        res = br.run(["sh", "-c", "exec 3<&0; setsid sleep 40 <&3 & echo $! > {pid_file}; exit 0"], "x" * (4 << 20),
                     timeout_s=10, out_max=1000, err_max=1000)
        print(json.dumps({{"problem": res.problem, "threads": threading.active_count()}}), flush=True)
    """, deadline_s=30)
    assert got["problem"] is None
    assert got["threads"] == 1, f"{got['threads'] - 1} helper thread(s) (the request writer) outlived run()"


def test_limited_kills_a_helper_that_sits_in_a_session_of_its_own(tmp_path):
    """The scenario hangs past its deadline while `bounded_run` has a helper in another session; the helper must be gone afterwards."""
    pid_file = tmp_path / "helper.pid"
    with pytest.raises(pytest.fail.Exception):
        limited(f"""
            br.run(["sh", "-c", "echo $$ > {pid_file}; exec sleep 300"], "", timeout_s=200, out_max=1000, err_max=1000)
        """, deadline_s=3)
    assert gone(int(pid_file.read_text())), "a helper in a session of its own survived the failed scenario"


def test_reap_tagged_kills_only_the_processes_that_carry_the_tag():
    tag = f"unit-{os.getpid()}-{time.monotonic_ns()}"
    tagged = subprocess.Popen(["sleep", "60"], env={**os.environ, TAG_VAR: tag}, start_new_session=True)
    other = subprocess.Popen(["sleep", "60"], env={**os.environ, TAG_VAR: tag + "x"}, start_new_session=True)
    try:
        assert reap_tagged(tag) == 1
        assert tagged.wait(timeout=5) == -signal.SIGKILL and other.poll() is None
    finally:
        for p in (tagged, other):
            if p.poll() is None:
                p.kill()
            p.wait()


def _listed_test_run(tmp_path, spawn, boards=1):
    """One configuration's test through `Supervisor._run_listed_test` (the stage 4 path), with `spawn` standing in for the hardware run.
    Returns the mask-input ledger entries the run wrote."""
    from pathlib import Path
    from types import SimpleNamespace
    from orchard import stages
    from orchard.ledger import Ledger, read_entries
    from orchard.supervisor import Supervisor
    run = tmp_path / "run"
    md = run / "stages" / "4" / "configs" / "1" / "model-dir"
    md.mkdir(parents=True)
    (md / "tokenizer.json").write_text("tok")
    sup = object.__new__(Supervisor)
    sup.run_dir = run
    sup.ledger = Ledger(run / "ledger.jsonl")
    sup._spawn = spawn
    sup._moved_aside_record = lambda p: {}
    sup._run_listed_test_body = lambda *a: Supervisor._run_listed_test_body(sup, *a)
    sup._reset_test_chips = lambda *a: None
    test = SimpleNamespace(chips=2, boards=boards, command=lambda n: "cmd", deadline_s=10.0, cache="/c",
                           output_path=lambda sd: Path(sd) / "tests" / "2" / "output.txt", record_path=lambda sd: Path(sd) / "tests" / "2" / "test-result.json")
    lease = SimpleNamespace(chips=["a", "b"], dev_indices=[0, 1], lease_id="L")
    try:
        sup._run_listed_test(SimpleNamespace(number=4, budget_s=100.0), run / "stages" / "4", test, [lease])
    finally:
        sup.ledger.close()
    return [e["data"] for e in read_entries(run / "ledger.jsonl")
            if e["event"] == "decision" and e["data"].get("decision") == stages.MASK_INPUTS_DECISION]


def _ok_spawn(*a, **k):
    return {"command": "c", "returncode": 0, "timed_out": False, "seconds": 5.0, "chips": ["a"], "output": {}}


def test_a_stage_4_configuration_test_records_the_mask_input_digests(tmp_path):
    """The wiring for the listed-test path: `_run_test` records them, and stage 4 does not go through `_run_test`."""
    found = _listed_test_run(tmp_path, _ok_spawn)
    assert len(found) == 1 and found[0]["dir"] == "stages/4/configs/1/model-dir" and found[0]["files"]["tokenizer.json"], found


def test_the_digests_are_taken_once_after_the_retry_not_before_it(tmp_path):
    """A 2-board test that fails at startup is run again. The record is of the files as the final attempt left them."""
    import hashlib
    calls = []
    model_dir = tmp_path / "run" / "stages" / "4" / "configs" / "1" / "model-dir"

    def spawn(*a, **k):
        calls.append(1)
        model_dir.joinpath("tokenizer.json").write_text(f"tok{len(calls)}")
        a[5].write_text("")                          # the output file the retry moves aside
        return {"command": "c", "returncode": 1 if len(calls) == 1 else 0, "timed_out": False, "seconds": 1.0, "chips": ["a"], "output": {}}
    found = _listed_test_run(tmp_path, spawn, boards=2)
    assert len(calls) == 2 and len(found) == 1
    assert found[0]["files"]["tokenizer.json"] == hashlib.sha256(b"tok2").hexdigest(), found
