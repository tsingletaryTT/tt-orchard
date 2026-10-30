"""The prefill benchmark's guards (orchard/prefill_bench.py). CPU only: no chip, no model.

The benchmark exists because a timing run once hung and left a chip unreadable (issue 290). What these tests pin down is that a hang
is killed, is reported with its iteration number, and that the log says where the run stopped."""
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from orchard import prefill_bench as pb

ROOT = Path(__file__).resolve().parent.parent


def events(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def test_line_is_on_disk_before_the_call_returns(tmp_path):
    log = pb.BenchLog(tmp_path / "b.log")
    log.line("before", label="step", iteration=1)
    # read through a second handle: the line must already be flushed
    assert events(tmp_path / "b.log")[0]["event"] == "before"
    assert events(tmp_path / "b.log")[0]["iteration"] == 1


def test_run_iteration_writes_before_and_after_in_order(tmp_path):
    log = pb.BenchLog(tmp_path / "b.log")
    seen = []

    def fn():
        # at this moment the "before" line must exist and the "after" line must not
        seen.append([e["event"] for e in events(tmp_path / "b.log")])
        return "ok"

    out = pb.run_iteration(log, pb.Watchdog(log, kill=lambda code: None), "step", 3, 60, fn)
    assert out[0] == "ok"
    assert seen == [["before"]]
    ev = events(tmp_path / "b.log")
    assert [e["event"] for e in ev] == ["before", "after"]
    assert ev[1]["iteration"] == 3 and ev[1]["seconds"] >= 0


def test_a_slow_iteration_is_killed_and_reported_with_its_number(tmp_path):
    """A real child process: iteration 2 sleeps far past its limit. The child must die with the hang code, quickly, and the log's last line
    must name iteration 2. A guard that only returned a value would let this run for 60 seconds."""
    script = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(ROOT)!r})
        from orchard import prefill_bench as pb
        log = pb.BenchLog({str(tmp_path / 'h.log')!r})
        wd = pb.Watchdog(log)
        pb.run_iteration(log, wd, "chunked", 1, 5, lambda: time.sleep(0.01))
        pb.run_iteration(log, wd, "chunked", 2, 1, lambda: time.sleep(60))
        print("not reached")
    """)
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert time.time() - t0 < 20
    assert p.returncode == pb.HANG_EXIT
    assert "not reached" not in p.stdout
    last = events(tmp_path / "h.log")[-1]
    assert last["event"] == "hang" and last["iteration"] == 2 and last["label"] == "chunked" and last["timeout_s"] == 1


def test_a_fast_iteration_cancels_its_timer(tmp_path):
    """If the timer were not cancelled the kill hook would fire after the iteration finished."""
    log = pb.BenchLog(tmp_path / "b.log")
    fired = []
    wd = pb.Watchdog(log, kill=lambda code: fired.append(code))
    pb.run_iteration(log, wd, "step", 1, 0.3, lambda: None)
    time.sleep(0.6)
    assert fired == []


def test_trace_on_is_refused_without_a_warmup_line(tmp_path):
    log = pb.BenchLog(tmp_path / "b.log")
    with pytest.raises(pb.TraceWithoutWarmup):
        pb.require_warmup(log, trace=True)
    # trace off needs no warm-up line
    pb.require_warmup(log, trace=False)
    # a "before" line without its "after" does not count: the warm-up did not finish
    log.line("before", label="chunked", iteration=0, phase="warmup")
    with pytest.raises(pb.TraceWithoutWarmup):
        pb.require_warmup(log, trace=True)
    log.line("after", label="chunked", iteration=0, phase="warmup", seconds=1.0)
    pb.require_warmup(log, trace=True)


def test_run_plan_refuses_trace_on_with_no_warmup_and_runs_nothing(tmp_path):
    log = pb.BenchLog(tmp_path / "b.log")
    called = []
    with pytest.raises(pb.TraceWithoutWarmup):
        pb.run_plan(log, pb.Watchdog(log, kill=lambda c: None), "chunked", lambda: called.append(1), warmups=0, runs=1, trace=True,
                    warmup_timeout=5, timeout=5)
    assert called == []


def test_run_plan_warms_up_first_then_times_and_logs_every_iteration(tmp_path):
    log = pb.BenchLog(tmp_path / "b.log")
    res = pb.run_plan(log, pb.Watchdog(log, kill=lambda c: None), "step", lambda: None, warmups=1, runs=3, trace=True, warmup_timeout=5, timeout=5)
    assert len(res) == 3
    ev = events(tmp_path / "b.log")
    assert [(e["event"], e["phase"]) for e in ev] == [("before", "warmup"), ("after", "warmup")] + [(k, "timed") for _ in range(3) for k in ("before", "after")]
    assert [e["iteration"] for e in ev if e["phase"] == "timed" and e["event"] == "after"] == [1, 2, 3]


def test_summary_labels_n_and_reports_every_run():
    s = pb.summarize(512, [2.0, 4.0, 4.0])
    assert s["n"] == 3 and s["prompt_tokens"] == 512
    assert s["seconds"] == [2.0, 4.0, 4.0]
    assert s["tok_s"] == [256.0, 128.0, 128.0]
    assert s["tok_s_median"] == 128.0 and s["tok_s_min"] == 128.0 and s["tok_s_max"] == 256.0


def test_total_budget_is_under_ten_minutes():
    # the whole process runs under `timeout`; the plan must fit inside it
    assert pb.PROCESS_TIMEOUT_S < 600


# ---- issue 293 ----------------------------------------------------------------------------------------------------------------
def test_the_kill_runs_even_when_the_hang_line_cannot_be_written(tmp_path):
    """Issue 293 item 1: if writing the hang line raises OSError the timer thread used to end without calling kill."""
    class BrokenLog(pb.BenchLog):
        def line(self, event, **fields):
            if event == "hang":
                raise OSError("disk full")
            super().line(event, **fields)

    log = BrokenLog(tmp_path / "b.log")
    fired = []
    wd = pb.Watchdog(log, kill=lambda code: fired.append(code))
    wd.arm("chunked", 2, 0.05, "timed")
    time.sleep(0.5)
    assert fired == [pb.HANG_EXIT]


def test_a_failing_iteration_logs_the_exception_text_before_it_propagates(tmp_path):
    """Issue 293 item 2: the trace-on wedge ended by an exception whose text was lost when the cleanup hung."""
    log = pb.BenchLog(tmp_path / "b.log")
    seen_at_raise = []

    def fn():
        raise RuntimeError("device buffer corrupted")

    with pytest.raises(RuntimeError, match="device buffer corrupted"):
        try:
            pb.run_iteration(log, pb.Watchdog(log, kill=lambda c: None), "chunked", 4, 60, fn)
        finally:
            # the caller's cleanup would run here: the error line must already be on disk
            seen_at_raise.append([e["event"] for e in events(tmp_path / "b.log")])
    assert seen_at_raise == [["before", "error"]]
    err = events(tmp_path / "b.log")[-1]
    assert err["iteration"] == 4 and err["label"] == "chunked" and err["phase"] == "timed"
    assert err["type"] == "RuntimeError" and "device buffer corrupted" in err["message"]


def test_an_unwritable_error_line_does_not_replace_the_original_exception(tmp_path):
    class BrokenLog(pb.BenchLog):
        def line(self, event, **fields):
            if event == "error":
                raise OSError("disk full")
            super().line(event, **fields)

    log = BrokenLog(tmp_path / "b.log")
    with pytest.raises(ValueError, match="original"):
        pb.run_iteration(log, pb.Watchdog(log, kill=lambda c: None), "x", 1, 60, lambda: (_ for _ in ()).throw(ValueError("original")))


def test_greedy_tokens_steps_only_when_another_token_is_needed():
    """Issue 293 item 3: the loop called step after the last token and threw the result away, which could pass the context limit."""
    import torch
    calls = []

    def step(t):
        calls.append(t)
        return torch.tensor([0.0, 0.0, float(t + 2)])

    first = torch.tensor([0.0, 1.0, 0.0])             # argmax 1
    out = pb.greedy_tokens(first, step, 4)
    assert len(out) == 4 and out[0] == 1
    assert len(calls) == 3                            # three steps give tokens 2 to 4; no step after the fourth
    assert pb.greedy_tokens(first, step, 1) == [1] and len(calls) == 3


def test_greedy_tokens_refuses_a_nonpositive_count():
    import torch
    for n in (0, -3):
        with pytest.raises(ValueError, match="at least 1"):
            pb.greedy_tokens(torch.tensor([0.0, 1.0]), lambda t: torch.tensor([1.0, 0.0]), n)
