# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Targeted tests for scripts/acestep/acestep_profile.py: the stage clock (inclusive and exclusive wall time, nesting, threads) and the table. No chip, no weights."""
import importlib.util
import sys
import threading
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "acestep"
spec = importlib.util.spec_from_file_location("acestep_profile", SCRIPTS / "acestep_profile.py")
prof = importlib.util.module_from_spec(spec)
sys.modules["acestep_profile"] = prof
spec.loader.exec_module(prof)


class Thing:
    def inner(self, s):
        time.sleep(s)
        return s

    def outer(self, s):
        time.sleep(s)
        return self.inner(s)


def test_wrap_keeps_return_value_and_counts_calls():
    clock, t = prof.StageClock(), Thing()
    clock.wrap(t, "inner", "inner")
    assert t.inner(0.01) == 0.01 and t.inner(0.01) == 0.01
    rows = clock.rows()
    assert rows["inner"]["calls"] == 2 and rows["inner"]["inclusive_s"] >= 0.02


def test_exclusive_time_subtracts_children():
    clock, t = prof.StageClock(), Thing()
    clock.wrap(t, "inner", "inner")
    clock.wrap(t, "outer", "outer")
    t.outer(0.05)
    r = clock.rows()
    assert r["outer"]["inclusive_s"] >= 0.1 - 1e-3
    # the outer call slept 0.05 itself and 0.05 inside the inner call
    assert 0.04 <= r["outer"]["exclusive_s"] <= 0.08
    assert abs(r["outer"]["inclusive_s"] - r["outer"]["exclusive_s"] - r["inner"]["inclusive_s"]) < 1e-6


def test_a_raising_call_still_records_and_pops_the_stack():
    clock = prof.StageClock()

    class Boom:
        def go(self):
            raise ValueError("x")

    b = Boom()
    clock.wrap(b, "go", "go")
    try:
        b.go()
    except ValueError:
        pass
    assert clock.rows()["go"]["calls"] == 1
    assert clock.depth() == 0


def test_threads_have_their_own_stacks():
    clock, t = prof.StageClock(), Thing()
    clock.wrap(t, "inner", "inner")
    clock.wrap(t, "outer", "outer")
    th = threading.Thread(target=t.inner, args=(0.05,))
    th.start()
    t.outer(0.02)           # runs while the thread's inner call is open: must not take the thread's time as its child
    th.join()
    r = clock.rows()
    assert r["outer"]["exclusive_s"] >= 0.02 - 1e-3 and r["outer"]["exclusive_s"] < 0.045


def test_table_lists_every_stage_with_a_share_of_the_total():
    rows = {"a": {"calls": 1, "inclusive_s": 1.0, "exclusive_s": 0.75}, "b": {"calls": 8, "inclusive_s": 0.25, "exclusive_s": 0.25}}
    text = prof.table(rows, total_s=2.0)
    assert "a" in text and "b" in text and "37.5" in text and "12.5" in text


def test_wrap_keeps_the_attributes_of_the_wrapped_function():
    # acestep_fast puts a `reset` attribute on prepare_condition; a clock wrapped around it must not hide it
    def f():
        return 1

    f.reset = lambda: "reset"
    holder = type("H", (), {})()
    holder.f = f
    prof.StageClock().wrap(holder, "f", "f")
    assert holder.f.reset() == "reset" and holder.f() == 1
