"""The telegraph ledger on the project database (orchard/telegraph_store.py): the budget rules are the rules of the jsonl ledger,
processes cannot pass a cap together, a killed process leaves nothing half written, and the files it reads are hostile input.

`legacy_telegraph` is the frozen jsonl implementation (tests/legacy_telegraph.py). The "twin" tests run the same operations
through it and through the product and compare every answer and every record.
"""
import json
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import legacy_telegraph as old
from dbtools import CHILD, REPO, KILLED, make_history, reserve, run_child, settle, store_worker
from ledger_io import cache_rows, database_bytes, ledger_lines, ledger_records, seed_ledger, tree_text
from orchard import db as dbm
from orchard import telegraph as tg
from orchard import telegraph_store as ts

KEY = "k3y-VALUE-0123456789"
PRICES = {"small": [1.0, 3.0], "big": [5.0, 15.0]}
NO_USAGE = {"choices": [{"message": {"content": "fine"}, "finish_reason": "stop"}]}


class Wire:
    def __init__(self, reply=None, status=200):
        self.requests, self.status = [], status
        self.reply = reply or (lambda body: openai_reply("use ttl.block.fill for constants", 120, 30))

    def __call__(self, url, headers, body, timeout):
        self.requests.append(json.loads(body))
        out = self.reply(self.requests[-1])
        return self.status, (out if isinstance(out, bytes) else json.dumps(out).encode())


def openai_reply(text, tin, tout):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": tin, "completion_tokens": tout}}


def table(**kw):
    base = {"enabled": True, "endpoint": "https://llm.example.org/v1", "models": ["small", "big"], "key_env": "TG_KEY"}
    base.update(kw)
    return base


def raises(exc):
    def transport(url, headers, body, timeout):
        raise exc
    return transport


def build(module, directory, base, transport=None, **kw):
    cfg = module.load(table(**kw), base=base)
    return module.Telegraph(cfg, directory, transport=transport or Wire(), environ={"TG_KEY": KEY}, clock=lambda: 1000.0,
                            hostname="node9", home="/nonexistent/someone")


# =============================================================================================== the twin: same answers, same records

def normalise(records):
    """The records with each call id replaced by the order it first appears in (ids are random), and the reply file names with it."""
    names = {}

    def swap(match):
        return names.setdefault(match.group(0), f"id{len(names)}")
    text = json.dumps(records, sort_keys=True)
    ids = set(re.findall(r'"(?:call_id|reserve|origin|report)": "([0-9a-f]{12})"', text))
    for found in sorted(ids, key=lambda i: text.index(i)):
        names.setdefault(found, f"id{len(names)}")
    for found, name in names.items():
        text = text.replace(found, name)
    return json.loads(text)


def answers_of(got):
    return [(a.ok, a.reason, a.model, a.cached, a.tokens_in, a.tokens_out, a.cost_usd, a.text) for a in got]


def s_plain(m, d, base):
    t = build(m, d, base)
    return [t.ask("p", "t", "f1"), t.ask("p", "t", "f2")]


def s_timeout_token_cap(m, d, base):
    t = build(m, d, base, transport=raises(TimeoutError("slow")), max_total_tokens=500, max_output_tokens=1000)
    return [t.ask("p", "t", "f1"), t.ask("p", "t", "f2")]


def s_unknown_usage_cost_cap(m, d, base):
    t = build(m, d, base, transport=Wire(lambda b: NO_USAGE), max_cost_usd=0.5, max_output_tokens=100, prices=PRICES)
    return [t.ask("p", "t", "f1"), t.ask("p", "t", "f2")]


def s_rejected_before_processing(m, d, base):
    wire = Wire(status=401)
    t = build(m, d, base, transport=wire, max_cost_usd=0.005, max_output_tokens=1000, prices=PRICES)
    out = [t.ask("p", "t", "f1")]
    wire.status = 200
    return out + [t.ask("p", "t", "f2")]


def s_process_dies(m, d, base):
    t = build(m, d, base, transport=raises(SystemExit(1)), max_calls=1)
    with pytest.raises(SystemExit):
        t.ask("p", "t", "f1")
    return [build(m, d, base, max_calls=1).ask("p", "t", "f2")]


def s_unpriced_legacy_record(m, d, base):
    record = {"call_id": "old1", "problem": "p", "model": "small", "outcome": "ok", "cached": False, "tokens_in": 9000,
              "tokens_out": 900, "cost_usd": None, "usage_known": True, "ts": 1.0, "latency_s": 1.0}
    if m is old:
        Path(d).mkdir(parents=True)
        (Path(d) / "telegraph.jsonl").write_text(json.dumps(record) + "\n")
    else:
        seed_ledger(d, [record])
    return [build(m, d, base, max_cost_usd=100.0, prices=PRICES).ask("other", "t", "f")]


def s_per_problem_caps(m, d, base):
    t = build(m, d, base, max_calls_per_problem=2, max_tokens_per_problem=400)
    return [t.ask("p", "t", f"f{i}") for i in range(3)] + [t.ask("q", "t", "f")]


def s_cache_hit_reports_and_the_ladder(m, d, base):
    t = build(m, d, base)
    first = t.ask("p", "t", "f1")
    again = t.ask("p", "t", "f1")
    t.report(first.call_id, False)
    return [first, again, t.ask("p", "t", "f2")]


def s_unreadable_answer(m, d, base):
    t = build(m, d, base, transport=Wire(lambda b: b"not json at all"), max_cost_usd=0.5, max_output_tokens=100, prices=PRICES)
    return [t.ask("p", "t", "f1"), t.ask("p", "t", "f2")]


def s_no_price_under_a_cost_cap(m, d, base):
    return [build(m, d, base, max_cost_usd=1.0, prices={"big": [5.0, 15.0]}).ask("p", "t", "f")]


def s_per_problem_cost_cap(m, d, base):
    t = build(m, d, base, max_cost_per_problem_usd=0.0005, max_output_tokens=100, prices=PRICES)
    return [t.ask("p", "t", "f1"), t.ask("q", "t", "f1")]


def s_many_calls_one_cap(m, d, base):
    t = build(m, d, base, max_calls=4, max_calls_per_problem=9)
    return [t.ask(f"p{i % 3}", "t", f"f{i}") for i in range(7)]


SCENARIOS = [s_plain, s_timeout_token_cap, s_unknown_usage_cost_cap, s_rejected_before_processing, s_process_dies,
             s_unpriced_legacy_record, s_per_problem_caps, s_cache_hit_reports_and_the_ladder, s_unreadable_answer,
             s_no_price_under_a_cost_cap, s_per_problem_cost_cap, s_many_calls_one_cap]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.__name__)
def test_the_product_gives_the_same_answers_and_the_same_records_as_the_jsonl_ledger(tmp_path, scenario):
    legacy_dir, new_dir = tmp_path / "old" / "tg", tmp_path / "new" / "tg"
    want = scenario(old, legacy_dir, tmp_path / "old")
    got = scenario(tg, new_dir, tmp_path / "new")
    assert answers_of(got) == answers_of(want)
    legacy_records = [json.loads(line) for line in (legacy_dir / "telegraph.jsonl").read_text().splitlines()]
    assert normalise(ledger_records(new_dir)) == normalise(legacy_records)
    assert tg.accounting(ledger_records(new_dir)) == old.accounting(legacy_records)


def test_the_scenarios_reach_a_refusal_an_unsettled_reservation_a_cache_hit_and_an_error(tmp_path):
    """The twin tests could agree by both being empty; this checks the scenario set reaches the corners."""
    kinds = set()
    for scenario in SCENARIOS:
        d = tmp_path / scenario.__name__ / "tg"
        scenario(tg, d, tmp_path / scenario.__name__)
        recs = ledger_records(d)
        for r in recs:
            kinds.add("reserve" if "reserve" in r else "report" if "report" in r else r.get("outcome") or "other")
            if r.get("cached"):
                kinds.add("cached")
        if tg.summarise_records(recs)["unsettled"]:
            kinds.add("unsettled")
    assert {"reserve", "report", "ok", "error", "budget", "cached", "unsettled"} <= kinds


# =============================================================================================== I1: scope

def test_i1_a_cached_answer_is_served_only_to_its_own_scope(tmp_path):
    with ts.TelegraphStore.open(tmp_path / "s", scope="run-a") as a:
        settle_with_cache(a, "a1", "identity-1")
        assert a.cache_get("identity-1")["call_id"] == "a1"
    for other in ("run-b", "default"):
        with ts.TelegraphStore.open(tmp_path / "s", scope=other) as b:
            assert b.cache_get("identity-1") is None


def settle_with_cache(store, call_id, identity):
    reserve(store, call_id)
    store.settle({"call_id": call_id, "problem": "p", "model": "small", "outcome": "ok", "cached": False, "tokens_in": 1,
                  "tokens_out": 1, "cost_usd": 0.0, "usage_known": True, "reply_file": f"replies/{call_id}.txt"},
                 cache=(identity, f"replies/{call_id}.txt", f"text of {call_id}", call_id))


def test_i1_the_same_request_in_two_scopes_keeps_two_cache_rows(tmp_path):
    for scope, cid in (("run-a", "a1"), ("run-b", "b1")):
        with ts.TelegraphStore.open(tmp_path / "s", scope=scope) as store:
            settle_with_cache(store, cid, "same-identity")
    for scope, cid in (("run-a", "a1"), ("run-b", "b1")):
        with ts.TelegraphStore.open(tmp_path / "s", scope=scope) as store:
            assert store.cache_get("same-identity") == {"call_id": cid, "reply_file": f"replies/{cid}.txt", "text": f"text of {cid}"}


def test_i1_records_caps_and_totals_are_counted_per_scope(tmp_path):
    cfg = tg.load(table(max_calls=1, max_calls_per_problem=9), base=tmp_path)
    tels = {s: tg.Telegraph(cfg, tmp_path / "s", transport=Wire(), environ={"TG_KEY": KEY}, scope=s) for s in ("a", "b", "c")}
    assert tels["a"].ask("p", "t", "f1").ok
    assert "call cap" in tels["a"].ask("p", "t", "f2").reason
    assert tels["b"].ask("p", "t", "f1").ok                           # another scope has its own allowance
    for scope, calls in (("a", 1), ("b", 1), ("c", 0)):
        assert tg.accounting(tg._read_ledger_strict(tmp_path / "s", scope))["calls"] == calls
    assert tels["a"].ask("p", "t", "f1").cached                       # its own answer, from its own cache row
    assert len(cache_rows(tmp_path / "s", "a")) == 1 and cache_rows(tmp_path / "s", "c") == []


def test_i1_every_lookup_names_the_scope(tmp_path):
    """The cache lookup and the record reads each carry the scope: a mutation that drops it would serve scope b's answer to a."""
    with ts.TelegraphStore.open(tmp_path / "s", scope="b") as b:
        settle_with_cache(b, "b1", "x")
    with ts.TelegraphStore.open(tmp_path / "s", scope="a") as a:
        assert a.cache_get("x") is None and a.records() == [] and a.export_lines() == []


# =============================================================================================== accounting: exactness (issue 153)

def test_totals_add_in_ledger_order_so_they_equal_the_jsonl_sums_exactly(tmp_path):
    """Issue 153: floating-point sums depend on the order of the additions. A sum taken in SQL is free to add in another order."""
    costs = [1e16, 1.0, 1.0]                                          # 1e16 + 1.0 + 1.0 is 1e16, and 1.0 + 1.0 + 1e16 is not
    records = [{"call_id": f"c{i}", "problem": f"p{i % 3}", "model": "small", "outcome": "ok", "cached": False,
                "tokens_in": 10, "tokens_out": 5, "cost_usd": c, "usage_known": True, "ts": float(i)} for i, c in enumerate(costs)]
    seed_ledger(tmp_path / "s", records)
    expected = 0.0
    for c in costs:
        expected += c
    got = tg.accounting(tg._read_ledger_strict(tmp_path / "s"))
    assert got["cost"] == expected                                     # exactly, with no rounding
    assert got == old.accounting(records)
    assert tg.summarise_records(tg._read_ledger_strict(tmp_path / "s")) == old.summarise_records(records)
    assert expected == 1e16 and sum(reversed(costs)) != expected      # another order gives another sum: ledger order is the rule


def test_the_accounting_on_a_realistic_backfilled_ledger_is_equal_before_and_after_the_database(tmp_path):
    rows = []
    for i in range(40):
        model = ("sonnet-5-5", "haiku-5-5", "qwen3-32b")[i % 3]
        rows.append({"call_id": f"b{i:02d}", "problem": f"prob{i % 7}-None-r{i % 2}", "model": model, "outcome": "ok", "cached": False,
                     "tokens_in": 20000 + 1000 * i, "tokens_out": 1500 + 37 * i, "latency_s": 3.0 + i / 10,
                     "cost_usd": 0.0 if model.startswith("qwen") else round((20000 + 1000 * i) * 3e-6 / 7 + 0.001234567, 9),
                     "usage_known": True, "ts": 1000.0 + i})
    seed_ledger(tmp_path / "s", rows)
    got = tg._read_ledger_strict(tmp_path / "s")
    assert got == rows
    assert tg.accounting(got) == old.accounting(rows) and tg.summarise_records(got) == old.summarise_records(rows)


def test_a_reservation_is_one_transaction_the_check_and_the_insert_together(tmp_path, monkeypatch):
    with ts.TelegraphStore.open(tmp_path / "state") as store:
        real, count = store.db.write, []
        monkeypatch.setattr(store.db, "write", lambda: (count.append(1), real())[1])
        assert reserve(store, "c1") is None
        assert len(count) == 1


# =============================================================================================== many processes

CASES = [
    ("global calls", dict(max_calls=150, max_calls_per_problem=1000, max_total_tokens=10**9), ["p0", "p1", "p2"], 150),
    ("per problem", dict(max_calls=10**6, max_calls_per_problem=60, max_total_tokens=10**9), ["p0", "p1", "p2"], 180),
    ("both", dict(max_calls=150, max_calls_per_problem=60, max_total_tokens=10**9), ["p0", "p1", "p2"], 150),
    ("tokens", dict(max_calls=10**6, max_calls_per_problem=10**6, max_total_tokens=20000), ["p0", "p1"], 20),
    ("cost", dict(max_calls=10**6, max_calls_per_problem=10**6, max_total_tokens=10**9, max_cost_usd=10.0, prices=PRICES),
     ["p0", "p1"], 40),
]


@pytest.mark.parametrize("name,caps,problems,expected", CASES, ids=[c[0] for c in CASES])
def test_four_processes_making_200_reservations_each_cannot_pass_a_cap(tmp_path, name, caps, problems, expected):
    directory = tmp_path / "state"
    with dbm.Database(directory):
        pass
    ctx = multiprocessing.get_context("spawn")
    args = [(str(directory), table(**caps), i, 200, problems, 1000, 0.25) for i in range(4)]
    with ctx.Pool(4) as pool:
        results = pool.starmap(store_worker, args)
    granted = [cid for r in results for cid, ok in r if ok]
    assert len(granted) == expected, (name, len(granted))
    assert len(set(granted)) == len(granted)                           # nothing granted twice
    acct = tg.accounting(tg._read_ledger_strict(directory))
    assert acct["calls"] == expected
    per = {p: acct["problems"].get(p, {"calls": 0})["calls"] for p in problems}
    if caps.get("max_calls_per_problem", 1000) < 1000:
        assert all(v <= caps["max_calls_per_problem"] for v in per.values())
    assert sum(per.values()) == expected
    with dbm.Database(directory) as db:
        assert db.integrity_ok()


def test_four_processes_that_also_settle_leave_one_settled_call_each_and_the_same_totals(tmp_path):
    directory = tmp_path / "state"
    with dbm.Database(directory):
        pass
    ctx = multiprocessing.get_context("spawn")
    table_ = table(max_calls=10**6, max_calls_per_problem=10**6, max_total_tokens=10**9)
    with ctx.Pool(4) as pool:
        results = pool.starmap(store_worker, [(str(directory), table_, i, 200, ["a", "b"], 1000, 0.25, True) for i in range(4)])
    assert len([1 for r in results for _cid, ok in r if ok]) == 800
    records = tg._read_ledger_strict(directory)
    acct = tg.accounting(records)
    assert acct["calls"] == 800 and acct["tokens"] == 800 * 100        # each was settled at 60 in and 40 out
    assert len([r for r in records if "reserve" in r]) == 800 == len([r for r in records if "call_id" in r])


# =============================================================================================== kills

def child(args):
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    return subprocess.Popen([sys.executable, str(CHILD), *args], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)


def read_until(proc, word):
    line = proc.stdout.readline()
    assert line.startswith(word), (line, proc.stderr.read() if proc.poll() is not None else "")
    return line


def test_a_writer_killed_between_reserve_and_settle_leaves_the_reservation_counted_at_its_worst_case(tmp_path):
    directory = tmp_path / "state"
    proc = child(["hold-after-reserve", str(directory), "-", "call-1"])
    try:
        read_until(proc, "RESERVED")
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
    cfg = tg.load(table(max_calls=1, max_total_tokens=10**9), base=tmp_path)
    t = tg.Telegraph(cfg, directory, transport=Wire(), environ={"TG_KEY": KEY})
    acct = tg.accounting(tg._read_ledger_strict(directory))
    assert (acct["calls"], acct["tokens"], acct["cost"]) == (1, 1500, 0.5)
    got = t.ask("p", "t", "f")
    assert not got.ok and "call cap" in got.reason
    with dbm.Database(directory) as db:
        assert db.integrity_ok()


def test_a_writer_killed_inside_the_transaction_leaves_nothing_and_no_stale_lock(tmp_path):
    directory = tmp_path / "state"
    proc = child(["hold-in-transaction", str(directory), "-", "call-1"])
    try:
        read_until(proc, "INSIDE")                                     # the INSERT ran, the COMMIT has not
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
    started = time.monotonic()
    with ts.TelegraphStore.open(directory, busy_timeout_ms=1000) as store:
        assert store.records() == []                                   # nothing remains of the killed write
        assert reserve(store, "after") is None                        # and a new writer is not blocked by it
    assert time.monotonic() - started < 3


def test_six_writers_killed_at_different_points_leave_every_reported_commit_and_no_half_row(tmp_path):
    directory = tmp_path / "state"
    with dbm.Database(directory):
        pass
    procs = [child(["storm", str(directory), "-", f"w{i}", str(3 + i)]) for i in range(6)]
    reported = {"R": set(), "S": set()}
    try:
        time.sleep(1.0)
        for i, proc in enumerate(procs):
            time.sleep(0.05 * i)
            proc.send_signal(signal.SIGKILL)
        for proc in procs:
            out, _err = proc.communicate(timeout=20)
            for line in out.splitlines():
                kind, _, call_id = line.partition(" ")
                if kind in reported and call_id:
                    reported[kind].add(call_id)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
    records = tg._read_ledger_strict(directory)
    reserved = {r["reserve"] for r in records if "reserve" in r}
    settled = {r["call_id"] for r in records if "call_id" in r}
    assert reported["R"] and reported["R"] <= reserved                 # every commit a child reported is in the file
    assert reported["S"] <= settled
    assert settled <= reserved                                         # a settlement never exists without its reservation
    with dbm.Database(directory) as db:
        assert db.integrity_ok()
    acct = tg.accounting(records)
    assert acct["calls"] == len(reserved)
    by_call = {r["call_id"]: r for r in records if "call_id" in r}
    expected = 0
    for r in records:
        if "reserve" in r:
            done = by_call.get(r["reserve"])
            expected += (done["tokens_in"] + done["tokens_out"]) if done and done.get("usage_known") is not False else r["tokens"]
    assert acct["tokens"] == expected


# =============================================================================================== the guards (I3, I10)

def old_directory(tmp_path):
    return_dir = tmp_path / "old" / "tg"
    t = build(old, return_dir, tmp_path / "old")
    t.ask("p", "t", "f1")
    return return_dir


def test_an_unmigrated_jsonl_ledger_refuses_the_call_and_names_the_command(tmp_path):
    d = old_directory(tmp_path)
    wire = Wire()
    got = build(tg, d, tmp_path, transport=wire).ask("p", "t", "f2")
    assert not got.ok and "db migrate telegraph" in got.reason and wire.requests == []
    assert not (d / "project.db").exists() or ledger_records(d) == []     # no second history was started beside the file


def test_an_unmigrated_jsonl_ledger_is_not_read_as_an_empty_history_by_a_page(tmp_path):
    d = old_directory(tmp_path)
    with pytest.raises(ts.NotMigrated):
        tg._read_ledger_strict(d)
    assert tg._read_ledger(d) == []                                    # the display reader shows nothing, never a guess


def test_a_migration_that_did_not_finish_refuses_the_call(tmp_path):
    d = tmp_path / "state"
    make_history(d, 1)
    with dbm.Database(d) as db:
        with db.write() as conn:
            db.meta_set(conn, "cutover:default", "imported")
    got = build(tg, d, tmp_path).ask("p", "t", "f")
    assert not got.ok and "did not finish" in got.reason


def test_a_marker_and_a_live_jsonl_together_refuse_the_directory(tmp_path):
    d = tmp_path / "state"
    make_history(d, 1)
    (d / "MIGRATED-TO-SQLITE").write_text("{}")
    (d / "telegraph.jsonl").write_text('{"call_id": "late"}\n')
    got = build(tg, d, tmp_path).ask("p", "t", "f")
    assert not got.ok and "old writer" in got.reason


def test_a_directory_with_the_migrated_marker_and_no_database_record_is_refused(tmp_path):
    d = tmp_path / "state"
    d.mkdir(mode=0o700)
    (d / "MIGRATED-TO-SQLITE").write_text("{}")
    got = build(tg, d, tmp_path).ask("p", "t", "f")
    assert not got.ok and "does not record it" in got.reason


# =============================================================================================== recovery states seen through telegraph

def test_after_a_recovery_every_ask_is_refused_until_a_person_confirms_and_cache_hits_still_work(tmp_path):
    d = tmp_path / "state"
    t = build(tg, d, tmp_path, max_calls=10)
    assert t.ask("p", "t", "f1").ok
    with dbm.Database(d) as db:
        db._write_marker("restored for the test")
        db._write_unreconciled("restored for the test")
    wire = Wire()
    blocked_t = build(tg, d, tmp_path, transport=wire, max_calls=10)
    got = blocked_t.ask("p", "t", "f2")
    assert not got.ok and "confirm" in got.reason and wire.requests == []
    assert blocked_t.ask("p", "t", "f1").cached                        # a cache hit sends nothing and spends nothing
    with dbm.Database(d) as db:
        db.confirm_reconciled("checked")
    assert blocked_t.ask("p", "t", "f2").ok


def test_charging_reply_files_with_no_record_adds_each_once_and_only_raises_the_counted_spend(tmp_path):
    d = tmp_path / "state"
    make_history(d, 2)
    with ts.TelegraphStore.open(d) as store:
        before = tg.accounting(store.records())
        assert store.add_recovered_reservations(["c0", "c1", "new1", "new2"], 1500, 0.5) == 2       # c0 and c1 have records already
        assert store.add_recovered_reservations(["new1", "new2"], 1500, 0.5) == 0                   # a second pass adds nothing
        after = tg.accounting(store.records())
    assert after["calls"] == before["calls"] + 2 and after["tokens"] == before["tokens"] + 3000 and after["cost"] >= before["cost"] + 1.0


# =============================================================================================== hostile input: the files the store reads

@pytest.mark.parametrize("damage", ["garbage", "empty-ledger-table-gone", "truncated"])
def test_a_damaged_database_file_is_never_read_as_an_empty_history_and_telegraph_refuses_or_recovers(tmp_path, damage):
    d = make_history(tmp_path / "state", 3)
    path = d / "project.db"
    data = path.read_bytes()
    if damage == "garbage":
        path.write_bytes(b"garbage" * 1000)
    elif damage == "truncated":
        path.write_bytes(data[: len(data) // 2 + 7])
    else:
        import sqlite3
        conn = sqlite3.connect(path)
        conn.execute("DROP TABLE telegraph_events")
        conn.close()
    for suffix in ("-wal", "-shm"):
        (d / ("project.db" + suffix)).unlink(missing_ok=True)
    wire = Wire()
    got = build(tg, d, tmp_path, transport=wire, max_calls=3).ask("p", "t", "f")
    assert not got.ok and wire.requests == []                          # the cap of 3 is not available again
    assert "confirm" in got.reason or "cannot be written or read" in got.reason


def test_a_page_reading_a_damaged_file_gets_an_error_and_does_not_move_or_change_it(tmp_path):
    d = make_history(tmp_path / "state", 3)
    (d / "project.db").write_bytes(b"garbage" * 1000)
    for suffix in ("-wal", "-shm"):
        (d / ("project.db" + suffix)).unlink(missing_ok=True)
    before = (d / "project.db").read_bytes()
    with pytest.raises(dbm.LedgerUnavailable):
        ts.read_snapshot(d)
    assert (d / "project.db").read_bytes() == before and not list(d.glob("*.corrupt-*"))


@pytest.mark.parametrize("kind", ["link", "directory", "fifo"])
def test_a_database_path_that_is_not_a_regular_file_is_refused_by_the_store_and_by_the_reader(tmp_path, kind):
    d = tmp_path / "state"
    d.mkdir(mode=0o700)
    target = d / "project.db"
    if kind == "link":
        outside = tmp_path / "outside.db"
        outside.write_bytes(b"")
        target.symlink_to(outside)
    elif kind == "directory":
        target.mkdir()
    else:
        os.mkfifo(target)
    results = []

    def attempt():
        results.append(build(tg, d, tmp_path).ask("p", "t", "f").reason)
        try:
            ts.read_snapshot(d)
        except dbm.LedgerRefused as exc:
            results.append(type(exc).__name__)
    t = threading.Thread(target=attempt, daemon=True)
    t.start()
    t.join(15)
    assert not t.is_alive(), "a special file made the store block"
    assert "cannot be written or read" in results[0] and results[1] == "UnsafeDirectory"
    assert not list(d.glob("*.corrupt-*"))


def test_a_stored_record_that_is_not_json_is_an_error_and_not_an_empty_history(tmp_path):
    import sqlite3
    d = make_history(tmp_path / "state", 2)
    conn = sqlite3.connect(d / "project.db")
    conn.execute("UPDATE telegraph_events SET raw_json = '{not json' WHERE seq = 1")
    conn.commit()
    conn.close()
    with pytest.raises(dbm.LedgerUnavailable):
        ts.read_snapshot(d)
    got = build(tg, d, tmp_path, max_calls=2).ask("p", "t", "f")
    assert not got.ok and "cannot be written or read" in got.reason


def test_a_record_with_a_value_json_cannot_carry_is_refused_not_stored(tmp_path):
    with ts.TelegraphStore.open(tmp_path / "state") as store:
        for bad in (float("nan"), float("inf")):
            with pytest.raises(dbm.LedgerUnavailable):
                store.append({"call_id": "x", "outcome": "ok", "cost_usd": bad})
        assert store.records() == []


# =============================================================================================== credentials never reach the database

SECRET = "sk-live-0123456789abcdefghij-SECRET"


def test_the_key_reaches_no_byte_of_the_database_its_wal_or_an_export_through_any_public_write_function(tmp_path):
    d = tmp_path / "state"
    wire = Wire(lambda b: openai_reply(f"the key is {SECRET} and Bearer abcdefghijklmnop", 10, 10))
    cfg = tg.load(table(), base=tmp_path)
    t = tg.Telegraph(cfg, d, transport=wire, environ={"TG_KEY": SECRET}, hostname="node9", home="/home/someone")
    got = t.ask(f"problem-{SECRET}", f"task {SECRET} /home/someone/x", f"failure {SECRET}")
    t.report(got.call_id, True, note=f"note {SECRET}")
    again = t.ask(f"problem-{SECRET}", f"task {SECRET} /home/someone/x", f"failure {SECRET}")      # the cache hit record
    assert got.ok and again.cached
    e = Wire(lambda b: b'{"error": "bad key ' + SECRET.encode() + b'"}', status=401)
    tg.Telegraph(cfg, d, transport=e, environ={"TG_KEY": SECRET}).ask(f"other-{SECRET}", "t", "f")   # an error record
    tg.Telegraph(cfg, d, transport=Wire(), environ={}).ask("nokey", "t", "f")                          # a no-key refusal
    exported = "\n".join(ledger_lines(d))
    for blob in (database_bytes(d).decode("latin-1"), tree_text(d), exported):
        assert SECRET not in blob and "abcdefghijklmnop" not in blob and "/home/someone" not in blob
    assert exported and "sk-live" not in exported


# =============================================================================================== one snapshot

def test_a_read_during_concurrent_writes_never_shows_a_settlement_without_its_reservation(tmp_path):
    d = tmp_path / "state"
    stop = threading.Event()
    errors = []

    def writer():
        with ts.TelegraphStore.open(d) as store:
            i = 0
            while not stop.is_set():
                reserve(store, f"w{i}")
                settle(store, f"w{i}")
                i += 1

    t = threading.Thread(target=writer)
    with dbm.Database(d):
        pass
    t.start()
    try:
        for _ in range(150):
            try:
                records = ts.read_snapshot(d).records
            except dbm.LedgerUnavailable as exc:
                errors.append(str(exc))
                continue
            reserved = {r["reserve"] for r in records if "reserve" in r}
            assert {r["call_id"] for r in records if "call_id" in r} <= reserved
    finally:
        stop.set()
        t.join(10)
    assert not errors


def test_the_page_builds_every_part_from_one_read_transaction(tmp_path, monkeypatch):
    from orchard import telegraph_view as view
    d = tmp_path / "state"
    make_history(d, 2)
    calls = []
    real = ts.read_snapshot
    monkeypatch.setattr(ts, "read_snapshot", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    view._cache.clear()
    facts = view._ledger_facts(d)
    assert facts["error"] is None and len(calls) == 1


# =============================================================================================== reading from the CLI shell of an agent

def test_a_second_open_of_the_same_directory_sees_what_the_first_committed_before_its_request(tmp_path):
    seen = {}

    def transport(url, headers, body, timeout):
        seen["during"] = [r for r in ledger_records(tmp_path / "state") if "reserve" in r]       # another process reading
        return 200, json.dumps(openai_reply("ok", 10, 10)).encode()
    cfg = tg.load(table(), base=tmp_path)
    assert tg.Telegraph(cfg, tmp_path / "state", transport=transport, environ={"TG_KEY": KEY}).ask("p", "t", "f").ok
    assert len(seen["during"]) == 1                                    # the reservation was committed before the send
