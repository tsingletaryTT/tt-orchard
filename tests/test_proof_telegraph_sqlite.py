"""Proof only (the product does not use it): a SQLite reservation ledger keeps the accounting rules of orchard.telegraph.

Nothing here is used by the product. The tests check the claims in
docs/superpowers/specs/2026-10-28-project-database-design.md:

  * the same totals and the same budget decisions as the jsonl ledger, for the scenarios of tests/test_telegraph.py
    (run through the real Telegraph) and for random operation sequences;
  * four processes making 200 reservations each cannot pass a cap and cannot spend the same allowance twice;
  * a process killed with SIGKILL between a reservation and its settlement leaves the reservation counted, and a process
    killed inside a transaction leaves nothing half written;
  * the file and its WAL files are private, and a damaged file is detected.
"""
import json
import multiprocessing
import os
import random
import signal
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pytest

import legacy_telegraph as tg                 # the jsonl implementation, frozen: the proof compares against it
from orchard._proofs import telegraph_sqlite as proof

REPO = Path(tg.__file__).resolve().parents[1]
KEY = "k3y-VALUE-0123456789"


# ----------------------------------------------------------------------------------------------- helpers

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


def telegraph(tmp_path, transport=None, **kw):
    cfg = tg.load(table(**kw), base=tmp_path)
    transport = transport or Wire()
    return tg.Telegraph(cfg, tmp_path / "tg", transport=transport, environ={"TG_KEY": KEY},
                        clock=lambda: 1000.0, hostname="node9", home="/nonexistent/someone"), transport


def raises(exc):
    def transport(url, headers, body, timeout):
        raise exc
    return transport


PRICES = {"small": [1.0, 3.0], "big": [5.0, 15.0]}
NO_USAGE = {"choices": [{"message": {"content": "fine"}, "finish_reason": "stop"}]}


def assert_same_accounting(want: dict, got: dict):
    """The jsonl accounting and the SQLite accounting agree: integers exactly, dollars to a billionth."""
    assert (got["calls"], got["tokens"], got["unknown_cost"]) == (want["calls"], want["tokens"], want["unknown_cost"])
    assert got["cost"] == pytest.approx(want["cost"], abs=1e-9)
    assert set(got["problems"]) == set(want["problems"])
    for name, w in want["problems"].items():
        g = got["problems"][name]
        assert (g["calls"], g["tokens"], g["unknown_cost"]) == (w["calls"], w["tokens"], w["unknown_cost"])
        assert g["cost"] == pytest.approx(w["cost"], abs=1e-9)


# ----------------------------------------------------------------------------------------------- scenarios of tests/test_telegraph.py

def s_plain(tmp_path):
    tel, _ = telegraph(tmp_path)
    tel.ask("p", "t", "f1")
    tel.ask("p", "t", "f2")


def s_timeout_token_cap(tmp_path):
    tel, _ = telegraph(tmp_path, transport=raises(TimeoutError("slow")), max_total_tokens=500, max_output_tokens=1000)
    tel.ask("p", "t", "f1")
    tel.ask("p", "t", "f2")


def s_unknown_usage_cost_cap(tmp_path):
    tel, _ = telegraph(tmp_path, transport=Wire(lambda b: NO_USAGE), max_cost_usd=0.5, max_output_tokens=100,
                       prices=PRICES)
    tel.ask("p", "t", "f1")
    tel.ask("p", "t", "f2")


def s_rejected_before_processing(tmp_path):
    tel, wire = telegraph(tmp_path, transport=Wire(status=401), max_cost_usd=0.005, max_output_tokens=1000, prices=PRICES)
    tel.ask("p", "t", "f1")
    wire.status = 200
    tel.ask("p", "t", "f2")


def s_process_dies(tmp_path):
    tel, _ = telegraph(tmp_path, transport=raises(SystemExit(1)), max_calls=1)
    with pytest.raises(SystemExit):
        tel.ask("p", "t", "f1")
    tel2, _ = telegraph(tmp_path, max_calls=1)
    tel2.ask("p", "t", "f2")                                 # refused: the reservation counts


def s_torn_tail(tmp_path):
    tel, _ = telegraph(tmp_path)
    tel.ask("p", "t", "f1")
    with open(tmp_path / "tg" / "telegraph.jsonl", "a") as f:
        f.write('{"call_id": "half')
    tel.ask("p", "t", "f2")                                  # quarantines the fragment: one call of unknown cost


def s_legacy_unpriced_record(tmp_path):
    (tmp_path / "tg").mkdir()
    old = {"call_id": "old1", "problem": "p", "model": "small", "outcome": "ok", "cached": False, "tokens_in": 9000,
           "tokens_out": 900, "cost_usd": None, "usage_known": True, "ts": 1.0, "latency_s": 1.0}
    (tmp_path / "tg" / "telegraph.jsonl").write_text(json.dumps(old) + "\n")
    tel, _ = telegraph(tmp_path, max_cost_usd=100.0, prices=PRICES)
    tel.ask("other", "t", "f")                               # refused: an earlier call has no known cost


def s_per_problem_caps(tmp_path):
    tel, _ = telegraph(tmp_path, max_calls_per_problem=2, max_tokens_per_problem=400)
    for i in range(3):
        tel.ask("p", "t", f"f{i}")
    tel.ask("q", "t", "f")


def s_cache_hit_and_reports(tmp_path):
    tel, _ = telegraph(tmp_path)
    first = tel.ask("p", "t", "f1")
    tel.ask("p", "t", "f1")                                  # a cache hit: no request, and it is not counted
    tel.report(first.call_id, False)
    tel.ask("p", "t", "f2")                                  # the next model up


def s_unreadable_answer(tmp_path):
    tel, _ = telegraph(tmp_path, transport=Wire(lambda b: b"not json at all"), max_cost_usd=0.5, max_output_tokens=100,
                       prices=PRICES)
    tel.ask("p", "t", "f1")
    tel.ask("p", "t", "f2")


SCENARIOS = [s_plain, s_timeout_token_cap, s_unknown_usage_cost_cap, s_rejected_before_processing, s_process_dies,
             s_torn_tail, s_legacy_unpriced_record, s_per_problem_caps, s_cache_hit_and_reports, s_unreadable_answer]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.__name__)
def test_the_sqlite_ledger_gets_the_same_totals_as_the_jsonl_ledger_for_the_telegraph_scenarios(tmp_path, scenario):
    scenario(tmp_path)
    recs = tg.read_ledger(tmp_path / "tg")
    assert recs, "the scenario left no ledger"
    want = tg.accounting(recs)
    with proof.Ledger(tmp_path / "db" / "project.db") as db:
        db.import_records(recs)
        assert_same_accounting(want, db.accounting())
        summary = tg.summarise_records(recs)
        totals = db.totals()
        assert (totals["calls"], totals["tokens_in"], totals["tokens_out"]) == \
               (summary["calls"], summary["tokens_in"], summary["tokens_out"])
        assert totals["cost_usd"] == pytest.approx(summary["cost_usd"], abs=1e-9)


def test_a_boolean_cost_is_unknown_and_not_one_dollar(tmp_path):
    """A JSON true is not a number in telegraph, so it must not be a cost of 1.0 here (SQLite would store it as 1)."""
    rec = {"call_id": "old1", "problem": "p", "model": "small", "outcome": "ok", "cached": False, "tokens_in": 5,
           "tokens_out": 5, "cost_usd": True}
    with proof.Ledger(tmp_path / "project.db") as db:
        db.import_records([rec])
        got = db.accounting()
    assert_same_accounting(tg.accounting([rec]), got)
    assert got["unknown_cost"] == 1 and got["cost"] == 0


def test_the_scenarios_include_a_refusal_a_reservation_that_was_never_settled_and_a_torn_record(tmp_path):
    """The scenario set must reach the corners, or the totals above could agree by both being zero."""
    kinds = set()
    for scenario in SCENARIOS:
        root = tmp_path / scenario.__name__
        root.mkdir()
        scenario(root)
        for r in tg.read_ledger(root / "tg"):
            kinds.add("torn" if "torn" in r else "reserve" if "reserve" in r else "report" if "report" in r else
                      r.get("outcome") or "other")
            if r.get("cached"):
                kinds.add("cached")
        recs = tg.read_ledger(root / "tg")
        if tg.summarise_records(recs)["unsettled"]:
            kinds.add("unsettled")
    assert {"torn", "reserve", "report", "ok", "error", "budget", "cached", "unsettled"} <= kinds


# ----------------------------------------------------------------------------------------------- the same decisions

def jsonl_decision(cfg, recs, model, worst, problem):
    stub = types.SimpleNamespace(cfg=cfg)
    return tg.Telegraph._over_budget(stub, model, worst, tg.accounting(recs), problem)


@pytest.mark.parametrize("seed", range(40))
def test_random_operations_give_the_same_decisions_and_the_same_accounting(tmp_path, seed):
    rng = random.Random(seed)
    caps = dict(max_calls=rng.randint(3, 12), max_total_tokens=rng.randint(2000, 9000),
                max_calls_per_problem=rng.randint(1, 5), prices=PRICES)
    if rng.random() < 0.6:
        caps["max_cost_usd"] = rng.choice([0.5, 1.0, 2.0])
    if rng.random() < 0.4:
        caps["max_cost_per_problem_usd"] = rng.choice([0.25, 0.5, 1.0])
    if rng.random() < 0.4:
        caps["max_tokens_per_problem"] = rng.randint(800, 3000)
    cfg = tg.load(table(**caps), base=tmp_path)
    recs, pending = [], []
    with proof.Ledger(tmp_path / f"s{seed}.db") as db:
        for step in range(70):
            op = rng.choices(["reserve", "settle", "torn", "legacy"], [6, 5, 0.5, 0.5])[0]
            if op == "reserve":
                problem, model = rng.choice("pqr"), rng.choice(["small", "big"])
                cost = rng.choice([None, 0.0, 0.0625, 0.125, 0.25, 0.5])      # multiples of 1/16: sums are exact
                worst = {"tokens": rng.randint(100, 1500), "cost": cost}
                want = jsonl_decision(cfg, recs, model, worst, problem)
                call_id = f"c{step:03d}"
                got = db.reserve(call_id, problem, model, worst["tokens"], worst["cost"], cfg)
                assert got == want, (seed, step, got, want)
                if not want:
                    recs.append({"reserve": call_id, "problem": problem, "model": model, "tokens": worst["tokens"],
                                 "cost_usd": worst["cost"]})
                    pending.append(call_id)
            elif op == "settle" and pending:
                call_id = pending.pop(rng.randrange(len(pending)))
                res = next(r for r in recs if r.get("reserve") == call_id)
                outcome = rng.choice(["ok", "ok", "error"])
                known = rng.choice([True, True, False])
                tin, tout = (rng.randint(1, 400), rng.randint(1, 300)) if known else (0, 0)
                cost = rng.choice([None, 0.0, 0.0625, 0.125]) if known else rng.choice([None, 0.0])
                rec = {"call_id": call_id, "problem": res["problem"], "model": res["model"], "outcome": outcome,
                       "cached": False, "tokens_in": tin, "tokens_out": tout, "cost_usd": cost, "usage_known": known}
                recs.append(rec)
                db.settle(call_id, problem=res["problem"], model=res["model"], outcome=outcome, tokens_in=tin,
                          tokens_out=tout, cost_usd=cost, usage_known=known)
            elif op == "torn":
                recs.append({"torn": f"telegraph.jsonl.torn-{step}", "tokens": 1500})
                db.record_torn(f"telegraph.jsonl.torn-{step}", 1500)
            elif op == "legacy":
                rec = {"call_id": f"old{step}", "problem": rng.choice("pq"), "model": "small", "outcome": "ok",
                       "cached": False, "tokens_in": 50, "tokens_out": 20, "cost_usd": rng.choice([None, 0.0625])}
                recs.append(rec)
                db.import_records([rec])
            assert_same_accounting(tg.accounting(recs), db.accounting())


def test_a_probe_after_each_scenario_is_decided_the_same_way(tmp_path):
    for scenario in SCENARIOS:
        root = tmp_path / scenario.__name__
        root.mkdir()
        scenario(root)
        recs = tg.read_ledger(root / "tg")
        for caps in ({"max_calls": 1}, {"max_total_tokens": 500}, {"max_cost_usd": 0.01, "prices": PRICES},
                     {"max_calls_per_problem": 1}, {"max_cost_per_problem_usd": 0.001, "prices": PRICES}, {}):
            cfg = tg.load(table(**caps), base=root)
            with proof.Ledger(root / "probe.db") as db:
                db.import_records(recs)
                for problem in ("p", "q", "other"):
                    for worst in ({"tokens": 700, "cost": 0.004}, {"tokens": 10, "cost": None}):
                        got = db.would_refuse(cfg, "small", worst["tokens"], worst["cost"], problem)
                        assert got == jsonl_decision(cfg, recs, "small", worst, problem), (scenario.__name__, caps, problem)
            (root / "probe.db").unlink()
            for suffix in ("-wal", "-shm"):
                (root / f"probe.db{suffix}").unlink(missing_ok=True)


# ----------------------------------------------------------------------------------------------- the real backfilled shape

def backfilled_shape(path: Path, n: int = 40):
    """A ledger shaped like the backfilled one: settled calls with no reservations, then one verdict for most of them."""
    rows = []
    for i in range(n):
        model = ("sonnet-5-5", "haiku-5-5", "qwen3-32b")[i % 3]
        rows.append({"call_id": f"b{i:02d}", "problem": f"prob{i % 7}", "model": model, "outcome": "ok", "cached": False,
                     "tokens_in": 20000 + 1000 * i, "tokens_out": 1500 + 37 * i, "latency_s": 3.0 + i / 10,
                     "cost_usd": 0.0 if model.startswith("qwen") else round((20000 + 1000 * i) * 3e-6 / 7 + 0.001, 9),
                     "usage_known": True, "ts": 1000.0 + i, "prompt_chars": 100, "prompt_sha256": "0" * 64,
                     "reply_file": f"replies/b{i:02d}.txt"})
    for i in range(0, n, 1):
        if i % 8 != 7:
            rows.append({"report": f"b{i:02d}", "helped": i % 3 != 0, "ts": 2000.0 + i, "note": ""})
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return rows


def test_import_check_passes_on_a_ledger_shaped_like_the_backfilled_one(tmp_path):
    src = tmp_path / "tg"
    src.mkdir()
    rows = backfilled_shape(src / "telegraph.jsonl")
    with proof.Ledger(tmp_path / "project.db") as db:
        result = proof.import_and_compare(src, db)
    assert result["equal"], result
    assert result["jsonl"]["calls"] == 40 and result["records_imported"] == len(rows)
    assert result["jsonl"] == result["sqlite"]


def test_import_check_fails_when_one_call_is_changed_after_the_import(tmp_path):
    """The check itself must be able to fail."""
    src = tmp_path / "tg"
    src.mkdir()
    backfilled_shape(src / "telegraph.jsonl")
    with proof.Ledger(tmp_path / "project.db") as db:
        proof.import_and_compare(src, db)
        db._conn.execute("UPDATE calls SET tokens_in = tokens_in + 1 WHERE call_id = 'b03'")
        result = proof.compare(src, db)
    assert not result["equal"]


def test_every_line_is_kept_verbatim_so_the_export_equals_the_ledger(tmp_path):
    src = tmp_path / "tg"
    src.mkdir()
    backfilled_shape(src / "telegraph.jsonl")
    with proof.Ledger(tmp_path / "project.db") as db:
        proof.import_and_compare(src, db)
        exported = list(db.export_records())
    original = [json.loads(line) for line in (src / "telegraph.jsonl").read_text().splitlines()]
    assert exported == original


# ----------------------------------------------------------------------------------------------- files and pragmas

def test_the_file_its_directory_and_the_wal_files_are_private(tmp_path):
    path = tmp_path / "state" / "project.db"
    with proof.Ledger(path) as db:
        db.reserve("a", "p", "small", 100, 0.1, tg.load(table(), base=tmp_path))
        files = [path, Path(str(path) + "-wal"), Path(str(path) + "-shm")]
        assert all(f.exists() for f in files)
        assert all(f.stat().st_mode & 0o777 == 0o600 for f in files), [oct(f.stat().st_mode) for f in files]
        assert path.parent.stat().st_mode & 0o777 == 0o700


def test_an_existing_file_with_loose_permissions_is_made_private(tmp_path):
    path = tmp_path / "project.db"
    sqlite3.connect(path).close()
    path.chmod(0o644)
    with proof.Ledger(path):
        assert path.stat().st_mode & 0o777 == 0o600


def test_the_pragmas_are_set(tmp_path):
    with proof.Ledger(tmp_path / "project.db") as db:
        got = db.pragmas()
    assert got["journal_mode"] == "wal" and got["synchronous"] == 2 and got["foreign_keys"] == 1
    assert got["busy_timeout"] >= 5000


def test_a_newer_schema_version_is_refused_and_an_older_one_is_migrated(tmp_path):
    path = tmp_path / "project.db"
    with proof.Ledger(path) as db:
        assert db.user_version() == proof.SCHEMA_VERSION
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {proof.SCHEMA_VERSION + 1}")
    conn.close()
    with pytest.raises(proof.DatabaseTooNew):
        proof.Ledger(path)


def test_a_damaged_file_is_detected_and_moved_aside_and_a_new_one_is_started(tmp_path):
    """Page 1 stays intact, so the file opens and runs its pragmas; only the integrity check can find the damage."""
    path = tmp_path / "project.db"
    cfg = proof.Limits()
    with proof.Ledger(path) as db:
        for i in range(400):
            db.reserve(f"call-{i}", "p", "small", 100, 0.1, cfg)
    for suffix in ("-wal", "-shm"):
        assert not Path(str(path) + suffix).exists()          # the last close folded the log into the file
    data = bytearray(path.read_bytes())
    assert len(data) > 4096 * 4
    for i in range(4096, len(data), 4096):                    # damage the inside of every page but the first
        data[i + 20:i + 400] = os.urandom(380)
    path.write_bytes(bytes(data))
    with proof.Ledger(path) as db:                            # opens by moving the bad file aside
        assert db.accounting()["calls"] == 0
        assert db.moved_aside is not None and db.moved_aside.exists()
        assert db.integrity_ok()


# ----------------------------------------------------------------------------------------------- many processes

CASES = [
    # (name, caps, problems, expected number of reservations that must succeed)
    ("global calls", dict(max_calls=150, max_calls_per_problem=1000, max_total_tokens=10**9), ["p0", "p1", "p2"], 150),
    ("per problem", dict(max_calls=10**6, max_calls_per_problem=60, max_total_tokens=10**9), ["p0", "p1", "p2"], 180),
    ("both", dict(max_calls=150, max_calls_per_problem=60, max_total_tokens=10**9), ["p0", "p1", "p2"], 150),
    ("tokens", dict(max_calls=10**6, max_calls_per_problem=10**6, max_total_tokens=20000), ["p0", "p1"], 20),
    ("cost", dict(max_calls=10**6, max_calls_per_problem=10**6, max_total_tokens=10**9, max_cost_usd=10.0,
                  prices=PRICES), ["p0", "p1"], 40),
]


@pytest.mark.parametrize("name,caps,problems,expected", CASES, ids=[c[0] for c in CASES])
def test_four_processes_making_200_reservations_each_cannot_pass_a_cap(tmp_path, name, caps, problems, expected):
    cfg_table = table(**caps)
    ctx = multiprocessing.get_context("spawn")
    path = tmp_path / "state" / "project.db"
    with proof.Ledger(path):
        pass                                                   # create the file once
    args = [(str(path), cfg_table, str(tmp_path), i, 200, problems, 1000, 0.25) for i in range(4)]
    with ctx.Pool(4) as pool:
        results = pool.starmap(proof.worker, args)
    granted = [cid for r in results for cid, ok in r if ok]
    assert len(granted) == expected, (name, len(granted))
    assert len(set(granted)) == len(granted)                   # nothing granted twice
    with proof.Ledger(path) as db:
        acct = db.accounting()
        assert acct["calls"] == expected
        per = {p: acct["problems"].get(p, {"calls": 0})["calls"] for p in problems}
        if "max_calls_per_problem" in caps and caps["max_calls_per_problem"] < 1000:
            assert all(v <= caps["max_calls_per_problem"] for v in per.values())
        assert sum(per.values()) == expected
        assert db.integrity_ok()


def test_four_processes_that_also_settle_leave_one_row_per_call_and_the_same_totals(tmp_path):
    cfg_table = table(max_calls=10**6, max_calls_per_problem=10**6, max_total_tokens=10**9)
    ctx = multiprocessing.get_context("spawn")
    path = tmp_path / "project.db"
    with proof.Ledger(path):
        pass
    args = [(str(path), cfg_table, str(tmp_path), i, 200, ["a", "b"], 1000, 0.25, True) for i in range(4)]
    with ctx.Pool(4) as pool:
        results = pool.starmap(proof.worker, args)
    granted = [cid for r in results for cid, ok in r if ok]
    assert len(granted) == 800
    with proof.Ledger(path) as db:
        acct = db.accounting()
        assert acct["calls"] == 800
        assert acct["tokens"] == 800 * 100                     # each was settled at 60 in + 40 out
        assert db.count_rows() == 800


# ----------------------------------------------------------------------------------------------- kill tests

def child(args, tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    return subprocess.Popen([sys.executable, "-m", "orchard._proofs.telegraph_sqlite", *args], cwd=tmp_path,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


def read_until(proc, word):
    line = proc.stdout.readline()
    assert line.startswith(word), (line, proc.stderr.read() if proc.poll() is not None else "")
    return line


def test_a_writer_killed_between_reserve_and_settle_leaves_the_reservation_counted(tmp_path):
    path = tmp_path / "project.db"
    cfg = tg.load(table(max_calls=1, max_total_tokens=10**9), base=tmp_path)
    proc = child(["hold-after-reserve", str(path), "call-1", "p", "1500", "0.5"], tmp_path)
    try:
        read_until(proc, "RESERVED")
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        assert proc.returncode == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait()
    with proof.Ledger(path) as db:
        acct = db.accounting()
        assert (acct["calls"], acct["tokens"], acct["cost"]) == (1, 1500, 0.5)       # counted at its worst case
        assert db.unsettled() == ["call-1"]
        assert db.reserve("call-2", "p", "small", 10, 0.0, cfg) is not None
        assert "call cap" in db.reserve("call-3", "p", "small", 10, 0.0, cfg)        # the cap sees the dead writer's call
        assert db.integrity_ok()


def test_a_writer_killed_inside_a_transaction_leaves_nothing_and_no_lock(tmp_path):
    path = tmp_path / "project.db"
    cfg = tg.load(table(max_calls=5), base=tmp_path)
    proc = child(["hold-in-transaction", str(path), "call-1", "p", "1500", "0.5"], tmp_path)
    try:
        read_until(proc, "INSIDE")
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait()
    with proof.Ledger(path, busy_timeout_ms=1000) as db:        # a stale lock would fail here after one second
        assert db.accounting()["calls"] == 0 and db.count_rows() == 0
        assert db.reserve("call-2", "p", "small", 10, 0.0, cfg) is None
        assert db.integrity_ok()


def test_a_writer_killed_at_random_points_never_leaves_a_half_written_call(tmp_path):
    path = tmp_path / "project.db"
    seen_reserved, seen_settled = set(), set()
    for round_ in range(6):
        proc = child(["storm", str(path), f"r{round_}", "3"], tmp_path)
        try:
            for _ in range(40 + 17 * round_):                   # the cut falls at a different point each round
                line = proc.stdout.readline().split()
                if not line:
                    break
                (seen_reserved if line[0] == "R" else seen_settled).add(line[1])
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                os.kill(proc.pid, signal.SIGKILL)
                proc.wait()
    assert len(seen_reserved) > 50 and seen_settled
    with proof.Ledger(path) as db:
        rows = db.rows()
        ids = {r["call_id"] for r in rows}
        assert seen_reserved <= ids                              # everything reported committed is on disk
        assert {r["call_id"] for r in rows if r["settled"]} >= seen_settled
        assert all(r["has_reserve"] or r["settled"] for r in rows)
        expect_tokens = sum((r["tokens_in"] + r["tokens_out"]) if r["settled"] and r["usage_known"] != 0
                            else r["reserved_tokens"] for r in rows)
        acct = db.accounting()
        assert acct["tokens"] == expect_tokens and acct["calls"] == len(rows)
        assert db.integrity_ok()


# ----------------------------------------------------------------------------------------------- no credential, no SQL outside

def test_a_reservation_row_holds_no_text_that_is_not_an_id_a_name_or_a_number(tmp_path):
    with proof.Ledger(tmp_path / "project.db") as db:
        db.reserve("abc123", "problem", "small", 100, 0.1, tg.load(table(), base=tmp_path))
        rows = db.rows()
    assert set(rows[0]) >= {"call_id", "rproblem", "rmodel", "reserved_tokens", "reserved_cost"}
    assert all(isinstance(v, (int, float, str, type(None))) for v in rows[0].values())
