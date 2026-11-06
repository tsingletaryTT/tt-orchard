# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The helper drives stage 2 of a rehearsal, the best port is kept across attempts, and a stub is not answered from the helper's cache
(orchard/repair_help.py, Supervisor._port_repair_rounds, orchard/telegraph.py). Run12 of the tiny Qwen4 rehearsal (2026-11-04) is the model:
the port reached `prefill` on the chips, the attempt ended, and the next attempt started again from the skeleton; the coder looped after
every placed file; and a stub came back from the cache at no cost a second time.

The chips, the coder and the helper's endpoint are fakes, as in tests/test_supervisor_telegraph_in_repair.py."""
import hashlib
import json
import os

import pytest

from orchard import phone_a_friend as paf
from orchard import port_facts, port_repair, repair_help as rh
from orchard import supervisor as sv
from orchard import telegraph as tg
from orchard.supervisor import EXIT_ABORTED, Control
from port_check_fakes import HONEST
from run_fakes import bringup, where
from test_supervisor import Rig
from test_supervisor_full_port import PREPARE, make, with_snapshot
from test_supervisor_port_repair import BROKEN, broken, repairs
from test_supervisor_telegraph_in_repair import PREFILL_BROKEN, STUB, decs, port_text, stage_ends, wire_helper
from test_telegraph import KEY, Wire, openai_reply, table

pytestmark = pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")

PORT = "port/port_model.py"
STILL_LOAD = BROKEN.replace("model.layers.2.ple.conv1d.weight", "model.layers.3.other.weight")


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.full_port_rehearsal = True          # the helper drives by default in a rehearsal
    yield r
    r.close()


def start(rig, port, per_round):
    """Stage 2's prepare writes `port`; a coder repair round k writes `per_round[k - 1]` (the last repeats). Returns the coder's repair prompts."""
    make(rig, over={(2, "prepare"): {**PREPARE, PORT: port}})
    seen = repairs(rig, [{PORT: text} for text in per_round])
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    return seen


# ---- the loop: the helper's file is the round's repair and the coder does not run -----------------------------------------------------------

def test_in_a_rehearsal_the_helper_repairs_every_round_and_no_coder_step_runs(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN, HONEST])
    seen = start(rig, BROKEN, ["# the coder must not be asked\n"])
    assert rig.run() == EXIT_ABORTED
    assert stage_ends(rig) == ["pass"]
    assert seen == []                                                          # no repair step reached the coder
    driven = decs(rig, rh.DRIVEN)
    assert [(d["round"], d["op"]) for d in driven] == [(1, "load"), (2, "prefill")]
    assert all(len(d["port_sha256"]) == 64 for d in driven)
    assert len(wire.requests) == 2 and [d["round"] for d in decs(rig, rh.ASKED)] == [1, 2]
    assert port_text(rig).endswith(HONEST.rstrip("\n") + "\n")


def test_the_driven_round_holds_the_cost_and_the_change_the_supervisor_computed(rig, monkeypatch, tmp_path):
    wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN, HONEST])
    start(rig, BROKEN, ["# c\n"])
    rig.run()
    placed = decs(rig, rh.PLACED)
    assert [p["driven"] for p in placed] == [True, True]
    assert all("lines added" in p["change"] and p["cost_usd"] >= 0 for p in placed)
    assert "functions changed: " in placed[0]["change"]


def test_the_second_request_tells_the_helper_what_the_first_round_failed_at_and_changed(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(monkeypatch, tmp_path, [STILL_LOAD, HONEST])
    start(rig, BROKEN, ["# c\n"])
    rig.run()
    first = wire.requests[0]["body"]["messages"][-1]["content"]
    second = wire.requests[1]["body"]["messages"][-1]["content"]
    assert "What the previous round did" not in first
    assert "What the previous round did" in second and "Round 1 failed at the 'load' request" in second
    assert "lines added" in second and "Do not repeat a change that did not move the failure" in second


def test_the_helper_never_touches_the_check_and_the_check_still_runs_under_a_lease(rig, monkeypatch, tmp_path):
    wire_helper(monkeypatch, tmp_path, [HONEST])
    start(rig, BROKEN, ["# c\n"])
    rig.run()
    assert not decs(rig, "port repair refused")
    assert len(decs(rig, port_repair.RERUN)) == 1 and len(decs(rig, port_repair.TESTED)) >= 2


# ---- when the helper does not drive -----------------------------------------------------------------------------------------------------

def test_outside_a_rehearsal_the_coder_repairs_unless_the_operator_asks_for_the_helper(rig, monkeypatch, tmp_path):
    rig.args.full_port_rehearsal = False
    tel, wire = wire_helper(monkeypatch, tmp_path, [HONEST])
    seen = start(rig, BROKEN, [HONEST + "# the coder's repair\n"])
    rig.run()
    assert seen and not decs(rig, rh.DRIVEN)


def test_the_operator_can_switch_the_driving_on_outside_a_rehearsal(rig, monkeypatch, tmp_path):
    rig.args.full_port_rehearsal = False
    rig.args.helper_drives_port = True
    wire_helper(monkeypatch, tmp_path, [HONEST])
    seen = start(rig, BROKEN, ["# c\n"])
    rig.run()
    assert seen == [] and len(decs(rig, rh.DRIVEN)) == 1


def test_no_helper_drives_port_leaves_the_repair_to_the_coder_in_a_rehearsal(rig, monkeypatch, tmp_path):
    rig.args.helper_drives_port = False
    wire_helper(monkeypatch, tmp_path, [HONEST])
    seen = start(rig, BROKEN, [HONEST + "# the coder's repair\n"])
    rig.run()
    assert seen and not decs(rig, rh.DRIVEN)


def test_without_a_helper_nothing_is_driven(rig, monkeypatch, tmp_path):
    monkeypatch.setattr(sv, "_telegraph_from", lambda *a, **k: None)
    seen = start(rig, BROKEN, [HONEST + "# the coder's repair\n"])
    rig.run()
    assert seen and not decs(rig, rh.DRIVEN) and not decs(rig, rh.BEST)


def test_with_the_round_asks_off_the_helper_does_not_drive(rig, monkeypatch, tmp_path):
    wire_helper(monkeypatch, tmp_path, [HONEST])
    monkeypatch.setattr(sv.Supervisor, "REPAIR_HELP_DEFAULT", False)
    seen = start(rig, BROKEN, [HONEST + "# the coder's repair\n"])
    rig.run()
    assert seen and not decs(rig, rh.DRIVEN)


# ---- handing back to the coder -----------------------------------------------------------------------------------------------------------------

def test_a_stub_reply_hands_the_round_to_the_coder(rig, monkeypatch, tmp_path):
    wire_helper(monkeypatch, tmp_path, [STUB, HONEST])
    seen = start(rig, BROKEN, [HONEST + "# the coder's repair\n"])
    rig.run()
    assert seen and not decs(rig, rh.DRIVEN)
    assert len(decs(rig, rh.STUB)) == 1


def test_two_rounds_that_fail_the_same_way_hand_the_third_to_the_coder_and_the_helper_is_not_asked_again_in_it(rig, monkeypatch, tmp_path):
    same_a, same_b = BROKEN + "# a\n", BROKEN + "# b\n"
    tel, wire = wire_helper(monkeypatch, tmp_path, [same_a, same_b, HONEST])
    seen = start(rig, BROKEN, [HONEST + "# the coder's repair\n"])
    rig.run()
    # round 1: the helper's file fails at load with the same message; round 2: the failure is the same as round 1's, so the coder gets it.
    assert [d["round"] for d in decs(rig, rh.DRIVEN)] == [1]
    assert seen and all("round 2 of" in text for text in seen)
    assert [d["round"] for d in decs(rig, rh.ASKED)] == [1] and len(wire.requests) == 1


def test_the_stage_cap_on_asks_hands_the_round_to_the_coder(rig, monkeypatch, tmp_path):
    monkeypatch.setattr(rh, "HELPER_TOTAL", 1)
    tel, wire = wire_helper(monkeypatch, tmp_path, [STILL_LOAD, HONEST])
    seen = start(rig, BROKEN, [HONEST + "# the coder's repair\n"])
    rig.run()
    assert len(decs(rig, rh.ASKED)) == 1 and seen
    refusals = decs(rig, rh.NOT_ASKED)
    assert refusals and "limit is 1" in refusals[0]["reason"]


def test_the_rounds_are_bounded_by_the_helper_rounds_constant(rig, monkeypatch, tmp_path):
    monkeypatch.setattr(rh, "HELPER_ROUNDS", 4)
    monkeypatch.setattr(rh, "HELPER_TOTAL", 99)
    distinct = [BROKEN.replace("model.layers.2", f"model.layers.{i + 3}") for i in range(12)]       # each round fails with its own message
    wire_helper(monkeypatch, tmp_path, distinct)
    start(rig, BROKEN, ["# c\n"])
    rig.run()
    rounds = [d["round"] for d in decs(rig, rh.DRIVEN)]
    assert rounds and max(rounds) == 4 and all(r <= 4 for r in rounds)
    assert [d["of"] for d in decs(rig, port_repair.DECISION)][0] == 4


# ---- the best port is kept across attempts --------------------------------------------------------------------------------------------------------

def attempts(rig, restored_prompts):
    """Attempt 1's prepare writes a port that fails at prefill, every coder repair round writes one that fails at load; a later attempt's prepare
    writes the working port when the supervisor's message says the best port is in place (as a coder that starts from it would). The prepare
    prompts are kept, and so is what port_model.py held when each prepare step began."""
    make(rig, over={(2, "prepare"): {**PREPARE, PORT: PREFILL_BROKEN}})
    repairs(rig, [{PORT: broken(1)}, {PORT: broken(2)}, {PORT: broken(3)}])
    base = rig.script

    def script(request):
        if "tools" in request and where(request) == (2, "prepare") and len(request["messages"]) == 2:       # the first request of the step
            text = request["messages"][1]["content"]
            on_disk = (rig.run_dir / "stages" / "2" / "port" / "port_model.py")
            restored_prompts.append((text, on_disk.read_text() if on_disk.is_file() else None))
            if "furthest version an earlier attempt reached" in text:
                return bringup(request, overrides={(2, "prepare"): {**PREPARE, PORT: HONEST}})
        return base(request)
    rig.script = script
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")


def test_the_furthest_port_is_recorded_with_its_op_and_hash_and_only_when_it_is_further(rig, monkeypatch, tmp_path):
    wire_helper(monkeypatch, tmp_path, [BROKEN])           # the helper drives: its reply fails like the rounds before it
    attempts(rig, [])
    rig.run()
    best = decs(rig, rh.BEST)
    # the load failures of the rounds are not further; attempt 2's working port (its check ran to the end) is, and ranks above every failure
    assert [(b["op"], b["rank"]) for b in best] == [("prefill", 3), (None, rh.PASSED_RANK)]
    assert best[0]["port_sha256"] == hashlib.sha256(PREFILL_BROKEN.encode()).hexdigest()
    copy = rig.run_dir / "stages" / "2.best" / best[0]["snapshot"] / "port_model.py"
    assert copy.is_file() and hashlib.sha256(copy.read_bytes()).hexdigest() == best[0]["port_sha256"]


def test_a_later_attempt_starts_from_the_best_port_not_from_the_skeleton(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(monkeypatch, tmp_path, [BROKEN])
    prompts = []
    attempts(rig, prompts)
    assert rig.run() == EXIT_ABORTED
    restored = decs(rig, rh.RESTORED)
    assert len(restored) == 1 and restored[0]["op"] == "prefill" and restored[0]["port_sha256"] == hashlib.sha256(PREFILL_BROKEN.encode()).hexdigest()
    assert len(restored[0]["skeleton_sha256"]) == 64
    first_text, first_disk = prompts[0]
    last_text, last_disk = prompts[-1]
    assert "furthest version" not in first_text and "furthest version" in last_text
    assert last_disk == PREFILL_BROKEN or last_disk.rstrip("\n") == PREFILL_BROKEN.rstrip("\n")
    assert not decs(rig, paf.ASKED)                                               # a restored best port makes the attempt-start ask unnecessary


def _stage_with_port(tmp_path, text: bytes, **extra):
    """stages/2/port holding port_model.py and the extra files (name: bytes)."""
    stage = tmp_path / "stages" / "2"
    (stage / "port").mkdir(parents=True)
    (stage / "port" / "port_model.py").write_bytes(text)
    for name, data in extra.items():
        (stage / "port" / name).write_bytes(data)
    return stage


def _next_attempt(stage, skeleton, **extra):
    """The same stage directory as a later attempt starts it: the port directory holds the skeleton (and the extra files, name: bytes)."""
    import shutil
    shutil.rmtree(stage / "port")
    (stage / "port").mkdir()
    (stage / "port" / "port_model.py").write_bytes(skeleton)
    for name, data in extra.items():
        (stage / "port" / name).write_bytes(data)


def test_a_port_the_coder_has_written_is_not_replaced_by_the_best_port(tmp_path):
    skeleton = b"# the skeleton\n"
    stage = _stage_with_port(tmp_path, PREFILL_BROKEN.encode())
    entry = rh.save_best(stage)
    assert entry is not None
    (stage / "port" / "port_model.py").write_bytes(b"# the coder wrote this\n")
    placed, why = rh.restore_best(stage, entry, skeleton)
    assert placed is False and "not the unchanged skeleton" in why
    assert (stage / "port" / "port_model.py").read_bytes() == b"# the coder wrote this\n"
    (stage / "port" / "port_model.py").write_bytes(skeleton)
    assert rh.restore_best(stage, entry, skeleton) == (True, "")
    assert (stage / "port" / "port_model.py").read_bytes() == PREFILL_BROKEN.encode()


def test_a_best_copy_that_does_not_match_the_ledgers_hash_is_not_put_in_place(tmp_path):
    skeleton = b"# the skeleton\n"
    stage = _stage_with_port(tmp_path, b"# something else than the ledger recorded\n")
    entry = rh.save_best(stage)
    (stage / "port" / "port_model.py").write_bytes(skeleton)
    copy = tmp_path / "stages" / "2.best" / entry["snapshot"] / "port_model.py"
    copy.write_bytes(b"# changed after the ledger entry\n")
    placed, why = rh.restore_best(stage, entry, skeleton)
    assert placed is False and "does not match" in why
    assert (stage / "port" / "port_model.py").read_bytes() == skeleton


def test_a_link_in_place_of_the_best_copy_is_refused(tmp_path):
    skeleton = b"# the skeleton\n"
    stage = _stage_with_port(tmp_path, b"# a port\n")
    entry = rh.save_best(stage)
    (stage / "port" / "port_model.py").write_bytes(skeleton)
    copy = tmp_path / "stages" / "2.best" / entry["snapshot"] / "port_model.py"
    elsewhere = tmp_path / "elsewhere.py"
    elsewhere.write_bytes(b"# a port\n")                      # the same bytes, so only the link can refuse it
    copy.unlink()
    os.symlink(elsewhere, copy)
    placed, why = rh.restore_best(stage, entry, skeleton)
    assert placed is False and "link" in why


# ---- round 2 of the review of PR 497: snapshots, supporting files, links -------------------------------------------------------------------

def test_the_snapshot_keeps_the_supporting_files_and_restores_them_all(tmp_path):
    """Finding: only port_model.py was saved, so a port that imports a neighbour lost it when the failed attempt was moved aside."""
    skeleton = b"# the skeleton\n"
    stage = _stage_with_port(tmp_path, b"import helper\n", **{"helper.py": b"X = 1\n"})
    (stage / "port" / "pkg").mkdir()
    (stage / "port" / "pkg" / "__init__.py").write_bytes(b"Y = 2\n")
    (stage / "port" / "__pycache__").mkdir()
    (stage / "port" / "__pycache__" / "helper.cpython-312.pyc").write_bytes(b"\0")
    entry = rh.save_best(stage)
    assert [f["path"] for f in entry["files"]] == ["helper.py", "pkg/__init__.py", "port_model.py"]
    _next_attempt(stage, skeleton)
    fresh = stage
    assert rh.restore_best(fresh, entry, skeleton) == (True, "")
    assert (fresh / "port" / "helper.py").read_bytes() == b"X = 1\n"
    assert (fresh / "port" / "pkg" / "__init__.py").read_bytes() == b"Y = 2\n"
    assert not (fresh / "port" / "__pycache__").exists()


def test_a_supporting_file_the_attempt_already_wrote_is_not_replaced_by_the_restore(tmp_path):
    skeleton = b"# the skeleton\n"
    stage = _stage_with_port(tmp_path, b"import helper\n", **{"helper.py": b"X = 1\n"})
    entry = rh.save_best(stage)
    _next_attempt(stage, skeleton, **{"helper.py": b"# the coder's own\n"})
    fresh = stage
    placed, why = rh.restore_best(fresh, entry, skeleton)
    assert placed is False and "already exists" in why
    assert (fresh / "port" / "port_model.py").read_bytes() == skeleton and (fresh / "port" / "helper.py").read_bytes() == b"# the coder's own\n"


def test_a_file_list_that_does_not_hash_to_the_snapshots_name_is_refused(tmp_path):
    skeleton = b"# the skeleton\n"
    stage = _stage_with_port(tmp_path, b"# a port\n")
    entry = rh.save_best(stage)
    forged = dict(entry, files=[dict(entry["files"][0], sha256=hashlib.sha256(b"# another port\n").hexdigest())])
    (stage / "port" / "port_model.py").write_bytes(skeleton)
    placed, why = rh.restore_best(stage, forged, skeleton)
    assert placed is False and "does not match the snapshot's name" in why


def test_a_restore_that_fails_part_way_leaves_the_skeleton_as_the_entry_module(tmp_path, monkeypatch):
    """The supporting files go first: when one cannot be written, port_model.py is still the skeleton and the next attempt can try again."""
    skeleton = b"# the skeleton\n"
    stage = _stage_with_port(tmp_path, b"import zz\n")
    (stage / "port" / "zz").mkdir()
    (stage / "port" / "zz" / "mod.py").write_bytes(b"Z = 1\n")
    entry = rh.save_best(stage)
    _next_attempt(stage, skeleton)
    real = rh.safeio.write_beneath_if

    def failing(fd, parts, data, **kw):
        if tuple(parts) == ("zz", "mod.py"):
            raise rh.safeio.PathRefused("the disk refused this file")
        return real(fd, parts, data, **kw)
    monkeypatch.setattr(rh.safeio, "write_beneath_if", failing)
    placed, why = rh.restore_best(stage, entry, skeleton)
    assert placed is False and "disk refused" in why
    assert (stage / "port" / "port_model.py").read_bytes() == skeleton


def test_a_new_best_port_never_replaces_the_snapshot_an_earlier_entry_names(tmp_path):
    """Finding: the single saved copy was replaced before its hash reached the ledger, so a stop between the two left the old entry pointing at other bytes."""
    skeleton = b"# the skeleton\n"
    stage = _stage_with_port(tmp_path, b"# first best\n")
    first = rh.save_best(stage)
    (stage / "port" / "port_model.py").write_bytes(b"# second best\n")
    second = rh.save_best(stage)                       # the ledger entry for this one was never written (a stop)
    assert first["snapshot"] != second["snapshot"]
    (stage / "port" / "port_model.py").write_bytes(skeleton)
    assert rh.restore_best(stage, first, skeleton) == (True, "")        # the earlier entry still restores its own bytes
    assert (stage / "port" / "port_model.py").read_bytes() == b"# first best\n"


def test_a_link_in_the_port_directory_refuses_the_snapshot(tmp_path):
    stage = _stage_with_port(tmp_path, b"# a port\n")
    os.symlink(tmp_path / "elsewhere.py", stage / "port" / "extra.py")
    assert rh.save_best(stage) is None


def test_the_restore_does_not_write_through_a_port_directory_that_became_a_link(tmp_path):
    """Finding: the write went to the port path by name, so a `port` directory swapped for a link after the skeleton check redirected it."""
    skeleton = b"# the skeleton\n"
    stage = _stage_with_port(tmp_path, b"# a port\n")
    entry = rh.save_best(stage)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "port_model.py").write_bytes(skeleton)
    import shutil
    shutil.rmtree(stage / "port")
    os.symlink(outside, stage / "port", target_is_directory=True)
    placed, why = rh.restore_best(stage, entry, skeleton)
    assert placed is False
    assert (outside / "port_model.py").read_bytes() == skeleton


# ---- the unit parts -------------------------------------------------------------------------------------------------------------------------

def test_best_rank_orders_the_request_chain_and_ranks_an_unknown_op_as_nothing():
    assert [rh.best_rank(op) for op in (None, "init", "load", "prefill", "decode", "banana")] == [0, 1, 2, 3, 4, 0]


def test_best_state_takes_the_highest_rank_and_the_latest_on_a_tie():
    def e(seq, rank, sha):
        return {"seq": seq, "stage": 2, "event": "decision", "data": {"decision": rh.BEST, "rank": rank, "port_sha256": sha, "op": "x"}}
    entries = [e(1, 2, "a"), e(2, 3, "b"), e(3, 1, "c"), e(4, 3, "d")]
    assert rh.best_state(entries, 2)["port_sha256"] == "d"
    assert rh.best_state(entries, 3) is None
    assert rh.best_state([{"seq": 1, "stage": 2, "event": "decision", "data": {"decision": rh.BEST, "rank": "3", "port_sha256": "z"}}], 2) is None


def test_change_summary_counts_lines_and_names_the_functions_that_changed():
    old = "def a():\n    return 1\n\ndef b():\n    return 2\n"
    new = "def a():\n    return 1\n\ndef b():\n    return 3\n\ndef c():\n    return 4\n"
    line = rh.change_summary(old, new)
    assert "lines added" in line and "functions changed: b, c" in line
    assert rh.change_summary(old, old).startswith("0 lines added, 0 removed") and "functions changed" not in rh.change_summary(old, old)


def test_change_summary_survives_text_that_does_not_parse():
    line = rh.change_summary("def a(:\n", "def b(:\n")
    assert "lines added" in line and "functions changed" not in line


def test_same_failure_twice_needs_the_same_op_and_the_same_message_in_two_rounds_of_the_attempt():
    def rnd(seq, k, op, failure):
        return {"seq": seq, "stage": 2, "event": "decision", "data": {"decision": port_repair.DECISION, "round": k, "op": op, "failure": failure}}
    start_entry = {"seq": 1, "stage": 2, "event": "stage_start", "data": {}}
    same = [start_entry, rnd(2, 1, "load", "boom"), rnd(3, 2, "load", "boom")]
    assert rh.same_failure_twice(same, 2, 2) is True
    assert rh.same_failure_twice(same, 2, 1) is False
    assert rh.same_failure_twice([start_entry, rnd(2, 1, "load", "boom"), rnd(3, 2, "prefill", "boom")], 2, 2) is False
    assert rh.same_failure_twice([start_entry, rnd(2, 1, "load", "boom"), rnd(3, 2, "load", "other")], 2, 2) is False
    assert rh.same_failure_twice([start_entry, rnd(2, 1, "load", ""), rnd(3, 2, "load", "")], 2, 2) is False
    later_start = {"seq": 2, "stage": 2, "event": "stage_start", "data": {}}
    assert rh.same_failure_twice([rnd(1, 1, "load", "boom"), later_start, rnd(3, 2, "load", "boom")], 2, 2) is False      # round 1 belongs to an earlier attempt


def test_a_cap_other_than_the_default_is_used_by_due():
    asked = [{"seq": i, "stage": 2, "event": "decision", "data": {"decision": rh.ASKED, "round": i}} for i in range(1, 8)]
    assert rh.due(asked, 2, 8)[0] is False
    assert rh.due(asked, 2, 8, total=rh.HELPER_TOTAL)[0] is True


# ---- the helper's cache does not serve an answer that was reported as not helpful -----------------------------------------------------------------

def make_helper(tmp_path):
    wire = Wire(lambda body: openai_reply("a useless stub", 100, 20))
    cfg = tg.load(table(models=["helper-model"], prices={"helper-model": [3.0, 15.0]}, max_calls_per_problem=8, max_calls=20), base=tmp_path)
    tel = tg.Telegraph(cfg, tmp_path / "tg", transport=wire, environ={"TG_KEY": KEY}, clock=lambda: 1000.0, hostname="node9", home="/home/someone")
    return tel, wire


def test_the_same_question_is_answered_from_the_cache_until_the_answer_is_reported_as_not_helpful(tmp_path):
    tel, wire = make_helper(tmp_path)
    first = tel.ask("p", "the question", "")
    again = tel.ask("p", "the question", "")
    assert again.cached is True and len(wire.requests) == 1
    tel.report(first.call_id, False, note="a stub")
    third = tel.ask("p", "the question", "")
    assert third.cached is False and len(wire.requests) == 2 and third.cost_usd > 0


def test_a_helpful_report_leaves_the_cache_in_use(tmp_path):
    tel, wire = make_helper(tmp_path)
    first = tel.ask("p", "the question", "")
    tel.report(first.call_id, True)
    assert tel.ask("p", "the question", "").cached is True and len(wire.requests) == 1


def test_an_unhelpful_report_against_a_cache_hit_blocks_the_origin_as_well(tmp_path):
    tel, wire = make_helper(tmp_path)
    tel.ask("p", "the question", "")
    hit = tel.ask("p", "the question", "")
    assert hit.cached is True
    tel.report(hit.call_id, False, note="a stub")                       # a report on a cache hit is recorded against the call that made the answer
    assert tel.ask("p", "the question", "").cached is False and len(wire.requests) == 2


# ---- review of PR 497, round 3: the restore keeps what a writer put there after the checks --------------------------------------------------

def test_an_edit_to_port_model_made_while_the_snapshot_is_read_is_kept(tmp_path, monkeypatch):
    """Finding: the skeleton check came first and `write_beneath` then replaced the file unconditionally, so a late edit was lost."""
    skeleton = b"# the skeleton\n"
    stage = _stage_with_port(tmp_path, PREFILL_BROKEN.encode())
    entry = rh.save_best(stage)
    _next_attempt(stage, skeleton)
    real = rh.safeio.read_beneath
    seen = []

    def editing(fd, parts, **kw):
        out = real(fd, parts, **kw)
        if tuple(parts) == ("port_model.py",) and not seen:
            seen.append(1)
            (stage / "port" / "port_model.py").write_bytes(b"# an edit that arrived late\n")
        return out
    monkeypatch.setattr(rh.safeio, "read_beneath", editing)
    placed, why = rh.restore_best(stage, entry, skeleton)
    assert placed is False and "edit is kept" in why
    assert (stage / "port" / "port_model.py").read_bytes() == b"# an edit that arrived late\n"


def test_a_supporting_file_created_after_its_existence_check_is_not_overwritten(tmp_path, monkeypatch):
    skeleton = b"# the skeleton\n"
    stage = _stage_with_port(tmp_path, b"import helper\n", **{"helper.py": b"H = 1\n"})
    entry = rh.save_best(stage)
    _next_attempt(stage, skeleton)
    real = rh._exists

    def late(pin, parts):
        out = real(pin, parts)
        if tuple(parts) == ("port", "helper.py"):
            (stage / "port" / "helper.py").write_bytes(b"# written by someone else\n")
        return out
    monkeypatch.setattr(rh, "_exists", late)
    placed, why = rh.restore_best(stage, entry, skeleton)
    assert placed is False and "already exists" in why
    assert (stage / "port" / "helper.py").read_bytes() == b"# written by someone else\n"
    assert (stage / "port" / "port_model.py").read_bytes() == skeleton


def test_a_port_directory_swapped_during_the_restore_is_refused_and_not_reported_as_restored(tmp_path, monkeypatch):
    """Review of PR 497, round 5: the writes went into the moved-away directory and `restore_best` still returned success."""
    import os
    skeleton = b"# the skeleton\n"
    stage = _stage_with_port(tmp_path, PREFILL_BROKEN.encode())
    entry = rh.save_best(stage)
    (stage / "port" / "port_model.py").write_bytes(skeleton)
    real = rh.safeio.write_beneath_if

    def swapping(fd, parts, data, **kw):
        out = real(fd, parts, data, **kw)
        if not (stage / "port.old").exists():
            os.rename(stage / "port", stage / "port.old")
            (stage / "port").mkdir()
            (stage / "port" / "port_model.py").write_bytes(skeleton)
        return out
    monkeypatch.setattr(rh.safeio, "write_beneath_if", swapping)
    placed, why = rh.restore_best(stage, entry, skeleton)
    assert placed is False and "replaced" in why
    assert (stage / "port" / "port_model.py").read_bytes() == skeleton      # the active port was not restored, and nothing says it was


def test_an_unswapped_restore_is_still_restored(tmp_path):
    skeleton = b"# the skeleton\n"
    stage = _stage_with_port(tmp_path, PREFILL_BROKEN.encode())
    entry = rh.save_best(stage)
    (stage / "port" / "port_model.py").write_bytes(skeleton)
    assert rh.restore_best(stage, entry, skeleton) == (True, "")
