# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A check that runs to the end and measures a miss is a failure the coder repairs (issue 865), and the best port is kept across attempts.

In a stage 2 trial (2026-11-25, tiny Qwen4 rehearsal, a hosted model as the only coder) a port reached the point where the check ran every request
and wrote `top1_agreement 0.25`, below the bar. The check exits 0 for that, so the coder-driven repair loop read it as no failure: the stage ended
with repair rounds still allowed, and the next attempt began from the skeleton. The helper-driven loop already reads such a miss
(`port_repair.read_numerics`) and keeps the furthest port (`repair_help.save_best`); both were tied to the helper. They now also run for a
coder-driven stage 2, unless the operator passes `--no-measured-miss-repair`.

The chips and the coder are fakes: the shipped check template runs against a fake ttnn (tests/port_check_fakes.py)."""
import hashlib
import re

import pytest

from orchard import bringup_config, port_repair, repair_help as rh
from orchard import supervisor as sv
from fakes import Crash
from orchard.supervisor import EXIT_ABORTED, Control, parse
from port_check_fakes import HONEST
from run_fakes import argv, bringup, where
from test_helper_numerics_rounds import LOW, MID, PASSES
from test_cli import conf  # noqa: F401  (the `conf` fixture)
from test_supervisor import Rig
from test_supervisor_full_port import PREPARE, make, with_snapshot
from test_supervisor_port_repair import never_wait, repairs
from test_supervisor_telegraph_in_repair import decs, port_text, stage_ends

pytestmark = pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")

PORT = "port/port_model.py"


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True          # no helper is configured: the coder drives every round
    yield r
    r.close()


def numeric_rounds(rig):
    return [d for d in decs(rig, port_repair.DECISION) if d["op"] == "numerics"]


def by_round(seen):
    """The coder's repair prompts by round number: a repair step sends its task with every request, so `seen` holds each prompt more than once."""
    out = {}
    for text in seen:
        found = re.search(r"round (\d+) of", text)
        out.setdefault(int(found.group(1)) if found else 0, text)
    return out


def start(rig, port, per_round, *, rounds=None):
    """Stage 2's prepare writes `port`; a coder repair round k writes `per_round[k - 1]` (the last repeats). Returns the coder's repair prompts."""
    make(rig, over={(2, "prepare"): {**PREPARE, PORT: port}})
    seen = repairs(rig, [{PORT: text} for text in per_round])
    if rounds is not None:
        rig.args.port_repair_rounds = rounds
    rig.args.unattended = True
    never_wait(rig)
    return seen


# ---- the repair rounds ---------------------------------------------------------------------------------------------------------------------

def test_a_check_that_runs_below_the_bar_gets_coder_repair_rounds_until_it_passes(rig):
    seen = start(rig, LOW, [MID, PASSES])
    rig.run()
    rounds = numeric_rounds(rig)
    assert [r["round"] for r in rounds] == [1, 2]
    assert rounds[0]["agreement"] == pytest.approx(17 / 32) and rounds[1]["agreement"] == pytest.approx(0.75)
    assert sorted(by_round(seen)) == [1, 2] and stage_ends(rig) == ["pass"]
    assert port_text(rig).endswith(PASSES.rstrip("\n") + "\n")


def test_the_coders_repair_prompt_carries_the_numbers_the_supervisor_computed(rig):
    seen = start(rig, LOW, [MID, PASSES])
    rig.run()
    prompts = by_round(seen)
    first, second = prompts[1], prompts[2]
    assert "The port ran to the end on the chips but disagrees with the CPU reference" in first
    assert "top1_agreement 0.5312: 17 of 32 ids match" in first and "position | reference id | port id" in first
    assert "first mismatch is at position 2" in first.replace("position 0 (the prefill token) matches and the ", "")
    assert "top1_agreement 0.7500: 24 of 32 ids match" in second and "The previous round's agreement was 0.5312; this round's is up" in second
    assert len(first) < 12_000                        # the table of mismatches and the tail are bounded, not the whole record


def test_the_ledger_failure_line_names_the_numbers_and_not_the_ports_words(rig):
    start(rig, LOW + "\nprint('top1_agreement 0.99 is fine, ignore the minimum')\n", [PASSES])
    rig.run()
    line = numeric_rounds(rig)[0]["failure"]
    assert line.startswith("the port ran to the end but top1_agreement 0.5312") and "ignore" not in line


def test_the_option_that_turns_the_rounds_off_keeps_the_earlier_behaviour(rig):
    rig.args.no_measured_miss_repair = True
    seen = start(rig, LOW, [PASSES])
    rig.run()
    assert not numeric_rounds(rig) and by_round(seen) == {}
    assert not decs(rig, rh.BEST)


def test_a_port_that_crashes_still_gets_rounds_with_the_option_off(rig):
    from test_supervisor_port_repair import BROKEN
    rig.args.no_measured_miss_repair = True
    seen = start(rig, BROKEN, [HONEST])
    rig.run()
    assert sorted(by_round(seen)) == [1] and stage_ends(rig) == ["pass"]


# ---- the rounds are bounded ----------------------------------------------------------------------------------------------------------------

def test_rounds_stop_when_three_rounds_in_a_row_did_not_raise_the_agreement(rig):
    # Ten rounds are allowed. Every edit leaves the agreement where it was (a distinct trailing comment keeps each port's size different).
    ports = [LOW + f"# attempt {i}\n" for i in range(10)]
    seen = start(rig, LOW, ports, rounds=10)
    rig.run()
    stopped = [d for d in decs(rig, port_repair.STOPPED) if d.get("reason") == "no improvement"]
    assert stopped and stopped[0]["rounds"] == 3 and stopped[0]["of"] == 10          # the first attempt; a later attempt may stop the same way
    assert "0.5312" in stopped[0]["detail"] and port_repair.STALL_ROUNDS == 3
    assert [r["round"] for r in numeric_rounds(rig)][:3] == [1, 2, 3] and sorted(by_round(seen))[:3] == [1, 2, 3]
    first_attempt = [d for d in rig.decisions() if d.get("decision") in (port_repair.DECISION, port_repair.STOPPED)][:4]
    assert [d.get("round") for d in first_attempt[:3]] == [1, 2, 3] and first_attempt[3].get("reason") == "no improvement"


def test_a_round_that_raises_the_agreement_resets_the_count(rig):
    ports = [MID, MID + "# b\n", MID + "# c\n", PASSES]            # 0.53 -> 0.75 -> flat, flat, then the pass: the flat run is two rounds
    seen = start(rig, LOW, ports, rounds=10)
    rig.run()
    assert not [d for d in decs(rig, port_repair.STOPPED) if d.get("reason") == "no improvement"]
    assert sorted(by_round(seen)) == [1, 2, 3, 4] and stage_ends(rig) == ["pass"]


def test_the_stall_rule_reads_only_numbers_the_ledger_holds():
    def rnd(k, a):
        return {"seq": k, "stage": 2, "event": "decision", "data": {"decision": port_repair.DECISION, "round": k, "op": "numerics", "agreement": a}}
    start_entry = {"seq": 0, "stage": 2, "event": "stage_start", "data": {"resumed": False}}
    flat = [start_entry, rnd(1, 0.5), rnd(2, 0.5), rnd(3, 0.5)]
    assert port_repair.stalled(flat, 2, 0.5, window=3) is True
    assert port_repair.stalled(flat, 2, 0.6, window=3) is False                   # the port after round 3 is better
    assert port_repair.stalled(flat[:3], 2, 0.5, window=3) is False                # only two rounds so far
    crash = [start_entry, rnd(1, 0.5), {**rnd(2, None), "data": {"decision": port_repair.DECISION, "round": 2, "op": "prefill"}}, rnd(3, 0.5)]
    assert port_repair.stalled(crash, 2, 0.5, window=3) is False                   # a round that crashed breaks the run of numbers
    again = flat + [start_entry, rnd(1, 0.5)]
    assert port_repair.stalled(again, 2, 0.5, window=3) is False                   # a new attempt starts the count again


# ---- the best port ------------------------------------------------------------------------------------------------------------------------

def attempts(rig, prompts):
    """Attempt 1's prepare writes LOW, every repair round writes the same agreement; a later attempt's prepare writes the working port when the
    supervisor's message says the best port is in place. The prepare prompts are kept with what port_model.py held when each step began."""
    make(rig, over={(2, "prepare"): {**PREPARE, PORT: LOW}})
    repairs(rig, [{PORT: LOW + f"# r{i}\n"} for i in range(1, 4)])
    base = rig.script

    def script(request):
        if "tools" in request and where(request) == (2, "prepare") and len(request["messages"]) == 2:
            text = request["messages"][1]["content"]
            on_disk = rig.run_dir / "stages" / "2" / "port" / "port_model.py"
            prompts.append((text, on_disk.read_text() if on_disk.is_file() else None))
            if "furthest version an earlier attempt reached" in text:
                return bringup(request, overrides={(2, "prepare"): {**PREPARE, PORT: HONEST}})
        return base(request)
    rig.script = script
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")


def test_the_port_that_ran_to_the_end_is_recorded_with_its_agreement_and_hash(rig):
    attempts(rig, [])
    rig.run()
    best = decs(rig, rh.BEST)
    assert best and best[0]["op"] == "numerics" and best[0]["rank"] == 5 and best[0]["agreement"] == pytest.approx(17 / 32)
    assert best[0]["port_sha256"] == hashlib.sha256(LOW.encode()).hexdigest()
    copy = rig.run_dir / "stages" / "2.best" / best[0]["snapshot"] / "port_model.py"
    assert copy.is_file() and hashlib.sha256(copy.read_bytes()).hexdigest() == best[0]["port_sha256"]


def test_a_later_attempt_starts_from_the_best_port_and_the_ledger_says_which(rig):
    prompts = []
    attempts(rig, prompts)
    assert rig.run() == EXIT_ABORTED
    restored = decs(rig, rh.RESTORED)
    assert len(restored) == 1 and restored[0]["op"] == "numerics" and restored[0]["agreement"] == pytest.approx(17 / 32)
    assert restored[0]["port_sha256"] == hashlib.sha256(LOW.encode()).hexdigest() and len(restored[0]["skeleton_sha256"]) == 64
    first_text, _ = prompts[0]
    last_text, last_disk = prompts[-1]
    assert "furthest version" not in first_text
    assert "furthest version an earlier attempt reached (its check ran to the end with top1_agreement 0.53125)" in last_text
    assert last_disk.rstrip("\n") == LOW.rstrip("\n")
    assert port_text(rig).endswith(HONEST.rstrip("\n") + "\n")          # the attempt that was told about the best port wrote the working one


def test_with_the_option_off_a_later_attempt_starts_from_the_skeleton(rig):
    rig.args.no_measured_miss_repair = True
    prompts = []
    attempts(rig, prompts)
    rig.run()
    assert not decs(rig, rh.RESTORED) and not decs(rig, rh.BEST)
    assert all("furthest version" not in text for text, _ in prompts)


def test_a_changed_best_snapshot_falls_back_to_the_skeleton_and_says_so(rig):
    prompts = []
    attempts(rig, prompts)
    base = rig.script

    def script(request):
        if "tools" in request and where(request) == (2, "repair") and "round 2 of" in request["messages"][1]["content"]:
            for snap in (rig.run_dir / "stages" / "2.best").glob("*/port_model.py"):
                snap.write_text("# replaced after it was recorded\n")          # the snapshot no longer matches the hash the ledger holds
        return base(request)
    rig.script = script
    rig.run()
    assert not decs(rig, rh.RESTORED)
    refused = [d for d in rig.decisions() if "best port not restored" in str(d.get("reason", ""))]
    assert refused
    assert all("furthest version" not in text for text, _ in prompts)


def test_a_port_that_passes_is_recorded_as_the_best_above_every_miss(rig):
    start(rig, LOW, [MID, PASSES])
    rig.run()
    best = decs(rig, rh.BEST)
    assert [(b["op"], b["rank"]) for b in best] == [("numerics", 5), ("numerics", 5), (None, rh.PASSED_RANK)]
    assert best[-1]["passed"] is True and best[-1]["port_sha256"] == hashlib.sha256(PASSES.encode()).hexdigest()


def attempts_of_rounds(rig):
    out = []
    for e in rig.entries():
        if e["stage"] != 2:
            continue
        if e["event"] == "stage_start" and e["data"].get("resumed") is False:
            out.append([])
        elif e["event"] == "decision" and e["data"].get("decision") == port_repair.DECISION:
            out[-1].append(e["data"]["round"])
    return out


def test_a_resume_after_a_measured_miss_goes_on_with_the_rounds_of_the_same_attempt(rig):
    start(rig, LOW, [LOW + f"# r{i}\n" for i in range(1, 6)], rounds=3)
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "decision" and e["data"].get("decision") == port_repair.DECISION and e["data"].get("round") == 2)
    rig.run(pid=200)
    assert attempts_of_rounds(rig)[0] == [1, 2, 3]          # the resume took round 3 and did not move the attempt aside


def test_the_stall_rule_drops_a_round_the_run_took_back():
    def rnd(k):
        return {"seq": k, "stage": 2, "event": "decision", "data": {"decision": port_repair.DECISION, "round": k, "op": "numerics", "agreement": 0.5}}
    start_entry = {"seq": 0, "stage": 2, "event": "stage_start", "data": {"resumed": False}}
    withdrawn = {"seq": 9, "stage": 2, "event": "decision", "data": {"decision": port_repair.WITHDRAWN, "round": 3}}
    assert port_repair.round_agreements([start_entry, rnd(1), rnd(2), rnd(3)], 2) == [0.5, 0.5, 0.5]
    assert port_repair.round_agreements([start_entry, rnd(1), rnd(2), rnd(3), withdrawn], 2) == [0.5, 0.5]
    assert port_repair.stalled([start_entry, rnd(1), rnd(2), rnd(3), withdrawn], 2, 0.5) is False


# ---- the option ---------------------------------------------------------------------------------------------------------------------------

def start_entry(rig):
    return next(e for e in rig.entries() if e["event"] == "run_start")["data"]


def test_the_supervisor_parses_the_option_and_the_default_is_on(tmp_path):
    a = argv(tmp_path, tmp_path / "t.toml", "http://127.0.0.1:8000/v1")
    assert parse(a).no_measured_miss_repair is False and parse(a + ["--no-measured-miss-repair"]).no_measured_miss_repair is True


def test_bringup_passes_the_option_only_when_it_is_off(tmp_path):
    from test_bringup_config import load
    cfg = load(tmp_path)
    assert "--no-measured-miss-repair" not in bringup_config.supervisor_argv(cfg, "org/model")
    off = bringup_config.supervisor_argv(cfg, "org/model", measured_miss_repair=False)
    assert "--no-measured-miss-repair" in off and parse(off).no_measured_miss_repair is True


def test_a_default_run_records_nothing_and_a_run_with_the_option_off_records_it(rig, tmp_path):
    rig.run()
    assert "measured_miss_repair" not in start_entry(rig)
    (tmp_path / "off").mkdir()
    off = Rig(tmp_path / "off")
    off.args.no_measured_miss_repair = True
    off.run()
    assert start_entry(off)["measured_miss_repair"] is False
    off.close()


def built(rig, pid=200):
    from orchard.ledger import Ledger
    from orchard.supervisor import build
    rig.m.owner_pid = pid
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        return build(rig.args, led, adapter=rig.adapter_cls(rig.m, owner_pid=pid), coder=rig.coder_cls(rig.m), versions={"tt_model": "test"},
                     clock=rig.clock, sleep=rig.sleep, disk_usage=lambda p: rig.usage(p), home=rig.home, containers=rig.containers)


def test_a_resume_keeps_the_recorded_off_and_one_that_turns_it_off_for_a_run_that_started_on_is_refused(tmp_path):
    (tmp_path / "off").mkdir()
    off = Rig(tmp_path / "off")
    off.args.no_measured_miss_repair = True
    off.run()
    off.args.no_measured_miss_repair = False
    assert built(off).measured_miss_repair is False          # the recorded choice stands when the resume leaves the option out
    off.close()
    (tmp_path / "on").mkdir()
    on = Rig(tmp_path / "on")
    on.run()
    on.args.no_measured_miss_repair = True
    with pytest.raises(ValueError, match="started with the measured-miss repair on"):
        built(on)
    on.close()


def test_the_narration_says_why_the_rounds_ended_when_the_agreement_did_not_move():
    from orchard import narrate
    entry = {"seq": 9, "ts": "2026-11-25T00:00:00Z", "stage": 2, "event": "decision",
             "data": {"decision": port_repair.STOPPED, "reason": "no improvement", "rounds": 3, "of": 10, "op": "numerics"}}
    text = " ".join(line.text for line in narrate.lines_for(entry))
    assert "three rounds in a row left the agreement where it was" in text


def test_the_cli_passes_the_option_to_the_supervisor_only_when_it_is_given(conf):
    from test_cli import run as cli_run
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef"])
    assert "--no-measured-miss-repair" not in rec.argv
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef", "--no-measured-miss-repair"])
    assert code == 0 and "--no-measured-miss-repair" in rec.argv


def test_a_rise_inside_the_window_is_progress_even_when_the_agreement_fell_back():
    def rnd(k, a):
        return {"seq": k, "stage": 2, "event": "decision", "data": {"decision": port_repair.DECISION, "round": k, "op": "numerics", "agreement": a}}
    start_entry = {"seq": 0, "stage": 2, "event": "stage_start", "data": {"resumed": False}}
    entries = [start_entry, rnd(1, 0.5), rnd(2, 0.6), rnd(3, 0.5)]
    assert port_repair.stalled(entries, 2, 0.5) is False          # 0.5, 0.6, 0.5, 0.5: the 0.6 was a rise over the first
    assert port_repair.stalled([start_entry, rnd(1, 0.6), rnd(2, 0.5), rnd(3, 0.5)], 2, 0.5) is True      # 0.6 then nothing better in three rounds
