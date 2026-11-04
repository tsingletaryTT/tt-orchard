# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 4 of the full-port route (issue 454): the gate of the entries (orchard/full_port_entries.py), the entry form of the stage 2 and 3
check (`full_port.check(..., entry=N)`), the supervisor's ledger checks (`full_port_stage4.vouch`) and the routing. The supervisor's side is
in tests/test_supervisor_full_port_stage4.py. No test here touches a chip.

`stage4_run` builds the directory a passing run leaves for the counts 1, 3 and 4: stage 3 on four chips, entries for 1 and 3, and a covered
entry for 4. Each guard test changes one fact and watches the gate name it."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import full_port, full_port_entries, full_port_stage4
from orchard.ledger import Ledger
from orchard.stages import FULL_PORT_STAGE_3, FULL_PORT_STAGE_4, STAGES, SKIP_4_FULL_PORT, gate_full_port_4, spec_for
from test_full_port import REF_IDS, TEXT, build, put, sha

SKILLS = Path(__file__).resolve().parent.parent / "orchard" / "skills"
CHIPS = [f"0000:0{i}:00.0" for i in range(1, 5)]
REQUIRED = (1, 3, 4)


def make_entry(run, n, *, chips=None, record=None, marker=None, result=None, model_src="import ttnn\n", template=None):
    """The files of one entry directory a passing test leaves."""
    base = f"stages/4/configs/{n}"
    names = list(chips if chips is not None else CHIPS[:n])
    ref, prm = run / "stages/1/evidence/reference/generated-ids.json", run / "stages/1/evidence/reference/prompt-ids.json"
    model = put(run, f"{base}/port/port_model.py", model_src)
    tpl = put(run, f"{base}/{full_port.TEMPLATE_NAME}", full_port.TEMPLATE.read_text() if template is None else template)
    config = put(run, f"{base}/{full_port.CONFIG_NAME}", {"model_snapshot": "snapshot"})
    rec = {"schema": full_port.RECORD_SCHEMA,
           "reference": {"path": "stages/1/evidence/reference/generated-ids.json", "sha256": sha(ref.read_bytes())},
           "prompt": {"path": "stages/1/evidence/reference/prompt-ids.json", "sha256": sha(prm.read_bytes())},
           "reference_ids": list(REF_IDS), "chip_ids": list(REF_IDS[:30]) + [0, 0], "top1_agreement": 30 / 32, "chips": names, "load_s": 90.5,
           "free_run_ids": [REF_IDS[0]] + list(range(1, 48)), "free_run_text": TEXT, "template_sha256": sha(full_port.TEMPLATE.read_bytes()),
           "config": {"path": f"{base}/{full_port.CONFIG_NAME}", "sha256": sha(config.read_bytes())},
           "port_sources": [{"path": f"{base}/port/port_model.py", "sha256": sha(model.read_bytes()), "role": "model"}], **(record or {})}
    rec_path = put(run, f"{base}/evidence/port-check.json", rec)
    out = put(run, f"{base}/evidence/hw-test-output.txt", f"loading\nPORT_CHECK_SHA256 {sha(rec_path.read_bytes())}\n")
    put(run, f"{base}/test-result.json",
        {"command": full_port.command(4, n), "returncode": 0, "timed_out": False, "seconds": 5.0, "chips": names,
         "output": {"path": f"{base}/evidence/hw-test-output.txt", "sha256": sha(out.read_bytes())}, **(marker or {})})
    put(run, f"{base}/result.json",
        {"runs": True, "coherent": True, "n_tokens": 32, "top1_agreement": 30 / 32, "load_s": 90.5, "free_run_text": TEXT,
         "evidence": [f"{base}/evidence/port-check.json", f"{base}/evidence/hw-test-output.txt"], **(result or {})})


def make_covered(run, n, stage=3, **over):
    ev = full_port.file_record(run, run / f"stages/{stage}/test-result.json")
    rec = full_port.file_record(run, run / f"stages/{stage}/evidence/port-check.json")
    put(run, f"stages/4/configs/{n}/covered.json", {"chips": n, "covered_by": stage, "evidence": ev, "record": rec, **over})


def stage4_run(tmp_path, *, required=REQUIRED, write_summary=True):
    """(stage_dir, run_dir) of a passing stage 4."""
    stage3, run = build(tmp_path, stage=3, record={"chips": CHIPS}, marker={"chips": CHIPS})
    stage_dir = run / "stages" / "4"
    for n in (1, 3):
        make_entry(run, n)
    make_covered(run, 4)
    plan, _ = full_port_entries.plan_entries(list(required), max_chips=4, budget_s=28800.0)
    full_port_entries.write_plan(stage_dir, plan)
    if write_summary:
        full_port_entries.write_json(stage_dir / "result.json", full_port_entries.summarize(stage_dir, run, required))
    return stage_dir, run


def gate(stage_dir, run, **kw):
    return gate_full_port_4(stage_dir, run, required=kw.pop("required", REQUIRED), **kw)


def one(g, word):
    assert not g.ok, "the gate passed a stage it must refuse"
    return [r for r in g.reasons if word in r]


def rewrite(path, **over):
    path.write_text(json.dumps({**json.loads(path.read_text()), **over}))


# ---- the gate: a pass ---------------------------------------------------------------------------------------------

def test_a_stage_4_with_a_tested_entry_for_each_count_and_a_covered_one_passes_and_lists_its_evidence(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    g = gate(stage_dir, run)
    assert g.ok, g.reasons
    assert "stages/4/configs/1/evidence/port-check.json" in g.evidence and "stages/4/configs/3/evidence/hw-test-output.txt" in g.evidence
    assert "stages/3/test-result.json" in g.evidence and "stages/4/tests/plan.json" in g.evidence and "stages/4/result.json" in g.evidence


def test_the_summary_has_one_row_per_count_with_numbers_recomputed_from_the_records(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    rows = json.loads((stage_dir / "result.json").read_text())["entries"]
    assert [(r["chips"], r["covered_by"]) for r in rows] == [(1, None), (3, None), (4, 3)]
    assert rows[0]["top1_agreement"] == 30 / 32 and rows[0]["n_tokens"] == 32 and rows[2]["evidence"] == ["stages/3/test-result.json", "stages/3/evidence/port-check.json"]


# ---- each guard, one at a time ------------------------------------------------------------------------------------

def test_an_entry_whose_record_lists_a_different_chip_count_is_refused(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    make_entry(run, 3, chips=CHIPS[:2])          # the record and the marker both list two chips for the three-chip entry
    full_port_entries.write_json(stage_dir / "result.json", full_port_entries.summarize(stage_dir, run, REQUIRED))
    assert one(gate(stage_dir, run), "the 3-chip entry") and one(gate(stage_dir, run), "configured for 3")


def test_a_marker_that_lists_more_chips_than_the_record_is_refused(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    rewrite(stage_dir / "configs/1/test-result.json", chips=CHIPS[:2])
    assert one(gate(stage_dir, run), "not the chips the supervisor leased")


def test_a_port_source_outside_the_entry_directory_is_refused(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    rec = json.loads((stage_dir / "configs/1/evidence/port-check.json").read_text())
    rec["port_sources"] = [{"path": "stages/3/port/port_model.py", "sha256": sha("import ttnn\n"), "role": "model"}]
    (stage_dir / "configs/1/evidence/port-check.json").write_text(json.dumps(rec))
    assert one(gate(stage_dir, run), "needs a file under stages/4/configs/1/port/")


def test_a_port_source_in_another_entrys_directory_is_refused(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    rec = json.loads((stage_dir / "configs/1/evidence/port-check.json").read_text())
    rec["port_sources"] = [{"path": "stages/4/configs/3/port/port_model.py", "sha256": sha("import ttnn\n"), "role": "model"}]
    (stage_dir / "configs/1/evidence/port-check.json").write_text(json.dumps(rec))
    assert one(gate(stage_dir, run), "the 1-chip entry") and one(gate(stage_dir, run), "needs a file under stages/4/configs/1/port/")


def test_a_template_copy_that_differs_from_the_shipped_file_is_refused(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    (stage_dir / "configs/3" / full_port.TEMPLATE_NAME).write_text(full_port.TEMPLATE.read_text() + "\n# edited\n")
    assert one(gate(stage_dir, run), "differs from the shipped template")


def test_a_marker_with_stage_3s_command_for_an_entry_is_refused(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    rewrite(stage_dir / "configs/1/test-result.json", command=full_port.command(3))
    assert one(gate(stage_dir, run), full_port.command(4, 1))


def test_a_marker_that_shows_a_failed_test_is_refused(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    rewrite(stage_dir / "configs/1/test-result.json", returncode=1)
    assert one(gate(stage_dir, run), "exited with code 1")


def test_a_plan_that_lists_other_counts_than_the_runs_is_refused(tmp_path):
    stage_dir, run = stage4_run(tmp_path, required=(1, 3))
    assert one(gate(stage_dir, run, required=REQUIRED), "tests/plan.json lists the counts [1, 3]")


def test_a_missing_entry_directory_is_refused_not_skipped(tmp_path):
    import shutil
    stage_dir, run = stage4_run(tmp_path)
    shutil.rmtree(stage_dir / "configs/3")
    assert one(gate(stage_dir, run), "the 3-chip entry: the entry directory configs/3/ does not exist")


@pytest.mark.parametrize("field,value", [("top1_agreement", 1.0), ("n_tokens", 31), ("covered_by", 2), ("chips", 5)])
def test_a_stage_result_that_differs_from_the_summary_of_the_records_is_refused(tmp_path, field, value):
    stage_dir, run = stage4_run(tmp_path)
    doc = json.loads((stage_dir / "result.json").read_text())
    doc["entries"][0 if field != "covered_by" else 2][field] = value
    (stage_dir / "result.json").write_text(json.dumps(doc))
    assert one(gate(stage_dir, run), "result.json is not the summary the gate builds")


def test_a_missing_stage_result_is_refused(tmp_path):
    stage_dir, run = stage4_run(tmp_path, write_summary=False)
    assert one(gate(stage_dir, run), "result.json is missing")


def test_the_gate_takes_the_counts_from_the_runs_option_and_names_a_bad_one(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    for bad in (None, (), (0,), (True,), ("4",)):
        assert one(gate(stage_dir, run, required=bad), "--required-chips")


@pytest.mark.parametrize("huge", [10 ** 20, 2 ** 63])
def test_a_very_large_count_is_a_missing_entry_and_never_a_pass(tmp_path, huge):
    stage_dir, run = stage4_run(tmp_path)
    assert one(gate(stage_dir, run, required=(1, 3, 4, huge)), f"the {huge}-chip entry")
    entries, problems = full_port_entries.plan_entries([huge], max_chips=4, budget_s=28800.0)
    assert entries == [] and "more chips than" in " ".join(problems)


def test_rehearsal_waives_only_the_coherent_field_of_each_entry(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    rewrite(stage_dir / "configs/1/result.json", coherent=False)
    full_port_entries.write_json(stage_dir / "result.json", full_port_entries.summarize(stage_dir, run, REQUIRED))
    assert one(gate(stage_dir, run), "coherent")
    waived = gate(stage_dir, run, rehearsal=True)
    assert waived.ok and waived.waived == ("coherent",)
    rewrite(stage_dir / "configs/1/result.json", coherent=False, top1_agreement=0.1)
    assert not gate(stage_dir, run, rehearsal=True).ok          # every other check still applies


# ---- a covered entry ----------------------------------------------------------------------------------------------

def test_a_covered_entry_whose_covering_marker_changed_after_the_covering_is_refused(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    rewrite(run / "stages/3/test-result.json", seconds=99.0)
    assert one(gate(stage_dir, run), "not the file the covering of this entry rested on")


def test_a_covered_entry_for_a_count_the_covering_stage_did_not_run_is_refused(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    make_covered(run, 4, stage=2)            # stage 2's marker is not there in this run
    assert one(gate(stage_dir, run), "stages/2/test-result.json")
    make_covered(run, 4, stage=3, chips=3)
    assert one(gate(stage_dir, run), "is for 3 chips, not 4")


def test_a_covered_entry_that_names_a_stage_that_cannot_cover_is_refused(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    make_covered(run, 4, covered_by=4)
    assert one(gate(stage_dir, run), "only stages [3, 2] cover an entry")


def test_an_entry_with_a_marker_and_a_covered_file_is_refused(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    make_entry(run, 4)
    assert one(gate(stage_dir, run), "either covered or tested")


def test_a_covering_stage_that_ran_on_other_chips_than_the_count_covers_nothing(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    make_covered(run, 1)                      # stage 3 ran on four chips, not one
    assert one(gate(stage_dir, run), "does not cover this entry")


# ---- the ledger's half --------------------------------------------------------------------------------------------

class Sup:
    """The parts of the supervisor `vouch` reads: the run directory, the ledger and the option."""
    def __init__(self, run, ledger, required=REQUIRED):
        self.run_dir, self.ledger, self.required_chips = run, ledger, required


def vouched_ledger(tmp_path, stage_dir, run, *, skip=()):
    """A ledger holding what the supervisor writes for `stage4_run`'s files (leave out the entries named in `skip`)."""
    led = Ledger(tmp_path / "ledger.jsonl")
    rec = lambda p: full_port.file_record(run, p)
    led.append("stage_start", 3, escalated=False, resumed=False)
    led.append("evidence", 3, what="hardware test", returncode=0, timed_out=False, **rec(run / "stages/3/test-result.json"))
    led.append("stage_end", 3, result="pass")
    led.append("stage_start", 4, escalated=False, resumed=False)
    rows = [("plan", "hardware test list", stage_dir / "tests/plan.json"), ("covered", full_port_stage4.COVERED_WHAT, stage_dir / "configs/4/covered.json"),
            ("stage", full_port_stage4.STAGE_RESULT_WHAT, stage_dir / "result.json")]
    for n in (1, 3):
        rows += [(f"marker{n}", None, stage_dir / f"configs/{n}/test-result.json"),
                 (f"result{n}", full_port_stage4.ENTRY_RESULT_WHAT, stage_dir / f"configs/{n}/result.json")]
    for key, what, path in rows:
        if key in skip:
            continue
        if what is None:
            led.append("evidence", 4, what="hardware test", returncode=0, timed_out=False, **rec(path))
        else:
            led.append("evidence", 4, what=what, **rec(path))
    return led


def vouch(sup, gate_result):
    return full_port_stage4.vouch(sup, FULL_PORT_STAGE_4, sup.run_dir / "stages" / "4", gate_result, waived_decision="waived")


def test_a_stage_4_whose_files_the_ledger_holds_passes_the_ledger_checks(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    led = vouched_ledger(tmp_path, stage_dir, run)
    g = gate(stage_dir, run)
    assert g.ok and vouch(Sup(run, led), g).ok


@pytest.mark.parametrize("skip,word", [("marker1", "the 1-chip entry: test-result.json is not the file the supervisor recorded"),
                                       ("marker3", "the 3-chip entry: test-result.json is not the file the supervisor recorded"),
                                       ("plan", "tests/plan.json is not the file the supervisor recorded"),
                                       ("covered", "the 4-chip entry: covered.json is not the file the supervisor recorded"),
                                       ("result1", "the 1-chip entry: result.json is not the file the supervisor recorded"),
                                       ("stage", "result.json is not the file the supervisor recorded")])
def test_a_file_the_ledger_does_not_hold_is_refused(tmp_path, skip, word):
    stage_dir, run = stage4_run(tmp_path)
    led = vouched_ledger(tmp_path, stage_dir, run, skip=(skip,))
    out = vouch(Sup(run, led), gate(stage_dir, run))
    assert not out.ok and any(word in r for r in out.reasons), out.reasons


def test_a_marker_changed_after_the_ledger_recorded_it_is_refused(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    led = vouched_ledger(tmp_path, stage_dir, run)
    rewrite(stage_dir / "configs/1/test-result.json", seconds=1.5)          # a change that leaves the gate's own checks satisfied
    out = vouch(Sup(run, led), gate(stage_dir, run))
    assert not out.ok and any("the 1-chip entry: test-result.json is not the file" in r for r in out.reasons)


def test_a_covering_stage_the_ledger_shows_failed_covers_nothing(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    led = vouched_ledger(tmp_path, stage_dir, run)
    led.append("stage_end", 3, result="fail")
    out = vouch(Sup(run, led), gate(stage_dir, run))
    assert not out.ok and any("covers it, and the ledger shows no pass of stage 3" in r for r in out.reasons)


def test_covering_stage_asks_the_ledger_and_the_marker_and_the_count(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    led = vouched_ledger(tmp_path, stage_dir, run)
    cover = full_port_stage4.covering_stage(led, run, 4)
    assert cover["covered_by"] == 3 and cover["evidence"]["path"] == "stages/3/test-result.json"
    assert full_port_stage4.covering_stage(led, run, 3) is None and full_port_stage4.covering_stage(led, run, 1) is None
    led.append("stage_end", 3, result="fail")
    assert full_port_stage4.covering_stage(led, run, 4) is None


def test_a_covering_marker_the_ledger_does_not_hold_covers_nothing(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    led = Ledger(tmp_path / "ledger2.jsonl")
    led.append("stage_end", 3, result="pass")           # the stage passed, but no entry vouches for its marker
    assert full_port_stage4.covering_stage(led, run, 4) is None


# ---- the plan -----------------------------------------------------------------------------------------------------

def test_the_plan_has_one_entry_per_distinct_count_in_order_and_refuses_what_cannot_run():
    entries, problems = full_port_entries.plan_entries([4, 1, 3, 1], max_chips=4, budget_s=28800.0)
    assert not problems and [e.chips for e in entries] == [1, 3, 4] and [e.boards for e in entries] == [1, 2, 2]
    assert entries[0].command(4) == full_port.command(4, 1) == "python3 -P -s stages/4/configs/1/port_check.py"
    for bad, word in (([], "no chip count"), ([0], "whole number"), ([True], "whole number"), ([8], "more chips than"), ([1, 2, 3, 4], "budget")):
        entries, problems = full_port_entries.plan_entries(bad, max_chips=4, budget_s=9000.0)
        assert problems and word in " ".join(problems) and entries == []


def test_a_plan_file_that_is_not_the_supervisors_shape_is_not_used(tmp_path):
    for doc in ({"tests": []}, {"tests": [{"chips": 1, "script": "serve_and_compare.py", "deadline_s": 60, "cache": ""}]},
                {"tests": [{"chips": 3, "script": "port_check.py", "deadline_s": 60, "cache": ""}, {"chips": 1, "script": "port_check.py", "deadline_s": 60, "cache": ""}]},
                {"tests": [{"chips": 1, "script": "port_check.py", "deadline_s": 60, "cache": "/x"}]}, [1], {"tests": "x"}):
        (tmp_path / "tests").mkdir(exist_ok=True)
        (tmp_path / "tests" / "plan.json").write_text(json.dumps(doc))
        entries, why = full_port_entries.load_plan(tmp_path)
        assert entries is None and why
    entries, why = full_port_entries.load_plan(tmp_path / "nowhere")
    assert entries is None and "missing" in why


# ---- the entry form of the check ----------------------------------------------------------------------------------

def test_the_entry_form_needs_stage_4_and_a_whole_count_and_its_count_is_the_expected_chips(tmp_path):
    stage_dir, run = stage4_run(tmp_path)
    edir = stage_dir / "configs/1"
    assert full_port.check(edir, run, stage=4, entry=1)[0] == []
    assert "entry 1 of stage 3" in full_port.check(edir, run, stage=3, entry=1)[0][0]
    assert "entry 0 of stage 4" in full_port.check(edir, run, stage=4, entry=0)[0][0]
    assert "entry True" in full_port.check(edir, run, stage=4, entry=True)[0][0]
    assert "expected count of 2" in full_port.check(edir, run, stage=4, entry=1, expected_chips=2)[0][0]


def test_the_stage_helpers_name_an_entrys_own_paths():
    assert full_port.port_dir(4, 2) == "stages/4/configs/2/port/" and full_port.port_module(4, 2) == "stages/4/configs/2/port/port_model.py"
    assert full_port.command(4, 2) == "python3 -P -s stages/4/configs/2/port_check.py"
    assert (full_port.port_dir(3), full_port.command(2)) == ("stages/3/port/", full_port.COMMAND)          # the stage forms are unchanged


def test_the_template_finds_the_run_directory_from_an_entry_directory(tmp_path):
    # The template once took the run directory as two levels above its own; an entry is four levels below the run.
    text = full_port.TEMPLATE.read_text()
    import re
    ns = {"Path": Path, "__file__": str(tmp_path / "run" / "stages" / "4" / "configs" / "3" / "port_check.py")}
    exec(re.search(r"^HERE = .*\nRUN_DIR = .*$", text, re.M).group(0), ns)
    assert ns["RUN_DIR"] == tmp_path.resolve() / "run"
    ns["__file__"] = str(tmp_path / "run" / "stages" / "2" / "port_check.py")
    exec(re.search(r"^HERE = .*\nRUN_DIR = .*$", text, re.M).group(0), ns)
    assert ns["RUN_DIR"] == tmp_path.resolve() / "run"


# ---- routing and the skill ----------------------------------------------------------------------------------------

def test_with_the_opt_in_a_full_port_class_gets_the_full_port_stage_4():
    s = spec_for(4, "full-port", None, "full-port", full_port=True)
    assert s is FULL_PORT_STAGE_4
    assert s.skill == "full-port-shrink" and s.gate is gate_full_port_4 and s.draft == "full_port_entries" and s.gate_file == "result.json"
    assert s.marker == "tests/plan.json" and s.boards == 2 and s.skip is None and not s.tests


def test_without_the_opt_in_or_for_another_route_stage_4_is_the_generic_or_its_own_stage():
    assert spec_for(4, "full-port", None, "full-port") == STAGES[4]
    assert spec_for(4, "full-port", None, "unknown", full_port=True) == STAGES[4]
    assert spec_for(4, "full-port", None, "full-port", full_port=True, kind="reranker") == STAGES[4]
    assert spec_for(4, "weights-only", None, "weights-only", full_port=True).skill == "weights-swap-configs"


def test_the_skip_reason_names_the_option():
    assert "required-chips" in SKIP_4_FULL_PORT


def skill_text():
    return (SKILLS / "full-port-shrink.md").read_text()


def test_the_skill_names_the_entry_paths_the_fixed_command_and_what_not_to_edit():
    text = skill_text()
    assert "stages/4/configs/<N>/port_check.py" in text and "stages/4/configs/<N>/" in text and "full-port-check.md" in text
    assert "Do not edit anything under `stages/2/` or `stages/3/`" in text and "no repair rounds" in text.lower()
    assert full_port.COMMAND not in text and full_port.command(3) not in text
