# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Tests added after the first mutation table for the sampling choice (issue 107): each one closes a guard whose removal left every earlier test green."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sampling_choice import NEEDED, NOT_NEEDED, side, variant, write_genconfig  # noqa: E402
from test_sampling_choice_files import make_run, report, save, write  # noqa: E402

from orchard import narrate, sampling_choice as sc  # noqa: E402
from orchard.supervisor import EXIT_READY  # noqa: E402

pytestmark = pytest.mark.usefixtures("stub_tools")


# ---- the rule ---------------------------------------------------------------------------------------------------------------------

def test_a_generation_config_listed_without_the_present_flag_is_not_the_models_own(tmp_path):
    write_genconfig(tmp_path, model_generation_config={"temperature": 0.6}, model_generation_config_present=False)
    assert sc.need(tmp_path)["needed"] is False


@pytest.mark.parametrize("cfg", [{"top_p": 0}, {"top_p": -0.5}])
def test_a_top_p_of_zero_or_below_is_no_sampling_default(tmp_path, cfg):
    write_genconfig(tmp_path, model_generation_config=cfg, model_generation_config_present=True)
    assert sc.need(tmp_path)["needed"] is False


def test_a_variant_exactly_at_the_floor_clears_it():
    d = sc.decide(NEEDED, side(), variant(greedy=10.0, sampled=10.0))
    assert d["choice"] == "drafter-off" and d["flag"] is None


def test_a_sampled_speed_of_zero_falls_back_to_the_greedy_speed():
    d = sc.decide(NEEDED, side(drafter="off", verdict="sampled-full-speed", greedy=18.0, sampled=0))
    assert d["numbers"]["served_tok_s"] == 18.0 and d["flag"] is None


def test_a_trial_that_says_it_served_with_no_swap_check_gives_a_served_side_with_no_verdict(tmp_path):
    run, s2 = make_run(tmp_path)
    write(s2 / sc.TRIAL_DIR / sc.TRIAL_RESULT, {"variant": {"served": True}})
    assert sc.variant_side(s2) == {"drafter": "off", "verdict": "not-run", "greedy_tok_s": None, "sampled_tok_s": None, "served": True}


def test_a_flag_that_is_not_one_the_rule_gives_is_refused_by_the_recomputation(tmp_path):
    run, s2 = make_run(tmp_path, needed=False)
    d = save(run, s2)
    d["flag"] = "loud"
    (s2 / sc.DECISION_FILE).write_text(json.dumps(d))
    assert any("'loud'" in p for p in sc.decision_problems(s2, run))


# ---- the sidecar gate reads the decision too ---------------------------------------------------------------------------------------

def test_the_sidecar_class_gate_refuses_a_forged_decision(tmp_path):
    from orchard.stages import gate_weights_swap_sidecar
    run, s2 = make_run(tmp_path, variant=report(drafter="off", ignored=False, seconds=6.0))
    d = save(run, s2)
    d["choice"] = "keep"
    (s2 / sc.DECISION_FILE).write_text(json.dumps(d))
    write(s2 / "result.json", {"serves": True})
    g = gate_weights_swap_sidecar(s2, run)
    assert any(sc.DECISION_FILE in r for r in g.reasons)


# ---- narration marks a flag as a problem --------------------------------------------------------------------------------------------

def test_a_flagged_decision_line_is_painted_as_a_problem_and_an_unflagged_one_is_not():
    def line(flag):
        e = {"event": "decision", "stage": 2, "seq": 1, "ts": "2026-10-07T05:20:31Z",
             "data": {"decision": "sampling decision", "choice": "keep", "flag": flag, "reason": "r"}}
        return narrate.lines_for(e)[0]
    assert line("unverified").bad is True and line(None).bad is False


# ---- the supervisor ---------------------------------------------------------------------------------------------------------------

from test_supervisor_sampling import TrialSpy, decision_file, overrides, rig, sd, spy, stage2  # noqa: E402,F401
from run_fakes import bringup  # noqa: E402
from fakes import Crash  # noqa: E402
from orchard import supervisor  # noqa: E402


def test_a_kill_right_after_the_decision_resumes_without_a_second_decision(rig, spy):
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "decision" and e["data"].get("decision") == "sampling decision")
    assert rig.run(pid=200) == EXIT_READY
    assert len(stage2(rig, "sampling decision")) == 1 and len(spy.sampling_trials) == 1


def test_a_timed_out_test_is_a_timeout_even_with_exit_code_zero(rig, spy):
    spy.s_code, spy.s_timed_out = 0, True
    assert rig.run() == EXIT_READY
    assert "timed out" in decision_file(rig)["reason"]


def test_the_trials_servers_are_swept_after_the_test(rig, spy, monkeypatch):
    calls = []
    monkeypatch.setattr(supervisor.Supervisor, "_stop_trial_servers", lambda self, n, d, when: calls.append((n, Path(d).name, when)))
    rig.run()
    assert (2, "sampling-trial", "after the sampling trial test") in calls


def test_the_stage_2_gate_is_checked_again_after_a_decision_is_written(rig, spy, monkeypatch):
    seen = []
    real = supervisor.Supervisor._check_gate

    def spying(self, spec, stage_dir):
        if spec.number == 2:
            seen.append((stage_dir / sc.DECISION_FILE).exists())
        return real(self, spec, stage_dir)
    monkeypatch.setattr(supervisor.Supervisor, "_check_gate", spying)
    rig.run()
    assert seen[-1] is True and False in seen          # once before the decision, and again after it


# ---- the package --------------------------------------------------------------------------------------------------------------------

from test_package_sampling_choice import drafter_on_run, section, stage_card, world  # noqa: E402,F401
from test_package_graft_auto import rows  # noqa: E402
from orchard.stages import gate_package  # noqa: E402


def test_the_gate_refuses_a_package_that_kept_the_drafter_after_a_drafter_off_decision(world):
    drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    run_sh = pkg / "run.sh"
    run_sh.write_text(run_sh.read_text().replace("CMD=(", 'export QWEN36_DRAFTER="dflash2"\nCMD=(', 1))
    g = gate_package(stage_dir, world["run"])
    assert not g.ok and any("drafter" in r for r in g.reasons), g.reasons


def test_the_gate_refuses_the_missing_mtp_head_reason_after_a_sampling_choice(world):
    drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    bullet = next(ln for ln in card.splitlines() if ln.startswith("- Speculative decoding is off. Stage 2 chose"))
    nearest = json.loads((pkg / "tt_kernel_manifest.json").read_text())["weights"]["repo_id"]
    (pkg / "README.md").write_text(card.replace(bullet, "- Speculative decoding is off. The drafter of X needs the MTP tensors of the model, and Y has none."))
    g = gate_package(stage_dir, world["run"])
    assert not g.ok and any("mtp" in r.lower() for r in g.reasons), g.reasons


def test_a_decision_with_no_probe_record_still_gets_a_sampling_section_and_the_gate_accepts_it(world):
    dec = drafter_on_run(world, needed=False)
    swap = world["run"] / "stages/2/evidence/swap-check.json"
    d = json.loads(swap.read_text())
    d.pop("sampling_probe")
    swap.write_text(json.dumps(d))
    sc.write_decision(world["run"] / "stages/2", sc.decide(sc.need(world["run"]), sc.served_side(world["run"] / "stages/2"), None), trial=False)
    stage_dir, pkg, card = stage_card(world)
    assert "kept the configuration it served" in " ".join(section(card).split())
    assert gate_package(stage_dir, world["run"]).ok


def test_the_decision_sentence_stands_in_its_own_paragraph_after_the_probe_text(world):
    dec = drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    assert "\n\n" + sc.card_sentence(dec) in card


# ---- the bundle -------------------------------------------------------------------------------------------------------------------

def test_the_bundle_labels_the_floor_as_a_choice(tmp_path):
    from test_operator_bundle_sampling_choice import files, put
    from test_operator_bundle_template import make_run as make_bundle_run
    run = make_bundle_run(tmp_path)
    put(run, sc.decide(NEEDED, side(), variant()))
    results, _ = files(run)
    row = next(ln for ln in results.splitlines() if ln.startswith("| Usability floor"))
    assert "| choice |" in row and "| measured |" not in row
