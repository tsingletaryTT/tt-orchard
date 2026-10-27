# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The sampling choice in the supervisor (issue 107): after stage 2 passes its gate (and the graft trial, when there is one), a model that needs sampling
and was served with the drafter on without sampling shown to work gets one more hardware test, the drafter-off variant. The supervisor records which
configuration the run ships. A failed trial is not a stage failure. The trial's command is replaced by a fake; every other command runs for real."""
import json
from pathlib import Path

import pytest

from orchard import defaults, sampling_choice as sc, supervisor
from orchard.supervisor import EXIT_READY
from fakes import Crash
from run_fakes import DELTA, FAKE_SWAP_TEST, FILES, bringup, hw, swap_prepare
from test_sampling_choice_files import report
from test_supervisor import Rig
from test_supervisor_graft import CANDIDATE, GRAFT_DELTA, TrialSpy as GraftSpy, trial_result

pytestmark = pytest.mark.usefixtures("stub_tools")

GENCONFIG_NEEDED = {"model_generation_config": {"temperature": 0.6, "top_p": 0.95}, "model_generation_config_present": True,
                    "card_sampling_hints": {}, "card_warns_greedy": []}
GENCONFIG_NOT = {"model_generation_config": {}, "model_generation_config_present": False, "card_sampling_hints": {}, "card_warns_greedy": []}


def overrides(*, genconfig=GENCONFIG_NEEDED, stage2_report=None, delta=None, extra2=None):
    """Stage 0 writes the genconfig evidence; stage 2's prepare step leaves a swap check (the fake hardware test does not run the real script)."""
    f0 = {**FILES[(0, "run")], "evidence/genconfig-license.json": genconfig, **({"delta.json": delta} if delta else {})}
    prep = {**hw(2), "evidence/swap-check.json": stage2_report if stage2_report is not None else report(drafter="on (dflash2)", ignored=True, seconds=1.6),
            **(extra2 or {})}
    # Stage 4's fake tests write a swap check with no `serving` record. When stage 2 chooses the drafter-off configuration, the stage 4 gate asks each
    # configuration for `serving.drafter: off`, so the fake tests say it (a keep decision asks nothing, and the extra key does no harm).
    off = FAKE_SWAP_TEST.replace('{"result_draft": {"top1_agreement": 0.94}}', '{"result_draft": {"top1_agreement": 0.94}, "serving": {"drafter": "off"}}')
    assert off != FAKE_SWAP_TEST
    prep4 = {k: (off if v == FAKE_SWAP_TEST else v) for k, v in swap_prepare().items()}
    return {(0, "run"): f0, (2, "prepare"): prep, (4, "prepare"): prep4}


def variant_files(*, drafter="off", ignored=False, seconds=6.0, serves=True):
    return report(drafter=drafter, ignored=ignored, seconds=seconds, serves=serves)


class TrialSpy(GraftSpy):
    """Handles the sampling trial's command: it writes trial-result.json and, when the variant serves, the variant's swap check. The graft trial's
    command is still handled by the parent."""

    def __init__(self, real, **kw):
        super().__init__(real, **kw)
        self.sampling_trials = []
        self.variant = variant_files()
        self.served = True
        self.s_code, self.s_timed_out, self.s_write = 0, False, True
        self.graft_variant = None

    def __call__(self, command, run_dir, env, timeout, out):
        if "sampling_trial.py" in command:
            self.commands.append(command)
            self.sampling_trials.append({"command": command, "timeout": timeout, "env": dict(env)})
            out.write(b"sampling trial output\n")
            d = Path(run_dir) / "stages" / "2" / sc.TRIAL_DIR
            if self.s_write and self.s_code == 0 and not self.s_timed_out:
                (d / "variant" / "evidence").mkdir(parents=True, exist_ok=True)
                if self.served:
                    (d / "variant" / "evidence" / "swap-check.json").write_text(json.dumps(self.variant))
                (d / sc.TRIAL_RESULT).write_text(json.dumps({"label": "measured", "steps": [], "evidence": [],
                                                              "variant": {"served": self.served, "failure": None if self.served else "the server exited"}}))
            return self.s_code, self.s_timed_out
        if "graft_trial.py" in command and self.graft_variant is not None:
            code = super().__call__(command, run_dir, env, timeout, out)
            d = Path(run_dir) / "stages" / "2" / "graft-trial" / "variant" / "evidence"
            d.mkdir(parents=True, exist_ok=True)
            (d / "swap-check.json").write_text(json.dumps(self.graft_variant))
            return code
        return super().__call__(command, run_dir, env, timeout, out)


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    r.script = lambda req: bringup(req, overrides=overrides())
    yield r
    r.close()


@pytest.fixture
def spy(monkeypatch):
    s = TrialSpy(supervisor.spawn_checked)
    monkeypatch.setattr(supervisor, "spawn_checked", s)
    return s


def sd(rig):
    return rig.run_dir / "stages" / "2"


def stage2(rig, decision):
    return [e["data"] for e in rig.entries() if e["stage"] == 2 and e["event"] == "decision" and e["data"].get("decision") == decision]


def decision_file(rig):
    return json.loads((sd(rig) / sc.DECISION_FILE).read_text())


# ---- the trial runs and the decision follows ----------------------------------------------------------------------------------------

def test_a_model_that_needs_sampling_and_was_served_with_a_drafter_that_ignores_it_gets_the_drafter_off_trial(rig, spy):
    assert rig.run() == EXIT_READY
    assert [t["command"] for t in spy.sampling_trials] == [f"python3 stages/2/{sc.TRIAL_DIR}/sampling_trial.py"]
    d = decision_file(rig)
    assert d["choice"] == "drafter-off" and d["flag"] is None and d["needed"] is True and d["need_source"] == "generation_config"
    assert d["trial"] == f"{sc.TRIAL_DIR}/{sc.TRIAL_RESULT}"
    assert [e["result"] for e in rig.ends(2)] == ["pass"]
    assert sc.drafter_off_chosen(rig.run_dir) is True


def test_the_trial_directory_holds_the_shipped_script_and_the_config_it_reads(rig, spy):
    rig.run()
    d = sd(rig) / sc.TRIAL_DIR
    shipped = Path(supervisor.__file__).parent / "skills" / "weights-swap-templates" / "sampling_trial.py"
    assert (d / "sampling_trial.py").read_text() == shipped.read_text()
    start = next(e["data"] for e in rig.entries() if e["event"] == "run_start")
    assert json.loads((d / "trial_config.json").read_text()) == {"stage_dir": str(sd(rig)), "orchard_dir": start["paths"]["orchard_dir"]}


def test_the_ledger_records_the_trial_and_the_decision_with_the_files_sha256(rig, spy):
    rig.run()
    assert len(stage2(rig, "sampling trial started")) == 1
    rec = stage2(rig, "sampling decision")
    assert len(rec) == 1 and rec[0]["choice"] == "drafter-off" and rec[0]["reason"] == decision_file(rig)["reason"]
    assert rec[0]["flag"] is None and rec[0]["needed"] is True and rec[0]["numbers"]["variant_tok_s"] == pytest.approx(16.0)
    assert rec[0]["decision_file"]["path"] == f"stages/2/{sc.DECISION_FILE}" and len(rec[0]["decision_file"]["sha256"]) == 64


def test_the_trial_test_is_its_own_record_and_leaves_stage_2s_test_alone(rig, spy):
    rig.run()
    ev = [e["data"] for e in rig.entries() if e["stage"] == 2 and e["event"] == "evidence" and e["data"].get("what") == "sampling trial test"]
    assert len(ev) == 1 and ev[0]["path"] == f"stages/2/{sc.TRIAL_DIR}/test-result.json" and ev[0]["returncode"] == 0
    assert len([e for e in rig.entries() if e["stage"] == 2 and e["event"] == "evidence" and e["data"].get("what") == "hardware test"]) == 1


def test_the_trial_gets_its_own_deadline_and_the_marker_variable_the_sweep_looks_for(rig, spy):
    rig.run()
    t = spy.sampling_trials[0]
    assert t["timeout"] == min(defaults.SAMPLING_TRIAL_DEADLINE_S, 14400.0)
    assert t["env"][supervisor.GRAFT_TRIAL_ENV] == str((sd(rig) / sc.TRIAL_DIR).resolve())
    assert t["env"]["TT_VISIBLE_DEVICES"]


def test_the_trial_leaves_no_servers_and_gives_the_chips_back(rig, spy):
    rig.run()
    taken, released = stage2(rig, "test lease taken"), stage2(rig, "test lease released")
    assert len(taken) == len(released) >= 1


# ---- no trial -------------------------------------------------------------------------------------------------------------------------

def test_a_model_that_does_not_need_sampling_is_not_tried_and_keeps_stage_2s_configuration(tmp_path, spy):
    r = Rig(tmp_path)
    try:
        r.script = lambda req: bringup(req, overrides=overrides(genconfig=GENCONFIG_NOT))
        assert r.run() == EXIT_READY
        assert spy.sampling_trials == []
        d = decision_file(r)
        assert d["choice"] == "keep" and d["flag"] is None and d["needed"] is False and d["trial"] is None
        assert sc.drafter_off_chosen(r.run_dir) is False
        assert len(stage2(r, "sampling decision")) == 1
    finally:
        r.close()


def test_a_served_configuration_that_applies_sampling_is_not_tried(tmp_path, spy):
    r = Rig(tmp_path)
    try:
        r.script = lambda req: bringup(req, overrides=overrides(stage2_report=report(drafter="off", ignored=False, seconds=6.0)))
        assert r.run() == EXIT_READY
        assert spy.sampling_trials == []
        d = decision_file(r)
        assert d["choice"] == "keep" and d["flag"] is None and d["needed"] is True
    finally:
        r.close()


def test_a_run_without_a_stage_2_swap_check_makes_no_decision_and_runs_no_trial(tmp_path, spy):
    """The stage 2 gate of a plain fake run has no swap check. Nothing to judge, so nothing is recorded and nothing is started."""
    r = Rig(tmp_path)
    try:
        assert r.run() == EXIT_READY
        assert spy.sampling_trials == [] and not (sd(r) / sc.DECISION_FILE).exists()
        assert stage2(r, "sampling decision") == []
    finally:
        r.close()


def test_the_operators_override_not_needed_skips_the_trial_for_a_model_with_evidence(rig, spy):
    rig.args.sampling = "not-needed"
    assert rig.run() == EXIT_READY
    assert spy.sampling_trials == []
    d = decision_file(rig)
    assert d["choice"] == "keep" and d["need_source"] == "operator" and d["override"] == "not-needed"


def test_the_operators_override_needed_runs_the_trial_for_a_model_with_no_evidence(tmp_path, spy):
    r = Rig(tmp_path)
    try:
        r.script = lambda req: bringup(req, overrides=overrides(genconfig=GENCONFIG_NOT))
        r.args.sampling = "needed"
        assert r.run() == EXIT_READY
        assert len(spy.sampling_trials) == 1
        d = decision_file(r)
        assert d["choice"] == "drafter-off" and d["need_source"] == "operator" and d["override"] == "needed"
    finally:
        r.close()


def test_a_container_configuration_is_not_tried_and_the_reason_is_recorded(tmp_path, spy):
    r = Rig(tmp_path)
    try:
        r.script = lambda req: bringup(req, overrides=overrides(extra2={"swap_config.json": {"package": "fake/pkg", "profile": "p2",
                                                                                           "tt_cache": "/nonexistent/c"}}))
        assert r.run() == EXIT_READY
        assert spy.sampling_trials == []
        d = decision_file(r)
        assert d["choice"] == "keep" and d["flag"] == "variant-not-tried" and "container" in d["reason"]
        assert [x["reason"] for x in stage2(r, "sampling trial skipped")] and "container" in stage2(r, "sampling trial skipped")[0]["reason"]
    finally:
        r.close()


def test_too_little_disk_skips_the_trial_and_says_so(rig, spy, monkeypatch):
    """Only the trial's own disk check sees a short disk (a distinct threshold), so the earlier stages' checks are not disturbed."""
    real = supervisor.check_disk
    monkeypatch.setattr(supervisor, "SAMPLING_TRIAL_DISK_GB", 12345.0)
    monkeypatch.setattr(supervisor, "check_disk", lambda path, gb, usage=None: (False, 7.0) if gb == 12345.0 else real(path, gb, usage=usage))
    assert rig.run() == EXIT_READY
    d = decision_file(rig)
    assert spy.sampling_trials == [] and d["flag"] == "variant-not-tried" and "12345 GB" in d["reason"] and "7.0 GB is free" in d["reason"]
    assert d["trial"] is None


def test_a_lab_run_skips_the_trial_and_says_why(tmp_path, monkeypatch):
    from test_supervisor_lab import LabRig
    r = LabRig(tmp_path)
    try:
        r.script = lambda req: bringup(req, overrides=overrides())
        s = TrialSpy(supervisor.spawn_checked)
        monkeypatch.setattr(supervisor, "spawn_checked", s)
        assert r.run() == EXIT_READY
        assert s.sampling_trials == []
        d = decision_file(r)
        assert d["choice"] == "keep" and d["flag"] == "variant-not-tried" and "lab" in d["reason"]
    finally:
        r.close()


# ---- the trial does not decide a stage ---------------------------------------------------------------------------------------------------

def test_a_variant_below_the_floor_keeps_the_fast_configuration_flags_it_and_the_stage_still_passes(rig, spy):
    spy.variant = variant_files(seconds=12.0)                    # 96 tokens in 12 s is 8 tokens/s, below the floor of 10
    assert rig.run() == EXIT_READY
    d = decision_file(rig)
    assert d["choice"] == "keep" and d["flag"] == "no-variant-serves-purpose" and "neither" in d["reason"]
    assert [e["result"] for e in rig.ends(2)] == ["pass"] and [e["result"] for e in rig.ends(8)] == ["pass"]


def test_a_variant_that_does_not_serve_keeps_the_fast_configuration_and_the_stage_still_passes(rig, spy):
    spy.served = False
    assert rig.run() == EXIT_READY
    d = decision_file(rig)
    assert d["choice"] == "keep" and d["flag"] == "no-variant-serves-purpose"


def test_a_variant_that_also_ignores_sampling_is_not_chosen(rig, spy):
    spy.variant = variant_files(ignored=True)
    assert rig.run() == EXIT_READY
    assert decision_file(rig)["choice"] == "keep" and decision_file(rig)["flag"] == "no-variant-serves-purpose"


def test_a_trial_that_exits_non_zero_is_not_a_stage_failure(rig, spy):
    spy.s_code = 3
    assert rig.run() == EXIT_READY
    d = decision_file(rig)
    assert d["choice"] == "keep" and d["flag"] == "variant-not-tried" and "did not finish" in d["reason"] and "code 3" in d["reason"]
    assert [e["result"] for e in rig.ends(2)] == ["pass"]


def test_a_trial_that_times_out_is_not_a_stage_failure(rig, spy):
    spy.s_timed_out, spy.s_code = True, None
    assert rig.run() == EXIT_READY
    d = decision_file(rig)
    assert d["flag"] == "variant-not-tried" and "timed out" in d["reason"]


def test_a_trial_that_wrote_no_result_is_recorded_and_keeps_the_served_configuration(rig, spy):
    spy.s_write = False
    assert rig.run() == EXIT_READY
    d = decision_file(rig)
    assert d["choice"] == "keep" and d["flag"] == "variant-not-tried" and sc.TRIAL_RESULT in d["reason"]


# ---- resume -----------------------------------------------------------------------------------------------------------------------------

def test_a_kill_after_the_trial_test_resumes_without_repeating_it(rig, spy):
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "evidence" and e["data"].get("what") == "sampling trial test")
    assert len(spy.sampling_trials) == 1
    assert rig.run(pid=200) == EXIT_READY
    assert len(spy.sampling_trials) == 1
    assert decision_file(rig)["choice"] == "drafter-off" and len(stage2(rig, "sampling decision")) == 1


def test_a_finished_stage_2_is_not_tried_again_on_a_resume(rig, spy):
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 2)
    assert len(spy.sampling_trials) == 1
    assert rig.run(pid=200) == EXIT_READY
    assert len(spy.sampling_trials) == 1 and len(stage2(rig, "sampling decision")) == 1


def test_a_kill_before_the_trial_test_starts_runs_it_after_the_resume(rig, spy):
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "decision" and e["data"].get("decision") == "hardware test started"
                and "sampling_trial.py" in e["data"].get("command", ""))
    assert spy.sampling_trials == []
    assert rig.run(pid=200) == EXIT_READY
    assert len(spy.sampling_trials) == 1 and sc.drafter_off_chosen(rig.run_dir) is True


# ---- the flag --------------------------------------------------------------------------------------------------------------------------

def test_the_flag_is_parsed_with_auto_as_the_default_and_three_choices(tmp_path):
    from run_fakes import argv, write_tiers
    tiers = write_tiers(tmp_path / "t.toml", "http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1")
    base = argv(tmp_path, tiers, "http://127.0.0.1:8000/v1")
    assert supervisor.parse(base).sampling is None
    for v in ("auto", "needed", "not-needed"):
        assert supervisor.parse(base + ["--sampling", v]).sampling == v
    with pytest.raises(SystemExit):
        supervisor.parse(base + ["--sampling", "maybe"])


def test_a_normal_run_start_records_the_setting(rig, spy):
    rig.run()
    assert next(e["data"] for e in rig.entries() if e["event"] == "run_start")["sampling"] == "auto"


def test_a_flag_is_recorded_in_run_start_and_kept_on_a_resume_that_leaves_it_out(rig, spy):
    rig.args.sampling = "not-needed"
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 1)
    assert next(e["data"] for e in rig.entries() if e["event"] == "run_start")["sampling"] == "not-needed"
    rig.args.sampling = None
    assert rig.run(pid=200) == EXIT_READY
    assert spy.sampling_trials == [] and decision_file(rig)["override"] == "not-needed"


def test_a_resume_with_a_different_flag_is_refused(rig, spy):
    rig.args.sampling = "not-needed"
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 1)
    rig.args.sampling = "needed"
    with pytest.raises(ValueError, match="sampling"):
        rig.run(pid=200)


def test_a_resume_with_the_same_flag_is_accepted(rig, spy):
    rig.args.sampling = "needed"
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 1)
    assert rig.run(pid=200) == EXIT_READY


# ---- the graft trial and the sampling choice, in order ------------------------------------------------------------------------------------

def graft_overrides(**kw):
    return overrides(delta={**DELTA, "graft_candidate": CANDIDATE}, **kw)


def test_a_graft_whose_variant_ignores_sampling_is_not_chosen_for_a_model_that_needs_it_and_no_drafter_off_trial_follows(tmp_path, monkeypatch):
    """Stage 2 served the drafter-off safe configuration (a graft candidate has no mtp tensors). The graft trial's variant, with the drafter on, ignored
    sampling. The graft is refused, and the sampling decision keeps the safe configuration, which applies sampling."""
    r = Rig(tmp_path)
    try:
        r.script = lambda req: bringup(req, overrides=graft_overrides(stage2_report=report(drafter="off", ignored=False, seconds=6.0)))
        s = TrialSpy(supervisor.spawn_checked)
        s.graft_variant = report(drafter="on (dflash2)", ignored=True, seconds=1.6)
        monkeypatch.setattr(supervisor, "spawn_checked", s)
        assert r.run() == EXIT_READY
        g = json.loads((sd(r) / "graft-decision.json").read_text())
        assert g["choice"] == "safe" and "sampling" in g["reason"] and g["numbers"]["variant_sampling_verdict"] == "sampled-ignored"
        assert s.sampling_trials == [] and len(s.trials) == 1
        d = decision_file(r)
        assert d["choice"] == "keep" and d["flag"] is None and d["numbers"]["served_drafter"] == "off"
    finally:
        r.close()


def test_a_graft_whose_variant_applies_sampling_is_chosen_and_the_sampling_decision_keeps_it(tmp_path, monkeypatch):
    r = Rig(tmp_path)
    try:
        r.script = lambda req: bringup(req, overrides=graft_overrides(stage2_report=report(drafter="off", ignored=False, seconds=6.0)))
        s = TrialSpy(supervisor.spawn_checked)
        s.graft_variant = report(drafter="on (dflash2)", ignored=False, seconds=1.6)
        monkeypatch.setattr(supervisor, "spawn_checked", s)
        assert r.run() == EXIT_READY
        assert json.loads((sd(r) / "graft-decision.json").read_text())["choice"] == "graft"
        d = decision_file(r)
        assert d["choice"] == "keep" and d["numbers"]["served_drafter"] == "on" and d["numbers"]["served_verdict"] == "sampled-full-speed"
        assert s.sampling_trials == []
    finally:
        r.close()


def test_a_graft_is_chosen_for_a_model_that_does_not_need_sampling_whatever_its_verdict(tmp_path, monkeypatch):
    r = Rig(tmp_path)
    try:
        r.script = lambda req: bringup(req, overrides=graft_overrides(genconfig=GENCONFIG_NOT, stage2_report=report(drafter="off", ignored=False, seconds=6.0)))
        s = TrialSpy(supervisor.spawn_checked)
        s.graft_variant = report(drafter="on (dflash2)", ignored=True, seconds=1.6)
        monkeypatch.setattr(supervisor, "spawn_checked", s)
        assert r.run() == EXIT_READY
        assert json.loads((sd(r) / "graft-decision.json").read_text())["choice"] == "graft"
        assert decision_file(r)["choice"] == "keep" and decision_file(r)["needed"] is False
    finally:
        r.close()


# ---- the gate ---------------------------------------------------------------------------------------------------------------------------

def test_the_stage_2_gate_refuses_a_decision_the_files_do_not_support(rig, spy):
    from orchard.stages import gate_weights_swap
    rig.run()
    assert gate_weights_swap(sd(rig), rig.run_dir).ok
    d = decision_file(rig)
    d["choice"] = "keep"
    (sd(rig) / sc.DECISION_FILE).write_text(json.dumps(d))
    gate = gate_weights_swap(sd(rig), rig.run_dir)
    assert not gate.ok and any(sc.DECISION_FILE in r for r in gate.reasons)
