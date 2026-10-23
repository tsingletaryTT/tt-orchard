# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The graft trial in the supervisor: after stage 2's own test passes its gate, a model with a graft candidate gets one more hardware test
that times the safe configuration and the grafted variant, and the supervisor records which one the run ships. A failed trial is not a
stage failure. The trial's command is replaced by a fake that writes the numbers a real one would; every other command runs for real."""
import hashlib
import json
from pathlib import Path

import pytest

from orchard import defaults, supervisor
from orchard.graft_trial import DECISION_FILE, TRIAL_DIR, TRIAL_RESULT, chosen
from orchard.supervisor import EXIT_READY
from fakes import Crash
from run_fakes import DELTA, FILES, bringup
from test_supervisor import Rig, Stop

pytestmark = pytest.mark.usefixtures("stub_tools")

CANDIDATE = {"base_tensors": 15, "base_revision": "a" * 40, "base_shards": ["model-00018-of-00018.safetensors"], "dtypes": ["BF16"],
             "hidden_size": 5120, "shapes_match": True, "graftable": True}
GRAFT_DELTA = {(0, "run"): {**FILES[(0, "run")], "delta.json": {**DELTA, "graft_candidate": CANDIDATE}}}


def trial_result(*, safe=20.0, variant=60.0, top1=0.94, coherent=True, served=True):
    return {"label": "measured", "steps": [], "safe": {"server_ready_s": 100.0, "decode_tok_s": safe, "ttft_s": 0.3},
            "variant": {"served": served, "coherent": coherent, "top1_agreement": top1, "n_tokens": 32, "server_ready_s": 240.0,
                        "decode_tok_s": variant, "ttft_s": 0.2, "failure": None if served else "the server exited"},
            "evidence": []}


class TrialSpy:
    """Replaces spawn_checked. The graft trial's command writes the result the test says; anything else runs for real."""

    def __init__(self, real, result=None, code=0, timed_out=False, write_anyway=False):
        self.real, self.result, self.code, self.timed_out = real, result if result is not None else trial_result(), code, timed_out
        self.write_anyway = write_anyway           # a trial that left a result and then failed or timed out
        self.trials, self.commands = [], []

    def __call__(self, command, run_dir, env, timeout, out):
        self.commands.append(command)
        if "graft_trial.py" not in command:
            return self.real(command, run_dir, env, timeout, out)
        self.trials.append({"command": command, "timeout": timeout, "env": dict(env)})
        out.write(b"trial output\n")
        if (self.write_anyway or (self.code == 0 and not self.timed_out)) and self.result is not False:
            d = Path(run_dir) / "stages" / "2" / TRIAL_DIR
            (d / TRIAL_RESULT).write_text(json.dumps(self.result))
        return self.code, self.timed_out


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    r.script = lambda req: bringup(req, overrides=GRAFT_DELTA)
    yield r
    r.close()


@pytest.fixture
def spy(monkeypatch):
    s = TrialSpy(supervisor.spawn_checked)
    monkeypatch.setattr(supervisor, "spawn_checked", s)
    return s


def stage2(rig, decision):
    return [e["data"] for e in rig.entries() if e["stage"] == 2 and e["event"] == "decision" and e["data"].get("decision") == decision]


def sd(rig):
    return rig.run_dir / "stages" / "2"


# ---- the trial runs, and the decision follows the numbers ---------------------------------------------------------------------------

def test_a_graft_candidate_gets_one_more_hardware_test_after_stage_2_passes_its_gate(rig, spy):
    assert rig.run() == EXIT_READY
    assert len(spy.trials) == 1
    assert spy.trials[0]["command"] == f"python3 stages/2/{TRIAL_DIR}/graft_trial.py"
    assert [d["result"] for d in rig.ends(2)] == ["pass"]
    # Stage 2's own test is untouched: its record still holds its own command.
    own = json.loads((sd(rig) / "test-result.json").read_text())
    assert "graft_trial.py" not in own["command"]
    # The ledger shows both tests, the trial's after the agent's finish step.
    started = [d["command"] for d in stage2(rig, "hardware test started")]
    assert len(started) == 2 and "graft_trial.py" in started[1] and "graft_trial.py" not in started[0]


def test_the_trial_directory_holds_the_shipped_script_and_the_config_it_reads(rig, spy):
    rig.run()
    d = sd(rig) / TRIAL_DIR
    shipped = Path(supervisor.__file__).parent / "skills" / "weights-swap-templates" / "graft_trial.py"
    assert (d / "graft_trial.py").read_text() == shipped.read_text()
    conf = json.loads((d / "trial_config.json").read_text())
    start = next(e["data"] for e in rig.entries() if e["event"] == "run_start")
    assert conf == {"stage_dir": str(sd(rig)), "orchard_dir": start["paths"]["orchard_dir"]}


def test_a_fast_variant_is_chosen_and_the_decision_is_recorded_twice(rig, spy):
    assert rig.run() == EXIT_READY
    d = json.loads((sd(rig) / DECISION_FILE).read_text())
    assert d["choice"] == "graft" and d["numbers"]["speedup"] == pytest.approx(3.0) and d["candidate"] == CANDIDATE
    assert chosen(rig.run_dir) is True
    ledger = stage2(rig, "graft decision")
    assert len(ledger) == 1 and ledger[0]["choice"] == "graft" and ledger[0]["reason"] == d["reason"]
    assert ledger[0]["numbers"]["variant_decode_tok_s"] == 60.0
    assert ledger[0]["decision_file"]["path"] == f"stages/2/{DECISION_FILE}"
    assert len(ledger[0]["decision_file"]["sha256"]) == 64


def test_a_variant_that_is_not_faster_keeps_the_safe_configuration_and_the_stage_still_passes(rig, spy):
    spy.result = trial_result(variant=21.0)
    assert rig.run() == EXIT_READY
    d = json.loads((sd(rig) / DECISION_FILE).read_text())
    assert d["choice"] == "safe" and "only" in d["reason"]
    assert chosen(rig.run_dir) is False
    assert [d["result"] for d in rig.ends(2)] == ["pass"] and [d["result"] for d in rig.ends(8)] == ["pass"]


def test_a_trial_that_exits_non_zero_is_not_a_stage_failure(rig, spy):
    spy.code = 3
    assert rig.run() == EXIT_READY
    d = json.loads((sd(rig) / DECISION_FILE).read_text())
    assert d["choice"] == "safe" and "did not finish" in d["reason"] and "code 3" in d["reason"]
    assert [d["result"] for d in rig.ends(2)] == ["pass"]


def test_a_trial_that_times_out_is_not_a_stage_failure(rig, spy):
    spy.timed_out, spy.code = True, None
    assert rig.run() == EXIT_READY
    d = json.loads((sd(rig) / DECISION_FILE).read_text())
    assert d["choice"] == "safe" and "timed out" in d["reason"]
    assert [d["result"] for d in rig.ends(2)] == ["pass"]


def test_a_trial_that_wrote_no_result_keeps_the_safe_configuration(rig, spy):
    spy.result = False                      # exit 0, no trial-result.json
    assert rig.run() == EXIT_READY
    d = json.loads((sd(rig) / DECISION_FILE).read_text())
    assert d["choice"] == "safe" and TRIAL_RESULT in d["reason"]


def test_the_trial_has_its_own_deadline_and_never_more_than_the_stage_budget(rig, spy):
    rig.run()
    assert spy.trials[0]["timeout"] == min(defaults.GRAFT_TRIAL_DEADLINE_S, supervisor_budget(2))


def supervisor_budget(n):
    from orchard.stages import STAGES
    return next(s for s in STAGES if s.number == n).budget_s


def test_the_trial_test_gets_the_leased_chips_like_stage_2s_own_test(rig, spy):
    rig.run()
    env = spy.trials[0]["env"]
    assert env["TT_VISIBLE_DEVICES"] and env["TT_METAL_CACHE"].endswith("kernels")


def test_the_trial_test_is_recorded_as_a_graft_trial_in_the_ledger(rig, spy):
    rig.run()
    ev = [e["data"] for e in rig.entries() if e["stage"] == 2 and e["event"] == "evidence" and e["data"].get("what") == "graft trial test"]
    assert len(ev) == 1 and ev[0]["returncode"] == 0 and ev[0]["path"] == f"stages/2/{TRIAL_DIR}/test-result.json"
    own = [e["data"] for e in rig.entries() if e["stage"] == 2 and e["event"] == "evidence" and e["data"].get("what") == "hardware test"]
    assert len(own) == 1                                   # stage 2's own test keeps its entry


def test_the_trial_leases_chips_and_gives_them_back(rig, spy):
    rig.run()
    assert rig.m.leases == {} or all(owner == rig.m.owner_pid for _, owner in rig.m.leases.values())
    released = [d for d in stage2(rig, "test lease released")]
    taken = [d for d in stage2(rig, "test lease taken")]
    assert len(taken) == len(released) and len(taken) >= 1


# ---- when there is no trial ---------------------------------------------------------------------------------------------------------

def test_a_run_with_no_graft_candidate_has_no_trial_and_no_decision_file(tmp_path, monkeypatch):
    r = Rig(tmp_path)
    try:
        s = TrialSpy(supervisor.spawn_checked)
        monkeypatch.setattr(supervisor, "spawn_checked", s)
        assert r.run() == EXIT_READY
        assert s.trials == [] and not (sd(r) / DECISION_FILE).exists() and not (sd(r) / TRIAL_DIR).exists()
        assert stage2(r, "graft decision") == [] and chosen(r.run_dir) is False
    finally:
        r.close()


@pytest.mark.parametrize("change", [{"shapes_match": None}, {"shapes_match": False}, {"graftable": False}])
def test_a_candidate_that_may_not_fit_is_not_tried(tmp_path, monkeypatch, change):
    r = Rig(tmp_path)
    try:
        delta = {**DELTA, "graft_candidate": {**CANDIDATE, **change}}
        r.script = lambda req: bringup(req, overrides={(0, "run"): {**FILES[(0, "run")], "delta.json": delta}})
        s = TrialSpy(supervisor.spawn_checked)
        monkeypatch.setattr(supervisor, "spawn_checked", s)
        assert r.run() == EXIT_READY
        assert s.trials == []
    finally:
        r.close()


def test_the_opt_out_skips_the_trial_and_says_so(rig, spy):
    rig.args.no_graft_trial = True
    assert rig.run() == EXIT_READY
    assert spy.trials == []
    d = json.loads((sd(rig) / DECISION_FILE).read_text())
    assert d["choice"] == "safe" and "--no-graft-trial" in d["reason"] and d["trial"] is None
    assert [x["reason"] for x in stage2(rig, "graft trial skipped")] == [d["reason"]]
    assert chosen(rig.run_dir) is False


def test_the_opt_out_is_recorded_in_run_start_and_kept_on_a_resume(rig, spy):
    rig.args.no_graft_trial = True
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 1)
    start = next(e["data"] for e in rig.entries() if e["event"] == "run_start")
    assert start["graft_trial"] is False
    rig.args.no_graft_trial = False                    # the resume leaves the flag out
    assert rig.run(pid=200) == EXIT_READY
    assert spy.trials == []


def test_a_normal_run_start_records_that_the_trial_is_on(rig, spy):
    rig.run()
    assert next(e["data"] for e in rig.entries() if e["event"] == "run_start")["graft_trial"] is True


# ---- a kill and a resume --------------------------------------------------------------------------------------------------------

def test_a_kill_after_the_trial_test_resumes_without_repeating_it(rig, spy):
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "evidence" and e["data"].get("what") == "graft trial test")
    assert len(spy.trials) == 1
    assert rig.run(pid=200) == EXIT_READY
    assert len(spy.trials) == 1                               # the marker stopped a second boot
    assert json.loads((sd(rig) / DECISION_FILE).read_text())["choice"] == "graft"
    assert len(stage2(rig, "graft decision")) == 1


def test_a_kill_before_the_trial_test_ends_runs_it_again(rig, spy):
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "decision" and e["data"].get("decision") == "hardware test started"
                and "graft_trial.py" in e["data"].get("command", ""))
    assert len(spy.trials) == 0                               # killed as it was about to start
    assert rig.run(pid=200) == EXIT_READY
    assert len(spy.trials) == 1 and chosen(rig.run_dir) is True


def test_a_finished_stage_2_is_not_tried_again_on_a_resume(rig, spy):
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 2)
    assert len(spy.trials) == 1
    assert rig.run(pid=200) == EXIT_READY
    assert len(spy.trials) == 1 and len(stage2(rig, "graft decision")) == 1


# ---- the decision reaches the gate --------------------------------------------------------------------------------------------------

def test_the_stage_2_gate_refuses_a_graft_decision_the_trial_does_not_support(rig, spy):
    """The supervisor writes a consistent decision, so this edits the trial result after the fact and runs the gate again."""
    from orchard.stages import gate_weights_swap
    rig.run()
    assert gate_weights_swap(sd(rig), rig.run_dir).ok
    (sd(rig) / TRIAL_DIR / TRIAL_RESULT).write_text(json.dumps(trial_result(variant=19.0)))
    gate = gate_weights_swap(sd(rig), rig.run_dir)
    assert not gate.ok and any("do not support" in r for r in gate.reasons)


def test_a_lab_run_skips_the_trial_and_says_why(tmp_path, monkeypatch):
    """The trial's command opens this box's chips; a lab run's tests go to the lab, so the trial is left out with a recorded reason."""
    from test_supervisor_lab import LabRig
    r = LabRig(tmp_path)
    try:
        r.script = lambda req: bringup(req, overrides=GRAFT_DELTA)
        s = TrialSpy(supervisor.spawn_checked)
        monkeypatch.setattr(supervisor, "spawn_checked", s)
        assert r.run() == EXIT_READY
        assert s.trials == []
        d = json.loads((sd(r) / DECISION_FILE).read_text())
        assert d["choice"] == "safe" and "lab" in d["reason"] and d["trial"] is None
        assert [x["reason"] for x in stage2(r, "graft trial skipped")] == [d["reason"]]
    finally:
        r.close()


def test_the_run_flag_is_parsed(tmp_path):
    from run_fakes import argv, write_tiers
    tiers = write_tiers(tmp_path / "t.toml", "http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1")
    base = argv(tmp_path, tiers, "http://127.0.0.1:8000/v1")
    assert supervisor.parse(base).no_graft_trial is False
    assert supervisor.parse(base + ["--no-graft-trial"]).no_graft_trial is True


def narrated(run_dir):
    import io
    from datetime import timezone
    from orchard import narrate, ui
    out = io.StringIO()
    narrate.Narrator(run_dir, ui.PLAIN, out, replay=True, clock=lambda: 0.0, heartbeat_s=60, tz=timezone.utc).poll()
    import re
    # The narrator wraps long lines; one entry per time stamp puts each line back together.
    return [x.strip() for x in re.split(r"(?=\b\d\d:\d\d:\d\d )", " ".join(out.getvalue().split())) if x.strip()]


def test_the_trial_is_narrated_when_it_starts_and_when_it_is_decided(rig, spy):
    rig.run()
    lines = narrated(rig.run_dir)
    started = [x for x in lines if "graft trial" in x and "mtp" in x]
    assert len(started) == 1 and "beside the safe configuration" in started[0] and "minutes" in started[0]
    decided = [x for x in lines if "graft decision" in x]
    assert len(decided) == 1 and "kept" in decided[0] and "3.0 times" in decided[0]
    assert any("graft trial test passed" in x for x in lines)


def test_a_kept_safe_decision_is_narrated_with_its_reason(rig, spy):
    spy.result = trial_result(variant=21.0)
    rig.run()
    decided = [x for x in narrated(rig.run_dir) if "graft decision" in x]
    assert len(decided) == 1 and "safe configuration stays" in decided[0] and "only" in decided[0]


def test_a_failed_trial_test_is_narrated_as_failed(rig, spy):
    spy.code = 3
    rig.run()
    assert any("graft trial test failed (exit 3)" in x for x in narrated(rig.run_dir))


def test_a_skipped_trial_is_narrated_with_its_reason(rig, spy):
    rig.args.no_graft_trial = True
    rig.run()
    assert any("graft trial skipped" in x and "--no-graft-trial" in x for x in narrated(rig.run_dir))


def test_the_trials_output_goes_to_its_own_file_and_not_to_stage_2s(rig, spy):
    rig.run()
    assert (sd(rig) / TRIAL_DIR / "evidence" / "hw-test-output.txt").read_text() == "trial output\n"
    assert "trial output" not in (sd(rig) / "evidence" / "hw-test-output.txt").read_text()


def test_a_trial_that_left_a_result_and_then_exited_non_zero_is_not_believed(rig, spy):
    spy.code, spy.write_anyway = 3, True
    rig.run()
    d = json.loads((sd(rig) / DECISION_FILE).read_text())
    assert d["choice"] == "safe" and "code 3" in d["reason"]


def test_a_trial_that_left_a_result_and_then_timed_out_is_not_believed(rig, spy):
    spy.timed_out, spy.code, spy.write_anyway = True, None, True
    rig.run()
    d = json.loads((sd(rig) / DECISION_FILE).read_text())
    assert d["choice"] == "safe" and "timed out" in d["reason"]


def test_a_variant_that_lost_more_than_the_margin_against_stage_2s_own_run_is_not_chosen(rig, spy):
    """Stage 2's result.json says 0.94 (the fake's); 0.86 clears stage 2's bar of 0.85 and loses 0.08."""
    spy.result = trial_result(top1=0.86)
    rig.run()
    d = json.loads((sd(rig) / DECISION_FILE).read_text())
    assert d["choice"] == "safe" and "margin" in d["reason"] and d["numbers"]["safe_top1_agreement"] == 0.94


def test_a_failed_gate_does_not_start_the_trial_and_the_escalated_attempt_gets_one(rig, spy):
    from test_supervisor import escalation_aware
    from run_fakes import SWAP_LOW
    rig.script = escalation_aware(2, {(2, "finish"): SWAP_LOW})
    base = rig.script
    rig.script = lambda req: bringup(req, overrides={**GRAFT_DELTA, **({(2, "finish"): SWAP_LOW} if False else {})}) if False else base(req)
    # escalation_aware calls bringup without the graft delta, so add it back for stage 0
    def script(req):
        if "tools" in req and __import__("run_fakes").where(req) == (0, "run"):
            return bringup(req, overrides=GRAFT_DELTA)
        return base(req)
    rig.script = script
    assert rig.run() == EXIT_READY
    assert [d["result"] for d in rig.ends(2)] == ["escalate", "pass"]
    assert len(spy.trials) == 1 and len(stage2(rig, "graft decision")) == 1
    order = [(e["event"], e["data"].get("decision") or e["data"].get("result")) for e in rig.entries() if e["stage"] == 2]
    assert order.index(("stage_end", "escalate")) < order.index(("decision", "graft trial started"))


def test_a_decision_the_gate_does_not_accept_fails_the_stage_and_escalates(rig, spy, monkeypatch):
    """The supervisor's own decision is consistent. Force a file that says graft over a slow trial: the gate run after the trial refuses it."""
    real = supervisor.graft.write_decision

    def forced(stage_dir, decision, **kw):
        return real(stage_dir, {**decision, "choice": "graft"}, **kw)
    monkeypatch.setattr(supervisor.graft, "write_decision", forced)
    spy.result = trial_result(variant=19.0)
    def stop():
        raise Stop()                       # the escalated attempt fails the same way and the run pauses
    rig.on_sleep = stop
    with pytest.raises(Stop):
        rig.run()
    first = rig.ends(2)[0]
    assert first["result"] == "escalate" and any("do not support" in r for r in first["reasons"])


def test_a_full_port_run_has_no_trial_even_with_a_candidate_in_its_delta(rig, spy):
    delta = {**DELTA, "path": "full-port", "graft_candidate": CANDIDATE}
    rig.script = lambda req: bringup(req, overrides={(0, "run"): {**FILES[(0, "run")], "delta.json": delta}})
    rig.on_sleep = lambda: supervisor.Control(rig.run_dir).write("resume")
    assert rig.run() == EXIT_READY
    assert spy.trials == [] and not (sd(rig) / DECISION_FILE).exists()


def test_a_kill_after_the_decision_was_recorded_resumes_without_a_second_decision_or_test(rig, spy):
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "decision" and e["data"].get("decision") == "graft decision")
    assert rig.run(pid=200) == EXIT_READY
    assert len(spy.trials) == 1 and len(stage2(rig, "graft decision")) == 1
    assert [d["result"] for d in rig.ends(2)] == ["pass"]


def test_a_disk_too_full_for_the_variants_tensor_cache_skips_the_trial_and_says_so(rig, spy):
    """The variant converts a second tensor cache (34 GB measured for 2 chips). Short of space, the trial is not run, and the run goes on."""
    from collections import namedtuple
    from run_fakes import plenty
    sd2 = rig.run_dir / "stages" / "2"

    def usage(path):
        if (sd2 / "result.json").exists() and not (sd2 / DECISION_FILE).exists():
            return namedtuple("U", "total used free")(0, 0, (defaults.GRAFT_TRIAL_DISK_GB - 1) * 1e9)
        return plenty(path)
    rig.usage = usage
    assert rig.run() == EXIT_READY
    assert spy.trials == []
    d = json.loads((sd2 / DECISION_FILE).read_text())
    assert d["choice"] == "safe" and d["trial"] is None and "GB" in d["reason"] and "tensor cache" in d["reason"]
    assert stage2(rig, "graft trial skipped")[0]["reason"] == d["reason"]


def test_enough_disk_runs_the_trial(rig, spy):
    from collections import namedtuple
    sd2 = rig.run_dir / "stages" / "2"

    def usage(path):
        free = defaults.GRAFT_TRIAL_DISK_GB * 1e9 if (sd2 / "result.json").exists() and not (sd2 / DECISION_FILE).exists() else 10 ** 15
        return namedtuple("U", "total used free")(0, 0, free)
    rig.usage = usage
    assert rig.run() == EXIT_READY and len(spy.trials) == 1
