# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A hardware-test plan that is there and refused is not a missing plan (issue 386): the ledger says why before the prepare step replaces it."""
import pytest

from orchard.hwtests import plan_refusal
from orchard.ledger import Ledger
from orchard.stages import WEIGHTS_ONLY_STAGE_4, evidence_record, open_stage_dir


class _Stop(Exception):
    pass


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "run" / "ledger.jsonl") as led:
        yield led


def _supervisor_at_the_prepare_step(tmp_path, ledger):
    """A Supervisor stripped to what _run_test_list touches before its prepare step, which stops the call."""
    from orchard.supervisor import Supervisor
    sup = Supervisor.__new__(Supervisor)
    sup.run_dir = tmp_path / "run"
    sup.ledger = ledger
    sup.lab = None
    sup.paths = type("P", (), {"tt_model_root": tmp_path / "models", "operator_home": tmp_path / "op"})()

    def stop(spec, stage_dir):
        raise _Stop
    sup._draft = stop
    return sup


def _resumed_stage_4(tmp_path, ledger, body):
    """Stage 4 with a plan file `body` that the supervisor recorded (so the marker check passes on its hash)."""
    run = tmp_path / "run"
    d, _ = open_stage_dir(run, WEIGHTS_ONLY_STAGE_4, resuming=False, ledger=ledger)
    (d / "tests").mkdir(parents=True, exist_ok=True)
    (d / "tests" / "plan.json").write_text(body)
    ledger.append("evidence", 4, what="hardware test list", **evidence_record(run, d / "tests" / "plan.json"))
    d, resumed = open_stage_dir(run, WEIGHTS_ONLY_STAGE_4, resuming=True, ledger=ledger)
    assert resumed
    return d


@pytest.mark.parametrize("body,why", [
    ('{"tests": [{"chips": NaN}]}', "NaN"),
    ('{"tests": [{"chips": 2}]}', "not a list of tests"),
])
def test_a_refused_plan_is_recorded_before_the_prepare_step_replaces_it(tmp_path, ledger, body, why):
    d = _resumed_stage_4(tmp_path, ledger, body)
    assert why in plan_refusal(d)
    sup = _supervisor_at_the_prepare_step(tmp_path, ledger)
    with pytest.raises(_Stop):
        sup._run_test_list(WEIGHTS_ONLY_STAGE_4, d, False, True)
    said = [e["data"] for e in ledger.read() if str(e["data"].get("decision", "")).startswith("hardware test plan refused")]
    assert len(said) == 1 and said[0]["file"] == "tests/plan.json" and why in said[0]["reason"], said


def test_a_missing_plan_is_not_recorded_as_refused(tmp_path, ledger):
    run = tmp_path / "run"
    d, _ = open_stage_dir(run, WEIGHTS_ONLY_STAGE_4, resuming=False, ledger=ledger)
    assert plan_refusal(d) is None
    sup = _supervisor_at_the_prepare_step(tmp_path, ledger)
    with pytest.raises(_Stop):
        sup._run_test_list(WEIGHTS_ONLY_STAGE_4, d, False, True)
    assert not any("refused" in str(e["data"].get("decision", "")) for e in ledger.read())


# ---- issue 344, point 3: stage 7 files copied into the bundle ----------------------------------------------------------------

def test_copy_package_skips_a_link_that_leaves_the_run_and_a_fifo(tmp_path):
    import os
    from types import SimpleNamespace
    from orchard.supervisor import Supervisor
    run = tmp_path / "run"
    s7 = run / "stages" / "7"
    s7.mkdir(parents=True)
    outside = tmp_path / "secret.txt"
    outside.write_text("outside the run")
    os.symlink(outside, s7 / "package.json")
    os.mkfifo(s7 / "PUBLISH_COMMANDS.txt")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    Supervisor._copy_package(SimpleNamespace(run_dir=run), bundle)
    assert not (bundle / "package").exists()
    (s7 / "package.json").unlink()
    (s7 / "package.json").write_text("{}")
    Supervisor._copy_package(SimpleNamespace(run_dir=run), bundle)
    assert (bundle / "package" / "package.json").read_text() == "{}"
    assert not (bundle / "package" / "PUBLISH_COMMANDS.txt").exists()
