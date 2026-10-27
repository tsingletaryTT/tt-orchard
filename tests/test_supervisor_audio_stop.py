# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""An audio run stops before stage 1 until the audio stages exist (plan docs/superpowers/plans/2026-10-28-audio-bringup.md, task A3).
Without the stop it would go on into the text weights-only stages, which measure text and would give an audio model a meaningless pass."""
import pytest

from orchard import blocked, narrate, status, supervisor
from orchard.supervisor import EXIT_BLOCKED, EXIT_READY
from run_fakes import DELTA, FILES, bringup
from test_supervisor import Control, Rig, Stop

pytestmark = pytest.mark.usefixtures("stub_tools")

AUDIO = {(0, "run"): {**FILES[(0, "run")], "delta.json": {**DELTA, "class": "weights-only",
                                                         "facts": {"form": "full", "modality": "audio"}}}}
TEXT = {(0, "run"): {**FILES[(0, "run")], "delta.json": {**DELTA, "facts": {"form": "full", "modality": "text"}}}}


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    r.on_sleep = lambda: (_ for _ in ()).throw(Stop("the run waited for the operator"))
    yield r
    r.close()


def decisions(rig, name="pause"):
    return [d for d in rig.decisions() if d.get("decision") == name]


def started(rig, n):
    return [e for e in rig.entries() if e["event"] == "stage_start" and e["stage"] == n]


def test_an_attended_audio_run_pauses_before_stage_1(rig):
    rig.script = lambda r: bringup(r, overrides=AUDIO)
    with pytest.raises(Stop):
        rig.run()
    pauses = decisions(rig)
    assert pauses and supervisor.AUDIO_UNBUILT == pauses[-1]["reason"]
    assert not started(rig, 1) and started(rig, 0)


def test_an_operator_resume_acknowledges_the_stop_once(rig):
    rig.script = lambda r: bringup(r, overrides=AUDIO)
    with pytest.raises(Stop):
        rig.run()
    seen = []

    def resume():
        seen.append(1)
        if len(seen) == 1:
            Control(rig.run_dir).write("resume")
        else:
            raise Stop("a second wait")
    rig.on_sleep = resume
    assert rig.run() == EXIT_READY
    assert started(rig, 1), "after the operator's resume the run goes on"
    assert len(decisions(rig)) == 1


def test_an_unattended_audio_run_blocks_with_its_own_code(rig):
    rig.args.unattended = True
    rig.script = lambda r: bringup(r, overrides=AUDIO)
    assert rig.run() == EXIT_BLOCKED
    assert decisions(rig, "blocked")[-1]["code"] == "audio-path-unbuilt"
    assert not started(rig, 1)


def test_an_unattended_retry_does_not_pass_the_stop(rig):
    rig.args.unattended = True
    rig.script = lambda r: bringup(r, overrides=AUDIO)
    assert rig.run() == EXIT_BLOCKED
    assert rig.run() == EXIT_BLOCKED
    assert not started(rig, 1)


def test_a_text_run_is_not_stopped(rig):
    rig.script = lambda r: bringup(r, overrides=TEXT)
    assert rig.run() == EXIT_READY


def test_a_run_with_no_facts_is_not_stopped(rig):
    assert rig.run() == EXIT_READY


def test_the_pause_is_classified_and_the_code_has_an_unblock_text():
    assert status.classify_pause({"reason": supervisor.AUDIO_UNBUILT})[0] == "audio_unbuilt"
    assert blocked.block_code({"reason": supervisor.AUDIO_UNBUILT}) == "audio-path-unbuilt"
    assert "audio-path-unbuilt" in blocked.REASONS and "audio-path-unbuilt" in narrate.UNBLOCK
    assert "audio-path-unbuilt" not in blocked.AUTO_RETRY


def test_the_status_hint_says_a_person_decides():
    facts = {"state": "paused", "pause": {"kind": "audio_unbuilt"}, "control_pending": False, "disk": {"run_dir_free_gb": 500}, "pauses_at_stage": 1, "quiet_s": 0}
    hint = next(h for name, test, h in status.HINT_RULES if test(facts))
    assert "needs a decision" in hint
