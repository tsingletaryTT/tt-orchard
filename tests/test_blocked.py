"""The blocked end state (orchard/blocked.py): which pauses an unattended run turns into a named block,
and the bundle it leaves. Pure functions of ledger entries, so no supervisor is needed here."""
import json

import pytest

from orchard import blocked
from orchard.supervisor import FULL_PORT

T0 = 1_791_000_000


def pause(reason, **extra):
    return {"decision": "pause", "reason": reason, **extra}


@pytest.mark.parametrize("data,code", [
    (pause(FULL_PORT), "needs-new-model-code"),
    (pause("3 escalations since the last resume (cap 3)"), "retry-budget-spent"),
    (pause("3 cold coder boots since the last resume (cap 3)"), "retry-budget-spent"),
    (pause("the run has lasted 90000 s since it started or was last resumed (cap 86400 s)"), "retry-budget-spent"),
    (pause("stage 2 failed after escalation: serves is false"), "stage-failed"),
    (pause("stage 7 (package) failed; it is supervisor code and is not escalated: x"), "stage-failed"),
    (pause("watchdog: 20 model turns and no file written", watchdog=True,
           findings=[{"detector": "no_file_written", "summary": "s", "evidence": {}}]), "agent-stuck"),
    (pause("blocked: only 12 GB free on the run disk, the stage needs 40"), "disk-full"),
    (pause("blocked: the lease could not be released: chip hung"), "hardware-unhealthy"),
    (pause("blocked: gozer reset failed on board 0"), "hardware-unhealthy"),
    (pause("blocked: the coder canary failed"), "coder-unusable"),
    (pause("blocked: server did not become healthy"), "coder-unusable"),
    (pause("blocked: something nobody wrote a rule for"), "blocked"),
    (pause("a reason nobody classified"), "unclassified"),
])
def test_each_pause_maps_to_one_named_block_reason(data, code):
    assert blocked.block_code(data) == code


def test_an_operator_pause_is_never_a_block():
    assert blocked.block_code(pause("operator")) is None


def test_every_code_is_in_the_documented_set():
    assert {"needs-new-model-code", "retry-budget-spent", "stage-failed", "agent-stuck", "disk-full",
            "hardware-unhealthy", "coder-unusable", "blocked", "unclassified"} <= set(blocked.REASONS)


def entry(seq, event, stage, **data):
    return {"seq": seq, "ts": time_str(T0 + seq * 60), "event": event, "stage": stage, "data": data}


def time_str(t):
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


ENTRIES = [
    entry(0, "run_start", None, model="Cloudflare/clef", versions={}, inputs={}),
    entry(1, "stage_start", 0), entry(2, "stage_end", 0, result="pass", path="weights-only"),
    entry(3, "stage_start", 1), entry(4, "stage_end", 1, result="pass"),
    entry(5, "stage_start", 2), entry(6, "stage_end", 2, result="escalate", reasons=["serves is false"]),
    entry(7, "stage_start", 2), entry(8, "stage_end", 2, result="fail", reasons=["serves is false"]),
    entry(9, "decision", 2, **pause("stage 2 failed after escalation: serves is false")),
]


def test_the_bundle_names_the_model_the_reason_the_stage_and_how_to_retry(tmp_path):
    blocked.write_bundle(tmp_path, ENTRIES, "stage-failed", "stage 2 failed after escalation: serves is false",
                         now=T0 + 3600)
    text = (tmp_path / "BLOCKED.md").read_text()
    for needle in ("Cloudflare/clef", "stage-failed", "stage 2 failed after escalation", "tt-orchard bringup Cloudflare/clef"):
        assert needle in text
    assert "pass" in text and "fail" in text                     # the stage table
    assert "never published" in text or "nothing was published" in text.lower()


def test_the_bundle_has_a_machine_readable_twin(tmp_path):
    blocked.write_bundle(tmp_path, ENTRIES, "stage-failed", "why", now=T0 + 3600)
    j = json.loads((tmp_path / "blocked.json").read_text())
    assert (j["code"], j["reason"], j["model"], j["stage"]) == ("stage-failed", "why", "Cloudflare/clef", 2)
    assert j["stages"][0] == {"stage": 0, "name": "intake and delta triage", "status": "pass", "wall_s": 60, "attempts": 1}


def test_a_second_bundle_replaces_the_first_and_is_never_appended(tmp_path):
    blocked.write_bundle(tmp_path, ENTRIES, "stage-failed", "first", now=T0)
    blocked.write_bundle(tmp_path, ENTRIES, "disk-full", "second", now=T0)
    text = (tmp_path / "BLOCKED.md").read_text()
    assert "second" in text and "first" not in text


def test_a_run_that_never_started_still_gets_a_bundle(tmp_path):
    blocked.write_bundle(tmp_path, [], "unclassified", "no ledger", now=T0)
    assert "no ledger" in (tmp_path / "BLOCKED.md").read_text()


def test_the_last_pause_is_found_in_the_ledger():
    assert blocked.last_pause(ENTRIES) == pause("stage 2 failed after escalation: serves is false")
    assert blocked.last_pause(ENTRIES[:3]) is None


def test_a_pause_that_was_resumed_is_no_longer_the_last_pause():
    resumed = ENTRIES + [entry(10, "decision", None, decision="resume", by="retry")]
    assert blocked.last_pause(resumed) is None
    again = resumed + [entry(11, "decision", 2, **pause("a later pause"))]
    assert blocked.last_pause(again) == pause("a later pause")
