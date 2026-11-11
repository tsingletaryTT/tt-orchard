"""Issue 704: an escalation records which tier and model served the escalated attempt, and the status summary does not count an escalation to
the same model as an escalation (it is a fresh-context retry). The run's escalation cap still counts every escalation, because a fresh
context on the same model is still a loop that has to end."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import stages, watchdog
from orchard.tiers import load

SAME = """
[tiers.large]
role = "plan and diagnose"
endpoint = "http://127.0.0.1:8000/v1"
model = "coder-model"
placement = "chips"
context_tokens = 262144

[tiers.small]
role = "routine steps"
endpoint = "http://127.0.0.1:8011/v1"
model = "coder-model"
placement = "chips"
context_tokens = 262144

[tiers.cpu]
role = "stand-in"
endpoint = "http://127.0.0.1:11434/v1"
model = "cpu-model"
placement = "cpu"

[stages.0]
run = "small"
[stages.1]
run = "small"
[stages.2]
run = "small"
diagnose = "large"
[stages.3]
run = "small"
diagnose = "large"
[stages.4]
run = "small"
plan = "large"
diagnose = "large"
[stages.5]
run = "small"
[stages.6]
run = "small"
[stages.7]
run = "none"
[stages.8]
run = "small"

[escalation]
default = "large"
"""
OTHER = SAME.replace('model = "coder-model"\nplacement = "chips"\ncontext_tokens = 262144\n\n[tiers.small]', 'model = "big-model"\nplacement = "chips"\ncontext_tokens = 262144\n\n[tiers.small]', 1)


def cfg(tmp_path, text):
    p = tmp_path / "tiers.toml"
    p.write_text(text)
    return load(p)


def test_an_escalation_to_a_tier_with_the_same_model_is_marked_and_served_by_the_coder(tmp_path):
    c = cfg(tmp_path, SAME)
    info = stages.escalation_info(c, 2, coder_tier="small")
    assert info == {"tier_before": "small", "tier_after": "large", "served_tier": "small", "served_model": "coder-model", "same_model": True}


def test_an_escalation_to_another_model_is_marked_as_a_change(tmp_path):
    c = cfg(tmp_path, OTHER)
    info = stages.escalation_info(c, 2, coder_tier="small")
    assert info["tier_after"] == "large" and info["served_tier"] == "large" and info["served_model"] == "big-model" and info["same_model"] is False


def test_a_stage_that_loads_no_model_has_no_escalation_info(tmp_path):
    assert stages.escalation_info(cfg(tmp_path, SAME), 7, coder_tier="small") == {}


from orchard.ledger import Ledger
from orchard.watchdog import Finding, Ladder


class Actuator:
    def nudge(self, agent, message): pass
    def escalate(self, agent, stage): pass
    def pause(self, reason, evidence): pass


def _finding():
    return Finding("identical_responses", "coder", 1.0, "3 identical responses in a row", {"count": 3})


def _escalate_entry(tmp_path, **kw):
    with Ledger(tmp_path / "ledger.jsonl") as led:
        ladder = Ladder(Actuator(), led, {"coder"}, **kw)
        assert [ladder.respond("coder", [_finding()], 2) for _ in range(2)] == ["nudge", "escalate"]
        return next(e["data"] for e in led.read() if e["event"] == "escalate")


def test_the_watchdog_ladder_adds_the_served_by_fields_to_its_escalate_entry(tmp_path):
    data = _escalate_entry(tmp_path, escalation_info=lambda stage: {"served_tier": "small", "same_model": True, "stage_seen": stage})
    assert data["served_tier"] == "small" and data["same_model"] is True and data["stage_seen"] == 2 and data["watchdog"] is True


def test_the_ladder_without_a_provider_writes_the_entry_as_before(tmp_path):
    assert "served_tier" not in _escalate_entry(tmp_path)


def test_a_provider_that_raises_leaves_the_entry_without_the_fields(tmp_path):
    def boom(stage):
        raise KeyError("no such tier")
    data = _escalate_entry(tmp_path, escalation_info=boom)
    assert data["rung"] == "escalate" and "served_tier" not in data
