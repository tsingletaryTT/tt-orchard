# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Sampling options of an agent step (B1): greedy by default, exactly as before; temperature, top_p and seed sent only
when a caller sets them. The request is captured through the step's `http` function, as the bytes post_json would send,
so no model server is called."""
from __future__ import annotations

import json

import pytest

from fake_model import final
from orchard.agent import AgentStep, Tools, agent_env
from orchard.ledger import Ledger

SAMPLING_KEYS = ("temperature", "top_p", "seed")


@pytest.fixture
def run(tmp_path):
    (tmp_path / "run" / "stages" / "0" / "evidence").mkdir(parents=True)
    with Ledger(tmp_path / "run" / "ledger.jsonl") as led:
        yield tmp_path / "run", led


class _Control:
    def take_nudges(self, agent):
        return []

    def stop_reason(self):
        return None


def _bodies(run, ledger, **kw):
    """Run one step against a capturing http function; return the JSON text of each request, as post_json encodes it."""
    sent = []

    def http(url, body, timeout):
        sent.append(json.dumps(body))
        return {"choices": [{"index": 0, "message": final("done"), "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    sd = run / "stages" / "0"
    step = AgentStep(agent="stage-agent", endpoint="http://127.0.0.1:9/v1", model="fake",
                     tools=Tools(run, sd, agent_env(run)), ledger=ledger, stage=0, phase="run", feed=lambda e: None,
                     control=_Control(), run_dir=run, evidence_dir=sd / "evidence", log_path=sd / "log" / "r.jsonl",
                     http=http, clock=lambda: 1000.0, **kw)
    step.run("system", "user")
    return sent


def test_a_step_built_the_old_way_sends_exactly_temperature_0_and_no_other_sampling_key(run):
    run_dir, ledger = run
    (body,) = _bodies(run_dir, ledger)
    assert '"temperature": 0,' in body and '"temperature": 0.0' not in body
    parsed = json.loads(body)
    assert [k for k in SAMPLING_KEYS if k in parsed] == ["temperature"]


def test_temperature_top_p_and_seed_are_sent_when_set(run):
    run_dir, ledger = run
    (body,) = _bodies(run_dir, ledger, temperature=0.7, top_p=0.9, seed=1234)
    parsed = json.loads(body)
    assert (parsed["temperature"], parsed["top_p"], parsed["seed"]) == (0.7, 0.9, 1234)


def test_only_the_options_that_are_set_are_sent(run):
    run_dir, ledger = run
    (body,) = _bodies(run_dir, ledger, temperature=1.0)
    parsed = json.loads(body)
    assert parsed["temperature"] == 1.0 and "top_p" not in parsed and "seed" not in parsed


def test_a_float_zero_temperature_is_sent_as_the_same_bytes_as_before(run):
    """The probe passes 0.0 for greedy decoding; the request must match the old `"temperature": 0` exactly."""
    run_dir, ledger = run
    (body,) = _bodies(run_dir, ledger, temperature=0.0)
    assert '"temperature": 0,' in body and '"temperature": 0.0' not in body
