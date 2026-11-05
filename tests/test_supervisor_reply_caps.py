# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 458 through the real supervisor: the tier's cap and sampling reach the coder's request, the request after a capped reply uses the
retry profile, and a step restarts once after two capped replies. The coder is a script; the supervisor, the agent loop and the ledger are real."""
import pytest

from orchard import runaway
from orchard.supervisor import EXIT_READY
from run_fakes import bringup
from test_supervisor import Rig

pytestmark = pytest.mark.usefixtures("stub_tools")

LOOP = ("Now I see the issue clearly. The model dictionary opens at position 5350 and never closes.\n"
        "- Position 17628: } (closes something, but not the model dictionary)\n") * 120
CAPPED = {"role": "assistant", "content": LOOP, "finish_reason": "length", "completion_tokens": 16384}


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def tool_requests(rig):
    return [r for r in rig.chip_server.requests if "tools" in r]


def agent_steps(rig):
    return [d for d in rig.decisions() if d.get("decision") == "agent step"]


def test_the_tiers_cap_and_sampling_reach_every_request_and_the_ledger(rig):
    text = rig.tiers.read_text().replace('model = "qwen-27b"\n', 'model = "qwen-27b"\nmax_tokens = 4096\ntemperature = 0.1\n'
                                         'frequency_penalty = 0.05\nunstick_temperature = 0.6\n', 1)
    rig.tiers.write_text(text)
    assert rig.run() == EXIT_READY
    reqs = tool_requests(rig)
    assert reqs and all(r["max_tokens"] == 4096 and r["temperature"] == 0.1 and r["frequency_penalty"] == 0.05 for r in reqs)
    first = agent_steps(rig)[0]["sampling"]
    assert first["max_tokens"] == 4096 and first["temperature"] == 0.1 and first["frequency_penalty"] == 0.05
    assert first["retry_after_cap"]["temperature"] == 0.6 and first["restarts_on_caps"] == 1


def test_a_tier_without_the_keys_sends_what_it_always_sent_and_the_ledger_records_the_defaults(rig):
    assert rig.run() == EXIT_READY
    assert all(r["max_tokens"] == 16384 and r["temperature"] == 0 and "frequency_penalty" not in r for r in tool_requests(rig))
    sampling = agent_steps(rig)[0]["sampling"]
    assert sampling["max_tokens"] == 16384 and sampling["temperature"] == 0 and sampling["retry_after_cap"] == runaway.Unstick().record()


def test_the_request_after_a_capped_reply_uses_the_retry_profile_through_the_supervisor(rig):
    seen = []

    def script(request):
        if "tools" in request and not seen:
            seen.append(1)
            return CAPPED
        return bringup(request)
    rig.script = script
    assert rig.run() == EXIT_READY
    first, second = tool_requests(rig)[:2]
    assert first["temperature"] == 0 and first["max_tokens"] == 16384
    assert second["temperature"] == 0.5 and second["max_tokens"] == 8192 and second["frequency_penalty"] == 0.2
    note = [d for d in rig.entries() if d["event"] == "notice" and d["data"].get("reply_capped")]
    assert len(note) == 1 and note[0]["data"]["repetition"] is True and note[0]["data"]["capped_in_stage"] == 1


def test_a_step_restarts_once_after_two_capped_replies_through_the_supervisor(rig):
    seen = []

    def script(request):
        if "tools" in request and len(seen) < 2:
            seen.append(1)
            return CAPPED
        return bringup(request)
    rig.script = script
    assert rig.run() == EXIT_READY
    restarts = [d for d in rig.decisions() if d.get("decision") == "context restarted after capped replies"]
    assert len(restarts) == 1 and restarts[0]["capped_in_step"] == 2 and restarts[0]["repetition_loops"] == 2
    after = tool_requests(rig)[2]["messages"]
    assert [m["role"] for m in after] == ["system", "user"] and "restarted" in after[1]["content"]
