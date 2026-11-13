# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The coder starts lazily at stage 2 of a `--thin-port` full port (issue 713).

When stage 2 of a thin-port run begins with no coder in the run (a resume after the coder's lease was released, as when an arm reuses a run
directory), the supervisor does not boot the coder for the stage. It boots it at the first model request of the stage: a stub the thin port
step must fill, an agent step (prepare, repair, finish, escalation) or a thin repair request. A plan with no stubs whose port passes reaches
the stage 2 pass with no coder boot, and the ledger says "coder not started" with the reason. The chips, the coder and the model are fakes; the
supervisor, the stage machine, the gate and the check template are the real ones.
"""
from __future__ import annotations

import pytest

from orchard import thin_port
from orchard.supervisor import EXIT_ABORTED, EXIT_BLOCKED, Control

pytestmark = pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")


def entries_after_resume(r):
    """The ledger entries written by the second run (after the first run's last entry)."""
    return r.entries()[r.first_run_end:]


def ledger_now(r):
    """The ledger while the supervisor runs: it holds the ledger's lock, so the file is read, never opened through Ledger."""
    import json
    out = []
    for line in (r.run_dir / "ledger.jsonl").read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass                                       # a line the supervisor is writing right now
    return out[r.first_run_end:]


def names(es, stage=None):
    return [e["data"].get("decision") for e in es if e["event"] == "decision" and (stage is None or e["stage"] == stage)]


def first_run_blocks_at_stage_2(r, tmp_path, monkeypatch):
    """Run 1: stages 0 and 1 pass with the coder up; stage 2's thin step fails, the unattended run blocks and gives the coder's lease back."""
    from test_supervisor_full_port import HONEST, coder_writes, with_snapshot
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.thin_port = True
    coder_writes(r, {}, port=HONEST)
    monkeypatch.setattr(thin_port, "run", lambda **kw: thin_port.Outcome("error", "the model request failed and was retried once (OSError)"))
    assert r.run() == EXIT_BLOCKED
    assert "hardware released" in names(r.entries())                       # the run that resumes has no coder
    r.first_run_end = len(r.entries())


def second_run(r, monkeypatch, fake):
    monkeypatch.setattr(thin_port, "run", fake)
    r.args.unattended = False
    seen = []

    def on_sleep():
        # The first pause is left by the blocked stage, and the operator resumes it. The next pause (before stage 3 or at stage 5) ends the run.
        seen.append(1)
        Control(r.run_dir).write("resume" if len(seen) == 1 else "abort")
    r.on_sleep = on_sleep
    return r.run(pid=200)


def writes_port(kw, text):
    port = kw["stage_dir"] / "port"
    port.mkdir(parents=True, exist_ok=True)
    (port / "port_model.py").write_text(text)


def test_a_plan_with_no_stubs_reaches_the_stage_2_pass_without_booting_the_coder(tmp_path, monkeypatch):
    from test_supervisor import Rig
    from test_supervisor_full_port import HONEST
    r = Rig(tmp_path)
    try:
        first_run_blocks_at_stage_2(r, tmp_path, monkeypatch)
        calls = []

        def fake(**kw):
            calls.append(kw)
            writes_port(kw, HONEST)
            return thin_port.Outcome("done", requests=0)          # no stub: `connect` is never called
        assert second_run(r, monkeypatch, fake) == EXIT_ABORTED
        after = entries_after_resume(r)
        assert [d["result"] for d in r.ends(2)][-1] == "pass"
        # no boot anywhere in the resumed run: stages 3 and 4 are skipped (no chip count is listed) and stage 5 is where it pauses
        assert "coder starting" not in names(after) and "coder started" not in names(after)
        assert [e["data"]["result"] for e in after if e["event"] == "stage_end" and e["stage"] in (3, 4)] == ["skipped", "skipped"]
        assert "coder start deferred" in names(after, 2)
        note = next(e["data"] for e in after if e["event"] == "decision" and e["data"].get("decision") == "coder not started")
        assert note["reason"] == "the plan has no stubs"
        assert "hardware test started" in names(after, 2) and "park" not in names(after, 2)        # the board was free: nothing to park
        assert calls and calls[0]["endpoint"] is None and callable(calls[0]["connect"])
    finally:
        r.close()


def test_a_plan_with_a_stub_starts_the_coder_before_the_first_stub_request(tmp_path, monkeypatch):
    from test_supervisor import Rig
    from test_supervisor_full_port import HONEST
    r = Rig(tmp_path)
    try:
        first_run_blocks_at_stage_2(r, tmp_path, monkeypatch)
        seen = {}

        def fake(**kw):
            before = names(ledger_now(r))
            seen["started_before_connect"] = "coder started" in before
            seen["endpoint"], seen["model"], seen["sampling"] = kw["connect"](1)
            seen["started_after_connect"] = "coder started" in names(ledger_now(r))
            writes_port(kw, HONEST)
            return thin_port.Outcome("done", filled=["_B_ple"], requests=1)
        assert second_run(r, monkeypatch, fake) == EXIT_ABORTED
        assert seen["started_before_connect"] is False and seen["started_after_connect"] is True
        assert seen["endpoint"] and seen["model"] and "temperature" in seen["sampling"]
        after = entries_after_resume(r)
        n = names(after, None)
        assert n.index("coder started for a model request") < n.index("coder started") < n.index("thin port step")
        reason = next(e["data"]["reason"] for e in after if e["event"] == "decision" and e["data"].get("decision") == "coder started for a model request")
        assert "1 stub" in reason
        assert "coder not started" not in n
        assert [d["result"] for d in r.ends(2)][-1] == "pass"
    finally:
        r.close()


def test_a_later_model_request_of_the_stage_boots_the_coder_first(tmp_path, monkeypatch):
    """Falsification check of the issue: the plan has no stubs, the port fails the board check, and the repair round needs the coder. It is booted
    before the repair request, so no step finds it missing."""
    from test_supervisor import Rig
    from test_supervisor_port_repair import BROKEN
    r = Rig(tmp_path)
    try:
        first_run_blocks_at_stage_2(r, tmp_path, monkeypatch)
        second_run(r, monkeypatch, lambda **kw: (writes_port(kw, BROKEN), thin_port.Outcome("done"))[1])
        after = entries_after_resume(r)
        n = names(after)
        assert "coder not started" in n                                        # the thin step needed no coder
        boot = n.index("coder started for a model request")
        assert n.index("coder not started") < n.index("port repair round") < boot < n.index("coder started")
        # the thin repair round's request is the first model request: the coder boots for it, before its tier is probed
        reason = next(e["data"]["reason"] for e in after if e["event"] == "decision" and e["data"].get("decision") == "coder started for a model request")
        assert reason == "a thin repair request"
        first_request = next(i for i, x in enumerate(n) if x in ("thin repair round", "thin repair handed to the coder", "agent step"))
        assert boot < first_request
    finally:
        r.close()


def test_a_run_whose_coder_is_up_is_not_deferred(tmp_path, monkeypatch):
    """A fresh run boots the coder for stages 0 and 1 (their steps are model requests); stage 2 then uses it, and nothing is deferred."""
    from test_supervisor import Rig
    from test_supervisor_full_port import HONEST, coder_writes, with_snapshot
    r = Rig(tmp_path)
    try:
        with_snapshot(r, tmp_path)
        r.args.full_port = True
        r.args.thin_port = True
        coder_writes(r, {}, port=HONEST, unattended=False)
        r.on_sleep = lambda: Control(r.run_dir).write("abort")

        def fake(**kw):
            writes_port(kw, HONEST)
            return thin_port.Outcome("done")
        monkeypatch.setattr(thin_port, "run", fake)
        r.run()
        n = names(r.entries())
        assert "coder started" in n and "coder start deferred" not in n and "coder not started" not in n
    finally:
        r.close()


def test_an_agent_step_as_the_first_model_request_boots_the_coder_first(tmp_path, monkeypatch):
    """The generator cannot plan (thin step "skipped"), so the agent's prepare step is the stage's first model request: the coder boots before
    the step's tier is probed."""
    from test_supervisor import Rig
    r = Rig(tmp_path)
    try:
        first_run_blocks_at_stage_2(r, tmp_path, monkeypatch)
        second_run(r, monkeypatch, lambda **kw: thin_port.Outcome("skipped", "no reference interpreter"))
        after = entries_after_resume(r)
        n = names(after)
        assert "coder not started" not in n
        boot = n.index("coder started for a model request")
        assert boot < n.index("coder started") < n.index("agent step")
        reason = next(e["data"]["reason"] for e in after if e["event"] == "decision" and e["data"].get("decision") == "coder started for a model request")
        assert "prepare agent step" in reason
        assert [d["result"] for d in r.ends(2)][-1] == "pass"
    finally:
        r.close()


def test_only_stage_2_of_a_thin_port_full_port_with_no_coder_in_the_run_is_deferred(tmp_path, monkeypatch):
    from test_supervisor import Rig
    from orchard import supervisor as sup_mod
    from orchard.ledger import Ledger
    from orchard.stages import FULL_PORT_STAGE_2, FULL_PORT_STAGE_3, STAGES
    from orchard.supervisor import build
    r = Rig(tmp_path)
    try:
        first_run_blocks_at_stage_2(r, tmp_path, monkeypatch)
        with Ledger(r.run_dir / "ledger.jsonl") as led:
            sup = build(r.args, led, adapter=r.adapter_cls(r.m, owner_pid=300), coder=r.coder_cls(r.m), home=r.home, containers=r.containers)
            assert sup._coder_deferrable(FULL_PORT_STAGE_2) is True
            assert sup._coder_deferrable(FULL_PORT_STAGE_3) is False
            assert sup._coder_deferrable(STAGES[0]) is False and sup._coder_deferrable(STAGES[2]) is False       # not a full port's stage 2
            monkeypatch.setattr(sup_mod, "thin_port", lambda entries: False)
            assert sup._coder_deferrable(FULL_PORT_STAGE_2) is False                                             # a run without --thin-port
            monkeypatch.undo()
            monkeypatch.setattr(sup_mod.Supervisor, "_coder_in_use", lambda self: True)
            assert sup._coder_deferrable(FULL_PORT_STAGE_2) is False                                             # a coder already in the run
    finally:
        r.close()
