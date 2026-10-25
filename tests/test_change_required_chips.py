# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Changing a run's required chip counts on purpose (review of PR 70).

The stage 4 block for a required count that nothing serves says to run again with fewer required counts. A run keeps the counts it started with, so that
command was refused. `--change-required-chips` is the explicit way: the supervisor records a "required chips set" decision with the old and new counts, the
latest decision wins on a later resume, and it is refused once stage 4 has passed (its results then rest on the old counts)."""
import json

import pytest

from orchard.ledger import Ledger
from fakes import Crash
from orchard.supervisor import build, parse
from test_supervisor import rig  # noqa: F401
from run_fakes import argv


def resumed(rig, new, *, change=True, pid=200):
    rig.args.required_chips = (2, 4)
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 0)
    rig.args.required_chips = new
    rig.args.change_required_chips = change
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        sup = build(rig.args, led, adapter=rig.adapter_cls(rig.m, owner_pid=pid), coder=rig.coder_cls(rig.m),
                    versions={"tt_model": "test"}, home=rig.home)
        return sup, led.read()


def test_the_flag_parses_and_defaults_off(tmp_path, rig):
    assert rig.args.change_required_chips is False
    assert parse(argv(tmp_path, rig.tiers, rig.chip_server.endpoint) + ["--change-required-chips"]).change_required_chips is True


def test_a_different_count_without_the_flag_is_refused_and_names_the_flag(rig):
    with pytest.raises(ValueError, match="--change-required-chips"):
        resumed(rig, (4,), change=False)


def test_with_the_flag_the_new_counts_apply_and_the_ledger_records_the_change(rig):
    sup, entries = resumed(rig, (4,))
    assert sup.required_chips == (4,)
    rec = [e["data"] for e in entries if e["data"].get("decision") == "required chips set"]
    assert len(rec) == 1 and rec[0]["old"] == [2, 4] and rec[0]["new"] == [4]


def test_the_latest_recorded_change_is_what_a_later_resume_uses(rig):
    resumed(rig, (4,))
    rig.args.required_chips = None
    rig.args.change_required_chips = False
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        sup = build(rig.args, led, adapter=rig.adapter_cls(rig.m, owner_pid=300), coder=rig.coder_cls(rig.m),
                    versions={"tt_model": "test"}, home=rig.home)
    assert sup.required_chips == (4,)


def test_the_same_counts_record_nothing(rig):
    sup, entries = resumed(rig, (2, 4))
    assert sup.required_chips == (2, 4)
    assert not [e for e in entries if e["data"].get("decision") == "required chips set"]


def test_the_flag_with_no_counts_changes_nothing(rig):
    sup, entries = resumed(rig, None)
    assert sup.required_chips == (2, 4)
    assert not [e for e in entries if e["data"].get("decision") == "required chips set"]


def test_a_change_after_stage_4_passed_is_refused(rig):
    rig.args.required_chips = (2, 4)
    rig.run()                                                  # the whole fake run: stage 4 passes
    rig.args.required_chips = (4,)
    rig.args.change_required_chips = True
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        with pytest.raises(ValueError, match="stage 4 has passed"):
            build(rig.args, led, adapter=rig.adapter_cls(rig.m, owner_pid=400), coder=rig.coder_cls(rig.m),
                  versions={"tt_model": "test"}, home=rig.home)


def test_bringup_passes_the_flag_to_the_supervisor(tmp_path):
    from test_bringup_config import load
    from orchard import bringup_config as bc
    cfg = load(tmp_path)
    assert "--change-required-chips" not in bc.supervisor_argv(cfg, "a/b")
    assert "--change-required-chips" in bc.supervisor_argv(cfg, "a/b", change_required_chips=True)
