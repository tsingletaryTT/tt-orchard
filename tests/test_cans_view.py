# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Cans view module (orchard/cans_view.py): what it reads, what it shows for a missing or broken piece, and what it never does.

A temporary checkout holds the specs and a temporary cans folder holds the staged bundles, so no real can, chip or network is involved.
Hostile input (links, FIFOs, huge or malformed files, NUL) is in tests/test_cans_view_hostile.py."""
import json
import os
import socket
import subprocess

import pytest

from cans_fixture import CARD, add_spec, by_name, dump, make_checkout, passed_boot, record, stage_bundle
from orchard import can
from orchard import cans_view as view


@pytest.fixture
def world(tmp_path):
    root = make_checkout(tmp_path)
    cans = tmp_path / "cans"
    cans.mkdir()
    return root, cans


def build(world, **kw):
    root, cans = world
    return view.CansView(root / "config", root, cans, **kw).build()


def test_a_can_with_a_bundle_and_records_shows_every_fact(world):
    root, cans = world
    name = add_spec(root, "toy", mesh="P300", devices=2)
    b = stage_bundle(cans, name)
    record(cans, name, b, gate=can.gate_record([]), boot=passed_boot())
    row = by_name(build(world), name)
    assert (row["kind"], row["mesh"], row["device_count"], row["card_profile"]) == ("tt-dit-server", "P300", 2, "catalog")
    assert row["weights"]["repo"] == "someone/toy-model" and row["weights"]["revision"] == "d18cca4c121edb87d022116d281ce212c9136f57"
    assert row["bundle"]["state"] == "staged"
    assert row["sums"]["state"] == "ok" and row["sums"]["text"] == "all 6 files match"
    assert row["gate"]["state"] == "passed" and row["boot"]["state"] == "passed"
    assert (row["boot"]["decode_tok_s"], row["boot"]["cache_state"], row["boot"]["ready_s"]) == (31.4, "empty", 212.0)
    assert "decode 31.4 tok/s" in row["boot"]["text"] and "cache empty" in row["boot"]["text"]
    assert row["card"]["state"] == "ok"
    assert row["card"]["performance"] == {"header": ["Metric", "Value"], "rows": [["Decode", "31.4 tok/s"], ["Start", "212 s"]], "more": 0}
    assert row["card"]["limitations"] == ["Greedy decoding only.", "Context is 2048 tokens."]
    assert (row["tone"], row["summary"]) == ("good", "booted")


def test_the_hub_is_a_link_and_a_not_checked_here_never_a_revision_check(world):
    root, cans = world
    name = add_spec(root, "toy")
    row = by_name(build(world), name)
    assert row["weights"]["hub_url"] == "https://huggingface.co/someone/toy-model"
    assert row["weights"]["hub_text"] == "not checked here"


def test_a_weights_repo_that_is_not_a_plain_org_slash_name_gets_no_link(world):
    root, cans = world
    name = add_spec(root, "toy", repo='a b/c<script>')
    row = by_name(build(world), name)
    assert row["weights"]["hub_url"] is None and row["weights"]["repo"] == 'a b/c<script>'


def test_a_spec_with_no_card_profile_says_the_card_is_not_checked_and_still_shows_its_rows(world):
    root, cans = world
    name = add_spec(root, "toy", profile=None)
    stage_bundle(cans, name)
    row = by_name(build(world), name)
    assert row["card_profile"] is None and row["card"]["state"] == "not-required" and row["card"]["performance"]["rows"][0] == ["Decode", "31.4 tok/s"]


def test_no_cans_folder_shows_every_can_as_not_staged_and_says_how_to_set_it(tmp_path):
    root = make_checkout(tmp_path)
    name = add_spec(root, "toy")
    answer = view.CansView(root / "config", root, tmp_path / "nowhere", redact=lambda t: t).build()
    assert answer["folder"]["state"] == "missing" and "ORCHARD_CANS_DIR" in answer["folder"]["text"]
    row = by_name(answer, name)
    assert row["bundle"]["state"] == "not-staged" and row["bundle"]["text"] == "no staged bundle"
    assert (row["tone"], row["summary"]) == ("dim", "not staged")
    assert row["sums"]["state"] == "none" and row["gate"]["state"] == "none" and row["boot"]["state"] == "none"


def test_a_spec_the_loader_refuses_is_a_row_of_its_own_and_the_others_still_show(world):
    root, cans = world
    good = add_spec(root, "good")
    (root / "config" / "can.broken.toml").write_text("[can]\nname = 'x'\n")
    (root / "config" / "can.garbled.toml").write_text("this is [ not toml")
    answer = build(world)
    assert [c["slug"] for c in answer["cans"]] == ["broken", "garbled", "good"]
    broken = by_name(answer, "broken")
    assert broken["spec"]["state"] == "refused" and broken["spec"]["text"].startswith("spec refused: ")
    assert (broken["tone"], broken["summary"]) == ("bad", "spec refused") and broken["weights"] is None
    assert by_name(answer, good)["spec"]["state"] == "ok"


def test_a_spec_that_names_a_missing_source_file_is_refused_with_the_checkout_path_removed(world):
    root, cans = world
    add_spec(root, "toy")
    (root / "pkg" / "toy_model.py").unlink()
    row = by_name(build(world), "toy")
    assert row["spec"]["state"] == "refused" and str(root) not in dump(build(world))


def test_a_changed_file_is_a_sums_difference_that_names_the_file(world):
    root, cans = world
    name = add_spec(root, "toy")
    b = stage_bundle(cans, name)
    (b / "run.sh").write_text("#!/bin/sh\necho changed\n")
    row = by_name(build(world), name)
    assert row["sums"]["state"] == "differs" and row["sums"]["tone"] == "bad"
    assert row["sums"]["problems"] == ["run.sh changed after SHA256SUMS was written"]
    assert (row["tone"], row["summary"]) == ("bad", "failing")


def test_a_file_the_sums_do_not_list_and_a_listed_file_that_is_gone_are_both_problems(world):
    root, cans = world
    name = add_spec(root, "toy")
    b = stage_bundle(cans, name)
    (b / "extra.txt").write_text("x")
    (b / "install.sh").unlink()
    problems = by_name(build(world), name)["sums"]["problems"]
    assert "SHA256SUMS lists install.sh, which is not in the bundle" in problems
    assert "SHA256SUMS does not list extra.txt" in problems


def test_a_missing_sums_file_is_shown_as_missing(world):
    root, cans = world
    name = add_spec(root, "toy")
    b = stage_bundle(cans, name)
    (b / can.SUMS_FILE).unlink()
    row = by_name(build(world), name)
    assert row["sums"]["state"] == "missing" and row["sums"]["text"] == "SHA256SUMS is missing" and row["tone"] == "warn"


def test_files_in_subfolders_are_checked_and_listed_with_their_folder(world):
    root, cans = world
    name = add_spec(root, "toy")
    b = stage_bundle(cans, name, extra={"extra_code/a/b.py": "x = 1\n"})
    assert by_name(build(world), name)["sums"]["text"] == "all 7 files match"
    (b / "extra_code" / "a" / "b.py").write_text("x = 2\n")
    assert by_name(build(world), name)["sums"]["problems"] == ["extra_code/a/b.py changed after SHA256SUMS was written"]


# ---- records ----------------------------------------------------------------------------------------------------------------

def test_a_failed_gate_record_is_red_with_its_problems_and_a_count(world):
    root, cans = world
    name = add_spec(root, "toy")
    b = stage_bundle(cans, name)
    record(cans, name, b, gate=can.gate_record([f"scrub: thing {i}" for i in range(14)]))
    row = by_name(build(world), name)
    assert row["gate"]["state"] == "failed" and row["gate"]["tone"] == "bad" and row["gate"]["problem_count"] == 14
    assert row["gate"]["problems"][0] == "scrub: thing 0" and len(row["gate"]["problems"]) == 10
    assert "14 problems" in row["gate"]["text"] and (row["tone"], row["summary"]) == ("bad", "failing")


def test_a_record_for_an_earlier_bundle_is_stale_and_does_not_count_as_passed(world):
    root, cans = world
    name = add_spec(root, "toy")
    b = stage_bundle(cans, name)
    record(cans, name, b, gate=can.gate_record([]), boot=passed_boot())
    can.write_sums(b)                                    # unchanged files: same digest
    assert by_name(build(world), name)["gate"]["state"] == "passed"
    (b / "README.md").write_text(CARD + "\nmore\n")
    can.write_sums(b)                                    # the bundle changed after the record
    # a new view: the first one cached its answer
    row = by_name(build(world), name)
    assert row["gate"]["state"] == "stale" and "earlier bundle" in row["gate"]["text"]
    assert row["boot"]["state"] == "stale" and row["boot"]["decode_tok_s"] == 31.4
    assert (row["tone"], row["summary"]) == ("warn", "incomplete")


def test_no_record_says_so_and_names_the_command(world):
    root, cans = world
    name = add_spec(root, "toy")
    stage_bundle(cans, name)
    row = by_name(build(world), name)
    assert row["gate"]["state"] == "none" and "--state-dir" in row["gate"]["text"]
    assert row["boot"]["state"] == "none" and row["boot"]["decode_tok_s"] is None
    assert (row["tone"], row["summary"]) == ("warn", "incomplete")


def test_a_failed_boot_names_the_failed_checks(world):
    root, cans = world
    name = add_spec(root, "toy")
    b = stage_bundle(cans, name)
    ev = {"passed": False, "checks": {"health": {"ok": True}, "decode_rate": {"ok": False}}, "ready_s": 9.0}
    record(cans, name, b, boot=can.boot_record(ev, "warm"))
    row = by_name(build(world), name)
    assert row["boot"]["state"] == "failed" and row["boot"]["failed"] == ["decode_rate"] and "decode_rate" in row["boot"]["text"]


def test_a_boot_with_no_cache_state_says_it_was_not_recorded(world):
    root, cans = world
    name = add_spec(root, "toy")
    b = stage_bundle(cans, name)
    record(cans, name, b, boot=passed_boot(cache=None))
    assert "cache state not recorded" in by_name(build(world), name)["boot"]["text"]


def test_a_state_file_with_a_bad_field_is_unreadable_and_shows_nothing_from_it(world):
    root, cans = world
    name = add_spec(root, "toy")
    b = stage_bundle(cans, name)
    path = record(cans, name, b, gate=can.gate_record([]), boot=passed_boot())
    raw = json.loads(path.read_text())
    raw["boot"]["decode_tok_s"] = "fast"
    path.write_text(json.dumps(raw))
    row = by_name(build(world), name)
    assert row["gate"]["state"] == "unreadable" and row["boot"]["state"] == "unreadable" and row["boot"]["decode_tok_s"] is None
    assert row["sums"]["state"] == "ok", "the live pieces still show"


def test_a_state_file_for_another_can_is_not_used(world):
    root, cans = world
    name = add_spec(root, "toy")
    b = stage_bundle(cans, name)
    path = record(cans, name, b, gate=can.gate_record([]))
    raw = json.loads(path.read_text())
    raw["name"] = "someone-else"
    path.write_text(json.dumps(raw))
    assert by_name(build(world), name)["gate"]["state"] == "unreadable"


# ---- the card ---------------------------------------------------------------------------------------------------------------

def test_a_card_without_the_two_sections_is_a_problem_when_the_profile_asks_for_them(world):
    root, cans = world
    name = add_spec(root, "toy")
    stage_bundle(cans, name, card="# Toy\n\nNo sections.\n")
    row = by_name(build(world), name)
    assert row["card"]["state"] == "problems" and row["card"]["performance"] is None and row["card"]["limitations"] == []
    assert "Expected performance" in row["card"]["text"]
    assert (row["tone"], row["summary"]) == ("bad", "failing")


def test_a_heading_inside_a_code_fence_does_not_count_as_a_section(world):
    root, cans = world
    name = add_spec(root, "toy")
    stage_bundle(cans, name, card="# T\n\n```\n## Expected performance\n| a | b |\n| - | - |\n| 1 | 2 |\n## Limitations\ntext\n```\n")
    row = by_name(build(world), name)
    assert row["card"]["state"] == "problems" and row["card"]["performance"] is None


def test_a_missing_readme_is_shown_as_a_missing_card(world):
    root, cans = world
    name = add_spec(root, "toy")
    b = stage_bundle(cans, name)
    (b / "README.md").unlink()
    can.write_sums(b)
    row = by_name(build(world), name)
    assert row["card"]["state"] == "missing" and row["card"]["text"] == "the card (README.md) is missing"


def test_card_rows_and_lines_are_capped_and_cut(world):
    root, cans = world
    name = add_spec(root, "toy")
    rows = "".join(f"| m{i} | {i} {'x' * 400} |\n" for i in range(30))
    lines = "".join(f"limit {i}\n" for i in range(30))
    stage_bundle(cans, name, card=f"## Expected performance\n\n| M | V |\n| - | - |\n{rows}\n## Limitations\n\n{lines}")
    card = by_name(build(world), name)["card"]
    assert len(card["performance"]["rows"]) == view.CARD_MAX_ROWS and card["performance"]["more"] == 30 - view.CARD_MAX_ROWS
    assert all(len(c) <= view.CELL_MAX for r in card["performance"]["rows"] for c in r)
    assert len(card["limitations"]) == view.CARD_MAX_LINES and card["limitations_more"] == 30 - view.CARD_MAX_LINES


# ---- what the view never does -----------------------------------------------------------------------------------------------

def test_the_answer_holds_no_path_of_this_machine(world, tmp_path):
    root, cans = world
    name = add_spec(root, "toy")
    b = stage_bundle(cans, name)
    record(cans, name, b, gate=can.gate_record([f"scrub: {b}/run.sh holds a thing"]))
    (b / "run.sh").write_text("changed")
    (root / "config" / "can.bad.toml").write_text("not toml [")
    text = dump(build(world, redact=lambda t: t))
    assert str(tmp_path) not in text.replace(f"{b}/run.sh holds a thing", "")      # the recorded problem is the operator's text; redact handles it below
    text = dump(build(world, redact=lambda t: t.replace(str(tmp_path), "<tmp>")))
    assert str(tmp_path) not in text


def test_strings_from_files_pass_through_redact(world):
    root, cans = world
    name = add_spec(root, "toy")
    b = stage_bundle(cans, name)
    record(cans, name, b, gate=can.gate_record(["scrub: /home/someone/secret path"]))
    (b / "extra-HOST.txt").write_text("x")
    out = build(world, redact=lambda t: t.replace("/home/someone", "~").replace("HOST", "<host>"))
    row = by_name(out, name)
    assert row["gate"]["problems"] == ["scrub: ~/secret path"]
    assert "SHA256SUMS does not list extra-<host>.txt" in row["sums"]["problems"]


def test_building_the_answer_writes_nothing_and_never_runs_the_gate_or_touches_the_network(world, monkeypatch):
    root, cans = world
    name = add_spec(root, "toy")
    b = stage_bundle(cans, name)
    record(cans, name, b, gate=can.gate_record([]))

    def forbidden(*a, **k):
        raise AssertionError("the Cans view must not do this")
    monkeypatch.setattr(can, "gate", forbidden)
    monkeypatch.setattr(can, "boot_check", forbidden)
    monkeypatch.setattr(can, "stage", forbidden)
    monkeypatch.setattr(can, "write_state", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)

    def snapshot():
        return sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for base in (root, cans) for p in base.rglob("*"))
    before = snapshot()
    assert by_name(build(world), name)["bundle"]["state"] == "staged"
    assert snapshot() == before


def test_the_answer_is_cached_for_a_while_and_rebuilt_after(world):
    root, cans = world
    name = add_spec(root, "toy")
    b = stage_bundle(cans, name)
    now = [1000.0]
    v = view.CansView(root / "config", root, cans, clock=lambda: now[0])
    assert by_name(v.build(), name)["sums"]["state"] == "ok"
    (b / "run.sh").write_text("changed")
    now[0] += view.CACHE_S - 1
    assert by_name(v.build(), name)["sums"]["state"] == "ok"
    now[0] += 2
    assert by_name(v.build(), name)["sums"]["state"] == "differs"


def test_only_the_first_sixty_specs_are_listed_and_the_rest_are_counted(world):
    root, cans = world
    for i in range(view.MAX_SPECS + 5):
        (root / "config" / f"can.s{i:03d}.toml").write_text("not toml [")
    (root / "config" / "notacan.toml").write_text("x")
    (root / "config" / "can.x.txt").write_text("x")
    answer = build(world)
    assert len(answer["cans"]) == view.MAX_SPECS and answer["specs"]["more"] == 5
    assert all(c["slug"].startswith("s") for c in answer["cans"])


def test_an_unreadable_specs_folder_is_an_empty_list_and_says_so(tmp_path):
    root = make_checkout(tmp_path)
    answer = view.CansView(tmp_path / "nowhere", root, tmp_path / "cans").build()
    assert answer["cans"] == [] and answer["specs"]["folder_unreadable"] is True


def test_the_real_specs_in_the_repository_all_load_or_are_shown_as_refused_without_an_error():
    from pathlib import Path
    checkout = Path(__file__).resolve().parent.parent
    answer = view.CansView(checkout / "config", checkout, checkout / "does-not-exist").build()
    assert len(answer["cans"]) >= 6
    for row in answer["cans"]:
        assert row["spec"]["state"] in ("ok", "refused")
        assert row["bundle"]["state"] == ("not-staged" if row["spec"]["state"] == "ok" else "unknown")
        if row["spec"]["state"] == "ok":
            assert row["weights"]["hub_url"].startswith("https://huggingface.co/")
