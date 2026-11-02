# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Regression tests for the findings of the first review of PR 342 (the Cans view and the can state file).

Each test names the finding it covers: the boot record is bound to the copy that was booted, a file that grows while it is hashed is not
a match, a repo id that redaction would change gets no link, a record of another can is replaced, a spec that names a big tree is not
walked by the view, a malformed `expect` is a refused row, and keyboard focus survives a redraw (layout and browser tests)."""
import hashlib
import io
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from cans_fixture import add_spec, by_name, dump, make_checkout, stage_bundle
from orchard import can
from orchard import cans_view as view
from test_can_state import args, bundle, fake, read   # noqa: F401  (fake is a fixture)
from test_webui_cans import app_for, get, root         # noqa: F401  (root is a fixture)


@pytest.fixture
def world(tmp_path):
    checkout = make_checkout(tmp_path)
    cans = tmp_path / "cans"
    cans.mkdir()
    return checkout, cans


def build(world, **kw):
    checkout, cans = world
    return view.CansView(checkout / "config", checkout, cans, **kw).build()


# ---- the boot record is bound to the copy that was booted --------------------------------------------------------------------

def test_boot_binds_the_record_to_the_copy_it_booted_not_to_the_source_rebuilt_during_the_boot(tmp_path, fake, monkeypatch):
    booted = "0" * 64 + "  run.sh\n"

    def boot(source, work, spec, **kw):
        copy = Path(work) / "bundle"
        copy.mkdir(parents=True)
        (copy / can.SUMS_FILE).write_text(booted)                              # what the boot served
        (Path(source) / can.SUMS_FILE).write_text("1" * 64 + "  run.sh\n")      # the source is edited and its sums regenerated meanwhile
        return {"passed": True, "checks": {}, "ready_s": 1.0}
    monkeypatch.setattr(can, "boot_check", boot)
    assert can.main(args(tmp_path, "boot", state_dir=str(tmp_path / "st")), out=io.StringIO()) == 0
    assert read(tmp_path / "st")["boot"]["bundle_sums_sha256"] == hashlib.sha256(booted.encode()).hexdigest()


# ---- the state file ----------------------------------------------------------------------------------------------------------

def test_a_record_of_another_can_is_replaced_not_merged(tmp_path):
    b = bundle(tmp_path)
    st = tmp_path / "st"
    can.write_state(st, "other", b, boot=can.boot_record({"passed": True, "checks": {}}, "warm"))
    (st / "toy.can-state.json").write_text((st / "other.can-state.json").read_text())      # a copied file under this can's name
    can.write_state(st, "toy", b, gate=can.gate_record([]))
    s = read(st)
    assert s["name"] == "toy" and "boot" not in s and s["gate"]["passed"] is True
    can.check_state(s)


def test_a_trailing_newline_is_not_accepted_in_a_name_a_digest_or_a_time(tmp_path):
    with pytest.raises(can.CanError):
        can.write_state(tmp_path / "st", "toy\n", bundle(tmp_path), gate=can.gate_record([]))
    with pytest.raises(can.CanError):
        can.check_state({"schema": 1, "name": "toy\n"})
    part = {"passed": True, "problem_count": 0, "problems": [], "bundle_sums_sha256": "a" * 64, "spec_sha256": None, "at": "2026-11-02T10:00:00Z"}
    assert can.check_state({"schema": 1, "name": "toy", "gate": part})
    for key, value in (("bundle_sums_sha256", "a" * 64 + "\n"), ("at", "2026-11-02T10:00:00Z\n")):
        with pytest.raises(can.CanError):
            can.check_state({"schema": 1, "name": "toy", "gate": {**part, key: value}})


# ---- a file that grows while it is hashed ------------------------------------------------------------------------------------

def test_a_file_that_grows_while_it_is_hashed_is_not_reported_as_matching(world, monkeypatch):
    checkout, cans = world
    name = add_spec(checkout, "toy")
    b = stage_bundle(cans, name, extra={"grow.bin": "x" * 61})
    real = view.hashlib.sha256
    appended = []

    class Hasher:
        def __init__(self):
            self._h = real()

        def update(self, data):
            if len(data) == 61 and not appended:
                appended.append(1)
                with open(b / "grow.bin", "ab") as f:
                    f.write(b"appended while the file was being read")
            self._h.update(data)

        def hexdigest(self):
            return self._h.hexdigest()
    monkeypatch.setattr(view.hashlib, "sha256", lambda *a: real(*a) if a else Hasher())
    sums = by_name(build(world), name)["sums"]
    assert appended and sums["state"] == "differs" and "grow.bin changed while it was read" in sums["problems"]


# ---- the repository link -----------------------------------------------------------------------------------------------------

def test_a_repo_id_that_redaction_would_change_gets_no_link_and_none_of_it_appears_in_the_answer(world):
    checkout, cans = world
    token = "hf_" + "A" * 36
    name = add_spec(checkout, "toy", repo=f"someone/{token}")
    out = build(world, redact=lambda t: t.replace(token, "<token>"))
    row = by_name(out, name)
    assert row["weights"]["hub_url"] is None and row["weights"]["repo"] == "someone/<token>"
    assert token not in dump(out)


def test_a_token_in_a_repo_id_is_not_in_the_route_answer_either(root, tmp_path):
    app, checkout = app_for(root, tmp_path)
    token = "hf_" + "B" * 36
    add_spec(checkout, "toy", repo=f"someone/{token}")
    code, data = get(app)
    assert code == 200 and token not in dump(data)
    assert by_name(data, "toy-blackhole")["weights"]["hub_url"] is None


def test_a_repo_id_with_a_trailing_newline_gets_no_link(world):
    checkout, cans = world
    name = add_spec(checkout, "toy", repo="someone/model\\n")           # a TOML escape: the value ends in a newline
    assert by_name(build(world), name)["weights"]["hub_url"] is None


def test_a_spec_file_name_with_a_trailing_newline_is_not_a_spec(world):
    checkout, cans = world
    (checkout / "config" / "can.x.toml\n").write_text("x")
    assert build(world)["cans"] == []


# ---- loading specs for the view ----------------------------------------------------------------------------------------------

def test_a_spec_that_names_a_big_tree_is_not_walked_by_the_view(world, monkeypatch):
    checkout, cans = world
    name = add_spec(checkout, "toy")
    (checkout / "tree").mkdir()
    (checkout / "tree" / "a.py").write_text("x = 1\n")
    path = checkout / "config" / "can.toy.toml"
    path.write_text(path.read_text() + '\n[[extra_tree]]\nsrc = "tree"\ndst = "t"\n')
    walked = []
    real = os.walk
    monkeypatch.setattr(can.os, "walk", lambda *a, **k: walked.append(a) or real(*a, **k))
    row = by_name(build(world), name)
    assert row["spec"]["state"] == "ok" and walked == []
    assert len(can.load_spec(path, base=checkout).extra_code) == 1 and walked, "the command line's loader still walks the tree"


def test_a_spec_whose_extra_tree_is_not_a_directory_is_still_refused_without_walking(world):
    checkout, cans = world
    name = add_spec(checkout, "toy")
    path = checkout / "config" / "can.toy.toml"
    path.write_text(path.read_text() + '\n[[extra_tree]]\nsrc = "missing"\ndst = "t"\n')
    assert by_name(build(world), "toy")["spec"]["state"] == "refused", name      # a refused spec is named by its file's slug


def test_a_spec_with_a_number_for_expect_is_a_refused_row_and_the_others_still_show(world):
    checkout, cans = world
    good = add_spec(checkout, "good")
    add_spec(checkout, "bad")
    path = checkout / "config" / "can.bad.toml"
    path.write_text(path.read_text() + "\n[check]\nexpect = 1\n")
    answer = build(world)
    assert by_name(answer, "bad")["spec"]["state"] == "refused"
    assert by_name(answer, good)["spec"]["state"] == "ok"
    with pytest.raises(can.CanError):
        can.load_spec(path, base=checkout)


def test_any_exception_from_the_loader_is_one_refused_row_with_no_detail(world, monkeypatch):
    checkout, cans = world
    add_spec(checkout, "toy")
    monkeypatch.setattr(can, "load_spec", lambda *a, **k: (_ for _ in ()).throw(ValueError("/secret/path inside")))
    answer = build(world)
    row = by_name(answer, "toy")
    assert row["spec"]["state"] == "refused" and "ValueError" in row["spec"]["text"] and "/secret/path" not in dump(answer)


# ---- second review: records bound to the spec, budget charged on every path, card columns, quadratic scan, string fields -------------

def test_changing_only_the_spec_after_a_recorded_pass_makes_the_record_stale(world):
    from cans_fixture import passed_boot, record
    checkout, cans = world
    name = add_spec(checkout, "toy")
    b = stage_bundle(cans, name)
    record(cans, name, b, gate=can.gate_record([]), boot=passed_boot())
    assert by_name(build(world), name)["summary"] == "booted"
    path = checkout / "config" / "can.toy.toml"
    path.write_text(path.read_text().replace("d18cca4c121edb87d022116d281ce212c9136f57", "e" * 40))     # a new pinned weights revision
    row = by_name(build(world), name)
    assert row["gate"]["state"] == "stale" and row["boot"]["state"] == "stale" and row["summary"] == "incomplete"
    assert "spec file" in row["gate"]["text"]


def test_a_record_with_no_spec_fingerprint_is_not_counted_as_passed(world):
    from cans_fixture import passed_boot, record
    checkout, cans = world
    name = add_spec(checkout, "toy")
    b = stage_bundle(cans, name)
    record(cans, name, b, gate=can.gate_record([]), boot=passed_boot(), spec=None)
    assert by_name(build(world), name)["gate"]["state"] == "stale"


def test_the_command_line_records_the_spec_fingerprint(tmp_path, fake):
    spec = tmp_path / "spec.toml"
    spec.write_text("x = 1\n")
    assert can.main(args(tmp_path, "gate", spec=str(spec), state_dir=str(tmp_path / "st")), out=io.StringIO()) == 1
    assert read(tmp_path / "st")["gate"]["spec_sha256"] == hashlib.sha256(b"x = 1\n").hexdigest()


def test_files_that_change_while_hashed_still_use_their_share_of_the_budget(world, monkeypatch):
    from cans_fixture import CARD
    checkout, cans = world
    name = add_spec(checkout, "toy")
    b = stage_bundle(cans, name, extra={"a-grow.bin": "x" * 61, "b-grow.bin": "y" * 61})
    monkeypatch.setattr(view, "HASH_MAX_BYTES", len(CARD) + 100)
    real = view.hashlib.sha256
    grown = []

    class Hasher:
        def __init__(self):
            self._h = real()

        def update(self, data):
            if len(data) == 61 and not grown:
                grown.append(1)
                with open(b / "a-grow.bin", "ab") as f:
                    f.write(b"more")
            self._h.update(data)

        def hexdigest(self):
            return self._h.hexdigest()
    monkeypatch.setattr(view.hashlib, "sha256", lambda *a: real(*a) if a else Hasher())
    text = by_name(build(world), name)["sums"]["text"]
    # the budget after README.md is 100: a-grow (61, changed) is charged and leaves 39, so b-grow and toy_model.py are skipped
    assert "2 files not hashed" in text


def test_card_columns_are_capped(world):
    checkout, cans = world
    name = add_spec(checkout, "toy")
    cols = view.CARD_MAX_COLS + 5
    head = "| " + " | ".join(f"c{i}" for i in range(cols)) + " |\n| " + " | ".join("-" for _ in range(cols)) + " |\n"
    row = "| " + " | ".join(str(i) for i in range(cols)) + " |\n"
    stage_bundle(cans, name, card=f"## Expected performance\n\n{head}{row}\n## Limitations\n\ntext\n")
    perf = by_name(build(world), name)["card"]["performance"]
    assert len(perf["header"]) == view.CARD_MAX_COLS and len(perf["rows"][0]) == view.CARD_MAX_COLS and perf["columns_more"] == 5


def test_many_separator_rows_do_not_make_the_card_check_quadratic():
    text = "## Expected performance\n\n| a | b |\n" + "| - | - |\n" * 20000 + "\n## Limitations\n\ntext\n"
    done = []
    t = threading.Thread(target=lambda: done.append(can.card_problems(text, "catalog")), daemon=True)
    t.start()
    t.join(10)
    assert done, "the card check did not finish in 10 s"
    assert any("no row with a number" in p for p in done[0])


@pytest.mark.parametrize("field", ["repo", "name", "revision"])
def test_a_spec_field_that_is_not_a_string_is_a_refused_row_and_the_others_show(world, field):
    checkout, cans = world
    good = add_spec(checkout, "good")
    add_spec(checkout, "bad")
    path = checkout / "config" / "can.bad.toml"
    old = {"repo": 'repo = "someone/toy-model"', "name": 'name = "bad-blackhole"',
           "revision": 'revision = "d18cca4c121edb87d022116d281ce212c9136f57"'}[field]
    path.write_text(path.read_text().replace(old, f'{field} = ["org/model"]'))
    answer = build(world)
    assert by_name(answer, "bad")["spec"]["state"] == "refused" and by_name(answer, good)["spec"]["state"] == "ok"
    with pytest.raises(can.CanError, match="must be a string"):         # the loader itself refuses it, for the command line too
        can.load_spec(path, base=checkout)


def test_a_failure_while_building_a_row_is_a_refused_row_not_a_failed_page(world, monkeypatch):
    checkout, cans = world
    add_spec(checkout, "bad")
    stage_bundle(cans, "bad-blackhole")
    monkeypatch.setattr(view.CansView, "_sums", lambda self, pin: (_ for _ in ()).throw(RuntimeError("boom")))
    assert by_name(build(world), "bad")["spec"]["state"] == "refused"
