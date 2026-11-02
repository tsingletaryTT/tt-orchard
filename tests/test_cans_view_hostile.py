# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hostile input for the Cans view (orchard/cans_view.py): links, FIFOs, huge and malformed files, NUL and control characters, paths that
leave the bundle. Each test names the guard it exercises. A FIFO test runs the build in a thread with a deadline, because the failure these
tests look for is a hang. See tests/hostile_inputs.py for the shared list of hostile shapes."""
import json
import os
import threading

import pytest

from cans_fixture import add_spec, by_name, dump, make_checkout, passed_boot, record, stage_bundle
from orchard import can
from orchard import cans_view as view


@pytest.fixture
def world(tmp_path):
    root = make_checkout(tmp_path)
    cans = tmp_path / "cans"
    cans.mkdir()
    name = add_spec(root, "toy")
    return root, cans, name


def build(world, **kw):
    root, cans, _ = world
    return view.CansView(root / "config", root, cans, **kw).build()


def build_with_deadline(world, seconds=10):
    """The answer, or a failed test when the build does not finish (a FIFO opened for read blocks until a writer comes)."""
    out = {}
    t = threading.Thread(target=lambda: out.update(a=build(world)), daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), "the build hung"
    return out["a"]


def row(world):
    return by_name(build_with_deadline(world), world[2])


# ---- links ------------------------------------------------------------------------------------------------------------------

def test_a_bundle_that_is_a_link_is_not_read(world, tmp_path):
    root, cans, name = world
    real = stage_bundle(tmp_path / "elsewhere", name)
    (cans / name).symlink_to(real, target_is_directory=True)
    r = row(world)
    assert r["bundle"]["state"] == "link" and r["sums"]["state"] == "none" and r["card"]["state"] == "none"
    assert r["card"]["performance"] is None, "nothing was read through the link"


def test_a_cans_folder_that_is_a_link_is_refused(world, tmp_path):
    root, cans, name = world
    stage_bundle(tmp_path / "real-cans", name)
    link = tmp_path / "linked-cans"
    link.symlink_to(tmp_path / "real-cans", target_is_directory=True)
    answer = view.CansView(root / "config", root, link).build()
    assert answer["folder"]["state"] == "unreadable" and by_name(answer, name)["bundle"]["state"] == "not-staged"


def test_a_listed_file_that_is_a_link_is_a_problem_and_the_target_is_not_hashed(world, tmp_path):
    root, cans, name = world
    b = stage_bundle(cans, name)
    secret = tmp_path / "secret.txt"
    secret.write_text("outside")
    (b / "run.sh").unlink()
    (b / "run.sh").symlink_to(secret)
    sums = row(world)["sums"]
    assert sums["state"] == "differs"
    assert "run.sh is a link" in sums["problems"]
    assert any(p.startswith("run.sh ") and "link" in p for p in sums["problems"])


def test_a_link_to_a_directory_inside_the_bundle_is_not_entered(world, tmp_path):
    root, cans, name = world
    b = stage_bundle(cans, name)
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "big.bin").write_text("x")
    (b / "sub").symlink_to(tmp_path / "outside", target_is_directory=True)
    sums = row(world)["sums"]
    assert "sub is a link" in sums["problems"] and not any("big.bin" in p for p in sums["problems"])


def test_a_spec_file_that_is_a_link_is_not_listed(world, tmp_path):
    root, cans, name = world
    target = tmp_path / "real.toml"
    target.write_text("x")
    (root / "config" / "can.linked.toml").symlink_to(target)
    assert [c["slug"] for c in build(world)["cans"]] == ["toy"]


def test_a_state_file_that_is_a_link_is_unreadable(world, tmp_path):
    root, cans, name = world
    b = stage_bundle(cans, name)
    real = record(cans, name, b, gate=can.gate_record([]))
    moved = tmp_path / "moved.json"
    real.rename(moved)
    real.symlink_to(moved)
    assert row(world)["gate"]["state"] == "unreadable"


# ---- FIFOs and special files ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["SHA256SUMS", "README.md", "state", "listed-file", "unlisted-file"])
def test_a_fifo_in_place_of_a_file_never_hangs_the_build(world, which):
    root, cans, name = world
    b = stage_bundle(cans, name)
    if which == "state":
        record(cans, name, b, gate=can.gate_record([]))
        path = cans / f"{name}{can.STATE_SUFFIX}"
    elif which == "unlisted-file":
        path = b / "pipe"
    else:
        path = b / ("run.sh" if which == "listed-file" else which)
    if path.exists():
        path.unlink()
    os.mkfifo(path)
    r = row(world)
    assert r["bundle"]["state"] == "staged"
    if which == "SHA256SUMS":
        assert r["sums"]["state"] == "unreadable"
    elif which == "README.md":
        assert r["card"]["state"] == "unreadable"
    elif which == "state":
        assert r["gate"]["state"] == "unreadable"
    elif which == "listed-file":
        assert r["sums"]["state"] == "differs" and any("run.sh" in p for p in r["sums"]["problems"])
    else:
        assert r["sums"]["state"] == "differs" and "pipe is not a regular file or directory" in r["sums"]["problems"]


def test_a_listed_fifo_is_not_waited_on_even_when_the_listing_could_not_tell_it_from_a_file(world, monkeypatch):
    """With the listing cut short, a listed path is opened without the walk having classified it; a FIFO there must not block the open."""
    root, cans, name = world
    b = stage_bundle(cans, name)
    (b / "run.sh").unlink()
    os.mkfifo(b / "run.sh")
    monkeypatch.setattr(view, "SCAN_MAX_ENTRIES", 0)
    r = row(world)
    assert r["sums"]["state"] == "differs" and "run.sh is not a regular file" in r["sums"]["problems"]


def test_a_bundle_that_is_a_plain_file_is_said_not_to_be_a_directory(world):
    root, cans, name = world
    (cans / name).write_text("not a directory")
    assert row(world)["bundle"]["state"] == "not-a-directory"


# ---- huge and malformed files -----------------------------------------------------------------------------------------------

def test_a_sums_file_over_the_limit_is_not_read(world):
    root, cans, name = world
    b = stage_bundle(cans, name)
    (b / can.SUMS_FILE).write_bytes(b"a" * (view.SUMS_MAX_BYTES + 1))
    r = row(world)
    assert r["sums"]["state"] == "unreadable" and "larger than 1 MB" in r["sums"]["text"]


def test_a_sums_file_with_too_many_lines_is_refused_not_walked(world):
    root, cans, name = world
    b = stage_bundle(cans, name)
    (b / can.SUMS_FILE).write_text(("0" * 64 + "  f\n") * (view.SUMS_MAX_LINES + 1))
    assert row(world)["sums"]["state"] == "unreadable"


def test_a_readme_over_the_limit_is_not_read(world):
    root, cans, name = world
    b = stage_bundle(cans, name, card="x" * (view.CARD_MAX_BYTES + 1))
    r = row(world)
    assert r["card"]["state"] == "unreadable" and "larger than 256 KB" in r["card"]["text"]


def test_a_state_file_over_the_limit_or_not_json_or_deeply_nested_is_unreadable(world):
    root, cans, name = world
    b = stage_bundle(cans, name)
    path = cans / f"{name}{can.STATE_SUFFIX}"
    for body in ("x" * (can.STATE_MAX_BYTES + 1), "{not json", "[" * 5000, '{"schema": 1, "name": "toy-blackhole", "gate": NaN}', "[]", ""):
        path.write_text(body)
        view_ = view.CansView(world[0] / "config", world[0], cans)
        r = by_name(view_.build(), name)
        assert r["gate"]["state"] == "unreadable" and r["boot"]["state"] == "unreadable", body[:20]
        assert r["sums"]["state"] == "ok"


def test_a_state_file_with_a_repeated_key_is_unreadable(world):
    root, cans, name = world
    stage_bundle(cans, name)
    (cans / f"{name}{can.STATE_SUFFIX}").write_text('{"schema": 1, "schema": 1, "name": "toy-blackhole"}')
    assert row(world)["gate"]["state"] == "unreadable"


def test_a_spec_over_the_loader_limit_is_a_refused_row(world):
    root, cans, name = world
    (root / "config" / "can.huge.toml").write_text("# " + "x" * (can.MAX_SPEC_BYTES + 10))
    answer = build(world)
    assert by_name(answer, "huge")["spec"]["state"] == "refused" and by_name(answer, name)["spec"]["state"] == "ok"


def test_a_card_that_is_not_utf8_is_read_with_replacement_and_does_not_raise(world):
    root, cans, name = world
    b = stage_bundle(cans, name)
    (b / "README.md").write_bytes(b"## Expected performance\n\n| M | V |\n| - | - |\n| \xff\xfe | 3 |\n\n## Limitations\n\ntext\n")
    can.write_sums(b)
    r = row(world)
    assert r["card"]["state"] == "ok" and r["card"]["performance"]["rows"][0][1] == "3"


# ---- NUL, control characters and paths that leave the bundle ----------------------------------------------------------------

def test_nul_and_control_characters_in_a_sums_path_are_shown_as_spaces_and_refused_as_a_path(world):
    root, cans, name = world
    b = stage_bundle(cans, name)
    (b / can.SUMS_FILE).write_bytes(b"0" * 64 + b"  run\x00.sh\n" + b"1" * 64 + b"  a\x1b[31mb\n")
    r = row(world)
    assert r["sums"]["state"] == "differs"
    text = dump(r["sums"])
    assert "\\u0000" not in text and "\\u001b" not in text


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/hostname", "a/../../outside.txt", "./run.sh", "a//b", "..", "dir/"])
def test_a_sums_path_that_leaves_the_bundle_is_a_problem_and_nothing_outside_is_read(world, tmp_path, path):
    root, cans, name = world
    b = stage_bundle(cans, name)
    outside = tmp_path / "cans" / "outside.txt"
    outside.write_text("content")
    import hashlib
    digest = hashlib.sha256(b"content").hexdigest()
    (b / can.SUMS_FILE).write_text(f"{digest}  {path}\n")
    r = row(world)
    assert r["sums"]["state"] == "differs"
    assert r["sums"]["problem_count"] >= 1
    assert not any("match" in r["sums"]["text"] for _ in [0])


def test_a_malformed_sums_line_is_a_problem_naming_its_line_number(world):
    root, cans, name = world
    b = stage_bundle(cans, name)
    (b / can.SUMS_FILE).write_text("not a digest line\n")
    assert row(world)["sums"]["problems"][0] == "SHA256SUMS line 1 is not a digest and a path"


def test_a_name_with_a_newline_in_the_bundle_is_shown_without_the_newline(world):
    root, cans, name = world
    b = stage_bundle(cans, name)
    (b / "two\nlines").write_text("x")
    problems = row(world)["sums"]["problems"]
    assert "SHA256SUMS does not list two lines" in problems


# ---- budgets ----------------------------------------------------------------------------------------------------------------

def test_when_the_hash_budget_is_spent_the_check_says_partly_checked_and_never_matches(world, monkeypatch):
    root, cans, name = world
    stage_bundle(cans, name)
    monkeypatch.setattr(view, "HASH_MAX_BYTES", 20)
    r = row(world)
    assert r["sums"]["state"] == "partial" and r["sums"]["tone"] == "warn" and "not hashed" in r["sums"]["text"]
    assert (r["tone"], r["summary"]) == ("warn", "incomplete")


def test_a_difference_found_before_the_budget_ran_out_is_still_reported_as_a_difference(world, monkeypatch):
    root, cans, name = world
    b = stage_bundle(cans, name)
    (b / "README.md").write_text("changed")
    monkeypatch.setattr(view, "HASH_MAX_BYTES", 20)
    r = row(world)
    assert r["sums"]["state"] == "differs" and "partly checked" in r["sums"]["text"]


def test_when_the_listing_is_cut_short_the_check_says_so_and_does_not_report_unlisted_or_missing_files(world, monkeypatch):
    root, cans, name = world
    stage_bundle(cans, name)
    monkeypatch.setattr(view, "SCAN_MAX_ENTRIES", 2)
    r = row(world)
    assert r["sums"]["state"] in ("partial", "differs") and "cut short" in r["sums"]["text"]
    assert not any("is not in the bundle" in p for p in r["sums"]["problems"])
    assert r["sums"]["state"] != "ok"


def test_a_bundle_nested_deeper_than_the_walk_allows_is_not_called_a_match(world, monkeypatch):
    root, cans, name = world
    deep = "/".join("d" for _ in range(view.SCAN_MAX_DEPTH + 2)) + "/f.txt"
    stage_bundle(cans, name, extra={deep: "x"})
    r = row(world)
    assert r["sums"]["state"] != "ok" and "cut short" in r["sums"]["text"]
