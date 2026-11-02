# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Findings of the Copilot review of PR 398: stage 1 checks the token ids that stage 2's template reads, media is hashed through a pinned root
under a size cap, and a scoring model of class full-port has one pause after stage 2 and not two."""
import hashlib
import os

import pytest

from orchard import scoring_check as sc
from orchard import supervisor
from orchard.supervisor import EXIT_BLOCKED, Control
from scoring_check_fakes import HONEST, make_run
from scoring_embed_fakes import MEDIA_EMBED, embed_rows, make_embed_run, media_entry
from test_scoring_stage1_gate import PNG, edit_json, gate, reference_dir
from test_supervisor_scoring import FULL, Stop, make, never_wait, rig, started  # noqa: F401  (rig is a fixture)

BAD_IDS = [None, [], "abc", [1, "2"], [1, True], [-1], [1.5], [None], [10 ** 400], 7]


def retarget(run, ev):
    """The reference names the pair file by its hash, so it is rewritten to name the edited pair file."""
    edit_json(ev / "scoring-reference.json", lambda d: d["pairs"].update(sha256=hashlib.sha256((ev / "scoring-pairs.json").read_bytes()).hexdigest()))


@pytest.mark.parametrize("ids", BAD_IDS + [list(range(sc.MAX_IDS + 1))])
@pytest.mark.parametrize("kind", ["embedding", "reranker"])
def test_the_gate_refuses_a_pair_row_whose_token_ids_stage_2_would_refuse(tmp_path, kind, ids):
    run, stage, _ = make_embed_run(tmp_path) if kind == "embedding" else make_run(tmp_path, HONEST)
    ev = reference_dir(run)
    edit_json(ev / "scoring-pairs.json", lambda d: d["pairs"][3].update(input_ids=ids))
    retarget(run, ev)
    got = gate(run)
    word = "an integer has too many digits" if ids == [10 ** 400] else "token ids"        # the safe reader refuses a 401-digit integer first
    assert not got.ok and any(word in r for r in got.reasons), got.reasons


def test_the_gate_refuses_a_relevant_label_that_is_not_true_or_false(tmp_path):
    run, stage, _ = make_run(tmp_path, HONEST)
    ev = reference_dir(run)
    edit_json(ev / "scoring-pairs.json", lambda d: d["pairs"][0].update(relevant="yes"))
    retarget(run, ev)
    got = gate(run)
    assert not got.ok and any("relevant is not true or false" in r for r in got.reasons)


def media_run(tmp_path, **files):
    return make_embed_run(tmp_path, MEDIA_EMBED, rows=embed_rows({0: [media_entry("a.png", PNG)]}), media_files={"a.png": PNG, **files})


def test_a_media_file_replaced_by_a_link_to_identical_bytes_is_refused(tmp_path):
    run, stage, _ = media_run(tmp_path, **{"real.png": PNG})
    assert gate(run).ok
    media = reference_dir(run) / "media"
    (media / "a.png").unlink()
    os.symlink(media / "real.png", media / "a.png")                  # the same bytes, so the same sha256, but not a regular file
    got = gate(run)
    assert not got.ok and any("can be read within" in r for r in got.reasons), got.reasons


def test_a_media_file_over_the_size_cap_is_refused_whatever_digest_the_pair_file_gives(tmp_path):
    run, stage, _ = media_run(tmp_path)
    with open(reference_dir(run) / "media" / "a.png", "wb") as f:
        f.truncate(sc.MEDIA_MAX + 1)                                 # a sparse file just over the cap: refused before any read
    got = gate(run)
    assert not got.ok and any("can be read within" in r and "larger than" in r for r in got.reasons), got.reasons


def test_a_media_directory_that_is_a_link_inside_the_run_directory_is_refused(tmp_path):
    run, stage, _ = media_run(tmp_path)
    ev = reference_dir(run)
    (ev / "media").rename(ev / "real-media")
    os.symlink(ev / "real-media", ev / "media")
    got = gate(run)
    assert not got.ok and any("can be read within" in r for r in got.reasons), got.reasons


@pytest.mark.parametrize("path", ["stages/1/evidence/reference/media/", "stages/1/evidence/reference/media/sub/a.png"])
def test_a_media_path_that_is_not_a_file_directly_under_the_media_directory_is_refused(tmp_path, path):
    run, stage, _ = media_run(tmp_path)
    edit_json(reference_dir(run) / "scoring-pairs.json", lambda d: d["pairs"][0]["media"][0].update(path=path))
    got = gate(run)
    assert not got.ok and any("not a file directly under" in r for r in got.reasons), got.reasons


# ---- the pause of a full-port scoring run -------------------------------------------------------------------------------------

def test_an_unattended_full_port_scoring_run_ends_blocked_with_scoring_route_and_not_full_port_stages_unbuilt(rig):
    rig.args.full_port = True
    rig.args.unattended = True
    make(rig, delta=FULL)
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    assert [d["code"] for d in rig.decisions() if d["decision"] == "blocked"] == ["scoring-route"]
    assert [d["reason"] for d in rig.decisions() if d.get("decision") == "pause"] == [supervisor.SCORING_STAGES_UNBUILT]
    assert not started(rig, 3)


def test_an_attended_full_port_scoring_run_pauses_once_before_stage_3_and_one_resume_is_enough(rig):
    rig.args.full_port = True
    make(rig, delta=FULL)
    waits = []

    def on_sleep():
        waits.append(1)
        if len(waits) > 1:
            raise Stop("the run paused a second time")
        Control(rig.run_dir).write("resume")
    rig.on_sleep = on_sleep
    try:
        rig.run()
    except Stop:
        pass
    pauses = [d["reason"] for d in rig.decisions() if d.get("decision") == "pause"]
    assert supervisor.FULL_PORT_STAGES_UNBUILT not in pauses and pauses.count(supervisor.SCORING_STAGES_UNBUILT) == 1
    assert started(rig, 3)
