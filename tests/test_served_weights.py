# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 7 checks that stage 2 served the weights stage 0 names, from the files on disk.

Stage 0's delta.json names the model as `<repo>@<revision>`. Stage 2's swap-check.json holds
`new_model_id`, a label the stage agent typed, and `model_dir`, the directory the test served. The
label alone proves nothing about the weights, so `served_weights_problems` follows the links of the
`*.safetensors` files in model_dir into the Hugging Face cache and reads the revision there.

Two link layouts are covered. A link into `snapshots/<revision>/` names the revision in its path.
prepare_swap.py links each file to its blob (`os.path.realpath` of the snapshot file), so the path
names no revision; then the revision is the snapshot whose same-named file is that blob.

Everything here is tmp dirs and symlinks. Nothing opens a device or reaches the network.
"""
import os

import pytest

from orchard.package import served_weights_problems, split_model_id
from package_fakes import NEW, NEW_REV, hf_snapshot

OTHER_REV = "2b6f474b4ee3dd2d1cc456c28b8abbb403b955aa"
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
STAGE0 = f"{NEW}@{NEW_REV}"


@pytest.fixture
def hub(tmp_path):
    """One HF home holding NEW at two revisions with different weights, and another repo at
    NEW_REV."""
    home = tmp_path / "hf"
    snaps = {rev: hf_snapshot(home, NEW, rev, {s: f"{rev} {s}" for s in SHARDS})
             for rev in (NEW_REV, OTHER_REV)}
    other = hf_snapshot(home, "Someone/Else", NEW_REV, {s: f"else {s}" for s in SHARDS})
    return {"snaps": snaps, "other": other, "tmp": tmp_path}


def model_dir(tmp, links: dict) -> str:
    """A model dir whose files are symlinks to the given targets (name -> target path)."""
    md = tmp / "model-dir"
    md.mkdir()
    (md / "config.json").write_text("{}")
    for name, target in links.items():
        os.symlink(target, md / name)
    return str(md)


def snapshot_links(snap):
    """Links into snapshots/<revision>/, the layout the HF cache itself uses."""
    return {s: str(snap / s) for s in SHARDS}


def blob_links(snap):
    """Links straight to the blobs, as prepare_swap.py makes them."""
    return {s: os.path.realpath(snap / s) for s in SHARDS}


@pytest.mark.parametrize("links", [snapshot_links, blob_links])
def test_a_short_label_and_links_to_stage_0s_revision_pass(hub, links):
    md = model_dir(hub["tmp"], links(hub["snaps"][NEW_REV]))
    assert served_weights_problems(STAGE0, NEW, md) == []


def test_the_live_case_passes_a_short_label_with_matching_links(hub):
    # Run 3: stage 0 names 'Altworld/Hemmingway-1@<rev>', stage 2's label is 'Altworld/Hemmingway-1'.
    md = model_dir(hub["tmp"], blob_links(hub["snaps"][NEW_REV]))
    assert served_weights_problems("Altworld/Hemmingway-1@" + NEW_REV, "Altworld/Hemmingway-1", md) == []


def test_a_label_with_the_right_revision_passes(hub):
    md = model_dir(hub["tmp"], snapshot_links(hub["snaps"][NEW_REV]))
    assert served_weights_problems(STAGE0, STAGE0, md) == []


def test_a_label_with_another_revision_fails(hub):
    md = model_dir(hub["tmp"], snapshot_links(hub["snaps"][NEW_REV]))
    problems = served_weights_problems(STAGE0, f"{NEW}@{OTHER_REV}", md)
    assert problems and OTHER_REV in problems[0] and NEW_REV in problems[0]


@pytest.mark.parametrize("links", [snapshot_links, blob_links])
def test_links_to_another_revision_fail_and_name_both(hub, links):
    md = model_dir(hub["tmp"], links(hub["snaps"][OTHER_REV]))
    problems = served_weights_problems(STAGE0, NEW, md)
    assert len(problems) == 1
    assert OTHER_REV in problems[0] and NEW_REV in problems[0]


@pytest.mark.parametrize("links", [snapshot_links, blob_links])
def test_links_to_mixed_revisions_fail(hub, links):
    a, b = links(hub["snaps"][NEW_REV]), links(hub["snaps"][OTHER_REV])
    md = model_dir(hub["tmp"], {SHARDS[0]: a[SHARDS[0]], SHARDS[1]: b[SHARDS[1]]})
    problems = served_weights_problems(STAGE0, NEW, md)
    assert problems and "different revisions" in problems[0]
    assert NEW_REV in problems[0] and OTHER_REV in problems[0]


def test_a_label_naming_another_repo_fails(hub):
    md = model_dir(hub["tmp"], snapshot_links(hub["snaps"][NEW_REV]))
    problems = served_weights_problems(STAGE0, "Someone/Else", md)
    assert problems and "Someone/Else" in problems[0] and NEW in problems[0]


def test_links_into_another_repos_snapshot_fail(hub):
    md = model_dir(hub["tmp"], snapshot_links(hub["other"]))
    problems = served_weights_problems(STAGE0, NEW, md)
    assert problems and "Someone/Else" in problems[0] and NEW in problems[0]


def test_weight_files_that_are_not_links_fail_with_a_clear_reason(hub):
    md = model_dir(hub["tmp"], {})
    for s in SHARDS:
        (hub["tmp"] / "model-dir" / s).write_text("copied weights")
    problems = served_weights_problems(STAGE0, NEW, md)
    assert problems and "no Hugging Face snapshot" in problems[0] and SHARDS[0] in problems[0]


def test_a_link_to_a_file_outside_any_snapshot_fails(hub):
    stray = hub["tmp"] / "stray.safetensors"
    stray.write_text("weights")
    md = model_dir(hub["tmp"], {SHARDS[0]: str(stray)})
    problems = served_weights_problems(STAGE0, NEW, md)
    assert problems and "no Hugging Face snapshot" in problems[0]


def test_a_model_dir_without_weight_files_fails(hub):
    md = model_dir(hub["tmp"], {})
    problems = served_weights_problems(STAGE0, NEW, md)
    assert problems and "no *.safetensors" in problems[0]


def test_a_missing_model_dir_fails(hub):
    problems = served_weights_problems(STAGE0, NEW, str(hub["tmp"] / "nowhere"))
    assert problems and "not a directory" in problems[0]


def test_stage_0_must_name_a_revision(hub):
    md = model_dir(hub["tmp"], snapshot_links(hub["snaps"][NEW_REV]))
    problems = served_weights_problems(NEW, NEW, md)
    assert problems and "names no revision" in problems[0]


def test_ids_split_into_repo_and_revision():
    assert split_model_id(STAGE0) == (NEW, NEW_REV)
    assert split_model_id(NEW) == (NEW, None)
    assert split_model_id(None) == ("", None)
