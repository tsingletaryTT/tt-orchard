# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 1's gate for a scoring model (orchard/stages.py `gate_scoring_reference`, orchard/scoring_check.py `reference_reasons`, issue 378): the
files `python3 -m orchard.scoring_reference` writes, checked in the run directory before stage 2 uses them."""
import hashlib
import json

import pytest

from orchard import scoring_check as sc
from orchard.stages import gate_scoring_reference
from scoring_check_fakes import HONEST, make_run
from scoring_embed_fakes import EMBED_DIM, HONEST_EMBED, MEDIA_EMBED, embed_rows, make_embed_run, media_entry

PNG = b"\x89PNG\r\n\x1a\n" + b"fake image bytes" * 20


def reference_dir(run):
    return run / "stages" / "1" / "evidence" / "reference"


def gate(run):
    return gate_scoring_reference(run / "stages" / "1", run)


def test_the_gate_accepts_the_recorders_files_of_both_kinds_and_lists_them_and_the_media(tmp_path):
    rows = embed_rows({n: [media_entry("a.png", PNG)] for n in (0, 5)})
    run, stage, _ = make_embed_run(tmp_path, MEDIA_EMBED, rows=rows, media_files={"a.png": PNG})
    got = gate(run)
    assert got.ok and list(got.evidence) == [sc.REFERENCE_FILE, sc.PAIRS_FILE, sc.MEDIA_DIR + "a.png"]
    run2, _, _ = make_run(tmp_path / "r", HONEST)
    assert gate(run2).ok


def edit_json(path, fn):
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc))


def as_reranker(ev):
    """The embedding files turned into a reranker set whose scores are one short."""
    pairs = json.loads((ev / "scoring-pairs.json").read_text())
    pairs["model_kind"] = "reranker"
    for p in pairs["pairs"]:
        p["group"] = "g"
    text = json.dumps(pairs) + "\n"
    (ev / "scoring-pairs.json").write_text(text)

    def fix(doc):
        doc["model_kind"] = "reranker"
        doc.pop("vectors")
        doc["scores"] = [1.0]
        doc["pairs"]["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    edit_json(ev / "scoring-reference.json", fix)


@pytest.mark.parametrize("what,word", [
    ("missing_pairs", "scoring-pairs.json"), ("missing_ref", "scoring-reference.json"), ("pairs_changed", "names a pair file other than"),
    ("kind_flipped", "is for kind"), ("short_scores", "has 1 scores for the 30 pairs"), ("nan_vector", "NaN is not allowed"),
])
def test_the_gate_refuses_each_way_the_recorders_files_can_be_wrong(tmp_path, what, word):
    run, stage, _ = make_embed_run(tmp_path)
    ev = reference_dir(run)
    if what == "missing_pairs":
        (ev / "scoring-pairs.json").unlink()
    elif what == "missing_ref":
        (ev / "scoring-reference.json").unlink()
    elif what == "pairs_changed":
        (ev / "scoring-pairs.json").write_text((ev / "scoring-pairs.json").read_text() + " ")
    elif what == "kind_flipped":
        edit_json(ev / "scoring-reference.json", lambda d: d.update(model_kind="reranker"))
    elif what == "short_scores":
        as_reranker(ev)
    else:
        edit_json(ev / "scoring-reference.json", lambda d: d["vectors"].__setitem__(3, [float("nan")] * EMBED_DIM))
    got = gate(run)
    assert not got.ok and any(word in r for r in got.reasons), got.reasons


@pytest.mark.parametrize("text", ["[]", "null", '{"model_kind": 5}', '{"model_kind": []}', "{", "", '{"model_kind": "embedding", "pairs": 3}'])
def test_the_gate_never_raises_on_a_pair_file_of_the_wrong_shape(tmp_path, text):
    run, stage, _ = make_embed_run(tmp_path)
    (reference_dir(run) / "scoring-pairs.json").write_text(text)
    got = gate(run)
    assert not got.ok and got.reasons


def test_a_media_file_named_by_the_pair_file_that_is_missing_is_a_reason(tmp_path):
    run, stage, _ = make_embed_run(tmp_path, MEDIA_EMBED, rows=embed_rows({0: [media_entry("a.png", PNG)]}))
    got = gate(run)
    assert not got.ok and any("not a regular file directly under" in r for r in got.reasons)
