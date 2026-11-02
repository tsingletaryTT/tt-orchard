# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Tests added after the mutation run of the scoring slice 2 found guards that no test depended on alone: the mean cosine bound (the minimum bound
also fired), the worker's check of a media file at the point of use (the parent's check also fired), the media byte total, and the ledger check of
the test marker (the template's own checks also fired)."""
import importlib.util
import json
import os

import pytest

from orchard import scoring_check as sc
from orchard import supervisor
from orchard.supervisor import EXIT_BLOCKED
from scoring_check_fakes import run_template
from scoring_embed_fakes import HONEST_EMBED, MEDIA_EMBED, NOISY_EMBED, embed_rows, make_embed_run, media_entry
from test_scoring_embedding import PNG, gate, mentions, media_rows
from test_supervisor_scoring import Rig, make, never_wait, rig, with_model  # noqa: F401  (rig is a fixture)

spec = importlib.util.spec_from_file_location("scoring_check_template_survivors", sc.TEMPLATE)
tpl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tpl)


def test_the_mean_cosine_bound_alone_refuses_a_set_whose_minimum_is_fine(tmp_path):
    cfg = sc.draft_config(str(tmp_path / "snapshot"), "embedding")
    cfg["bounds"]["cosine_mean_min"] = 0.9999999
    port = NOISY_EMBED.replace("0.5", "0.001")                 # every cosine is about 0.99999: far above the minimum bound of 0.985
    run, stage, _ = make_embed_run(tmp_path, port, config=cfg)
    assert run_template(tmp_path, run, stage).returncode == 0
    reasons = gate(run, stage)
    mentions(reasons, "mean cosine to the reference vector for all inputs")
    mentions(reasons, "mean cosine to the reference vector for kind text")
    assert not [r for r in reasons if "minimum cosine" in r]


# A port that rewrites the media file of a later input while it embeds the first: the parent hashed the file before the port ran.
TAMPER = '''
import ttnn, random
def load(model_dir, mesh_device):
    return {}
def embed(model, ids, media=None):
    n = ids[0] - 1
    if n == 0:
        with open("stages/1/evidence/reference/media/a.png", "ab") as f:
            f.write(b"changed while the port ran")
    r = random.Random(n)
    return [r.uniform(-1, 1) for _ in range(8)]
def close(model):
    pass
'''


def test_the_worker_checks_a_media_files_hash_when_it_reads_it_and_not_only_when_the_parent_hashed_it(tmp_path):
    rows = embed_rows({0: [media_entry("a.png", PNG)], 9: [media_entry("a.png", PNG)]})      # input 9 is in a later request than input 0
    run, stage, _ = make_embed_run(tmp_path, TAMPER, rows=rows, media_files={"a.png": PNG})
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "does not match the sha256 in the pair file" in done.stdout, done.stdout
    assert not (stage / "evidence" / "scoring-check.json").exists()


def test_the_media_bytes_of_one_set_are_limited_in_all(tmp_path, monkeypatch):
    run, stage, _ = make_embed_run(tmp_path, MEDIA_EMBED, rows=embed_rows({0: [media_entry("a.png", PNG)], 1: [media_entry("a.png", PNG)]}),
                                   media_files={"a.png": PNG})
    monkeypatch.setattr(tpl, "RUN_DIR", run)
    rows = json.loads((run / "stages" / "1" / "evidence" / "reference" / "scoring-pairs.json").read_text())["pairs"]
    monkeypatch.setattr(tpl, "MEDIA_TOTAL_MAX", len(PNG) * 2 - 1)               # one file fits, two do not
    with pytest.raises(tpl.CheckFailed, match="larger than"):
        tpl.check_media(rows)
    monkeypatch.setattr(tpl, "MEDIA_TOTAL_MAX", len(PNG) * 2)
    assert tpl.check_media(rows)[0][0]["sha256"]


def test_more_media_entries_than_a_row_may_have_are_refused_by_the_check_and_the_gate(tmp_path, monkeypatch):
    rows = embed_rows({0: [media_entry("a.png", PNG)] * (tpl.MEDIA_PER_ROW + 1)})
    run, stage, _ = make_embed_run(tmp_path, MEDIA_EMBED, rows=rows, media_files={"a.png": PNG})
    monkeypatch.setattr(tpl, "RUN_DIR", run)
    with pytest.raises(tpl.CheckFailed, match="at most 8 entries"):
        tpl.check_media(json.loads((run / "stages" / "1" / "evidence" / "reference" / "scoring-pairs.json").read_text())["pairs"])
    mentions(sc.reference_reasons(run, "embedding")[0], "media must be a list of at most 8 entries")


def test_a_test_marker_rewritten_after_the_hardware_test_is_refused_because_the_ledger_holds_the_hash_of_the_one_written(rig, monkeypatch):
    rig.args.unattended = True
    make(rig)
    never_wait(rig)
    original = supervisor.Supervisor._check_gate

    def rewrite_then_check(self, spec, stage_dir):
        marker = stage_dir / "test-result.json"
        if spec.number == 2 and marker.is_file():
            doc = json.loads(marker.read_text())
            doc["note"] = "edited after the test"                    # still a marker the template's own checks accept
            marker.write_text(json.dumps(doc))
        return original(self, spec, stage_dir)
    monkeypatch.setattr(supervisor.Supervisor, "_check_gate", rewrite_then_check)
    assert rig.run() == EXIT_BLOCKED
    block = [d for d in rig.decisions() if d["decision"] == "blocked"][0]
    assert block["code"] == "stage-failed" and "not the file the supervisor recorded" in block["reason"]
