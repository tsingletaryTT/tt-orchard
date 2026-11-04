# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 leaves no evidence file from an earlier attempt for a check that this attempt skips (issue 452).

`readout-facts.json` is written only for a restricted head, `vocab-compare.json` only for a vocabulary trim, `audio-compare.json` only on
the audio route. A file from an earlier attempt of the same run stayed in `stages/0/evidence/` when a later attempt did not write it, and
a reader of the directory could take it for the current snapshot's. The template now removes every evidence file it owns at the start of
an attempt. A file it does not own (the agent's own notes) is left alone. The script runs as a subprocess, the way the agent runs it."""
from __future__ import annotations

import json
import os
import re

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("tokenizers")
from test_delta_triage_alignment import decider_case, run, template  # noqa: E402,F401  (decider_case is a fixture)
from test_delta_triage_template import TEMPLATE, run_triage, stage_for, tokenizer_json  # noqa: E402,F401
from test_delta_triage_audio import audio_case, run_ok  # noqa: E402,F401  (audio_case is a fixture)

CONDITIONAL = ("readout-facts.json", "vocab-compare.json", "audio-compare.json")


def plant(stage, names=CONDITIONAL):
    ev = stage / "evidence"
    ev.mkdir(exist_ok=True)
    for name in names:
        (ev / name).write_text(json.dumps({"stale": "from an earlier attempt"}))
    (ev / "agent-notes.json").write_text("{}")          # not a file the template writes


def test_a_text_attempt_that_skips_each_conditional_check_leaves_no_file_from_an_earlier_attempt(decider_case):
    tmp_path, base, new, head, ids, with_rows = decider_case          # no sidecar: no restricted head, no vocabulary trim
    stage = stage_for(tmp_path, new, base, model_id="perplexity-ai/pplx-decider-v1.2")
    plant(stage)
    r = run_triage(stage)
    assert r.returncode == 0, r.stderr[-800:]
    for name in CONDITIONAL:
        assert not (stage / "evidence" / name).exists(), name
    assert (stage / "evidence" / "config-compare.json").is_file()      # this attempt's own evidence is there
    assert (stage / "evidence" / "agent-notes.json").is_file()         # a file the template does not own is kept


def test_a_second_attempt_without_the_sidecar_drops_the_readout_facts_of_the_first(decider_case):
    tmp_path, base, new, head, ids, with_rows = decider_case
    stage, d = run(tmp_path, with_rows(head[ids]), base)
    assert d["class"] == "restricted-head" and (stage / "evidence" / "readout-facts.json").is_file()
    (new / "readout.safetensors").unlink()
    r = run_triage(stage)
    assert r.returncode == 0, r.stderr[-800:]
    assert json.loads((stage / "delta.json").read_text())["head"]["restricted"] is False
    assert not (stage / "evidence" / "readout-facts.json").exists()


def test_an_audio_attempt_leaves_none_of_the_text_files_of_an_earlier_attempt(audio_case):
    stage = audio_case()
    plant(stage, ("readout-facts.json", "vocab-compare.json", "head-compare.json", "tensor-compare.json"))
    run_ok(stage)
    ev = stage / "evidence"
    for name in ("readout-facts.json", "vocab-compare.json", "head-compare.json", "tensor-compare.json"):
        assert not (ev / name).exists(), name
    assert (ev / "audio-compare.json").is_file() and (ev / "agent-notes.json").is_file()


def test_a_link_with_an_evidence_name_is_removed_and_its_target_is_left_alone(decider_case):
    tmp_path, base, new, head, ids, with_rows = decider_case
    stage = stage_for(tmp_path, new, base, model_id="perplexity-ai/pplx-decider-v1.2")
    target = tmp_path / "precious.txt"
    target.write_text("keep me")
    (stage / "evidence").mkdir()
    os.symlink(target, stage / "evidence" / "vocab-compare.json")
    r = run_triage(stage)
    assert r.returncode == 0, r.stderr[-800:]
    assert not os.path.lexists(stage / "evidence" / "vocab-compare.json")
    assert target.read_text() == "keep me"


def test_a_stale_file_that_cannot_be_removed_stops_the_attempt_and_names_the_file(decider_case):
    tmp_path, base, new, head, ids, with_rows = decider_case
    stage = stage_for(tmp_path, new, base, model_id="perplexity-ai/pplx-decider-v1.2")
    (stage / "evidence" / "readout-facts.json").mkdir(parents=True)        # a directory cannot be unlinked as a file
    (stage / "evidence" / "readout-facts.json" / "x").write_text("x")
    r = run_triage(stage)
    assert r.returncode != 0 and "readout-facts.json" in r.stderr
    assert not (stage / "delta.json").exists()


def test_every_evidence_name_the_template_writes_is_in_its_list_of_owned_files():
    # a new conditional file added without its name in the list would bring the problem back
    src = TEMPLATE.read_text(encoding="utf-8")
    written = set(re.findall(r'write_json\("([^"]+)"', src))
    ns: dict = {}
    m = re.search(r"EVIDENCE_FILES = \((.*?)\)\n", src, re.S)
    assert m, "the template has no EVIDENCE_FILES list"
    listed = set(re.findall(r'"([^"]+)"', m.group(1)))
    # gguf-header.json is removed by its own step with its own soft failure (stage 0 then says the run has no GGUF evidence; PR 441), so it
    # is deliberately not in the hard-failing list. The template must still show that removal.
    self_cleared = {"gguf-header.json"} if 'stale = EVIDENCE / "gguf-header.json"' in src else set()
    assert written <= listed | self_cleared, sorted(written - listed - self_cleared)
