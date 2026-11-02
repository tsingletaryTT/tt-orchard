# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 1 recorder for a scoring model (orchard/scoring_reference.py, issue 358). The prompt and pair-set tests need no torch. The
end-to-end tests build a tiny random Qwen3 in the reference interpreter (ORCHARD_REFERENCE_PYTHON) and skip without it."""
import hashlib
import json
import subprocess
import sys

import pytest

from orchard import scoring_reference as sr
from scoring_fakes import PLAIN_SCORE_SCRIPT, build_tokenizer, make_tiny_snapshot, needs_reference, repo_env


@pytest.fixture(scope="module")
def tok():
    return build_tokenizer()


# ---- the prompt (no torch) ---------------------------------------------------------------------------------

def test_the_prompt_is_the_card_prefix_then_the_body_then_the_card_suffix(tok):
    ids = sr.prompt_ids(tok, "Who?", "Me.")
    prefix = tok.encode(sr.PREFIX, add_special_tokens=False).ids
    suffix = tok.encode(sr.SUFFIX, add_special_tokens=False).ids
    assert ids[:len(prefix)] == prefix and ids[-len(suffix):] == suffix
    body = tok.encode("<Instruct>: " + sr.INSTRUCTION + "\n<Query>: Who?\n<Document>: Me.", add_special_tokens=False).ids
    assert ids == prefix + body + suffix
    im_start = tok.token_to_id("<|im_start|>")
    assert ids[0] == im_start and ids.count(im_start) == 3          # system turn, user turn, assistant turn


def test_the_instruction_is_the_card_default_and_a_caller_can_change_it(tok):
    assert sr.prompt_ids(tok, "q", "d") == sr.prompt_ids(tok, "q", "d", instruction=sr.INSTRUCTION)
    assert sr.prompt_ids(tok, "q", "d") != sr.prompt_ids(tok, "q", "d", instruction="Find the Document that is about cooking")


@pytest.mark.parametrize("limit", [60, 80, 120])
def test_a_long_pair_is_cut_at_the_end_of_the_body_and_keeps_its_prefix_and_suffix(tok, limit):
    long_doc = "capital of France " * 400
    ids = sr.prompt_ids(tok, "What is the capital of France?", long_doc, max_length=limit)
    assert len(ids) == limit
    full = sr.prompt_ids(tok, "What is the capital of France?", long_doc, max_length=100000)
    suffix = tok.encode(sr.SUFFIX, add_special_tokens=False).ids
    assert ids[-len(suffix):] == suffix and ids[:limit - len(suffix)] == full[:limit - len(suffix)]


@pytest.mark.parametrize("limit", [0, -5, 10])
def test_a_limit_with_no_room_for_the_pair_is_refused(tok, limit):
    with pytest.raises(ValueError, match="leaves no room"):
        sr.prompt_ids(tok, "q", "d", max_length=limit)


def test_the_limit_that_leaves_exactly_one_id_for_the_body_is_accepted(tok):
    n = len(tok.encode(sr.PREFIX, add_special_tokens=False).ids) + len(tok.encode(sr.SUFFIX, add_special_tokens=False).ids)
    assert len(sr.prompt_ids(tok, "q", "d", max_length=n + 1)) == n + 1
    with pytest.raises(ValueError):
        sr.prompt_ids(tok, "q", "d", max_length=n)


def test_yes_and_no_are_looked_up_in_the_vocabulary_and_a_vocabulary_without_them_is_refused(tok):
    yes, no = sr.yes_no_ids(tok)
    assert (yes, no) == (tok.token_to_id("yes"), tok.token_to_id("no")) and yes != no
    from tokenizers import Tokenizer, models
    bare = Tokenizer(models.WordLevel({"<unk>": 0, "yes": 1}, unk_token="<unk>"))
    with pytest.raises(ValueError, match="yes.*no"):
        sr.yes_no_ids(bare)


# ---- the fixed pair set (no torch) -------------------------------------------------------------------------

def test_the_pair_set_has_every_kind_and_one_relevant_document_per_group():
    rows = sr.pair_set()
    assert {r["kind"] for r in rows} == set(sr.KINDS)
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    assert len(groups) >= 6 and all(len(g) == 4 and sum(r["relevant"] for r in g) == 1 for g in groups.values())
    assert all(len({r["kind"] for r in g}) == 1 for g in groups.values())          # a group is one kind
    assert any(not r["document"].isascii() for r in rows) and any("def " in r["query"] for r in rows)
    long_rows = [r for r in rows if r["kind"] == "long"]
    assert all(len(r["document"]) > 4000 for r in long_rows)


def test_the_pair_set_is_the_same_on_every_call(tok):
    a = sr.encode_json(sr.build_pairs(tok))
    assert a == sr.encode_json(sr.build_pairs(tok))
    assert sr.build_pairs(tok)["pair_set"] == sr.PAIR_SET_VERSION


def test_every_pair_carries_its_ids_and_the_ids_are_the_prompt(tok):
    pairs = sr.build_pairs(tok)["pairs"]
    for p in pairs:
        assert p["input_ids"] == sr.prompt_ids(tok, p["query"], p["document"]) and p["input_ids"]
    assert len({tuple(p["input_ids"]) for p in pairs}) == len(pairs)           # no two pairs became the same prompt


# ---- scores from a tiny random model (reference interpreter) ----------------------------------------------

def _record(py, tmp_path, *extra):
    snap = make_tiny_snapshot(py, tmp_path / "snap")
    out = tmp_path / "ref"
    done = subprocess.run([py, "-s", "-m", "orchard.scoring_reference", "rerank", "--model-dir", str(snap), "--out", str(out), *extra],
                          capture_output=True, text=True, timeout=600, env=repo_env(), cwd=str(tmp_path))
    return snap, out, done


def test_the_recorder_writes_scores_equal_to_yes_minus_no_of_the_full_logits(tmp_path):
    py = needs_reference()
    snap, out, done = _record(py, tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    ref = json.loads((out / "scoring-reference.json").read_text())
    pairs_bytes = (out / "scoring-pairs.json").read_bytes()
    assert ref["schema"] == sr.REFERENCE_SCHEMA and ref["precision"] == "fp32" and ref["model_kind"] == "reranker"
    assert ref["pairs"] == {"file": "scoring-pairs.json", "sha256": hashlib.sha256(pairs_bytes).hexdigest(), "count": len(ref["scores"])}
    pairs = json.loads(pairs_bytes)
    assert len(ref["scores"]) == len(pairs["pairs"]) == len(sr.pair_set())
    plain = subprocess.run([py, "-I", "-c", PLAIN_SCORE_SCRIPT, str(snap), str(out / "scoring-pairs.json"),
                            str(ref["yes_token_id"]), str(ref["no_token_id"])], capture_output=True, text=True, timeout=600)
    assert plain.returncode == 0, plain.stderr
    expected = json.loads(plain.stdout)
    assert max(abs(a - b) for a, b in zip(ref["scores"], expected)) < 1e-4
    assert len({round(s, 6) for s in ref["scores"]}) > len(ref["scores"]) // 2          # the scores are not one constant


def test_the_recorded_pair_file_is_the_one_the_reference_names_and_a_second_run_gives_the_same_files(tmp_path):
    py = needs_reference()
    snap, out, done = _record(py, tmp_path)
    assert done.returncode == 0, done.stderr
    first = {f: (out / f).read_bytes() for f in (sr.PAIRS_FILE, sr.REFERENCE_FILE)}
    second = subprocess.run([py, "-s", "-m", "orchard.scoring_reference", "rerank", "--model-dir", str(snap), "--out", str(out)],
                            capture_output=True, text=True, timeout=600, env=repo_env(), cwd=str(tmp_path))
    assert second.returncode == 0, second.stderr
    assert {f: (out / f).read_bytes() for f in first} == first


def test_a_snapshot_without_a_tokenizer_json_is_refused_with_a_message(tmp_path):
    py = needs_reference()
    (tmp_path / "empty").mkdir()
    done = subprocess.run([py, "-s", "-m", "orchard.scoring_reference", "rerank", "--model-dir", str(tmp_path / "empty"), "--out", str(tmp_path / "o")],
                          capture_output=True, text=True, timeout=120, env=repo_env(), cwd=str(tmp_path))
    assert done.returncode == 2 and "tokenizer.json" in done.stderr and not (tmp_path / "o").exists()


def test_the_command_line_names_its_flags():
    done = subprocess.run([sys.executable, "-m", "orchard.scoring_reference", "rerank", "--help"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and "--model-dir" in done.stdout and "--dtype" in done.stdout and "--max-length" in done.stdout
