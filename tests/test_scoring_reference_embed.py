# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 1 recorder for an embedding model (orchard/scoring_reference.py `embed`, issue 376). The input-set tests need no torch. The
end-to-end tests build a tiny random Qwen3 in the reference interpreter (ORCHARD_REFERENCE_PYTHON) and skip without it."""
import hashlib
import json
import math
import subprocess

import pytest

from orchard import scoring_check as sc
from orchard import scoring_reference as sr
from scoring_fakes import build_tokenizer, make_tiny_snapshot, needs_reference, repo_env

EMBED_PLAIN = '''
import json, sys, torch
from transformers import AutoModelForCausalLM
model_dir, pairs_file = sys.argv[1], sys.argv[2]
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
out = []
with torch.no_grad():
    for p in json.load(open(pairs_file))["pairs"]:
        h = model(input_ids=torch.tensor([p["input_ids"]]), output_hidden_states=True).hidden_states[-1][0, -1]   # the final-normed state
        out.append((h / h.norm()).tolist())
print(json.dumps(out))
'''


def test_the_embedding_set_has_six_inputs_of_each_of_four_kinds_and_queries_carry_the_instruction():
    rows = sr.embed_set()
    assert len(rows) == 24 and [r["kind"] for r in rows] == [k for k in sr.EMBED_KINDS for _ in range(6)]
    assert len({r["id"] for r in rows}) == 24
    for r in rows:
        assert r["text"].startswith("Instruct: ") == (r["role"] == "query")
        if r["role"] == "query":
            assert r["text"].startswith(f"Instruct: {sr.EMBED_INSTRUCTION}\nQuery:")
    assert sum(1 for r in rows if r["role"] == "query") == 8


def test_the_embedding_set_meets_the_gates_minimums():
    rows = sr.embed_set()
    assert len(rows) >= sc.EMBEDDING_BOUNDS["min_inputs"] and len({r["kind"] for r in rows}) >= sc.EMBEDDING_BOUNDS["min_kinds"]


def test_the_ids_are_the_tokenizers_own_with_its_special_tokens_cut_at_the_right_end():
    tok = build_tokenizer()
    text = "What is the capital of France?"
    assert sr.embed_ids(tok, text) == tok.encode(text, add_special_tokens=True).ids
    assert sr.embed_ids(tok, text, max_length=3) == tok.encode(text, add_special_tokens=True).ids[:3]
    with pytest.raises(ValueError, match="at least 1"):
        sr.embed_ids(tok, text, max_length=0)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """One real recording of a tiny random Qwen3 (a forward over 24 inputs, about 1,000 tokens for the long ones): shared by the tests below."""
    py = needs_reference()
    tmp_path = tmp_path_factory.mktemp("embed")
    snap = make_tiny_snapshot(py, tmp_path / "snap")
    out = tmp_path / "ref"
    done = subprocess.run([py, "-s", "-m", "orchard.scoring_reference", "embed", "--model-dir", str(snap), "--out", str(out)],
                          capture_output=True, text=True, timeout=900, env=repo_env(), cwd=str(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    return py, snap, out


def test_the_recorder_writes_unit_vectors_equal_to_the_last_final_normed_hidden_state(recorded):
    py, snap, out = recorded
    ref = json.loads((out / "scoring-reference.json").read_text())
    pairs_bytes = (out / "scoring-pairs.json").read_bytes()
    pairs = json.loads(pairs_bytes)
    assert ref["model_kind"] == "embedding" and pairs["model_kind"] == "embedding" and ref["precision"] == "fp32"
    assert ref["pairs"] == {"file": "scoring-pairs.json", "sha256": hashlib.sha256(pairs_bytes).hexdigest(), "count": 24}
    assert len(ref["vectors"]) == 24 and ref["dim"] == len(ref["vectors"][0]) == 32
    for v in ref["vectors"]:
        assert math.sqrt(sum(x * x for x in v)) == pytest.approx(1.0, abs=1e-5)
    plain = subprocess.run([py, "-I", "-c", EMBED_PLAIN, str(snap), str(out / "scoring-pairs.json")], capture_output=True, text=True, timeout=600)
    assert plain.returncode == 0, plain.stderr
    expected = json.loads(plain.stdout)
    assert max(abs(a - b) for va, vb in zip(ref["vectors"], expected) for a, b in zip(va, vb)) < 1e-5
    assert len({tuple(round(x, 5) for x in v) for v in ref["vectors"]}) > 12          # the vectors are not one constant


def test_the_recorded_files_have_the_shapes_the_gate_and_the_template_read(recorded):
    py, snap, out = recorded
    ref = json.loads((out / "scoring-reference.json").read_text())
    pairs = json.loads((out / "scoring-pairs.json").read_text())
    assert [r["kind"] for r in pairs["pairs"]] == [k for k in sr.EMBED_KINDS for _ in range(6)]
    assert all(isinstance(r["input_ids"], list) and r["input_ids"] and all(type(t) is int for t in r["input_ids"]) for r in pairs["pairs"])
    assert len(ref["vectors"]) * ref["dim"] <= sc.MAX_VECTOR_NUMEL and ref["dim"] <= sc.MAX_DIM


def test_stage_1s_gate_accepts_the_recorded_files_and_the_check_reads_them(recorded, tmp_path):
    # The recorded files, copied into a run directory, pass stage 1's gate as an embedding set, and fail it as a reranker set.
    py, snap, out = recorded
    run = tmp_path / "run"
    dest = run / "stages" / "1" / "evidence" / "reference"
    dest.mkdir(parents=True)
    for name in (sr.PAIRS_FILE, sr.REFERENCE_FILE):
        (dest / name).write_bytes((out / name).read_bytes())
    reasons, seen = sc.reference_reasons(run, "embedding")
    assert reasons == [] and seen == [sc.REFERENCE_FILE, sc.PAIRS_FILE]
    assert any("not 'reranker'" in r for r in sc.reference_reasons(run, "reranker")[0])


def test_a_snapshot_without_a_tokenizer_json_is_refused_with_a_message(tmp_path):
    py = needs_reference()
    (tmp_path / "empty").mkdir()
    done = subprocess.run([py, "-s", "-m", "orchard.scoring_reference", "embed", "--model-dir", str(tmp_path / "empty"), "--out", str(tmp_path / "o")],
                          capture_output=True, text=True, timeout=120, env=repo_env(), cwd=str(tmp_path))
    assert done.returncode == 2 and "tokenizer.json" in done.stderr and not (tmp_path / "o").exists()
