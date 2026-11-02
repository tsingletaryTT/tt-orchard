# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fakes for the scoring-model tests (issue 358): a word-level tokenizer that knows every word of the fixed pair set, and a tiny random
Qwen3 snapshot built in the reference interpreter (torch and transformers; the tests skip without ORCHARD_REFERENCE_PYTHON)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import scoring_reference as sr

REPO = Path(__file__).resolve().parent.parent
SPECIALS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def build_tokenizer():
    """A `tokenizers.Tokenizer`: one id per distinct word of the prompt texts (split on whitespace and punctuation), the four chat
    markers as special tokens, and `yes` and `no` in the vocabulary."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    pre = pre_tokenizers.Whitespace()
    words = {"yes", "no", "system", "user", "assistant"}
    texts = [sr.PREFIX, sr.SUFFIX, sr.INSTRUCTION, "<Instruct>", "<Query>", "<Document>"]
    for row in sr.pair_set():
        texts += [row["query"], row["document"]]
    for t in texts:
        for w, _ in pre.pre_tokenize_str(t):
            words.add(w)
    vocab = {"<unk>": 0}
    for w in sorted(words):
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre
    tok.add_special_tokens(SPECIALS)
    return tok


def save_tokenizer(path: Path) -> None:
    build_tokenizer().save(str(path))


# Run in the reference interpreter: a random Qwen3 (2 layers, tied embeddings) next to the tokenizer the caller saved.
TINY_MODEL_SCRIPT = '''
import sys, torch
from transformers import Qwen3Config, Qwen3ForCausalLM
out, vocab = sys.argv[1], int(sys.argv[2])
torch.manual_seed(7)
cfg = Qwen3Config(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=8, max_position_embeddings=16384, tie_word_embeddings=True)
Qwen3ForCausalLM(cfg).save_pretrained(out)
'''

# Run in the reference interpreter: the score by the plain route (the whole logits matrix), for comparison with the recorder's.
PLAIN_SCORE_SCRIPT = '''
import json, sys, torch
from transformers import AutoModelForCausalLM
model_dir, pairs_file, yes, no = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
dtype = torch.bfloat16 if len(sys.argv) > 5 and sys.argv[5] == "bf16" else torch.float32
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).eval()
pairs = json.load(open(pairs_file))["pairs"]
out = []
with torch.no_grad():
    for p in pairs:
        last = model(input_ids=torch.tensor([p["input_ids"]])).logits[0, -1]
        out.append(float(last[yes]) - float(last[no]))          # two logits as the model rounds them, then the difference
print(json.dumps(out))
'''


def reference_python() -> str | None:
    """The interpreter with torch and transformers, or None. Needs a Qwen3 class, which is what the recorder loads."""
    cand = os.environ.get("ORCHARD_REFERENCE_PYTHON")
    if not cand or not os.path.isfile(cand):
        return None
    probe = subprocess.run([cand, "-I", "-c", "import torch; from transformers import Qwen3Config, Qwen3ForCausalLM"],
                           capture_output=True, timeout=120)
    return cand if probe.returncode == 0 else None


def needs_reference():
    py = reference_python()
    if py is None:
        pytest.skip("needs ORCHARD_REFERENCE_PYTHON: an interpreter with torch and a transformers that has Qwen3ForCausalLM")
    return py


def make_tiny_snapshot(py: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer()
    tok.save(str(dest / "tokenizer.json"))
    done = subprocess.run([py, "-I", "-c", TINY_MODEL_SCRIPT, str(dest), str(tok.get_vocab_size())], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    return dest


def repo_env() -> dict:
    return {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(REPO), "HOME": os.environ.get("HOME", "/"), "HF_HUB_OFFLINE": "1"}
