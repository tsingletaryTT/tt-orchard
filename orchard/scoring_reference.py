# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 1 for a scoring model (issue 358): record a fixed pair set and the Hugging Face scores for it, on the CPU.

A reranker or an embedding model produces scores or vectors and no tokens, so the stage 1 record that other models keep (generated token
ids) does not apply. This module writes the record that the stage 2 scoring check (skills/scoring-check-templates/scoring_check.py and
orchard/scoring_check.py) compares a port against. The first slice covers one model class: a Qwen3-style yes/no reranker.

    python3 -m orchard.scoring_reference rerank --model-dir SNAPSHOT --out DIR [--dtype fp32|bf16] [--max-length 8192]

Run it in an interpreter that has torch and transformers (ORCHARD_REFERENCE_PYTHON). It writes two files into DIR:

    scoring-pairs.json       the fixed pair set: for each pair its group, its kind (short, multilingual, code, long), the query, the
                             document and the token ids of the whole prompt. The port is given only the ids.
    scoring-reference.json   the Hugging Face score for each pair (same order), the precision, the yes and no token ids, and the sha256
                             of scoring-pairs.json. The check refuses a reference whose pair file has changed.

The prompt and the score are the Qwen3-Reranker model card's (huggingface.co/Qwen/Qwen3-Reranker-4B, "Transformers Usage"), which the
public bundle raahemnabeel/qwen3-reranker-4b follows token for token (its inputs.py and reference.py):

    prefix  <|im_start|>system\\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note
            that the answer can only be "yes" or "no".<|im_end|>\\n<|im_start|>user\\n
    body    <Instruct>: {instruction}\\n<Query>: {query}\\n<Document>: {document}      (cut at the right end to fit the length limit)
    suffix  <|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
    score   logit("yes") - logit("no") at the last position of the prefix + body + suffix

The card tokenizes with `truncation='longest_first'` on one sequence, which cuts the end of the body; this module does the same with the
`tokenizers` library alone (the card's tokenizer adds no special tokens). The card turns the two logits into a probability for display;
the check compares the raw difference, as the bundle's accuracy table does.

What this does not show: it is the reference, so nothing here proves a port right. The Hugging Face model runs in fp32 by default, one
pair at a time with no padding, so the numbers do not depend on a batch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from orchard import safeio

PAIRS_SCHEMA = "orchard-scoring-pairs/1"
REFERENCE_SCHEMA = "orchard-scoring-reference/1"
PAIRS_FILE = "scoring-pairs.json"
REFERENCE_FILE = "scoring-reference.json"
PAIR_SET_VERSION = "scoring-pairs/1"        # bump when a text below changes: a record names the version it was made with
MODEL_KIND = "reranker"
MAX_LENGTH = 8192                           # the model card's max_length, prompt included
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
          'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
SOURCES = ("Qwen/Qwen3-Reranker-4B model card, Transformers Usage",
           "raahemnabeel/qwen3-reranker-4b: code/models/autoports/qwen_qwen3_reranker_4b/inputs.py and reference.py")
DTYPES = ("fp32", "bf16")
FILE_MAX = 64 * 1024 * 1024                 # choice: a pair file holds a few dozen prompts; this leaves room for long documents
LONG_REPEATS = 40                           # choice: about 1,000 tokens of filler around the answer; the 4B bundle's set goes to 8K

KINDS = ("short", "multilingual", "code", "long")

_FILLER = ("The committee reviewed the quarterly logistics schedule and noted that regional warehouses reported steady throughput. "
           "Maintenance windows were moved to weekends, and the procurement office asked for revised estimates. ")


def _long(sentence: str) -> str:
    """A long document: the sentence that matters sits in the middle of repeated filler that answers nothing."""
    half = _FILLER * (LONG_REPEATS // 2)
    return half + sentence + " " + half


# (group name, kind, query, [(document, is_relevant)]). Every group has one relevant document and three that are not.
_GROUPS = [
    ("short-capital", "short", "What is the capital of France?",
     [("Paris is the capital and largest city of France.", True),
      ("Berlin is the capital of Germany and sits on the river Spree.", False),
      ("France is known for wine, cheese and the Eiffel Tower.", False),
      ("The capital gains tax applies to profit from selling an asset.", False)]),
    ("short-boil", "short", "At what temperature does water boil at sea level?",
     [("At sea level, water boils at 100 degrees Celsius (212 degrees Fahrenheit).", True),
      ("Water freezes at 0 degrees Celsius at standard pressure.", False),
      ("A boiler heats water for radiators in a house.", False),
      ("Sea level is the average height of the ocean's surface.", False)]),
    ("multilingual-zh", "multilingual", "世界上最高的山峰是什么？",
     [("珠穆朗玛峰是世界上最高的山峰，海拔约八千八百四十八米。", True),
      ("尼罗河是非洲最长的河流之一。", False),
      ("长城是中国古代的军事防御工程。", False),
      ("太平洋是地球上面积最大的海洋。", False)]),
    ("multilingual-de", "multilingual", "Wer hat die Relativitätstheorie entwickelt?",
     [("Albert Einstein entwickelte die spezielle und die allgemeine Relativitätstheorie.", True),
      ("Isaac Newton beschrieb die Gesetze der Bewegung und der Gravitation.", False),
      ("Die Relativität von Preisen hängt vom Markt ab.", False),
      ("Berlin ist die Hauptstadt von Deutschland.", False)]),
    ("code-reverse", "code", "How do I reverse a list in Python?",
     [("Use `items.reverse()` to reverse a list in place, or `items[::-1]` to get a reversed copy.", True),
      ("Use `sorted(items)` to return a new list in ascending order.", False),
      ("In Java, `Collections.reverse(list)` reverses a List in place.", False),
      ("Python lists are mutable sequences that can hold values of different types.", False)]),
    ("code-http", "code", "def fetch(url): return requests.get(url, timeout=5).json()  # what does this function do",
     [("It sends an HTTP GET request to the url with a five second timeout and returns the parsed JSON body.", True),
      ("It sorts the rows of a database table by the url column.", False),
      ("It deletes the file at the given path.", False),
      ("A timeout is the longest time a program waits for an answer.", False)]),
    ("long-photosynthesis", "long", "Which gas do plants take in during photosynthesis?",
     [(_long("During photosynthesis, plants take in carbon dioxide and release oxygen."), True),
      (_long("During cellular respiration, animals take in oxygen and release carbon dioxide."), False),
      (_long("Roads in the district were repaved last spring."), False),
      (_long("The library extended its opening hours on Saturdays."), False)]),
    ("long-treaty", "long", "In which year was the Treaty of Westphalia signed?",
     [(_long("The Treaty of Westphalia was signed in 1648 and ended the Thirty Years' War."), True),
      (_long("The Treaty of Versailles was signed in 1919 and ended the First World War."), False),
      (_long("A westward-facing window gives afternoon light."), False),
      (_long("The council postponed the vote on the new parking rules."), False)]),
]


def pair_set() -> list[dict]:
    """The fixed pair set as (group, kind, query, document, relevant) rows, in a fixed order. No ids yet."""
    rows = []
    for name, kind, query, docs in _GROUPS:
        for doc, relevant in docs:
            rows.append({"group": name, "kind": kind, "query": query, "document": doc, "relevant": relevant})
    return rows


def prompt_ids(tokenizer, query: str, document: str, *, instruction: str = INSTRUCTION, max_length: int = MAX_LENGTH) -> list[int]:
    """The token ids of the whole prompt for one pair, as the model card builds them. `tokenizer` is a `tokenizers.Tokenizer`.
    The body is cut at its right end so that prefix, body and suffix together are at most `max_length` ids."""
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False).ids
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False).ids
    room = max_length - len(prefix) - len(suffix)
    if room < 1:
        raise ValueError(f"max_length {max_length} leaves no room for the prompt around the pair ({len(prefix)} + {len(suffix)} ids)")
    body = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    return prefix + tokenizer.encode(body, add_special_tokens=False).ids[:room] + suffix


def yes_no_ids(tokenizer) -> tuple[int, int]:
    """The ids of the tokens "yes" and "no", looked up as the card does (convert_tokens_to_ids). A vocabulary without either is refused."""
    yes, no = tokenizer.token_to_id("yes"), tokenizer.token_to_id("no")
    if yes is None or no is None or yes == no:
        raise ValueError("the tokenizer has no single token for both \"yes\" and \"no\"")
    return yes, no


def build_pairs(tokenizer, *, max_length: int = MAX_LENGTH) -> dict:
    """The contents of scoring-pairs.json."""
    pairs = []
    for row in pair_set():
        pairs.append({**row, "input_ids": prompt_ids(tokenizer, row["query"], row["document"], max_length=max_length)})
    return {"schema": PAIRS_SCHEMA, "pair_set": PAIR_SET_VERSION, "model_kind": MODEL_KIND, "instruction": INSTRUCTION,
            "max_length": max_length, "prefix": PREFIX, "suffix": SUFFIX, "sources": list(SOURCES), "pairs": pairs}


def encode_json(obj) -> bytes:
    return (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def score_pairs_hf(model_dir: str, pairs: dict, yes_id: int, no_id: int, *, dtype: str = "fp32") -> list[float]:
    """Hugging Face scores, one pair at a time with no padding. Needs torch and transformers (import here, so the rest of this module
    imports without them). The score is the hidden state at the last position, after the final norm, times the difference of the
    yes and no rows of the output matrix: logit(yes) - logit(no) without computing the whole vocabulary."""
    import torch
    from transformers import AutoModelForCausalLM
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32 if dtype == "fp32" else torch.bfloat16).eval()
    w = model.lm_head.weight
    direction = (w[yes_id].float() - w[no_id].float())
    out = []
    with torch.no_grad():
        for p in pairs["pairs"]:
            ids = torch.tensor([p["input_ids"]], dtype=torch.long)
            h = model.model(input_ids=ids).last_hidden_state[0, -1].float()
            out.append(float(h @ direction))
    return out


def record(model_dir: str, out_dir: str, *, dtype: str = "fp32", max_length: int = MAX_LENGTH) -> dict:
    """Write scoring-pairs.json and scoring-reference.json into `out_dir`. Returns the reference record."""
    from tokenizers import Tokenizer
    tok_path = Path(model_dir) / "tokenizer.json"
    try:
        data = safeio.read_bytes(tok_path, max_bytes=FILE_MAX)
    except safeio.SafeReadError as exc:
        raise ValueError(f"{tok_path} cannot be read: {exc.reason}") from exc
    tokenizer = Tokenizer.from_str(data.decode("utf-8"))
    yes_id, no_id = yes_no_ids(tokenizer)
    pairs = build_pairs(tokenizer, max_length=max_length)
    pairs_bytes = encode_json(pairs)
    scores = score_pairs_hf(model_dir, pairs, yes_id, no_id, dtype=dtype)
    import torch
    import transformers
    ref = {"schema": REFERENCE_SCHEMA, "model_kind": MODEL_KIND, "precision": dtype,
           "score": "logit(yes) - logit(no) at the last position", "yes_token_id": yes_id, "no_token_id": no_id,
           "pairs": {"file": PAIRS_FILE, "sha256": hashlib.sha256(pairs_bytes).hexdigest(), "count": len(pairs["pairs"])},
           "scores": scores, "torch": torch.__version__, "transformers": transformers.__version__}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safeio.atomic_write(out / PAIRS_FILE, pairs_bytes)
    safeio.atomic_write(out / REFERENCE_FILE, encode_json(ref))
    return ref


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m orchard.scoring_reference", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rr = sub.add_parser("rerank", help="record the pair set and the Hugging Face scores for a Qwen3-style yes/no reranker")
    rr.add_argument("--model-dir", required=True, help="the Hugging Face snapshot directory (config, weights, tokenizer.json)")
    rr.add_argument("--out", required=True, help="the directory that receives scoring-pairs.json and scoring-reference.json")
    rr.add_argument("--dtype", choices=DTYPES, default="fp32", help="precision of the reference run (default fp32)")
    rr.add_argument("--max-length", type=int, default=MAX_LENGTH, help="prompt length limit in ids (default 8192, the model card's)")
    args = ap.parse_args(argv)
    try:
        ref = record(args.model_dir, args.out, dtype=args.dtype, max_length=args.max_length)
    except (ValueError, OSError) as exc:
        print(f"scoring reference: {exc}", file=sys.stderr)
        return 2
    print(f"scoring reference: {ref['pairs']['count']} pairs, {ref['precision']}, written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
