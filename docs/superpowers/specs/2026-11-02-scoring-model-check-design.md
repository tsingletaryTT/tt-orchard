# Scoring models: a stage 2 check on score parity

Date: 2026-11-02. Issue 358. Status: the first slice is built (a stage 1 recorder for a Qwen3-style reranker, a supervisor-owned check
template, and a gate). It runs against fakes on the CPU. Nothing here ran on a chip, and nothing is wired into the stage machine yet.

## 1. The problem

Stage 2 of a bring-up compares generated tokens: top-1 agreement between the chip and a Hugging Face reference on the same ids. A
reranker or an embedding model produces scores or vectors and no tokens, so that check has nothing to compare. Three bundles that
raahemnabeel published on 2026-11-02 (`qwen3-reranker-4b`, `qwen3-vl-reranker-8b`, `qwen3-vl-embedding-8b`, one chip each) show what a
check for them looks like. Each reports its own parity table against a Hugging Face run on exactly the same token ids. This note takes
the measurements those tables make and puts them under the supervisor, with the same protections as the full-port check (the coder
writes the model code, the supervisor writes the check and the record).

## 2. The model kind

A model has a kind: `generative` (every model before this note), `reranker` or `embedding`. The kind decides which stage 1 reference is
recorded and which stage 2 check runs. This slice builds `reranker` only. The check template refuses `embedding` with a message.

How the kind is chosen is not built. The proposal is that stage 0 reads it from the model card (`pipeline_tag` of `text-ranking`,
`feature-extraction` or `sentence-similarity`) and the architecture name, writes `model_kind` into `delta.json`, and the operator can
override it with a flag. The kind is a fact about how the model is used. It is not an outcome class (`classes.py`), because a reranker
with the same tensors as its base model is `weights-only` and one with a different encoder is `full-port`. The two choices are
independent.

## 3. What stage 0 does for a scoring model

Stage 0 is unchanged in this slice. What it would do, from reading `hazards.py`, `nearest.py` and `preflight.py`:

- Nearest bundle. Today stage 0 names the closest installed bundle and compares tensors and config with it. The 4B reranker is a
  Qwen3 dense decoder with tied embeddings, so a Qwen3 text bundle of similar size is its nearest bundle, and the tensor comparison
  finds no difference. Class `weights-only`: the swap path loads the new weights into that bundle. What differs is how the output is
  read, and no weights-only stage knows that. The bundles solve this with a separate serving stack (a `/v1/rerank` API over a two-row
  head). A scoring model therefore needs the check in this note and a serving route that stage 5 does not have yet (section 8).
- The embedding model has no `lm_head` and no tied output head to compare. Stage 0 would list the nearest bundle's `lm_head` under "only
  in the nearest model". How that should classify is an open question (section 8). The 8B VL embedding bundle is a Qwen3-VL with its text
  tower read at the last position after the final norm, so the nearest bundle would be a Qwen3-VL one.
- Hazard scan. The scan predicts failures from the config alone. Its vocabulary-per-die hazard (65,536 columns per die) concerns a full
  output head, and a reranker reads two rows of it. The scan keeps running and a hit on that row needs a note for this kind: "reads two
  rows, so the sampler limit does not apply". The tile and mixer checks apply as they are, because the backbone is the same.

## 4. Stage 1: the reference

A generative model records generated ids. A reranker records the scores a Hugging Face run gives to a fixed pair set.
`orchard/scoring_reference.py` writes two files under `stages/1/evidence/reference/`:

- `scoring-pairs.json`: the pair set. Each pair has a group, a kind (`short`, `multilingual`, `code`, `long`), the query, the document and
  the token ids of the whole prompt. There are 8 groups of 4 documents, one relevant each, so 32 pairs. The long documents are about
  1,000 tokens of filler around one sentence. The 4B bundle's own set is larger (188 pairs, documents up to 8K tokens). The set is
  versioned (`scoring-pairs/1`), because the bounds in section 6 are judged on it.
- `scoring-reference.json`: one score per pair in the same order, the precision (fp32 by default, bf16 on request), the yes and no token
  ids, the library versions, and the sha256 of the pair file. A reference that names a different pair file is refused.

The prompt and the score are the Qwen3-Reranker model card's. The prefix is the system turn and the start of the user turn. The body is
`<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}`, cut at its right end so the prompt fits 8192 ids. The suffix ends the
user turn and opens an assistant turn with an empty think block. The score is logit("yes") minus logit("no") at the last position. The
recorder follows the card's Transformers example and the bundle's `inputs.py` and `reference.py`, which say they match the card token for
token. It looks the two ids up with the tokenizer, as the card does, and the test that builds a tiny random Qwen3 compares the recorded
scores with the same difference taken from the full logits matrix.

Both the Hugging Face run and the port are given the same ids. The reference does not use a batch or padding, so a score does not depend
on what else was in the batch. The port is also called with a group at a time, and its scores must still match.

## 5. Stage 2: the check

The shape follows the full-port check (`docs/superpowers/specs/2026-11-02-full-port-supervisor-design.md`, section 3.3).

The supervisor copies `orchard/skills/scoring-check-templates/scoring_check.py` to `stages/2/scoring_check.py`, drafts
`stages/2/scoring_check_config.json`, and runs `python3 -P -s stages/2/scoring_check.py` under the lease. The coder writes one file,
`stages/2/port/port_model.py`, with this interface:

| Function | Used for | In this slice |
|---|---|---|
| `load(model_dir, mesh_device) -> model` | all kinds | called, timed as `load_s` |
| `score_pairs(model, pairs: list[list[int]]) -> list[float]` | `reranker` | called, one group of prompts at a time |
| `embed(model, ids: list[int]) -> list[float]` | `embedding` | documented, not called |
| `close(model) -> None` | all kinds | called |

The template runs the port in a second process and takes only a list of finite numbers per request. The parent holds the pair set, the
reference, the numbers and the record. A port that prints a mark line, writes the record, replaces the template, patches functions or
installs an import hook cannot reach the parent. When the run ends the parent freezes the worker (which stops every thread and every exit
callback in it), freezes and kills every process below the parent (the parent is a Linux subreaper, so a process whose own parent exited is still found), kills the worker, and only then writes the record through a temporary file and
a rename. The parent hashes the port's source files before the worker starts and fails the run if they differ afterwards. It refuses a
port directory that holds a link, bytecode (`__pycache__`, `.pyc`, `.pyo`), a compiled module (`.so`, `.pyd`) or a `.pth` file, because those
can be imported without being in the manifest. The worker runs with `-B` so it writes no bytecode.

The numbers, computed by the parent and recomputed by the gate from the scores in the record:

| Number | Meaning |
|---|---|
| Pearson r | of the chip score against the reference score over all pairs (0.0 when either is constant) |
| mean and max absolute difference | in logits |
| top-1 groups | groups where the chip and the reference pick the same top document (ties go to the earlier document) |
| order kept | share of document pairs within a group that keep the reference's order, among pairs the reference separates by at least 0.5 |

These are the four measurements the 4B and 8B VL reranker bundles report.

The gate (`orchard/scoring_check.py`) refuses: a test that did not exit 0 or ran another command; an output file that is not the one the
supervisor recorded or has other than one mark line that matches the record; a template copy that is not the shipped file; a record
without the shipped template's sha256; a config that is not the one the record names; a record whose pair file, reference file, scores or
groups differ from the files on disk; metrics that differ from the gate's own recomputation; numbers outside the bounds; chips other than
the leased ones; port files that changed, were added, are links, do not mention `ttnn`, or refer to the reference files.

## 6. Bounds

| Bound | Default | Evidence |
|---|---|---|
| Pearson r at least | 0.998 | 4B selected policy 0.9997; 8B VL 0.9991; the 4B bundle's LoFi policy, which it did not select, 0.99885 |
| mean absolute difference at most | 0.25 logits | 4B selected 0.140; 8B VL 0.107; bfp8 policy 0.217 |
| max absolute difference at most | 1.0 logits | 4B selected 0.64; 8B VL 0.39; LoFi policy 0.93 |
| groups with the same top document at least | 0.95 | both bundles 1.0; bfp8 policy 31 of 32 |
| order kept at least | 0.98 | both bundles 1.0 |
| pairs, groups, separated pairs at least | 24, 6, 10 | the shipped set has 32 pairs and 8 groups |

The defaults are `DEFAULT_BOUNDS` in `orchard/scoring_check.py`. The supervisor copies them into `scoring_check_config.json` under
`bounds`. The gate applies the default for each key and replaces it with the config's value only when the config is stricter. A looser
value, a value that is not a finite number and an unknown key are each a reason. So a coder who edits the config cannot loosen a bound,
and an operator who wants a stricter bar for one model can set it there. Letting an operator loosen a bound for one model (for example a
bfp8 weight policy that costs more score error) needs a ledger entry and a card line that says so. That is not built.

These numbers are choices made from the public tables. They have not been compared with a run through this supervisor on a chip. The
bundles' absolute differences depend on the score scale of their model, and a model with a different scale needs its own review of the
two absolute bounds.

## 7. What this does not show

The same limits as the full-port check. The supervisor ran a test and the record is tied to it, but a port that returns the reference's
scores without computing them cannot be told from one that computed them. The source checks make the plain forms visible. A person who
wants the proof replays the command on a chip.

Parity with Hugging Face on 32 pairs is not a quality measure. The bundles also report retrieval benchmarks (NDCG@10 on subsets of MTEB
reranking and MIRACL) as parity rows, and this slice has none.

## 8. Out of scope for this slice (filed as issues)

1. (issue 375) A real run on a board: the 4B reranker port under `--full-port` or the weights-swap path, with the recorder run on the real snapshot.
2. (issue 376) Embedding models: `embed` in the template, cosine to the Hugging Face vector (mean and minimum) per input kind, bounds, and the
   recorder for an embedding model (the last-token hidden state after the final norm, L2 normalised).
3. (issue 377) Vision pairs and video inputs: the pair set carries ids only, and a vision-language reranker or embedder needs pixel inputs.
4. (issue 378) Stage machine wiring: the `model_kind` field in stage 0, the route to this check, a finish step and `result.json`, and a stage 5
   serving check for a `/v1/rerank` style API.
5. (issue 379) Retrieval benchmark parity rows (NDCG, Hit@1) as an optional second stage 2 record.
6. (issue 380) How stage 0 classifies a model with no `lm_head`.
