# Scoring models: a stage 2 check on score parity

Date: 2026-11-02, updated 2026-11-03. Issue 358. Status: built and run against fakes on the CPU only. Nothing here ran on a chip.
Slice 1 (PR 383): a stage 1 recorder for a Qwen3-style reranker, a supervisor-owned check template, and a gate. Slice 2 (issues 376 to 380):
the embedding kind, the model kind in stage 0 and the stage machine, the stage 0 rule for a model with no output head, retrieval rows
(NDCG and Hit@1) computed from the recorded scores, and media (image and video files) in the pair set. Section 8 lists what is left.

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

How the kind is chosen (`orchard/model_kind.py`, issue 378). The supervisor decides it when stage 0 passes, in this order: the operator's
`--model-kind`; the model card's `pipeline_tag` (`text-ranking` is a reranker, `feature-extraction` and `sentence-similarity` are embedding
models); an architecture name that contains `Embedding`; otherwise `generative`. The answer and its source go into stage 0's passing
`stage_end` (`model_kind`), where a resume reads it, and into `hazard-scan.json` (`facts.model_kind`). It is not written into `delta.json`,
which the triage template writes and an agent can edit. A bare backbone architecture alone does not make a model an embedding model,
because vision and audio encoders are also bare backbones. Qwen3-Reranker names `Qwen3ForCausalLM`, so only the card or the operator can
say it is a reranker. The kind is a fact about how the model is used. It is not an outcome class (`classes.py`), because a reranker
with the same tensors as its base model is `weights-only` and one with a different encoder is `full-port`. The two choices are
independent.

## 3. What stage 0 does for a scoring model

The notes below were written from reading `hazards.py`, `nearest.py` and `preflight.py` before slice 2. Slice 2 builds the two changes marked
"built".

- Nearest bundle. Today stage 0 names the closest installed bundle and compares tensors and config with it. The 4B reranker is a
  Qwen3 dense decoder with tied embeddings, so a Qwen3 text bundle of similar size is its nearest bundle, and the tensor comparison
  finds no difference. Class `weights-only`: the swap path loads the new weights into that bundle. What differs is how the output is
  read, and no weights-only stage knows that. The bundles solve this with a separate serving stack (a `/v1/rerank` API over a two-row
  head). A scoring model therefore needs the check in this note and a serving route that stage 5 does not have yet (section 8).
- The embedding model has no `lm_head` and no tied output head to compare. Before slice 2 stage 0 listed the nearest bundle's `lm_head`
  under "only in the nearest model", which made the class `full-port`. Built (issue 380): the triage template reads the architecture names
  of the new model's config. When none of them reads a vocabulary head (`ForCausalLM`, `ForConditionalGeneration`, `LMHeadModel`,
  `ForMaskedLM`, `ForSeq2SeqLM`, `ForPreTraining`) and no tensor of the model is an `lm_head`, the model has no output head to use
  (`head.not_used` in `delta.json`). The nearest model's `lm_head` is then not a missing tensor, the class follows the backbone (`weights-only`
  when the backbone matches), and a hazard in `delta.json` says the generated-token check does not apply. The class says nothing about how the
  output is read: the scoring check, chosen by the kind, does that. A `ForCausalLM` model with tied embeddings names a head architecture and is
  unchanged. The 8B VL embedding bundle is a Qwen3-VL with its text tower read at the last position after the final norm, so the nearest bundle
  would be a Qwen3-VL one.
- Hazard scan. The scan predicts failures from the config alone. Its vocabulary hazards (65,536 columns per die, and a vocabulary that is not
  whole tiles) concern a full output head. Built (issue 380): for a reranker (two rows of the head), for an embedding model (no head) and for a
  config that describes a bare backbone, `scan_model_hazards` leaves out those two hazards and its notes say which would have hit and why they
  do not apply. The tile, expert and mixer checks apply as they are, because the backbone is the same.

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
| a `media=` argument on `score_pairs` and `embed` | rows that name files | passed only when the pair file has media (issue 377) |
| `embed(model, ids: list[int]) -> list[float]` | `embedding` | called once for each input (issue 376) |
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

For an embedding model (issue 376) the parent computes the cosine of each chip vector to the Hugging Face vector (the last position's
final-normed hidden state, L2 normalised, fp32, recorded by `python3 -m orchard.scoring_reference embed`), and the mean and the minimum over all
inputs and for each input kind (text, multilingual, code, long). The check refuses a vector of the wrong length, a value that is not a finite
number, and a vector of all zeros. The record keeps the chip's vectors, because the gate recomputes every cosine. The reference vectors stay in the
stage 1 file, which the record names by sha256. A record of 24 inputs of 4,096 numbers is about 2 MB, and the limit is 500,000 numbers (about 10 MB).
The shape of the bounds is the public 8B VL embedding bundle's accuracy table (section 6). The nearest-document count that bundle also reports
(3 of 29 queries had a different nearest document) is not built.

Retrieval rows (issue 379). When the pair file labels each pair relevant or not, the check computes NDCG@10 and Hit@1 of the reference ranking
and of the chip ranking from the recorded scores, per group, and records both with their difference. The gate recomputes them and bounds how far
the chip falls below the reference (`ndcg_drop_max`, `hit1_drop_max`). The set is the shipped 8 groups of 4 documents, so these are parity rows
on a small set. The bundles' rows on subsets of MTEB reranking, MIRACL, ViDoRe and MMEB are not built: they need datasets whose licences and
sizes have to be checked before any text goes into the repository.

Media (issue 377). A row of the pair file may list up to 8 `media` entries (`kind` image or video, `path` under
`stages/1/evidence/reference/media/`, `sha256`). The recorder stores a file there and the hash in the row, so the reference and the port get
the same bytes. The check reads each file without following a link and under 16 MiB (at most 64 files and 128 MiB in all), checks the sha256,
and hands the bytes to the port through `media=`; the worker checks the sha256 again when it reads the file. The gate reads the files again and
refuses one that changed, was removed or became a link after the run. The record stays small because it holds hashes, not media. The recorder for
a vision-language model (a processor that turns an image into the model's tensors on the CPU) is not built, so the interface runs against fakes only.

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

Two more reranker bounds are choices with no public table behind them: NDCG@10 of the chip may fall below the reference's by at most 0.03, and
Hit@1 by at most 0.15 (one group in the shipped 8 is 0.125).

The embedding bounds (`EMBEDDING_BOUNDS`) come from the public card of the 8B VL embedding bundle (raahemnabeel/qwen3-vl-embedding-8b,
`doc/accuracy.md`): over 81 inputs the cosine to the Hugging Face fp32 vector is 0.9977 on average and 0.9926 at the minimum, and 0.9960 on
average for the image and the video inputs.

| Bound | Default | Evidence |
|---|---|---|
| mean cosine at least, overall and for each kind | 0.995 | the bundle's 0.9977 overall and 0.9960 for image and video |
| minimum cosine at least, overall and for each kind | 0.985 | the bundle's 0.9926 |
| inputs, kinds at least | 24, 3 | the shipped set has 24 inputs of 4 kinds |

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

A process the port starts in a session of its own (`setsid`, a double fork) survives the supervisor's process-group kill on a timeout or
an abort, because the template's cleanup dies in the same kill. That needs a cgroup or a PID namespace (issue 390).

Parity with Hugging Face on 32 pairs is not a quality measure. The bundles also report retrieval benchmarks (NDCG@10 on subsets of MTEB
reranking and MIRACL) as parity rows. Slice 2 computes NDCG and Hit@1 on the shipped set only (section 5).

## 8. What is built and what is left

Built in slice 2 (CPU and fakes only):

1. (issue 376) The embedding kind: the recorder (`embed`), `embed` in the template, cosine mean and minimum per input kind, bounds, the gate.
   The embedding bounds in section 6.
2. (issue 378) The model kind in stage 0 and the stage machine: `--model-kind`, the card and architecture rules, `stage_end` `model_kind`,
   stage 1 (skill `scoring-reference`, gate `gate_scoring_reference`) and stage 2 (skill `scoring-check`, gate `gate_scoring`, draft
   `scoring_check`) for class `weights-only`, or `full-port` with `--full-port`; no finish step (the check's record is the result); the
   ledger holds the sha256 of the template and the config as copied, and stage 2 refuses a reference that is not the file stage 1's gate
   accepted; a pause before stage 3 and a pause before stage 1 for a class with no route (block code `scoring-route`).
3. (issue 379) NDCG@10 and Hit@1 from the recorded scores, with bounds.
4. (issue 380) The stage 0 rule for a model with no output head, and the hazard scan's handling of it (section 3).
5. (issue 377) Media in the pair set and in the port interface, with size bounds and hash checks, against fakes.

Left, each in a public issue:

1. (issue 375) A real run on a board: the 4B reranker port, the recorder on the real snapshot, the bounds checked against the run.
2. A stage 5 serving check for a `/v1/rerank` style API, and stages 3, 4 and 6 for a scoring model (the run pauses after stage 2).
3. A recorder for a vision-language reranker or embedding model (a processor and real image and video inputs) and a real run of one.
4. Retrieval benchmark subsets (MTEB reranking, MIRACL, ViDoRe, MMEB) with their licences checked, and the nearest-document count.
5. A scoring route for the classes other than `weights-only` and `full-port` (`weights+sidecar`, `vocab-trim`, `stock-runtime`).
