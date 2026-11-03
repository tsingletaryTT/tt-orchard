# Tiny Qwen4-shaped model for rehearsals: design

Date: 2026-11-02. Status: the checkpoint generator is built (`scripts/tiny_qwen4/make_checkpoint.py`). The pilot run in section 7
has not been run and needs the operator's approval. Nothing here ran on a chip.

## 1. Why

Two things have never been exercised:

1. The supervisor's `--full-port` path (PR 331, `docs/superpowers/specs/2026-11-02-full-port-supervisor-design.md`). No local
   coder has written a port through it. A real attempt on a 27B model costs about 100 GiB of disk and hours per hardware test.
2. The Qwen4 layer types (gated delta net, sparse-indexer attention, hyper-connections, n-gram embedding) on a TT chip, at a
   size where one run takes minutes.

A model with the Qwen4 layer layout, random weights and a hidden size of 64 answers the first question in minutes and gives the
coder a port target for the second. Random weights are enough because the stage 2 gate compares chip tokens with CPU reference
tokens for the same weights. No part of the check needs a trained model.

Out of scope: training. ttml has no backward for the gated delta net, hyper-connections or the sparse indexer. Output quality:
random weights produce noise, so stages that judge text quality (stage 5 and later) are not part of this rehearsal.

Known blocker (found in the PR 356 review): the stage 2 gate for a full port requires `coherent: true` in `result.json` and a
free-run text of at least 20 non-space characters and 4 different words (`orchard/full_port.py`, `MIN_FREE_RUN_CHARS` and
`MIN_FREE_RUN_WORDS`; the skill `full-port-check.md` tells the agent to write `coherent: false` for unreadable output). A model with
random weights emits byte noise, so a correct port of it fails stage 2 on that one field. The rehearsal therefore cannot end in a stage 2
pass without either a gate change or a false `coherent`. This note takes this policy: the pilot is read for the other gate reasons
(top-1 agreement, token count, provenance, port sources), and a failure whose only reason is `coherent` counts as a pass of the
rehearsal. The agent must never write `coherent: true` for noise; a run where it does is a finding about the agent.

Update (issue 357): `tt-orchard bringup ... --full-port --full-port-rehearsal` (or `python3 -m orchard.supervisor run ... --full-port
--full-port-rehearsal`) waives the `coherent` field and no other check. The option is given when the run starts, is recorded in
`run_start` as `full_port_rehearsal`, and a resume keeps it and cannot add it. The supervisor passes it to the gate from the ledger; a
field in `result.json` does nothing. When stage 2 passes with `coherent` not true, the ledger gets a decision
`full-port rehearsal: coherence waived`. The text floor (20 non-space characters, 4 different words) still applies: a free run of
noise that is shorter than that fails, and that failure is data about the port.

## 2. What the model is

A flat `qwen4_exp_text` config (the text tower alone, class `Qwen4ExpForCausalLM` in transformers 5.19.0), written with the key
names of the Flash-Next config. Three variants share every dimension except the feed-forward block:

| Variant | Feed-forward blocks | Why |
|---|---|---|
| `dense` | all layers a plain gated MLP, no `num_experts` key | The dense Qwen4 reading. Exercises `qwen4_dense.py`. |
| `moe` | all layers sparse MoE (8 experts, top 2, plus a shared expert) | The Flash-Next shape. Loads without the shim. |
| `mixed` | layer 0 dense (`mlp_only_layers: [0]`), the rest MoE | Exercises the layer rule inside one model. |

Dimensions (full-size reading in brackets):

| Item | Tiny | Full-size reading | Note |
|---|---|---|---|
| `hidden_size` | 64 | 5120 | |
| `num_hidden_layers` | 4 (option `--layers`, a multiple of 4) | 64 | |
| `layer_types` | `linear_attention` x3, `full_attention` x1, repeated | assumed 3:1 | The schedule is an assumption in the repo's fixture too. |
| Attention heads | 8 query, 2 KV, `head_dim` 32 | 24 / 4 / 256 | Ratio 4:1 here, 6:1 full size. Both divide 2 and 4 dies (KV by replication). `head_dim` 32 is one whole tile; 16 trips `head_dim_not_whole_tiles`. |
| Gated delta net | 2 key heads, 6 value heads, width 32, conv kernel 4 | 16 / 48 / 128 | Same 1:3 ratio. |
| `hc_count` / `hc_lowrank` | 4 / 8 | 4 / 640 | `hc_lowrank` is set from the config, never defaulted. 8 keeps the 1/8 ratio to the hidden size. |
| `intermediate_size` (dense) | 256 | about 14336 (inferred) | A multiple of 128, so 2 and 4 dies get whole 32-column tiles. |
| MoE (variants `moe`, `mixed`) | 8 experts, top 2, width 64, shared expert 64 | 512 / 10 / 640 | |
| Vocabulary | 384 | 248320 | 384 gives whole tiles on 2 and 4 dies. |
| Sparse indexer | budget 16, compress ratio 4, 4 heads, width 32, 1 KV head | 2048 / 4 / 4 / 128 / 1 | The full-attention layer needs these keys or it does not build. 4 heads divide 2 and 4 dies; 2 heads trip `qsa_heads_not_divisible_by_dies`. |
| N-gram embedding | `ngram_size` 3, base vocabulary 1000, 2 heads per n-gram, 4 split parts, `ple_layer_ids` [2], `ple_embed_dim` 64 | 20,000,000 base | The table is 4096 x 16 (65k values) here. Full-size Flash-Next has a 102 GB table. |
| Output gate | `output_gate_type` sigmoid | sigmoid | |
| Context | `max_position_embeddings` 512 | 262144 | |
| MTP | not written | 1 layer | The reference loader ignores `mtp.*` keys, so no gate would see it. |

Parameter counts are under 2 M for every variant; a bf16 checkpoint is a few MB.

## 3. What each layer exercises

| Layer (default 4) | Block | Qwen4 feature |
|---|---|---|
| 0 | gated delta net, gated MLP or MoE | Short causal conv (kernel 4, depthwise), delta-rule recurrence, output gate. Two hyper-connection mixes. |
| 1 | gated delta net, plus the n-gram embedding module | N-gram hashed embedding (PLE) with its own conv. |
| 2 | gated delta net | The recurrent decode step (one token) against the chunked prefill. |
| 3 | full attention with the sparse indexer | Query/KV head grouping, sigmoid output gate, block-pooled indexer scoring and top-k block selection. |
| after the last layer | hyper-connection mixer, final norm, LM head | Merging the 4 residual streams. |

Chip-side facts from `docs/run-logs/2026-11-02-qwen4-ops-probe.md` that a port will meet at these shapes:

- `ttnn.conv1d` returns wrong values for a depthwise convolution with dilation above 1 (PCC 0.18 to 0.28). The plain kernel-4
  short convolution in the gated delta net has dilation 1 and is correct. Dilation shows up only if the port copies the
  Flash-Next indexer or n-gram convolution as written.
- `chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule` match the reference at Flash-Next head counts. At these tiny
  head counts the probe also ran (2 key heads, 4 value heads, width 64) with the same error level. Six value heads of width 32 is a
  combination the probe did not run.
- The gated residual composed from ttnn ops matches at hidden 256 and 2560 (PCC 0.99996).

### Width presets (`--width`, `--set`, `--max-gb`)

The tiny widths hide problems that only a real width shows: tiling and sharding at hidden size 5120, the delta-net chunk size at 48
value heads, the memory a 248320-row vocabulary takes on each die, and speed. `--width 27b` keeps four layers and random weights and
takes the full-size column of the table above: hidden size 5120; 24 query and 4 KV heads of width 256; 16 key and 48 value delta-net
heads of width 128; `hc_lowrank` 640; dense feed-forward width 14336 (the inferred figure in the table; the Qwen3.8-27B config has
17408); vocabulary 248320; indexer budget 2048 with 4 heads of width 128 and 1 KV head; 4096 positions. For `--variant moe` and
`mixed` it also takes 512 experts, top 10, expert and shared-expert width 640. Not scaled, because no public reading gives a width:
the per-layer embedding (`ple_embed_dim` 64, `ple_layer_ids` [2]) and the n-gram table (base vocabulary 1000; the full-size table is
102 GB). `max_position_embeddings` is 4096, a working length, where the model has 262144.

Sizes, from the parameter count on torch's meta device (nothing allocated): the 27b dense checkpoint is 4.12 billion parameters,
8.2 GB with its tokenizer (measured: 8,235,973,392 bytes of weights against an estimate of 8,235,973,536). The 27b MoE variants are far
larger (512 experts of width 640 in every MoE layer): `mixed` is 37 GB and `moe` 47 GB, and the default `--max-gb 12` refuses them;
raise it only with the disk to match. A 27b dense checkpoint with a 3-prompt, 8-token CPU reference took 58 seconds to write on a busy box. The script refuses an estimate over `--max-gb` or over the free space
of the output filesystem minus 40 GB, before it creates the directory.

`--set KEY=VALUE` (repeatable) changes one width after the preset. The keys are the width keys only (not the layer count, the layer
schedule, the per-layer embedding or the n-gram table), only keys the variant already has (an expert key on a dense config would make
transformers fill in 512 experts), and positive whole numbers. A scaled config (a preset other than `tiny`, or any `--set`) must
tile on 2 and 4 dies: hidden, feed-forward, expert and vocabulary widths are multiples of 128 (whole 32-column tiles on 4 dies); head
widths are multiples of 32; the attention, delta-net and indexer head counts divide by 4; a KV head count divides by 4 or divides 4
(replication); the value heads are a multiple of the key heads; the vocabulary holds the 260 special and byte tokens. The tiny default
is not held to these rules and is written exactly as before: the config, prompts, weights and tokenizer files are byte-identical to the
output of the script before the presets existed (the manifest gained `width`, `overrides` and `size`).

## 4. How the checkpoint is made

`scripts/tiny_qwen4/make_checkpoint.py` runs in the reference interpreter (`ORCHARD_REFERENCE_PYTHON`, torch and transformers
5.19.0 on the CPU). It never imports ttnn or ttml and opens no device.

1. Build the config for the variant (`build_config`). A dense config has no expert keys at all, because the transformers default
   of 512 experts would build the wrong model without an error.
2. Build the model through the dense shim (`qwen4_dense.py`) under `torch.manual_seed(seed)`, so one seed gives one set of
   weights. Float tensors are cast to bf16 and written as `model.safetensors` (integer buffers keep their dtype). The key names
   come from the transformers class, so they match what the stage 1 loader expects.
3. Write `config.json`, `generation_config.json` and, unless `--no-tokenizer` is given, a byte-level tokenizer (`tokenizer.json`,
   `tokenizer_config.json`). The tokenizer has the 256 byte tokens, four special tokens (ids 0 to 3) and 124 filler tokens so
   that its size equals the config's `vocab_size`. It is built with the `tokenizers` library, offline. Without it, stage 1's
   `reference_gate.py` stops at its tokenizer step, because that template calls `AutoTokenizer.from_pretrained`. The tokenizer was
   added beyond the brief for that reason. The id-based prompts below do not use it.
4. Write `prompts.json`: seeded id lists (lengths 8, 16 and 33, ids in the byte-token range 4 to 259). Length 33 is not a multiple
   of the delta-net chunk size, which covers the padding path.
5. Run the CPU reference (`--reference`, default on) through `reference_gate.load_model`, the loader stage 1 uses. The run refuses
   to record anything unless the load report is clean (no missing, unexpected or mismatched keys). For each prompt it decodes 32
   tokens greedily with one full forward pass per token, checks the first token against a separate forward pass over the prompt, and
   checks that every step appends one token at the end. It refuses a degenerate result (fewer than 4 distinct generated ids in a
   prompt), because a constant output would let any wrong port pass.
6. Write `reference/` (`prompt-ids.json` and `generated-ids.json` for the first prompt in the shape stage 1 writes, plus
   `reference-ids.json` for all prompts with top-5 ids and logits) and `tiny-qwen4-manifest.json` (seed, variant, dimensions, the
   sha256 of every file, the transformers and torch versions, the shim's sha256).

The output directory must be outside the repository checkout. Checkpoints are generated into a temp directory and are never
committed. The generator is deterministic: two runs with the same seed and variant give identical files and identical reference ids.

## 5. What stages 0, 1 and 2 see

Stage 0 (`orchard/hazards.py` and the classifier). The config scan has no default for `hc_count` and `hc_lowrank` and reads
both. With `qwen3.8-flash-next.json` as the nearest config:

- dense: `dense_model_nearest_is_moe` and `dense_qwen4_not_flash_next_shape`. All division checks pass at 2 and 4 dies (a test
  checks this on the generated config).
- moe: the Flash-Next hazards only, with no dense-shape notes.
- The stage 0 class for a local snapshot depends on the tensor comparison with the nearest model, which this document did not run.
  The run needs class `full-port`. If stage 0 gives another class, the runbook stops and reports it.

Stage 1 (`reference_gate.py` copied into the stage directory). With the tokenizer present it loads through the shim for `dense`
and `mixed`, and directly for `moe`, writes `dense_qwen4` (layers, note, shim sha256) into the load report, decodes the
template's English prompt through the byte tokenizer, and writes the ids files that stage 2 reads. The generated text is random
bytes, so the card check ("matches the card") passes on form only: it needs non-empty text, and the filler tokens decode to
non-empty strings. The gate that compares the chip with this reference is the stage 2 gate; the template's prompt and the id-based
prompts here are separate reference sets, and the stage 2 port check should use the id-based ones (see section 7).

Stage 2 (`gate_full_port`, `evidence/port-check.json`). The gate recomputes top-1 agreement between chip ids and the stage 1 ids
and needs at least 16 positions (`SWAP_MIN_TOKENS`) at a top-1 agreement of 0.85 (`SWAP_TOP1_MIN`). At 32 tokens per prompt and three prompts the generator gives
96 positions. The record must name the stage 1 file and its sha256. A bf16 port of random weights can diverge after the first few
tokens because ties between near-equal logits are common; the reference records the top-5 logits so a divergence can be told apart
from a wrong port (the gap between the top two logits at the first differing position).

## 6. Tests and guards

`tests/test_tiny_qwen4.py` (no network, no hardware; the tests that need torch and transformers skip without
`ORCHARD_REFERENCE_PYTHON`). Each guard has a mutation that must turn a test red:

| Guard | Test |
|---|---|
| Layer schedule is 3 linear, 1 full, repeated | `layer_types` matches for 4 and 8 layers; a count that is not a multiple of 4 is refused |
| Dense config has no expert keys; MoE and mixed have all of them | key check per variant |
| `hc_count` 4, `hc_lowrank` present and not 320 or 640 | config check; the stage 0 scan reads both |
| Dimensions divide 2 and 4 dies | the scan reports none of the division hazards on the generated configs |
| Same seed gives the same bytes, a different seed gives different bytes | two builds compared |
| Floats are bf16, the hybrid-layer keys exist, a dense checkpoint has no expert tensor | the safetensors header is read |
| The first-token check, the greedy choice and the manifest are consistent | `first_token_ok` is true for every prompt; `greedy_token` equals `torch.argmax` on tied bf16 logits (PR 356 review) |
| A failed refresh leaves no old ids; a successful one rewrites the manifest | `--reference-only` on a broken and on a bare snapshot (PR 356 review) |
| Stage 1's own template passes on the checkpoint | the unmodified `reference_gate.py` is run in a stage directory |
| Prompt ids are inside the byte-token range and seeded | pure function test |
| Output directory inside the repository is refused | refusal test |
| Reference refuses a load with missing keys | a key removed from a copy of the checkpoint |
| Reference refuses a degenerate decode | zero-scale weights |
| Reference ids are the same on two runs | two reference runs compared |
| Manifest hashes match the files | recomputed |

`tests/test_tiny_qwen4_width.py` covers the width options. Each guard has a mutation that turns a test red:

| Guard | Test |
|---|---|
| The default output is unchanged | `config.json`, `generation_config.json` and `prompts.json` hashes pinned from the script before the presets; the weights and tokenizer hashes too, when the torch, transformers and tokenizers versions are the pinned ones |
| The tiny default is not held to the tile rules; a scaled config is | the rules reject the tiny config when called directly; a one-key override on the tiny base is refused |
| `27b` has the full-size widths in four layers, no expert key on a dense config, the full-size expert widths on `moe` and `mixed` | config checks, and the stage 0 scan reports no division hazard at 2 and 4 dies |
| Widths tile on 2 and 4 dies | 24 refused widths (not a multiple of 128 or 32, heads not dividing 4, a KV count that neither divides nor is divided by 4, value heads not a multiple of key heads, expert count above the pick count, a vocabulary under 260), each with the key named |
| An override names a width key the variant has, as a positive whole number | refusals for a non-width key, an expert key on a dense config, a dense width on a MoE config, booleans, floats, strings, zero, negatives, nan and inf |
| An override is bounded (1,048,576; 4,096 experts) so counting the parameters cannot hang | `10**20`, a vocabulary of 2**21, 4,097 experts are refused; the limits themselves pass |
| `--set` parses to whole numbers | missing `=`, empty key, repeated key, non-digits |
| The size estimate is checked before the directory exists | over `--max-gb`, over the free space minus the margin, exactly at the limit, a bad limit or estimate; the run stops before `mkdir`; the estimate is printed |
| The estimate counts parameters without allocating them | the 27b config is counted in a child under 3 GB resident, and the estimate is within 2% of the weights file on the tiny model and a scaled one |
| The tokenizer is sized to the config's vocabulary | a 512-wide vocabulary, round trip |
| The manifest records width, overrides, estimated and actual sizes, and a refresh keeps them | read back after a build and after `--reference-only` |

## 7. Runbook for the pilot run (operator approval needed; not started)

Goal: watch a local coder write a port of the tiny dense model through `--full-port`, with stage 2 comparing chip tokens with the
CPU reference. Cost: one board for each hardware test, a few minutes each.

1. Generate. `ORCHARD_REFERENCE_PYTHON=<reference interpreter> python3 scripts/tiny_qwen4/make_checkpoint.py --variant dense --out <temp dir>/snapshot`.
   Read `tiny-qwen4-manifest.json`; the reference ids exist under `reference/`.
2. Place the snapshot where the run reads it: `<HF_HOME>/hub/models--local--tiny-qwen4-dense/snapshots/<revision>/`, with
   `refs/main` holding the revision. Use a scratch `HF_HOME` (`--hf-home`). With the hub unreachable for a model that does not
   exist, the preflight's hub check is a warning when a local snapshot exists (`check_hub`), and `--no-fetch` forbids a download.
3. Dry run first: `tt-orchard bringup local/tiny-qwen4-dense --no-fetch --offline --full-port --dry-run --base <Flash-Next or the
   nearest bundle>`. Check that the preflight passes and stage 0's class would be `full-port`. If the preflight blocks on a model
   with no hub entry, stop and file an issue; do not work around it.
4. Run: `tt-orchard bringup local/tiny-qwen4-dense --no-fetch --offline --full-port --base ...` under the operator's usual
   coder and lease settings. Expected end state: the run stops after stage 2 with code `full-port-stages-unbuilt` (the design's
   slice 1) only if the gate accepts the port. Run it with `--full-port-rehearsal` (issue 357), which waives the `coherent` field. Without
   that option the expected end state is a stage 2 failure whose reasons are exactly the coherence ones (`coherent must be true`, and
   possibly the free-run word count). Any other reason (top-1 agreement, token count, provenance, port sources) is data about the coder or the
   supervisor. Read the gate's full list of reasons in the run's ledger before deciding.
5. Review `stages/2/port/` by hand. The gate checks provenance and agreement, and a person decides whether the port is what it
   claims to be (design section 3.3 lists what the gate does not prove).
6. Repeat with `--variant mixed` only after the dense run ends in a known state.

What to record in the run log: the stage 0 class, the time each stage took, the number of coder turns, whether the port used
`chunk_gated_delta_rule` or a torch fallback, the first differing position (if any) and the logit gap there, and every place the
supervisor or its skills had to be corrected by hand.

Known gaps that the pilot may hit: the stage 2 prepare step's template compares ids from the stage 1 file for the template prompt,
and the id-based prompt set here is separate, so the port-check command in `hw_test.json` needs to read `prompts.json` for the
rehearsal to cover length 33. Whether the coder or the supervisor wires that is part of what the pilot learns. No supervisor
code was changed for this document.
