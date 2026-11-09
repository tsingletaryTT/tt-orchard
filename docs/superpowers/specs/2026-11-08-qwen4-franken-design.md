# Qwen4 frankenmodel: real weights in a Qwen4-shaped skeleton

Date: 2026-11-08. Status: the generator and its CPU proofs are built (`scripts/tiny_qwen4/make_franken.py`, `tests/test_make_franken.py`).
Nothing here ran on a chip. The hardware run in section 6 has not been started and needs the chips.

## 1. Why

The tiny Qwen4 rehearsal (`2026-11-02-tiny-qwen4-rehearsal-design.md`) ports a 64-wide model with random weights. It shows that the
harness can port the Qwen4 layer types. It cannot show two things: that the port handles the real widths (hidden 5120) and real weight
values, and that a wrong port is caught by something other than a token comparison against noise.

No Qwen4 weights exist. The frankenmodel borrows real weights from the nearest model that does exist. Qwen3.8-27B has the skeleton the
public readings give for dense Qwen4-27B: hidden 5120, three gated delta net layers for each gated attention layer, attention with 24
query heads, 4 key-value heads of 256 and an output gate, delta net with 16 key heads and 48 value heads of 128. What the readings add
is hyper-connections with four streams, QSA indexers on the attention layers, and a multi-token-prediction head.

The frankenmodel keeps the base's first K layers (K is 4 or 8) with their real weights and adds the hyper-connections and the indexers
in a way that leaves the function unchanged. A port of it is then checked against a model whose behaviour is known from outside: the
base model run through transformers' own Qwen3.5 class.

## 2. The Qwen4 features, and what is proven

The feature list is the public reading recorded in `docs/run-logs/2026-11-02-dense-qwen4-prep.md` and the project log. No published
Qwen4 configuration confirms any row.

| Feature (public reading) | Proven already, with the record | What the frankenmodel adds | Needs real Qwen4 weights |
|---|---|---|---|
| Hybrid gated delta net and gated attention, 3:1 | Tiny Qwen4 run 23 passed all stages with the helper's port (random weights, 4 layers, top-1 1.0 over 32 tokens; `docs/run-logs/2026-11-03-tiny-qwen4-rehearsal-first-run.md` has the first run, the later runs are in the project log). Olmo-Hybrid-7B, a 24:8 delta net and attention model with real weights, passed stages 0 to 6 on four chips; its can and run log are in PR 633. | A real-width delta net and attention in the Qwen4 class, with real values. | Nothing for the layer types. The real layer schedule and the head counts. |
| Hyper-connections, 4 streams | Run 23 ported them with random weights and `hc_lowrank` 8. | Real width (20480 streams wide, `hc_lowrank` 640), and a perturbed variant that a port must reproduce. | The trained mixing weights, which only change values. |
| QSA indexers on the attention layers | The Qwen3.8-27B-QSA bring-up (the release with trained indexers) passed stages 2 to 4 on the chips; its ledger is under the run directory and is not yet written up in the repo. | The release's real indexer tensors, mapped into transformers' indexer class (section 4). | Whether the indexer's behaviour matches the release past the budget. A prompt longer than the budget (2048 tokens). |
| Multi-token prediction head | None. transformers' `Qwen4Exp` class has no MTP head, and the harness serves none. | Nothing: the head is left out and the manifest says so. | The head and a speculative-decoding path to use it. |
| Per-layer embedding and n-gram table | Run 23 ported a tiny one. | Nothing: `ple_layer_ids` is empty. | The widths. The public readings give none, and the full-size n-gram table would be 102 GB. |
| Long context (the reading of a very large window) | Nothing. | Nothing: the proofs use prompts of 8, 16 and 33 tokens. | A long-context run, and the model. |

## 3. How the equality is built

transformers' `Qwen4ExpForCausalLM` (5.19.0) is the reference for the shape. Its hyper-connection is a gated residual:

* The embedding is repeated into four streams.
* Before a block, `hc_norm` normalises each stream in groups of the hidden size. The block reads
  `mean over streams of sigmoid(up(silu(down(streams) / 4))) * hc_norm(streams)`.
* After the block, every stream gets `streams + 2 * sigmoid(inject(streams) / 4) * block_output`.
* After the last layer, the same mixer (without the injection) reads the streams. It plays the part of the final norm, because the class
  has no `norm` of its own.

If `down`, `up` and `inject` are all zero, the mix weight is `sigmoid(0) = 0.5` and the injection weight is `2 * sigmoid(0) = 1`. The
streams stay equal. The block input is `0.5 * rms_norm(x) * (1 + hc_norm.weight)`. The base block input is `rms_norm(x) * (1 + w)`. So
`hc_norm.weight = 1 + 2 * w` makes the two equal, and the base's `input_layernorm`, `post_attention_layernorm` and final `norm` become the
`hc_norm` of the attention block, the feed-forward block and the final mixer. Every operation on the way is a scaling by a power of two or a
mean of equal values, so the result is exact in float32. In bf16 the stored `1 + 2 * w` is rounded and the match is approximate.

One more setting matters. The delta net's gated norm in `Qwen4ExpText` reads `output_gate_type`, and the class default is `sigmoid`. The base
uses `silu` (it calls it swish). With `sigmoid` the first layer's output differs from the base by about 0.2 on a small model. The generator
writes `silu`, and a test pins it.

The feed-forward block is dense. transformers builds a mixture-of-experts block in every layer, so the checkpoint is loaded through
`orchard/skills/reference-gate-templates/qwen4_dense.py`, as every dense Qwen4 checkpoint here is.

## 4. The indexers

QSA selects, for each query, the blocks of four tokens with the highest score, up to a budget of tokens (2048). The class keeps the
current partial block and every block when there are fewer blocks than the budget allows. A prompt shorter than the budget therefore
attends to everything, and the indexer weights cannot change the output. They are random in a default frankenmodel.

With `--indexers DIR` the generator reads `qsa-indexers.safetensors` from the higa2515 release (safetensors only, never a `.pt` file or the
release's code) and maps, per attention layer:

| Release tensor | transformers tensor |
|---|---|
| `q_proj.weight` (512 x 5120) and `k_proj.weight` (128 x 5120) | `index_qk_proj.weight` (640 x 5120), the two joined |
| `q_norm` | `q_layernorm.weight` |
| `k_block_norm` | `k_layernorm.weight` |
| `k_norm` | not carried |

The release's norms are zero-centred like the class's, which its `indexer.py` says in text. The release also applies `k_norm` to the raw
keys before pooling, and the class does not. The shapes fit, and the mapping is a reading that nothing has verified. It matters only for prompts
longer than the budget, where a lowered budget (`--set indexer_budget=N`) tests that the indexer path runs and changes the output.

## 5. What the checks show and what they do not

`make_franken.py` runs these checks. The first two run on every generation, the rest with `--prove`, which writes `proofs.json`.

1. **Degenerate-reference guard (every generation).** The stage 1 prompt ("Write the text I send my landlord about the broken boiler.",
   through the checkpoint's chat template) is greedy-decoded for 32 tokens on the CPU. A decode with fewer than 4 distinct ids exits 3 (the
   checkpoint stays on disk with `degenerate: true` in its manifest; `--allow-degenerate` turns the exit into a warning). Reason: stage 2
   compares chip tokens with these 32 reference tokens, and a constant reference passes a port that outputs the same constant.
   `make_checkpoint.py` has the same bar. `--skip-decode-check` skips it for a generation that cannot run a forward pass.
2. **Teacher-forced reference (every generation).** `reference/teacher-forced.json` holds the model's top-1 id at each of the 155 positions of a
   fixed passage, and the five highest logits of the last 8. A greedy decode of a truncated model can be constant while its top-1 answers
   at 155 positions still vary, so this is the informative part of the reference.
3. **Equality with the base.** The franken's logits against the same K layers loaded through transformers' real Qwen3.5 class, on three seeded
   prompts of 8, 16 and 33 tokens. The proof record names the dtype each model ran in. float32 rule: a difference of at most 5e-4 and the same
   top-1 token everywhere. bf16 rule: 5% of the largest logit, and top-1 may flip on near ties.
4. **Perturbation.** With `--hc-sigma` the franken must differ from the base by more than the rule allows.
5. **Indexer.** The identity and perturbation claims are judged on the model with every block selected (a budget of at least the prompt length
   plus a block), so a small configured budget cannot satisfy either claim through sparse attention. The comparison of the all-selected model
   with a budget of 8 tokens always runs, and a missing indexer proof counts as a failure.
6. **Determinism.** Two runs give identical logits.

What a pass means: the hyper-connection and indexer wiring computes the base model's function, and the port target is a model with real
weights whose correct output is known. A port that gets a hyper-connection or the gate activation wrong fails against it.

What it does not mean: the franken is not a Qwen4. Its weights are the first K layers of a 64-layer model, so its text is not coherent and no
quality figure can come from it. The agreement proves equality with Qwen3.8 on the same layers. It says nothing about how Qwen4's trained
hyper-connection or indexer weights behave. The stage 2 text floor (readable free-run text) will not pass on its output, and the
supervisor's `--full-port-rehearsal` option waives only that field.

## 5a. Results on the real weights (2026-11-08, CPU, no chip)

The records are in `docs/run-logs/evidence/2026-11-08-qwen4-franken/`. The base is the Qwen3.8-27B snapshot `1d4bf0f2`, the indexers are the
higa2515 release snapshot `aedae2c5`, the proof prompts are seeded token ids of 8, 16 and 33 tokens, and every run used the reference
interpreter (torch 2.14.1, transformers 5.19.0).

A first version of these proofs loaded the bf16 franken in float32 (the loader mistook transformers' normalised `config.dtype` for "not a
string"), so the first bf16 figure (0.119) compared a float32 franken with a bf16 base. The loader now reads the dtype from the written config,
the record names both execution dtypes, and the bf16 figures below are bf16 against bf16.

**How many layers before the stage 1 decode is not degenerate.** Greedy decode of the stage 1 prompt, 32 tokens, franken against the same layers
loaded through the Qwen3.5 class:

| K layers | Size (bf16) | Distinct ids | First 12 generated ids | Base gives the same ids |
|---|---|---|---|---|
| 8 | 12.1 GB | 1 | 220 x 12 (a space) | yes |
| 12 | 15.6 GB | 2 | 220, 220, 11, 220, 220, 220, 11, 220, 220, 11, 220, 220 | yes |
| 16 | 19.0 GB | 2 | 220, 220, 220, 220, 220, 11, 11, 220, 220, 11, 220, 220 | yes |
| 20 | 22.5 GB | 14 | 0, 0, 0, 0, 0, 2415, 261, 518, 477, 477, 344, 19937 | yes |

The base shows the same ids at every K, so the degeneracy comes from truncating the model and the identity hyper-connection path adds nothing.
K = 8, 12 and 16 fail the guard. K = 20 passes it, with 14 distinct ids and text that is varied and unreadable.

**Equality and perturbation.** Largest logit difference to the base over the three prompts, with every block selected:

| Checkpoint | Largest logit difference | Rule | Top-1 | Holds |
|---|---|---|---|---|
| K = 4, float32, identity | 0.0 | 5e-4 and top-1 equal | equal | yes |
| K = 8, bf16, identity | 0.125 | 0.86 | equal | yes |
| K = 12, bf16, identity | 0.125 | about 0.87 | equal | yes |
| K = 16, bf16, identity | 0.125 | about 0.87 | equal | yes |
| K = 20, bf16, identity | 0.25 | 0.87 | equal | yes |
| K = 8, bf16, hyper-connections from N(0, 0.02^2) | 5.20 | 0.87 | differs | differs, as intended |

* The float32 difference is exactly zero because every operation on the identity path is a scaling by a power of two or a mean of equal
  values. The bf16 differences are the rounding of the stored `1 + 2 * w` norm weights plus bf16 arithmetic, at most 1.5% of the largest logit.
* Teacher-forced reference over 155 positions: K = 20 has 49 distinct top-1 ids and agrees with the base at 154 of 155 positions (99.35%; the
  one flip is a bf16 near tie). K = 4 float32 has 5 distinct ids and agrees at all 155. The perturbed K = 8 agrees at 89.0%.
* The indexer with a budget of 8 tokens moves the logits by 1.11 (K = 8), 2.17 (K = 20) and 0.88 (K = 4, float32) against the same model with
  every block selected, so the indexer path runs and changes the output once the prompt is longer than the budget.
* Two runs of each checkpoint give identical logits. The size estimate made before writing was within 21 kB of the file written.

**Consequence for a hardware run.** The K = 20 bf16 checkpoint (22.5 GB) is the smallest tested truncation whose stage 1 decode is not degenerate,
and it is the candidate for the hardware run. Stage 2 still compares only 32 greedy tokens with 14 distinct ids. The 155-position teacher-forced
reference (`reference/teacher-forced.json`) is the larger test. The supervisor does not read it, so a hardware run has to compare the chip's
top-1 ids at those positions by hand or through a stage 2 gate change (proposed, not made).

The other checkpoints were deleted after their proofs to save disk. They regenerate with the commands in each manifest.

## 6. The hardware run

Not run. A later run uses the supervisor driver pattern of the qwen-like runs, with a local model path, `--full-port` and
`--full-port-rehearsal`, the dense shim and no remote code (the checkpoint has no code of its own):

```
python3 -m orchard.supervisor run --model local/qwen4-franken-k8 --run-dir RUN --tiers TIERS --coder-target raahemnabeel/qwen3-coder-next \
  --coder-kind container --coder-profile p300 --coder-port PORT --coder-chips 2 --required-chips 2,4 --gozer gozer --skills-dir SKILLS \
  --env HF_HUB_OFFLINE=1 --input model=CHECKPOINT --input reference_python=~/orchard-venvs/reference/bin/python \
  --input base_candidates=BASE_CANDIDATES.json --unattended --full-port --full-port-rehearsal --accept-credentials-visible
```

`BASE_CANDIDATES.json` names `Qwen/Qwen3.8-27B` as the nearest model, in the format of `~/orchard-runs/notes/qwen-like/runs/*/base-candidates.stage0.json`.
Use the K = 20 bf16 checkpoint (22.5 GB). A K = 8 checkpoint fails stage 1: its CPU reference is the same space token 32 times, so the
'matches the card' form check fails and stage 2's token agreement could not tell a wrong port from a right one. Pass `--input base=`
pointing at the Qwen3.8-27B snapshot so stage 0 can compare tensor names.
