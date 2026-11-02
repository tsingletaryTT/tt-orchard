# Dense Qwen4: what changes in tt-orchard, and what waits for the real config (2026-11-02)

Status: provisional. No Qwen4 weights and no `config.json` exist. Everything below comes from public reports of an unreleased model, and the code says so in its comments and in the files it writes. Replace the assumptions when the real config is published.

Sources (public):

- TensorFold issue 1244: https://github.com/ashhart/TensorFold/issues/1244
- llama.cpp issue 31072: https://github.com/ggml-org/llama.cpp/issues/31072
- The `qwen4_exp` code in transformers 5.19.0 (the reference interpreter), read and run on the CPU.
- Intel's public vLLM patch (`intel/llm-scaler`, `build/v0.31.0`) for the layer rule.

## How a dense Qwen4 config differs

What the reports give (the fixture `tests/fixtures/model-configs/qwen4-dense-provisional.json` holds these as dimensions only):

| Item | Reading |
|---|---|
| Hidden size | 5120 |
| Gated delta net | 48 value heads and 16 key heads, head width 128 |
| Attention | 24 query heads, 4 KV heads, head dimension 256, sigmoid output gate |
| Hyper-connections | `hc_count` 4 (width 20480); `hc_lowrank` 640 at full size and 320 per rank at TP2 |
| FFN | about 14336 (inferred, not reported) |
| MTP | present |
| Experts | none: a plain gated MLP in every layer |

The fixture also assumes 64 layers in a 3:1 pattern of linear and full attention, and the Qwen3.8 vocabulary of 248320. Those three are copied from Qwen3.8-27B and have no Qwen4 source.

Against Qwen3.8-Flash-Next (the only public `qwen4_exp` config) the differences that matter for tt-orchard are:

- Flash-Next has 512 routed experts and a shared expert in every layer. A dense Qwen4 has an ordinary MLP, so the router, expert and shared-expert paths do not apply.
- Hidden size 5120 against 2560, KV heads 4 against 2, FFN replaces `moe_intermediate_size`.
- `hc_lowrank` is 640 at full size, 320 in Flash-Next. The 320 in the TP2 reading is a per-rank shard, so no code may use 320 as a default.
- transformers builds a mixture-of-experts block in every decoder layer of `qwen4_exp` and its config refuses `num_experts <= 0`. There is no dense reference upstream. A config that omits `num_experts` gets the class default of 512, which builds the wrong model without any error.

## What the shims assume

### Stage 0 hazard scan (`orchard/hazards.py`)

- `hc_count` and `hc_lowrank` are read from the config. There is no default. A missing value is `None` in the facts, and the notes say which one is missing. A value that is not a whole number of at least 1 is refused.
- A config with no routed experts, set against a nearest bundle that has them, reports `dense_model_nearest_is_moe`. A `qwen4_exp` or `qwen4_exp_text` config with no routed experts reports `dense_qwen4_not_flash_next_shape`, whether or not a nearest bundle is given.
- New division checks run at 2 and 4 dies and at the scan's own die count: `hyper_connection_width_not_divisible_by_dies` (`hc_count` x hidden size), `ffn_per_die_not_whole_tiles` (dense layers only; a multiple of 32 columns per die), `attention_heads_not_divisible_by_dies`, `kv_heads_not_divisible_by_dies` and `qsa_heads_not_divisible_by_dies`. Fewer KV heads than dies counts as replication and is accepted when the die count is a multiple of the KV heads (Flash-Next on 4 dies).
- The Flash-Next hazards are unchanged on 2 and 4 dies (a test checks this).
- On the fixture, with Flash-Next as the nearest bundle at 4 dies, the scan reports `kv_heads_above_nearest`, `dense_model_nearest_is_moe` and `dense_qwen4_not_flash_next_shape`. All division checks pass at 2 and 4 dies.

### Stage 1 reference (`orchard/skills/reference-gate-templates/qwen4_dense.py`)

- The rule is Intel's: layer `i` is MoE when `i` is not in `mlp_only_layers`, `num_experts > 0` and `(i + 1) % decoder_sparse_step == 0`. Every other layer is dense. A missing `mlp_only_layers` is empty, a missing `decoder_sparse_step` is 1, and a missing `num_experts` is 0.
- The shim does not edit transformers. `prepare_config` writes 1-wide placeholder expert settings when the config has no experts at all, so the config validates. Inside `dense_layers()`, `Qwen4ExpTextDecoderLayer.__init__` runs and then swaps the MoE block of each dense layer for the model's own `Qwen4ExpTextMLP` (width `intermediate_size`, activation `hidden_act`). The class is restored on exit. The swap happens at construction, so it works for `from_pretrained` as well as for `cls(config)`.
- `reference_gate.py` uses the shim only when the snapshot's `model_type` is `qwen4_exp` or `qwen4_exp_text` and at least one layer is dense. A config with experts in every layer loads exactly as before. If the shim file is not next to the script, the load stops with a message that names the file. The load report gains a `dense_qwen4` entry with the dense layers and the provisional note.
- `ops_inventory.py` uses the same shim and adds `dense_layers` and `dense_note` to its output for a dense config.
- The stage gate fails a stage whose copy of `qwen4_dense.py` differs from the shipped file. A stage with no copy is not failed for it.
- Not covered: the multimodal wrapper `Qwen4ExpForConditionalGeneration` with `from_pretrained` was not run, only the text model class. Reading the vision tower was out of scope.

### Op inventory on a dense config

The run used the fixture's text dimensions as a flat `qwen4_exp_text` config (64 layers in the config, 4 traced, 64 positions, meta device, `--stub-data-dependent`), with the Flash-Next indexer keys added in a scratch copy so the QSA layer exists. The fixture itself has no indexer or n-gram keys, because no source gives them for a dense model. Without indexer keys, the multimodal form of the fixture fails to build (`TypeError` on a missing indexer setting), which is a finding in itself: the real config must say whether the full-attention layers use the indexer.

Result: no failures, 67 ATen ops, the QSA indexer stubbed (3 GDN layers and 1 QSA layer, all dense MLPs). Compared with the Flash-Next 4-layer trace in `2026-11-02-qwen4-new-ops.md`:

- Gone: `transformers.grouped_mm_fallback`, `aten.cummax`, `aten.histc`, `aten.index_put_`. That trace had a MoE router, experts and an n-gram embedding. This trace has none of the three. Which of them used which op was not traced one by one.
- Still present from the old unmapped list: `aten.diff`, `aten.eye`, `aten.unflatten.int` (the gated delta net).
- The `topk` in the trace belongs to the stubbed indexer. The dense MLP's own ops are `linear`, `silu` and `mul`.
- The op classes (`mapped`, `candidate`, `unmapped`) were not computed for this run, because `tt-orchard inventory` asks the shared venv's interpreter for the installed ttnn version, and this task imported no ttnn. The counts above are from the raw template output. Run `tt-orchard inventory <dir> --stub-data-dependent` to get the classes.

## What waits for the real config.json

- The layer schedule: whether the layers alternate 3:1 as assumed, whether `layer_types` is written, and whether `mlp_only_layers` or `decoder_sparse_step` appear.
- The FFN width (`intermediate_size`), and whether the MLP is gated with three projections.
- The MTP shape: the number of layers (`mtp_num_hidden_layers`), whether the MTP block is dense, and whether it shares the embedding. The reference loader ignores `mtp.*` keys on purpose, so the MTP head is not checked by stage 1.
- The quantisation format: BF16, FP8 or a packed 4-bit form, and how the scales are stored. The sizing and fit estimates assume BF16 until this is known.
- Whether the n-gram table (`ngram_*` keys, PLE) exists. Flash-Next carries a 102 GB table. The fixture assumes there is none.
- Whether the full-attention layers use the sparse indexer (`indexer_*` keys), and its budget.
- `hc_lowrank` at full size. The reports say 640; the real file decides.
- Whether the checkpoint's key names match the `qwen4_exp` class (the dense MLP keys are `mlp.gate_proj.weight`, `mlp.up_proj.weight` and `mlp.down_proj.weight` in the shim).

When the file exists, run `python3 -m orchard.hazards <config> --nearest <config> --dies 2` and `--dies 4`, then the stage 1 template, and replace the fixture.
