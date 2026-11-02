# Qwen3.8-Flash-Next graft study: what exists on Tenstorrent hardware and what tt-orchard can reuse for Qwen4 (2026-11-01)

Date: 2026-11-01. Branch `flash-next-graft-study`. Read-only study: no chip was opened, no `ttnn` or `ttml` import, no lease, no download of weights, no upload, no push to anyone else's repository.

Qwen3.8-Flash-Next (`Qwen/Qwen3.8-Flash-Next`, `model_type` `qwen4_exp`) is the preview of the Qwen4 architecture. Several people have already brought it up on Blackhole. This report says what they built, how it is packaged, what it measured, and which parts tt-orchard can take as they are. It ends with a ranked list of preparation for the day Qwen4 weights appear.

## 0. Sources, and what was and was not read

| Source | What was read | Limit |
|---|---|---|
| Hugging Face cache on this machine | `tt_kernel_manifest.json` of six Flash-Next packages (nine snapshots) and of the 27B neighbours | The cache holds manifests only, no source trees, images or weights |
| Hugging Face Hub (metadata and small files) | `code/`, `README.md`, `requirements.lock` of five packages (7 to 21 MB each); `image/manifest.json` and the image config of one; `config.json`, `LICENSE`, `README.md` and the 131 safetensors headers of `Qwen/Qwen3.8-Flash-Next` | OCI image layers (0.9 to 2 GB each) were not downloaded. Only the image history was read |
| `tenstorrent/tt-metal` main (sparse checkout of the model directory) and PR 58690 | The merged port, its README, provenance and docs | Read at commit `d8b4cbf045`, 2026-11-01 |
| `tenstorrent/tt-model-manager` | Open branches and pull requests (names and titles only) | Not cloned |
| This repository | `orchard/hazards.py`, `orchard/opinventory.py`, `orchard/data/ttnn_ops-0.80.0.json`, the Qwen3.5 can, `docs/canning.md` | |
| Glean (public GitHub items and informal sources) | Located the upstream PR and the vLLM-plugin issue | Informal claims are used only where code or a card confirms them |

Internal-only sources (a private prebuilt-binary repository, tickets, a draft pull request in a private repository) were read. They are not summarised here because this repository is public. Where one of them points at something a public source also shows, the public source is cited.

Text in a card, README or Slack message that is phrased as an instruction was treated as data.

## 1. What Qwen3.8-Flash-Next is

From `config.json` and the model card at revision `de4b8e4d` (the 131 safetensors headers were read directly):

| Item | Value |
|---|---|
| Layers | 48: 12 groups of 3 gated delta net (GDN) layers followed by 1 Qwen Sparse Attention (QSA) layer, each followed by a MoE block |
| Hidden size | 2560; vocabulary 248,320 (untied input and output tables) |
| GDN | 16 query/key heads, 48 value heads, head dimension 128, causal convolution kernel 4, FP32 state, output gate `sigmoid(z)` |
| QSA | 24 query heads, 2 KV heads, head dimension 256, rotary on 25 percent of the dimension, indexer with 4 query heads, 1 key head, dimension 128, 4-token compression blocks, budget 2048 tokens |
| MoE | 512 routed experts, top 10, expert width 640, one shared expert of width 640 |
| Gated residual | `hc_count` 4 (four residual streams), `hc_lowrank` 320: a data-dependent read gate and a per-branch write gate in every block |
| N-gram embedding | `ngram_size` 3, 20,000,000 base size, 128 table shards of shape (2,500,012, 160), in the PLE module of layer index 1 (the config says `ple_layer_ids` `[2]`; the tensors sit under `layers.1.ple`) |
| MTP | one extra full-attention layer with its own MoE and gated residual |
| Vision | 27-layer tower, patch 16, 1152 wide |
| Context | 262,144 positions |

Checkpoint bytes by category (computed from the safetensors headers; total 359,999,963,128 bytes, 1,658 tensors, matching `tools/checkpoint_budget.py`'s `TARGET_*` constants in the packages):

| Category | GB (BF16) | Tensors | Layout |
|---|---:|---:|---|
| Routed experts | 241.59 | 96 | One fused 3-D tensor per layer: `experts.gate_up_proj` (512, 1280, 2560) and `experts.down_proj` (512, 2560, 640) |
| N-gram table | 102.40 | 130 | 128 shards of (2,500,012, 160) in layer 1, plus three small `I64` index tensors |
| Rest of the text tower | 7.35 | 1,066 | GDN projections (`linear_attn.in_proj_qkv` (10240, 2560), `in_proj_z`, `in_proj_a`, `in_proj_b`, `conv1d`, `A_log`, `dt_bias`, `norm`, `out_proj`), QSA (`self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj`, `indexer.index_qk_proj`), router, shared expert, three gated-residual modules per block |
| Embedding and LM head | 2.54 | 2 | (248320, 2560) each |
| MTP | 5.21 | 31 | |
| Vision | 0.90 | 333 | |

The press figure of 125B parameters with 6B active leaves out the 51B n-gram table and the 4B MTP layer. The operator plan's 180B counts everything. Both numbers are right for what they count.

## 2. (a) Hardware shapes brought up, with numbers

`p300x2` means two P300 boards. Each P300 carries two Blackhole chips, so `p300x2` is four chips, the same as `p150x4` (four P150 boards). This machine is two P300c boards, four chips (`gozer status` lists 2 boards, 4 chips). The card text in `tt-hous/qwen3.8-flash-next-p300x2` says "two P300 boards (four Blackhole chips)".

Every number below comes from a package card or a README. This study re-measured none of them. "Claim" means the author states it and the evidence is not in the package or not seen here.

| Package (HF id) | Hardware | Runtime kind | Precision | Context | Decode | Prefill and first token | Accuracy evidence | Status |
|---|---|---|---|---|---|---|---|---|
| `tt-hous/qwen3.8-flash-next-p300x2` (cached manifest 2026-09-17; Hub head 2026-10-29) | 2 P300 = 4 chips, mesh (4,1) | `vllm-plugin`, vLLM 0.24.0, one sequence | BFP4 experts (lofi), BFP8 dense and KV, bf16 LM head | 262,144 | 24.3 tok/s up to 1k tokens, 13.8 above 4k | TTFT 340 ms at 128 tokens, 1.59 s at 1k, 7.99 s at 4k, 79 s at 32k, 338 s at 131k (388 to 642 tok/s) | top-1 agreement with a BF16 CPU reference: decode 96.3 percent (14 prompts, 1,386 positions), prefill 88.8 percent; GSM8K 92 of 100 (8 answers cut at the token cap) | Card table with run counts (one run per row). No log in the package |
| `sjettTT/qwen3.8-flash-next_p150x4` (snapshot 2026-09-29) | 4 P150; profile `c32k-quietbox2` for QB2 (card: validated 2026-09-21) | `tt-dit-server` (own chat server) | BF4 experts, BF8 dense, FP32 state, BF16 vision | 32k, 64k, 128k profiles | With MTP 4: sampled 59.0 tok/s (17.0 ms per token), greedy 68.1 to 111.4 by prompt type; four lanes 134.9 tok/s aggregate | TTFT 0.92 s at 560 tokens, 1.08 s at 2.1k, 7.67 s at 20.9k, 11.97 s at 31.7k (2,650 tok/s) | JSON record 96/96 against the CPU at every start; 13 pinned records replayed | Card with dated runs; acceptance replay runs at each boot |
| `DeAIafdeling/qwen38-flash-next` (2026-10-01) | 4 chips on a QB2, mesh (1,4) | `http-server` (chat server) | as above | 131,072 | 74.4 tok/s single stream (MTP 5, 64k stack, 4 runs, mean 74.2 +/- 0.5 percent); 70.7 tok/s aggregate on 4 lanes | TTFT about 41 ms; 1,995 tok/s on a 3.3k-token prompt, 604.5 tok/s cold | Boot refuses to serve unless the acceptance corpus lines up | Card claims; `QWEN38_HARDWARE_MODE=diagnostic_non_promoting` in the manifest |
| `DeAIafdeling/qwen38-flash-next-qb2-256k` (2026-10-04) | QB2, 4 chips | `vllm-plugin`, vLLM 0.26.0, one slot | as above | 262,080 and 131,008 | 59.27 +/- 0.22 tok/s at 256k (6-run protocol), 74.4 at 131k, both through the chat server | TTFT 0.076 s | Card: chunked prefill "tolerance-class against the CPU reference", prefix agreement only | "The vLLM adapter path on this package is declared, not yet independently benchmarked"; no MTP through the adapter |
| `changh95/qwen3.8-flash-next-p150x8` (2026-09-19) | 8 P150, TP8 and EP8 | `tt-dit-server` | BF4 experts, 107 GB expert cache | 32k | 20.4 tok/s per user at one user (TPOT 46.6 ms), 205 tok/s aggregate at 28 users | TTFT 363 ms at 128 tokens, 764 ms at 1k, 8.9 s at 16k | JSON 96/96 at every start | Card table from `vllm bench serve`; no MTP row |
| `tenstorrent/tt-metal` main, PR 58690 (merged 2026-10-29) | 4 P150b (QuietBox); QB2 listed "untested" | standalone chat server (`tools/qwen38_chat_server.py`); vLLM "in a later PR" | BF4 experts, BF8 dense | 32,768 | With MTP 4: 18.14 ms per token (about 55 tok/s) short, 16.06 ms after 21.8K tokens; plain 26.18 ms | TTFT slab prefill 3.3, 7.1, 12.4, 23.0 s at 4K, 8K, 16K, 32K | JSON 96/96; teacher-forced against Hugging Face over 8,416 positions: top-1 0.9525, KL 0.0528; GSM8K 58 of 64 (demo) | PR description with a validation table on a named tt-metal commit. The only entry with an upstream validation table, a named commit and a teacher-forced comparison |

Observations:

1. The package that runs the whole model on this machine's shape and is easiest to serve (`tt-hous`, vLLM with tool-call and reasoning parsers) is the slowest at 14 to 24 tok/s. The chat-server packages (`sjettTT`, `DeAIafdeling`) reach 59 to 74 tok/s with MTP and do not speak vLLM.
2. Three numbers describe the same weights at the same context on a QB2 and differ by 3x: 24.3 (`tt-hous`), 59.3 (`qb2-256k`) and 74.4 tok/s (`DeAIafdeling`). They use different runtimes and MTP settings. They are not comparable until the settings are matched.
3. Only `tt-hous` reports a task score (GSM8K 92 of 100). The upstream demo reports 58 of 64. Neither has run the benchmarks the Qwen model card reports. The `tt-hous` card says so itself.
4. `tt-hous` card limits worth keeping: no prefix caching, TTFT of about 2.6 ms per prompt token, chains of thought beyond about 25,000 to 30,000 reasoning tokens drift, requests with a seed, penalties, `logprobs` or structured output sample on the host (about 18 ms slower per token).

Inconsistencies found in the cards:

- `changh95/qwen3.8-flash-next-p150x8` describes the model as "51.2B MoE". 51.2B is the n-gram table. The MoE is the 125B.
- `sjettTT/.../README.md` and the port README call it a "256-expert MoE". `config.json` and the checkpoint tensors say 512, and `tools/checkpoint_budget.py` sets `TARGET_EXPERT_COUNT = 512`.
- Three licence strings name the same weights: `qwen-community-1.0` (`tt-hous`), `qwen-license-apache-2.0` (`qb2-256k`), none (`sjettTT`, `changh95`). Section 7 covers the licence.

## 3. (b) How each piece is implemented

Two independent implementations exist, and the upstream port is the cleaned-up first one.

- **Route A, hand-written C++ programs through `ttnn.generic_op`** (`sjettTT`, `DeAIafdeling`, `changh95`, and upstream `tt-metal` main). All model code lives under `models/demos/blackhole/qwen38_flash_next/`. `ttnn/fused/program.py` is "a thin layer over `ttnn.generic_op` for this model's fused decode kernels". Main carries about 20 kernel directories with 109 C++ and header files. No tt-lang is used anywhere in the tree (grep for `tt-lang`, `ttlang`, `import ttl` found nothing).
- **Route B, Python autoport through the `ttm-*` skills** (`tt-hous`). `models/autoports/qwen_qwen3_8_flash_next/tt/` holds `functional_decoder.py`, `fused_decoder.py`, `optimized_decoder.py`, `multichip_decoder.py`, `generator_vllm.py` and `doc/datatype_sweep/selected_precision_config.json`. It calls `ttnn.experimental.kda.*` ops (`gdn_decode_step`, `hc_inject`, `hc_mix_post`, `moe_route_topk`, `moe_sort_slabs`, `moe_swiglu`, `moe_weighted_sum`, `recurrent_gated_delta_rule`, `qkv_causal_conv1d_silu`, `sigmoid_gated_rms_norm`). Upstream `tt-metal` has a `kda` directory with only `sigmoid_gated_rms_norm`, `qkv_causal_conv1d_silu`, `recurrent_chunk_scan` and related scan helpers. The others (`gdn_decode_step`, `hc_inject`, `hc_mix_post`, `moe_route_topk`) were not found by `gh search code` in `tenstorrent/tt-metal`. The manifest records tt-metal `283ee08863` on branch `hous/qwen3.8-flash-next`, "dirty": true, "pushed": false, and that branch does not exist on GitHub. Route B cannot be rebuilt from public sources as published.

| Piece | Where and how | Custom ops or kernels | Host involvement |
|---|---|---|---|
| GDN, prefill | Upstream: `ttnn.transformer.chunk_gated_delta_rule` (tt-metal PR 57440) for every chunk recurrence. The earlier forks had six model-owned kernels for this; the PR note says "Greedy streams change at near-ties" after the swap | Public op | none |
| GDN, decode | `ttnn/gdn.py` (3,815 lines): TP4, Q/K heads split 4 and V heads 12 per chip, FP32 recurrent state in preallocated buffers; fused programs `gdn_step`, `gdn_pre_rows`, `gdn_post_rows`, `gdn_rows_scan` | Model-owned C++ through `generic_op` | none |
| QSA | `ttnn/qsa.py` (7,940 lines): index queries sharded one per chip and scored with `ttnn.experimental.indexer_score_dsa`; top-k with `ttnn.experimental.topk_large_indices`; sparse value kernel `ttnn.transformer.sparse_sdpa` with `K_DIM=512`, `v_dim=256`; `qsa_block` and `qsa_rows` fused programs; cache write through `ttnn.experimental.deepseek_prefill.update_padded_kv_cache` | DeepSeek sparse-attention ops that `tt-metal` already has, plus fused programs | none |
| Gated residual | `ttnn/gr.py`: residual held as (1, 4, 1, 640) per chip with the branch axis explicit; the 1/4 branch mean folded into the RMS gamma; fused programs `gr_read`, `gr_write`, `gr_fold`, `gr_recip_last`; the final mixer is its own module (`ttnn/final_mixer.py`) | Model-owned C++ | none |
| N-gram embedding (PLE) | `ttnn/ple.py`: "The 51.2B-parameter n-gram table remains on the host. Only the sixteen selected 160-wide rows are read for one token ... uploaded as four disjoint 640-wide ROW_MAJOR shards (1,280 B per device)". Projection, gating, dilated convolution and residual run on the mesh. `tools/prewarm_ple_table.py` pages the 104 GB table into the host page cache (13 s warm) | `ple` fused program | Lookup runs on the host; needs about 104 GB of page cache |
| MoE | `ttnn/moe.py`: `ttnn.experimental.moe_compute` with BF4 experts packed by `ttnn.experimental.moe_compute_utils` (the ring-aware packer); expert-parallel across the four chips; router tail, shared expert and combine as fused programs. Main's `moe_compute` takes at most 128 tokens per call, so a 2048-row prefill slab makes 16 calls per layer | Public op (patched in the forks, see section 6) | BF4 conversion runs on the host once (about 30 min on 4 P150, 1,448 s on a QB2) |
| MTP | `ttnn/mtp*.py`: one extra layer, drafts of 3, 4 or 5, exact acceptance for greedy and exact speculative sampling for sampled requests; the input mixer takes the four-branch gated-residual state as its hidden input | `mtp_accept` fused program | none |
| Sampling | `ttnn/device_sampler.py`: greedy picks one (value, index) per chip and reads back 8 scalars; the sampled path draws after a top-32 candidate row | `sampler_tail`, `greedy_tail` | Host sampler for requests with penalties, seeds, `logprobs` |
| Vision | `ttnn/vision*.py`, `mrope.py`, `vision_splice.py`: resident tower at BF16, eight row buckets run once at start; `image_url` data URLs only; 3-axis interleaved MRoPE sections `[11, 11, 10]` | | Image preprocessing on the host |
| Route B (`tt-hous`) | Text only: "requests with image or video content are rejected". MTP as lossless k=1 speculative decoding (`tt/spec_decode.py`), eager, with the n-gram history snapshotted and rolled back on rejection | `kda.*` ops (unpublished) | PLE rows assembled on the host with a row cache |

Comparison with Qwen3.5 (the GDN model tt-orchard already cans):

| GDN setting | Qwen3.5-35B-A3B (`pkgsrc/can/qwen35/config.json`) | Flash-Next |
|---|---|---|
| Hidden size | 2048 | 2560 |
| Layers | 40 | 48 |
| Q/K heads, dimension | 16, 128 | 16, 128 |
| V heads | 32 | 48 |
| Convolution kernel | 4 | 4 |
| Output gate | `silu(z)` | `sigmoid(z)` (`ttnn/gdn.py` docstring) |
| Full-attention interval | 4 | 4 (and the full-attention layer is QSA) |
| Experts, top-k | 256, 8 | 512, 10 |
| Vocabulary | 248,320 | 248,320 |

The recurrence is the same family. The Qwen3.5 can's decode path steps the state in FP32 "(kda.recurrent_decode)" per a comment in `model.py` of its models wheel. Whether the Qwen3.5 wheel and Flash-Next's `gdn_step` share any kernel is not verified. The Flash-Next differences that matter for reuse: 48 value heads (4/12 split per chip works; a 2-chip split gives 24 per chip), the sigmoid output gate, and a hidden size of 2560.

ttnn 0.80.0 name coverage (this repository's name dump, `orchard/data/ttnn_ops-0.80.0.json`, 821 names; a name match that makes no claim about shapes). Of the `ttnn.*` calls the upstream port makes, 124 are in the dump. The ones that matter are all there: `experimental.indexer_score_dsa`, `experimental.topk_large_indices`, `transformer.sparse_sdpa`, `transformer.chunk_gated_delta_rule`, `experimental.moe_compute`, `experimental.deepseek_moe_fast_reduce_nc_fused`, `generic_op`. Two are missing: `experimental.deepseek_prefill.update_padded_kv_cache` (the QSA cache write) and the Python module `experimental.moe_compute_utils` (the BF4 packer). The PyPI wheel is older than tt-metal main of 2026-10-29, and the port was validated on main (`207549a61f`), so a name match here does not mean the wheel runs the port.

## 4. (c) Packaging forms, and what a Qwen4 package changes

All Flash-Next packages are v5.1 container manifests (`schema_version` "5.1"). No Flash-Next thin v6 bundle exists. The 27B neighbours use the same schema (`tt-hous/qwen3.8-27b-p150x4`, `vllm-plugin`, 131,584 context; `changh95/qwen3.8-27b-pd4x4-p150x8`, `tt-dit-server`, a prefill/decode split across 8 chips).

| Field | What the Flash-Next manifests put there |
|---|---|
| `weights` | `repo_id` `Qwen/Qwen3.8-Flash-Next` with a pinned `revision` (`f5d08274`, later `de4b8e4d`); weights are a pointer, never in the image |
| `container.kind` | `vllm-plugin`, `tt-dit-server` (an ASGI `app`) or `http-server` (a `command` and a `ready_line`) |
| `container.runtime` | vLLM 0.24.0 or 0.26.0, the plugin sha (one is `dirty`), `extra_models_dir`, `overrides` (`transformers==5.16.1`), `requirements.lock` |
| `container.serve` | `hardware` (`p300x2`, `p150x4`, `p150x8`), `mesh_device`, limits, `additional_config.tt` (`fabric_config` `FABRIC_1D`, `trace_region_size`, `l1_small_size`, `sample_on_device_mode`), `args`, `env` (`QWEN38_*`, `TT_MESH_GRAPH_DESC_PATH`) |
| `serve_profiles` | Context rungs and mesh variants: `c32k`, `c64k`, `c128k`, `c32k-lanes4`, `c32k-quietbox`, `c32k-quietbox2`, `long-context`, `qb2-256k`, `qb2-131k`, `batch28` |
| `verify` | Python assertions that run in the image: the mesh descriptor shipped, the acceptance prompts shipped, the model class imports, `ttnn.experimental.kda` has the op |
| `built` | tt-metal `sha`, `describe`, `dirty`, `branch`, `remote`, `pushed`, `code_sha256`, image digest |
| `card` | description, performance tables, limitations, risks, licence |

Each package's `code/` directory is the model tree copied into the image's `/opt/tt-metal`. The image itself copies a prebuilt `/opt/tt-metal/build` and `ttnn/` from a patched tt-metal (the image history of `qb2-256k` shows `COPY /opt/tt-metal/build ...` then `COPY code/ /opt/tt-metal/`). So the compiled ops with their C++ patches exist only inside the image layers.

A thin v6 bundle for this model needs what the Qwen3.5 can needed (`docs/canning.md`): a `ttnn` wheel carrying the ops (here the patched tt-metal build or main's build), a models wheel, an ASGI adapter for `kind = "tt-dit-server"` (the `sjettTT` repository has `tools/qwen38_asgi.py`; main has the chat server but no ASGI wrapper), a pinned SFPI (section 6), a model directory step for the checkpoint, and a pointer to weights. `tt-model package-thin` is documented as beta and unsupported.

What a Qwen4 package would change:

| Piece | Model-name-specific | Generic |
|---|---|---|
| Module path `models/demos/blackhole/qwen38_flash_next`, env prefix `QWEN38_*`, architecture override `TTQwen4ExpForConditionalGeneration` | yes | |
| Checkpoint pins: `checkpoint.py` `PINNED_CHECKPOINT_REVISION`, per-shard header digests, `checkpoint_budget.py` `TARGET_TENSOR_COUNT` 1658, `TARGET_CHECKPOINT_BYTES`, `TARGET_EXPERT_COUNT` 512. The server "refuses a checkpoint that differs, with the digests printed" | yes (every pin changes with new weights) | |
| Acceptance corpus: 12 greedy prompts and `cpu-oracle-two-step.jsonl`, regenerated from the CPU oracle | yes | the replay mechanism, `--require-json-96` |
| BF4 expert cache key (`resident-experts-f5d08274-p300-tp4-ep4-bfp4-v1`) | yes: keyed by checkpoint and converter source | the converter |
| Mesh descriptors (`qb2_p300_1x4_line_mesh_graph_descriptor.textproto`), hardware profiles | | yes |
| Run directory, evidence files, stall watchdog (exit 1 in the standalone server, 75 in the p150x8 bundle), runtime admission | | yes |
| Layer geometry in `config.py` (48 layers, 12 QSA layers, 512 experts, 2560 hidden, per-chip splits 6, 12, 128) | yes, if Qwen4 changes any of them | |
| GSM8K benchmark tool, `tools/report.py` | | yes |

Weights remap: the checkpoint stores the experts fused per layer, so no per-expert tensor fusion is needed (contrast with the Sage-A1.5 row in `2026-10-29-next-round-plan.md`, where 31,333 per-expert tensors need fusing). A Qwen4 checkpoint with a different key layout would need `tt/*.py` `from_checkpoint` edits, which sit beside the pinned names above.

## 5. (d) Weight handling and fit

Conversion: on first boot the server converts the routed experts of all 49 MoE layers (48 plus the MTP layer) from BF16 to BF4 on the host, one layer at a time (about 10 GB of host tensors in flight, 27 s per layer on 4 P150, 1,448 s on a QB2 per `README.md` of `DeAIafdeling/qwen38-flash-next`), packed by the ring-aware packer into a compact layout, and writes a 65 to 69 GB cache (`tt-hous`: about 64 GiB). Dense weights go to BF8. The cache is "keyed by the checkpoint and the converter's sources, not by the tt-metal revision". A BF4 cache is verified against the converter at every boot. `tools/bf4_host_packer.cpp` is the native packer. `tools/checkpoint_budget.py` and `tools/safetensors_metadata.py` print the budget and list tensors.

Disk and host memory per the READMEs: 360 GB checkpoint, 65 to 69 GB BF4 cache, about 23 GB of component caches for the first context and 10 GB for each other context, about 1.3 GB of kernel cache. Host RAM 64 GB is "comfortable"; the 104 GB n-gram table is read through the page cache. The optional CPU oracle needs 170 to 240 GB of RAM. This machine has about 249 GB.

Fit, as arithmetic from the table in section 1 and the cards (all estimates):

| Shape | Fits Flash-Next? | Reason |
|---|---|---|
| 1 chip, 2 chips (one P300) | No | BF4 experts alone are about 68 GB (241.59 GB x 0.28125, the BF4 tile is 576 bytes for 1,024 values); two chips hold about 58 to 64 GB with no space for dense weights, KV and traces |
| 4 chips (P150x4, P300x2, this machine) | Yes | Every working package at this shape. The sjett card reports 1.5 to 1.7 GB free per DRAM bank after the captures at 32k, 64k and 128k; if a chip has 8 DRAM banks (my assumption; no source read) that is about 12 GB free per chip, so roughly 18 GB of each 32 GB chip is resident (unsupported estimate) |
| 8 chips (P150x8) | Yes | `changh95` runs TP8 and EP8 with 28 users |

Per-chip memory: the cards and this repository's logs treat a Blackhole chip as 32 GB (`docs/run-logs/2026-10-29-overnight-minicpm3-4b.md`: "Blackhole p300 board has two chips with 32 GB each"), and the next-round plan says capacity "is arithmetic on parameter counts" and not stated as a measurement. The usable part is lower: a P300 board reaches `/dev/tenstorrent` as two chips, and the allocator reserves space for traces and caches. I did not read a usable figure from a live allocator.

What a Qwen4 size would need on this machine: the cards give no figure for the overhead (KV, state, traces, caches) that a larger model would still need, so this study states no cutoff for routed-expert size. The only supported figures are Flash-Next's own: about 68 GB of BF4 routed experts runs on four chips, with 1.5 to 1.7 GB free per DRAM bank after the captures on the sjett packages. A Qwen4 with more routed parameters needs the headroom recomputed from its own headers and a measured allocator reading (proposed issue A).

The four-chip fit has a consequence for tt-orchard. A Flash-Next-class model uses both boards. While it is loaded, no board is free, so every hardware stage parks it, as the Qwen3.8-27B-on-four-chips layout already does (`config/tiers.qb2-27b.toml`).

## 6. (e) Known failure modes, patches and pins

Version pins:

| Package | tt-metal | vLLM and plugin | transformers |
|---|---|---|---|
| `tt-hous` (2026-09-17) | `283ee08863` (v0.77.0-dev20260818-415, dirty, branch not pushed) | 0.24.0, plugin `a48857ac` (dirty) | 5.16.0 (lock) |
| `sjettTT` c32k vLLM snapshot | `5738626b` | 0.26.0, plugin `bef89e42` | 5.16.1 |
| `sjettTT` unified server | `02e1453c` (branch `q38-serve-unified`, pushed) | n/a | 5.16.1 |
| `DeAIafdeling` base | `22aea2f1` (`v2-mirror-40`) | n/a | |
| `DeAIafdeling` 256k | `fe2de7d9` (`v2-mirror-72`, dirty) | 0.26.0, plugin `3506696c` | |
| `changh95` p150x8 | `b8733f1f` | n/a | 5.12.1, safetensors 0.8.0 |
| upstream port | validated on main `207549a61f`; README: clang-20, "the default SFPI toolchain", transformers 5.16.1 | vLLM adapter "in a later PR" | |

The reference venv of this repository has transformers 5.19.0, which contains `qwen4_exp` (`Qwen4ExpForCausalLM`, `Qwen4ExpTextGatedDeltaNet`, `Qwen4ExpTextQSAIndexer`, `Qwen4ExpTextNGramEmbedding`, `Qwen4ExpTextGatedResidual`). Stage 1's CPU reference can already load the architecture class.

tt-metal patches the forks carry that main does not (`DeAIafdeling` README, section 1; `changh95` `docs/REBASE_SURVEY.md`): `moe_compute` fixes (empty-rank metadata ownership, idle-expert combine sync skipped at B=1, source buffer double counting, `local_combine=True`, a 512-token block), all-gather and fabric guards, four BF4-cache loading commits, and an extended `indexer_score` op (`per_row_slots`, `kv_len_tensor`). Upstream rewrote `indexer_score` in six commits after the forks branched. Two of the fork's `moe_compute` changes are merged (PRs 57564 and 57448). Main's port drops four fused kernels and uses the public GDN op, so its greedy streams differ from the forks at near-ties.

Compiler constraints: `docs/REBASE_SURVEY.md` records that moving from SFPI 7.74.0 to 7.79.0 "recompiles every SFPU kernel" and forces a new bitwise baseline of the acceptance record (exp, silu, sigmoid, reciprocal, rsqrt, topk tie order). The Qwen3.5 can in this repository already ships SFPI 7.74.0 in its ttnn wheel because 7.89.0 on the host broke the kernels (`2026-10-30-qwen35-thin-bundle.md`). Open `tt-model-manager` pull request 200 ("optionally provision SFPI into the ttnn wheel at install") and branch `feat/sfpi-autoprovision` address the same problem.

Failure modes recorded in the packages:

- First boot takes 30 to 60 minutes (expert conversion) and kernels compile for tens of minutes (`DeAIafdeling` card).
- "A SIGKILLed container can leave the mesh dirty; use `tt-model stop` (SIGTERM), and reset the chips if a later boot wedges at the first large host->device DMA" (card, `risks`). Gozer's `reset <lease>` covers the reset.
- A boot refuses to serve if profiler variables leak into the environment, if the worktree is dirty, or if the acceptance corpus is absent (a stated design choice).
- Stall watchdog: `--stall-seconds 300`; a wedge ends the server with exit status 1 (75 in the p150x8 bundle) and expects a supervisor to restart it. `docs/SERVER.md` lists the clock rules.
- The `qb2-256k` card: max one sequence, no MTP through vLLM, MTP depth 6 is "a measured NO-GO (stay at 4-5)", chunked prefill agrees with the CPU for a prefix and then diverges.
- Acceptance numerics: the JSON record matches 96/96. The other records leave the CPU stream after 2 to 75 tokens; "a replay that leaves earlier than the pinned table is a regression". The pinned tables are the 32k context's; at 128k one slab record diverges earlier ("an open numerics item").
- Reasoning loops: the `tt-hous` card warns that very long chains of thought drift and lose the answer.
- `moe_compute` takes at most 128 tokens per call on main, which makes slab prefill slower than the forks.

## 7. (f) Licences and terms

| Item | Terms | Evidence |
|---|---|---|
| Weights `Qwen/Qwen3.8-Flash-Next` | Qwen Community License 1.0: permission to use, copy, modify, distribute, host and fine-tune, with the notice kept. Two conditions: a product with more than 100,000,000 monthly active users or more than US$ 20,000,000 monthly revenue must show the model name prominently; a licensee that runs a "Model as a Service" or "AI Work Assistant" business needs a separate licence from Qwen before any commercial use, except internal use that does not make the software, its outputs or its capabilities available to a third party | `LICENSE` of the weights repository (3,235 bytes) |
| Port code in the packages | Apache-2.0 (card texts; `PROVENANCE.md` upstream: "Apache-2.0, as the rest of tt-metal"; one header file also names a third-party copyright) | `PROVENANCE.md`, cards |
| Upstream port provenance | Ported from `sjettTT/tt-qwen-3.8-flash-next` (Apache-2.0 on GitHub); the source's first commit names an AI tool as author and committer | `PROVENANCE.md` |
| Image contents | An Ubuntu 22.04 base, OpenMPI 5.0.7 (ULFM build), a Python venv, the patched tt-metal build. The licences of the bundled compiler (SFPI) and OpenMPI were not reviewed here | image history |
| Licence strings in cards | `qwen-community-1.0`, `qwen-license-apache-2.0`, or none | cards |

For tt-orchard: a can or card for Flash-Next or Qwen4 carries weights as a pointer, names the licence as the Qwen Community License 1.0 with the link, and states the Model-as-Service condition. The `qwen-license-apache-2.0` name in one card suggests the weights are Apache-2.0; they are not. Serving the model to third parties through a hosted endpoint is the case that needs a separate licence from Qwen. Running it for the operator's own bring-up and tests is internal use. The licence text for Qwen4 is not known and must be read when the weights appear.

## 8. What tt-orchard already has and what is missing (g)

Measured in this study, with no device:

1. **Stage 0 hazard scan** (`orchard/hazards.py`, issue 183) on the real Flash-Next config against the Qwen3.5 config as nearest bundle. Before this branch it reported mixers `['attention', 'linear_attention']` and two hazards (`routed_experts_above_256`, `router_top_k_above_nearest`). It missed the sparse attention, the gated residual and the n-gram embedding because the config spells them `indexer_*`, `hc_count` and `ngram_size`. After the change in this branch it reports five mixers and three more hazards, each with a stage 2 check. With 2 dies it also reports `vocab_per_die_above_65536` (124,160 columns per die); with 4 dies the figure is 62,080 and the hazard does not fire.
2. **Op inventory** (`tt-orchard inventory`) on a 2-layer meta-device trace of `Qwen4ExpForCausalLM` (transformers 5.19.0, torch 2.14.1+cpu): 36 ops `mapped`, 1 `composable`, 10 `verified`, 25 `candidate`, 7 `unmapped` (`aten.unflatten.int`, `transformers.grouped_mm_fallback.default`, `aten.cummax.default`, `aten.eye.default`, `aten.histc.default`, `aten.index_put_.default`, `aten.diff.default`), 2 `ignored`. The same trace with 4 layers fails at the QSA layer: `GuardOnDataDependentSymNode ... (u0//4) > 0` at `modeling_qwen4_exp.py:728`. The reference indexer loops over every query position with `torch.nonzero` and a data-dependent count of complete blocks, so `torch.export` cannot trace it. The inventory therefore sees the GDN, MoE, gated-residual and n-gram layers but never the QSA layer.

Reusable as they are:

| Item | Why |
|---|---|
| Stage 0 hazard scan, stage 1 reference venv (transformers 5.19.0 loads `qwen4_exp`) | Config-only and CPU-only |
| Park and restore, lease handling (`gozer`), `config/tiers.qb2-27b.toml` shape | A 4-chip model parks the same way the 27B does |
| Thin-bundle can machinery (`orchard/can.py`, `tt-dit-server` kind, boot check, gate, SFPI in the wheel) | Qwen3.5 is the worked example; the SFPI 7.74.0 fix applies to ports built on the 2026-09 tt-metal base |
| Weights as a pointer, `[model_dir]` and `prepare_model_dir.py` | The checkpoint needs no key fusion |
| Sizing tool, decode-floor report (`orchard/decode_floor.py`) | Give a floor for 4 dies at BF4 without a chip |

Must be adapted:

| Item | Change |
|---|---|
| Container weight swap (stages 2 and 4 via a `four_chip_package`) | Flash-Next servers pin the checkpoint by per-shard digest and key the BF4 cache by revision, so a swap to other weights is refused or rebuilds the cache. The swap probe has to read the pin and the cache key |
| Package scrub and card gate | Add the Qwen Community License 1.0 terms and check that a card's licence name matches the weights repository's |
| Op inventory | Needs a way to trace past the QSA indexer (a static stub for the block selection) |
| Hardware facts | Per-chip DRAM is an assumption in the docs; a residency estimator from safetensors headers is missing |

Missing:

| Item | Note |
|---|---|
| A vLLM adapter for Flash-Next in tt-metal | "in a later PR". Until then a thin bundle is `tt-dit-server`, which has no tool-call or batching front end of its own beyond the chat server |
| A Flash-Next result on this machine's silicon from any public source other than the `sjettTT` QB2 profile (validated 2026-09-21 per its card) | The upstream README lists QB2 as untested |
| A task benchmark beyond GSM8K | None of the packages has run the Qwen card's own benchmarks |
| Any Flash-Next support in the stock-runtime route or the nearest-model logic | `grep` finds no reference to `qwen4` or `flash-next` in `orchard/` |

## 9. Ranked preparation for the day Qwen4 weights appear

Effort is in working days (my estimate, written before the work). Risk is the chance the item does not yield what it says.

| Rank | Item | Now (no hardware) | Later (hardware) | Effort | Risk |
|---:|---|---|---|---:|---|
| 1 | Read the new `config.json`, the transformers class diff against `qwen4_exp` and the safetensors headers; run `python3 -m orchard.hazards` against the Flash-Next fixture; compute bytes by category with `huggingface_hub.get_safetensors_metadata` | Yes. The scan and the fixture are in this branch | none | 0.5 | Low |
| 2 | Choose the route: take main's `models/demos/blackhole/qwen38_flash_next` as the base, change the pins and constants in section 4, regenerate the acceptance corpus | Yes (read and plan) | One boot and the acceptance replay | 2 to 3 with hardware | Medium. The CPU oracle needs 170 to 240 GB of host RAM on a host that has about 249 GB, so nothing else can run beside it |
| 3 | Residency estimator: bytes by category from headers, a quant policy, chips needed, free space per chip. Use it to say 4 chips or 8 before any boot | Yes | One allocator reading to replace the 32 GB assumption | 1 | Low |
| 4 | Make the container-swap probe read digest pins and cache keys (section 8) | Yes, with fake packages | One swap against a real package | 2 | Medium |
| 5 | A static QSA stub for the inventory trace so the QSA layer's ops appear | Yes | none | 1 | Medium. The stub must keep the op mix of the real indexer |
| 6 | A Qwen4 coder layout (`bringup` and `tiers` files for four chips, large tier only, CPU tier for parked stages) after a role-fit run | Template only | Role-fit run | 0.5 template, 1 with the run | Medium. A model that has not passed the role-fit test should not become a default |
| 7 | Thin bundle for the chosen route: ASGI adapter over the chat server, pinned ttnn wheel with its SFPI, boot check | Spec skeleton only | Boot check and a bench against the package numbers in section 2 | 5 to 8 | High. Needs a ttnn build that has the ops, and the SFPI choice changes the numerics baseline |
| 8 | Licence gate for cards (Qwen Community License terms; name consistency) | Yes | none | 0.5 | Low |
| 9 | Baseline run of the closest published package on this machine, to compare against the cards | Plan only | One session, 1 to 2 hours after a first boot of 30 to 60 minutes | 1 | Low value if Qwen4 changes the geometry; the study avoided repeating it |

What this branch does for item 1: `orchard/hazards.py` now reads the Qwen4-style keys, a real-config fixture and tests are added, and the table has a new row for the n-gram embedding. Items 2 to 9 are not done here.

## 10. Proposed GitHub issues (text only; none filed)

Each body is written for the public repository: no machine paths, host names or tokens.

**Issue A. Residency estimator: chips needed from safetensors headers and a quantization policy**
Body: Stage 0 knows parameter counts but not bytes by category, and the documents treat a 32 GB chip as an assumption (`docs/run-logs/2026-10-29-next-round-plan.md`, "Per-chip DRAM"). For Qwen/Qwen3.8-Flash-Next the headers give 241.59 GB of routed experts, 102.40 GB of n-gram table (host-side), 9.89 GB of dense text weights, 5.21 GB of MTP and 0.90 GB of vision (total 359,999,963,128 bytes). Add `orchard/residency.py`: read headers with `huggingface_hub.get_safetensors_metadata`, group tensors by name pattern, apply a policy (BF4 for routed experts at 0.28125 of BF16, BF8 for dense), and report per-chip residency for 1, 2, 4 and 8 chips with a stated reserve. Label every figure as arithmetic. Done when tests cover a fake index and the Flash-Next numbers above, and the report names the allocator figure it still needs.

**Issue B. Container weight swap must read the checkpoint pin and the cache key**
Body: The Flash-Next container servers verify every shard header and `config.json` digest and refuse a different checkpoint (`tools/checkpoint_budget.py`, `tools/verify_checkpoint_files.py`, `PINNED_CHECKPOINT_REVISION`), and the BF4 expert cache name carries the revision (`resident-experts-<revision>-p300-tp4-ep4-bfp4-v1`). A weights swap through stages 2 and 4 would fail at boot or rebuild a 65 to 69 GB cache. Teach the swap probe to list pins and cache keys found in a package's `verify` lines, `serve.env` and `code/`, and to report "swap refused by the package" before any chip time. Related: issue 73 (a container that ignores `MODEL_WEIGHTS_DIR`).

**Issue C. Inventory cannot trace sparse-attention layers**
Body: `tt-orchard inventory` on `Qwen4ExpForCausalLM` (transformers 5.19.0) traces two GDN layers and fails on the first QSA layer with `GuardOnDataDependentSymNode` at `modeling_qwen4_exp.py:728`. The reference indexer builds block lists with `torch.nonzero` per query position. Add an option that replaces a named module's forward with a static stub that has the same op mix (matmul, relu, sum, topk over a fixed block count) so the QSA layer appears in the inventory, and label the stub's ops "stubbed". Done when a 4-layer Flash-Next config traces and the QSA ops are listed with that label.

**Issue D. Qwen4 coder layout, after a role-fit run**
Body: When Qwen4 weights and a package exist, add `config/bringup.qb2-qwen4.toml` and `config/tiers.qb2-qwen4.toml` for four chips with only the `large` tier on chips and the CPU tier for parked stages, mirroring `tiers.qb2-27b.toml`. Gate it on `python3 -m orchard.rolefit` results for the package. Add a `CODERS` entry in `orchard/setup_machine.py` and update `tests/test_settings.py`'s expected presets in the same change. Not before a role-fit run: the Flash-Next packages on file have not been tested as an agent model.

**Issue E. Thin bundle for a Flash-Next-class model (`tt-dit-server`)**
Body: Spec `config/can.qwen4.toml` with `kind = "tt-dit-server"`, an ASGI adapter over `tools/qwen38_chat_server.py`, a ttnn wheel built from tt-metal main with its SFPI bundled (as in `config/can.qwen35.toml`), `[model_dir]` pointing at the pinned snapshot, and a boot check that sends the acceptance JSON prompt and requires 96/96. Open questions: the wheel needs `experimental.deepseek_prefill.update_padded_kv_cache` and `moe_compute_utils` (absent from the 0.80.0 name dump); the SFPI version changes the acceptance baseline; the licence terms of the bundled compiler were not reviewed.

**Issue F. Package gate: licence name must match the weights repository's licence**
Body: Three cards name the licence of the same weights differently (`qwen-community-1.0`, `qwen-license-apache-2.0`, none). Add a gate check to `orchard/can.py` and `package_card.py` that reads the weights repository's `LICENSE` name and requires the card's licence field and `licensing` paragraph to match it, and that a licence with a Model-as-a-Service clause is quoted on the card.

## 11. Method notes and limits

- No part of the Hugging Face cache beyond manifests was available, so every code reading used the Hub copy of `code/` from the current head of each repository. The `tt-hous` Hub head (2026-10-29) is newer than the cached manifest (2026-09-17); its README and card were read at the head and its manifest at the cached snapshot. They may differ in details.
- `tt-hous/qwen3.8-flash-next` and `tt-hous/qwen3.8-flash-next-p300x2` list the same files at the same revision on the Hub.
- A telemetry snapshot (`tt-smi -s`) and `gozer status` were run once to confirm the board count and host memory. Neither opens a device for compute.
- Section 2's numbers are quoted from cards and one README. A card number was not checked against logs, because no logs ship in the packages.
- The ttnn name coverage in section 3 compares names only.
