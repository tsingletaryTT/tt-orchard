# tier-plan against the Hy4-flash bundle (2026-11-25)

Hy4-flash is Tencent's 220B mixture-of-experts model. Raahem's bundle `raahemnabeel/hy4-flash` serves it on four dies (p300x2) with 8 users sharing a
202,752-token context pool. The card says 220B parameters in total, 11.3B active per token, and a 63.5B Engram n-gram memory held in host RAM.
Nothing here was run on a chip. The config is the public `config.json` of `tencent/Hy4-flash` at revision b2498eba (committed as
`tests/fixtures/model-configs/hy4-flash.pinned.config.json`). The upstream main config differs in one key, `engram_fp8_embedding` (false on main, true at
the pinned revision), so a plan has to use the revision the bundle pins.

## Before this change

`tt-orchard tier-plan tencent/Hy4-flash --chips 4` stopped at once:

```
unknown family: the layer type 'deepseek_sparse_attention' is not one the floor can count (known: linear_attention, mamba, full_attention, sliding_attention, attention)
```

Three more things were wrong once that was bypassed (found by recomputing the card's figures from the config):

| What | Before | After | Source of the right value |
| --- | --- | --- | --- |
| Layer types | 12 `deepseek_sparse_attention`, 30 `sliding_attention`: refused | both counted as multi-latent attention, 32 and 64 query heads | `num_attention_heads` 32, `swa_num_attention_heads` 64, same latent shapes (the bundle's `tt/config.py` asserts that) |
| Head width printed by the hazard facts | 64 (the `head_dim` key) | 256 | `qk_nope_head_dim` 192 + `qk_rope_head_dim` 64 = `qk_head_dim` 256; the key names the rope part |
| Routed expert parameters | 297.2B (each expert 3 x 3072 x 1536) | 148.6B (each expert 3 x 1536 x 1536) | `routed_expert_hidden_size` 1,536: the experts work in a latent, with a 3072-to-1536 projection before them and a 1536-to-3072 projection after (the bundle's `tt/moe.py`) |
| Engram memory | not mentioned | named, 63.4B parameters and 60.2 GiB | `engram_layer_head_vocab_sizes`, `engram_n_embed_per_ngram`, `engram_n_head_per_ngram` |

The doubled expert size moved every host-memory figure: with it, all-bfp4 on four chips was "fits with host experts" (308 of 512 experts per layer on the
chips, 165.7 GiB on the host). With the latent width it is "fits on chips" (22.11 GiB per chip), which is what the bundle does.

## After this change

```
local hy4-flash.pinned.config.json: 42 layers, 41 with 512 routed experts (top 16); 5.05 B always-resident and 148.58 B routed-expert parameters [config-derived]
combination           chips  verdict                 GiB/chip  experts on chips  host GiB
all bf16                  4  fits with host experts     25.76        121 of 512     286.8
all bfp8                  4  fits with host experts     25.82        290 of 512     157.0
bfp8 + bfp4 experts       4  fits on chips              24.46        512 of 512         -
all bfp4                  4  fits on chips              22.11        512 of 512         -
```

The plan lists what it leaves out: hyper-connections, the sparse-attention indexer, the Engram memory (63.4B parameters, 60.2 GiB as FP8 rows with a bf16
scale each, kept in host memory by the published bundle; the plan does not place it) and the vision tower.

## Recomputed numbers against the card

| Figure | Card | Recomputed from the config | Note |
| --- | --- | --- | --- |
| Total parameters | 220B | 217.0B (5.05B always resident, 148.6B experts, 63.4B Engram) | the 3B left is the vision tower, the indexer, the hyper-connection weights and the drafter, which the count leaves out |
| Engram parameters | 63.5B | 63.36B (660,006,416 rows x 96 values) | matches |
| Engram size in host memory | about 60 GiB | 60.24 GiB (one byte a value plus a bf16 scale a row) | matches |
| Parameters read per token | 11.3B active | 9.32B (8.95B without the output head) | about 2B unexplained: the indexer, hyper-connections, the Engram projections and the drafter are not counted |
| Routed experts | 148.6B at bfp4 = 77.8 GiB | 148.58B | matches |

The card's 11.3B and the count's 9.3B differ by about 2B, and the config does not say where they go. That stays open.

## KV cache: the flat allowance against an estimate from the config

`--context 202752 --users 8` gives a model-aware estimate for a latent-attention model. For Hy4-flash it is 12 full-attention layers (the
`deepseek_sparse_attention` ones) x 576 values x 2 bytes (bf16) x 202,752 tokens = 2.610 GiB, plus 30 sliding-window layers x 512 rows x 1,152 bytes
for each user (17.7 MB a user), which is 2.627 GiB for one user and 2.74 GiB for eight. The latent row is shared by all heads, so a split over heads
repeats it on every die and the estimate is not divided by the die count (the flat 4.0 GiB was divided by four).

The study of the bundle's code put the cache at about 2.9 GiB a die (15 KB a token a die). The 0.2 to 0.3 GiB between that and 2.63 GiB is the
sparse-attention indexer's group keys, which the config-only estimate leaves out and says so. The estimate changes one row of the four-chip plan:
`bfp8 + bfp4 experts` no longer fits with the experts split in the old layout (24.46 GiB a chip against a budget of 24.15 GiB) and is placed in the
tensor-parallel layout (21.32 GiB a chip), which still fits.

## Against the running container (added 2026-11-25 evening)

The bundle was later brought up on four dies here. The container's own memory lines (its serve log, not a tt-orchard measurement) report the DRAM used
per die: 29.76 GiB after the model load, 30.50 GiB after the warm-up passes and 30.79 GiB after trace capture, with 1.04 GiB free of 31.83 GiB. The plan's
per-die figure for the closest precision row (`bfp8 + bfp4 experts`) is 24.46 GiB with the old layout and 21.32 GiB in the tensor-parallel layout, so the
plan is 6.4 to 9.5 GiB below what the container used. The verdict "fits on chips" matched the run. The missing line items are listed in issue 845 (the
drafter, the vision tower, the trace region, the long-prefill scratch, the indexer, the hyper-connection and Engram weights). This change does not
calibrate the plan against that run; it fixes what the config can say (layer types, the latent expert width, the Engram size, a KV estimate).
The Engram memory as a host-resident table in the residency categories is issue 844, and the first-start preflight is issue 843; neither is changed here.

## What is not done here

- The Engram memory is named and sized. The plan does not place it, because it has no row for host-resident embedding tables.
- The two hazard rows `routed_experts_above_256` and `mla_flash_decode_needs_core_grid` stay in the list. They describe the stock ttnn ops. The bundle's card
  claims its own gather kernels serve the 512 experts and its own latent attention serves the model; nobody here has measured that, so the scan's notes
  say so and keep both rows.

The bundle is Raahem's work; the model is Tencent's under the tencent-hy-license (see the card); nothing was downloaded except a 10 KB config file.
