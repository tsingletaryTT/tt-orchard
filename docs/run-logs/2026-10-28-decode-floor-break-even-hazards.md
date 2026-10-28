# Decode floor, speculative break-even and the dimension hazard scan (issues 181, 182, 183)

Date: 2026-10-28. Hardware: none used. Everything here is computed from model configs and from numbers already on record.

Four small modules, each with tests: `orchard/model_config.py` (reads and normalises a `config.json`), `orchard/decode_floor.py` (181),
`orchard/spec_break_even.py` (182) and `orchard/hazards.py` (183). None is wired into a stage yet. See "What is not wired".

## 181: decode speed as a share of the DRAM-bandwidth floor

The floor is the time to read the active weights once per token: `bytes per token / (dies x per-die bandwidth)`. It is an estimate and
the result says so and lists its inputs.

Bandwidth source. The public Tenstorrent docs site (repository `tenstorrent/tenstorrent.github.io`, page `core/aibs/blackhole/index.md`,
the "Memory Bandwidth" row) lists 448 GB/s for the p100a and 512 GB/s for the p150a and p150b. The page `core/aibs/blackhole/p300.md`
lists 1024 GB/s for the p300c, which carries two ASICs, so 512 GB/s per die. These are datasheet peaks (a 256-bit bus at 16 GT/s).
The tt-metal tech report on saturating DRAM bandwidth reports over 92% on Wormhole and Grayskull; no Blackhole figure for a streaming
kernel is on record here, so 100% of the floor is not reachable and the share should be read with that in mind. The bandwidth argument
of `decode_floor` has no default, so a caller who does not trust the table passes its own figure.

Counting. The count was checked against the shard headers of both models' safetensors files, summed by tensor group. Stored
parameters match to within a thousandth after leaving out the vision tower (460,730,096) and the multi-token-prediction head
(424,699,392) of Qwen3.8-27B. The check found one fault in the first version: the Qwen3-Coder-Next config has no `attn_output_gate`
key, yet its query projection is twice as wide, so the model type `qwen3_next` is treated as gated.

Inputs for the table below: the Qwen3.8-27B bundle ships bfp8 weights (run logs 2026-10-19 and 2026-10-20). The Qwen3-Coder-Next
`p300` profile uses bfp4 routed experts (run log 2026-10-06); the dtype of its attention and shared weights is not on record, so the
table assumes bfp8 for them and also shows all-bfp4 as a reference. Layer layouts are taken from the config by model family: MLPs have three projections except for a listed set of two-projection families (GPT-NeoX and similar), a sliding window is applied by family (Mistral, Qwen2 and Qwen3 rules) and a window on an unknown family is refused. Block formats are counted as bfp8 = 1.0625 bytes per weight and
bfp4 = 0.5625 (16 values share one 8-bit exponent). Both models on one p300 board: 2 dies, 512 GB/s each.

| model | floor tok/s (estimate) | GB per token | measurement | tok/s | share of the estimated floor |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B, all bfp8 (bundle as shipped), p300 2 dies | 37.6 | 27.2 | drafter off, 2 chips, low | 17.8 | 47.3% |
| Qwen3.8-27B, all bfp8 (bundle as shipped), p300 2 dies | 37.6 | 27.2 | drafter off, 2 chips, high | 19.1 | 50.8% |
| Qwen3.8-27B, all bfp8 (bundle as shipped), p300 2 dies | 37.6 | 27.2 | drafter on, 2 chips, about | 60.0 | 159.5% |
| Qwen3.8-27B, all bf16 (reference only), p300 2 dies | 20.0 | 51.2 | | | |
| Qwen3.8-27B, all bfp4 (reference only), p300 2 dies | 71.0 | 14.4 | | | |
| Qwen3-Coder-Next, bfp4 routed experts, other weights assumed bfp8, p300 2 dies | 337.8 | 3.0 | fused-decode bundle, median of 28 | 79.3 | 23.5% |
| Qwen3-Coder-Next, bfp4 routed experts, other weights assumed bfp8, p300 2 dies | 337.8 | 3.0 | earlier package | 38.5 | 11.4% |
| Qwen3-Coder-Next, all bfp4 (reference only), p300 2 dies | 510.9 | 2.0 | | | |

Every share is a share of an estimated floor. What the table says:

- Qwen3.8-27B with the drafter off reaches about half of the floor that its bfp8 weights allow. The drafter-on figure is above the
  floor because a drafter emits several tokens for each read of the weights. The floor does not apply to it, and `share_text` says so.
- Qwen3-Coder-Next moves 3.0 GB per token, so its floor is 338 tok/s on this estimate. The fused bundle reaches about a quarter of it,
  and the earlier package about a ninth. Reading only 3 GB per token makes small ops, collectives and expert reads matter more than
  the matmul bandwidth, which is the pattern issue 181 describes.
- The 27B figure depends strongly on the dtype assumption (20 tok/s at bf16, 71 at bfp4). The share is only as good as the dtype
  policy that went in. A bundle's real policy should be read from the bundle when stage 0 records the floor.

## 182: speculative break-even

`speculative_break_even(plain_step_ms, accepted_per_pass, verify_pass_ms)` returns the ratio against plain decode, the break-even pass
time (`accepted_per_pass x plain_step_ms`) and a verdict. The published example (2.7 accepted per pass, a 4.6 ms plain step) gives a
break-even pass of 12.42 ms, shown as 12.4 ms. The verdict uses a margin of 5% (an input) around a ratio of 1. Zero acceptance gives a
ratio of 0 and a loss. Negative, non-finite, boolean and non-numeric inputs are refused.
`verdict_by_prompt_kind` takes a list of measured acceptance values per prompt kind and states which kind it used: the lowest mean by
default, `pooled`, or a named kind, and it reports every kind's own verdict. `drafter_decision` returns True only for a gain.
The test `test_drafter_is_correctly_turned_off_when_the_targeted_kind_loses` is the done-when case for the drafter turned off.
No measured plain step time or verify pass time exists in this repository yet (stages 2 and 4 record tokens per second, not
per-pass times), so the harness cannot reproduce a real package's decision from measured numbers. That is the part left open.

## 183: hazard scan over catalog configs

The table in `orchard/hazards.py` has 19 rows (the five tile rows are one per dimension). It scans: routed experts above 256 and above 512,
vocabulary per die above 65,536 (rounded up), vocabulary and the hidden, intermediate, head, expert and shared widths as tile counts,
a hidden size of 3 tiles, top-k and KV heads above the nearest bundle's, mixers the nearest bundle lacks (linear attention,
multi-latent attention, sparse attention, hyper-connections), and caller-supplied limits for top-k, KV heads and sliding-window
layers. No figure for the router kernel's supported width or for KV heads per group is on record (the sliding-window limit counts layers over the whole model, because a config does not say how layers are grouped), so those limits are inputs and
the scan invents none. A hit that the nearest bundle's own config also has is marked `also_on_nearest`.

I scanned every cached model whose `config.json` is in the local Hugging Face cache, against the Qwen3.8-27B config as the nearest
bundle, with 2 dies. 33 configs scanned. 163 cached repositories have no `config.json` (container packages and diffusion models) and 12
have a config the scan could not read dimensions from (11 without `hidden_size`, such as GPT-2 which says `n_embd`, and 1 without
`intermediate_size`); the scan did not run on them.

| hazard | models that hit it |
| --- | --- |
| `routed_experts_above_256` (512 experts) | Qwen/Qwen3-Coder-Next |
| `vocab_per_die_above_65536` on 2 dies | 18 models (a vocabulary above 131,072 is what the check means on 2 dies). The Qwen3.8-27B family hits it with 124,160 columns per die and is marked as also on the nearest bundle, which runs on 2 chips, so the limit does not bind there |
| `kv_heads_above_nearest` (nearest has 4) | 17 models, mostly Llama-style models with 8 KV heads and encoders |
| `vocab_not_whole_tiles` | bert-base-uncased and three sentence-transformers encoders (not language models for this harness) |
| `mixer_multi_latent_attention_not_in_nearest`, `mixer_hyper_connections_not_in_nearest` | XingChen-AGI/Xing4.0-29B-A4B (its first scan also reported `head_dim_not_whole_tiles`, a false hit from the head width defaulting to hidden size over heads; multi-latent attention heads are the query/key width, 192 here, which is 6 whole tiles) |

Qwen3-Coder-Next also hits `vocab_per_die_above_65536` on 2 dies (75,968 columns per die), marked as also on the nearest bundle.
The scan was repeated with 4 dies. It added no hit that the 2-die scan lacked, and it dropped the vocabulary hit for every model with a
vocabulary of at most 262,144. The Qwen3-Coder-Next fused bundle runs with 512 experts, so a bundle that handles the 256-bit mask
limit exists; a new 512-expert model should start from it, and stage 2 should still run the high-index router check.

## What is not wired

Stage 0's facts record is checked by `orchard/audio_facts.py` (`facts_problems` refuses any key other than `form`, `serving`,
`modality`) and the record is written by the template `orchard/skills/delta-triage-templates/delta_triage.py`, which runs as a script
in a run directory and cannot import `orchard`. Adding a field means editing `audio_facts.py`, `stages.py` gate code that the audio
work (PR 187) owns, and a copy of the scan inside the template with a test that pins it to the module, as the audio detection already
does. I did not do that. The floor and the scan are available as `python3 -m orchard.decode_floor` and `python3 -m orchard.hazards`.
What is left: add `hazards` and `floor` to stage 0's facts record after PR 187 merges; show the share in stage 2, stage 4 and the
bundle; record plain step and verify pass times in stage 2 so 182 can write its verdict to the ledger; read the dtype policy from the
bundle.
