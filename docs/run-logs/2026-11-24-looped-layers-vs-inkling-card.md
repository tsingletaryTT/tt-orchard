# Looped layers in the decode floor, checked against the Inkling-2-Mini card (2026-11-24)

Source of the comparison: the Lottolabs Inkling-2-Mini card says decode streams 3.72 GB of weights per token with BFP8 attention (map g4 in
its `prec_maps.md`). The upstream model's config has `loop_config` (`core_start` 6, `core_end` 18, `num_passes` 3): layers 6 to 17 run three
times for each token, so 24 stored layers make 48 layer applications. Nothing here was measured on a chip. The card's figure is the card's own.

## What the floor said before

`orchard/decode_floor.py` counted each stored layer once. With the policy BFP8 for everything and BFP4 for the routed experts, on the
public config at the head of the upstream repository (revision `7a67ae23`, read on 2026-11-24; the card names revision `6e7c8248`, which
the Hub does not know under that repository, so the revision the card was measured on could not be read):

| | bytes per token | floor on one p150 at 512 GB/s |
| --- | --- | --- |
| before (each layer once) | 1.841 GB | 278.1 tok/s |
| after (loop and attention gate counted) | 3.551 GB | 144.2 tok/s |
| card, map g4 | 3.72 GB | 126 tok/s at 470 GB/s |

The share of the floor for the card's 86.8 tok/s single-stream speed was therefore 31 percent of the old floor, and is 60 percent of the new
one at 512 GB/s (69 percent at the card's 470 GB/s roofline).

## Where the new 3.551 GB comes from

| group | GB per token | how |
| --- | --- | --- |
| routed experts | 1.720 | 48 applications x 8 of 128 experts x 3 x 2304 x 1152 weights x 0.5625 bytes. This equals the card's expert figure. |
| attention | 1.324 | 48 applications of the attention layer, with the output gate (below) |
| LM head | 0.492 | once |
| router | 0.015 | 48 applications |

The attention figure needed a second change. The checkpoint's safetensors header (read with `get_safetensors_metadata`, which fetches headers
only) lists `attn.gate_attn.weight` as [3072, 2304] in every layer, the same size as the query projection, and the config has no key for it.
`orchard/model_config.py` now counts the query twice as wide for model type `inkling2_loop_text`, the same rule that exists for `qwen3_next`.
Without it the total would be 3.19 GB.

## What is still missing: 0.17 GB, 4.6 percent under the card

The header also lists, for each looped layer, four loop controllers (`controllers.{attn,k,mlp,v}`: a [64, 4608] down matrix and two [2304, 64]
matrices each, about 2.4 million weights per layer, BF16), and a [384, 2304] relative-position projection in every layer. The config names
only `gate_rank`, so the floor does not count them. Counted by hand they add about 0.17 GB (36 looped applications) and 0.05 GB, which is
the gap to 3.72 GB. `orchard/tier_plan.py` now says in its caveats that the loop controllers are not counted, and
`orchard/decode_floor.py` says so in its assumptions.

## What changed in the code

- `orchard/model_config.py`: `ModelDims.loop_passes` (one count per layer) and `loop_variable`, read from `loop_config`; a bad loop is refused
  with the key named. A config without the key, or with it null, counts every layer once.
- `orchard/decode_floor.py`: active bytes count each looped layer once per pass; stored size does not change; the summary records the passes.
- `orchard/tier_plan.py`: a caveat for looped layers. The fit (what is stored) is unchanged apart from the attention gate.
- `orchard/hazards.py`: a `looped_layers` mixer and the row `mixer_looped_layers_not_in_nearest`.
