# tier-plan with a tensor-parallel layout, against the Inkling-2-Mini bundle (2026-11-24)

The card of `raahemnabeel/inkling-2-mini` says its 2- and 4-die profiles divide the model over the dies: the attention heads, the experts' intermediate width and the output head. The card and the bundle's own checks assert that the query and key-value head counts divide by the die count. `tt-orchard tier-plan` offered two placements that fit on chips, "replicated" and "experts split over the chips", so for a model that fits only when the attention weights are divided as well, the plan said "fits with host experts".

## What the plan now does

When neither of those two placements fits and the die count is above one, the plan tries a third: **tensor-parallel**. It divides the attention weights, any dense or shared MLP, the output head and the routed experts by the die count. The routers and the embedding table stay on every die. This is a layout the plan assumes, taken from the bundle's description, and no port of it exists in tt-orchard. The row says so in its note.

The layout needs these counts to divide by the die count (`tp_counts` in `tier_plan.Shape`, read from the config): the attention heads, the key-value heads, the linear-attention key and value heads and the Mamba heads when the model has them, and the experts' intermediate width in whole 32-wide tiles per die. A row that cannot use the layout names the count that blocks it (`tensor_parallel_blocked`). Nothing changes for a row that already fits as "replicated" or "experts split".

## Inkling-2-Mini, recomputed with the repository's own tools

Counts from the public config (`tests/fixtures/model-configs/inkling-2-mini.config.json`): 24 attention heads, 8 key-value heads, expert width 1152 (36 tiles). The layout divides by 2 and by 4. It does not divide by 8, because 1152 is 36 tiles and 36 does not divide by 8; the plan reports that.

Weights per die, tensor-parallel (the plan's arithmetic; the card's figures are not used here):

| Format (non-expert and expert) | 1 die, replicated | 2 dies | 4 dies |
| --- | --- | --- | --- |
| bf16 | 48.46 GiB | 24.67 GiB | 12.77 GiB |
| bfp8 | 25.75 GiB | 13.11 GiB | 6.79 GiB |
| bfp4 | 13.63 GiB | 6.94 GiB | 3.59 GiB |

Effect on the rows with the default constants: "all bf16" on 2 chips changes from "fits with host experts" to "fits on chips" (24.67 GiB per die against a budget of 24.89 GiB). The other rows are unchanged, because they already fit as replicated or split. The 4-die all-bfp8 row still prints "replicated, 25.75 GiB per chip", since the simpler placements are tried first; the tensor-parallel figure for that row is 6.79 GiB per die, which leaves most of each die for KV cache. The card's 4-die profile reaches a 1,048,576-token context.

## What was not checked

- The plan's split of the weights is the bundle's description as the card gives it. The bundle's code was not read here, so details such as whether the embedding table is divided over the vocabulary are not confirmed. The plan keeps it on every die, which makes the per-die figure a little high.
- Nothing ran on a chip. A fit says only that the weights fit; speed is not modelled. The card's measured one-user decode on 1, 2 and 4 dies shows that dividing over more dies helps much less than the byte count alone suggests (`docs/run-logs/2026-11-24-decode-floor-scaling-inkling.md`, when that note is merged).
