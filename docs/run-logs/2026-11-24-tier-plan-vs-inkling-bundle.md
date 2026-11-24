# tier-plan against Raahem's Inkling-2-Mini bundle: where the fit budget disagrees (2026-11-24)

Source of the card figures: the published card of `raahemnabeel/inkling-2-mini` (profile table, row `p150`). Every figure from the card below is the card's own and was not measured here. Every figure from `tt-orchard tier-plan` was recomputed on 2026-11-24 from the public config of the upstream model (`tests/fixtures/model-configs/inkling-2-mini.config.json`); nothing ran on a chip.

## The disagreement

`tt-orchard tier-plan` on the config, one chip, all bfp8, says "fits with host experts": 22.72 GiB per chip, 112 of 128 experts on the chip. The card says the same model runs on one die with all experts at bfp8 ("the bfp8 experts take 27 of the die's 31.5 GiB", profile `p150`: 8 sequences sharing a 32,768-token context).

## The arithmetic

| Quantity | Plan | Card |
| --- | --- | --- |
| DRAM of one chip that weights may use | 29.89 GiB (93.4 percent of 32, `orchard/plan.py`, "probe 14") | 31.5 GiB |
| Held back before weights: reserve | 3.0 GiB (`orchard/defaults.py`) | not stated |
| Held back before weights: KV cache | 4.0 GiB for the whole model, a flat figure for every model | not stated; the profile has 8 sequences over 32,768 tokens |
| Weight budget per chip | 29.89 - 3.0 - 4.0 = 22.89 GiB | 31.5 GiB less whatever the runtime keeps |
| Routed experts at bfp8 | 24.21 GiB (24.46 B parameters x 1.0625 bytes) | 27 GiB |
| Always-resident weights (attention, routers, embedding, head) | 1.54 GiB | not stated |
| Everything replicated on one chip | 25.75 GiB | at least 27 GiB plus the rest |

By the plan's own figures the weights need 25.75 GiB and the usable share is 29.89 GiB, so the weights fit if the reserve and the KV share together stay under 4.14 GiB. The plan assumes 7.0 GiB. The card's figures leave 31.5 - 27 - 1.54 = 2.96 GiB for the reserve and the KV share if the rest of the model is as the plan counts it. Both numbers are below the plan's 7.0 GiB, which is the assumption that decides the row.

Three of the plan's inputs differ from what the card implies:

1. **Usable share.** 29.89 GiB against 31.5 GiB. The plan's figure comes from one probe (29.9 GB usable of a 32 GB chip); the unit in that probe note is GB, and whether it is the same quantity as the card's 31.5 GiB was not checked.
2. **Reserve plus KV share.** 7.0 GiB against about 3 GiB. The KV allowance does not depend on the model. Inkling-2-Mini has full attention in 4 of its 24 layers and 1,024-token sliding windows in the other 20 (the config's `local_layer_ids` lists 20 layers and `sliding_window_size` is 1024), so its KV cache at 32,768 tokens is much smaller than the flat 4 GiB.
3. **Expert bytes.** 24.21 GiB against 27 GiB, a difference of 2.8 GiB (11.6 percent). The difference is close to the size of three more expert layers at bfp8 (3.03 GiB). That would match a drafter with three expert layers of the same size. The card does not say what the 27 GiB holds, and this was not checked.

## What changed

`tier-plan` keeps every constant and every verdict. A row that does not fit on the chips only because of the reserve and the KV share now carries `overhead_limit_for_chips_bytes` (in the JSON) and a line in the text output. For this model:

```
all bfp8, 1 chip(s): would fit on the chips if the reserve plus this chip's share of the KV cache were at most 4.1 GiB per chip; the plan assumes 7.0 GiB (--reserve-gib and --kv-gib set them). ...
```

A reader can then run `tt-orchard tier-plan ... --reserve-gib 2 --kv-gib 1` and see the row fit, with the flag values visible in the output as the caller's. No constant was changed to match the card.

## What is still open

- The card's 27 GiB of experts is 2.8 GiB above the plan's count. Reading the weight index of the published repository (`tt-orchard tier-plan raahemnabeel/inkling-2-mini` reads only the index) or the bundle's loader would settle whether it is the drafter or padding.
- The usable share of a Blackhole chip's DRAM has one probe behind it. A second probe on a free board would show whether 29.9 or 31.5 GiB is right.
- A model-aware KV allowance (from the config's attention layout and a context length) would remove the flat figure. That needs a context length as an input and was left out of this change.
