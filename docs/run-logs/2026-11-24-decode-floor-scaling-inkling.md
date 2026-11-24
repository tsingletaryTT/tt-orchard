# The decode floor against measured scaling over dies, Inkling-2-Mini (2026-11-24)

`orchard/decode_floor.py` divides the bytes read per token by the sum of the dies' bandwidth, so its floor doubles when the die count doubles. The published cards for Inkling-2-Mini give measured single-user decode speeds, which show how far a real build is from that. The speeds below are the cards' figures. None was measured here, and nothing ran on a chip. The floors and shares were recomputed on 2026-11-24 with `python3 -m orchard.decode_floor` on the public config (`tests/fixtures/model-configs/inkling-2-mini.config.json`), 512 GB/s per die.

## Raahem's bundle: 1, 2 and 4 dies

All weights at bfp8 (the card's format), 5.08 GB read per token (the looped layers counted once per pass, PR 824). Plain decode without the drafter, temperature 1, three 300-token prompts, from the card's performance table.

| Dies | Floor (tok/s) | Measured (tok/s) | Share of the floor | Speed-up over 1 die | Ideal | Scaling efficiency |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 100.8 | 53.2 | 53% | 1.00 | 1 | 1.00 |
| 2 | 201.6 | 74.8 | 37% | 1.41 | 2 | 0.70 |
| 4 | 403.2 | 92.3 | 23% | 1.73 | 4 | 0.43 |

Each doubling of the die count adds 41 percent (1 to 2 dies) and 23 percent (2 to 4 dies) to the speed of a single user. The floor gains 100 percent each time. The card does not say where the rest goes. Collectives between dies at layer boundaries, a per-die launch cost and a fixed host cost per token would all produce this shape, and the numbers here do not separate them.

## Lottolabs' build: one die, mixed formats

Experts at BFP4 and the rest at BFP8 (the policy `default=bfp8,routed_experts=bfp4` approximates the card's mixed format; the card's attention q and gate are also BFP4). 3.55 GB per token, floor 144.2 tok/s. The card's 86.8 tok/s (greedy, mean of 8 prompts) is 60 percent of it.

The two cards use different protocols (temperature 1 and three 300-token prompts against greedy and eight prompts), so the 53 and 60 percent shares are not a like-for-like comparison of the two builds. Within Raahem's table the three rows share one protocol.

## What changed in the code

- `decode_floor.scaling_table(config, policy, bandwidth, measured_by_dies)` returns one row per die count with the floor, the share of it, the speed-up over the smallest die count, the ideal speed-up and their quotient. Every row is labelled as supplied by the caller and not measured by the module.
- `python3 -m orchard.decode_floor CONFIG --dies 4 --per-die-gb-s 512 --policy default=bfp8 --measured-by-dies 1=53.2,2=74.8,4=92.3` prints the table.
- `ASSUMPTIONS` names the gap. The floor itself did not change, and a test compares the floor before and after a table call.

## What the table does not say

Four dies at bfp8 reach 92.3 tok/s. One die at the mixed format reaches 86.8 tok/s on the other card's protocol. By bytes alone, the mixed format reads 30 percent less per token (3.55 against 5.08 GB), and it needs one die where the bfp8 build needs four. Whether the 4-die build would do better with the mixed format is not known. That needs a run on a board.
