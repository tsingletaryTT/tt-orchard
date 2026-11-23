# tier-plan against three published hybrid-MoE bundles (2026-11-23, issue 739 step 1)

`tt-orchard tier-plan` was run on the config.json of each base model, with no weight downloaded, and its figures were compared with the figures the
published bundle cards state. The constants were not changed to make the figures agree: every constant is the value in the code (32 GiB DRAM per chip,
93.4 percent usable, a 3 GiB reserve per chip, the bytes per element in `orchard/plan.py`), and the only flag set is the KV cache allowance, which a
card's context length fixes.

The bundles are `raahemnabeel/minimax-m2.7-hybrid-qb2`, `raahemnabeel/inkling-small-hybrid-qb2` and `raahemnabeel/ornith-1.5-397b-hybrid-qb2`
(public cards, read on 2026-11-23). Each runs on a p300x2, which is 4 chips.

## What the config count gives

| Model | Config says | Plan counts | Card says |
| --- | --- | --- | --- |
| MiniMaxAI/MiniMax-M2.7 | 62 layers, 256 experts, top 8 | 4.01 B always-resident, 224.68 B routed (228.7 B in all) | 230B parameters, 62 layers x 256 experts |
| thinkingmachines/Inkling-Small | 42 layers, 256 experts, top 6 | 5.57 B always-resident, 270.58 B routed (276.2 B in all) | 276B parameters, 42 layers, 256 routed + 2 shared, top 6 |
| ornith-ai/Ornith-1.5-397B-FP8 | 60 layers, 512 experts, top 10 | 9.80 B always-resident, 386.55 B routed (396.4 B in all) | 397B parameters, 60 layers x 512 experts, top 10 |

The parameter totals agree with the published sizes to within 1 percent. For Ornith the weight index of the FP8 checkpoint says 389.6 GiB and the
config count in bf16 is 738.3 GiB (minus 47 percent), and the command prints that as a note: the checkpoint is stored in FP8, so a bf16 count is about
twice the file size.

## Host memory and layers on the chips: plan says X, card says Y

The plan prints two layouts. "All pinned" keeps the same number of experts in every layer on the chips and pins every expert in host memory with a
10 GiB margin (the rule in the issue). "Whole layers" keeps complete layers on the chips and needs host memory only for the other layers' experts plus
the margin (what the MiniMax and Ornith cards describe). Rows below use the plan's `bfp8 + bfp4 experts` row (attention and tables in bfp8, experts in
bfp4, which matches the cards' bfp4 device experts) unless stated.

| Bundle and profile | Plan: layers on the chips | Plan: host GiB, whole layers | Plan: host GiB, all pinned | Card |
| --- | --- | --- | --- | --- |
| MiniMax-M2.7, 32K context (KV allowance 7.75 GiB: 253,952 bytes per token in bf16 times 32,768) | 44 of 62 | 44.2 | 127.7 | 44 full layers, 18 hybrid layers, about 43 GB free host RAM |
| MiniMax-M2.7, 64K context (KV 15.5 GiB) | 40 of 62 | 51.8 | 127.7 | 36 full layers, 26 hybrid, about 57 GB |
| MiniMax-M2.7, default KV allowance (4 GiB) | 46 of 62 | 40.4 | 127.7 | not a card profile |
| Ornith-1.5-397B, `all bfp4` row | 24 of 60 | 131.5 | 212.5 | `p300x2-128k-lowmem` about 135 GB; `p300x2-128k` (default) about 210 GB |
| Inkling-Small, `all bfp4` row | 27 of 42 | 60.6 | 151.8 | about 135 GB of free host RAM |

Reading of the comparison:

- MiniMax: the layer count agrees with the card (44 against 44 at 32K, 40 against 36 at 64K) and the whole-layer host figure agrees with the card's
  (44.2 against about 43, 51.8 against about 57). The KV allowance is the only input set to fit a context length. It comes from the config
  (62 layers, 8 KV heads, head size 128, bf16) and does not come from the card. The all-pinned figure (127.7 GiB) is about three times the card's, because the
  card keeps most layers fully on the chips and pins nothing for them.
- Ornith: the all-pinned figure (212.5 GiB) is within 2 percent of the default profile's about 210 GB, and the whole-layer figure for the all-bfp4
  row (131.5 GiB) is within 3 percent of the low-memory profile's about 135 GB. The two profiles differ by pinned device-format copies of experts, which the
  plan does not model, so some of this agreement is a coincidence of the two layouts and should not be read as a validation of either.
- Inkling-Small: the all-pinned figure (151.8 GiB) is 12 percent above the card's about 135 GB. The whole-layer figure (60.6 GiB) is less than half of it.
  The card's host memory holds two copies of the experts that are not on the chips (about 49 GB of int4 copies for the CPU and about 66 GB of pinned
  device-format copies, plus the server), and the plan counts one. The card's 175 static device slots are not comparable with the plan's 147 to 165
  experts per layer, because the card does not say whether the slots are per layer.

What this does not show: nothing here checks that a port with these layouts runs, how fast it decodes, or how the model's own router distributes its
picks. The hit shares the plan prints for the all-pinned layout come from OLMoE-1B-7B (64 experts, top 8), and the plan labels them that way in every row.
