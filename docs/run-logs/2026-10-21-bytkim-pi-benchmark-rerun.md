# Re-run of the cut-off items at 32768 tokens: `bytkim/Qwen3.8-27B-pi` and the base

Date: 2026-10-20 to 2026-10-21 (UTC). Evidence: `evidence/2026-10-21-bytkim-pi-benchmark-rerun/`. Follows `2026-10-20-bytkim-pi-benchmark-full.md`.

## What was run

The full run (8192-token limit, thinking on) left 80 cut-off replies for the fine-tune and 100 for the base on HumanEval, MBPP and IFEval. Each cut-off is an empty reply, so
the items were listed from the stored samples (`orchard.bench_rerun items`): 81 for the fine-tune (6, 44, 31) and 98 for the base (8, 53, 37). They were asked again with
`--max-gen-toks 32768`, same bundle, same `run.sh`, same thinking setting, temperature 0, four requests at once, one 2-chip lease per model. `bench_session` gained `--items`
(passed to the harness as `--samples`, refused with `--limit` and for bfcl). `orchard.bench_rerun merge` replaces each re-run item's result and recomputes the score over every item.

The fine-tune's session ran 17:43Z to 20:43Z, the base's 20:43Z to 00:18Z. No request failed in either session. A watcher stopped on any error, on more than half of the items
still capped, on a 45-minute silence, or on a set where the re-run lost more items than it gained. It never fired.

## Result

| Set | Fine-tune before | after | Base before | after | Re-run items (ft / base) | Still cut off (ft / base) |
|---|---|---|---|---|---|---|
| HumanEval | 0.957 | 0.976 | 0.939 | 0.963 | 6 / 8 | 1 / 2 |
| MBPP | 0.896 | 0.940 | 0.864 | 0.928 | 44 / 53 | 18 / 19 |
| IFEval prompt-strict | 0.874 | 0.885 | 0.850 | 0.863 | 31 / 37 | 15 / 13 |

No re-run item got worse. Paired sign tests at the higher limit: HumanEval 4 against 2 (p 0.69), MBPP 19 against 13 (p 0.38), IFEval 30 against 18 (p 0.11). At 8192 tokens MBPP was 26
against 10 (p 0.011). GSM8K and BFCL had no cut-offs and are unchanged: GSM8K is still the one clear difference (101 against 39).

So the MBPP gap that the card called clear came mostly from the length limit. The base lost more items to it, and it gained more from the re-run (32 against 22 items).

## Limits of this check

- The list of items is the empty replies. The proxy's cut-off count differs by one for three of the six model and set pairs (fine-tune IFEval 30 cut-offs and 31 empty replies; base MBPP 54 and 53; base IFEval 38 and 37). A reply cut off after its answer began has text and is not in the list; the request log has no document ids, so it cannot be found.
- Items that were still cut off at 32768 tokens count as wrong. A higher limit would still help a little.
- The document ids of a `--samples` run match the first run's (the merge refuses an id the first run lacks, and none were refused).

## Gate finding

`gate_package` on `main` rejects the package's `run.sh` (`HF_MODEL` set twice: the thin-package default line and the `model-dir` line). The `run.sh` format fix is commit `e61df0f` on `bytkim-package`,
which was not in `main`. The card gate first passed with that branch's code. This branch now carries `e61df0f` as a cherry-pick (PR 13), so the gate passes here; it brings `orchard/measure_serving.py` and the 10-19 packaging run log with it, and `bytkim-package` will overlap this branch when it lands.

The updated card was uploaded to `episod/qwen3.8-27b-pi-p300` (commit `84d3c49`) and published again on 2026-10-21, after the gate passed.

## Not done

Speed sweep, MMLU-Pro, long context, thinking off, a limit above 32768 tokens, `tt-model pull` and a boot from the Hub.
