# Full-size benchmark of bytkim/Qwen3.8-27B-pi against Qwen/Qwen3.8-27B

Date: 2026-10-19 to 2026-10-20 (UTC). Plan: `docs/superpowers/plans/2026-10-19-model-card-benchmark.md`. Pilot: `2026-10-19-bytkim-pi-benchmark-pilot.md`.
Files: `evidence/2026-10-20-bytkim-pi-benchmark-full/` (`benchmark.json`, the compile script, the card edit script).

## Conditions

Both models served one after the other under `gozer run --chips 2 --exact 2`, through the same bundle (`episod/qwen3.8-27b-dflash2-p300`: bfp8, DFlash2
drafter on, sampling on the device), the same `run.sh` apart from the model path, warm tensor caches. Every item of each set, temperature 0, thinking as
the chat template defaults it, `max_gen_toks` 8192, concurrency 4. lm-evaluation-harness 0.4.13, `humaneval_chat` and `mbpp_chat` from this repository,
BFCL v3 revision `61fc0608cfd8`. Session wall time: pi 03:56Z to 08:14Z, base 08:14Z to 13:09Z. No request errors in any phase.

## Results (pass rate, standard error; replies cut off at the length limit)

| Task | pi | base | cut off (pi / base) | pairs where only pi / only base was right |
|---|---|---|---|---|
| HumanEval (164) | 0.957 ± 0.016 | 0.939 ± 0.019 | 6 / 8 | 6 / 3 (p 0.51) |
| MBPP (500) | 0.896 ± 0.014 | 0.864 ± 0.015 | 44 / 54 | 26 / 10 (p 0.011) |
| GSM8K flexible-extract (1,319) | 0.839 ± 0.010 | 0.792 ± 0.011 | 0 / 0 | 101 / 39 (p below 0.001) |
| GSM8K strict-match (format) | 0.603 | 0.562 | | |
| IFEval prompt-level strict (541) | 0.874 ± 0.014 | 0.850 ± 0.015 | 30 / 38 | 31 / 18 (p 0.085) |
| BFCL simple (400) | 0.932 ± 0.013 | 0.930 ± 0.013 | 0 / 0 | not saved per item |
| BFCL multiple (200) | 0.955 ± 0.015 | 0.935 ± 0.017 | 0 / 0 | not saved per item |

The p values are two-sided sign tests on the items where the two models differ.

- The pilot (20 items) could not tell the models apart. At full size the fine-tune is ahead on every set. The gap is clear on GSM8K and MBPP and within noise on the others.
- Most failures on HumanEval and MBPP are empty replies (the 8192-token limit reached while thinking): pi 6 of 7 and 44 of 52, base 8 of 10 and 53 of 68. Both scores are lower than a higher limit would give.
- Drafter acceptance (server log, per phase): pi 0.46 to 0.60 of drafted tokens, 4.2 to 5.2 tokens per iteration; base 0.46 to 0.56, 4.2 to 4.9. Highest on BFCL, lowest on MBPP.
- The pilot's decode estimate was too high: the full run took 4.8 hours for pi and 4.9 for the base, 9.8 hours in all.

## Card and publish

The card gained the 14 score rows, a "Benchmark against the base model" section, new limitations and a new not-measured list; `gate_package` passed.
`hf upload --private`, then `tt-model publish episod/qwen3.8-27b-pi-p300` (public, listed in the community catalog), on the operator's instruction.
Not tested: `tt-model pull` and a boot from the Hub.

## Not done

Speed sweep by input length and concurrency, MMLU-Pro, long context, thinking-off and higher-limit runs, per-item BFCL results.
