# KL and top-1 limits beside the perplexity bar, checked against published cards (2026-11-24)

The fidelity rule in `orchard/plan.py` (`FIDELITY_BAR = 9.9`, issue 731) reads one number: 10 x reference perplexity / port perplexity, which is
about 1 percent of perplexity or less. The Lottolabs releases publish more than that for each build: KL against a CPU BF16 reference, top-1
agreement and the perplexity change, on 10,000 to 58,000 scored positions in four kinds of text. Nothing below was measured here. The figures are
the cards' own, and the scores are worked out from the cards' perplexity changes with `score = 10 / (1 + dPPL)`. A card's perplexity change is
measured on natural text, and our score is measured on text the reference sampled, so the two are not the same quantity. The conversion shows where
the rules disagree and is not a measurement of a port.

| build (card figures) | perplexity change | KL | worked-out score | ratio bar 9.9 | `lfm3-card` rule |
| --- | --- | --- | --- | --- | --- |
| LFM3-1.7B | +0.55 percent | 0.0243 | 9.945 | ok | ok |
| LFM2.5-24B-A2B | +0.29 percent | 0.0458 (top-1 0.915) | 9.971 | ok | below the bar: KL above 0.03 |
| Inkling-2-Mini, default map | +0.15 percent | 0.0112 (top-1 0.966) | 9.985 | ok | ok |
| Qwen3.6-35B-A3B | +4.78 percent | 0.075 (top-1 0.918) | 9.544 | below the bar | below the bar |
| Qwen3.8-27B, code corpus | +4.25 percent | 0.055 | 9.592 | below the bar | below the bar |

LFM2.5 is the case the new limits exist for: it clears the perplexity bar with a KL 53 percent above the LFM3 card's own limit.

## What changed

- `orchard/plan.py`: `FidelityRule` (a ratio bar plus optional `kl_max`, `top1_min` and `abs_dppl_percent_max`), the named rules `perplexity`
  (the default: the bar alone, as before) and `lfm3-card`, `gate_detail` (the gate and a reason), and the options `--fidelity-kl`, `--fidelity-top1`,
  `--fidelity-dppl-percent`, `--fidelity-rule`, `--fidelity-kl-max` and `--fidelity-top1-min`. A bare `(score, tokens)` pair gates exactly as before.
  A rule that needs a measure the record lacks gives `unscored` and names the measure. The perplexity change is worked out from the score when the
  record does not state it. Each row carries `format_gate_reason`, and the plan names the rule it used.
- `orchard/tier_plan.py`: the same options and rule, on every row.
- `orchard/cli.py`: the `plan` wrapper built the plan's argument list by hand and dropped `--fidelity-bfp4` and `--fidelity-tokens`; it now passes
  every fidelity option.
- `scripts/fidelity_score.py`: each scored result carries `dppl_percent` and `plan_flags`, the options to paste into `plan` or `tier-plan`.

## What is not done

Stage 2's `fidelity` and `sampled_fidelity` blocks hold the log-probability of each sampled token on the port and on the reference, so they have a
perplexity ratio and no KL. A KL needs the port's whole next-token distribution at each scored position (or its top candidates), which the stage 2 check
does not return today. Stage 2 does record a top-1 agreement. The CPU fake-quant row beside the device row (the Qwen3.6 card: KL 0.046 simulated against
0.075 on the device, so the runtime adds as much error as the weights) needs the same distribution on the device side. Neither was added to the stage 2
record in this change. `scripts/fidelity_score.py --simulate bfp4` already produces the simulated row on CPU.
