# Choosing the served configuration by what the model is for (issue 107)

Status: built on branch `sampling-choice`, version 0.9.12. Nothing here ran on a chip. The measurements it uses are in
`docs/run-logs/2026-10-27-greedy-only-sampling.md`.

## The problem

The Qwen3.8 DFlash2 bundle has two serving configurations on a 2-chip board. With the drafter on it decodes at about 60 tokens/s, and it accepts
a request with temperature above 0 but samples only the first token; every later token is greedy. With the drafter off and host sampling it
decodes at about 18 tokens/s and samples correctly. A model whose purpose needs sampling is not served well by the first configuration, however
fast it is. A model whose purpose does not need sampling loses two thirds of its speed in the second.

The operator's ruling (issue 107): stage 2 chooses between a drafter-on and a drafter-off variant by measurement, and the choice has to be a
practical one. A configuration that is fast and cannot do what the model is for has no value. No change goes into `vllm-tt-plugin`.

## What the rule reads

1. Whether sampling is needed, from files (`need`). In order:
   - the operator's flag `--sampling needed` or `--sampling not-needed` (default `auto`, recorded in `run_start`, kept on resume);
   - the model card warns against greedy decoding (stage 0 records this in `genconfig-license.json` as `card_warns_greedy`, with the quoted line);
   - the model's own `generation_config.json` (not the copy staging takes from the nearest model) has `do_sample` true, a temperature above 0,
     a top_p below 1 or a top_k above 0;
   - the model card recommends such settings (`card_sampling_hints`);
   - otherwise sampling is not needed, and the record says no evidence was found.
   An explicit `do_sample: false` with no other evidence counts as evidence that sampling is not needed. The record keeps the file and the value
   or line that decided it.
2. The sampling probe's verdict for the configuration stage 2 serves (after the graft decision, see below). `sampled-full-speed` and
   `sampled-slower` mean sampling is applied. Every other verdict means it is not, or that nothing is known.
3. The speed of each configuration, from the probe's own requests (the mean of the two sampled requests, else the greedy request).
4. A floor, `SAMPLING_MIN_TOKS_PER_S` = 10 tokens/s. Measured drafter-off speeds are 16.5 to 19.1 tokens/s on 2 chips (one run). Ten is
   about 55 percent of the lowest measured value. At 10 tokens/s a 1,000-token reply takes 100 seconds, which a person waiting for a chat reply
   still tolerates and an agent loop does not enjoy. It is a choice and is labelled as one in `defaults.py`.

## The decision

The choice is `keep` (the configuration stage 2 served stays) or `drafter-off` (the drafter-off variant replaces it). Cases:

| Sampling needed | Served configuration | Variant | Choice | Flag |
|---|---|---|---|---|
| no | any | not tried | keep | none |
| yes | applies sampling, at or above the floor | not tried | keep | none |
| yes | applies sampling, below the floor | not tried | keep | `below-floor` |
| yes | ignores sampling, drafter on | serves, applies sampling, at or above the floor | drafter-off | none |
| yes | ignores sampling, drafter on | serves, applies sampling, below the floor | keep | `no-variant-serves-purpose` |
| yes | ignores sampling, drafter on | does not serve, or does not apply sampling | keep | `no-variant-serves-purpose` |
| yes | probe missing or inconclusive, drafter on | serves, at or above the floor (or speed unknown) | drafter-off | `unverified` when no probe verdict backs it |
| yes | probe missing or inconclusive, drafter on | does not serve | keep | `unverified` |
| yes | ignores sampling or unknown, drafter off or not known | not possible | keep | `unverified` or `no-variant-serves-purpose` |

When the flag is `no-variant-serves-purpose` the run continues with the faster configuration, the decision file says in plain words that neither
configuration does what the model needs, and the stage 8 bundle lists it under the risks the operator must read. A flag never fails the stage:
the operator decides. A configuration that ignores sampling is kept in that case because the alternative (a variant that is too slow to use)
is no better, and the record says which of the two problems was chosen.

The variant is measured by `sampling_trial.py`, a hardware test under the same lease rules as the graft trial. It builds the stage's
configuration with `"drafter": "off"` in `sampling-trial/variant/` (own tensor cache), runs the stage 2 check (which includes the sampling
probe), and writes `trial-result.json`. It runs only when `needs_variant` says the rule cannot be decided without it: sampling is needed, the
served configuration has the drafter on, and the probe does not show that sampling is applied.

## Order with the graft trial

A graft keeps the drafter on. Its variant is served with the same sampling probe as every swap check, so the graft trial already holds the
sampling verdict of the grafted configuration. The order is explicit:

1. Stage 2 serves its configuration and passes the gate.
2. The graft trial runs when there is a graft candidate. The graft is chosen only when its usual checks pass and, when sampling is needed, the
   grafted variant's own probe shows sampling applied. A grafted drafter that ignores sampling is not chosen for a model that needs it; the
   reason says so. The graft's decision file records `sampling_needed` and the variant's verdict.
3. The sampling decision then judges the configuration that is now served: the graft when it was chosen, else stage 2's own. Because of step 2,
   a chosen graft always applies sampling when it is needed, so the sampling decision keeps it. When the served configuration is the safe one
   (drafter off), the decision keeps it and the floor is checked.

A graft candidate never needs the drafter-off trial: its served configuration already has the drafter off.

## Downstream

- Stage 4 bundle configurations get `"drafter": "off"` when the decision is `drafter-off` (prepare_swap writes the same two run.sh edits as
  for a model with no mtp tensors). Container configurations get the key too (the container script honours it). The stage 4 gate reads the
  serving record of each configuration and refuses one that disagrees with the decision.
- Stage 7 reads the decision: a `drafter-off` choice makes the package drafter-off with host sampling, as for a model with no mtp tensors.
  `--graft-mtp` with that decision is refused, because the two settings ask for opposite packages.
- The package card's Sampling section states the choice and the reason in plain sentences built from the recorded numbers, and its rows come
  from the configuration the package serves. The gate refuses a card that leaves the sentence out.
- The stage 8 bundle's Sampling section states the choice, the numbers and the flag.

## Gate

`decision_problems(stage_dir, run_dir)` recomputes the decision from the files on disk: the need from stage 0's files and the override recorded
in the ledger, the sides from the swap checks, the speeds from the probe's raw requests. A file whose choice or flag differs from the
recomputation is refused. This is best effort against an edited file, as the graft gate is.

## Not decided here

- Whether every Qwen3.8 fine-tune should count as "needing sampling" because the nearest model's generation config sets a temperature. The
  rule counts only the model's own file, and nearly every Qwen3.8 fine-tune ships one with a temperature, so most of them flip. The operator
  can override per run.
- A drafter-on configuration that applies sampling (a fixed bundle) is handled by the table and never seen on hardware.
- 4 chips: the drafter-off speed was measured on 2 chips only.
