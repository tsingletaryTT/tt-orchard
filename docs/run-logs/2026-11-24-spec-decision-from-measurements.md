# The drafter decision from a measurement record (2026-11-24, issue 182)

## What exists

`orchard/spec_break_even.py` (PR 190) turns three measured times into a verdict. Nothing in a run produced those times, so no package had a decision made from numbers. This change builds everything around the calculator that needs no board:

- `orchard/spec_decision.py` reads `evidence/spec-measure.json` from stage 2's directory and decides from it. The record holds the plain step time, the verify pass time, the accepted tokens per pass for each prompt kind, the sampling the numbers were measured at and where the prompts came from. Only `real-text` prompts are accepted, because acceptance measured on random tokens misleads. The verdict uses the prompt kind with the lowest mean acceptance and says which kind that was. Only a gain turns the drafter on; break-even and loss leave it off. A temperature of 0 marks the measurement as covering greedy decoding only, and the sentence says so.
- `Supervisor._spec_decision_record` runs when stage 2 passes. A record produces `evidence/spec-decision.json`, a ledger decision (`speculative decision`: enable, verdict, ratio, kind used, greedy only) and the file's evidence entry. A record that cannot be read or fails its checks produces a ledger decision (`speculative measurement refused`) with the reason, and the stage still passes. Without a record nothing is written.
- Stage 7's card builder (`package.write_card`) adds the decision's sentence to the card's limits for the required profile (`CardFacts.spec_statement`).

## Checked with recorded numbers

The published example is a 4.6 ms plain step and 2.7 accepted tokens per pass, a break-even pass of 12.4 ms. A verify pass of 8.0 ms gives a gain and the drafter on; 12.42 ms is break-even and the drafter stays off; 20.0 ms is a loss and the drafter stays off, which is the case the issue asks a test to show. With code at 5.0 and Thai at 1.0 accepted per pass, the verdict uses Thai and the drafter stays off.

## What is left for a board

- A measuring run that writes `evidence/spec-measure.json`: the plain step time and the verify pass time on the served model, and acceptance on a small set of real text in each prompt kind at the sampling the card claims. No such run is built, so the decision cannot be made for a real package yet, and the issue's "done when" (reproduce the decision for one drafter-on package from measured numbers) is not met.
- `run.sh` enabling the drafter only on a gain. That changes the package builder's handling of the source bundle's drafter settings and needs the measurement first.
- The folded issue 340 (speculative decoding for the Qwen3-Coder-Next coder) is untouched.
