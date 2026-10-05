---
name: serving-check
description: Stages 5 and 6 of a tt-orchard run. Check a served model from the outside (boot, passkey, canary), then run a short qualitative check and a benchmark, with every number labelled measured or TODO.
status: draft. A local tt-orchard copy, thinner than the vllm-integration, qualitative-check and benchmark-model skills it points to. It lives in tt-orchard.
---

# Serving check

## How the hardware test works

You do not run the server yourself. In the prepare phase, write a Python script in your stage
directory that does the whole check and exits, and give `python3 stages/<n>/<script>.py` as the
command in `hw_test.json`. The supervisor runs it on a leased board with `TT_VISIBLE_DEVICES`
set, with a deadline. The script must:

- start the server for the new model on the chips in `TT_VISIBLE_DEVICES`, using the recipe the
  earlier stages used (stage 0's path and stage 4's configuration);
- wait for it to be ready, with a time limit;
- run the checks below and write their outputs under `stages/<n>/evidence/`;
- stop the server before it exits, whatever happened. The supervisor checks that the chips are
  quiet afterwards and resets them. Never reset chips yourself.

In the finish phase, read `test-result.json` and the evidence, and write `result.json`.

## Stage 5: does it serve

    {"checks": {"boots":   {"pass": true, "seconds": 41.2, "evidence": ["stages/5/evidence/boot.log"]},
                "passkey": {"pass": true, "lengths": [2048, 32768], "evidence": ["..."]},
                "canary":  {"pass": true, "evidence": ["..."]}}}

- `boots`: the server reaches ready. Record the seconds to ready.
- `passkey`: a passkey (needle) hidden in filler text at two or more prompt lengths, the longest
  near the configured context, is answered exactly.
- `canary`: one fixed greedy prompt, asked twice, gives the same answer both times.

## Stage 6: how well and how fast

    {"numbers": [{"name": "decode", "value": 80.1, "unit": "tok/s/user", "label": "measured",
                  "workload": "prompt 128, generate 128, 1 user", "evidence": ["..."]},
                 {"name": "ttft", "value": null, "unit": "ms", "label": "TODO"}],
     "qualitative": {"summary": "...", "evidence": ["stages/6/evidence/qualitative.md"]}}

- Qualitative: five prompts suited to what the model is for (for a creative-writing model,
  writing tasks). Save the prompts and the full outputs, and write your reading of them in
  `qualitative.md`: wrong language, base-model autocomplete, repetition or broken formatting are
  failures to report.
- Benchmark: decode tokens per second per user and time to first token, at the workload the
  nearest supported model's package reports, so the numbers can be compared. Record the workload
  next to each number.
- Every number is labelled `measured` (the evidence file shows it) or `TODO` with a null value.
  The supervisor writes each one to the ledger as a measurement.

## Related skills

`vllm-integration`, `qualitative-check` and `benchmark-model` describe the full methods. Their
paths are listed in your context when they are installed.
