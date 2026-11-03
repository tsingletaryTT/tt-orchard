# Granite chunked prefill: follow-ups of issue 301 (2026-11-03)

Operator assignment: finish the hardware items left open in issue 301 after PR 296. Hardware: one chip of board 0 per run, each run under
`gozer run --chips 1`; every lease ended normally and the reset after it completed. Nothing wedged. Evidence is in
`docs/run-logs/evidence/granite-301-followups/`.

## Result

| Item | Outcome |
| --- | --- |
| 1. Smoke on the final runtime code | Done. 9 of 9 checks pass against the final code (below). The final-code smoke of 2026-10-30 (`granite-chunked-trace-safe/round3/smoke.json`, 34.6 tok/s) was already on record. |
| 2. Boot the staged package 0.3.0 with `bash run.sh` and its own venv | Done, with the smoke: 9 of 9. |
| 3. Agreement of the chunked prefill with the step loop past 1,024 tokens | Done at 1,536 and 2,000 tokens. |
| 4. Trace-on run of the old two-trace flow | Not run. The reason is below. |
| Label for the pre-round-2 smoke | Done. The old smoke was in `results/s4/round2-code/smoke.json`, and `results/s4/conditions.txt` pointed at `round1-code/`, which holds a benchmark. The pointer is corrected and a test checks it. |

## Items 1 and 2: the package, booted as published

- Package: revision `b5749cc0` of the Hub repository for Granite 0.3.0 (the recipe package), downloaded into a scratch directory. `sha256sum -c SHA256SUMS`
  reports no mismatch and `package-record.json` says version 0.3.0. The repository's `pkgsrc/runtime/granite_chip.py`, `granite_trace.py` and `pkgsrc/server/serve.py`
  are byte-identical to the package's copies, so this run is also a run of the final runtime code. `tools/smoke.py` differs only by the host-load fields added in PR 396.
- Venv: built from the package's own `requirements.txt` with `uv pip install` (CPU torch 2.15.0, numpy 2.5.x, tokenizers 0.23.3). ttnn came from the tt-metal checkout
  through `TT_METAL_HOME`, as the package's README says.
- Start: `bash run.sh` with `GRANITE_MODEL_DIR` pointing at the cached model snapshot, on a copy of the package (the pristine download was left alone).
  Ready line: `traced: true`, `chunked_prefill: true`, `ssd_precision: fp32`, context 2,048, `load_s` 13.4. `tools/smoke.py --port 20110`: 9 of 9 pass, `/health` 200.
  The server log held no warning about an active trace.
- Speed in this run is not a measurement of the server. Host load average was 23.0 before the start and 43.9 after the stop, other jobs were running on the CPU.
  Decode of the 64-token completion was 13.8 tok/s and the 1,481-token prompt took 4.96 s to the first token, against 34.6 tok/s and 1.24 s at host load 2.6 in
  the final-code smoke on the card. Per the earlier finding in this issue, decode follows the host load. The card's numbers are unchanged.

## Item 3: agreement past 1,024 tokens

`scripts/granite_prefill_bench.py agree`, trace on, chunk 128, `max_len` 2,048, the first N tokens of `docs/project-log.md` (the token-id sha256 is in each record).
The reference is the step loop on the same chip with the decode trace; the candidate is the server's prefill (`GraniteChip.prefill`: release traces, chunked blocks,
capture again). Each run reads four greedy tokens after the prompt.

| Prompt tokens | Logit cosine | Largest logit difference (logits up to) | Top-1 | Top-5 overlap | Next 4 tokens | Largest relative Mamba state difference | Host load before, after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1,536 | 1.00001 | 0.33 (41.7) | equal | 5 of 5 | equal | 2.6e-2 | 6.9, 28.4 |
| 2,000 | 1.00001 | 0.33 (42.5) | equal | 5 of 5 | equal | 2.1e-2 | 30.0, 20.1 |

Earlier lengths, same harness: 512 tokens 3.1e-2, 1,024 tokens 4.8e-2 (trace off), 129 tokens 8.1e-2. The state difference does not grow with length. The relative
state difference is the largest element difference over the largest element of the reference state, taken over every state tensor; it is the quantity reported earlier, so
the rows are comparable. The limits: one text, one prompt per length, one run each. The step loop is the reference, and the step loop itself differs from the
fp32 CPU reference by the amounts in `2026-10-29-granite-package.md`; this comparison does not measure that. Seconds in the records include the first-use kernel
compile for the new shapes (82 s for the chunked side at 1,536 tokens, 3 s at 2,000 tokens in a warm cache) and are not timings.

## Item 4: the old flow was not repeated

Two older flows are called "two-trace" in the records, and they behave differently.

1. The flow of the first trace-on attempt (commit `b507dc8`, project log 2026-10-30): the chunked blocks ran while the decode trace was alive. The timed run did not
   finish, the watchdog ended the process and the chip read all ones. `gozer release` exited 19 and the chip stayed unreadable; the log counts it as the
   third wedge of that day on that board and leaves the board to the operator. The only ttnn message was the warning that device buffers are allocated while a trace is active.
2. The flow of commit `10b1691` (release the traces, run the blocks, capture again, with both the full and the no-head trace captured): six trace-on runs completed
   with no hang (`granite-chunked-trace-safe/two-trace/`: 128, 512 and 1,024 tokens, two requests in one process, agreement), and the warning appeared once per process.
   The warning disappeared when the second capture was removed.

The records therefore already answer the question for flow 2: capturing a second trace while the first was alive allocated device buffers (the warning's source) and
did not wedge a chip in those six runs. What wedged a chip was flow 1, where block work allocates while a trace is alive. Repeating flow 1 means choosing to wedge a
chip; it wedged a chip in the only trace-on run of that flow, a failed reset leaves the chip unreadable until the operator acts, and the other board of this host
belongs to other work. That is not a cheap or safe check, and it would only restate the earlier finding. Item 4 stays open in the issue with that blocker. A repeat
needs a board the operator is willing to reset by hand afterwards.

## Not done

- Item 4's literal repeat (above).
- No speed number was taken at a quiet host load; the host load stayed above 20 throughout.
