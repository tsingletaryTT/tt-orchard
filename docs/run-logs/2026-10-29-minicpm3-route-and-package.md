# MiniCPM3-4B through the text-model route, and its package (2026-10-29)

Prompt (coordinator, from the operator): fix the four review findings of PR 210 (issue 219), build the harness route for a text model (issues 61, 90, 206, 214;
issue 143 for the coder), run MiniCPM3-4B through it on board 1 only, and package it as a tt-model package. No upload, no publish, no merge. Plain English.

Times are UTC. Hourly updates are added at the end of each section as the work goes on.

## Update 1 (about 21:00)

- Issue 219 fixed in PR 242: `--layers` reaches both constructors (`run_chip.build_model`), `make_reference.py` builds the native model from the model directory
  and the weights file (`native_model.py`; `--native-model PATH` is the explicit alternative), `serve.py` takes `finish_reason` from the token that ended the loop,
  `summarize.py` documents `REF_DIR CHIP_RUN ...`. Each guard turned a test red under a mutation.
- The text route is written and tested without a chip (branch `text-route`): `orchard/text_reference.py` (generic recorder), `orchard/text_bars.py` (the bar for a chip
  run from CPU-only floors), `orchard/text_runtime.py` (the adapter boundary and the MiniCPM3 adapter), `orchard/text_stages.py` (the gates), the stage table
  in `orchard/stages.py` and the supervisor's `_text_body`. See `docs/text-runtime.md`.

## Update 2 (about 22:30)

First real run through the route (run directory outside the repository; board 1 only, leases through the supervisor under the name `claude:minicpm3-route`):

- Stage 0 and stage 1 passed with no coder boot. Stage 1 converted the weights, ran the float32 and two bfloat16 CPU runs and the float32 perplexity window.
- Stage 2 failed the first time: the chip script does not record `final_norm`, and the bar compared it as a required array. The bar now compares `final_norm` only when both runs hold it (found by the real run, then reported by Copilot as well). The run was resumed; the old stage directory was moved aside and the chip test ran again.
- A gozer claim failed once with "unreadable tt_serial" while another agent's lease was being reset; the supervisor stopped on the adapter error, released nothing it did not hold, and the same command resumed the run.
- Stage 2 (second attempt) and stage 6 passed. bf16: 16 of 16 teacher-forced tokens, perplexity 8.007 on the chip against 8.0574 on the CPU in float32 (0.6 percent lower), worst block error 2.4 times the CPU bfloat16 error (at layer 15), decode 33.87 tok/s median (33.86 to 33.88, three runs after one warm-up), 53.9 percent of a 62.8 tok/s floor. 8-bit MLP: 16 of 16, perplexity 8.047, decode 40.14 tok/s (41.5 percent of 96.8), and the numeric bar was not met: the worst block error is 15 times the CPU bfloat16 error and the first block over its tolerance is layer 14. The stage passes on the default variant; the card says the 8-bit variant missed the bar.
- Stage 8 built the bundle with the new "Text model on a chip" section.

Package boot check, first attempt: the server answered the first chat request and then failed the second with `MMIO per-op timeout`, and the chip needed a reset through gozer (acquire the unit, release it). It happened again with a sequence of four requests: the first worked and the second returned a 500. The cause is in `serve.py` from PR 210: it called the batched prefill, which allocates device buffers while the decode trace is live. PR 210's smoke test used one request, and the batched prefill was added afterwards. The server now feeds the prompt through traced steps by default (`--batched-prefill` keeps the old behaviour and says why it is off). With that change four requests in a row worked (a greeting, two short answers and a streamed answer with 47 events).

Copilot reviewed PRs 242, 245 and 247 as critical. Every finding was fixed with a red test (see the project log); a second review was requested once on each.
