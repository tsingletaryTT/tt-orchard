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
