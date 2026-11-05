---
name: full-port-serving
description: Stage 5 of a tt-orchard run started with --full-port, after the port passed stage 2 (and stage 3 when it ran). No agent works in this stage. The supervisor boots the port behind its chat server, replays an acceptance corpus recorded from the stage 1 reference, runs a passkey check and a canary, and refuses a server whose replay differs.
status: draft. Written on 2026-11-04 for issue 329, stage 5 half (docs/superpowers/specs/2026-11-02-full-port-supervisor-design.md). Nothing here has met a chip.
---

# Full port behind the chat server

## When this stage runs

The run started with `--full-port`, stage 0 classed the model `full-port`, and the port passed stage 2 (one board) and, when the run
configures more chips, stage 3. Stage 4 of the full-port route has its own work. This stage starts when stage 4 ends.

No agent step runs in this stage, so this file is for the operator and for anyone reading a run. The coder is not asked anything and is not
started for the stage.

## What the supervisor does

1. **Prepare.** It reads the port-check record of the last port stage that passed (stage 3, or stage 2 when stage 3 was skipped) and checks its
   sha256 against the one the ledger holds for the file that stage's gate accepted. It copies that stage's port files to `stages/5/port/` and
   puts the sha256 of each in the ledger. It copies the shipped templates `port_serve.py` and `port_check.py` (the worker the port runs in), records the
   acceptance corpus in `stages/5/corpus.json` and writes `port_serve_config.json` and `hw_test.json`.
2. **Test.** One command under a lease of the same number of chips the accepted stage used: `python3 -P -s stages/5/port_serve.py`. The
   supervisor writes `test-result.json` and `evidence/hw-test-output.txt` when it ends.
3. **Result.** The supervisor writes `result.json` from `evidence/port-serve.json`. The gate recomputes every figure in it.

## The acceptance corpus

`corpus.json` is built from the stage 1 reference files and the accepted record. It holds two entries.

- `teacher-forced`: the prompt, the first N reference ids, and the ids the accepted port gave when those were fed in (`expected_ids`). The server
  answers the port's argmax at each position while the reference ids are fed in, the way stage 2's check does it.
- `free-run`: the prompt, and the ids the accepted port gave when it chose its own tokens.

The server replays both through `POST /v1/completions` before it accepts any client. The bar is every position. A single position that differs ends the
test with `FAILED: the corpus replay matched M of N positions` and no `READY` line. The gate rebuilds the corpus from the files on disk and requires
the file to be the same, and it recomputes the replay's top-1 agreement with the reference against the stage 2 bar.

## The checks after the replay

- **Boot.** The readiness line (`READY orchard-port on 127.0.0.1:<port> pid <pid> corpus N/N positions`), the load time, the time from the start of
  the test to the line, and whether the kernel cache was warm. Warm means the cache directory (`TT_METAL_CACHE`, else `~/.cache/tt-metal-cache`) held
  at least one file before the test started. It is a count of files. The runtime's own hit counters are not read.
- **Passkey.** A five-digit key hidden in filler text at two or more lengths (a short one and the longest that fits the context, at most 8192 tokens),
  asked through `POST /v1/chat/completions`, twice each. The keys come from the corpus hash. The answer must hold the key, the server's token count
  must be near the requested length, and the two runs must give the same ids.
- **Canary.** The corpus prompt asked, a different prompt asked, then the first prompt again. The ids of the two asks must be the same and must be the
  first ids of the accepted port's free run.

## The rehearsal option

`--full-port-rehearsal` waives the field `coherent` and nothing else. In this stage `coherent` means that the passkey was answered at every length. A
model with random weights cannot answer it. The corpus replay, the readiness line, the canary, the passkey runs that must be accepted and repeat their
ids, and every provenance check still apply. The ledger records the waiver.

## What a pass does and does not show

It shows that the port, running in the worker behind the chat server, answers the recorded positions with the ids it gave when stage 2 or 3 checked it,
that those ids agree with the CPU reference at the stage 2 bar, that two requests do not disturb each other in the canary, and that the server accepts
the passkey prompts and answers them the same way twice.

It does not show correctness at positions or lengths that stage 1 did not record. The passkey prompts have no CPU reference, so a long prompt is checked
only for repeatability and for the key in the answer. It does not show speed (stage 6 does) or serving more than one request at a time (the server runs one
request at a time). The chat template is the model's own when `tokenizer_config.json` carries one and `jinja2` is installed; the record says `plain`
when it was not. For a model with random weights the corpus is token-level numeric agreement of the port with itself and with the CPU reference.
A pass is not a promotion. Publishing is a decision for a person.
