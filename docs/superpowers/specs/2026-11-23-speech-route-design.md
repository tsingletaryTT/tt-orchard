# Speech route design (2026-11-23)

Operator decisions (2026-11-23): the harness gets a real speech route, and it never distributes a speech model's weights. It serves and checks what a bundle's
author published, and it carries the model's licence as the bundle's card states it.

## What exists

PR 818 recognises a speech model in the preflight (`orchard/speech_models.py`), names the published bundle that serves it and stops the run with the block
`speech-model`. `tt-orchard speech-check URL` checks a server that someone else started. No stage compares a speech model with a reference.

## What a speech run is

A speech model whose weights a published bundle already serves has no port to write. The run has three questions to answer: does the bundle answer what a
CPU reference answers, does it keep answering when started the way an operator starts it, and what are the licence terms the result is used under. That is
the shape of the text-runtime route (`docs/text-runtime.md`): no coder boots, no stage reads a port module, and stages 3 to 5 have
nothing to do. The scoring route needs the coder's port as its subject, and a speech run has none.

## Choice: follow the text-runtime pattern

Two other shapes were possible.

- Pattern of the scoring route: stage 2 runs a check on a port module the coder wrote. A speech run has no port module, so the stage would have nothing to run.
- A standalone command only (what PR 818 ships): no ledger, no gates, no resume, no operator bundle. It stays, because it is the check a person runs by hand.

The chosen shape: a run started with the input `speech_route=<task>` (`speech-to-text` or `text-to-speech`) takes a stage table like `TEXT_TABLE`. The preflight
sets the input when the model is a speech model and a published bundle for that task is known, so the `speech-model` block becomes a route exactly where its
pieces exist. A speech model with no known bundle keeps the block.

| Stage | What runs | Hardware | Gate |
|---|---|---|---|
| 0 | triage from the card and the bundle table: task, bundle, licence | none | `gate_speech_triage` |
| 1 | CPU reference, in the reference interpreter: a pinned transcription of the clips (speech to text) or a calibrated scorer (text to speech) | none | `gate_speech_reference` |
| 2 | the bundle served under a lease, checked with `speech-check`'s own functions, the answers saved | one lease | `gate_speech_check` |
| 3 to 7 | skipped, each with a reason | | |
| 8 | the operator bundle, with the section "Speech model on a chip" and the licence line | none | `gate_bundle` |

No stage starts the coder.

## What each stage records, and what the gates recompute

Stage 1, speech to text: the operator's clips (copied into the stage so the hashes can be checked), and the transcript the reference model gave for each, with
the versions that produced it. The expected text the operator gave for a clip, when there is one, is kept next to it.

Stage 1, text to speech: the prompts, and a calibration of the scorer. The scorer is a speech-to-text model on CPU. It transcribes the operator's clips, and its
word error rate against their expected text has to be low, otherwise a poor score on the bundle's speech would say nothing about the bundle. The reference model's
own audio is not recorded: no run of the upstream text-to-speech model under transformers has been verified (the card measures "the original model, run on CPU"
without saying how), so the route does not claim one. The result says the bar is absolute and says why.

Stage 2: `python3 -m orchard.speech_hw` runs inside the lease. It starts the bundle with `tt-model serve`, reads the address from the command's output, calls the
checks of `orchard/speech_check.py`, saves every answer, and stops the bundle in a `finally`. For text to speech the audio files are saved and the host scores
them afterwards with the same scorer, outside the lease. `speech_result.json` holds the answers. The gate recomputes every number from the saved answers and the
stage 1 record: word error rate with `speech_check.wer`, the WAV measurements from the saved files, the hashes. A recorded number that disagrees with the
recomputed one fails the gate.

Bars, all provisional until the first chip run:

- speech to text: the bundle's transcript against the reference transcript, at most 0.05 word error rate. The card's longest clip differed in 2 of 521 words (0.004).
  Against the operator's expected text, at most 0.10, the same limit `speech-check` uses.
- text to speech: the scorer's transcript of the bundle's audio against the prompt, at most 0.15, the limit `speech-check` uses through an ASR server. The card's
  worst non-Hindi single clip is 0.11.
- structure: health, model list and the 4xx checks all pass, and the bundle was stopped.

## Licence

Every speech run records the bundle's licence id, whether it is non-commercial, and a line that says the result is used under that licence's terms, as the
bundle's card states them. The line is written once (`speech_route.license_line`), goes into stage 0's record and the ledger, and into the operator bundle's
RESULTS.md and RISKS.md. The route never packages or uploads weights: no code path of the route calls `tt-orchard can`, `hf upload`, `tt-model push` or
`tt-model publish`, and a test reads the route's source to hold that. The text does not give a legal opinion.

## Out of scope in this PR

No chip ran. Stages 1 and 2 are exercised with fake interpreters and fake servers; the first run on a bundle is owed. `speech_reference_run.py` is written against
the public transformers API for Voxtral and Whisper and is untested on real weights. Questions about audio through `/v1/chat/completions`, concurrency, long audio
and speed are not checked.
