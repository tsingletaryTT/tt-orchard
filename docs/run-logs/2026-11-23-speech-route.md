# Speech route (2026-11-23)

Operator decisions of 2026-11-23: the harness grows a real speech route, and it never distributes a speech model's weights. It serves and checks what a bundle's
author published, and it carries the model's licence as the bundle's card states it.

This log says what was built, what the tests show, and what has not run. No chip, no gozer lease and no real `tt-model` was used.

## What exists

Before this change (PR 818) a Voxtral model stopped at the preflight with the block `speech-model`. With `tt-orchard bringup MODEL --speech-clips DIR` the block becomes
a run when the model is the upstream model of a known bundle (`mistralai/Voxtral-Mini-3B-2507`, `mistralai/Voxtral-4B-TTS-2603`). The design is in
`docs/superpowers/specs/2026-11-23-speech-route-design.md`. The run starts with the inputs `speech_route=<task>`, `speech_clips=DIR`, and optionally
`speech_prompts=FILE` and `speech_language=CODE`.

| Stage | Code | What it does |
|---|---|---|
| 0 | `Supervisor._speech_triage_body` | reads the bundle table, finds the clips, checks the reference interpreter, writes `delta.json` (class full-port, facts modality audio and serving container, a hazard for a non-commercial licence), records the selection and the licence line in the ledger |
| 1 | `Supervisor._speech_stage_1`, `orchard/speech_reference_run.py` | copies the clips, runs the producer in the reference interpreter: Voxtral transcripts of the clips (speech to text), or Whisper transcripts of the clips to calibrate the scorer (text to speech); writes `speech_reference.json` |
| 2 | `Supervisor._speech_stage_2`, `orchard/speech_hw.py` | under a one-board lease: `tt-model serve BUNDLE`, the checks of `speech_check`, every answer saved, `tt-model stop BUNDLE` in a `finally`; for text to speech the host then scores the saved audio with the scorer after the lease; writes `speech_result.json` |
| 3 to 7 | stage table | skipped, each with a reason |
| 8 | `build_bundle.py` | RESULTS.md section "Speech model on a chip" and RISKS.md section "Speech run", each with the licence line |

`orchard/speech_route.py` holds what the stages share: which run takes the route, the licence record, the clip and prompt handling, the reference record and its
gate, the result record and its gate. Every number a gate relies on is recomputed from the saved clips, the saved audio and the transcripts.

## Bars (all provisional)

- speech to text: word error rate of the bundle's transcript against the CPU reference transcript at most 0.05. The card's longest clip differed from its reference in 2 of 521
  words (0.004). Against the operator's expected text, when given, at most 0.10 (the limit `speech-check` uses).
- text to speech: word error rate of the scorer's transcript of the bundle's audio against the prompt at most 0.15 (the limit `speech-check` uses through an ASR server). The
  card's worst non-Hindi single clip is 0.11. The scorer has to score at most 0.10 on the operator's own clips in stage 1.
- the WAV checks of `speech-check` on the saved audio (24 kHz, 16-bit, mono, audible, 0.02 to 0.4 seconds per character), recomputed by the gate from the saved files.
- health, model list and the 4xx checks pass, the bundle started, and the bundle stopped with exit code 0.

The text-to-speech bar is absolute. No run of the upstream text-to-speech model under transformers has been verified (the card says "the original model, run on CPU"
and does not say how), so the route records no reference audio and does not claim a comparison with one.

## Licence

Every speech run records the bundle's licence id, whether it is non-commercial, and one sentence built from the bundle table: the model "is used under its licence, X, as the
bundle's card states it. This run served the bundle B and did not copy or republish any weights." The sentence is in stage 0's `delta.json` and ledger, in stage 1's
reference record, and in RESULTS.md and RISKS.md. The operator bundle prints the sentence derived from the table, never the record's own text, and RISKS.md says when
the record differs from the table. The sentence gives no opinion about what a licence allows.

No code path of the route calls `tt-orchard can`, `hf upload`, `tt-model push` or `tt-model publish`. A test reads the route's modules, the supervisor's `_speech*` methods and
the builder's `speech_*` functions and fails on a packaging or upload import, name or command; each rule is shown to fail against a source that breaks it.

## Tests

| File | What it holds |
|---|---|
| `tests/test_speech_route.py`, `test_speech_route_result.py`, `test_speech_route_stage1.py`, `test_speech_route_stage2.py` | records, gates, clip and prompt handling, producer config and environment |
| `tests/test_speech_hw.py` | the in-lease command against a fake `tt-model` and a fake server |
| `tests/test_speech_reference_run.py` | the producer with stand-in `transformers` and `torch` modules, and its config reading |
| `tests/test_speech_stages.py` | the stage table and the three gates |
| `tests/test_speech_route_preflight.py` | the base row and the bringup inputs |
| `tests/test_supervisor_speech_route.py` | whole runs through the supervisor, including the real bundle builder |
| `tests/test_speech_route_no_upload.py` | the licence and no-upload guards |
| `tests/test_speech_check_audio_hook.py` | the `on_audio` hook added to `speech_check` |

The reference producer is tested with stand-in library modules. It has not run on real weights.

## Not done, and owed

- A run on a chip. Stage 2 has run only against a fake `tt-model` and a fake server that copy what the two bundles' cards describe. The first run should serve one bundle under a
  lease with three clips of known text (speech to text) or the default prompts (text to speech), and it sets the real bars.
- A run of `speech_reference_run.py` on real weights (the Voxtral transcription request, and the Whisper pipeline) in the reference interpreter (transformers 5.24.0). The code is
  written against the public transformers API.
- Text to speech downloads the upstream model's snapshot in the preflight although no stage reads it. The route could skip that download.
- Questions about audio through `/v1/chat/completions`, languages other than the clips', concurrency, long audio and speed are not checked.
- The Mini bundle serves one request at a time and the TTS bundle runs in batches of 32; the route sends one request at a time to both.
