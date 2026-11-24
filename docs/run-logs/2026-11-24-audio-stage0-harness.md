# Stage 0 of an audio run without the coder (2026-11-24, issue 143)

## What changed

A run for an audio model booted the local coder for stage 0, held a board and a cold start for it, and then stopped or went on to stages that ask the coder nothing. The preflight already knows a model is audio before stage 0 (`orchard/audio_facts.py`). The change uses that:

- `bringup` records two `run_start` inputs when the base row says the model is audio, names its base and the base's snapshot is known: `audio_model=1` and `audio_base=<the base repository id>` (`orchard/cli.py`, `_audio_stage0_inputs`). A resume keeps what the run started with, so a run from before this change keeps the agent stage 0.
- `stages.AUDIO_STAGE_0` is stage 0 as supervisor code (`harness=True`, `needs_coder=False`) with the same gate as the agent stage. `spec_for(0, ..., audio_model=True)` returns it, after the stock-runtime stage 0.
- `Supervisor._audio_triage_body` copies the shipped `delta_triage.py` and its side file into the stage, writes `triage_config.json` from the run's inputs, runs it with the agent's environment (no token, HOME inside the run), and runs the gate. `orchard/audio_stage0.py` holds the selection, the preparation and the run.
- The template stays the judge. When its `delta.json` has no `facts.modality` of `audio`, the stage fails with a reason that says the template reports a text model. A failed harness stage is not escalated; the run pauses for the operator.

## Check on real snapshots

The stage was run by hand on this machine's cached `stabilityai/stable-audio-3-small-music` (revision 8bdfb53c) against `stabilityai/stable-audio-3-small-music-base` (revision 4249e917), with `audio_stage0.prepare` and `audio_stage0.run_template` and then `stages.gate_delta`: the template finished in 0.2 s, wrote class `weights-only`, path `weights-only`, modality `audio`, and the gate passed with no reasons. The agent stage 0 wrote extra evidence files by hand; the gate needs none of them.

## Tests

`tests/test_audio_stage0_harness.py` (44 tests) runs the supervisor on small audio snapshots with the real template: zero coder starts through stage 0 and through a complete run, a resume after the pause, a run killed at the start of stage 0, an abort, a text model that fails the stage, a missing input, a template that exits with an error or does not finish, an edited template copy that the gate refuses, and the environment the template gets. It also covers the stage table, the selection from the ledger and the command's inputs. The strict `xfail` of `tests/test_audio_stage0_coder.py` is gone: its wanted behaviour is the first test of the new file, and the old file pins that a run without the input still boots the coder once.

## What is left

- The stock-runtime stage 0 is also supervisor code and still boots the coder, because its spec keeps `needs_coder` true. The same flag would skip it; it needs its own test matrix and is not part of this change.
- The agent stage 0 stays for every model that is not audio.
- No chip is used by any of this.
