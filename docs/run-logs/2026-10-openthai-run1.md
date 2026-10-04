# Second-model run: iapp/openthai2.0-qwen3.8-27b, as hands-off as possible

Kept by Claude for the operator. Times are UTC unless marked. Numbers are measured unless marked.
Purpose: measure how often the harness needs a person on a model it has never seen. The first run
(Altworld/Hemmingway-1, see `2026-10-hemmingway-1.md`) needed a person many times and ended at
"ready for operator review". This run uses the code as it stands after that run. Nobody tunes the
harness for this model in advance.

## The model and why

`iapp/openthai2.0-qwen3.8-27b`: an open Thai knowledge, document-understanding and agentic model from a
Thai organization, a full fine-tune of Qwen/Qwen3.8-27B. Pinned revision
`a64f2b125f481320522ded36dcbc3c8c2737bb33` (the repository changed hours before the run, so the
revision is fixed). Apache-2.0, bf16, 14 shards, 56.5 GB, 64 layers, hidden 5120, vocabulary 248320.
Chosen because it is useful (few Thai models run on Tenstorrent hardware), different from
Hemmingway-1 (it keeps the base model's vision-language structure, so its tensor names and config
differ from a text-only export) and a hard case for the tokenizer (Thai depends on combining marks).
Not chosen: quantized and abliterated variants (not full bf16 weights), CC BY-NC models.

## Protocol

Setup done before the start (not counted): freed disk by deleting three scratch caches from earlier today
(34 + 34 + 27 GB: a base-weights cache, a prototype cache, an interrupted partial cache) and the 34 GB tensor
cache inside Hemmingway-1's stage 7 install copy (the venv and the evidence stay); downloaded the pinned
revision (14 shards, sizes verified, 53 GB). Free space on `/mnt/bonus` before the start: 208 GB.

Setup that is not counted (what an operator does before pressing Enter): choosing the model, downloading
its weights, freeing disk space, writing the run script, starting the supervisor with its flags.
The command is `/mnt/bonus/models/orchard-runs/openthai-run1.sh`. The operator chose not to isolate the
agents, so the run passes `--accept-credentials-visible`, as the first run did.

An **intervention** is anything done to the run or its environment after the supervisor starts:
a restart, a `control` command (pause, resume, abort), a code change or merge, killing a process, editing
a file, freeing disk, or any step that unblocks a stop. Each is logged below with the time and the reason.
Reading the ledger, logs and chip state is observation and is not counted.
Not an intervention: anything the harness resolves by itself (a stage retry, an escalation, a coder
recovery, a park and restore, a watchdog nudge).

Success: the run prints `ready for operator review` with **zero interventions**. Other numbers kept:
wall time per stage, stage retries and escalations, watchdog firings, and every stop that needed a person.

## Interventions

1. **16:05:18 Z: `control resume` after the first stop.** The run paused at 16:03:51Z in stage 0 (delta triage) after 9
   minutes. Attempt 1 (15:56:36Z) had looped on two `ls` commands for 17 turns and the `turn_repeat`
   detector nudged, then escalated it. Attempt 2 (escalated) spent 19 turns reading its own transcripts, the coder
   log and this model's repository (including an `adapter/` directory and the blobs) and wrote its first file on
   turn 20; the `no_file_written` detector (20 turns) paused the run at that moment, as the last rung of the
   ladder. No code was changed. Cause: stage 0 is still an open-ended agent step with no template, unlike
   stages 2, 4 and 7. Count: 1.

## Observations

(entries are added as the run proceeds)

- **2026-10-04T15:54:25Z run started** (supervisor pid 2624896, run directory `openthai2-run1`). Hands off from here.

- **16:23:44Z the run stopped a second time at stage 0, and this run ends here.** After the resume (intervention 1) the
  stage restarted from a fresh attempt; it used all 60 turns without a final answer ("stage 0 failed after
  escalation") and the run paused. What the agent did: it wrote 8 evidence files (config, files, generation
  config, license, tensor names, tensor shapes and dtypes, tokenizer, tokenizer encode test) and two scripts
  by about turn 35, never wrote `delta.json`, and spent turns 53 to 60 repeating nearly identical `grep`
  commands on an unrelated package's files. **Result of this run: not ready for operator review. 1
  intervention, then stopped at stage 0 again on a model the harness had never seen.** Findings:
  (1) stage 0 has no template, so the agent explores; (2) the `turn_repeat` detector needs identical
  arguments, and near-duplicates (the same command with a different trailing argument) escape it; (3) when the
  turn budget runs out with evidence on disk, nothing asks the agent to write the deliverable. Next: template
  stages 0 and 1 (scripts that do the measuring, the agent reviews and adds the hazards), catch near-duplicate
  repetition, add a wrap-up step at turn exhaustion, then rerun this model from a clean start. The first run's
  stage 0 and 1 results were produced under a hand-written delta-triage skill that took 27 and 8 minutes and
  needed no resume; this run shows that was not reliable on a second model.

## Attempt 2: the same model, after templating stages 0 and 1

Setup (not counted): the harness fixes above were built and merged (stage 0 and stage 1 template scripts, a
near-duplicate repetition detector, a wrap-up step at turn exhaustion; suite 1701 passed, 1 skipped), and the
run script `openthai-run2.sh` was written with a new run directory (`openthai2-run2`). Same model, same pinned
revision, same flags. The intervention count restarts at zero. Attempt 1's result stays as recorded: 1
intervention, stopped at stage 0.

**Interventions (attempt 2):** (none yet)
- **2026-10-04T17:37:08Z attempt 2 started** (supervisor pid 3074828, run directory `openthai2-run2`). Hands off.
- **17:37Z to 19:53Z attempt 2, stages 0 to 7: zero interventions, about 2 hours 16 minutes of wall time.** Stage 0 pass
  in 2 min 38 s (attempt 1 had run out of turns); stage 1 pass in 8 min with one watchdog nudge that resolved
  itself; stage 2 pass (cold boot, ready in 1970 s, agreement 0.906); stage 3 skipped by design; stage 4 pass
  on all three configurations (1 chip ready 1810 s, 2 chips 266 s, 4 chips 300 s, each agreement 0.906, one
  park for the 4-chip test); stages 5 and 6 skipped by design; stage 7 pass at 19:53:12Z (packages staged, an
  installed copy booted in the verify step). The delta triage found, with no hand-tuning: weights-only, 1199
  shared tensors with 0 differing in shape or dtype, the base model's vision tower kept (333 tensors), the
  same Apache-2.0 license, and the four standing hazards. Escalations 0, pauses 0, nudges 1.
- **19:53Z to 20:01Z attempt 2 stopped at stage 8 (operator bundle).** The first attempt (19:53:12Z) read stage
  7's evidence with 33 `cd ... && cat` commands, wrote nothing, and was escalated at 19:56:56Z after the
  `identical_responses` nudge and the 20-turn no-file watchdog; the escalated attempt read for 20 more turns,
  wrote nothing, and the watchdog paused the run at 20:00:59Z. Cause: stage 8 is the last stage that is still an
  open-ended agent step with no template; it must read every stage's results before it can write
  `RESULTS.md`, and a model that reads before it writes runs into the 20-turn limit. Its pass on the
  first model was not reliable. Fix: a template for stage 8 (a script that builds the bundle deterministically
  from the ledger and stage results; the agent reviews and adds risks), then resume this run to finish it
  (those steps count as interventions: a merge, a restart and a resume), and then a third model, fully clean.

## Attempt 2, continued: finishing at stage 8

**Interventions (attempt 2), counted from the stop at stage 8:**
- **2026-10-04 (about 13:40 local) merge of the stage 8 template** (`build_bundle.py`, skill rewrite, 26 new tests; suite 1728 passed, 1 skipped). Intervention 1.
- **`kill -9` of the paused supervisor (pid 3074828) and restart with `openthai-run2.sh`** (new pid 3924428). Intervention 2. The old lease was held by the dead pid; the new supervisor took over and booted the coder on board 1.
- **`control resume`** sent to the run directory. Intervention 3.

Interventions in attempt 2 so far: 3, all caused by stage 8 having no template. Stages 0 to 7 needed none.

- **2026-10-04T20:53Z attempt 2 reached `ready for operator review`.** Stage 8 passed on its first attempt after the
  resume, using the new template. Total attempt 2: stages 0 to 7 with zero interventions, then 3 interventions
  (merge, restart, resume), all at stage 8. The run never reached "ready for operator review" with zero
  interventions, so the zero-intervention test is still open.
  Checks after the run: ledger hash chain verifies (144 entries); bundle scrub reported 0 findings; grep of the
  bundle's text files for `hf_` tokens and the operator home path found 0 hits; the bundle is 191 MB with 39 files
  listed (sha256) in `stages/8/evidence/bundle-build.json`; all four chips free; no supervisor running.
  Contents (from `RESULTS.md`): weights-only; 1, 2 and 4 chip serving checks at top-1 agreement 0.906; two v6
  packages staged (p150 and p300); the p300 package passed its boot check and the p150 package was not boot-checked.
  Nothing was published.

- **2026-10-04 (23:55Z) creative sample from the 2-chip package.** At the operator's request the same "building
  remembering its 1930s past" prompt used for Hemmingway-1 was sent to the Thai model, in English and in Thai
  (thinking off, greedy, 900-token cap). Both answers are coherent and in persona; the Thai answer sets the
  building in Thailand and has a few doubtful phrases. Full text and notes:
  `docs/run-logs/samples/2026-10-04-openthai2-building.md`. This needed a fresh boot of the package on board 1
  (lease 823e9c, released and reset after); the boot took about 33 minutes warm, which is the same slow warm
  boot the benchmark agent measured (31 to 34 minutes).
