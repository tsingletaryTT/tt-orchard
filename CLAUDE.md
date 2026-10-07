# tt-orchard project log

## Original prompt (2026-10-01)
"What would it take to have a suite of skills that can handle moving from 4 chips back down to 2
chips or even 1 chip on the Quietbox 2? ... hand off to a CPU-based model ... 'hold my beer while I
test this new model'." Then widened: take a new model release (for example Qwen3.7) and bring it up
using the skills, with open-source models only. The run ends at an operator review bundle.

## Key decisions
- Supervisor is a state machine in code. Each stage starts a fresh short model context from the ledger.
- Lease handling goes through an adapter. tt-gozer owns leases. The supervisor holds each board lease
  under its own pid (`gozer acquire --owner-pid`) through a model swap and never releases it, so no
  other agent can take the board mid-swap. gozer gains `reset <lease>` (reset in place) and an ownership
  check that counts the owner's child processes. The earlier `yield`/`redeem` reservation design is the
  documented fallback (spec section 8). Changed on 2026-10-02 after reading gozer's `keymaster.py`,
  `gatekeeper.py` and `queue.py`.
- The watchdog reports on agents it did not launch and acts only on agents it launched.
- The supervisor never publishes. A command runner refuses push, publish and reset commands.
- Config is TOML because the supervisor uses only the standard library.
- The command runner (`orchard/runner.py`) is a fail-closed lexer. It reads a small subset of shell
  and refuses everything else, with a message that names the construct and says how to rewrite it.
  The first version split commands with `shlex`. Review showed it could be bypassed with comments,
  shell keywords, `eval`, shells reading stdin, and wrapper options such as `sudo --user`. The
  lexer replaced it. The runner is best effort. Credential removal and read-only mounts are the
  outer layers and belong to plans 3 and 4.
- The ledger (`orchard/ledger.py`) has a single-writer lock, rolls back a failed append, keeps each
  torn tail in its own sidecar file (synced before the cut), and reports any malformed content as
  `LedgerCorrupt`. A retried stage appears once in the `completed` list.
- Tier config validation is strict (`orchard/tiers.py`). It refuses remote endpoints, unedited
  `CHANGE-ME` values, unknown keys and tables, and stages nobody owns. Stage 4 needs a `plan` tier
  (spec: large plans, small runs). Stages 2, 3 and 4 need a `diagnose` tier. Stage 7 has neither.
  A required `[escalation]` table with `default = "<tier>"` names the tier that takes over when a
  stage without a `diagnose` tier escalates.
- The sizing tool (`orchard/sizing.py`) labels what it measures. `load_cold`, `prefill_cached` and
  `decode_complete` say how far a number can be trusted. `prompt_chars` and `prompt_sha256` show
  whether two runs used the same prompt. `mem_available_bytes` is read with the measured model (and
  any other loaded model) resident. In a ledger entry, `label` covers every field the tool reports.
  A `null` under `measured` means the source did not report it. `TODO` entries are placeholders for
  a measurement not taken, written by the operator-bundle stage.
- Tiers (operator, 2026-10-02): large = Qwen3.8-27B on all 4 chips (both boards); small =
  Qwen3.8-27B on 2 chips (one board); a CPU tier chosen by measurement. While the large tier is
  loaded no board is free, so every hardware stage parks it. `handoff.decide_park` makes that call
  from the boards the coder holds and the boards a stage needs.
- Park and restore is a step-by-step state machine. Each step writes a ledger entry; a restart
  replays the ledger and then believes the machine (docker, ps, gozer status) where they differ.
  After a crash the old lease belongs to a dead pid. If the park had sent its stop, recovery stops
  the coder if it still runs, takes a new lease, resets it and restores. A park killed before
  `stop_sent` is abandoned: the coder never left, so it stays up under the dead supervisor's lease
  (plan 4 re-leases it). A recorded stand-in pid is signalled only if its process start time still
  matches the ledger.
- The watchdog acts only through an injected actuator, which plan 4 supplies. Each ladder rung is
  written to the ledger before it is acted on, so a restart cannot repeat a rung.
- Plan 4 (2026-10-02): the supervisor starts the coder itself under its own owner-pid lease, because a
  lease cannot pass from the controller's pid to the supervisor's. It stops the coder and releases the
  lease at the end of the run and on abort. The CPU tier (ollama) is an external stand-in that the
  operator runs.
- A hardware stage is three parts: a prepare step writes `hw_test.json` and `handoff.json`, the
  supervisor runs that one command under the lease (parking the coder when no board is free), and a
  finish step writes the result from the test output. `test-result.json` is the resume marker.
- When the named tier is down, a chip tier with the same model serves the step, and the ledger says
  so. On this box only the 4-chip large server runs, so it serves every step.
- Agent shells get an allow-listed environment with HOME inside the run directory. Read-only mounts
  are not built.
- Portable paths (2026-10-03, prompt: "Someone else cannot run the harness as written"). Skills
  name machine paths as placeholders (`{{ORCHARD_DIR}}`, `{{HF_HOME}}`, `{{OPERATOR_HOME}}`,
  `{{TT_MODEL_ROOT}}`, `{{CACHE_ROOT}}`, orchard/paths.py). The context fills them in and refuses
  an unknown one. `run` gained `--cache-root`, `--hf-home` and `--operator-home`; run_start records
  the values and a resume keeps them, as with `--required-chips`. A repo test fails on any `/home/`
  or `/mnt/` path in orchard/skills or orchard/package_templates. hardware_check.py lost its
  machine defaults: `--gozer` and the child paths come from flags or `ORCHARD_*` variables.

## Layout
Spec: `docs/superpowers/specs/`. Plans: `docs/superpowers/plans/`. Code: `orchard/`. Tests: `tests/`.

## Status
Plan 1 is implemented through Task 4 (ledger, runner, tiers, sizing). Task 5 (run the sizing tool
against a real ollama, which downloads models and loads the host) was not run. It needs the
operator. Plan 2 is merged to `main` in tt-gozer. Plan 3 (adapters, server control, park and restore, watchdog) is implemented on branch `plan3-supervisor-behavior`. Plan 4 (stage machine, agent steps, supervisor loop, draft stage skills) is implemented on branch
`plan4-stage-machine`. The Hemmingway-1 run in the runbook has not been run. Plan 5 (stage 7: a v6
thin package, its boot check, card and publish commands as text) is implemented on branch
`stage7-wiring` (modules merged to main earlier); its hardware run has not been done.

## Open decision for the operator
No adversarial search for bypasses of the command runner has been done. Decide before plan 4 ships:
sanction an adversarial review, or accept best effort plus the outer layers (spec sections 10 and 14).
Also deferred: backticks or `$(` inside single quotes are refused as substitution (a false denial
that touches the lexer), tightening `tests/test_tiers.py`, splitting `runner.py` into modules, and a
two-process lock test.

## Log
- 2026-10-01: spec approved; plan 1 (ledger, runner, tiers, sizing) written.
- 2026-10-01: plan 1 executed with subagents. Tasks 1 and 2 by sonnet, Task 3 by haiku, Task 4 by
  sonnet, review by opus and sonnet, final review by fable.

- 2026-10-02: plan 3 (supervisor behavior). Prompt: "write plan 3: supervisor behavior (adapters, park
  and restore, watchdog)". Executed task by task. Notable moment in Task 14: the replay found a second
  five-call repeat in the loop chat (03:40 to 03:53Z, 5281 thinking tokens), so the identical-response
  detector fires twice there and the thinking cap fires five times. Task 14 mutation 3 as first written
  (`THINKING_CAP = 16000`) could not fail, because the 16861-token response had a tool call; the
  plan now uses 1500. Task 4: the five gozer contract tests ran against the real gozer with fake roots
  and none skipped. Task 7: the live server test ran and did not skip; one mutation first passed because
  of a stale `.pyc` (same file size, same second), so every mutation run now clears `__pycache__`. Task 15
  wrote the park-check driver and its tests. The controller then ran the park check on real hardware
  (2026-10-02, board 1, default options, fake servers): exit 0, resets of 41.676 s and 41.657 s. The only skip in
  the suite is the opt-in replay. The plan was revised before execution after a review
  (`.superpowers/plan3-review.md`, 1 critical and 10 important findings, all accepted): a tripwire gozer
  in the park-check tests, abandoning a park killed before its stop, recording the stand-in pid first,
  an explicit abandoned step, a single-tenant adapter that can be rebuilt after a crash and leases whole
  boards, salted fixture hashes, and per-attempt evidence files.
- 2026-10-02: plan 3 fix pass after the final review (`.superpowers/plan3-final-review.md`; report in
  `.superpowers/plan3-fixpass-report.md`). Prompt: fix C1 and I1 to I4 by test-first steps, one commit each,
  plus the minors. Key decisions: single-tenant tests get a `FakeRun` that fails on any call; the stand-in
  is spawned, recorded, then waited for; a recorded pid is signalled only if its start time and boot id
  still match; recovery and restore stop checks each got a test that fails without them.
- 2026-10-02: plan 4 written. Prompt: the smallest loop that lets the harness bring up a model whose
  architecture matches a supported one, stage 0 to 6 plus a minimal stage 8 bundle, first target
  Altworld/Hemmingway-1. Every code block was run from the plan text in a scratch copy before hand-over.
  Found while checking it: counting every coder start against a cap paused the run at stage 5, because
  each park restarts the coder; the cap now counts cold boots (a start slower than 10 min). The run
  also left the coder serving under a dead pid's lease after it finished, so the run now releases the
  hardware at the end.
- 2026-10-02: plan 4 implemented (tasks 9 to 11 in this batch). The suite has 962 passed and 1 skipped
  (the opt-in replay). The supervisor was killed after every ledger event of a scripted bring-up:
  162 kill points with the coder on four chips and 107 with the coder on two. Each resumed run
  finished with the same stage results, bundle and stage 6 numbers, and no lease left behind. The
  Hemmingway-1 hardware run has not been run.
- 2026-10-02: plan 4 fix pass after the final review (`.superpowers/plan4-final-review.md`; report in
  `.superpowers/plan4-fixpass-report.md`). Prompt: fix C1, C2 and I1 to I5 test-first, one commit each,
  with a mutation per new guard; the operator's rulings fixed the scope. Key decisions: the runner now
  refuses gozer writes (gozer run included), tt CLI resets, docker and tt-model control, kill and
  systemctl stop, ssh/scp/rsync to a host, uploads and curl/wget writes; agent shells get a no-chip
  device mask (`0000:ff:00.0`; checked on hardware 2026-10-02: a device open fails with a RuntimeError); a preflight refuses to start while credential
  files are visible unless the operator accepts it; SIGINT and SIGTERM take the abort path and other
  errors release the hardware (as `except` clauses, so the kill test's Crash still models SIGKILL);
  a coder boot that never reached "coder started" is finished before any stage. Shell-running tests
  use a stub PATH. The suite has 1141 passed and 1 skipped. Notable moment: a mutation that removed
  the signal handlers made SIGTERM kill pytest, so the mutation helper crashed before restoring the
  file; `git checkout` in the next run restored it, and that run was repeated.
- **2026-10-03 canary and first-boot check** (branch `canary-sanity`). The first hardware run blocked at the
  coder canary. The canary now sends `chat_template_kwargs: {enable_thinking: false}` and explains a null
  `content` with `finish_reason: length` as a reply cut off mid-reasoning. A coder's first start in a run
  (no baseline canary yet) is also asked "What is 7 times 6?" and must answer with text containing 42, or
  the run blocks and does not retry. A restart with a baseline skips the question; a resume after a block
  asks it again. Each guard was checked by mutation: remove it and a test goes red.

## Notable moments
- The `shlex` runner was shown to be bypassable (comments, keywords, `eval`, shells on stdin,
  wrapper options). Two fix rounds replaced it with the lexer. Later reviews were limited to
  reading code and running the tests.
- One reviewer tried an adversarial bypass search and was stopped by a safety classifier. The
  search was not dispatched again. The decision is open (see above).
- The haiku implementer of Task 3 needed four fix rounds and twice skipped the request for a full
  mutation table. A sonnet implementer finished it, and Task 4 went to sonnet from the start.
  Lesson: ask for a mutation table explicitly, then verify it by running the mutations on a copy.
- The final review found that the stage 4 rule misread the spec. The spec says stage 4 is "large
  plans, small runs". The fix added the `plan` key and the `[escalation]` table. It was cheapest to
  change before any operator config existed.
- The final review also found ledger exceptions other than `LedgerCorrupt`, zsh lexed as bash, and
  redirect targets that lost their `$` flag. All three are fixed.
- 2026-10-02: the hardware-check driver (`orchard/hardware_check.py`, built and reviewed with
  subagents; a safety review found five Critical problems before any real run, and a narrow check
  found two more) ran on board 1 with the tt-tnt agent's agreement: 22 checks passed. The 900 second
  idle test, board 0, a container server and two drivers at once are not done. Details in the spec,
  section 14, item 2.
- 2026-10-02 (later): the full hardware validation is done on both boards. Board 1 passed with the full
  960 s idle (the owner-pid lease survived a reconcile), board 0 passed, two drivers overlapped, and a
  container-server check on board 0 passed (warm boot 20 s, stop 1.6 s, reset 41.7 s). Findings: a
  reset opens every device on the box, so the other board looks busy for about 42 s; on this box a
  `tt-model serve` container's server runs as the same user and shows as `HELD-FOREIGN`. The tt-tnt
  agent yielded the four chips at a checkpoint and was told when the hardware was free again.
  A lapse of mine: I stopped the H5 lease holder with `pkill -f "^sleep 3600$"`, which matches any
  process of the same user with that command line; use the recorded pid instead.

## 2026-10-02: tier models chosen
Operator decisions: large = Qwen3.8-27B on all 4 chips; small = Qwen3.8-27B on 2 chips (one board); CPU = `qwen3-coder:30b`
(ollama, models in `/mnt/bonus/models/ollama`). Measured with chips idle: qwen3-coder:30b decode 14.1 tok/s, prefill 93 tok/s
(1,605-token prompt, short 92-token answer); dense 27B bf16 on CPU decode 0.70 tok/s, prefill 35 tok/s at 512 tokens with the slow
fallback kernels (the 64-token figure measured disk paging and is not valid). Because the large tier holds both boards, every hardware
stage needs park and restore. First target model: Altworld/Hemmingway-1 (spelled with two m's); the operator wants the harness, not
Claude by hand, to do the bring-up. Hand-done stage 0 findings are a reference answer only, at
`/mnt/bonus/models/hemmingway-1/work/stage0-reference.md`.

## 2026-10-03: truncated replies no longer end a step as done
On a real run the Qwen reasoning model spent all 8192 max_tokens thinking on turn 8 and returned finish_reason "length" with no
content and no tool calls. The loop took "no tool calls" as the final answer, so the stage ended with no output files and was
escalated. `AgentStep.run` now reads finish_reason: a "length" reply without tool calls is left out of the conversation, logged
(every log record has `finish_reason`; a `notice` ledger entry with `watchdog=True` is added), and answered with a user message
asking for a short think and a tool call. A second one in a row ends the step with status "error". The supervisor treats "error"
like any step that did not finish: the first time it escalates the stage, and a failure after escalation pauses the run.

## 2026-10-03: empty replies, a larger token budget, and one gate-feedback continuation
Prompt: the live Qwen3.8-27B run failed stage 1 three times with "reference.json is missing". On the third attempt the agent
worked 48 turns and wrote real evidence. Turn 48 was cut off at max_tokens (already handled). Turn 49 came back empty: 159
completion tokens of reasoning, content "", no tool calls, finish_reason "stop". The loop took that as the final answer, the gate
failed, and the escalation started a fresh context. Three changes, each test-first with a mutation check (branch
`agent-continuation`):
- A reply with no tool calls and no text, whatever its finish_reason, is nudged like a cut-off reply. Both kinds share the
  count of two in a row before the step ends with status "error".
- `AGENT_MAX_TOKENS` is now 16,384. Thinking tokens count toward it, and 8,192 was used up by reasoning in 3 of the failed
  replies. At about 80 tokens/s a full reply takes about 200 s.
- When a step ends "done" and the exit gate fails, the supervisor records a "gate feedback" decision and gives the same
  conversation one continuation (`AGENT_CONTINUATION_TURNS` = 20, its own log file) with the gate's reasons word for word. It
  happens once per run of the stage body and never after another status. A kill during the continuation loses that
  conversation; the resumed stage starts fresh and may use its own continuation (the e2e kill test covers this case).

## 2026-10-03: the stage machine follows the path stage 0 chose
Prompt: make stages 2 and 3 path-aware for a weights-only delta, using the draft `weights-swap-check` skill (branch
`weights-swap`, TDD, one commit per change). On `weights-only`, stage 2 uses that skill and the new `gate_weights_swap`
(serves, coherent, `n_tokens` >= 16, `top1_agreement` >= `SWAP_TOP1_MIN` = 0.6 (raised to 0.85 later that day, see below), positive `server_ready_s`, evidence files);
stage 3 is recorded as skipped the way stage 7 is, and the run goes on to stage 4. `full-port` and an unknown path keep the
old table. Decision: stage 0's passing `stage_end` records the path, and `run_path` reads it from the ledger before each stage,
so a resume makes the same choice even after an agent edits `delta.json`; older ledgers fall back to `delta.json`. No per-path
budget was added: the stage 2 budget (14,400 s) already exceeds the skill's 2,400 s test deadline. Mutations (swap skill on every
path, stage 3 skipped on every path, a low `top1_agreement` accepted, an unknown path read as weights-only) each turned tests red.

## 2026-10-03: swap templates and two watchdog detectors
Prompt: replace "the model writes a 400-line script from a description" with tested templates it copies, and add two
watchdog detectors for the loops a live stage 2 step fell into (branch `swap-templates`, worktree, tests first, a mutation
per guard). On that run the agent spent 60 turns grepping vLLM source for an unrelated timeout and wrote nothing; part of it
was two grep commands repeated in each of 5 turns.
- `orchard/skills/weights-swap-templates/`: `prepare_swap.py` builds `model-dir/` and the edited `run.sh` from
  `swap_config.json`, and exits 2 when an expected edit does not happen exactly once. `serve_and_compare.py` refuses a
  non-empty tensor cache without a matching `.orchard-model` marker (exit 3), serves, measures coherence and teacher-forced
  top-1 agreement, and stops the server's process group in a `finally`. Tests run both on a stdlib fake server and a
  WordLevel tokenizer. Mutations seen red: `proc.pgid`, no `finally` stop, no marker check, any text counted as a match.
- The skill now says: find four facts, write `swap_config.json` first, copy the templates from the main checkout, run
  `prepare_swap.py`, write `hw_test.json` and `handoff.json`. The finish phase writes `serves` false with the failure text
  when the test failed.
- `NoFileWritten` (`WRITELESS_TURNS` = 20, a choice) nudges once when no file was written for 20 turns, and re-arms after
  a write. `TurnRepeat` (`TURN_REPEAT_N` = 3) fires when one turn's set of calls repeats in 3 turns in a row. `Event`
  gained `wrote` and `turn`, fed by `AgentStep`. Both stay quiet on the committed transcript signatures. They are not in
  `transcript_detectors()`, so the opt-in replay of `~/.qwen` (not run here) is unchanged.
- Correction (same day): the hand prototype that the skill was written from served the BASE Qwen3.8-27B weights. The TT
  runtime takes its weights directory from `MODEL_WEIGHTS_DIR`, then `HF_MODEL`, then the config path, and the bundle's
  `run.sh` exports `HF_MODEL=Qwen/Qwen3.8-27B`. Its 25 of 32 (0.78) agreement compared the base model on the chip with the
  Hemmingway-1 CPU reference. Re-run with `MODEL_WEIGHTS_DIR` set: 30 of 32 (0.94), and the free-run text matched the
  CPU text (`docs/run-logs/2026-10-hemmingway-1.md`, 17:45Z and 17:55Z entries). The templates now set
  `MODEL_WEIGHTS_DIR` and `HF_MODEL` in the server's environment and rewrite the run.sh copy's `HF_MODEL` line; the skill
  lists this as its fourth fact. `SWAP_TOP1_MIN` went from 0.6 to 0.85, because the base weights passed 0.6. Both
  numbers come from one prompt of 32 tokens, so 0.85 is a choice with a thin margin. Mutations seen red: drop either
  variable, leave the `HF_MODEL` line at the nearest model id, set the bar back to 0.6.

## 2026-10-03: no gate feedback after a failed hardware test
Prompt: on a live run the stage 2 hardware test failed, the finish step honestly wrote `serves` false, and the
gate-feedback continuation told the agent to make `serves` true. It could not do that honestly, wrote nothing
for 20 turns, and `no_file_written` paused the run. Branch `no-feedback-after-failed-test`, TDD, one commit per
change:
- In a hardware stage, when `test-result.json` shows the test did not succeed (exit code other than exactly 0,
  `timed_out` true, or an unreadable record), the supervisor records "no gate feedback: the hardware test
  failed" with the exit code and timeout flag and ends the stage through the usual fail or escalate path.
  After a test that succeeded, the one continuation works as before.
- Found by the kill-around test: a kill between the escalate entry and the stage_end resumed the escalated
  attempt from the failed test record, and the run paused. That attempt now moves the directory aside and
  tests again ("not resuming from a failed hardware test").
- The finish task and the weights-swap-check Finish section say: record the failure (a `failure` field with
  the text from `hw-test-output.txt`, `serves` false, unmeasured numbers null), then stop. The swap gate
  still fails that result, and its serves reason now quotes the failure text.
- Mutations seen red: always continue, never continue, a missing exit code counted as success, no resume
  guard, no failure text in the gate reason, the old finish text.

## 2026-10-03: stage 4 tests each chip configuration on the weights-only path
Prompt: stage 4 must show the new model working on the configurations the packages ship for: 2 and 4 chips required, 1
optional (plan `docs/superpowers/plans/2026-10-03-multichip-stage-4.md`). Decisions:
- One hardware test per configuration, listed by the prepare step in `hw_tests.json` and validated by
  `orchard/hwtests.py`; the supervisor builds each command, runs the list in order of chip count, each under its own
  lease, and records each test in `tests/<N>/test-result.json`. `tests/plan.json` is the resume marker. The gate
  (`gate_mesh_swap`) counts a pass only with that record, and the supervisor refuses a record whose sha256 differs from
  the sha256 the ledger holds.
- The 4-chip test (`serve_and_compare_container.py`) runs the printed `docker run` itself with exact-count argv edits:
  fresh tensor cache, empty `/hf`, model directory and blobs mounted read-only at their own paths, `MODEL_WEIGHTS_DIR`
  and `HF_MODEL` set, only the leased device nodes, a run label. `tt-model serve` takes no extra docker arguments.
- A test that needs the coder's boards leases any further board first, then parks; a failed test still restores the
  coder. A container left by a test is stopped by label; one that survives blocks before any release. A cache whose last
  test did not exit 0 is moved aside before its next test.
- After a failed hardware test there is no gate feedback (main, `no-feedback-after-failed-test`). For a list, "failed"
  means a test that must pass did not: a required configuration's, or any listed one when no counts are required. An
  optional configuration that failed still gets gate feedback (one continuation) for a malformed `result.json`.
Not measured: the 4-chip cold conversion time, the 4-chip coder's restart, the 1-chip cache size.

## 2026-10-03: stages 5 and 6 are skipped on the weights-only path
Prompt: skip stages 5 and 6 on the weights-only path the way stage 3 is skipped (branch `stage7-wiring`, TDD). `spec_for`
returns a skipped spec for 5 and 6 on that path, and the supervisor writes `stage_start {skip: true}` and
`stage_end {result: "skipped", reason}` as it does for stage 3. Stage 5's reason: stage 4's serve-and-compare tests boot
each configuration, compare the output with the CPU reference and check that it is coherent. Stage 6's reason: the operator
deferred the qualitative check and benchmark. The full-port path and an unknown path keep both stages. Supervisor tests
that used stage 5 or 6 for escalation, numbers or a coder death now use stages 1, 2 or 4, or a full-port run. The
kill-after-every-ledger-event test covers the new skip entries.

## 2026-10-03: plan 5, stage 7 packages a weights-only model
Prompt (operator): the harness must produce, from a model it brought up, a tt-model package for the profiles already
shipped (2 and 4 chips; 1 optional), and never publish; the run ends at the operator bundle with the publish commands
as text. Plan: `docs/superpowers/plans/2026-10-03-packaging-stage-7.md`. Tasks 1 to 8 (the modules) merged earlier;
tasks 9 to 11 (stage table, supervisor, documents) on branch `stage7-wiring`. Key decisions:
- Stage 7 is supervisor code (`orchard/package.py`, `StageSpec.harness`), with no agent and no model. It runs only on
  the weights-only path of a run started with `--package-format v6` and `--package-namespace`; other runs skip it as
  before. Options can be added on a resume before stage 7 starts (a "package options set" decision). A failure pauses
  the run and is not escalated.
- The package is built from the nearest model's installed v6 bundle with `tt-model package-thin --out` and `--weights`
  naming the new model. `run.sh` runs `prepare_model_dir.py` first, and `--model`, `HF_MODEL` and `MODEL_WEIGHTS_DIR`
  all name the built `model-dir`.
- The boot check installs a copy and serves it on a leased board with the one-test hardware phase stage 2 uses. Only
  stage 2's profile (2 chips here) is boot-checked; other installed v6 bundles give optional profiles whose publish
  lines stay commented out. v5.1 is refused at start: it needs an image build.
- `gate_package` checks again from disk: license, scrub, `run.sh` wiring, manifest weights, card, the boot check's
  numbers and server environment, and the publish commands.
- Changes from the plan text, because main had moved: `--package-models-root` defaults to the run's recorded
  `tt_model_root` (orchard/paths.py), so it follows `--operator-home`, and `TT_MODEL_MODELS_ROOT` was removed. The
  supervisor test passes `FakeContainers`, because stage 4's list of tests searches for leftover containers. The
  README flag table gained the three flags in the supervisor commit, because `tests/test_readme.py` requires a row
  for every run flag.

## 2026-10-03: stage 7 reads the served revision from the weight links
Prompt: run 3's stage 7 failed at once with `packaging: stage 2 served 'Altworld/Hemmingway-1'; stage 0 names
'Altworld/Hemmingway-1@1a5f363a...'`. The check compared stage 0's `<repo>@<revision>` with a label the stage agent
typed. Fix it so a label cannot fool it (branch `package-id-check`, TDD). Key decisions:
- `served_weights_problems` (orchard/package.py) takes stage 0's model and swap-check.json's `new_model_id` and
  `model_dir`. The label must name stage 0's repo, and stage 0's revision if it has an `@revision`. Every
  `*.safetensors` file in `model_dir` must link into stage 0's repo, all at one revision, equal to stage 0's.
- prepare_swap.py links each weight file to its blob (`os.path.realpath` of the snapshot file), so the resolved path
  names no revision. A blob link is matched to the snapshots whose same-named file is that blob. A link into
  `snapshots/<rev>/` names the revision directly. Both layouts are tested.
- Stage 0's ids are split into repo and revision before stage 7 uses them, and `gate_package` does the same. With
  the real `<repo>@<revision>` format, the nearest-model check and the manifest check failed one step later.
- The fake HF cache names blobs by content hash, as the real cache does. Before, two revisions of one repo shared a
  blob file and the mixed-revision test could not fail.
- Both swap skills tell the agent to copy stage 0's `model` value verbatim into `new_model_id`.
Mutations seen red: skipping the link check, comparing labels strictly (the live case), accepting mixed revisions.
Not run on hardware.

## 2026-10-04: templates for stages 0 and 1, near-duplicate repeats, wrap-up at turn exhaustion
Prompt: the second-model run (iapp/openthai2.0-qwen3.8-27b, `docs/run-logs/2026-10-openthai-run1.md`)
stopped twice at stage 0. Template stages 0 and 1 as stages 2, 4 and 7 were, catch near-duplicate
repetition, and give a step that runs out of turns one wrap-up (branch `template-stages-0-1`, worktree,
TDD, one commit per change, a mutation per guard). Key decisions:
- `orchard/skills/delta-triage-templates/delta_triage.py` reads `triage_config.json`, measures from
  safetensors headers (stdlib) and small JSON files, runs a tokenizer encode test (106 fixed strings, 100
  fixed-seed random ones) and drafts `delta.json`. `weights-only` needs equal decisive text config values,
  equal shapes and dtypes for every shared tensor, and no text tensor in only one model (vision, projector
  and `mtp.*` tensors may differ; this last rule is stricter than the prompt asked). A tokenizer difference
  does not change the path. An id without `@revision` gets the snapshot's revision.
- `orchard/skills/reference-gate-templates/reference_gate.py` loads in bf16 with low_cpu_mem_usage, decodes
  32 tokens with one full forward pass per token (no KV cache, so hybrid-attention models need no cache
  handling), checks the first token against a separate prompt-only forward pass, and writes evidence after
  each step. The card check is a form check only.
- Both skills now say: write the config first, copy and run the template, review the draft, never read the
  ledger, logs or transcripts.
- `TurnRepeat` has a shape track (`command_shape`, `TURN_SHAPE_N` = 4); `AgentStep` puts the shape on each
  shell call's event.
- Wrap-up (`AGENT_WRAPUP_TURNS` = 12): one continuation per attempt when a step ends `turns`, its
  deliverable is not written by this step and evidence exists. A wrapped step gets no gate feedback.
  Found by the kill test: a kill after a wrap-up wrote its file left that file behind, so "not written"
  means missing or unchanged since the step started.
Mutations seen red: no prefix normalization, always weights-only, no encode test, a weakened append check,
no missing-keys report, shape track off, no shape on the event, no wrap-up, a wrap-up with no evidence.
Not run on hardware: neither template has run on a real model.

## 2026-10-04: a template for stage 8
Prompt: on the second-model run both stage 8 attempts stopped. The agent read stage 7's evidence with 33
`cat` commands, wrote nothing for 20 turns, and `no_file_written` stopped it. Template stage 8 the way
stages 0 and 1 were (branch `template-stage-8`, worktree, TDD, one commit per change). Key decisions:
- `orchard/skills/operator-bundle-templates/build_bundle.py` reads `bundle_config.json` (`run_dir`) and
  rebuilds `stages/8/bundle/` from scratch from the ledger and each stage's files. It finds the checkout
  from the ledger's `paths.orchard_dir` and uses `orchard.stages` for stage names and `orchard.scrub`
  for the scrub, with the operator's home from the ledger, as the gate sees it.
- The tensor-cache hazard is `Dealt with: yes` only when every cache stage 2 and stage 4 recorded is a
  distinct directory, each is empty or holds a `.orchard-model` marker naming the model, and none is
  under a path the stage 0 finding names, tt-model's cache directory or a served bundle. The caches
  are checked when the script runs. Every other hazard says `Not shown` unless a result file settles it.
- Text copied from result files has the run's paths replaced by labels (`<CACHE_ROOT>` and so on) and
  any home path, hostname or token removed. Copied package files are not edited, so a planted token
  shows up in the scrub output and the gate fails the bundle.
- When stage 7 was skipped, `PUBLISH_COMMANDS.txt` holds only comments. When stage 7 ran and left no
  `PUBLISH_COMMANDS.txt`, the script exits 2 and names the file. This is a choice: the prompt said to
  refuse when stage 7 produced none, and a run without `--package-format v6` must still reach review.
- The package copy leaves out every directory and suffix the package scrub forbids, links, wheels the
  manifest does not list, and files over 1 GB. The gate's scrub reads shipped wheels as text.
- The skill: write the config first, copy and run the template, read RESULTS.md and RISKS.md once,
  rewrite the summary paragraph, add risks only with evidence, never read the ledger or transcripts.
Mutations seen red: skipped-stage reasons dropped, every hazard marked dealt with, venv copied, scrub
skipped, shared-cache check off, marker check off, nearest-cache check off, unlisted wheels copied,
redaction off, size limit off, unverified profile left out of the TODO list.
Not run on a real run: the template has only run on fake run directories.

## 2026-10-05: an operator surface for a small local model
Prompt: let a small local LLM agent (qwen-code on Qwen3.8-27B, small context, weak at long reading) act as the
operator of a run. Two deliverables: a read-only status command and an operator runbook skill. Built test first,
with a mutation per key guard (state from supervisor liveness, the disk rule, the writer lock, the pid wiring, the
pause detector, the runbook's command lines and NEVER list).
- `python3 -m orchard.supervisor status --run-dir DIR [--json]` (`orchard/status.py`) prints about 30 lines: state,
  how it was decided, stages with wall time, counts, pause reason and detector, last 5 events, disk, leases and a
  `next:` hint from a table (`HINT_RULES`). It never opens the ledger writer. `ledger.read_entries(path)` is the
  lock-free, repair-free reader that `Ledger.read` now calls. Exit 0 for a status, 2 for a bad directory or a corrupt
  ledger.
- The supervisor did not record its pid. It now writes `supervisor.pid` at the start of `run()`. Older runs have none,
  so status falls back to the holder of the `ledger.jsonl.lock` flock, found in `/proc/locks` (reading it takes no
  lock). A pid from the file counts only if the command line still mentions the supervisor.
- `orchard/operator_checks.py` runs the post-run checks the runbook names. On the real data it first flagged
  `hf upload` text that stage 8 wrote with `write_file` (PUBLISH_COMMANDS.txt). Only `shell` calls are commands, so it
  now reads those. The hub lookup then found `episod/openthai2.0-qwen3.8-27b-p300` public, because a person published
  it after the run (see docs/run-logs/2026-10-publishing.md); that is a true finding for the check.
- `orchard/skills/operator-runbook.md` is a 150-line loop (status, table, one action, one log line, `sleep 300`).
  Tests parse every command it quotes with the real argparse, and check each state and the NEVER list.
- Decision (the user, 2026-10-05): the stage skills stay in tt-orchard and are not moved to the tt-model-bringup plugin
  in tenstorrent/skills. The `status:` lines, the spec and the plans that said "their home is the plugin" were reworded.
  The existing plugin skills the stages name (`model-bringup`, `tt-device-usage` and the rest) are still read through
  `--skills-dir`.

## 2026-10-06: the orchard view of `status` (branch `orchard-ux`)
Prompt (operator): "make tt-orchards UX/DX more pleasant and delightful, playing on the orchard metaphor and what
it means with model bring up, and the roles each actor plays. maybe we get some color and some emojis even."
Short design shown in chat, approved, built test first with a mutation per guard (17, all seen red).
- `orchard/ui.py` detects what a stream can show. Colour and emoji appear only on a capable terminal (`auto`);
  a pipe, a dumb terminal, `ORCHARD_PLAIN=1` or `--style plain` get plain text, `NO_COLOR` removes colour only,
  `--style pretty` forces decoration. Palette is the docs-site theme from the global CLAUDE.md. Left bar and
  bottom bar only.
- `orchard/lexicon.py` is the one table of orchard names (states, stages, actors, closing lines). A test ties it
  to `status.STATES` and `stages.STAGES`. The real state and stage names always stay on the page next to the
  orchard names.
- `orchard/orchard_view.py` renders the page from the facts dict. `status.render` and `--json` are unchanged;
  piped output is byte-identical to before (tested). The operator runbook now pins `--style plain`, because a
  model reading through a pseudo-terminal would otherwise get the decorated page (tested).
- Found while testing: `textwrap` broke `ready-for-operator-review` at its hyphens. A test now forbids a line
  ending in a hyphen, and the wrapper has `break_on_hyphens=False`.
- Not done: the class and rootstock line (the outcome classes are in the bringup spec, not built), a live
  `watch` view, and the start banner (it belongs to the `tt-orchard bringup` command).
- Suite: 1884 passed, 1 skipped. Package version 0.0.2.

## 2026-10-06: the `tt-orchard` command, Phase 2 of the bringup spec (branch `bringup-command`)
Prompt (operator): "Our goal is an unsupervised model bring up of Cloudflare/clef. Make a plan for running this in a
way that will leave a pattern of success for future runs where users initiate with `tt-orchard bringup
Cloudflare/clef` and it goes from there, hell or highwater." Then: approach A (outcome-class contracts), write the
spec and start. Then: "build the orchard UX, then confirm the hidden-state change, and then proceed."
Spec `docs/superpowers/specs/2026-10-06-bringup-command-design.md`, plan
`docs/superpowers/plans/2026-10-06-tt-orchard-command.md`, spike log `docs/run-logs/2026-10-06-bringup-spike.md`.
Key decisions and findings:
- Spike (Phase 1): Coder-Next boots on one p300 board (`p300` profile, bfp4 routed experts, 681 s cold, 39 tok/s
  decode, about 3,560 tok/s prefill, one sample each). The default profile takes all four chips. The package was
  built from a dirty, unpushed tt-metal tree. The analysis docs' success rates and the arbiter's scores are
  unmeasured, and the arbiter is not used.
- Clef: the backbone has no tensor Qwen3.8-27B lacks except 15 `mtp.*` ones (allowed), so triage should say
  weights-only. The 122-tensor head is a sidecar. The head reads the final hidden state of every token and the
  lm_head matrix. The TT stack keeps only the last row, so the parity check is a template in this repo that
  runs the layer loop itself (spec section 6). No tt-metal change. Source read, not run.
- The bringup defaults live in `config/bringup.toml`, not in `tiers.toml`: the tier loader refuses unknown tables.
- `bringup` never uses the operator's Hugging Face token (a gated model is a block), never clears a gozer lease,
  and never accepts visible credentials by itself. A blocked preflight reaches neither the download nor the
  supervisor; a dry run starts nothing. Each of these has a test that was seen red under a mutation (about 100
  mutations over the new modules; the ones that survived led to new tests).
- Found while testing: the UX label column was one column too narrow for the longest check name (the width test
  caught it); `orchard.__version__` had drifted from `pyproject.toml` (now tested); `textwrap` is told not to
  break at hyphens.
- The real dry run for Clef passes every check except credentials: this home has an HF token, gh and docker
  credentials and three SSH keys visible to agent shells. Accepting that is the operator's decision (README 5.6).
- Not done (Phase 3 and 4): outcome classes, the `blocked` end state, the run budget, the sidecar parity template,
  the role-fit test for Coder-Next, the chaos run on Hemmingway-1, the Clef run itself.
- Suite: 2032 passed, 1 skipped. Package version 0.0.3.

## 2026-10-06 (later): the blocked end state, Phase 3 part 1 (branch `bringup-command`)
Prompt (operator): "yes just proceed -- for hours, you have the conch. i don't mind you using my creds and all
that or the child processes. no harm." Credentials may be used where a run needs them. Publishing, pushing and
other agents' leases stay off limits.
- `supervisor run --unattended` (always passed by `tt-orchard bringup`): when the run would pause, it records a
  `blocked` decision with a code, writes `BLOCKED.md` and `blocked.json` in the run directory, releases the
  hardware and exits 5. Running the command again records `resume by retry` and goes on from the ledger. A pause
  the operator asked for still waits. `status` has a `blocked` state; the runbook has a row for it.
- The run budget needed no new code: the escalation, cold-boot and wall-clock caps already pause a run, and an
  unattended pause is now a block with the code `retry-budget-spent`.
- Found while designing the tests: on the full-port path an operator's resume lets an attended run go on into
  the full-port stages. An unattended retry must never do that, so a full port blocks every time when unattended.
- `_block` already existed in the supervisor (for notices); the new method is `_end_blocked`.
- 27 mutations over the new code; two survivors led to tests (last_pause across a resume, and an attended run
  never auto-resuming). Suite: 2077 passed, 1 skipped.

## 2026-10-06 (later still): outcome classes and the sidecar gate, Phase 3 part 2 (branch `bringup-command`)
- `orchard/classes.py`: `weights-only`, `weights+sidecar`, `full-port`, `unknown`. Each maps to a path the stage
  machine already knows (`weights+sidecar` -> `weights-only`; `unknown` -> `full-port`), so nothing that ignores
  the class changes. Stage 0's passing `stage_end` records `class` beside `path`; `run_class` reads it back the
  way `run_path` does, and an older ledger with only a path gets the class that path implies.
- Found by reading the triage template for Clef: `read_tensors` read every `*.safetensors` file in the snapshot,
  so Clef's `joint_head.safetensors` (122 tensors) would have counted as extra text tensors and forced the
  full-port path. The backbone is now the shards the model's index lists (or `model*.safetensors` without an
  index). Any other weights file the nearest model lacks is a sidecar; an unreadable or overlapping one makes
  the class `unknown`. `.py` files are hashed and listed in `code_files`, never imported. 12 more mutations on
  the triage code; 6 survivors each led to a test.
- `gate_weights_swap_sidecar` = the swap gate plus `gate_sidecar_parity`, which also ties the parity run to the
  exact head file and code file hashes stage 0 recorded. The five thresholds in `defaults.py`
  (`SIDECAR_*`) are provisional and labelled so; the first hardware measurement replaces them.
- `orchard/skills/weights-sidecar-check.md` is the stage 2 skill for the class. The three templates it names
  (`sidecar-parity-templates/`) are being built and run on hardware by a forked agent in its own worktree.
- A `git add -A` swept the fork's worktree directory into a commit as an embedded repository. Amended out, and
  `.claude/worktrees/` is now ignored.
- Suite: 2159 passed, 1 skipped.

## 2026-10-06 (evening): sidecar templates merged, role-fit test, `read_file`, Coder-Next chosen
- The forked agent built and hardware-tested `orchard/skills/sidecar-parity-templates/` in its own worktree.
  Clef on one board: the sidecar head agrees with the CPU reference on 15 of 16 questions, hidden-state
  correlation min 0.954, largest probability difference 0.123 (docs/run-logs/2026-10-06-sidecar-parity-prototype.md).
  The `SIDECAR_*` bars in `defaults.py` are now measured-and-chosen, not provisional. The loader ignores
  `joint_head.safetensors`, as predicted. The merge was clean (new files only). `tests/test_sidecar_skill.py`
  checks the skill's config example against the scripts' required keys and the files it copies.
- `orchard/rolefit.py` replays recorded agent turns against a candidate server (format, repeated failures,
  speed). Coder-Next on one board (`p300` profile): first run 46 of 50 well-formed, all 4 misses a `read_file`
  call the loop did not offer; 35 replays of the 27B's 7 failure turns repeated none. The loop now offers a
  read-only `read_file` tool; second run 50 of 50, same zero repeats, 41.8 / 38.4 tok/s decode at 8K / 32K.
  Result files and the write-up are in docs/run-logs/2026-10-06-rolefit-qwen3-coder-next*.
- Machine config (not committed): `config/tiers.toml` and `config/bringup.toml` now name Coder-Next on one
  board (port 8001, `--coder-chips 2`), run directory and caches on the root drive (stage 4's disk check reads
  the run directory's filesystem and wants 110 GB; `/mnt/bonus` had 101 GB free). The 27B configs are in
  `config/local-27b.*.toml`. The operator's standing instruction was to proceed for hours and credentials may
  be used, so the first Clef run passes `--accept-credentials-visible`.
- Mistakes worth keeping: a mutation run that lets a remote endpoint through made a test try a real host and
  hang for minutes (the guard test should fail before any network call); and my first `git add -A` swept in a
  fork's worktree.
- Suite: 2346 passed, 1 skipped.

## 2026-10-06 (night): the first Clef run found three harness gaps
Stage 0 passed in 3m47s and found class `weights+sidecar`, one sidecar (122 tensors) and one code file, with
the same sha256 values the fork measured on hardware. Stage 1 then stalled with Coder-Next as the agent:
- It wrote files as `stages/1/reference.json` (the run-relative form it reads with `read_file`). `write_file` read
  that as stage-relative and wrote `stages/1/stages/1/reference.json`; the agent moved and removed files in a
  loop until the repeated-call detector escalated, and the escalated attempt did the same. Fix: a path that
  starts with `stages/` is run-relative and must land in this stage (`Tools.target`; other stages refused).
- It ran `pip install --upgrade transformers` and `--force-reinstall` into `~/.tenstorrent-venv`, the machine's
  shared venv: transformers 5.19.0, tokenizers 0.23.2, huggingface_hub 1.33.0, safetensors 0.8.0, requests and
  certifi changed at 21:45 PDT on 2026-10-06. `tt-model`, `hf` and `tt-smi` still answer. `pip check` reports
  many conflicts and the suite now warns that SciPy wants numpy below 2.5 (numpy is 2.5.3); I do not know
  which of these predate the run, because the earlier versions were not recorded. Fix: the runner refuses
  pip, uv pip, pipx and conda installs unless confined to the run directory, and stage 1 gets a dedicated
  interpreter (`reference_python` in bringup.toml, a venv built at `~/orchard-venvs/reference`, handed to agents
  as an input and checked by the preflight).
- The first stage 1 attempt had in fact produced a good reference before the write loop: the reference gate
  ran on Clef with the upgraded transformers.
- Each fix has tests seen red under mutations (runner rule 21 mutations, 5 survivors led to tests; write
  paths 6, one equivalent; reference input 12, one survivor led to a test). Suite: 2429 passed, 1 skipped.

## 2026-10-07 (early): stage 2 on Clef needed two more fixes; it then passed
Stage 2 (class `weights+sidecar`, swap check plus sidecar parity) failed three times, each for a new reason, and
each failure was found by reading the server log, not by guessing:
1. The two prepare scripts both built `stages/2/model-dir`. The parity script wiped the swap script's directory, so
   the swap server had no `preprocessor_config.json` and died at once. Each template passed its own tests;
   nothing ran both in one directory. Now `parity-model-dir`, and `tests/test_sidecar_swap_together.py` runs both
   scripts in one directory in both orders.
2. The nearest bundle (`episod/qwen3.8-27b-dflash2-p300`) serves with a DFlash2 speculative drafter that needs the
   model's `mtp.*` tensors (the spike log noted Clef dropped them; I did not connect that to the serving bundle).
   Engine start died with "model has no MTP head". `prepare_swap.py` now clears `QWEN36_DRAFTER` in its `run.sh`
   copy when the new model has no `mtp.*` tensor (read from the weight index, else the shard headers; when it
   cannot tell it changes nothing). `serve_and_compare_container.py` does the same for the container's
   environment. Stage 0 now records the counts in `delta.json` (`mtp`) and adds a drafter hazard.
3. With the drafter off, plain decoding asked for on-device sampling (`sample_on_device_mode` in the bundle's
   `--additional-config`), which the model code refuses on a 1x2 mesh. The key is removed too, so the host samples.
I tested fix 3 by hand: a scratch stage directory inside the run directory, a gozer lease on the free board, and
`serve_and_compare.py` with the supervisor's environment (`agent_env` plus the leased chips). It served in 1022 s
cold and agreed with the CPU reference on 31 of 32 tokens. That was much faster than another supervised attempt.
The supervised retry then passed stage 2: swap check 31 of 32 (ready in 236 s on warm caches) and the sidecar
parity gate, with the same numbers as the prototype. Stage 3, 5 and 6 are skipped on the weights-only path.
Also seen: Coder-Next's finish step investigates a failed hardware test with repeated greps instead of
recording the failure (the watchdog escalated it twice); a passing test avoids that. Disk was tight: failed
attempts leave 31 GB tensor caches in their partial stage directories, so stage 4 (110 GB) nearly blocked; I
deleted the caches of abandoned attempts.
Stage 7 will need the same drafter handling in `orchard/package.py` when a run packages a model without
`mtp.*` tensors; it is not done, because this run does not use `--package-format`.

## 2026-10-07: the first Clef bring-up reached `ready for operator review`
`tt-orchard bringup Cloudflare/clef` (Coder-Next on one board, run directory `~/orchard-runs/cloudflare--clef`)
ended `ready` after 11,092 s of ledger time. Stages 0, 1, 2 and 4 passed; 3, 5, 6 and 7 were skipped by design.
- Stage 2 (swap check plus sidecar parity): 31 of 32 tokens, ready in 236 s on warm caches; parity hidden_pcc_min
  0.954, top-1 agreement 15 of 16, largest probability difference 0.123 (the prototype's numbers, to the digit).
- Stage 4: 1 chip 31 of 32 (1,256 s cold), 2 chips 31 of 32 (218 s), 4 chips 30 of 32 (254 s). The 4-chip test
  parked the real coder (canary, stand-in, `tt-model stop`, reset) and restored it: the first park and restore
  of a real model in a real run.
- It was not hands-off. It ended blocked twice (stage 1 `stage-failed`, stage 2 `agent-stuck`) and stage 2 failed
  three times; every cause was a harness bug, fixed in code before the same command was run again. The retry
  path (`resume by retry`, a new coder boot, a fresh stage directory) worked each time.
- What the bundle (`stages/8/bundle`) lacked, found by reading it: it did not mention the sidecar parity, the
  class, or that every swap check ran with the speculative drafter off, and it said "the bundle has no check for
  this hazard" about the sidecar. Fixed in `build_bundle.py` (class, sidecar and MTP rows, a parity section with
  the bars, per-check serving state, and hazard dispositions from the evidence). The supervisor's original bundle
  was left as built; a rebuilt copy is at `~/orchard-runs/cloudflare--clef-bundle-rebuilt`.
- I tested the stage 2 swap fix by hand in a scratch stage directory with my own lease before another supervised
  attempt; use that when a stage fails inside a long hardware test.
- A trap I fell into again: `pgrep -f` in a wait loop matches the loop's own shell (the global CLAUDE.md says
  so). Wait on the supervisor's recorded pid instead.
- Not done: stage 7 packaging for a model without `mtp.*` tensors (`orchard/package.py` needs the same drafter and
  sampling handling), a chaos run, the sidecar parity at 4 chips, image and video records, and a second sidecar
  model. The shared TT venv was changed by an agent's pip install before the runner refused it (transformers
  5.19.0, huggingface_hub 1.33.0, safetensors 0.8.0, tokenizers 0.23.2, requests, certifi at 21:45 PDT on
  2026-10-06; SciPy now warns about numpy 2.5.3). The earlier versions were not recorded.
- Suite: 2515 passed, 1 skipped.


## 2026-10-07: setup script, QuietBox 2 defaults, release to main (version 0.1.0)
Prompt (operator): "commit and push and merge everything in tt-orchard to main so anyone can make use of this. make sure we
have a setup script with pre-reqs handled with tt-gozer. position the model change as a strong suggestion or default for QB2.
publish the model card publicly on HF / tt-model-manager".
- `scripts/setup.sh` runs `orchard/setup_machine.py` (`tt-orchard setup` flags: `--coder`, `--yes`, `--check`,
  `--start-ollama`, `--runs-root`, `--venv-dir`, `--gozer-dir`). It checks Python 3.12+, tt-gozer 0.3.2+ (clones and installs
  it when missing), tt-model, docker, hf, ollama, hugepages, the reference venv and the coder package, and writes
  `config/tiers.toml` and `config/bringup.toml` from `config/*.qb2-*.toml`. Existing files are never overwritten. Other
  hardware gets a warning and no files. It never runs `reset`, `publish`, `push`, `reconcile`, `acquire` or `release`
  (`COMMANDS_NEVER_RUN`, tested with a fake machine whose every call is logged).
- Coder-Next on one board is the default for a QuietBox 2; `--coder 27b` gives the earlier arrangement. The README says what
  the evidence is and what it is not (single measurements, dirty unpushed tt-metal build, bfp4 experts).
- Process lapse: I wrote `setup_machine.py` before its tests. I moved it aside, wrote the tests, watched the import fail,
  restored it and ran 53 mutations. Three survived (a QuietBox 2 without "p300", the CPU torch index, the default coder); each
  now has a test and is red under its mutation.
- Not done: the Hugging Face / tt-model publish. No Clef package exists (stage 7 needs the drafter and sampling handling for
  a model without `mtp.*` tensors), and the bundle's `card.md` says no package was staged. Waiting for the operator to choose
  between building a package, publishing only a card, or skipping.

## 2026-10-07: stage 7 for a model without `mtp.*` tensors, and the Clef package (branch `clef-package`)
Prompt (operator): option 1, build a Clef package and publish it with tt-model. The operator also said the card
requirements in `~/code/tt-model-manager` had been updated.
- `orchard/package.py` reads from stage 2's `run.sh` copy whether the drafter was off and sampling was on the host
  (`stage_2_serving`). With the drafter off, `serving_env` passes `QWEN36_DRAFTER=` and drops `DFLASH_WEIGHTS` and
  `QWEN36_DFLASH_*`; `drop_device_sampling` removes `sample_on_device_mode` from the fixed arguments;
  `serving_problems` checks the staged files and `gate_package` runs it again. The card says speculative decoding is off,
  sampling is on the host, and the sidecar head is not served (gate checks it names every sidecar).
- Card layout follows tt-model-manager's standard (`model-card-standardizer` skill): Intended use, Expected performance,
  Limitations, Risks and safety considerations. The catalog's required-section check reads container packages only;
  a v6 thin bundle is not checked, but the card carries the sections anyway.
- 28 mutations over the new code; 4 survivors each led to a test.
- Stage 7 was run by hand on the finished Clef run (its ledger records stage 7 as skipped): `stage_all`, then the boot
  check under `gozer run --chips 2` with the supervisor's environment, then `finish` and `gate_package`. Not in the ledger.
  `clef-p300` (2 chips): boot check passed, 31 of 32 tokens (top1 0.96875), ready in 1,484 s on a fresh install and empty
  cache, 1,515 s for the whole check. `clef-p150` (1 chip) was staged and not boot-checked, so it is not to be published.
- Publish: `hf upload --private episod/clef-p300 ...` was blocked by the permission classifier (creates a Hub repo).
  Nothing was uploaded.

## 2026-10-07 (later): evidence ships with the package; `episod/clef-p300` is public
Prompt (operator): merge to main and push; fix the evidence-path gap; "you have my explicit permission to create the HF
repo, as a public repo".
- `copy_evidence` copies every file a measured card number cites into `evidence/<run-relative path>` with the run
  directory, home directory (any user's) and host name replaced by `<RUN_DIR>`, `<HOME>`, `<HOST>`, and an installed
  bundle path's org replaced (`<TT_MODEL_MODELS>/`), because that org is the operator's namespace and the scrub refuses
  it. Over 5 MB is refused. `gate_package` fails a card that cites a file missing from `evidence/`. A token in evidence
  fails the scrub in the gate. 13 mutations, 3 survivors led to tests, 1 was an equivalent mutant.
- Uploaded with `hf upload --private`, then `tt-model publish episod/clef-p300` made it public and listed it in the
  community catalog. Only the 2-chip profile. Pull and boot from the Hub is not tested. Delist: `tt-model unpublish`.

## 2026-10-07 (later): live narration, `tt-orchard watch`, and "what was tried" on a block (version 0.2.0)
Prompt (operator): "we need to log more in this tool to the screen. when it's starting things, what each role is doing,
etc. just had a failure / blocker but am just retrying it as it wasn't clear how to unblock. maybe if I'd seen what was
tried in real time?" Before this, a run printed almost nothing and BLOCKED.md said only "stage 1 failed after
escalation: ... turns no final answer after 60 turns".
- `orchard/narrate.py`: `Narrator` follows the ledger (lock-free `read_entries`) and the agent logs
  (`stages/*/log/*.jsonl`) and prints one line per event with the role: orchardist (supervisor), grafter (coder),
  head grower (large tier), seasonal hand (CPU tier), sheepdog (watchdog). Agent turns show the command run, the file
  read or written, and the result with the next turn. After 60 quiet seconds it says what it waits for; during a
  hardware test it shows the last output line. A failed hardware test is followed by the end of its output. A
  background thread polls once a second and gives up on its first error, saying so, so a narrator bug cannot stop a run.
- `tt-orchard bringup` prints it (`--quiet` turns it off); a resumed run first shows its last six entries.
  `tt-orchard watch` follows a run from another terminal (`--all`, `--once`) and exits when the run ends.
- A block now prints, and `BLOCKED.md` holds, "What was tried" (watchdog findings, the test's last output, the agent's
  last actions) and "How to unblock" per block code (`narrate.UNBLOCK`; a test ties its keys to `blocked.REASONS`).
- A test caught `stop()` joining a thread that never started. 31 mutations; 4 survivors led to tests and one
  (`replay=6` on a fresh run) is equivalent, because the ledger does not exist when the first poll runs.
- Not done: the stage 4 per-configuration test output is found by newest file under the stage, not by configuration;
  turn lines are not shown for the hardware tests' own subprocesses.
