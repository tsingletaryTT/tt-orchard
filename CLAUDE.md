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
