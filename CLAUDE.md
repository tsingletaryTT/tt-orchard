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

## 2026-10-17: `tt-orchard coder-speed`, and a correction (branch `ka-speedwatch`, version 0.6.2)
Prompt (operator): "can we add something that detects when coder gets too slow and needs a restart (or any model we're
using because it will change over time)". Context: during the kernel-probe batches I reported the coder had slowed from 38
to 7.6 tok/s, twice, and the operator approved a restart and reset for it. Both reports read vLLM's "Avg generation
throughput" log line, which is tokens over the whole 10 s window. An agent's turns are short tool calls with idle time
between them, so a healthy server reads 5 to 8 tok/s there. Measured idle, the same server decoded at 38.7 tok/s. The
restart was probably not needed. The ledger and the operator were told.
- `orchard/speedwatch.py` asks a server for one fixed greedy answer (about 300 tokens, thinking off) and divides the tokens
  the server reports by the seconds the client measured. It keeps the best speed seen per model and endpoint
  (`~/.cache/tt-orchard/speed-baseline.json`; a slow sample never lowers it) and calls a server `slow` when the last three
  samples are all below half of that. It reports and never restarts, because a restart interrupts every client.
- Found by running it live: a fresh request beside the batch's long conversation measured 7.3 tok/s, while two short requests
  together got 29.6 each. A decode step serves every running request, so one long context slows all of them. The check now
  reads `vllm:num_requests_running` before and after its own request; a sample taken beside another request is `busy`, is not
  counted and teaches nothing. A server with no such count is measured anyway and the message says it could not tell.
- `tt-orchard coder-speed [--endpoint URL --model NAME | --tiers] [--samples N] [--watch SECONDS] [--json]`; exit 0, 2
  (could not measure) or 3 (slow). `kernel-probe` measures before each attempt, records `coder_speed` in each attempt.json,
  lists it in the report and stops with exit 6 when the coder is slow (`--allow-slow-coder` overrides); `busy` and `error`
  never stop it, and the report marks those attempts as not comparable.
- Tests first. 85 in the two new files and 13 in the probe; about 55 mutations over the module and the wiring, each red,
  except one equivalent (an extra re-check that can never run). Tests that call the module never reach a server: an autouse
  fixture replaces the metrics call, because 127.0.0.1:8011 is a live coder on the machine they were written on.
- Not done: the supervisor's watchdog does not use it (restarting the coder inside a bring-up needs the park and restore
  machinery); the baseline needs `--reset-baseline` after a change on purpose; one pass over `--tiers` takes a sample per tier.

## 2026-10-17 (later): the check says how far off a wrong kernel is (branch `ka-numdiag`, version 0.6.3)
L4e's `rmsnorm` attempts 3 and 4 (temperature 0.7, different seeds) wrote the same wrong kernel: the mean of the squares was
divided by 8 tiles, not 256 elements. Their results match to three or four digits (max abs error 9094.3965 in both, slope
964). Each was told only `pcc 0.9796, bar 0.999`; the slope of 964, which says the output is about 964 times too big, was in the
result file and never reached them. The first-line summary stops at the first bar that fails, and PCC is checked first.
- `kernel_harness.py` records the three worst positions of each input set (`worst`: index, expected, got; a value that is not
  finite ranks first and is null). `kernel_check.py` `numeric_detail` turns the first failing set into one line, printed as
  `numbers: ...` under the `FAIL` line the way `where:` and `hint:` are: how far off the scale is (about Nx expected, the opposite
  sign, or about right), PCC, relative L2, the largest error beside the size of the expected values, the two worst places, and
  how many input sets fail. It states measured facts, names no op and gives no fix. The first line and `check.json` are unchanged.
- 14 mutations, 13 red; the survivor (`numbers:` for a non-numeric category) is equivalent in practice, because such a result
  has no failing numeric set. Two assertions I first wrote could not fail (one ended in `or True`); I rewrote them before running.
- Not run on the live coder yet: the probe batch (L4e) was still running on the other checkout. Whether the line helps the coder is
  not known.

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

## 2026-10-07 (later still): the base model preflight check (version 0.3.0)
Prompt (operator): after the `jialinyyzz/humanizer` run stopped, "yes build that preflight check. we should do a `tt search
gemma` in a case like this and ask if we want to select a matching model to base on."
Cause of that run (read from its ledger, logs and the cache): the model is a fine-tune of `google/gemma-4-12B`; that base
was not on disk and no installed bundle served it (only Qwen3.8-27B bundles were installed; the cache held manifest-only
Gemma repos). Three coder boots and 105 agent turns went to `ls` on cache directories.
- New `base` preflight check (`preflight.check_base`, `orchard/nearest.py`). The base is the card's `base_model` or
  `--base`. An installed bundle must serve it (`tt_kernel_manifest.json` `weights.repo_id`). A missing snapshot is
  downloaded and its size counts in the disk check. A card with no base is a warning. A base served only by a v5.1
  container package is a warning, because the swap template expects a v6 thin bundle (read from the template, not run).
- No bundle: block `nearest-model-missing`, after `tt model search <family>` (the official CLI first, `tt-model search`
  as the fallback). Candidates: installed bundles of the family first, then the search results.
  At a terminal `bringup` asks which to use and installs the pick with `tt-model pull`; in a script it prints the
  `--base <bundle>` commands. `--base` takes a bundle or a model id and needs no prompt. A dry run never prompts or pulls.
- The base snapshot path goes to the supervisor as `--input base=<path>` (the delta-triage skill already reads it).
- 30 mutations; 2 survivors led to tests (dry run with `--base`, a `--base` that still blocks). A first mutation run was
  meaningless because two tests were already failing; check the baseline is green before reading a mutation table.
- Real dry run on this machine for the humanizer: the check blocks and lists the two Gemma bundles in about 10 seconds.
- Not done: whether stage 2 can use a v5.1 container Gemma bundle; this run was not resumed.

## 2026-10-07 (later still): visible credentials are accepted by default (version 0.3.1)
Prompt (operator): "make --accept-credentials-visible the default please". `tt-orchard bringup` now passes
`--accept-credentials-visible` to the supervisor unless `--refuse-credentials-visible` is given. The preflight row is a
warning that names the files, and the ledger still records the acceptance. `--accept-credentials-visible` stays as a
no-op so older commands work. `supervisor run` itself is unchanged and still refuses until the flag is passed. The
exposure is the same as before (agent shells run as the operator and can read the files); only the default moved.
Four mutations, all red.

## 2026-10-07: `tt-orchard ui`, a browser page to watch and control runs (version 0.4.0)
Prompt (operator, ezietlow): "build a full UI that will allow for monitoring and control of the orchard project. It
needs to follow best practices for UI design and be an easy to use tool." Plan approved in chat: in this repository,
loopback only (reach it with `ssh -L`), the full lifecycle (watch; pause, resume, abort; retry a blocked run; start a
new bring-up through the preflight), stdlib server and plain HTML/CSS/JS with no build step.
- `orchard/webui.py`: a `ThreadingHTTPServer` on 127.0.0.1:8780 (`tiers.LOCAL_HOSTS` only). Reads with
  `status.collect` (gozer shared for 15 s), `ledger.read_entries`, and a `narrate.Narrator` subclass (`Feed`) whose
  lines go to the page as server-sent events. Writes only `supervisor.Control` words, and only when the run can act on
  them (pause or abort while running, resume while paused), and launches only a detached `tt-orchard bringup`
  (`start_new_session`), for a new model when the preflight has no block or for a blocked or stopped run (retry).
- Requests: the Host header must be a loopback name (DNS rebinding), every POST needs the per-process token from
  `/api/meta` and a loopback Origin when one is sent. Files come from an allow-list (gate files, evidence, the bundle's
  text, BLOCKED.md, coder.log; never the ledger, transcripts or the control file), stay inside the run (a symlink out
  is refused), are tailed past 256 KB, and are redacted with the scrub's token and home-path patterns plus the host name.
- `orchard/web/`: one page. Machine panel (boards and chips from `gozer status`, coder and CPU-tier ports, disk), runs
  sorted by what needs a person, the selected run's state with orchard and real names, the `next:` hint, block and pause
  panels, a nine-stage stepper, and tabs for the live feed, the ledger, the files and the details. Confirmations are
  `<dialog>`s that say what happens; abort needs a second click. Light and dark themes from `ui.ROLES`, keyboard tabs,
  `aria-live` feed, no external hosts (a CSP forbids them), every server string set with `textContent`.
- 57 tests in `tests/test_webui.py`, written first. 14 mutations, one per guard (loopback host, token, Origin, Host
  header, allow-list, symlink, redaction, control-state check, preflight block, one launch at a time, model id
  validation, gozer cache, new session, retry state): all red.
- Found in the browser, on node6 against real runs: an aborted run's open stage showed "running" with a clock still
  counting (`stage_rows` reports an open stage as running whatever the run's state), two runs of one model could not be
  told apart, polling rebuilt focused controls every few seconds (now only on change), and CLAIMED chips were coloured
  as a fault.
- The feed was not redacted at first: an agent's `ls ~/.cache/...` reached the page with the home path. Feed lines now
  pass through the same redaction as files (test first; the mutation that skips it is red).
- On node6's chips (QuietBox 2, gozer 0.3.3, Coder-Next on one board): the page ran the preflight for
  Altworld/Hemmingway-1, started the bring-up detached, and followed it. Coder-Next booted cold in 10 min 22 s
  (18:55:27 to 19:05:49Z) and passed its canary; stage 0 started. Pause from the page: paused within seconds, chips
  still leased. Resume: running again about 5 s later. Abort (two clicks): `abort: operator` at 19:08:04Z, `hardware
  released` at 19:08:47Z, then all four chips FREE, no container or supervisor left, port 8001 closed. While the
  Coder-Next container served, gozer showed its board HELD-FOREIGN; the page now says why.
- A hidden tab first stopped polling, so its title (the run's state) went stale; it now polls every 30 s.
- Not done: screenshots in the README, a test of the page's own JavaScript (it is checked by hand in a browser and
  by `node --check`), and `stage_rows` itself still calls an aborted run's open stage "running" (the page shows it
  as stopped).

## 2026-10-08: `tt-orchard ui --lan`, tt-toplike in the page, and the farm look
Prompts (operator): "open everything so I can access it from any box on lan"; "Can we use the tt-toplike project
for some of the card visualization? We also don't need login for now"; "please use everything from tt-toplike
don't re-write it"; "gamify the UI with some kind of stardew valley type animation ... theme the whole UI with
more stardew valley type color scheme and elements".
- `--lan`: listens beyond loopback with no login (the operator's call). The Host check is skipped there; every
  POST still needs the token and an Origin equal to the request's Host. The firewall is the operator's to open.
- A first attempt ported tt-toplike's portrait, colours and hwmon reader into the page. The operator asked for
  tt-toplike itself instead, so the port was removed. tt-toplike has no web build, so the page runs the real
  `tt-toplike-tui` in a pseudo-terminal per viewer and draws it with xterm.js (vendored). Forwarding keystrokes
  from an unauthenticated LAN page into it was refused by the session's safety check as a remote-execution
  surface; the operator chose view-only: no input route, `disableStdin`, and a `--mode` from TOPLIKE_MODES
  (`hivemind` left out). At most TOPLIKE_MAX per server; a closed page stops its tt-toplike (the server watches
  the socket, because a still screen sends nothing to fail on).
- tt-toplike printed in one colour at first: it gives up true colour when TMUX is set (src/ui/colors.rs), and
  the UI ran in tmux. It now gets the environment tt-toplike's own app gives it (src/bin/app.rs). Then the page's
  CSP blocked xterm.js's inline styles (thousands of console errors); styles may be inline now, scripts stay 'self'.
- The farm look: parchment menus in wooden frames on a grass field by day, indigo and lantern light by night,
  Pixelify Sans (vendored, OFL) for headings and buttons. `orchard/web/scene.js` draws the run as a pixel orchard
  in code: nine plots that grow into fruit trees as stages pass, the orchardist walking to the running stage,
  the grafter working while the agent talks, the sheepdog dashing by on a watchdog line, and weather for the
  state (sun, rain, frost, a soft storm with no strobe, falling leaves, harvest basket). Reduced motion draws it
  still. The scene is decorative; everything it shows is text on the page too.
- Checked on node6: tt-toplike 0.13.10 (built in ~/build/tt-toplike-orchard) shows all four p300c chips in
  colour in Table and Arcade views, inside the page.
- Later the same day (operator: "make weather conditions based on chip health. make the worker work based on
  utilization and rest based on the same. Make him go back and get tools from the shed when chips are switching
  leases"). tt-toplike's `--serve` stream was the first choice for the data, but it only serves with its json or
  hybrid backend, which polls `tt-smi` and opens the chips; gozer refuses a reset while a device is open, and the
  page's tt-toplike on its default backend had already shown up as BUSY-UNTRACKED in `gozer status`. So the page's
  tt-toplike now always runs `--backend sysfs`, and `read_chips` reads the same kernel files that backend reads.
  `weather()` and `utilization()` are pure and tested; a heartbeat that has not moved for 5 s is a storm. The farmer
  works, rests and fetches tools from the shed from `/api/health`; a lease change is seen within GOZER_TTL_S (5 s).
- Found: killing the UI's tmux session left tt-toplike children running (their own session, so no hangup). Now
  SIGTERM and SIGHUP close every session, and each child asks the kernel for SIGTERM when the server dies
  (PR_SET_PDEATHSIG); checked on node6 with `tmux kill-session` and with `kill -9` of the server: none left.
- Checked on node6's chips: four Blackholes at 34-36 °C, 12-13 W, aiclk 800 MHz, heartbeats moving: clear skies,
  farmer resting. A real `gozer acquire` and `release --no-reset` on board 0 sent him to the shed and back.
- Layout (operator: "cleaner and less like a series of panels"): one parchment sheet with a quiet rail; only the
  orchard scene and the tt-toplike TV keep wooden frames. The TV uses xterm.js's WebGL renderer, retints only
  tt-toplike's near-black backgrounds to the skin, and has zoom, wide and full-screen controls.
- Tested with a real bring-up started from the page (Altworld/Hemmingway-1, Coder-Next on board 0, 2026-10-08).
  The first readings showed the AI clock is not a work signal: with Coder-Next resident, tt_aiclk sits at 1350 MHz
  and the board draws 32-35 W with no request in flight, which the first formula called 60 % busy. Utilization is
  now power above each chip's lowest reading in the last ten minutes (FLOOR_WINDOW_S). Measured during stage 0:
  resting 34-36 W reads 0.00; token generation reads 44-75 W with peaks of 118-170 W (above power1_max, 125 W),
  0.2 to 1.0; the idle board stays at 0. The farmer also waited at plot 0 while the coder booted, because the facts
  name the next stage as current before it starts; the scene now works a plot only once that stage has started.

## 2026-10-12: base model research (branch `research-base-discovery`, version 0.5.0)
Prompt (operator): "How do we make this project better at finding and researching the model family it might need to base a new bring up on? Searching tt model manager, inspecting existing models, using HF or GH.. not just looking locally." The operator chose scope C (everything): hub lineage, architecture matching, a class estimate from safetensors headers before any download, a search for existing ports, and a written ranked report. Triggered by the `jialinyyzz/humanizer` run, which stopped because no installed bundle served its base. Plan: `docs/superpowers/plans/2026-10-12-base-research.md`. Tasks 0 to 10, test first, a mutation per guard.
Key decisions:
- The collectors are deterministic code. No model is involved. They live in `orchard/research/` (http, lineage,
  arch_match, headers, ports, report, triage_rules, types).
- The class rules are loaded from the delta-triage template (`triage_rules.py`), so the estimate and stage 0 use the same
  rules.
- The HTTP fetcher has no token and no Authorization header. It allows three hosts. Hugging Face redirects file downloads
  to a CDN, so redirects are followed only to an allow-listed host.
- `gh` is used for code search only, read only.
- `--dry-run` does the reads and prints the result but keeps no cache and writes no file.
- An inferred base (same architecture family and decisive config values) is a warning. Only a stated or inferred candidate
  with an installed v6 bundle is chosen on its own. Stage 0 records whether its measured class agreed with the estimate;
  the measurement decides.
- Flags: `--offline`, `--no-github`, `--refresh-research`. Files: `<run dir>/research/base-candidates.json` and `.md`.
Findings from planning: Hugging Face redirects file requests to a CDN; published bundles carry a manifest and tags that name
the model they serve; GitHub code search needs a login (`gh auth login`).
Measured on 2026-10-12 on this machine:
- Live test (`ORCHARD_LIVE_RESEARCH=1`, `no_github=True`, no bundles installed): passed in 11 s. Stated base
  `google/gemma-4-12B`. Ranked bundles: `stisiTT/gemma-4-12b-it-p150` first (inferred, group 2, weights-only), then
  `tt-hous/gemma-4-26B-A4B-it_p150_p150x2_p150x4` (weak, no class estimate). Nothing unavailable. The planning run with `gh`
  logged in also listed `google/gemma-4-12B` itself as stated with no bundle; this run excluded GitHub.
- Dry run `tt-orchard bringup jialinyyzz/humanizer --dry-run`: the base check blocks with `nearest-model-missing`, lists the
  same two ranked bundles, names tt-metal's `models/demos/gemma4` and `gemma4_31b_qb2` as existing work, and prints the
  `--base` commands. Nothing was started or written. No source was unavailable.
Not done:
- (Fixed the same day, version 0.5.1: see the entry below.) The UI preflight panel (`_ui_preflight`) did not run research.
- The sidecar class cannot be estimated from headers.
- A resumed run rebuilds the report and overwrites `base-candidates.json` and `.md` (`generated_at` changes).
- The preflight fails open. An exception from research leaves the earlier base check in place, and no row says that research failed.
- `report['model']` and the port text depend on upstream `clean()`.
- `_config` in `report.py` duplicates `arch_match.fetch_config`, because that function drops the failure reason.
- GitHub's unauthenticated limit is 60 requests per hour and a run makes about 6. This is an open point.
- Nothing was run on hardware.

## 2026-10-13: base research in the web UI (version 0.5.1)
Prompt (operator): "is this new featureset wired into the web UI too?" The answer was no (only the launched `bringup` had it).
Then: "add it as part of this PR too". Built in a worktree (`ui-base-research`), because an unattended bring-up of
`ThakiCloud/RAG-Gate-27B` was running from the main checkout. Tests first, a mutation per guard.
- The preflight panel runs the research as a dry run does (read-only, no cache, no file) and shows `research`: the ranked
  bundles, candidates, existing work and what was unavailable, bounded (8, 8, 5, 10 rows), redacted, https links only. The
  new-run dialog has a research setting (full, no GitHub, off) that becomes `--no-github` or `--offline` on the launch.
  The page asks again after every form edit, so one answer is kept for 5 minutes per model and setting (GitHub allows 60
  unauthenticated requests an hour; one research run makes about 6).
- `GET /api/runs/<name>/research` gives a run's `base-candidates.json` (2 MiB cap, no symlink) as the same view plus stage 0's
  `base_research` record (known fields only). `base-candidates.json` and `.md` are on the file allow-list; the stage 0 view,
  the cache and the temp files are not.
- 17 mutations; the one survivor (the size cap) was a test whose oversized file failed JSON parsing anyway; it now pads a valid
  report. Smoke test: the real server, a real preflight for `jialinyyzz/humanizer`, same two bundles as the CLI.
- Not tested: the page's JavaScript runs only under `node --check` and by hand in a browser; no browser run was done here.

## 2026-10-13: first unattended bring-up with the research setup (ThakiCloud/RAG-Gate-27B)
Prompt (operator): "put the new setup to a test and attempt an unsupervised bring up of a small, novel and useful and relatively
'official' model". Chosen with the new tooling: dry-run preflights of three Qwen3.8-27B fine-tunes found by a hub search
(`ThakiCloud/RAG-Gate-27B`, Apache-2.0, text only, a retrieval gate for RAG; ThinkingCap was gated; Omni-Edu's licence is "other").
None of the fine-tunes is small; "small" was read as a small change from the base. Run: `tt-orchard bringup ThakiCloud/RAG-Gate-27B`
(Coder-Next on one board). Run directory `~/orchard-runs/thakicloud--rag-gate-27b`.
- Research: stated base `Qwen/Qwen3.8-27B`, bundle `episod/qwen3.8-27b-dflash2-p150` chosen automatically, estimated class
  `weights-only`; stage 0 measured `weights-only` (`base_research.agreed` true). Stage 0 2m10s, stage 1 4m01s, stage 2 27m29s
  (31 of 32 tokens, top-1 agreement 0.96875, ready in 1,482 s cold). Stage 3, 5, 6 skipped by design.
- Stage 4: 1 chip passed (rc 0), 2 chips passed (rc 0), 4 chips failed twice (rc 4) with `Fabric Router Sync: Timeout after
  10000 ms ... Ethernet handshake likely failed -- the link may not be healthy`, on a different device each time (3, then 2).
  Same docker arguments and mesh as the Clef 4-chip test that passed on this box on 2026-10-07. The run ended `blocked`
  (`stage-failed`). Cause not found: the box's link state is the suspect; nothing was reset by hand.
- Three blocks on the way, none about the model: (1) ollama was not running, so the park's CPU stand-in failed its canary
  (`coder-unusable`); I started `OLLAMA_MODELS=/mnt/bonus/models/ollama ollama serve` and retried. (2) Stage 4 needed 110 GB
  and 109.6 GB was free; the operator approved deleting the Clef tensor cache (84 GB), then retried. (3) the 4-chip failure above.
- Gaps found, not fixed in this PR: preflight does not check that the CPU stand-in answers although stage 4 needs it for the
  park; `setup --start-ollama` runs a plain `ollama serve` that does not see `/mnt/bonus/models/ollama`; stage 4's disk
  check can only be met by deleting earlier runs' caches, and `setup` leaves no hint about that.
- Not a hands-off run: three resumes by me. Nothing was published, pushed or uploaded.
- Update (same day): the operator asked whether the chips had been reset before the 4-chip run. They had not all been. The ledger
  shows the park resets only the coder's board (chips 0 and 1, about 41 s); chips 2 and 3 ran the 1-chip and 2-chip tests and
  were not reset, and the fabric timeouts were on devices 3 and 2. Their ARC heartbeat read 1275 against 357 for the reset
  chips. I took all four under `gozer run --chips 4` (a `tt-smi -s` snapshot, then the release reset all four) and resumed.
  The run then passed stage 4 and stage 8 and ended `ready-for-operator-review`: 2 chips 118.0 s cold-cache-warm, top-1
  agreement 0.96875; 4 chips 269.9 s, 0.9375 (32 tokens each; draft off; the 1-chip configuration was optional and not in
  the final list). Counts: 1 retry, 1 escalation, 1 nudge, 3 pauses. Not run: stage 7 (no `--package-format`).
  The stale chip state is a hypothesis that fits the evidence and the pass after the reset; it was not isolated (no run with
  only the reset changed and a control without it). Harness gap, not fixed here: before a test that opens more chips than the
  coder's board, reset the chips earlier tests used.

## 2026-10-13: fixes for the review of PR 6 (version 0.5.2)
Prompt (operator): "fix requests on PR 6". The automated review left four findings, each checked against the code and each real:
- The UI's five-minute research memo was rebuilt for every request, because the server builds the preflight callable again each
  time (and per mode), so form edits repeated the hub and GitHub calls. The memo now lives in `_ui_preflight_for()`.
- An index the repo advertises but that is unreadable, lists none of its safetensors files or has no `weight_map` object fell back
  to file names silently, so the class estimate could be a confident guess. It now adds an error and the estimate is `unknown`.
- A sibling fine-tune was always `inferred`, whatever its config. Now the architecture comparison decides (`weak` with the
  differing keys, left out when the family differs); with no readable config it stays `inferred` on the relationship alone. The
  earlier test used a sibling with a different layer count and asserted `inferred`, which is the behaviour the review objected to.
- The page dropped stage 0's `candidate` because the template writes an object `{repo, confidence}` and the filter accepted only
  strings; my test used a string. The view now keeps a bounded, redacted object and the page shows both fields.
- Each fix has a test that failed first; 10 mutations, all red. Lesson: the UI test used the shape I assumed, not the shape the
  producer writes. Read the producing code when writing a test for a consumer.

## 2026-10-14: research hardening after PR 6 (branch `ezietlow/research-hardening`, version 0.5.3)
Prompt (operator, through a review of merged PR 6): fix forward the review's follow-ups, one commit each, each with a test
that failed first. Nothing was pushed or run on hardware.
- A safetensors header is read only up to 8 MiB (`headers.MAX_HEADER_READ` = the fetcher's body cap). The template's 100 MiB
  limit had overridden that cap for header reads; a larger claimed header is an error, so the estimate is `unknown`.
- An inferred base (the card names none; research picked one by architecture) blocks with `inferred-base-unconfirmed`
  until the operator confirms it: the chooser at a terminal, the candidate pick in the UI, `--base`, or
  `--accept-inferred-base`. Before, it was a warning and the run downloaded the base and started unattended. The model is
  never its own candidate.
- The fetcher has a budget per research run: 600 network requests, 240 s from the first one (`MAX_REQUESTS`, `DEADLINE_S`).
  After that the remaining requests fail at once and become unavailable rows; they are not cached and do not mark the host.
- An index that lists a shard the repo does not have makes the estimate `unknown` (it read part of the backbone before).
- Only `--base` and a direct base whose config does not disagree are `stated`. A direct base whose decisive values differ is
  `weak`; a base of the base is judged by its own config (`inferred` or `weak`), so it needs the confirmation above.
- A URL the fetcher opens, and every redirect, must have the bare allowed host as its authority (no user info, port or
  escape): urlsplit and urllib read `evil.com\@huggingface.co` differently.
- The report has `new_model_gated` and says in the Markdown and the UI that a gated model's short candidate list is not
  evidence of no base.
- The UI's research memo has a lock and keeps at most 32 answers. `arch_match.fetch_config` (only the tests called it) is gone.
Not done: one connection failure still marks a host unreachable for the rest of the run (by design); two UI requests that miss
the memo at the same moment both ask (no single-flight); `app.js` was not run in a browser here (no node on this box).

## 2026-10-14: fetcher fixes found in the review of PR 7 (version 0.5.4)
Prompt (operator): "merge #7. fix the fetcher bugs." Both bugs came in with PR 6 and were reproduced with a fake opener first.
- A body that fails while being read (timeout, `IncompleteRead`, reset) and an `http.client.HTTPException` such as
  `BadStatusLine` escaped `Fetcher.get` as raw exceptions, so one transient error dropped a whole collector. All of them are
  `FetchError` now. A broken body does not mark the host unreachable.
- One connect timeout marked the host unreachable for the rest of the run. A timeout now marks it on the second one
  (`TIMEOUTS_TO_GIVE_UP`); a timeout wrapped in a `URLError` counts the same. No route, DNS failure and refusal still mark at once.
- Five mutations (give up at one, each missing exception type, a read failure that marks the host, no `URLError` unwrap), all red.
- Not fixed: the UI memo reuses a degraded report for 5 minutes; the research cache never expires on a retry.

## 2026-10-14: UI layout on narrow screens, and which run is selected first (version 0.5.5)
Prompt (operator, watching the page in Firefox): the page makes you scroll a small column sideways; the running or most
recent run should be selected first or be the most obvious.
- Measured first (playwright, scrollable elements at 1440, 1024, 768 and 400 px): the left rail overflowed by 60 px (a state
  badge with `white-space: nowrap` widened the run list), the nine-stage strip scrolled sideways below about 860 px, and the
  tabs scrolled at 400 px. Now the rail's column is `minmax(0, 1fr)`, the stages wrap onto more rows, the tabs wrap, and
  below 900 px the selected run comes before the run list. Only the ledger table still scrolls sideways at 400 px.
- Run order (`webui.list_runs`): runs in progress (running, paused, or being launched) first, then every other run newest
  first. The old order put blocked and crashed runs above everything, so a 7-day-old stopped run was selected ahead of the run
  that had just finished. The state badge still says which runs need a person. The first row carries a `now` or `latest` tag.
- Mutations: no recency sort and no live-first rule both turn a test red. The first draft of the recency test passed
  without the sort, because the directory names were already in the right order; the names were changed.

## 2026-10-14: potential next choice, a music model (not run)
Prompt (operator): "is there music generation model we could try instead?" of `CiroN2022/ascii-art` (an SDXL LoRA; the
preflight blocks with `nearest-model-missing`, and the LoRA cannot apply to tt-animatediff, which serves SD 1.4).
- Base found: `MiniMaxAI/MiniMax-Music3` has a Tenstorrent bundle, `jashansinghTT/minimax-music3-blackhole` (`p150`,
  `tt-dit-server`, 0 downloads, not installed), and tt-metal has `models/demos/minimax_m3`. MusicGen and Stable Audio 3 have no bundle.
- Most MiniMax-Music3 fine-tunes are LoRA adapters (orchard has no merge step). `SonexaAI/Sonexa-Music-v0.1-Beta` ships full
  weights (57.4 GB, no license metadata, 213-byte README, 6 downloads) and its weights were not compared with the base.
- Dry run for Sonexa: blocks on `disk-full` (needs 137 GB, 89 GB free) and `nearest-model-missing` (ranks the bundle above as the
  stated base; `--base jashansinghTT/minimax-music3-blackhole` installs it). Nothing started.
- Open: the bundle's README was not read, so whether its server accepts swapped weights is unknown; the preflight warns stage 2
  expects a v6 thin bundle; the triage template was written for LLM tensors and has not seen this layout. Needs about 50 GB freed first.

## 2026-10-14: RAG-Gate-27B gate and logprobs check did not run: the host kernel compiler changed
Prompt (operator): "yes, run the gate check and the logprobs check" (to decide whether `ThakiCloud/RAG-Gate-27B` is worth a
published card; see the review above in this session). Prepared: 200 gate prompts from ChainCheck `musique_test` (70 cell A, 65
B, 65 D; at most 2,048 tokens; fixed hash order; expected Answer for A and B, Retrieve for D) in
`~/orchard-runs/thakicloud--rag-gate-27b/gate-check/prompts.jsonl`, and a driver that serves the model with stage 2's `run.sh`,
model directory and tensor cache, probes `logprobs`, then sends the prompts. Nothing was measured.
- Three boots failed in 13 s each with `trisc1 build failed ... 'vLut8si' is not a member of 'sfpi'`, a kernel JIT compile
  error: (1) default HOME and kernel cache, board 0; (2) the supervisor's kernel cache and a run-local HOME, board 0; (3) the same
  on board 1, the board stage 2 used. So it is not the board or the cache path.
- Cause found in `/var/log/dpkg.log.1`: `2026-10-13 10:45:43 upgrade sfpi 7.83.0 7.67.0`, a downgrade of the system SFPI compiler
  after stage 2 had booted this bundle (Oct 13 06:24, on 7.83.0). The bundle's kernel headers use `sfpi::vLut8si`, which 7.67.0
  does not define. Inferred, not tested: the kernel cache key includes the toolchain, so every kernel recompiles and the compile
  fails. Not done: reinstalling 7.83.0, because it is a system package that another agent or the operator changed on purpose, and
  it is the operator's call.
- Consequences: any host-served boot of the Qwen3.8 bundles on this box fails until SFPI is fixed. The 4-chip stage 4 test runs in
  a container and brings its own toolchain, so it is not affected. The preflight has no check that the box's SFPI can compile
  the nearest bundle's kernels (`lab_setup` compares SFPI between a lab and this box only).
- Side note: after each run, `gozer status` shows the idle board BUSY-UNTRACKED for about a minute, because the release's
  `tt-smi -r` and `-s` open every device (pids named in the line are those two commands).

## 2026-10-14 (later): RAG-Gate-27B gate and logprobs check ran after the SFPI upgrade
Prompt (operator): "Upgrade to latest SFPI". Installed `sfpi_7.87.0_x86_64_debian.deb` from the tenstorrent/sfpi release marked
Latest (sha256 matched the release digest and `sfpi-version`) with `sudo dpkg -i`, over 7.67.0. The bundle's kernels compiled; the server was
ready in 1,406 s (cold kernel cache). Results are in `~/orchard-runs/thakicloud--rag-gate-27b/gate-check/RESULTS.md`:
- Gate: 186 of 200 (0.930, interval [0.886, 0.958]) on ChainCheck items; over-refusal 0.074, unsupported answers 0.062; the card's own
  numbers on another set are .938, .071 and .055. Median 2.5 s per request with 4 in flight.
- Logprobs: `/v1/completions` `logprobs` works and returns the top 5 first-token logprobs, so the card's "threshold P(Answer)" use is possible
  on this server (drafter off, sampling on the host). `prompt_logprobs` is refused. The stage 2 template's comment says the server
  rejects logprobs; that did not hold here. Not tested with the drafter on.
- Not done: CPU agreement, 1-request latency, 4 chips. SFPI 7.87.0 is newer than the 7.83.0 the earlier runs used; this one run compiled with it.

## 2026-10-14 (later): CPU agreement, and a harness bug in stage 7's weight check
Prompt (operator): "cpu agreement -> v6 package -> boot check -> publish public card -> music model".
- CPU agreement for RAG-Gate-27B: 25 of 25 labels equal between the chips and the CPU model (every 8th of the 200 prompts); see
  `gate-check/RESULTS.md` in the run directory.
- Bug found by running stage 7 by hand: `served_weights_problems` could not trace the served weights on this machine, because the HF
  cache here is a shared blob store (`models--X/blobs/<hash>` links to `hub/blobs/<2 hex>/<hash>`), and the real path of a weight file
  names no repo. It said "links into no Hugging Face snapshot". `_weight_origin` now looks the file up in every `models--*`
  snapshot of the same hub, and reads a blob that two repos hold as the repo stage 0 names. 5 tests; mutations (no shared branch,
  no revision filter, no preference for stage 0's repo) turned them red; the last one first survived and got its own test.
  Clef's earlier stage 7 passed because its cache used the older layout.

## 2026-10-14 (later still): `episod/rag-gate-27b-p300` is public
Prompt (operator): "cpu agreement -> v6 package -> boot check -> publish public card -> music model". Stage 7 run by hand on the finished
RAG-Gate-27B run (the ledger still records it as skipped): `stage_all` (needed the shared-blob-store fix above), boot check under
`gozer run --chips 2 --exact 2` with the supervisor's environment (31 of 32 tokens, top-1 0.96875, ready in 1,546 s, fresh install, empty
cache), `finish`, `gate_package` ok. Only the 2-chip profile was staged: the 1-chip profile was skipped because stage 4 had no passing
1-chip configuration, and no 4-chip v6 bundle is installed.
- The generated card was rewritten for a gate: intended use and a curl example with `logprobs`, five measured rows (gate accuracy
  0.930, over-refusal 0.074, unsupported answers 0.062, 25 of 25 labels equal to the CPU's, median 2.5 s per request with 4 in flight),
  a section on what was and was not measured, and a section saying the same author's 4B, 8B and 9B models score at least as well.
  The evidence files (`evidence/gate-check/`) are redacted copies; the bundle's namespace in `RESULTS.md` became `<TT_MODEL_MODELS>`.
  `gate_package` passed after the rewrite (the first try failed on that namespace, which the scrub refuses).
- `hf upload --repo-type model --private`, then `tt-model publish episod/rag-gate-27b-p300` (public and listed in the community catalog),
  on the operator's explicit instruction. Delist with `tt-model unpublish`. Not tested: `tt-model pull` and a boot from the Hub.

## 2026-10-14 (later still): the Sonexa music run stopped at the base step; the only MiniMax-Music3 bundle is a container package
Prompt (operator): "ok go ahead and remove the caches and then go on." Deleted (operator-approved, all regenerable): the RAG-Gate-27B
tensor caches (`cache/thakicloud--rag-gate-27b`, 116 GB) and the four tensor caches of Clef's abandoned stage 2 attempts
(`stages/2.partial-2` and `2.partial-3`, 127 GB). 22 GB free became 259 GB. Kept: Clef's stage 2 cache, its stage 7 install, the shared
kernel cache. Machine config (uncommitted): the coder port moved from 8001 to 8011 (another agent's `tt-local-generator` prompt server holds
8001), and `ollama serve` with `OLLAMA_MODELS=/mnt/bonus/models/ollama` was started for the CPU stand-in. The stale lease on board 0 that
another agent's `animatediff-iguazu` left cleared on its own; nothing of theirs was touched.
- `tt-orchard bringup SonexaAI/Sonexa-Music-v0.1-Beta --base jashansinghTT/minimax-music3-blackhole` ran the preflight clean, then
  `tt-model pull` installed the base as a container package (4.1 GB image, profile p150, a `tt-dit-server`), and the command refused:
  "was pulled but is not among the installed bundles tt-orchard looks at". No run directory was created and no hardware was touched.
- Cause: orchard lists v6 thin bundles (a manifest that names the weights repo). Stages 2 and 4 swap new weights into such a bundle. A
  container package has no such manifest, so there is nothing to swap into. The refusal is the designed behaviour, and it names the
  problem. The preflight already warns about container bases for a model found by research; it does not say, before the pull, that a
  `--base` pointing at a published container bundle cannot be used.
- Not done: any bring-up of a non-LLM model. Options: pick a model whose base has a v6 thin bundle; add a container-package path to
  stage 2 (stage 4 already runs a container for its 4-chip test); or accept the blocked outcome as the documented result.

## 2026-10-14 (later still): `SonexaAI/Sonexa-Music-v0.1-Beta` is a copy of the base, not a fine-tune
Prompt (operator): "continue with #1. our goal here is to make music happen from a specialized model." Before building the audio check I
read the music container (`tt-model/minimax-music3`, from the pulled manifest) and compared the two Hub repos by file hash.
- The server reads `MUSIC3_SNAPSHOT`, or a directory in `HF_MODEL`, before the Hub (`config.resolve_snapshot`), so swapped weights can be
  injected. The base ignores `flowmatching_vae.pth`, `dav.pth` and `qwen_7B/*`, so a model directory needs those from somewhere else.
- Hub tree comparison (LFS sha256, or git oid for small files): both repos have the same 88 paths. 87 files are identical, including every
  safetensors shard and the LICENSE. Only `README.md` differs (11.7 KB in the base, 213 bytes in Sonexa). Sonexa is a re-upload of MiniMax-Music3.
- Consequence: a bring-up of Sonexa would measure the base. Nothing was started, downloaded or run on hardware, and no audio check was built.

## 2026-10-15: the MiniMax-Music3 container hangs on this box when it loads a cached tensor over about 32 MiB
Prompt (operator): "continue on" with option 3 (serve the base once for a clip, search for a real fine-tune). Done by hand, outside
the supervisor, with `~/orchard-runs/music/run_clip.sh` under `gozer run`. Nine boot attempts; none reached a healthy server.
- Cause found (attempt 7 and 9, then a small repro): the thread that builds the layer-0 KV cache spins in `ttnn.as_tensor` at
  `core.py:817`, `load_tensor_flatbuffer(cache_file_name, device=...)`. It runs at 100% system time, about 2.1 million page faults a
  second, on a `rw-p` mapping of the file; gdb and `py-spy` (blocking mode) cannot stop it, and `docker stop` needs its SIGKILL
  step. Repro (`kv_probe.py`, one chip, mesh 1x1): `as_tensor` of a zero tensor with a `cache_file_name` dumps in 0.1 s, and the
  load back hangs at (2,8,12288,128) bf16 TILE (50 MB) and (2,8,16384,128) (67 MB). It loads in 0.01 s at 6144 and 8192 (33.5 MB =
  2^25 bytes) and below. The same file written by an earlier boot also hangs (attempt 2 stalled loading the cached 134 MB embedding).
  The device and a plain `from_torch` of the same shape are fine (0.04 s). Serving the weights from tmpfs did not help. Host: Linux
  7.0.0-38, KMD 2.11.0, firmware 19.16.0, image tt-metal `v0.79.0-dev20260907`. Not known: whether this is the image, the kernel
  or both; other agents' LLM containers load large cached tensors here without trouble, and I did not find out why this one differs.
- Mistakes of mine, each cost a boot: (1) I started `tt-model serve` with no timeout, so a hung container held a lease for 8 hours;
  (2) `tt-model serve` pins `/dev/tenstorrent/0`, and gozer granted board 1, so the container opened a chip outside the lease
  (board 0 was free; no harm seen). The script now maps each leased chip to its device node; (3) I moved the tensor cache aside, and
  docker created the missing mount source as root, so the next boot failed with `Permission denied` and then sat idle until the stall
  check; (4) a launch while the chips still showed BUSY-UNTRACKED was queued and exited, and I read that as running for 15 minutes;
  (5) my cleanup removed the container before saving its log.
- For orchard (not built): a supervised hardware test has `deadline_s` (2,400 s stage 2, 3,600 s stage 4), so this would have ended
  in under an hour with `timed_out`. It has no stall detector, and 40 to 60 minutes is long for a log that stopped. A stall rule
  needs the longest silent stretch in healthy cold boots, which I have not measured.
- Findings about the models: `SonexaAI/Sonexa-Music-v0.1-Beta` is a copy of the base (see above). The real MiniMax-Music3 fine-tunes
  on the Hub are LoRA adapters (SimpleTuner `pytorch_lora_weights.safetensors`). The best one found is
  `terminusresearch/minimax-music3-lm-lora-fiona-crapple` (Apache-2.0, rank 64, alpha 64, language model only, trigger word
  `fiona crapple`; `regularised/checkpoint-750` is the author's pick). The reggae LoRA by bghira describes itself as a failed experiment.
  `orchard/skills/lora-merge-templates/merge_lora.py` merges an adapter into one component's shards (9 tests, 8 mutations red; the 9th
  was an equivalent mutant and the code was simplified). It has not been run on the real adapter.
- Not done: a clip from the base, the merge, a clip from the merged model, the audio check for stage 2.

## 2026-10-15 (later): MiniMax-Music3 base and a LoRA-merged copy both made music on one chip
With a local workaround for the cache-load hang (a patched `ttnn/operations/core.py` mounted read-only over the image's file, so
`as_tensor` never loads the flatbuffer cache), the container booted healthy in 76 to 165 s. Results in `~/orchard-runs/music/RESULTS.md`:
- Base clip, 20 s stereo 32 kHz, seed 42: peak 0.98, 84.5% of 100 ms windows above silence; 65 s of compute (denoise 23.9 s, depth
  decoder 19.1 s, language model 18.0 s, vocoder 14.6 s, VAE decode 3.5 s).
- `merge_lora.py` merged `fiona-crapple` `regularised/checkpoint-750` (252 matrices, largest relative change 4.8%) in 27 s. The container
  served the merged directory (`HF_MODEL` = the directory, mounted with the blob directories its links point to) and made clips.
  A repeated request gave a bit-identical file, so a seed is deterministic.
- Not shown: that the LoRA's style transferred. One seed per cell, no listening test; a tiny weight change alone changes an
  autoregressive sample. Nobody has listened to the clips (`~/orchard-runs/music/*.flac`).
- Still outside orchard: all of this ran by hand with `run_clip.sh`. No stage 2 audio check exists, and `bringup` still refuses a
  container-package base. The workaround patches vendor code in a mount; it is a local experiment, not something to ship.
- Leftover on disk: the 4.1 GB image, the base weights in the HF cache (27 GB), `fiona-merged` (27 GB), `tensors.*`/`weights.*` scratch
  caches in `~/.cache/tt-model/minimax-music3/`. Nothing published or uploaded.

## 2026-10-15 (later still): multi-seed clips, and the audio bring-up spec
Prompt (operator): "proceed! this is progress. I have confirmed these clips have MUSIC!", then option 1 (a real orchard path for audio).
- Multi-seed set (`~/orchard-runs/music/set1/`, `features.json` there): 5 seeds x 2 captions x 2 models, 20 clips, 20 s each, same
  seeds and captions for base and merged. Coarse spectral features (centroid, share of spectrum above 4 kHz, onsets per second), merged
  minus base per seed. With the trigger caption the share above 4 kHz is higher in 5 of 5 seeds (mean +0.138, sd 0.054) and the centroid
  in 4 of 5; with the synth-pop caption there is no consistent shift (high share 2 up, 3 down; onsets up in 5 of 5 by a mean 0.35).
  So the merge changes the sound when the trigger is present and little when it is absent. That fits an adapter that acts on its
  trigger; it does not show cafe-jazz style, and n = 5 with crude features cannot. Listening is still the test.
- Clip checks measured on 24 clips (these set the bars in the spec): duration 20.02 s for 20 s asked, 32 kHz stereo, at most 0.019%
  of samples at full scale, at least 59% of 100 ms windows above -40 dBFS.
- Brainstorming for the audio path: classified architectural; understanding note and design agreed in chat (facts `form`, `serving`,
  `modality` on the weights-only path, no new class; stage 1 materializes the merge; stage 2 boots base and new model in a container and
  generates clips; the harness probes for the cache-load hang and selects the workaround itself, the operator's ruling). The written spec is
  `docs/superpowers/specs/2026-10-15-audio-bringup-design.md`; it waits for the operator's review before a plan is written.
- Slips while running the set (all fixed in `run_clip.sh`): a slash in the clip label broke the log file names, so the first launch
  never started a container and my waiter did not notice for 30 minutes; the device node in `tt-model serve --print` depends on which
  chips are free, so a guard that expected node 0 failed once; two of my waiters exited on stale text in old logs.

## 2026-10-15 (later): PR 8 review (version 0.5.6)
Prompt (operator): "some comments on PR 8". Copilot's review left two findings on `fetcher-fixes`; each was checked against the code.
- `merge_lora.py`, component paths. A `--component` spelled `./language_model`, `language_model/` or `.` did not match the normalized
  paths `link_tree` compares, so the adapted shard was symlinked to the base first and then written. With safetensors 0.8.0 `save_file`
  replaces a link by rename, so I could not make the base change on this box (checked with real files). The mismatch was still real: a
  `save_file` that writes through a link overwrites the base weights, and the tests now use such a shim, which turns 9 of the new cases
  red on the old code. Fixed: the component is normalized (`.` for root-level weights, absolute paths and `..` refused), the
  skip set is normalized, a weight index that names a shard outside the component is refused, and `write_shard` refuses a destination that
  is a link. 24 tests; 8 mutations red, 1 equivalent (a second normalization in `link_tree`, removed).
- The audio spec relied on the template's `finally` to stop containers. Correct: on a timeout or an abort `orchard.agent._kill_session`
  sends SIGKILL to the process group, which skips cleanup, killing the Docker client does not stop its container, and `_hardware_phase`
  (the single-test path stage 2 would use) has no sweep; only the listed-tests path calls `_sweep`. The spec now requires labelled
  containers and a supervisor sweep after every stage 2 test, at resume, and before the coder is restored, with a block when one survives.
  This is design only; no supervisor code changed.

## 2026-10-15 (evening): the merged model served again, a Gradio demo, a tt-disco entry, and a card (repo `~/code/tt-minimax-music-blackhole`)
Prompt (operator): "load the model back up for me to utilize. add a gradio app and a tt-disco entry. make a model card according to
standards" (tt-model-manager's latest), then "does the gradio travel with the model repo itself" and "companion repo it is. episod for the
HF namespace". Built outside tt-orchard, in a new local git repo with no remote (planned `tsingletaryTT/tt-minimax-music-blackhole` on GitHub and
`episod/minimax-music3-fiona-crapple-blackhole` plus the weights repo `episod/minimax-music3-fiona-crapple` on Hugging Face). Nothing
was created remotely, pushed or uploaded; each needs the operator's explicit go-ahead.
- Serving: the merged model has run on port 20000 (hand-run under `claude:music-serve` on board 0) since about 08:25 PDT; stop it by
  killing the `run_clip.sh` pid in `~/orchard-runs/music/run-serve.log`. A second copy runs from the tt-disco entry on board 1 (server on 20001, page on 7871).
- Batch: 10 requests at once. The server accepts 9 jobs (one running, eight waiting) and answers the tenth with HTTP 429; all 10 finished
  in 641 s. One clip ended early (16.6 s of 20 s, `ended: true`), so the audio spec's duration check now allows that case.
- Repo contents: `demo/` (Gradio page and a stdlib client for the job API), `serve/serve_music.py` (launcher), `serve/kv_probe.py`, `serve/merge_lora.py`
  (a copy of orchard's, with its tests), `.disco/app.yaml`, `package/tt-model.yaml` (an unbuilt draft whose `card:` block is the card source),
  `package/render_preview.py` and `package/build_card.py`, `hf/README.md`, `weights/`. 111 tests, none needs a chip. Mutations: client 10 of 10 red, launcher 24 of 24
  after two added tests, app 14 of 14 after three added tests, card builder 5 of 6 (the title replacement had no test; added).
- Found by running it for real under tt-disco: (1) a systemd user unit has no `tt-model` on PATH, and the launcher died with a raw
  traceback (now a PATH fix and a clear error); (2) two containers on different boards blocked each other at "Waiting for lock
  'CHIP_IN_USE_0_PCIe'". The lock is a file in `/dev/shm`, which `--ipc host` shares, and its name uses the chip's number inside the
  container. The launcher had renumbered every chip to 0. `tt-model` and orchard's stage 4 template keep each node under its own number, and the
  launcher now does too (both containers then ran at once). The audio spec says so. (3) The probe on this host does hang, so `--workaround auto`
  selected the patched `core.py` on its own; a killed probe did not block the next boot (the earlier worry); the lock was the cause.
- The same seed, lyrics and caption gave a bit-identical file on the board-0 server and the board-1 server (one pair), and on repeat on one server.
- The card: the generator (`render_model_card`) accepts the `card:` block and renders the standard layout. A companion repository is not a package, so
  `build_card.py` changes the tags (no `tt-model-catalog` or `tt-model-container`), title, quickstart and provenance and leaves every other section as
  generated. Numbers are the server's own timings for 20 requests (median 65.6 s of compute for 20 s of audio; 3.3 s per second). Performance says no quality
  evaluation has been run; that is a fact, and the card carries no claim about how the clips sound.
- PR 8 (the review comments the operator mentioned) is handled in the entry above.
- Not done: a GitHub repo, any Hugging Face upload, the full container package (needs the image build and the tt-metal tree the base was built from),
  a quality evaluation, a browser run of the Gradio page (it was driven through `gradio_client` only).

## 2026-10-15 (evening, later): published on the operator's explicit instruction
Prompt (operator): "publish the model public and the GH repo public", after "works great!" and "those screenshots should go in the readme and model card".
- GitHub: https://github.com/tsingletaryTT/tt-minimax-music-blackhole, public, `main`, topics tenstorrent, blackhole, music-generation, gradio
  (`gh repo create ... --public --source . --push`).
- Hugging Face, public: `episod/minimax-music3-fiona-crapple-blackhole` (the companion: a `git archive` of the repo with the generated card as `README.md`,
  `hf/` dropped; 27 files, docs/ screenshots included).
- Hugging Face weights `episod/minimax-music3-fiona-crapple`: `hf upload` of the merged model directory (excluding its README and MERGE.json) is
  running at about 5 MB/s (Xet deduplicates the unchanged base files; the new data is mostly the 17 GB of rewritten language-model shards). The repo was
  made private while it fills, and `~/orchard-runs/music/finish_weights.sh` waits for the upload, adds the card and merge record, checks the four rewritten
  shards' sha256 against the local files and every other large file against the base repo, and only then makes the repo public. If it fails it leaves the repo
  private (`finish-weights.log`). Nothing was added to the tt-model catalog (no tt-model tags; it is a companion, not a package).
- The license (MiniMax-Music3 Community License) allows redistribution with the notice kept (the weights repo carries the LICENSE file) and asks commercial products
  to display "MiniMax-Music3" (the page title does). The screenshots were cropped to remove the browser's tab bar and address bar.
- Found from the operator's screenshots, the first browser run of the page: a 20 s request with two short sentences of lyrics gave 5.2 s of audio (the server ends a
  song when its lyrics are finished) and the page printed "5.18966 s" with no explanation. The page now rounds to one decimal and says why the song ended early
  (5 tests, 2 mutants added after the first mutation run). The published screenshots predate that wording and the running page still has the old code until its next
  restart, so the card and README make no claim about the note; a retake would show it.

## 2026-10-15 (evening, later still): new screenshots, and a bug the browser found
Prompt (operator): "get the new screenshot". No browser tool was available in the session, so I drove headless Chrome (`/usr/bin/google-chrome`) with the official
`playwright` package in a scratch venv (`executable_path`, no browser download), against a temporary copy of the page on port 7872 pointed at the server on 20000.
- Bug found by the first scripted run, missed by 23 tests: after the "More" panel was touched, a blank steps box reached Gradio as 0, and the component's own
  `minimum=1` failed the request ("Value 0 is less than minimum value 1") before `run_generation` could treat 0 as "server default". Fixed by removing the bounds on the
  component (the client still refuses steps outside 1 to 100 with a plain message); two new tests (the real app via `gradio_client` with steps 0, and steps 101 and 500),
  mutation (bounds put back) red. The tests had driven the app through `gradio_client` with None, never with 0.
- Screenshots: same inputs as the operator's own run (the two-sentence lyric, "acid jazz", seed 57393907). The seed gave 5.2 s again, so the early end repeats for a seed.
  Page text now reads "The song ended at 5.2 s, before the 20 s asked for: the lyrics were finished." The README and the card say the page explains an early end again.
- Pushed to GitHub (`2f8c3d5`) and re-uploaded to the Hugging Face companion repo. The page running under tt-disco on 7871 still has the old code, so it has the blank-steps
  bug until it is restarted from the catalog. The weights upload was at 7.6 GB; `finish_weights.sh` has not run yet.

## 2026-10-18: making unattended bring-up need fewer people (branch `autonomy-pr`, version 0.6.1)
Prompt (operator): "proceed in making the harness 'good at' bringing up models without Claude supervision", after the RAG-Gate replay
test. Method: list every stop in the past runs that a person cleared, keep the ones whose cause is known, and fix each test first with a
mutation per guard. The causes come from the entries above (RAG-Gate-27B, Clef, the SFPI downgrade, the music run). Each change is a
precaution or a check with a stated evidence base; none was seen to change a real run's outcome yet.
- Pre-test reset (`supervisor._reset_test_chips`): a test that needs two or more boards resets its leased chips in place before it
  runs. The park resets only the coder's board, and the RAG-Gate 4-chip test failed twice on chips earlier tests had used and passed
  after all four were reset. The cause was not isolated. A refused reset blocks with its text.
- Preflight `cpu` row: the CPU stand-in (ollama) must answer and list the tier's model. On the RAG-Gate run it was down and the park's
  canary failed three hours in. Blocks as `coder-unusable`.
- Preflight `sfpi` row (`orchard/sfpi.py`): the box's SFPI must be at least what the chosen bundle pins in its ttnn wheel. A system
  update to 7.67.0 broke every host boot of the Qwen3.8 bundle (pin 7.73.0); 7.83.0 and 7.87.0 booted it. Three observations, so the
  rule is stated as fitted to them. Blocks as `hardware-unhealthy`; installing the package stays the operator's.
- `bringup --auto-retries N` (default 2, 0 off): a run blocked as `coder-unusable`, `hardware-unhealthy` or `agent-stuck` is retried after
  60 s (a released board shows busy for about a minute). Other blocks are never retried (`blocked.AUTO_RETRY`).
- `bringup --prune-caches`: when disk is the only block and removing regenerable tensor caches can cover it, remove the oldest ones no
  unfinished run uses, just enough; never the kernel cache or anything outside the cache root; all or nothing; `--dry-run` only reports.
- Decided against (for now): a hardware-test stall rule. Six healthy cold boots had at most 198 s between log lines; the music hang was
  silent 40 to 60 minutes. But a test's own stdout is nearly silent during a healthy boot (its server log goes to a file), and the hang
  burns CPU, so neither signal the stall watcher has would catch it. A rule needs "no file under the stage directory grew for N s".
- RAG-Gate-27B replay (kernel probe): see the ledger for the comparison written before the run.
- Found on the live box: the Coder-Next container answered `/v1/models` while its prefill had hung after the overnight suspend; only a
  completion shows that. `coder-speed` timed out on it. A bring-up's coder canary asks a real question, so it would catch this at start.
- Not done: the live unattended run of a new model. It needs both boards, so the kernel-probe coder (also used through qwencode) has to stop
  first, which is the operator's decision. Candidates checked by dry run: `bytkim/Qwen3.8-27B-pi` (preflight passes apart from the coder
  port), `tsinghua-sigs-robot-lab/VeriLoop-E2` and `perplexity-ai/pplx-decider-v1.1-27b` (disk: 136 and 132 GB needed, 111 free).

## 2026-10-18 (later): the first unattended bring-up with the new setup (`bytkim/Qwen3.8-27B-pi`, version 0.6.1)
Prompt (operator): "go ahead" to stop the kernel-probe coder and run an unattended bring-up of a new model. Run directory
`~/orchard-runs/bytkim--qwen3.8-27b-pi`. It ended `ready-for-operator-review` after six launches of the same command. Stage 2 (one chip):
32 of 32 tokens, ready in 682 s. Stage 4: 2 chips 1.0, 4 chips 0.96875 (ready in 254 s). Stages 3, 5, 6, 7 skipped by design. Nothing was
published, pushed or uploaded. Interventions, each with its cause:
1. ollama was down after the suspend. The new `cpu` preflight row blocked before anything started (the row did its job). I restarted ollama
   with `OLLAMA_MODELS` set.
2. A cache entry for the model held only `config.json` and `README.md`. `local_snapshot` counted any snapshot with a config as local, so the
   59 GB of weights were never fetched, stage 0 compared an empty model and called it a full port, and stage 1 failed four times. Fixed: a
   snapshot is local only if a weight file exists behind its link (2 tests, mutation red).
3. A retry kept stage 0's stale verdict. `bringup` now moves the earlier run directory aside (`<run>.stale-<time>`) when it has to fetch the
   weights (3 tests, 2 mutations red). For this run I did the move by hand once, because the weights were already on disk.
4. Disk: stage 4 wants 110 GB free at entry. Twice the free space was short (49 GB, then 109 GB). The operator chose deleting the Clef and
   RAG-Gate run directories (kept small files in `*.kept`: ledger, evidence, package staging, gate-check results). Then I removed this run's
   own stage 2 and 4-chip caches. `--prune-caches` could not help: the only caches were this run's own.
5. The 4-chip test failed with `Fabric Router Sync: Timeout` after 57 s, the same as the RAG-Gate run. The pre-test reset had run and did
   not prevent it, so the 2026-10-13 "stale chip state" guess is not enough. By hand, one lease on all four chips, one `gozer reset`, then the
   supervisor's script passed (31 of 32, ready 302 s). Resetting board 1 then board 0 (the supervisor's order) got past fabric start and then
   hung loading the warm tensor cache, which looks like the cached-tensor load hang found on the music container (2026-10-15). I stopped my
   own container by id and released my leases. The cause of the fabric failure is still not isolated; the same procedure passed on the last
   attempt (260 s, top-1 0.969).
Added from this: a multi-board test that fails within 180 s without timing out is reset (every leased chip) and run once more
(`FAST_FAIL_S`, 5 tests, 5 mutations red). That retry did not fire in the final attempt, so it has not run on real hardware.
Also seen: a launcher that saves `$!` after `setsid nohup` saves the wrapper's pid, which exits at once; wait on the run's `supervisor.pid`.
The stall rule is still deferred. A shared-venv, no-weights stage 0 verdict is still possible if a snapshot is replaced mid-run.
Not done: the 4-chip warm-cache hang (not investigated here). The run's version numbers (0.7.0, 0.7.1) belong to the combined branch; split out, these changes are 0.6.1 (a self-review found that the set-aside also fired for a run that had passed stage 2 whose snapshot was later deleted; it now keeps such a run).
## 2026-10-15: design for attempting kernels when no equivalent model exists (branch `kernel-attempt`)
Prompt (operator): "What would it take for tt-orchard to take the initiative with models it can't find an equivalent for and
attempt to write the kernels (in tt-lang for example, using the present skills) instead?" Then: go with the op inventory and
the kernel probe, keep in step with the tt-model-manager branches ("metadata being part of what we need clarity on"), keep
provenance orchard-side, "don't draft an upstream fix. fix for yourself for now only. the goal is to get a branch where this
logic is in place." Spec: `docs/superpowers/specs/2026-10-15-kernel-attempt-design.md`. No product code yet; the spec waits for
the operator's review, then a plan.
- Corrected my own first answer: handing over agent-written ops is mostly possible already. The installed `tt-model package-thin`
  takes `--extra-code` and `--ops-wheel` (generic_op wheels); a new registered C++ ttnn op still needs v5 or v5.1, which orchard
  refuses. tt-model-manager's `deps.custom_ops` records only `--ops-wheel` wheels that `model.py` imports statically.
- ttnn op names are not readable without importing ttnn (zero hits in its `.py` files), and importing opens the device. The
  coverage list is a dump taken under a lease and stamped with the ttnn version. My first plan, an `ast` scan, would not have worked.
- tt-lang: the shared venv's `tt-lang 1.1.6` is the latest on PyPI; the repo is at v1.1.10 plus 23 commits. A simulator venv
  (`~/orchard-venvs/ttlang`) and a clean clone (`~/orchard-src/tt-lang`) were built outside the repo. The operator's own
  `~/code/tt-lang` (85 commits behind, edited submodule, 22 GB of builds) was left alone.
- Bug found in the tt-lang-sim wheel (1.1.6 and main): it loads `ttl/ttl/diagnostics.py`, a source-tree path the wheel does not
  ship, so a diagnosed kernel error becomes `FileNotFoundError`. Upstream's negative examples exit non-zero both with and
  without the bug, so an exit code cannot show it. Copying the source `diagnostics.py` into the venv fixes it locally; the
  environment command in the spec does that and then checks for the named diagnostic text. Not reported upstream, by decision.
- Process slips: my first run of all upstream examples used a launch setup unlike upstream's tests and gave 7 of 20; the second,
  with the source `python/` directory on the path, failed all 25 identically because it shadowed the wheel's `ttl` with an
  uncompiled one. Neither result was evidence about the simulator. The third run (helper package only) gave 12 of 25, and the
  causes of the 13 failures are listed in the memory note, partly unverified.

## 2026-10-16: the kernel attempt, implemented (branch `kernel-attempt`, version 0.6.0)
Prompt (operator): "approve and continue with subagent development. CONTINUE UNTIL IT WORKS OR I SAY OTHERWISE. Stop looping and
watch for bad behavior. Instead of pure timers, we need watchers." Spec `docs/superpowers/specs/2026-10-15-kernel-attempt-design.md`
(section 11 lists every change from the approved design), plan `docs/superpowers/plans/2026-10-15-kernel-attempt.md`, ledger
`.superpowers/sdd/2026-10-15-kernel-attempt/progress.md` with a report per task. The goal is a measurement: can the local coder write a
working tt-lang kernel for an op, and how many ops of a new model have no ttnn equivalent.
Built, by task:
- Task 1, `tt-orchard kernel-env`: a tt-lang checkout and a simulator venv, with the diagnostics patch. It completes an interrupted clone
  and never resets one at another commit (the plan's code deleted it, which blocked every retry after one failed fetch).
- Task 2, the command wiring in `orchard/cli.py` and the README section.
- Task 3 and 3b, `orchard/stallwatch.py`: every long child of `kernel-probe` (the simulator check, the smoke test, the hardware runs)
  runs under a watcher that stops it on a stall (no output and no CPU) or a repeating output line. The stop sends SIGTERM to the process group first and SIGKILL after a grace period, so `gozer run` can release
  its lease. The first SIGTERM is guarded (pgid above 1, a group other than the caller's, child leads it); the SIGKILL checks the first two and that the group has members, and skips leadership because the leader may be reaped.
- Tasks 4 and 4b, the op inventory template: traces a model on the meta device with `torch.export`, counts ops per static node, scales
  in-layer counts, follows higher-order bodies and keeps python-level nodes out of the op list.
- Tasks 5 and 6, `orchard/opinventory.py` and `inventory refresh-ttnn`: statuses mapped, composable, unmapped, ignored and unchecked; a
  committed ttnn dump.
- Task 7, the stage 0 hook (step 9 of the delta-triage skill, after `delta.json` is final) and Task 11, the bundle section.
- Tasks 8, 9 and 10, the kernel loop templates, `orchard/kernelprobe.py` and `tt-orchard kernel-probe`, with four ops (`add` as a
  calibration op, `silu_mul`, `rmsnorm`, `linear_scan`).
Review findings that changed the design:
- A PCC-only check passes a kernel whose output is scaled, because PCC ignores scale. The check now also needs relative L2 error, slope
  and max-abs-error bars (provisional, in the op specs). A best-effort `ast` gate stops a kernel that computes with torch and calls
  that a pass; it reports every violation with a remedy and adds the category `forbidden_code`.
- Trust boundary: the probe first read its verdict from a directory the agent could write and followed links. Now the probe re-runs the
  shipped check on its own copy under `.verdict/<op>-<n>/` (regular file, at most 256 KiB, `O_NOFOLLOW`), re-checks the sha256 before
  the gozer call and caps every read of an agent-writable file. It is best effort against shortcuts and accidents.
- A device-open failure was blamed on the kernel. The harness `setup` phase and the category `harness_error` fix that.
- The ttnn dump script was fragile: classes were listed as ops, and one failing attribute ended the script before its marker. It now
  skips classes and records names that raised under `skipped`.
- The first inventory missed ops inside higher-order bodies (the rotary embedding's cos, sin and to.device). Task 4b traverses them.
- A crash or interrupt lost the whole probe report. The report is now written from the records so far (exit 3 crash, 4 interrupt, 5
  hardware hang).
Live results:
- L1: `kernel-env --check` passed all six steps (exit 0).
- L2 and L3: the inventory ran on default configs of Qwen3, RWKV, Mamba2 and Moshi's text model (`MoshiForCausalLM`), 2 layers
  traced. The first `inventory refresh-ttnn` ran under `gozer run --chips 1 --exact 1` and wrote the dump for ttnn
  0.65.1rc17.dev6200: 821 names, none skipped, lease released, all four chips free afterwards. Final run on the final code
  (mapped / composable / unmapped / ignored): Qwen3 26/0/6/3; RWKV 17/1/2/2; Mamba2 30/1/4/3; Moshi's text model 26/1/7/3.
  Unmapped: Qwen3 `__and__`, `index`, `to.dtype_layout`, `diff`, `new_ones`, `to.device`; RWKV `copy_` (256 scaled), `div_` (4);
  Mamba2 `einsum` (320), `bitwise_not` (256), `masked_fill` (256), `split_with_sizes` (128); Moshi `type_as` (65), `to.device`
  (32) and the five Qwen3 ones other than `to.device`. An earlier run before the higher-order-body fix gave other numbers: the fix
  added the ops inside the rotary-embedding bodies and removed the python-level `getitem` and wrapper nodes. `rsub.Scalar` and
  `conv1d` are mapped now. By name, each architecture's ops are almost all in ttnn's name list. The unmapped ones are a short list
  of cast, in-place, indexing and masking ops plus Mamba2's `einsum`. `mapped` is a name match, so this does not say the models
  would run or run fast.
- L4 and L5 (the probe against the live coder, and the hardware phase) are run by the controller and added by the controller.
Notable moments and mistakes:
- The terminal died on 2026-10-16. Cause: a new test fed a symlink to `/dev/zero` to an uncapped `read_text()`, and the kernel OOM
  killer stopped it (exit 137). My first theory, `os.killpg(0, ...)` in the stallwatch tests, was wrong; the guards stay as hardening.
  That test now runs in a child process limited to 2 GiB with a 60 s timeout. Tests that spawn or kill processes run under `setsid -w`.
- While a suite was running I ran one `gozer run --chips 2 -- env` to see what a lease hands its child; it released and reset board 0, which was free and unused, and no chip was opened. It was avoidable.
- The first parse of the ttnn dump listed classes as ops, which led to the `callable()` fix above.
- Reviewers found the large gaps (the scale-blind check, the forgeable verdict, the lost report); the authors' own tests had passed.
Process: two parallel worktree streams (`ka-inventory`: tasks 4 to 7 and 11; `ka-probe`: tasks 8 to 10) merged into `kernel-attempt`;
more than 20 review rounds; most tasks had implementer-run mutation tables (the counts are the implementers' reports; reviewers read them and spot-checked some, and the controller did not re-run them); for two guards in Task 9 (the regular-file read layers) both layers were removed together, so no single layer was seen red alone; the ledger is the record. Full suite after
the merge: 3,860 passed, 2 skipped.
Not done:
- The hardware phase (L5) has not run. L4 and L5 results come from the controller.
- Success criterion A is not met: Qwen3 has 6 unmapped ops (`__and__`, `index`, `to.dtype_layout`, `diff`, `new_ones`, `to.device`). They
  stay unmapped because their ttnn semantics are not provably identical to the aten ops. Mapping them is a table decision for the
  operator or a later task.
- Only `kernel-probe` runs its children under the stall watcher. These still use timers only: `inventory refresh-ttnn` (a 1800 s
  `subprocess.run` timeout), the `inventory` trace, the `kernel-env` steps, and the agent's own shell commands (the `Tools` timeout).
  They can adopt the watcher later.
- The stall watcher's 60 s quiet period and the probe's other budgets are provisional and unmeasured.
- Stage wiring (option C in the spec) is the operator's decision from the probe's table.
- The `generic_op` export route is not built. The full-port stages are untouched.
- The tt-lang diagnostics patch is local and has not been reported upstream, by decision.
- Deferred minors for each task are in the ledger.

## 2026-10-16: fixes after the final review of the kernel attempt (branch `ka-final-fixes`)
Prompt (controller): fix the final whole-branch review's findings (one Critical, eight Important, two minor), then a re-review's
regression and its smaller items, test first, one commit per item, with a mutation per new guard. Worked in its own worktree while a
live probe batch ran the shipped templates from `kernel-attempt`. Reports: `.superpowers/sdd/2026-10-15-kernel-attempt/final-fixes-report.md`.
Key decisions (spec section 11, items 17 to 25):
- gozer gets 120 s after the watcher's SIGTERM, because its lease release runs a chip reset of about 42 s. The first 5 s grace
  killed gozer during the reset.
- The probe stops stray agent processes before any verdict: same uid, `ORCHARD_RUN_DIR` exactly the attempt directory, SIGTERM,
  2 s, a second scan, SIGKILL. `stopped` means the process is gone. It cannot see a process whose /proc environ no longer
  holds the value (an exec with a new environment; `clearenv()` alone does not change it) or a non-dumpable process.
- The report gives the watcher's longest quiet stretch and longest repeat for each hardware run, so its limits can be set from a
  measurement; it also has a status column and notes on `add` and the simulator's f32.
- The static gate allows only listed names from `typing` and `__future__` and refuses forbidden names as attributes and imports.
  `from ttnn import shutil as s` still passes: the gate cannot tell a module from a name without importing ttnn.
- `opinventory.main` defaults `--ttnn-python` under the passwd home, because in an agent shell HOME points inside the run directory.
- The ops-inventory template builds only a `PreTrainedModel` subclass.
- kernel_check runs the simulator in its own session with `PR_SET_PDEATHSIG` and a SIGTERM handler. The first version of this fix
  (session only) orphaned the simulator when the watcher stopped kernel_check; the re-review found it.
- `kernel-probe` holds an flock on `probe.lock` in its output directory. A first version (a pid file with a staleness
  check) let two probes run at once after both judged a stale lock dead; the re-review found it.
Notable moment: an I7 test written against the old code had a child print without end under a 60 s backstop. It wrote a 48 GB log
under `/tmp/pytest-of-ttuser` and took the root disk to 97% while the live probe ran. I deleted it at once. Tests that make large
output now run the child under `RLIMIT_FSIZE` (16 MiB) and write a bounded amount.

## 2026-10-16: kernel probe sampling (branch `ka-final-fixes`, round 3 of the fix wave)
Prompt (operator, through the controller): "as much as we can make determinism happen when it is useful, we should do it. but we
should also make it easy to adjust temperature so wild, zany or ingenious results can be found too".
- Live finding at the code before this change: every request sent `temperature: 0` (greedy decoding asked for), and attempts
  at one prompt were often the same sample. `add` gave 1 distinct kernel in 3 attempts, `linear_scan` 1 in 5 (identical: 9 turns
  and 198 s each), `rmsnorm` 3 in 5, and `silu_mul` 5 different kernels in 5 attempts, 26 to 37 turns each; why `silu_mul`
  diverged was not measured. The report's "independent draws" sentence was wrong, and so was calling greedy attempts
  repeatable: repeats are common at temperature 0 and not guaranteed.
- `AgentStep` takes temperature, top_p and seed; its default request is byte for byte the old one. `kernel-probe` takes
  `--temperature`, `--temperatures`, `--top-p` and `--seed`; greedy stays the default. No seed at temperature 0; above it,
  attempt k sends base + k, and a base derived from the probe id is recorded so a run can be repeated with `--seed` (if the
  server honours seeds, which is not verified).
- The report splits each op by temperature and counts distinct kernels, with a warning when all attempts wrote one kernel.
- Same round: kernel_check now exits by the signal that stopped it (it used to exit as SIGINT for every signal), and an
  interrupt can no longer be lost inside the sweep's SIGKILL pass. Spec section 11, items 26 and 27.


## 2026-10-16: the latest skills (Task 14, branch `ka-skills`, version 0.6.1)
Prompt (operator, through the controller): "WE USE THE LATEST tenstorrent/skills and tt-lang skills available. We check in
preflight whether we have the latest even." The live probe had given the coder only `kernel-probe.md` and `eltwise_add.py`;
`rmsnorm` passed 0 of 10 and `linear_scan` 0 of 6 on tt-lang API mistakes, and the installed tt-lang skill was never used.
- `orchard/skillsfresh.py` compares the tenstorrent/skills clone(s), the tt-lang checkout and the installed tt-lang skill with
  upstream, read only (a runner that refuses every git subcommand that could write, `gh api` GETs only, github.com only,
  `GIT_OPTIONAL_LOCKS=0`, a network budget). `kernel-probe` and `kernel-env --check` refuse a stale tt-lang source unless
  `--allow-stale-skills`; the bringup preflight's `skills` row warns and never blocks. Spec section 11, item 28.
- Found while reading the skill: it documents tt-lang specification 0.12 and the pinned checkout is at 0.20 (`ttl.kernel` is
  now `ttl.operation`, `reduce_sum` lost its scaler and takes a `shape`, `broadcast` and `fill` moved to `ttl.block`). The
  probe therefore also gives the coder the checkout's specification and `eltwise_broadcast_reduce.py`, and `kernel-probe.md`
  lists the renamed names. A test reads the simulator's source and checks each claim.
- On this machine on 2026-10-16 (read only): the tt-lang checkout is 1 commit behind upstream main (a compiler-only change),
  so the probe now refuses without `--allow-stale-skills` until the pin moves; the tt-lang skill matches its upstream; the
  tenstorrent/skills clone is 3 commits behind, dirty, on `tsingletaryTT/squeeze` (a bringup warning).
- Not measured: whether the reference files change the probe's pass counts.
- Fix wave after the independent review (`.superpowers/sdd/2026-10-15-kernel-attempt/task-14-review.md` in the
  `kernel-attempt` worktree): I1 to I3, M1 to M9, and git hardening (no redirects, no fsmonitor or hooks, process-group
  kill on a timeout), one commit each, test first. Notable: the conftest guard now fails a test that reaches the real
  git or gh runner; a process test first passed under its mutation because the call still waited the full 60 s for the
  grandchild to close the pipes, so it now bounds the call's duration.
- Task 14b (2026-10-16): the live rmsnorm probe (L4c) wrote a real kernel by turn 9 and then spent 22 turns on
  `no_kernel: kernel.py defines no OPERATION`, because it named the function `rmsnorm`. The check now names a single
  misnamed `@ttl.operation` function and says to rename it (still `no_kernel`); the skill says the name is literally
  `OPERATION`, gives a skeleton, and says inputs come in the order of the spec's `inputs` keys.
- Task 14c (2026-10-16): the next L4c rmsnorm attempt sliced in elements (tt-lang slices count tiles) and then
  alternated two identical `sed` commands for 9 turns, which TurnRepeat did not report. The skill now states units,
  dtype and the output as general facts, each checked against the harness, the specification or a simulator run.
  TurnRepeat gained a cycle track (period 2 or 3, three full periods), which applies to every agent step.
- Task 14d (2026-10-16): the coder ignored the literal-name rule a third time (`def rmsnorm` six times with the rule and
  a `def OPERATION` skeleton in its prompt, no check run between writes). The harness stopped requiring the name: the
  one `@ttl.operation` function is the operation, and the harness calls the function the gate names. Step 4 now says
  to run the check after each write. The 14b rename message is gone.

## 2026-10-18 (night): the first hardware runs of the kernel probe (L5, version 0.6.4)
Prompt (operator): "go ahead and use the full machine. when you're done, shut down the LLMs and then the machine itself." The tensixinet session
had finished its read-only demo and said the chips were free. Coder-Next ran on board 1 (chips 2 and 3, port 8011); the hardware phase ran on
chip 1 under `gozer run`, as the probe does.
- First launch refused: the latest-skills preflight found tt-lang upstream main one commit ahead of the pin (a CI-only commit; `diagnostics.py`
  unchanged, sha256 the same). Pin moved to `1cd7d41`, `kernel-env` rebuilt, version 0.6.4 (local commit, not pushed).
- `add`, 1 attempt: smoke test passed; simulator pass (7 turns, 45 s); hardware pass (PCC 0.999998, max abs error 0.0156). The kernel is upstream's
  `eltwise_add.py` renamed, so this checks the loop, not the coder.
- `silu_mul`, 3 attempts at T=0.7: 3 of 3 passed the simulator (10, 15 and 13 turns; 250 to 372 s) and 3 of 3 passed on hardware (PCC min 0.999996 each).
  Three distinct kernels. Prediction written before the run (at least 2 of 3 in the simulator, most of those on hardware) held.
- Correction of an earlier statement of mine: "no coder-written kernel has passed the simulator" was wrong. `silu_mul` passed 5 of 5 in L4.
  `rmsnorm` (0 of 16 over L4, L4e, L4f) and `linear_scan` (0 of 8) have not passed.
- Not done: a hardware run of `rmsnorm` or `linear_scan` (neither passes the simulator), repeated hardware runs for variance, any op beyond these four.

## 2026-10-19: PRs 9 and 10 merged locally; the Nemotron-3.5-Bolt-74B bring-up stopped at preflight
Prompt (operator): "Have this directory represent the combo of PRs 9 and 10, as if they were merged to main already", then "Perform an
unattended bring up of nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB. The only thing to 'help with' is HDD space. If it fails, we log the fail and
report back ... Don't make the changes yourself."
- Merged `pr-10` then `pr-9` into local `main` (not pushed). Conflicts were additive: `CLAUDE.md` keeps both sections, `preflight.py` keeps
  the CPU stand-in and SFPI signals beside the skills-freshness signal (my first resolution interleaved two `return Signals(` calls; fixed before
  the commit), the version is 0.6.4. The suite passes: 4422 passed, 2 skipped.
- Bring-up: blocked at preflight by `nearest-model-missing` (no bundle serves the BF16 base) and `coder-unusable` (ollama not running). Disk
  was enough. Nothing started. Write-up and what would be needed: `docs/run-logs/2026-10-19-nemotron-bolt-74b.md`. I did not start ollama,
  because the base block decides the outcome and the instruction was to change nothing else.

## 2026-10-19 (later): pplx-decider-v1.2 stopped at stage 2, and stage 1's pass was false
Prompt (operator): "OK let's do pplx-decider" (unattended, log the failure and report what we would need, change nothing). Ended `blocked`
(`needs-new-model-code`) after stage 0 (class `unknown`, path `full-port`, 20 m 31 s) and stage 1 (11 m 18 s). Write-up and the list of what
would be needed: `docs/run-logs/2026-10-19-pplx-decider-v1.2.md`. I started ollama (the CPU stand-in check blocks without it); nothing else was done by hand.
- Found by reading the evidence: stage 1's `summary.json` says `verdict: fail` (all 851 model keys missing, so the CPU reference ran on
  uninitialised weights and produced noise), the agent wrote `reference.json` with `verdict: pass`, and the gate accepted it. The gate trusts the
  agent's file instead of the evidence the template wrote.
- Stage 0's prefix normalization (`model.language_model.` to `model.`) matches 0 of 1183 names for a checkpoint that uses `language_model.*` and
  `visual.*`; the draft contradicts its own evidence on the prefix; `code_files` misses nested directories.
- The pplx-decider's real difference (a readout head and non-causal attention across a request's questions) is in the card and the code, not in the
  config or the tensor names; no stage can see it today.
## 2026-10-19: operator mode for the web UI (branch `ui-operator-mode`, version 0.6.5)
Prompt (operator): "add a very lightweight theme switcher to the tt-orchard UI. For now, call what we have now 'playful' mode. I want
'operator' mode, less focused on fun and more on what's happening now and what happened before and what to do next." Brainstormed first.
The operator's complaints about the page: what is running does not stand out enough, the font is hard to read, text is hard to copy, it is
hard to tell decoration from information, and numbers come with no context ("if perf is good I want it to tell me"). The operator chose
mockup B (two panes, "now" stays put) and said to build the switcher and layout first and the verdict line after.
- `orchard/web/mode.js` (a plain script in `<head>`, so no flash) sets `<html data-mode>` from `?mode=`, else the stored choice, else
  `playful`; only the two names are accepted and every storage call is in a `try`. Two buttons in the top bar switch it.
- The run view has two wrappers, `.op-left` (state band, Now, Next, block and pause panels) and `.op-right` (stages, results, tabs and
  panes). Outside operator mode they are `display: contents`, and a browser check showed every key element at the same position as on the
  old page. In operator mode the left pane is sticky and the right scrolls; below 900 px they stack.
- The state band takes its colour from `#run[data-state]` and `[data-tone]` (blue and pulsing for running; the pulse sits in a
  `prefers-reduced-motion: no-preference` query). Operator mode drops the pixel face by redefining `--pixel` as the system face, hides
  the scene (and `scene.js` stops drawing), and shows the real state name, stage name and feed role instead of the orchard words and icons.
- Checked in headless Chrome against the live pplx-decider run (light and dark, widths 1440 to 400, no horizontal scroll, storage blocked,
  an invalid `?mode=` ignored, Hardware and Settings views still open). 25 mutations in all; the first run left one survivor (the link
  value was not validated where it is read), which now has a test.
- A test of mine used an element id that did not exist (`run-head` is a class); it failed with a KeyError, not an assertion.
- Not done: the Verdict line (a plain sentence with a yardstick for each number; it needs the pass bars from the code and the numbers from
  earlier runs' ledgers), a "copy" button for long commands, and hiding the emoji inside the mode chip and the "Now" line. One 404 showed
  once in a browser console and did not recur in eight more loads on either UI. The page's own JavaScript still has no automated test;
  the browser checks were run by hand with Playwright in a scratch venv.

## 2026-10-19 (later): the Verdict line (branch `ui-operator-mode`, version 0.6.6)
Prompt (operator): "go ahead with the verdict line" (the second half of the operator-mode request: "if perf is good I want it to tell me instead of
just giving me numbers I don't have context for"). Tests first (`tests/test_verdict.py`, five more in `tests/test_webui.py`), 26 mutations.
- `orchard/verdict.py` turns `run_view.results` into {id, tone, headline, detail} lines, worst first. A swap or configuration result is set
  against `SWAP_TOP1_MIN` and the earlier runs' scores (named by model). A configuration's boot time is ranked (quicker than, about the same as,
  somewhat slower than, slower than) against the median of earlier runs of the same package and chip count that passed, and only when at least
  two exist; stage 2's boot time is stated and never ranked, because its caches are sometimes warm and the result does not say. A run with
  hardware results and no timing says "Not measured: generation speed". The cut-offs (a 15% band, 1.5 times for slow, a margin of 0.05) are choices.
- The reference line reads `stages/1/evidence/summary.json` and `load-report.json` (inside the run, under 2 MiB, no link out) and sets them
  beside `reference.json`. On the real pplx-decider run it shows, first and in red, that the template said fail with 851 of 851 keys missing while the
  agent's file said pass. The gate itself is unchanged: this is a display, and the gate fix is in the pplx-decider run log.
- `WebApp._peers` reads the other runs' results once per ledger change (mtime and size) because the page asks for the detail every few seconds;
  verdict strings pass through `redact`. The page shows the section in operator mode only; it is built with text nodes, never markup.
- Mutation lessons: three survived the first pass (a failed run counted as a yardstick, the minimum used instead of the median, a size cap
  tested with a file that was invalid JSON anyway); each now has a test that goes red.
- Checked in headless Chrome on the real runs: pplx-decider (bad, warn), bytkim (four good, three info, with 0.97 from RAG-Gate and clef and the
  118 s and 218 s boots for 2 chips). Not done: a hardware-free way to say whether tokens per second is good (nothing times generation yet, so the line
  says so), the same verdict in the playful view, and a copy button.

## 2026-10-19 (night): stage 0 matches by what the tensors are, and stops on a restricted head (branch `model-alignment`, version 0.6.7)
Prompt (operator): after the pplx-decider run, "we have proximal models right but the way we're matching isn't helping us yet"; asked what I
would have found if I drove the bring-up, I ran the analysis by hand (names, shapes, values, the readout rows) and then the operator ruled
"stop and report now" and "yes, build it in a worktree that has all our open PRs still merged together" (this branch is local `main`, which has
PRs 9 and 10 merged; the UI branch `ui-operator-mode` is not a PR and is not in it). Design note: `docs/superpowers/specs/2026-10-19-model-alignment-design.md`.
- Found by hand: all 1183 decider tensors have an analogue in Qwen3.8-27B (`language_model.X` and `visual.X` against `model.language_model.X` and
  `model.visual.X`, shapes equal); the base has 16 more (`lm_head`, 15 `mtp.*`); sampled values are near (cosine 0.9999) so it is a fine-tune; the
  255 readout rows are copies of `lm_head` rows for the tokens A, B, ... AA, AB (cosine at least 0.9999).
- Built, tests first (new files `test_delta_triage_alignment.py`, `test_restricted_head.py`, `test_gate_reference_evidence.py`): `align_leftovers`
  (suffix matching plus voted prefix rewrites), `sample_value_diffs` (standard library), `detect_restricted_head` (numpy, says when skipped), nested
  `code_files`, the class `restricted-head` (path full-port; the supervisor pauses before stage 1; unattended it blocks with the code `restricted-head`),
  and a stage 1 gate that compares `reference.json` with the template's `evidence/summary.json` and `load-report.json`.
- Run on the real checkpoints (4 s): class `restricted-head`; 1183 of 1183 aligned (850 and 333 by two prefix rewrites); values 1 identical and 7 near;
  head restricted, 255 rows, tokens A, B, C, ...; 5 code files found. The new gate fails the real pplx-decider stage 1 (three reasons) and still passes
  the bytkim, RAG-Gate and Clef stage 1 directories.
- Found while testing: `code_files` skipped symlinks, and a Hugging Face cache snapshot is all symlinks, so the decider's code was missed until a
  test with links went red. 38 mutations: 31 red first time; the 7 survivors (the "must not match" name guards, one vote as a rule, any one row as a
  head, a full-size matrix as a restriction, an unattended retry passing the stop) were tests that never reached their branch (the tie test's names
  shared too little to tie); each is fixed and red now. Two of my own assertions had `or True` and could not fail; fixed before any run.
- Not done: a reference loader that uses the name map (the gate now fails such a run instead of passing it); the decider's attention mask and marker
  readout (its `model.py` has not been read); a middle path that checks a restricted head against the producer's code on the host. The suite was run in
  full once at the end of this entry.

## 2026-10-19 (night, later): one branch, the loader's real cause, the mask answer and the parity run (branch `make-it-work-make-it-useful`, version 0.8.1)
Prompts (operator): "do the reference-loader and the attention mask. may as well follow that middle path. then, let's take all these PRs and branches
and decide: we're going to go one true branch for all these changes together. 'Make it work and make it useful' branch." Later: "i accidentally ran
tt-smi -r sorry" and "might need to re-run it before retrying and continuing".
- The branch: local `main` (PRs 9 and 10 merged) plus `ui-operator-mode` plus `model-alignment`, merged in that order; conflicts were CLAUDE.md (log
  entries) and the version, now 0.8.x. Nothing else carried commits `main` lacked: `autonomy` and `autonomy-keep` are the same commit, the old combined branch
  from before the split into #9 and #10, and `main` is ahead of it (a later self-review fix, a newer tt-lang pin). The full suite on the merge passed (4514).
  Not pushed; PRs 9 and 10 are not closed.
- The loader, corrected: I first said the template fell back to AutoModelForCausalLM because `transformers.Qwen3_5Model` is not exported. That was wrong
  (it is exported on both interpreters). The stage 1 agent had edited its copy of `reference_gate.py`. Fixes: both gates compare the stage's copy of
  `delta_triage.py` / `reference_gate.py` with the shipped file; the template resolves the class from the architecture (module, then by name), retries a
  load that misses keys once with a key_mapping from stage 0's `alignment.raw_rules`, and fails a model with no lm_head with evidence; every failure path
  writes `evidence/summary.json`. The skills say not to edit the scripts.
- The mask, read and measured (`docs/run-logs/2026-10-19-pplx-decider-parity.md`): only the 16 full-attention layers are bidirectional; a causal runtime
  changes probabilities by up to 0.47 on the CPU reference, about eight times the producer's own noise. Stage 0 now records the model's declared
  `attention_mode` and the code lines that mention masks, hooks and the readout (`evidence/readout-facts.json`).
- The middle path, built and run: `orchard/skills/restricted-head-templates/` (`restricted_head_parity.py`, `prepare_restricted_head.py`) and
  `docs/restricted-head-parity.md`. Host reference written from the protocol text, pinned to the producer's code by sha256, never imported. Two shims
  during the run only: a key_mapping into the TT loader's `Qwen3_5ForCausalLM.from_pretrained` (otherwise this checkpoint loads as random weights with
  a warning) and a wrapper that makes the prefill attention's one causal SDPA call non-causal with a mask hiding the padding. On one board: 14 of 14
  top-1 agree, largest probability difference 0.048, marker hidden-state correlation 0.9975 or better, wiring cosine 0.987, all 16 causal calls turned. A
  repeat after the operator's `tt-smi -r` gave identical numbers.
- Slips: my wrong first explanation of the loader (above); `AutoProcessor` needs torchvision; the readout was first read from the directory that holds
  only the backbone (a 3-minute boot lost); two assertions of mine ended in `or True`-style tails that could not fail and were tightened. 45 mutations over
  the new code: 40 red first time, the five survivors got tests (the sixth anchor was a redundant guard, removed).
- Not done: a `tt-orchard head-check` command; images, long states and 255-option questions; a 4-chip run; the bars are copied from the sidecar check, not
  measured for this head; serving.


## 2026-10-19 (night, last): a classification bench for stage 0 (version 0.8.2)
Prompt (operator): "is there a way to try just this new area of classification logic against like a group of 40 models (without downloading
their weights fully) and comparing the decisions it makes for those 40 against the decisions you, Claude, would make yourself?"
- `orchard/classify_bench.py` (`collect`, `compare`) reads each model's headers and configs through the research fetcher, runs the real stage 0 functions,
  and writes one evidence packet per model with no verdict. Three fresh labelers (opus, sonnet, fable) labeled the packets blind. Corpus and how-to: `docs/classify-bench/`.
- Result on 42 models (2 skipped: the Hub would not serve them): the tool matched the labelers' majority on 39 of 40. Against 6 models with a known outcome the tool and
  each labeler were right on 6 of 6. The 3 differences: Timestral-209M (labelers split unknown vs full-port), Sieve-27B-v2 (labelers split; no config.json, `head.pt` unreadable),
  and sabi-noema (all three labelers say weights-only; the tool says full-port because 96 small `A_log` and norm tensors are F32 where the base has BF16).
- Bugs the bench found in the tool, each fixed test first: a repo with its own `lm_head` could be called a restricted head; a sidecar problem hid a backbone mismatch
  (`decide_class` now checks the backbone first); subdirectory weights were listed as sidecars; the shard, body and request caps made large repos unreadable (the bench has
  its own limits and `headers.read_model` takes `max_shards`); `evidence_complete` counted GGUF and `.pt` sidecars as gaps.
- Quantized copies of the base (GPTQ, FP8) are full-port in the tool and in all three labelers' second round, but all three called it medium confidence: the guide has no class for a copy
  that differs only by quantization. First-round labels for those four packets said unknown or full-port; the second round used packets with the added evidence.
- Open: sabi-noema's dtype rule needs a hardware swap test before it is relaxed (TT casts `A_log` to float32 itself). The labelers saw packets the tool wrote, so a gap in a packet is a gap for both.

## 2026-10-19 (night, later): model card benchmark plan and tooling, and a Copilot finding (version 0.8.3)
Prompt (operator): "for these model cards, don't we want a full benchmark?" Then: "yes, write the benchmark plan and pilot on bytkim", and "don't use the HW yet".
The plan is `docs/superpowers/plans/2026-10-19-model-card-benchmark.md`. The pilot has not run: the operator said not to use the hardware yet, so
everything built so far needs no chip.
- Why: the bytkim card has 32 of 32 tokens against the CPU and decode speed on four prompts whose text differed between base and fine-tune. That is no
  evidence about the fine-tune. The plan compares base and fine-tune on the same bundle and server: GSM8K, HumanEval, IFEval, MBPP (lm-eval) and BFCL
  `simple` and `multiple` (a scorer in this repo), plus a speed sweep over input length and concurrency.
- Built, test first, a mutation table per module: `orchard/toolcall_bench.py` (BFCL scorer, 47 mutations), `orchard/bench_proxy.py` (adds thinking off, records
  tokens, cut-offs and latency per request, 20 mutations), `orchard/bench_session.py` (starts a bundle's run.sh, runs the phases under the stall watcher with a
  scrubbed environment, writes `summary.json`, 41 mutations). `tests/fixtures/bfcl` holds the first lines of the real data; `tests/fixtures/lm_eval` a trimmed real
  results file.
- Found while testing: the scrubbed harness environment hid the fake harness's log path, so the first session test failed for a reason that was the design working. A
  mutation run first crashed on a mutation that hung the tests; the harness now counts a hang as detected. `lm_eval`'s plain `humaneval` cuts chat replies at stop strings
  such as `\ndef`; the plan uses `humaneval_instruct` and `mbpp_instruct`.
- Copilot's review of PR 11 found that `used_caches()` ignored the tensor caches that sidecar-parity and restricted-head checks keep, so `--prune-caches` could delete one
  mid-run. Fixed in `79ddf24`: `caches.CACHE_CONFIGS` lists all three config names. Four tests were red without the change; a mutation table of 5 left no survivors. My first
  prune test could not fail (cache sizes round to 0.0 GB for small files, so nothing was covered); it now uses sparse 2 GB files and older protected caches.
- Not done: the pilot, a speed sweep script, the BFCL download helper, and any card change.

## 2026-10-20 (UTC): the benchmark pilot on bytkim/Qwen3.8-27B-pi, and a broken instrument (version 0.8.4)
Prompt (operator): "yes, write the benchmark plan and pilot on bytkim", then "don't use the HW yet", then "go". Results: `docs/run-logs/2026-10-19-bytkim-pi-benchmark-pilot.md`.
- Eight sessions on one 2-chip board (pi and base, thinking on and off, first pass and code pass), 20 items per task, each under its own `gozer run` lease. At that size the
  fine-tune and the base cannot be told apart on GSM8K, IFEval, BFCL, HumanEval or MBPP. Thinking off cut a session to a third of the time without a clear loss.
- The first pass scored HumanEval 0.0 for both models. Cause (read from the stored samples, 12 of 20 replies empty): lm-eval's `humaneval_instruct` expects a prefilled
  assistant turn and stops at `\ndef`; `mbpp_instruct` relies on the same prefill. My plan had named `humaneval_instruct` as the fix for exactly this problem, and it was not.
  `orchard/lm_eval_tasks/` has `humaneval_chat` and `mbpp_chat`, with 41 tests and an 18-mutation table. A control server that answers with the datasets' own solutions
  scores 1.0 and 1.0 (whole function and body-only replies) and a stub scores 0.0, on 40 items each.
- The first control run scored 0 for every mode, because my control server left `"index": 0` out of each choice; lm-eval turns that into an empty reply with no error. A
  score of 0 for every mode was the sign that the instrument, not the model, was wrong.
- While the pilot ran I developed in a second worktree (`bench-code-tasks`), because the driver imports `orchard.bench_session` for each session and an edit mid-run would have
  changed later sessions.
- Not done: the speed sweep, drafter acceptance from `server.log` (the packaging log said none is printed; each request logs `accept x/7`), a full-size run, and the card.

## 2026-10-20 (UTC): full-size benchmark of bytkim/Qwen3.8-27B-pi, and the card is public (version 0.8.5)
Prompt (operator): "push benchmark wor. run at full size with thinking. publish the card after", then "publish the model with tt-model-manager as public HF model",
then "tell the tensixinet agent it can continue (after you take down the running model)". Results: `docs/run-logs/2026-10-20-bytkim-pi-benchmark-full.md`.
- Every item of HumanEval, MBPP, GSM8K, IFEval and BFCL simple and multiple, for the fine-tune and then the base, 9.8 hours of chip time in all. The fine-tune is ahead on
  every set; the gap is clear on GSM8K (0.839 against 0.792) and MBPP (0.896 against 0.864) and within noise elsewhere. The 20-item pilot could not show this.
- The 8192-token limit with thinking on costs both models: 44 of 52 MBPP failures for the fine-tune and 53 of 68 for the base are empty replies. The card says so.
- Drafter acceptance now comes from the server log per phase (commit `1bfdaa8`): 0.46 to 0.60 of drafted tokens.
- `episod/qwen3.8-27b-pi-p300` is public and listed in the community catalog (delist with `tt-model unpublish`). The package gate passed after the card rewrite.
- tensixinet was told the chips were free after the server was down and `gozer status` showed all four chips FREE.
- Cleared 70 GB at the operator's request (three unused Hugging Face models, the audio8-asr tensor cache, four old animatediff image tags).
- Not done: the speed sweep, MMLU-Pro, long context, thinking-off and higher-limit runs, per-item BFCL results, `tt-model pull` and a boot from the Hub.

## 2026-10-20 (UTC): fixes for ezietlowTT's review of PR 11 (version 0.8.6)
Prompt (operator): "Respond to PR 11 comments and requests", then "push the fixes and reply to the review; review PR 12 against this env too". The review had 15
findings and some smaller items. Four agents each took a group in its own worktree (`pr11-a` to `pr11-d`), test first, one commit per finding, a mutation per guard;
I merged the four branches. The suite on the merged branch: 4930 passed, 2 skipped. Every finding was real; two were fixed differently from the suggestion.
- Lab mode: the `cpu` row is skipped (#1); the `sfpi` row reads the lab's version over ssh and is left out with no lab probe (#2). A run directory is not set aside while a
  supervisor holds its ledger lock, and `local_snapshot` accepts `.safetensors` at any depth and top-level `pytorch_model*.bin` (#4). A `.bin`-only model now passes preflight
  and may fail at stage 2, which reads top-level safetensors only.
- Gates: stages 0 and 1 need the template's copy and the template's own sha256 stamp (#3; a hand-forged stamp still passes, so this is best effort). Prefix-aligned checkpoints
  get a stage 0 hazard that says stages 2 and 4 do not rename weights (#10; a path reason would have undone the alignment feature). The reference loader frees the first model
  before a retry and keeps the first load report (#11). The restricted-head reference digest includes the revision and readout hash and loads with `weights_only=True` (#15).
- Credentials and leases: kernel probe children get a clean environment and are swept afterwards (#5). HumanEval and MBPP programs run in a Landlock and seccomp sandbox that
  refuses file reads outside Python's prefixes and every socket; user namespaces and bwrap are blocked on this box. The session refuses code phases when the sandbox cannot
  start unless `--allow-unsandboxed-code` is passed (#6). A block releases the test lease, and a restart gives back a local one (#7). `refresh-ttnn` stops the whole gozer
  group on a timeout (#8). A queued gozer request is reported as queued and its ticket cancelled (#9). A phase with any failed request is no longer ok.
- UI, verdict and speed: one malformed result file no longer breaks other runs' pages (#12); NaN is not a number (#13); a cut-off response is an error sample (#14).
- Check of the published card: the 1,328 stored HumanEval and MBPP programs (both models) were scored again by the new sandbox and gave the same verdict on every one, so the
  card's numbers stand. Group C wrote #6's code before its tests; its 38 mutations were run afterwards.
- Not done: `orchard/canary.py` `ask` has the same uncaught `IncompleteRead` as #14; the skills do not yet say the template copy is required.
## 2026-10-20 (UTC): the research catalog row falls back to `tt-model` (branch `ezietlow/research-catalog-fallback`, onto `make-it-work-make-it-useful`)
Prompt (operator, reviewing PR 11 on node6): "Lets get the dependecies onto node6 and try again", then "Lets fix the gap and contribute it back to the PR".
- With the test dependencies in place on node6 (gozer via `ORCHARD_TEST_GOZER`, tt-lang via `tt-orchard kernel-env`, `ORCHARD_LIVE_RESEARCH=1`) the suite was 4785 passed,
  1 failed, 3 skipped. The failure was `test_research_live.py`: `bundle_choices` empty, because the catalog row ran only `tt model search --catalog`, and none of node4, node5
  or node6 has a `tt` command (tt-model-manager ships `tt-model` and `tt-kernel`). `nearest.search` already fell back to `tt-model`; `research/ports.catalog` did not.
- `catalog()` now asks `tt` first and then `tt-model search <family> --catalog --json` when `tt` is missing or gives no answer; `nearest.parse_search` already reads both
  outputs. The unavailable row names every command that was tried, or says neither is installed. The plain-term check still runs before either command sees the family.
- Test first: 3 of 4 new tests red before the change (the fourth pins that `tt` answering means `tt-model` is not asked). Mutations, each red: no fallback, stop after the
  first attempt, `tt-model` asked first, the old reason text, `--catalog` dropped. The live humanizer test passes on node6 with no `tt` and no shim.
- Suite on node6 with the same dependencies: 4790 passed, 3 skipped (the two tests that need the skill installed as `ttm-tt-lang` by `tt-warp`, and the qwen transcript replay).

## 2026-10-21 (UTC): the cut-off items asked again at 32768 tokens (version 0.8.7)
Prompt (operator): "rerun the cut-off items with a higher token cap", then "let it finish, then update the card" and "if it ends up going really badly, let me know sooner". Results:
`docs/run-logs/2026-10-21-bytkim-pi-benchmark-rerun.md`.
- `bench_session --items FILE` runs chosen documents of a phase (the harness's `--samples`); `orchard/bench_rerun.py` lists the empty-reply items of a finished run and merges a re-run
  into it. Tests first; 13 mutations on the session change (2 survivors, one equivalent, one got a test) and 17 on the new module, all red.
- 81 items for the fine-tune and 98 for the base, 3.0 and 3.6 hours of chip time. No request failed and no item got worse. At the higher limit the models are close on HumanEval, MBPP
  and IFEval (paired p 0.69, 0.38, 0.11); the MBPP gap that was clear at 8192 tokens (p 0.011) is not. GSM8K is still the one clear difference.
- The card was updated (scores, a section on the re-run, intended use, limits) and passes `gate_package`. On `main` the gate rejected the package's `run.sh`, because the format-fix
  commit `e61df0f` was not merged; PR 13 carries it as a cherry-pick (with `orchard/measure_serving.py` and the 10-19 packaging log). The card was uploaded to
  `episod/qwen3.8-27b-pi-p300` (commit `84d3c49`) and published again on 2026-10-21.
- Slip: I ended one command with `pkill -f` and `pgrep -f`, which killed my own shell. It matched nothing else. Wait on a pid instead.

## 2026-10-21 (UTC): fixes for ezietlowTT's approval notes on PR 13 (version 0.8.8)
Prompt (operator): "fix all the notes before we merge". The review approved and left nine non-blocking notes; each was checked and fixed.
- `bench_rerun.merge` refuses: two rows for one item (lm-eval writes one per filter, so GSM8K would have kept one), a changed `doc_hash` or `prompt_hash`, a re-run that is not exactly
  the requested items (new `--items FILE`), and a sample with no value for a metric (it counted as 0). Re-running the six merges on the real run directories gives byte-identical
  output to the committed `merge-*.json` and passes every new check.
- `weights_wiring_problems` names the nearest model by whole id (`_names_model`): `org/model-v2` and `other-org/model` no longer match; `org/model.` and `${HF_MODEL:-org/model}` still do.
- 17 mutations over the two changes, all red (the first mutation run had no survivors). Tests first: 3 of the 4 package tests and 6 of the merge tests failed on the old code before the change.
- GSM8K's paired sign-test p was stored as `0.0` (the compile scripts rounded to 4 places). The scripts now keep 4 significant digits and the four `benchmark.json` copies hold
  1.607e-07 (the exact value is 1.6066e-07). The card says "below 0.001" and is unchanged. The copy on the Hub still has 0.0 until the file is uploaded again.
- README lists PyYAML as needed by `tests/test_lm_eval_code_tasks.py`. The commit bodies are empty; the mutation tables are in this log and the run log.

## 2026-10-21 (UTC): research finds tt-transformers and says what a tt-metal demo is (branch `base-code-research`, version 0.8.9)
Prompt (operator): the preflight blocked `Plurato123/OctLLM` with `nearest-model-missing`, and the operator expected tt-metal or tt-transformers to
already have the config. "Read qwen25_vl and see if it can be served, but also do A and B."
- Found first: research did list `models/demos/qwen25_vl` as existing work. It had no effect on the outcome, because stages 2 and 4 swap weights into an
  installed v6 thin bundle and a demo directory is code, not a bundle. Nothing in the repo read tt-transformers at all.
- `qwen25_vl` (read upstream, because the local tt-metal checkout is stale): Qwen2.5-VL-7B is supported, with a vLLM generator file
  (`tt/generator_vllm.py`), but the README, `PERF.md` and `model_config.py` name only N150, N300, T3K and TG, and the code gates on `is_wormhole_b0()`.
  Blackhole is not mentioned. That is not proof it fails on this box. Not run.
- A: `ports.tt_transformers` reads tt-metal's `models/tt_transformers/model_params` (named by Hugging Face model name, so a match is exact), its README
  table (which gives the boards each model runs on) and the standalone repo's `src/tt_transformers/models`. The report passes the stated bases and the
  model itself as `ids`. It runs under `--no-github`, because it uses the contents API like the tt-metal listing.
- B: `ports.demo_facts` adds, for the first two demos that matched by name, whether `tt/` has `generator_vllm.py` and which boards the README names (only
  board names reach the claim). `preflight._code_note` puts that first in the `nearest-model-missing` message ("code for it exists, but code is not a
  bundle"), warns when the code names no Blackhole board, and caps the searched bundle list at 8. The reason code did not change. The phrase the writer and
  the reader share lives in `research/types.py` (`LACKS_BLACKHOLE`), and a test feeds a real `demo_facts` claim into the preflight.
- Tests first, a mutation table per commit: 19, 23 and 28 mutations. Survivors, each given a test: an id that is not `org/name`, case in board names, p300,
  the count of demos read, the exact file name, the claim length limit, and the phrase being spelled in two modules. Two of my own tests were wrong first
  (a sort order I assumed, an import). The suite passed: 5054 passed, 2 skipped.
- Not done: nothing runs `qwen25_vl` on hardware; no bundle is built from demo code; `tt-transformers` is read only by name and README, not by its code.
- OctLLM itself is not a fit (checked the card, config and weight index): twelve layers carry extra `mesh_*` attention and MLP tensors, there is a separate
  `completion/model.safetensors`, and the output goes through CUDA decoders. Its official environment is conda with an NVIDIA GPU. Stage 0 would call it
  full-port or unknown.
- Candidate research for text-to-3D models: no installed v6 bundle serves the base of any model found (BlenderLLM on Qwen2.5-7B, LLaMA-Mesh on Llama-3.1-8B,
  BrickGPT as a LoRA on Llama-3.2-1B). BAAI/AREX-2 passes the preflight as a Qwen3.8-27B fine-tune; the operator is running it.
## 2026-10-21 (UTC): a progress bar for the weight download, and lines while the preflight runs (branch `bringup-progress`, version 0.8.10)
Prompt (operator): "tt-orchard should show a progress bar while downloading weights. it also waits a very long time before showing more info
when first starting up." Both were measured before anything was changed.
- Startup: `run_preflight` returned every check at once and the page printed after that. Timing each signal for BAAI/AREX-2: 84 s in all, 81 s of it
  research (65 s on a second run), which is three sequential reads of weight headers (14, 21 and 19 s; every shard's header, one request at a time).
  LLaMA-Mesh took 7.5 s. Now `run_preflight(progress=)` and `research.report.build(progress=)` print a line before each slow step, so the first line after
  `mode:` comes at 0.1 s, and the header lines name the model and its number of shards. `--quiet` turns them off. A callback that raises never reaches
  research or the preflight (`research.types.reporter` also cleans the text).
- Download: `hf download --quiet` ran with its output captured, so a 55 GB fetch printed nothing. Measured with huggingface_hub 1.33.0: without a terminal `hf`
  prints no progress at all; with one, and with `--format human`, it prints only "Fetching N files", never bytes. The bytes in the cache are the signal:
  `orchard/dlprogress.py` reads growth of `<hf_home>/hub/models--org--name/blobs` since the start and the files complete in the snapshot, in a thread beside the
  unchanged fetcher. With Xet a blob grows in bursts (205 MB for 24 s, then 1.5 GB), so the bar can stall and jump; the line says "waiting for data" and keeps
  an elapsed clock. A live run of a real 1.5 GB download under a pty redrew in place and ended with "downloaded 1.5 GB in 36s (42 MB/s average)". It showed
  "ETA 14m 48s" in the first second, so the ETA now waits ten seconds.
- Not terminal output: a pipe, a log, or `--style plain` gets a line at each 10% and at least once a minute.
- Tests first, a mutation table per commit: 47 + 5 (meter), 10 (command wiring), 36 (startup). Survivors got tests; two are equivalent (a write to a stream that
  already failed; `reporter(None)` that would raise a `TypeError` the `except` catches). Two of my own assertions were wrong first (`or`-tail that could not fail;
  1000 bytes is "1 KB").
- Review fixes (Copilot, PR 33): a blob that `hf` moves to the shared store leaves a link in the repo's `blobs/`, and the meter ignored links, so a finished shard dropped out of
  the count; it now counts the file a link reaches, once per inode. The closing summary used the planned size, which overstated a resumed download; it now takes a last
  measurement and reports the bytes that arrived (or "downloaded nothing new"). 6 and 7 mutations, all red.
- Not done: reading the shard headers in parallel would cut the 65 to 81 s itself (the fetcher keeps a request budget and a per-host state, so it needs care);
  the supervisor's coder boot (about 10 minutes cold) shows only what the narrator already shows; `hf download` was not replaced by our own downloader.

