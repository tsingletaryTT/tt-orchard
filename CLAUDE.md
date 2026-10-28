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

## Standing rules (short; the full history is in `docs/project-log.md`)
- Goal (operator): make tt-orchard bring models up, and bring them up well. Models that need new code are welcome: attempt them
  creatively. Work that is known and not ready goes in a public GitHub issue (no machine paths, host names or tokens).
- Test first, and for each guard run a mutation that removes it and see a test go red. Check that the baseline suite is green before
  reading a mutation table.
- Plain English in anything a person reads: state the finding, no "X, not Y" framing, no aphoristic closers, no metaphors. Sweep for
  `, not `, `rather than`, `instead of`, `reads as`, `honest`, `the one`, `which is why`, `This is what`.
- Branch is `main`. Use a worktree for parallel work. Versions (2026-10-28): never edit a version field. Each branch adds `changes/<slug>.toml` with `python3 scripts/version.py new <slug>` (its own target, summary and PR number). The highest target in `changes/` is the version main reports; `RELEASE` in `orchard/__init__.py` is the last release. Cut a release with `scripts/version.py release` (see `docs/releases.md`). Run `scripts/version.py check` before you push.
  Never merge without the operator's instruction. Never publish to Hugging Face without an explicit go-ahead.
- Hardware: take a lease through gozer (`gozer run`, or one `gozer acquire` and `gozer wait`, never a loop of `gozer run`). Never run
  `tt-smi -r` by hand. Do not touch ports 8000 or 8100 to 8103. Importing ttnn or ttml opens the device.
- Never `pkill -f` or `pgrep -f`; kill by exact pid.
- Credentials: `HF_TOKEN` is in the environment and is never printed. The public repo has no home paths, host names or tokens.
- A Copilot review is requested with the GraphQL `requestReviews` mutation and the bot id `BOT_kgDOCnlnWA`.

## Log
The dated log is in `docs/project-log.md`. Add each new entry at the end of that file, and keep this file under 30 KB.
