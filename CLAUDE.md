# tt-orchard project log

## Original prompt (2026-10-01)
"What would it take to have a suite of skills that can handle moving from 4 chips back down to 2
chips or even 1 chip on the Quietbox 2? ... hand off to a CPU-based model ... 'hold my beer while I
test this new model'." Then widened: take a new model release (for example Qwen3.7) and bring it up
using the skills, with open-source models only. The run ends at an operator review bundle.

## Key decisions
- Supervisor is a state machine in code. Each stage starts a fresh short model context from the ledger.
- Lease handling goes through an adapter. tt-gozer owns leases and gains `yield` and `redeem`.
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

## Layout
Spec: `docs/superpowers/specs/`. Plans: `docs/superpowers/plans/`. Code: `orchard/`. Tests: `tests/`.

## Status
Plan 1 is implemented through Task 4 (ledger, runner, tiers, sizing). Task 5 (run the sizing tool
against a real ollama, which downloads models and loads the host) was not run. It needs the
operator. Plans 2 to 4 are not started.

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
