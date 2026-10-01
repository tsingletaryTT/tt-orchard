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

## Layout
Spec: `docs/superpowers/specs/`. Plans: `docs/superpowers/plans/`. Code: `orchard/`. Tests: `tests/`.

## Log
- 2026-10-01: spec approved; plan 1 (ledger, runner, tiers, sizing) written.
