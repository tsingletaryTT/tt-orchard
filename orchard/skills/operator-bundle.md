---
name: operator-bundle
description: Stage 8 of a tt-orchard run. Write the bundle an operator reviews before anything is published - results, open risks and the exact publish commands as text - and never run those commands.
status: draft. A local tt-orchard copy. Its home is the tt-model-bringup plugin in tenstorrent/skills (spec section 11). Move it there after a real run has used it.
---

# Operator bundle

## Goal

The run ends at "ready for operator review". The operator decides whether anything is published,
whether it is public, and whether it enters any catalog. Write these files in `bundle/` in your
stage directory:

- `RESULTS.md`: what each stage found, from the stage result files in your context. Every number
  appears with its label (`measured` or `TODO`) and the evidence path behind it. Say that stage 7
  (package and container build) was skipped, so no package exists yet.
- `RISKS.md`: open risks. Include every `TODO` number, every stage 0 hazard and whether a later
  stage dealt with it, anything a gate passed on thin evidence, and what was never tested.
- `PUBLISH_COMMANDS.txt`: the exact commands the operator would run to package and publish, as
  text, one per line, private by default. You never run them. The supervisor refuses publish,
  push and upload commands.
- `card.md` (optional): a draft model card built only from measured numbers.

## The scrub check

After you finish, the supervisor copies the run ledger into `bundle/ledger.jsonl` and searches
every other file in `bundle/` for the machine's hostname, tokens and absolute home paths. A hit
blocks the bundle. Write paths relative to the run directory, and edit command output before you
paste it. Say in `RESULTS.md` that `ledger.jsonl` is the operator's copy of the run record, holds
absolute paths, and is not for publishing.
