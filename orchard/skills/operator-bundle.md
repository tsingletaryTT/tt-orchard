---
name: operator-bundle
description: Stage 8 of a tt-orchard run. Build the bundle an operator reviews before anything is published - results, open risks, a model card, the packages and the exact publish commands as text - and never run those commands.
status: draft. A local tt-orchard copy. Since 2026-10-04 a tested template script builds the whole bundle from the run's files; you configure it, run it and review what it wrote. It lives in tt-orchard.
---

# Operator bundle

## Goal

The run ends at "ready for operator review". The operator decides whether anything is published,
whether it is public, and whether it enters any catalog. The bundle is `bundle/` in your stage
directory (`stages/8/bundle/`).

A tested script builds all of it from the run's own files. You give it one config file, run it,
read the two reports it wrote, and fix only text that is wrong or unclear. You do not write the
bundle by hand, and you do not write a script.

## How to work

- Write `bundle_config.json` FIRST. It needs one fact you already have: the run directory.
- Do not investigate anything outside the bundle's own files. The script has read every stage's
  result files for you.
- Never read the run's ledger (`ledger.jsonl`), the step logs under `stages/*/log/`, or any
  transcripts. The script already counts the ledger's escalations, retries, pauses and operator
  commands. An earlier stage 8 agent read stage 7's files with 33 `cat` commands, wrote nothing
  for 20 turns, and was stopped.
- Do not download anything, and do not run any publish command.

## Steps

1. Write `stages/8/bundle_config.json` (an absolute path):

       {"run_dir": "<the run directory>"}

2. Copy the template into your stage directory:

       cp {{ORCHARD_DIR}}/orchard/skills/operator-bundle-templates/build_bundle.py stages/8/

3. Run `python3 stages/8/build_bundle.py` from the run directory. It prints a line for each file
   it writes, each file it left out of `package/`, and the scrub's findings. Exit 2 means a
   required input is missing; the message names the file. Do not create that file yourself. Put
   the message in your reply and stop. The gate then fails, and the operator reads your reply.
4. Read `bundle/RESULTS.md` and `bundle/RISKS.md` once. Do not open the other bundle files.
5. Edit only text that is wrong or unclear, with write_file (the whole file):
   - Replace the summary paragraph under `## Summary` at the top of `RESULTS.md` with a plain
     summary of three to six short sentences: what model the run brought up, on which path,
     what passed, what was skipped, what stage 7 staged, and what the operator must check first.
     Use only facts that are in `RESULTS.md` or `RISKS.md`.
   - Add a risk to `RISKS.md` only when you can justify it from a file the bundle names. Give
     each one an evidence path relative to the run directory.
   - Keep every number, label and evidence path the script wrote.
   - Do not change a `Dealt with: Not shown` line to `yes`. The script computes the tensor-cache
     line from the cache directories themselves. `Not shown` means the run's files cannot show
     it, and the operator needs to know that.
6. Reply with a short summary and no tool call.

## What the script writes

- `RESULTS.md`: each stage 0 to 8 with its result. Each number carries its label (`measured` or
  `TODO`) and its evidence paths, relative to the run directory. Skipped stages show the ledger's
  reason. Stage 4 is a table of chip configurations. Stage 7 names each package, its chip count,
  its license and whether its boot check passed. When stage 7 was skipped, it says so: no package
  exists. It also says that `ledger.jsonl` is the operator's copy of the run record, holds
  absolute paths and is not for publishing.
- `RISKS.md`: every `TODO` number (a profile whose boot check did not run, and everything the
  package cards list under Not measured), each stage 0 hazard with a `Dealt with:` line, the
  numbers that rest on thin evidence, and the license. When the license is non-commercial, it
  says so and says whether every card says so.
- `card.md`: the model-level card, with each package card copied in as it is.
- `PUBLISH_COMMANDS.txt`: the exact commands the operator would run, as text, private by default.
  You never run them. The supervisor refuses publish, push and upload commands. The script copies
  stage 7's file into `bundle/PUBLISH_COMMANDS.txt` and `bundle/package/PUBLISH_COMMANDS.txt`
  word for word, comments included. When stage 7 was skipped, the file holds only comments that
  say there is nothing to publish.
- `package/`: stage 7's package folders without venvs, caches, weights or wheels the package does
  not ship. A file larger than 1 GB is left out and listed in `RESULTS.md`.
- `ledger.jsonl`: a copy of the run ledger.
- `stages/8/evidence/bundle-build.json`: every file in the bundle with its sha256.

## The scrub check

After you finish, the supervisor copies the run ledger into `bundle/ledger.jsonl` again and
searches every other file in `bundle/` for the machine's hostname, tokens and absolute home
paths. A hit blocks the bundle. The script runs the same scrub and prints its findings. The text
it copies from result files already has machine paths replaced by labels such as
`<CACHE_ROOT>`. A finding in a copied package file comes from stage 7. Do not edit the package
copy to hide it. Name the file and the finding in your reply, so the operator can fix it at its
source. When you add text, write paths relative to the run directory.
