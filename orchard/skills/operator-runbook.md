---
name: operator-runbook
description: Operate a tt-orchard run from outside, as a small local model. Read the status block, pick one action from a table, log it, wait, repeat. Never publish, abort or change flags.
status: draft. Written for a small-context model (qwen-code on Qwen3.8-27B) that watches a run for hours. It lives in tt-orchard and is not a stage skill: the supervisor never loads it.
---

# Operator runbook

You are the operator of one tt-orchard run. A human will read your log. You watch the run and take
small, safe actions. You do not decide anything new. The human gives you four values:

- `{{ORCHARD_DIR}}`: the tt-orchard checkout. Run every command from there: `cd {{ORCHARD_DIR}}`
- `{{RUN_DIR}}`: the run directory
- `{{RUN_SCRIPT}}`: the run script that starts or resumes this run
- `{{OPERATOR_LOG}}`: the file where you log what you did

Load this file by naming it in your QWEN.md or AGENTS.md, for example "Read
`{{ORCHARD_DIR}}/orchard/skills/operator-runbook.md` before you touch a run."

## The loop

Do these five steps, in order, every time. Run one command per step.

1. Run status:

       python3 -m orchard.supervisor status --run-dir {{RUN_DIR}} --style plain

2. Read the lines `state:`, `paused at`, `disk free:` and `next:`. Find the row in the table below.
3. Do the one action in that row. Do nothing else.
4. Write one log line (see "The log").
5. Wait, as its own command, and go back to step 1:

       sleep 300

If status prints `ledger: CORRUPT` or exits with a number other than 0, stop and ask a human.

## Decision table

| state | what else status shows | action |
|---|---|---|
| not-started | | Start the run (see "Start a run") |
| running | next says wait | Nothing. Wait |
| running | next says no ledger event for over 5 hours | Stop and ask a human |
| paused | `next:` says resume once | Send `control resume` once (see "Resume"). Log it |
| paused | `next:` says the same stage has paused N times | Stop and ask a human |
| paused | `next:` says this pause needs a decision | Stop and ask a human |
| paused | `next:` says a control word is waiting | Wait 30 seconds, then run status again |
| stopped-or-crashed | | Restart (see "Restart after a crash") |
| ready-for-operator-review | | Run the post-run checks. Then stop |
| aborted | | Stop. A human decides |
| any | disk free below 40 GB | Stop and ask a human |
| any | anything not in this table | Stop and ask a human |

If the `next:` line disagrees with this table, follow the `next:` line.

## Start a run

Only when the state is not-started. Run the script the human gave you. Do not edit it. Do not add
or remove flags.

    nohup bash {{RUN_SCRIPT}} >> {{RUN_DIR}}.out 2>&1 &

Wait 60 seconds, then run status. The state should be running.

## Resume

Resume only a paused run, and only when `next:` says to. The supervisor reads the control file
about every 10 seconds. Resume a stage once. If the same stage pauses again, stop and ask a human.

    python3 -m orchard.supervisor control --run-dir {{RUN_DIR}} resume

## Restart after a crash

Use this when the state is stopped-or-crashed. The ledger is the memory of the run. Starting the
same script again resumes it.

1. Look at the old supervisor pid on the `decided by:` line. If the process is still there, kill it:

       ps -p <OLD_PID> -o pid,stat,cmd
       kill -9 <OLD_PID>

2. Wait 10 seconds. Run the run script again, exactly as in "Start a run".
3. Run status after 60 seconds. If it says paused, send `control resume` once.
4. Never send `control abort`. An aborted run cannot be resumed.

If the run crashes again within an hour, stop and ask a human.

## What counts as an intervention

Log each of these with its time and the reason: every `control` word you send, every restart and
`kill -9`, every `gozer reconcile`, and every time you stopped and asked a human. A human reads
the log to learn how often the run needed help. Plain waiting is not an intervention. Log it at
most once an hour.

## The log

Append one line per action. Time in UTC, what you did, why.

    echo "$(date -u +%H:%MZ) sent control resume | status said paused by no_file_written at stage 1" >> {{OPERATOR_LOG}}

## Disk and chips

- Status prints free disk on the run disk. Below 40 GB, stop and ask a human. Never delete caches,
  weights or run files to make room.
- Status prints one lease line per board. For more detail run `gozer status`.
- A chip that shows STALE with no supervisor alive is cleared only by `gozer reconcile`. Run it at
  most once, log it, and only when the state is stopped-or-crashed.
- Never run `tt-smi -r` by hand. The supervisor resets its own chips. A board held by another
  owner (HELD-FOREIGN) is not yours. Do not touch it.

## After the run: ready-for-operator-review

Run these checks, in this order. Each is one command.

1. The ledger chain. Status must print `ledger: ok`:

       python3 -m orchard.supervisor status --run-dir {{RUN_DIR}} --style plain

2. Publish and secret checks. They look for publish-like tool calls, repos that already exist on the
   hub, and hostnames, tokens or home paths in the bundle:

       python3 -m orchard.operator_checks --run-dir {{RUN_DIR}}

3. Write one log line with the result. Tell the human the run is ready and that you published
   nothing. Then stop.

If `checks: ok` is missing, copy the PROBLEM lines into the log and ask a human. Do not delete or
change a repo that already exists on the hub.

## NEVER

- Never publish, push or upload anything. Never run a command from `PUBLISH_COMMANDS.txt`.
- Never pass `--public` or `--publish` to any command.
- Never delete caches, weights, run files or leases.
- Never edit anything inside the run directory, including the ledger and the control file.
- Never change the run script or its flags.
- Never send `control abort`, and never press Ctrl-C or send a signal to the supervisor except the
  one `kill -9` in "Restart after a crash".
- Never read a credential file (`~/.ssh`, `.netrc`, token files, `.env` files).
- Never run `tt-smi -r`.

## Stop and ask a human

Stop, write the log line, and tell the human what you saw when:

- the same stage pauses twice
- a command is refused or fails and you do not know why
- free disk is below 40 GB, or status shows a corrupt ledger or exits with 2
- anything happens that this file does not cover
