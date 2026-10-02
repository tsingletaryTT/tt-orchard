# tt-orchard: design

Status: draft for operator review. Date: 2026-10-01. Author: Claude, with Taylor Singletary.
Plan 1 (the core library: ledger, command runner, tier config, sizing tool) is implemented through
Task 4. Task 5 of plan 1 (running the sizing tool on this machine) and plans 2 to 4 are not
implemented. Every number below is either cited to a
measurement made on this machine or marked **unmeasured**.

## 1. Purpose

tt-orchard runs the bring-up of a new model on a Tenstorrent Quietbox 2 using only open-source
models served from the same machine. A run starts from a Hugging Face model id (for example a
new Qwen release) and ends with a private package, evaluated, with an operator review bundle.
Shrinking a model from 4 chips to 2 or 1 chips is one stage of that run.

Success criteria for version 1:

1. A run reaches the operator bundle for a model in a family the stage skills already cover,
   with no Claude session in the loop.
2. Every claim in the bundle is backed by an evidence file listed in the ledger, and every number
   is labeled `measured` or `TODO`.
3. The run survives a supervisor crash at any ledger event and reaches the same final state.
4. A model loop (repeated calls, runaway thinking, no progress) is detected and handled by the
   response ladder in section 7. Foreign agents are only reported.
5. The run never publishes. It never calls `tt-model push`, `gh repo create`, or `git push`.

## 2. Boundary

The run ends at "ready for operator review". The operator decides whether the package goes to
Hugging Face, whether it is public, and whether it enters any catalog. The supervisor enforces this
in the command runner (section 10), so an ignored prompt cannot bypass it.

Out of scope for version 1: new-architecture kernel work beyond what the existing bring-up skills
cover, multi-box scheduling, and acting on agents the supervisor did not launch.

## 3. Evidence this design rests on

| Fact | Source | Used for |
|---|---|---|
| qwencode retried an identical 653 s call five times against a greedy-only server and looped | review of the failed Audio8 attempt, this machine | watchdog signals, retry rule |
| The same attempt built a CPU reference that decoded the wrong way and measured against it | same review | reference gate (stage 1) |
| Without prefix caching, a 130K-token context costs about 42 s of prefill per turn and 204K costs about 78 s | measured, qwencode on the 2-chip Qwen config | fresh short contexts per stage |
| `tt-model stop` clean shutdown 1.6-3.9 s; gozer release plus reset 20-40 s; 2-chip warm restart 2-3 min; 2-chip cold first boot about 30 min | measured, this machine | park/restore timing and budgets |
| A start within about 20 s of a release was queued (exit 10) | observed, gozer | yield requirement |
| gozer leases are board-granular; one board is two chips | gozer behavior, this machine | free-board check before parking |
| Container image builds take 1.5-2.5 h cold | measured, Audio8 v5.1 build | stage 7 runs with no model loaded |
| The first p150 repo exposed a hostname in an old revision | publishing history | scrub check in the bundle stage |
| gozer: every command has `--json`; the queue is FIFO with a 90 s claim window; tickets expire after 1 h; `acquire` is non-blocking | read from `~/code/tt-gozer/gozer/cli.py` and `queue.py` | gozer changes (section 8) |

Not measured, and required before the matching feature is trusted (section 12): CPU stand-in load
time and decode speed for any candidate model; free host memory while the large model is resident;
whether the canary answer is identical across a restart.

## 4. Architecture

Four layers. Each has one reason to change.

| Layer | Owns | Home |
|---|---|---|
| Lease mechanism | Board-granular leases, FIFO queue, release and reset, history, and a new `yield` | `tt-gozer` |
| Lease-aware skills | Park-and-restore guidance: when to yield, how to redeem, what to do on exit 10 or a stale ticket | `tt-gozer/skills/` |
| Stage skills | Delta triage, reference gate, the existing bring-up stages, operator bundle. They say "run hardware work under whatever lease the machine provides" and never name gozer | `tenstorrent/skills`, `tt-model-bringup` plugin (new skills added there) |
| Policy and loop | Supervisor: stage state machine, tier config, ledger, watchdog, command runner with denials | `tt-orchard` (this repo, local until the operator creates a remote) |

`tt-orchard` talks to the lease mechanism through an adapter with three calls: `acquire`, `release`,
`yield`. The gozer adapter ships in `tt-orchard`. A default adapter for machines without a lease
tool assumes a single tenant and refuses to start if another process holds a device node.

Repo layout (planned):

```
tt-orchard/
  CLAUDE.md                 project log: prompt, key decisions, notable moments
  README.md
  orchard/                  supervisor package (stdlib plus the model-server client)
    runner.py               command runner and denial list
    ledger.py               append-only ledger, replay, torn-line recovery
    stages.py               stage table and exit gates
    tiers.py                tier config loader
    handoff.py              park and restore
    watchdog.py             loop detection and response ladder
    adapters/gozer.py, adapters/single_tenant.py
  config/tiers.example.toml
  tests/
  docs/superpowers/specs/, docs/superpowers/plans/
```

## 5. Run structure

A run is a sequence of stages. Each stage has an owner skill, a model tier, and an exit gate. The
supervisor starts a stage with a fresh context holding only the stage skill, the relevant ledger
excerpt and the latest evidence files. A stage ends as `pass`, `fail` or `escalate`. `escalate`
hands the stage to the large model.

| # | Stage | Tier | Exit gate |
|---|---|---|---|
| 0 | Intake and delta triage against the nearest supported model | large | written list of what differs from the nearest supported model |
| 1 | Environment and CPU reference | small | reference reproduces the model card's published behavior |
| 2 | Functional decoder on one chip | small runs, large diagnoses | PCC and argmax thresholds |
| 3 | Full model | small runs, large diagnoses | end-to-end parity with the reference |
| 4 | Multichip, then shrink to 2 and 1 chips | large plans, small runs; large diagnoses | per-config evidence required by `mesh-shrink` |
| 5 | Serving integration | small | black-box server checks pass |
| 6 | Qualitative check and benchmark | small | measured numbers, each labeled measured or TODO |
| 7 | Package and container build | none; supervisor waits, chips released | image boots and passes the stage 5 checks |
| 8 | Operator bundle | small | results file, ledger, open risks, exact publish commands, scrub check clean |

Tiers are named in a local config file (`config/tiers.toml`, not committed; TOML because the
supervisor uses only the standard library). Model ids for each tier are chosen after the
measurements in section 12.

Each stage names the tier that runs it (`run`; `none` for stage 7, where no model is loaded).
Stages 2, 3 and 4 also name a `diagnose` tier, which takes over when the stage escalates. Stage 4
also names a `plan` tier (large plans, small runs); no other stage may have a `plan` key, and
stage 7 has neither `plan` nor `diagnose`. The config has a required `[escalation]` table with
`default = "<tier>"`. That tier takes over when any stage without a `diagnose` tier (0, 1, 5, 6 and
8) escalates. It must name a defined tier. The loader refuses keys it does not know, so a typo such
as `diagnos` is reported and not ignored.

### 5.1 Sizing the small and CPU tiers

Host, read on 2026-10-01: AMD Ryzen 7 9700X (8 cores, 16 threads, AVX-512 with bf16 and VNNI),
249 GB RAM in four 64 GB DIMMs rated 5600 MT/s and configured at 3600 MT/s, 133 GB available with
today's workload. Dual-channel DDR5 at 3600 MT/s has a theoretical peak of 57.6 GB/s.

CPU decode speed is limited by memory bandwidth. Each generated token reads every active weight
once, so tokens per second cannot exceed bandwidth divided by the bytes read per token. Ceilings at
57.6 GB/s, computed from model size and **not measured** (real speed will be lower):

| Model class | Bytes read per token (approx.) | Ceiling |
|---|---|---|
| 0.6B dense, 8-bit (the size tt-local-generator's prompt server uses) | 0.6 GB | 96 tok/s |
| 8B dense, 4-bit | 4.7 GB | 12 tok/s |
| 32B dense, 4-bit | 19 GB | 3 tok/s |
| mixture-of-experts, about 3B active, 4-bit | 1.8 GB | 32 tok/s |

A 0.6B model is enough for prompt polish. Agent steps need reliable tool calls, so the CPU tier
starts larger and is chosen by measurement. Dense models above about 8B are too slow for agent steps
on this memory. The class worth measuring first is a mixture-of-experts model with a few billion
active parameters: all weights must fit in RAM (about 18 GB for a 30B model at 4-bit, against 133 GB
available), and the per-token read stays small. This names a class to test and is not a model
decision. Each candidate is measured under load, with the large model serving, because both share one
memory bus. The sizing tool in plan 1 (`orchard/sizing.py`) takes these measurements.

Stage 0 decides the path. If the family matches a supported model, the run covers only what
changed. If it does not, the run is a full port and the large tier plans each later stage.

## 6. Park and restore

Runs when a stage needs a board the coder holds and no free board exists. If a free board exists,
the coder stays loaded and none of this runs.

1. **Handoff note.** The coder writes goal, stage, evidence so far, the next action and what to
   check on return. The supervisor confirms the file exists and parses.
2. **Stand-in first.** The supervisor starts the CPU stand-in and sends it a canary prompt. It stops
   the coder only after the stand-in answers.
3. **Yield.** `tt-model stop` shuts the coder down. `gozer yield` releases and resets the board and
   records a reservation for the holder with an expiry.
4. **Test.** The stand-in takes the board under a normal lease and runs the test with a deadline.
   A hang goes to the triage skill. Release resets the chips.
5. **Redeem.** The supervisor releases the test lease, redeems the reservation, starts the coder
   with `tt-model serve`, waits for ready, and sends the pre-park canary prompt. It compares the
   greedy answer with the pre-park answer.
6. **Resume.** The coder reads the handoff note and a short summary of the test result. The
   stand-in stops.

The stand-in never touches the coder's weight or tensor caches. A cleared cache turns a 2-3 min warm
restart into a cold boot of about 30 min.

Branches:

| Condition | Action |
|---|---|
| Reservation expired | queue normally, ledger warning |
| Coder does not return within the run's cold-boot budget | stage blocked, run paused for the operator, no blind retry |
| Supervisor crash while parked | restart reads `parked` from the ledger, then checks `gozer status` and `tt-model ps`; the machine wins on disagreement |
| Canary answer differs after restore | stage blocked until the operator or the large tier explains the difference |

## 7. Watchdog

Two kinds of agents, two sets of powers.

- **Agents the supervisor launched.** It sits between them and their model server and sees every
  request and response. It may interrupt, change the prompt, or escalate.
- **Agents someone else started.** It reads their session transcripts and lease activity. It writes a
  notice with the evidence and takes no action. It does not kill or edit them.

Signals:

- the same tool call with identical arguments, or an identical output hash, N times in a row
- thinking tokens past a per-step cap with no tool call or file written
- tokens flowing for T minutes with no new evidence file, ledger entry or test result
- a lease held while its chips sit idle (taken from gozer's comparison of leases with kernel state)
- a stage past its wall-clock budget

N, T and the caps are config values set at plan time from the replayed transcripts in section 12.

Response ladder for launched agents:

1. Nudge: add a message that names the repeat and requires a different approach. The greedy-only
   server cannot vary sampling, so the change comes from the prompt.
2. Escalate the step to the large tier.
3. Pause the run and notify the operator, with the evidence attached.

Each rung has a cap, so the ladder cannot loop.

## 8. tt-gozer changes

We are gozer's main consumer, so changes land in `tt-gozer` with tests, each in its own commit.

1. **`gozer yield`.** Release a board's chips (resetting them as `release` does) and record a
   reservation: board, holder, expiry. Allocation and `may_claim` refuse every other tenant for
   that board until the reservation is redeemed or expires. This is separate from the existing
   90 s claim window, which only entitles the head ticket for a short period after a release.
2. **`gozer redeem`.** Turn a live reservation into a lease on the same board.
3. **Expiry.** `gozer reconcile` (and the existing timer) clears an expired reservation. A crashed
   supervisor cannot hold a board past the expiry.
4. **JSON.** `--json` already exists on every command. The new commands support it. The adapter
   consumes only JSON.
5. **Known limit to account for.** Tickets expire after one hour regardless of polling, so a stage
   that waits longer than that on the queue must re-enqueue and record the loss of place.

Open design point for the plan: how reservations interact with a queue that already has waiting
tickets. `gatekeeper.py` allocation (750 lines) has not been read for this design. The plan must read
it first and state the rule.

## 9. Ledger

One append-only `ledger.jsonl` per run, with an `evidence/` directory beside it.

- Entry fields: sequence number, timestamp, stage, event, tier, model id, result, evidence paths with
  sha256.
- Events: `run_start`, `stage_start`, `stage_end`, `park`, `restore`, `retry`, `escalate`, `notice`,
  `measurement`, `decision`. A `measurement` entry carries a `label` of `measured` or `TODO`.
  The label applies to every field the tool reports in that entry. A `null` under `measured` means
  the source did not report that field, and the sibling flags say why (`load_cold`,
  `prefill_cached`, `decode_complete`, and the `resident_bytes` warning on stderr). `TODO`
  entries are placeholders for a measurement not taken; the operator-bundle stage (stage 8)
  writes them.
- At run start the ledger records the resolved versions: tt-metal commit, vLLM, `tt-model`, firmware.
  They stay fixed across resumes. A new run re-resolves them.
- Current state is computed by replaying the ledger. No separate state file exists.
- Each line holds the hash of the previous line and is flushed to disk when written. On resume, a
  torn last line is cut off the ledger and kept in a `.torn-<time>` file next to it (created
  exclusively and synced before the cut, so nothing is lost or overwritten).
- One process holds the ledger open at a time (an exclusive lock on a `.lock` file). A second
  opener gets an error. The lock is released when the process dies, so a restarted supervisor can
  reopen a crashed run.
- If an append fails partway (disk full, I/O error), the file is cut back to its size before the
  attempt. If that cut also fails, the writer refuses further appends until it is reopened.
- A file that is not a valid chain (a line that is not a JSON object, a non-UTF-8 byte, a wrong
  sequence number, a broken hash) is reported as corrupt and a person looks at it. The last line has
  no successor, so a change to it alone is not detected.

## 10. Failure handling and denials

| Condition | Action |
|---|---|
| Supervisor restart | replay the ledger, compare with the machine, machine wins |
| Stage with no `stage_end` | restart from its beginning or its declared resume marker; hour-long stages must declare one; the partial directory is moved aside, never deleted |
| Model server dies | one restart with the same config and a canary check; a second failure blocks the stage |
| Low disk | each stage declares required free space and the supervisor checks first; if short, the stage blocks; the supervisor never deletes a cache it did not create |
| Chip left dirty by a fatal DRAM OOM | release through the lease tool, which resets that board; if the reset fails the stage blocks and a notice is written; nothing runs `tt-smi -r` by hand |
| Devices missing during another agent's capture window | record an external-pause event and wait, with a limit; the stage is not failed |
| Queued (exit 10) | wait on the ticket within the stage budget; cancel stale tickets |
| Budget cap reached (wall clock, escalations, cold boots) | pause the run for the operator |

Operator commands: `pause`, `resume`, `abort`. Abort releases the hardware and closes the ledger.

Denials enforced by the command runner, which every stage agent's shell goes through:

- `tt-model push` and `tt-model publish`; `git push` (including `subtree push`, `lfs push`, git
  aliases set with `-c`, and the `git-push` helper); `gh repo create`; `hf upload` and
  `huggingface-cli upload` (any `upload*` subcommand).
- `tt-smi -r` in any spelling (long-option prefixes and clustered short flags included).
- `rm`, `rmdir` and `unlink` on any path outside the run directory, on a ledger file, through
  `xargs`, or with a glob directly in the run directory root (a glob there could match the ledger).
- A redirect to a ledger file, or to a target built from a variable or glob (the runner cannot tell
  whether it is the ledger). Other redirects are not judged.
- `ln -s` combined with a delete in the same string (the delete could go through the link).

The runner fails closed. Shell syntax it does not model is refused, and the message names the
construct and says how to rewrite the command. That covers command and process substitution,
heredocs, subshells, `eval`, `source`, shell keywords, shells that read stdin or a script, shells
other than bash, sh and dash, wrappers it does not list, and options it does not know. A prompt
that is ignored cannot bypass these, because the check sits where commands run.

The runner is best effort. It is a hand-written lexer and not a shell parser, and it does not stop
arbitrary code (`python3 -c`), tools it does not name (`mv`, `rsync --delete`, `tee`), or a script
that runs a denied command inside it. Outer layers belong to plans 3 and 4: agent shells run
without GitHub and Hugging Face tokens, and with read-only mounts everywhere outside the run
directory. The runner is one layer among these.

**OPEN DECISION.** No adversarial search for bypasses of the runner has been done. Reviews so far
read the code and ran its tests. One reviewer that tried an adversarial search was stopped by a
safety classifier, and the search was not re-run. The operator must decide, before plan 4 ships,
between sanctioning an adversarial review of the runner and accepting best effort plus the outer
layers.

Bundle scrub check (stage 8): search the package and card for the machine hostname, tokens and
absolute home paths. A hit blocks the bundle.

## 11. Skills

New skills, each named for the plugin that owns it:

- `tt-model-bringup`: `delta-triage`, `reference-gate`, `operator-bundle`. They never name a lease tool.
- `tt-gozer/skills`: `gozer-park` (park and restore from the agent's side).
- Existing skills used by stages: `model-bringup`, `functional-decoder`, `full-model`, `multichip`,
  `mesh-shrink`, `vllm-integration`, `qualitative-check`, `benchmark-model`, `tt-device-usage`,
  `stage-review`, `tti-release`, plus `gozer-keymaster` and `gozer-gatekeeper`.

Changes to `tenstorrent/skills` follow that repo's rules: both manifests, both catalogues, version
bump in both manifests, README update, CODEOWNER, and `pytest tests/`. No PR is opened without the
operator asking. The unreviewed `mesh-shrink` work and the drafts under `docs/superpowers/` on the
current branch are not part of this spec.

## 12. Measurements required before trust

| Measurement | Decides |
|---|---|
| CPU stand-in load time and decode speed, per candidate model | which model fills the small tier and whether CPU is usable beyond short steps |
| Free host memory with the large model resident | whether the stand-in can start before the coder stops |
| Canary answer identical across a coder restart | exact-text check or a looser one |
| Replay of the recorded qwencode loop (if the transcripts are still on disk; not yet checked) | N, T and the caps; the detector must fire |
| Replay of a transcript from a run that worked, such as the Audio8 bring-up | the detector stays quiet |

If the qwencode transcripts are gone, a synthetic loop is a weaker test and the plan must say so.

## 13. Testing

- Fault injection: kill the supervisor after each ledger event and confirm the resumed run reaches the
  same final state.
- Remove each guard (a denial, the stand-in-first order, the torn-line check) and watch the matching
  test fail before restoring it.
- Gozer changes follow `tt-gozer`'s existing test layout (`tests/test_*.py`), with a concurrency test for
  the reservation.
- A run against a model whose bring-up already succeeded (for example the Audio8 port) is the first
  end-to-end check, with hardware work under a lease. It compares the run's results with the known
  ones from `~/code/audio8-asr`.

## 14. Open questions

1. Which open-source models fill the large, small and CPU tiers. Decided after section 12.
2. How reservations interact with a queue that has waiting tickets (section 8).
3. Whether the watchdog's per-request view needs a proxy in front of the model server or the server
   already logs enough.
4. Remote repository name and owner. The working assumption is `tsingletaryTT/tt-orchard`. Nothing is
   created remotely until the operator asks.
5. Bypass search of the command runner (section 10). Sanction an adversarial review, or accept best
   effort plus the outer layers of plans 3 and 4. Decide before plan 4 ships.
