# tt-orchard

A supervisor that brings a new model up on a Tenstorrent Quietbox using only open-source models
served from the same machine. A run starts from a Hugging Face model id and ends with a private,
evaluated package and a bundle for an operator to review. The run never publishes anything.

Shrinking a model from 4 chips to 2 or 1 chip is one stage of a run. The skills for that live in
the `tt-model-bringup` plugin of the `tenstorrent/skills` repo (`mesh-shrink`).

## Status

Early. The hold-through-swap pattern has been checked on this machine's two boards (see below). No real model bring-up has been driven by the supervisor yet.

| Part | State |
|---|---|
| Run ledger (`orchard/ledger.py`) | built, tested |
| Command runner (`orchard/runner.py`) | built, tested; best effort, see "Safety" |
| Tier config loader (`orchard/tiers.py`) | built, tested |
| CPU model sizing tool (`orchard/sizing.py`) | built, tested against a fake server; not yet run against a real ollama |
| tt-gozer changes (`gozer reset`, descendant ownership, `gozer-park` skill) | built, reviewed and checked on real hardware (both boards); merged to `main` in `~/code/tt-gozer` |
| Hardware-check driver (`orchard/hardware_check.py`) | built, reviewed, run on both boards: 22 to 24 checks passed on each run, plus a container-server check (spec section 14, item 2) |
| Lease adapters (`orchard/adapters/`: gozer, single tenant) | built, tested against fakes; the gozer adapter also against the real gozer CLI with fake roots |
| Server control and canary (`orchard/server.py`, `orchard/canary.py`) | built, tested; stop checks also run against real ps, pgrep, ss and curl with a fake server |
| Park and restore (`orchard/handoff.py`) | built, tested against a fake machine, including a crash after every ledger event; the park check ran once on real hardware on 2026-10-02 (see below) |
| Watchdog (`orchard/watchdog.py`, `orchard/transcripts.py`) | built, tested; thresholds set from a replay of the recorded qwen-code loop and 30 quiet chats |
| Stage machine, agent steps and supervisor loop (`orchard/stages.py`, `orchard/agent.py`, `orchard/context.py`, `orchard/supervisor.py`, `orchard/scrub.py`) | built, tested against a fake machine and fake model servers, including a kill after every ledger event; not yet run on hardware (Hemmingway-1 entry in `docs/runbooks/hardware-validation.md`) |
| Stage skills (`orchard/skills/`: delta-triage, reference-gate, serving-check, operator-bundle) | local drafts; their home is the tt-model-bringup plugin |
| Model proxy for agents the supervisor did not launch, stage 7 (package and image build) | designed; no code yet |

The suite has 1199 passing tests and 1 skipped, and runs without hardware or network. 854 of them predate plan 4. The skipped test is the opt-in replay of local qwen-code transcripts (`ORCHARD_REPLAY=1 python3 -m pytest tests/test_replay_local.py`). 607 of the tests predate plan 3. The driver's tests use fake gozer roots and a stub child.

The park check (`orchard/park_check.py`) ran on real hardware on 2026-10-02, on board 1 (chips `0000:03:00.0` and `0000:04:00.0`), with the default options and fake coder and stand-in servers. It exited 0. The park reset took 41.676 s and the restore reset took 41.657 s. The record is in `runs/park-check/20261003T011558Z-0000-03-00.0/`, which git ignores. It is one run on one board; it did not use a real model.

Measured on this machine (one p300c board): an in-place `gozer reset` takes 41.7 s; `tt-model stop` takes 1.6 to 1.9 s; the Audio8 container reaches ready in about 20 s from a warm start; a lease owned by a live pid stays valid past gozer's 900 s detached window. During any reset the other board shows as busy for about 42 s, because `tt-smi -r` opens every device.

## How a run is meant to work

A run is a sequence of stages. Each stage has an owner skill, a model tier and an exit gate. The
supervisor starts a stage with a fresh, short model context built from the run ledger. When an
agent says it is finished and the stage's exit gate fails, the same conversation is told the gate's
reasons and gets one short continuation (up to 20 turns) before the stage is escalated. A reply with
no text and no command, or one cut off at max_tokens, does not count as finished.

| # | Stage | Model tier |
|---|---|---|
| 0 | Intake and delta triage against the nearest supported model | large |
| 1 | Environment and CPU reference, checked against the model card's published behavior | small |
| 2 | Functional decoder on one chip | small runs, large diagnoses |
| 3 | Full model | small runs, large diagnoses |
| 4 | Multichip, then shrink to 2 and 1 chips | large plans, small runs |
| 5 | Serving integration | small |
| 6 | Qualitative check and benchmark | small |
| 7 | Package and container build | none (the supervisor waits and the chips are released) |
| 8 | Operator bundle | small |

When a stage needs a board that the large model is using, the supervisor parks the large model,
resets the board in place, runs the work, and brings the large model back. It holds the board lease
under its own pid the whole time, so no other agent can take the board during the swap. A small
local CPU model keeps the loop running while the large model is stopped.

A watchdog looks for loops (repeated identical calls, the same set of calls turn after turn,
runaway thinking, no new evidence, 20 turns without writing a file) in agents the supervisor
launched, and nudges, escalates or pauses. For agents it did not launch, it only
writes a notice.

The full design is in `docs/superpowers/specs/2026-10-01-orchard-design.md`.

## What is built

- **Ledger.** An append-only JSON-lines file per run. Each line holds the hash of the previous
  line. It has a single-writer lock, rolls back a failed append, keeps each torn tail in its own
  file, and reports any malformed content as `LedgerCorrupt`. The state of a run is computed by
  replaying it.
- **Command runner.** The only way the supervisor runs commands for a stage agent. It reads a small
  subset of shell and refuses everything else, with a message that names the construct and says how
  to rewrite it. It refuses `tt-model push` and `publish`, `git push`, `gh repo create`,
  `hf upload`, `tt-smi -r`, deletion outside the run directory, and a few related commands.
- **Tier config.** Maps stages to model tiers in a local TOML file. It refuses remote endpoints,
  unedited `CHANGE-ME` values, unknown keys, and stages nobody owns.
- **Sizing tool.** Measures load time, prefill speed, decode speed, resident size and free host
  memory for a model served by a local ollama, and records each result in a ledger. It labels what
  it measures so a number from a cached prompt or a short decode is not mistaken for a normal one.
- **Lease adapters.** One interface (`LeaseAdapter`) over the lease tool. The gozer adapter drives
  the gozer CLI, never passes `--force`, never calls `gozer wait`, and claims a queue ticket by
  polling `acquire`. The single-tenant adapter is for a machine with no lease tool; it leases whole
  boards and can be rebuilt after a crash.
- **Server control and canary.** Starts, stops and checks a model server, and confirms a stop only
  when the server's own checks and the lease tool agree. The canary asks a fixed question before a
  park and after the restore and compares the two answers exactly. It asks for no reasoning phase, so
  a small token budget is enough. The supervisor also asks the coder one arithmetic question on its first
  start in a run and blocks if the answer lacks 42. A fake OpenAI-shaped server
  (`orchard/fake_server.py`) opens no device and is used in tests and in the park check.
- **Park and restore.** `orchard/handoff.py` stops the coder, resets the board in place, and brings
  the coder back. It writes one ledger entry per step. After a restart it replays the ledger and
  trusts the machine where the two differ. `orchard/park_check.py` runs it on one real board with
  fake servers (runbook: "Park check").
- **Watchdog.** A normalised event stream, eight detectors, a capped response ladder and a retry
  guard. `NoFileWritten` and `TurnRepeat` were added on 2026-10-03 after a live stage 2 step grepped
  for 60 turns and wrote nothing. It acts through an injected actuator that plan 4 supplies. `orchard/transcripts.py` turns
  a qwen-code transcript into that stream, and the detector thresholds come from a replay of the
  recorded loop.
- **Supervisor.** The supervisor (`python3 -m orchard.supervisor run ...`) runs stages 0 to 6 and 8 of a weights-only bring-up with local models, parks the coder for hardware stages, records every step in the ledger, and stops at ready for operator review. It never publishes. The path stage 0 writes in `delta.json` decides stages 2 and 3. On `weights-only`, stage 2 serves the new weights with the nearest model's existing TT implementation and compares the chip's tokens with the CPU reference (skill `weights-swap-check`, gate `gate_weights_swap`), and stage 3 is recorded as skipped. The agent copies two tested scripts from `orchard/skills/weights-swap-templates/` and fills in one config file; it does not write a serving script. On `full-port`, or a path the supervisor cannot read, stage 2 runs `functional-decoder` and stage 3 runs as before.

## Try it

Python 3.12 and pytest are the only requirements. There are no runtime dependencies.

```bash
python3 -m pytest -q
```

Copy the example tier config and fill in the models you chose. The loader refuses any value left as
`CHANGE-ME`, so an unedited copy cannot start a run. `config/tiers.toml` is ignored by git.

```bash
cp config/tiers.example.toml config/tiers.toml
```

Measure a CPU model. The tool talks to an ollama that is already running on this machine and has the
model pulled. It downloads nothing. Run it with the large model idle and again with it serving,
because both share one memory bus.

```bash
python3 -m orchard.sizing --model <name> --prompt-file prompt.txt \
    --gb-per-token <active weights in GB> --note "coder idle" \
    --ledger runs/sizing/ledger.jsonl
```

`--gb-per-token` turns on the ceiling: tokens per second cannot exceed memory bandwidth divided by the
gigabytes read per token. On this machine's configured 3600 MT/s memory the theoretical bandwidth is
57.6 GB/s, so the ceiling for a model that reads 1.8 GB per token is 32 tokens per second. Real
speed is lower.

## Safety

- A run ends at "ready for operator review". It never pushes, publishes, or lists a model in a
  catalog. The operator does that.
- Every model endpoint must be on this machine. The tier loader refuses anything else.
- The command runner is a best-effort guard. It tokenizes shell text, and it does not stop arbitrary
  code such as a Python one-liner that deletes files. Nobody has searched the runner for bypasses.
  The outer layers belong to the supervisor work that is not built yet: agent shells that run
  without GitHub and Hugging Face tokens, and read-only mounts outside the run directory.
- gozer cannot see device handles held by root-owned or container processes, so the supervisor
  confirms with the server's own tooling that a server has stopped before it resets a board.

## Open decisions

- Whether to sanction an adversarial review of the command runner, or accept best effort plus the
  outer layers. This must be settled before an agent gets a shell through the runner.
- Decided 2026-10-02: the large tier is Qwen3.8-27B on all four chips, the small tier is
  Qwen3.8-27B on two chips, and the CPU tier is `qwen3-coder:30b` on a local ollama. The CPU
  tier measured 14.1 tokens/s decode and 93 tokens/s prefill with the chips idle. Still open:
  its speed while the large model serves, and whether it makes reliable tool calls. A dense
  27B on CPU measured 0.70 tokens/s decode and was ruled out for anything but a canary.

## Where things are

| Path | What |
|---|---|
| `orchard/` | the supervisor package |
| `config/tiers.example.toml` | example tier config |
| `tests/` | the tests |
| `docs/superpowers/specs/` | the design |
| `docs/superpowers/plans/` | implementation plans: plan 1 (core, built) and plan 2 (tt-gozer changes, built on a branch) |
| `docs/runbooks/hardware-validation.md` | the checks for a real board, with the first-session procedure (run on 2026-10-02) |
| `docs/runbooks/driver-build-report.md`, `driver-review-findings.md` | how the hardware-check driver was built and what its safety review found |
| `CLAUDE.md` | project log: the original prompt, decisions, and notable moments |

Related work: `~/code/tt-gozer` (the chip lease tool; the supervisor depends on `gozer acquire
--owner-pid` and on the new `gozer reset`) and the `tt-model-bringup` plugin in the
`tenstorrent/skills` repo.
