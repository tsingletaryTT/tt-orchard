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
| tt-gozer changes (`gozer reset`, descendant ownership, `gozer-park` skill) | built, reviewed and checked on real hardware (both boards); on branch `orchard-hold-through-swap` in a worktree of `~/code/tt-gozer`; not merged |
| Hardware-check driver (`orchard/hardware_check.py`) | built, reviewed, run on both boards: 22 to 24 checks passed on each run, plus a container-server check (spec section 14, item 2) |
| Supervisor loop, park and restore, watchdog | designed, not started |
| Stage state machine, new bring-up skills, operator bundle | designed, not started |

The suite has 607 tests and runs without hardware or network. The driver's tests use fake gozer roots and a stub child.

Measured on this machine (one p300c board): an in-place `gozer reset` takes 41.7 s; `tt-model stop` takes 1.6 to 1.9 s; the Audio8 container reaches ready in about 20 s from a warm start; a lease owned by a live pid stays valid past gozer's 900 s detached window. During any reset the other board shows as busy for about 42 s, because `tt-smi -r` opens every device.

## How a run is meant to work

A run is a sequence of stages. Each stage has an owner skill, a model tier and an exit gate. The
supervisor starts a stage with a fresh, short model context built from the run ledger.

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

A watchdog looks for loops (repeated identical calls, runaway thinking, no new evidence) in agents
the supervisor launched, and nudges, escalates or pauses. For agents it did not launch, it only
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
