# Full port under the supervisor: design and first slice

Date: 2026-11-02. Status: slice 1 is built by the branch that adds this note; slices 2 to 6 are filed as issues.
Nothing here ran on a chip.

## 1. The problem

Stage 0 classifies a model as `full-port` when its tensors or config differ from the nearest supported model in a way the
existing TT code cannot load. A real example is `InternScience/Sage-A1.5` (nearest model Qwen3.5-35B-A3B): stage 0 and
stage 1 passed, then the run paused at stage 2 with "plan 4 runs weights-only bring-ups" and ended blocked with
`needs-new-model-code`. Three things are missing today:

1. An unattended run cannot be told to go on. `_full_port_unacknowledged` blocks every unattended full port, and the
   attended resume that gets past it runs stage 2 with the generic `functional-decoder` skill, which is not in this
   repository and has no gate that checks a measurement.
2. Stages 3, 5 and 6 on the full-port path use the generic table too. Their gates read a few fields an agent writes
   (`parity`, `top1`, `checks.*.pass`). Nothing ties those fields to a hardware test.
3. There is no record of what the hand ports did (MiniCPM3, Granite) in a form a supervised run can reuse.

## 2. What the hand ports and the studies show

Read for this note, with the limits noted.

- `scripts/minicpm3` (overnight 2). One chip, no supervisor. The shape that worked: a plain-torch CPU reference
  (`ref_model.py`), a chip model with a prefill and a decode step (`chip_model.py`), a teacher-forced agreement run against
  recorded reference ids (`run_chip.py agree`), then a speed run, a free-running quality check and a server. Every number in
  the run log comes from a script that writes a record file. The port code and the check are separate files.
- The text-runtime path (`orchard/text_runtime.py`, `orchard/text_stages.py`) already supervises these scripts, but only for
  models whose scripts are in this repository, and it uses no agent.
- Flash-Next study (`docs/run-logs/2026-11-01-flash-next-graft-study.md`). The ports that held up carry a pinned acceptance
  corpus that the server replays at every boot and refuses to serve without (96 of 96), a teacher-forced comparison with a
  Hugging Face reference (8,416 positions, top-1 0.9525, KL 0.0528), and a card that names the runtime kind. The ports that
  did not hold up give one decode number with no evidence file. Section 8 of that study lists what orchard lacks: a residency
  estimator and an inventory that can trace a sparse-attention layer.
- TTIS study (`docs/run-logs/2026-11-01-tt-inference-server-maturity-study.md`, branch `ti-server-maturity-study`, PR 320).
  A model graduates through `EXPERIMENTAL`, `FUNCTIONAL`, `COMPLETE` by passing evals and benchmark tiers, and the promotion
  is "a separate human decision". That maps to the shape used here: the supervisor produces tested evidence, and a person
  promotes.
- The `Sage-A1.5` delta. The only text tensor difference is that experts are stored one tensor per expert and projection, where
  the nearest model stores two fused 3-D tensors per layer. That is a weight repack, and stage 0 calls it a full port because
  the tensor names do not line up. Whether a repack class belongs in stage 0 is a separate question (slice 5).

## 3. Design

A full port gets its own route through stages 2 to 6, each stage with a local skill, a hardware test run by the supervisor
and a gate that checks the evidence and does not trust the agent's summary.

### 3.1 The route

| Piece | Rule |
|---|---|
| Opt-in | The operator starts the run with `--full-port` (`tt-orchard bringup MODEL --full-port`). The ledger records it in `run_start` and a resume keeps it. Without it nothing changes: a full port still blocks. |
| Which classes | Only stage 0's class `full-port`. `unknown` and `restricted-head` keep their pauses, because for those stage 0 either could not decide or the operator ruled the harness does not serve them. |
| Stage 2 | `full-port-check` skill and `gate_full_port` (slice 1). One board, one hardware test. |
| Stages 3 to 6 | Not built in slice 1. After stage 2 passes, the run pauses before stage 3 with a reason that names the unbuilt stages, and an unattended run ends blocked with code `full-port-stages-unbuilt`. Slices 3 and 4 replace the pause. |

### 3.2 Who writes what in stage 2

The stage keeps the shape every hardware stage already has:

1. Prepare (agent). Writes the port under `stages/2/port/` (model code) and a check script, then `hw_test.json` and
   `handoff.json`. The coder reads the stage 0 delta, the stage 1 reference and the nearest model's TT code as examples.
2. The supervisor sets aside any result file written during prepare (`swap_prepare.set_aside_results`, which now also covers
   `evidence/port-check.json`), then runs the one command under a lease. It parks the coder when no board is free, as for the
   weights-only stage 2.
3. Finish (agent). Reads the test output and writes `result.json`.
4. The gate decides.

### 3.3 The fabrication guards

The guards from PRs 313 and 314 carry over without change: results written before the test are set aside; the resume marker
`test-result.json` is trusted only when the ledger holds the supervisor's own entry for it; links out of `evidence/` are not
evidence. Slice 1 adds a gate (`orchard/full_port.py`, called by `gate_full_port`) that checks `evidence/port-check.json`, the
record the hardware command writes:

- The reference ids in the record are the ids in the stage 1 file, and the record names that file and its sha256. The gate
  re-reads the file and compares.
- The gate recomputes top-1 agreement from the record's chip ids and the reference ids. `result.json` must report the same
  number. The bar is `SWAP_TOP1_MIN`, the same value the weights-only stage 2 uses.
- At least `SWAP_MIN_TOKENS` positions are compared.
- The record's chips are the chips of the supervisor's `test-result.json`, whose exit code is 0 and which did not time out.
- `hw-test-output.txt` is the file the supervisor recorded in `test-result.json` (sha256 match), and it carries the sha256 of
  `port-check.json`, so the record and the test run belong together.
- The port sources listed in the record exist, match their sha256, include a file that mentions `ttnn`, and none of the model
  sources mentions the stage 1 reference directory. A model file that reads the reference could copy ids to the record.
- `free_run_text` in `result.json` equals the record's text and has enough distinct words to be text.

These are checks on shape and provenance. They do not prove the chip computed the ids. A person who wants that proof
replays the command, which the evidence makes possible. Slice 2 moves the comparison out of agent-written code into a
supervisor-owned template (as `serve_and_compare.py` does for weights swaps), which closes the gap where the agent writes the
script that writes the record.

### 3.4 Leases and parking

No change in slice 1. The stage spec asks for one board, and the existing hardware phase (`decide_park`, `_handoff`) handles the
coder. A model that does not fit one board stops at the prepare step with a refusal that says so; the residency estimate is
slice 4.

## 4. Slices

| Slice | Content | Issue |
|---|---|---|
| 1 (this PR) | Opt-in flag, route for class `full-port`, stage 2 skill and gate, pause before stage 3, block code | this PR |
| 2 | Supervisor-owned `port_check.py` template and a port module interface (load, prefill, decode step), so the agent writes only model code | filed |
| 3 | Stage 3 (full model on the configured chips) and stage 4 (shrink) for full ports, using the hardware test list (`hwtests.py`) | filed |
| 4 | Residency estimator from safetensors headers, used before stage 2 to say how many chips a port needs | filed |
| 5 | Stage 0: a weight-repack class for models like Sage-A1.5 whose only difference is the tensor layout of experts | filed |
| 6 | Stages 5 and 6 for full ports: a serving check with a pinned acceptance corpus that the server replays at boot, then quality and benchmark with repeats | filed |

## 5. Not decided here

- Whether a full port may be published. Nothing in this design publishes. Operator rule stays: never publish without an
  explicit go-ahead.
- Whether the local coder can write a port at all. Slice 1 only makes the route exist and keeps its evidence checkable. The
  first real attempt is an operator-run pilot.
