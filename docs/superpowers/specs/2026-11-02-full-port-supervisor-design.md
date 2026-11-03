# Full port under the supervisor: design and first slice

Date: 2026-11-02. Status: slice 1 (PR 331) and slice 2 (issue 325) are built; slices 3 to 6 are filed as issues. Since then: the rehearsal option on the stage 2 gate (issue 357) and a prefill-reset check from the plugin mapping
(issue 363, `2026-11-02-port-interface-vllm-plugin-mapping.md`) are built. Slice 5 (issue 328) needs a layout detector in the stage 0 template, a
class, and a converter in stage 2; none of the three exists yet.
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

1. Prepare (agent). Writes the port under `stages/2/port/` (model code only; since slice 2 the check is the supervisor's template), then `hw_test.json` and
   `handoff.json`. The coder reads the stage 0 delta, the stage 1 reference and the nearest model's TT code as examples.
2. The supervisor sets aside any result file written during prepare (`swap_prepare.set_aside_results`, which now also covers
   `evidence/port-check.json`), then runs that single command under a lease. It parks the coder when no board is free, as for the
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
replays the command, which the evidence makes possible.

Slice 2 (issue 325) moved the comparison out of agent-written code. The coder writes only a port module,
`stages/2/port/port_model.py`, with four functions: `load(model_dir, mesh_device)`, `prefill(model, ids)`,
`decode_step(model, token_id, position)` and `close(model)`. The supervisor copies the template
`orchard/skills/full-port-templates/port_check.py` and a config (`port_check_config.json`, the run's model snapshot) into
`stages/2/`, puts both back as shipped after the prepare step, rewrites `hw_test.json` to the fixed command
`python3 -P -s stages/2/port_check.py`, and runs it under the lease. The template reads the stage 1 ids, does the teacher
forcing, counts the agreement, makes the free run, decodes its text with the model's `tokenizer.json`, writes the record and
prints `PORT_CHECK_SHA256`. It runs the port module in a second process (the worker) and takes one integer id per request, so
a port that prints a mark line, writes the record, replaces the template, patches the template's functions or installs an import
hook cannot reach the process that makes the record. The gate refuses a template copy that differs from the shipped file, a record
without the shipped file's sha256, a command other than the fixed one, an output with other than one mark line, a record that
does not match its config, and a port file that is not in the record. What this still does not prove: a port that returns the
reference's ids without computing them cannot be told from one that computed them, and a coder who can edit the shipped template
in the orchard checkout is outside what the gate sees (the outer layers in specs sections 10 and 14 cover that).

### 3.4 Leases and parking

No change in slice 1. The stage spec asks for one board, and the existing hardware phase (`decide_park`, `_handoff`) handles the
coder. A model that does not fit one board stops at the prepare step with a refusal that says so; the residency estimate is
slice 4.

### 3.5 Repair rounds (issue 420)

The first unattended run of the tiny Qwen4 rehearsal (docs/run-logs/2026-11-03-tiny-qwen4-rehearsal-first-run.md) failed stage 2's port check in nine
seconds at `load`, on a mistake the failure text named: the port counted the config's one-indexed `ple_layer_ids` from zero. The stage ended at the failed
test (`_test_failure`, on purpose: the finish step records a failed test and cannot be asked to make it pass), and the next attempt began again.

A failed test of the port now earns the coder up to N repair rounds in the same attempt (`--port-repair-rounds`, default 3, recorded in `run_start`).

- **What earns a round.** The supervisor's record of the test shows exit code 2 without a timeout, and the output file the record names (read through the
  bounded reader, same sha256) holds a `PORT_CHECK_FAILURE` line of kind `port`. The template prints that line, unprefixed, after `FAILED:`; every line
  of the port's worker carries a prefix, so a port cannot write one. Kind `setup` (no leased chip, a stage 1 file that does not read), a timeout and any other
  exit code earn none.
- **What a round does.** The ledger decision comes first. A repair step edits only `stages/2/port/`. `test-result.json` and the output file must still be
  the files the supervisor wrote, or the attempt fails with the named reason. The template, config and `hw_test.json` are put back, results written during the
  step are set aside, and the same check runs under a lease. The new record is the supervisor's, with its own ledger hash. A round that changes nothing in
  the port ends the rounds without a test.
- **Setup and cleaning.** A failed device open at `init` is kind `setup` (the worker says so); a port import error is kind `port`. The output text is cleaned as a
  whole (home paths, URL user information, tokens) before the ledger's 300-character failure line or the prompt's tail is cut from it.
- **A kill between an edit and its rerun.** The ledger holds the digest of the port directory each test ran against (`port repair tested`). A resume that finds
  a different directory tests that edit first, without a new repair step or a new round, and only then goes on. This also holds for the last round of an attempt
  (an escalated attempt keeps its directory), and for a kill while the rerun is on the chips: the test opens its output file fresh, so the old record's hash stops
  matching. The decision `port repair rerun started` (with the seal of the old record) is written just before the rerun; with no `port repair tested` after it a
  resume tests again if the record is the sealed one, and records the missing `tested` entry if a new record exists. The round loop is bounded (`2 * rounds + 2`
  passes, then `did not settle`).
- **Where the count lives.** In the ledger, per attempt: it starts at a `stage_start` that is not a resume and at a moved-aside stage directory. A resume keeps
  it. A resume of an escalated attempt that still has rounds goes on in its stage directory and does not move it aside.
- **What does not change.** The gate, the template's checks, the hash that ties the final record to the supervisor, the finish step (it still records a failed test
  and does not investigate), and the failure path after the last round: a failed gate, then escalation, then a fresh attempt.
- **Cost.** Each round is one agent step and one 10-second to 1-hour test with a lease. The lease is taken again for each round and is not held between rounds: when the coder holds
  the board the test needs, the coder must be up for the repair step, so a lease held across the step would hold a board the coder is parked from.

## 4. Slices

| Slice | Content | Issue |
|---|---|---|
| 1 (this PR) | Opt-in flag, route for class `full-port`, stage 2 skill and gate, pause before stage 3, block code | this PR |
| 2 (built) | Supervisor-owned `port_check.py` template and a port module interface (load, prefill, decode step, close), so the agent writes only model code | #325 |
| 3 | Stage 3 (full model on the configured chips) and stage 4 (shrink) for full ports, using the hardware test list (`hwtests.py`) | #326 |
| 4 | Residency estimator from safetensors headers, used before stage 2 to say how many chips a port needs | #327 |
| 5 | Stage 0: a weight-repack class for models like Sage-A1.5 whose only difference is the tensor layout of experts | #328 |
| 6 | Stages 5 and 6 for full ports: a serving check with a pinned acceptance corpus that the server replays at boot, then quality and benchmark with repeats | #329 |

## 5. Not decided here

- Whether a full port may be published. Nothing in this design publishes. Operator rule stays: never publish without an
  explicit go-ahead.
- Whether the local coder can write a port at all. Slice 1 only makes the route exist and keeps its evidence checkable. The
  first real attempt is an operator-run pilot.
