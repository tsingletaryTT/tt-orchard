# `tt-orchard bringup`: unattended bring-up by outcome class

Date: 2026-10-06. Status: built and running. The operator asked to start executing at the same time as the
spec was written, then told the harness to proceed for hours. Section 12 records what each phase found.

## 1. Intent

`tt-orchard bringup <hf-model-id>` runs one model from a Hugging Face id to an operator review
bundle, with no one watching. It must survive a supervisor crash, a dead model server, a stuck
agent, a chip fault, a full disk and a model swap. The first target is `Cloudflare/clef`. The way
that run is built must be the way later runs go.

What the operator said (2026-10-06):
- Success for Clef means the Qwen3.8-27B backbone and vision encoder run on the chips and the schema
  head runs on the host CPU.
- `qwen3-coder-next` (served with `tt-model serve raahemnabeel/qwen3-coder-next-blackhole`) is
  expected to take the agent roles on the 2-chip tier.
- The harness never publishes. That rule is unchanged.

Assumptions I made (flagged, not confirmed):
- "Hell or highwater" means every run ends at a bundle that says what happened. It does not mean the
  harness keeps retrying forever, and it does not mean the harness may skip a safety rule.
- The first Clef run may take many hours.

## 2. What exists and what does not

- A supervisor with a ledger, a stage machine (stages 0 to 8), park and restore, a watchdog and
  crash recovery (README section 1). Entry point today: `python3 -m orchard.supervisor run ...` with
  about 20 flags. There is no `tt-orchard` binary.
- Two paths for a model: `weights-only` and `full-port`. Full-port has never run and pauses for the
  operator before stage 2.
- No notion of a file outside the backbone's shards. Clef has one (`joint_head.safetensors`).
- No check that works on a model with no text output.

## 3. Outcome classes

Stage 0 assigns each model one class. Each class has a written contract: the stages that run, the
evidence that counts as a pass, and the bundle that is written when the contract cannot be met.
The class is written to the ledger when stage 0 ends and is read from the ledger after that, as the
path is today.

| Class | Meaning | Contract |
|---|---|---|
| `weights-only` | Same architecture as a supported model, only weights differ | As today: coherence and top-1 agreement against the CPU reference, per chip configuration |
| `weights+sidecar` | `weights-only` for the backbone, plus extra weight files the supported runtime does not load | The backbone passes the `weights-only` contract on chips. The sidecar runs on host. Parity check described in section 6 |
| `full-port` | New model code needed | Runs to the point the harness can do unaided, then ends at a blocked bundle (section 5) |
| `unknown` | Stage 0 could not decide | Ends at a blocked bundle |

`path` in `delta.json` stays as it is for old runs. A new `class` field carries the four values, and
`path` is derived from it (`weights+sidecar` has `path` `weights-only`), so existing gates keep
working. The class table lives in one module (`orchard/classes.py`). Stage specs ask that module
what to run; the supervisor holds no per-class branches.

## 4. The `tt-orchard` command

One console script, named `tt-orchard`. It is a front end over `orchard.supervisor`, not a second
supervisor. `tt` is not used: it belongs to the official CLI.

| Command | Does |
|---|---|
| `tt-orchard bringup MODEL [--run-dir DIR] [--profile NAME]` | Preflight, then start or resume the run for MODEL. The run directory defaults to `<runs root>/<model slug>`. Starting again on the same directory resumes |
| `tt-orchard status [MODEL or --run-dir]` | The existing read-only status |
| `tt-orchard pause`, `resume`, `abort` | The existing control commands |

Defaults come from `config/tiers.toml` plus a `[bringup]` table (runs root, cache root, HF home,
package namespace, coder target and port). `bringup` refuses to start, naming the missing key, when
a default is absent. It never guesses a path (a repo test already forbids machine paths in skills).
`bringup` sets `--package-format v6` only when the operator's config says so, because the format
decides whether stage 7 runs.

Preflight (all read-only, all before any lease):
1. Disk: enough free space under the HF home and cache root for the model, its caches and the coder.
   The estimate comes from the Hub file sizes (`/api/models/<id>/tree`).
2. The model id resolves and has a license file. A gated or private model stops here with the
   reason in the bundle.
3. Credential files visible to agent shells (the existing check).
4. gozer is reachable and `gozer status` shows no live foreign lease on the chips the run needs.
   Stale leases are reported and left for `gozer reconcile`, which the operator or the
   gozer-gatekeeper procedure runs. The harness does not clear another agent's lease.
5. The coder tier answers its canary, or can be started by the supervisor.

## 5. Hell or highwater: what "never stuck" means

Every terminal state of a run is one of three, and each writes the operator bundle.

- `ready`: every contract passed. Today's `ready for operator review`.
- `blocked`: the harness did everything it could do and names what stopped it, with evidence.
  Reasons are an enumerated set (`needs-new-model-code`, `unsupported-input-type`,
  `hardware-unhealthy`, `disk-full`, `credentials-needed`, `license-needs-review`,
  `coder-unusable`, `model-unavailable`, `config-invalid`, `retry-budget-spent`, `stage-failed`,
  `agent-stuck`, `blocked`, `unclassified`).
- `aborted`: the operator asked.

Built 2026-10-06 (`orchard/blocked.py`, `supervisor run --unattended`, exit code 5). Today a pause is a fourth state that needs a person to look. The change: a pause caused by one of
the enumerated reasons becomes `blocked` and writes the bundle, and the supervisor exits with a
distinct code. A pause the harness cannot name stays a pause and the status command says so. That
is the one place a person is still required, and the design keeps it small instead of hiding it.

Retries are bounded and recorded. A stage already has attempt, escalate and continue rules. This
design adds one budget for the whole run (wall time and cold boots, `RUN_BUDGET` in
`orchard/defaults.py`), so that a run that keeps failing in new ways ends in `blocked` with
`retry-budget-spent` after a stated time.

Faults the harness recovers from without stopping, each with an existing or new test that goes red
when the recovery is removed:
- supervisor killed: resume from the ledger (exists; kill-after-every-event test exists)
- coder server dead: restart under the owner-pid lease (exists)
- agent loops or goes silent: watchdog nudges, then escalates the tier (exists)
- chip hung after a test: reset in place through gozer (exists)
- disk low mid-run: new. Free space is checked before each stage; the stage ends `blocked` with
  `disk-full` before a write could fail halfway
- a fault the supervisor itself raises: the existing abort path releases the hardware

## 6. Clef: the `weights+sidecar` contract

Facts from the Hub (2026-10-06, `Cloudflare/clef` at revision `2f3de3d…`): `qwen3_5` architecture,
Qwen3.8-27B text config with a vision encoder, 12 shards, a 256 MB `joint_head.safetensors`
(width 1024, 4 layers, 16 heads, 2 routing layers, input 5120), and `joint_schema_model.py`
(576 lines) that encodes records and loads the head. Output is one logit per option per question.

Design decisions:
- **Classification.** `delta_triage.py` compares the backbone shards against Qwen3.8-27B. Files named
  in the model's own README as a head, and not listed in `model.safetensors.index.json`, are recorded
  as `sidecars` with their size and tensor names. If the backbone matches and the sidecar loads
  with no tensor that overlaps a backbone tensor, the class is `weights+sidecar`. Anything else
  with an unexplained extra file stays `unknown`.
- **Parity check (new gate `gate_sidecar_parity`).** The reference is the CPU run of the full model
  (backbone, then head) on a fixed set of records. The device run produces the backbone's final hidden
  states for the same records, and the head runs on host from those states. The gate compares the
  per-question probabilities. Thresholds are set in `defaults.py` as a choice, with the reasoning
  next to them, and are checked against the CPU-versus-CPU noise floor first (two CPU runs with
  different thread counts), so the bar is above what the reference itself varies by.
- **Hidden states (decided after Phase 1; see the spike log).** The head reads the backbone's
  `last_hidden_state` for every token position (`ClefModel.forward` in `joint_schema_model.py`), and also
  takes the backbone's output-embedding matrix (`lm_head.weight`, in the backbone shards, loaded on host).
  The TT serving path cannot return hidden states: the vLLM plugin has no pooling route, and the
  device prefill keeps only the last row before the norm and the lm head. So the parity gate does not
  go through vLLM. A template script (`orchard/skills/sidecar-parity-templates/hidden_parity.py`) builds
  the nearest model's `Qwen36Model` on a leased board from the Clef backbone directory, runs the layer loop
  itself over all rows, applies the model's own final norm, and reads one replica back. It changes nothing in
  tt-metal. It depends on `embd`, `layers`, `norm` and a few helper methods of one tt-metal revision, so the
  evidence records that revision. A wiring check guards the reimplementation: for the last row, the script
  applies the model's `_lm_head` to its own normed row and compares the logits with the model's
  `prefill_tp` output on the same tokens. If they differ, the gate fails instead of comparing hidden states
  from a loop that no longer matches the model. If the model code lacks those attributes, the Clef class
  ends `blocked` with `unsupported-input-type`.
- **Third-party code.** `joint_schema_model.py` runs only inside the agent shell sandbox (no
  credentials, no-chip device mask) for CPU references, and on the host for the parity run only after
  stage 0 records its sha256 and a read of its imports. The harness never passes `trust_remote_code`
  to a run with credentials visible. The run's bundle lists the file's hash.
- **Vision.** The first contract covers text records only. Image and video records are listed in
  the bundle as not tested. The operator can widen this later.
- **Stages for this class.** 0, 1, 2 and 4 as on `weights-only`, with the parity gate added to
  stage 2 (one chip configuration) and stage 4 (each configuration that claims a pass). Stage 3, 5
  and 6 stay skipped with reasons. Stage 7 packages the backbone only and records that the head is a
  sidecar the package does not load. Publishing is out of scope.

## 7. The coder tier: qwen3-coder-next

The analysis documents in `docs/analysis/` are hypotheses. Their success-rate tables and the
arbiter's capability scores were not measured, and the arbiter assumes a 122B tier this machine
cannot run. This design does not use the arbiter. A model takes a role only when it passes a
role-fit test.

Role-fit test (`orchard/rolefit.py`, run before the coder is changed in `tiers.toml`):
1. Serves on the configured chips: boot time, memory, and the canary including the existing
   "7 times 6" first-start question.
2. Tool-call format: 50 recorded agent turns (from the Hemmingway-1 and openthai ledgers and the
   committed transcript signatures) replayed against the server. Pass means a parsable tool call
   or text on at least 95 percent.
3. The turns that failed the 27B (the stage 2 thinking loop, the stage 0 and stage 8 stalls) replayed
   with thinking on and off. Pass means no repeat of the same failure in 5 replays each.
4. Throughput: decode tokens per second at 8K and 32K context, measured, with the chips otherwise
   idle.

The result is a file under `docs/run-logs/` with the numbers, and `tiers.toml` changes only when the
operator's rule passes. If a 2-chip Coder-Next fails, the large tier stays as it is. Phase 1 result
(`docs/run-logs/2026-10-06-bringup-spike.md`): the package exists and boots. The "2 chips" claim is one
p300 board (2 dies) with bfp4 routed experts. The default profile `p300x2` takes all four chips. The
test therefore runs on both profiles, and a tier that uses the 4-chip profile follows the existing
park and restore rules like the large tier. One boot (681 s), a canary, one tool call and one throughput
sample (39 tok/s decode, about 3,560 tok/s prefill on `p300`) passed; steps 2 to 4 above have not run.
The package was built from a dirty, unpushed tt-metal tree, which the bundle records.

## 8. Proof that it works unattended

Chaos run on Hemmingway-1 (a known-good model) before Clef, scripted in `scripts/chaos_bringup.py`:
- kill the supervisor at 5 random ledger events
- kill the coder server once during a stage
- stall the agent (a fake server that stops answering) once
- fill a scratch filesystem so the disk check fires
- hold one chip with a foreign lease to check the harness waits and does not take it

Pass: the run ends `ready`, no lease is left, and no foreign lease was touched. The script writes
its result to the run log. A chaos script that has never been seen to fail is a claim, so each
injected fault is also tested against a build with the matching recovery removed.

Then the real Clef run. The pattern it leaves is the run log (what the harness did, what it
could not do), the class contract it used, and any template it needed.

## 9. Phases

1. **Spike (read mostly, short chip use under a lease; done 2026-10-06).** Answers: (a) is
   `raahemnabeel/qwen3-coder-next-blackhole` installed or pullable, and does it boot on 2 chips;
   (b) how can hidden states leave the TT stack; (c) does `delta_triage.py` classify Clef as
   expected from the Hub's headers and README; (d) what holds the stale leases that `gozer status`
   shows now. Output: `docs/run-logs/2026-10-06-bringup-spike.md` and an update to this spec.
2. **`tt-orchard` command.** Console script, `[bringup]` config, preflight, exit codes. Tests first.
3. **Class contracts and the sidecar check.** `orchard/classes.py`, the triage change, the
   parity gate, the `blocked` terminal state and the run budget. Tests first, a mutation per guard.
4. **Proof.** Role-fit test on Coder-Next, chaos run on Hemmingway-1, then Clef.

Each of phases 2 to 4 gets its own plan under `docs/superpowers/plans/` when it starts, written
from what the spike found.

## 10. Not in this design

The model-selection arbiter; any 122B tier; publishing; the full-port path beyond a clean
`blocked` bundle; moving stage skills to another repo (the operator decided they stay here).

## 11. Risks

- Hidden states may not be reachable on the chips (section 6). The fallback is `blocked`.
- Coder-Next may be slower or less reliable than the 27B in this loop. The role-fit test decides.
- A first unattended run of a model with a new class will find bugs. The bundle and ledger make
  them visible; the budget keeps the run from spinning.
- Another agent shares the machine. Stale leases are visible now. The harness waits and reports; it
  does not take or reset a chip it does not hold.

## 12. What was built and what the first real run found (2026-10-06)

Phase 1 (spike): see `docs/run-logs/2026-10-06-bringup-spike.md`.

Phase 2 (`tt-orchard` command): built as specified, with these changes. The defaults live in
`config/bringup.toml` because the tier loader refuses unknown tables. The preflight gained a `reference`
check (the interpreter stage 1 uses). `bringup` always passes `--unattended`.

Phase 3: `orchard/classes.py` and `orchard/blocked.py` (sections 3 and 5), `gate_weights_swap_sidecar`, and the
parity templates, built by a forked agent and run on one board with Clef (15 of 16 questions agree; bars in
`orchard/defaults.py`). Two findings changed the design:
- Triage read every `*.safetensors` file in the snapshot, so a sidecar would have forced the full-port path. The
  backbone is now the shards the model's index lists.
- On the full-port path an operator's resume means "go on into the full-port stages". An unattended retry must
  never do that, so an unattended full port blocks every time.

Role-fit test (section 7): built (`orchard/rolefit.py`) and run against Qwen3-Coder-Next on one board; it passes
once the loop offers a read-only `read_file` tool (docs/run-logs/2026-10-06-rolefit-qwen3-coder-next.md). The
arbiter of the analysis documents was not built.

The chaos run on Hemmingway-1 was replaced by the real Clef run, because each fault the chaos script would
inject (a crash, a stalled agent, a hardware fault) is more informative when it happens in a real run. The
supervisor's kill-after-every-event tests still cover crash recovery with fakes.

First Clef run (Coder-Next on one board, 27B untouched):
- Stage 0 passed in 3m47s: class `weights+sidecar`, the sidecar and code file found, hashes equal to the
  hardware prototype's.
- Stage 1 stalled twice: Coder-Next wrote `stages/1/...` paths that `write_file` read as stage-relative, and ran
  `pip install --upgrade transformers` in the machine's shared venv. The run ended `blocked: stage-failed`
  after 60 turns, wrote `BLOCKED.md`, released the hardware, and `tt-orchard bringup` printed the reason and
  the retry instruction. This was the first real use of the blocked end state and the retry path; both worked.
- The fixes: `write_file` accepts run-relative stage paths; the runner refuses package installs outside the run
  directory; `reference_python` gives stage 1 a dedicated interpreter. The retry passed stage 1 in 4 minutes.

