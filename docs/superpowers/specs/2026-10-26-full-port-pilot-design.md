# Full-port pilot: design and preparation (issue 61)

Date: 2026-10-26. Status: a proposal for the operator. Nothing here ran on a chip. The only code change is the
structured tt-transformers support record in research (section 9). Every decision that is the operator's is listed
as a question in section 8 with a recommendation and a default.

## 1. What the pilot has to show

Issue 61: no real run has taken the full-port stages (2 to 6) on a model. Every finished bring-up took the
weights-only path, and an unattended run that finds a full port ends `needs-new-model-code` at stage 0. The pilot
picks one model that tt-transformers already supports, so the port is mostly configuration, and uses it to find
out which parts of the full-port route work today.

The pilot answers four things (the four questions in the issue): how such a model gets past the `base` preflight
check (section 4), what an unattended run may write (section 5), what each stage measures (section 6) and whether
the local coder can drive the skills (section 7).

## 2. Pick the pilot model

Evidence read from the host tt-metal tree (`~/tt-metal`, tag v0.80.0, read only) and from each model's
`config.json` on the Hub (2026-10-26). The question is which candidate tt-transformers most plausibly serves on
this box's Blackhole boards (two p300 boards, four chips) with no new model code.

| | Llama-3.1-8B-Instruct | Llama-3.2-1B-Instruct | Llama-3.2-3B-Instruct | Qwen2.5-1.5B-Instruct | SmolLM2-135M-Instruct |
|---|---|---|---|---|---|
| `model_params/<name>` directory | yes | yes | yes | no | no |
| README table (verified list) and Blackhole boards named | yes, n150 / p100 / p150 | 1B base model: n150 only | 3B base model: n150 only | extended list only, N150 | no (only the 1.7B, N150) |
| `models/model_targets.yaml` (current CI targets; `PERF.md` is marked deprecated in favour of it) | Wormhole n150, n300; Blackhole `bh_p100`, `bh_p150` (top-1 90, top-5 98, batch 32 decode 42 tokens/s/user); `llama3.1-8b-qb2` on `p300x2` (IFEval 80.5) | Wormhole only (n150 79/97, n300 79/97, llmbox 80/97, galaxy 77/96) | Wormhole only (n150 89/98) | none | none |
| code that special-cases Blackhole boards | yes (`P150`, `P300`, `P150x4`, `P150x8` in `model_config.py`) | a tuning comment and a `P150x4` prefill entry | a `P150x4` prefill entry | none | none |
| reference outputs (`tests/reference_outputs`) | yes | yes | yes | no | no |
| attention heads / KV heads | 32 / 8 | 32 / 8 | 24 / 8 | 12 / 2 | 9 / 3 |
| ungated copy | `unsloth/Llama-3.1-8B-Instruct` | `unsloth/Llama-3.2-1B-Instruct` | `unsloth` copy | `Qwen/Qwen2.5-1.5B-Instruct` | `HuggingFaceTB/SmolLM2-135M-Instruct` |

What decides it:

- Llama-3.1-8B is the only candidate with explicit Blackhole targets: CI accuracy and speed rows for the p100 and
  p150 boards, and a `llama3.1-8b-qb2` entry for `p300x2`, which is this box's board layout. The 1B and 3B models
  have Wormhole targets only; their Blackhole evidence is a tuning comment (the matmul rule "swept on P300
  (Llama-3.2-1B, Llama-3.1-8B, ...)") and a `P150x4` entry in a prefill-size table.
- `tt/attention.py` asserts `n_heads % num_devices == 0` and `n_kv_heads % num_devices == 0` per tensor-parallel
  group. 8 KV heads divide across 1, 2 and 4 chips, so both Llama models fit the configurations a weights-only
  package ships for. Qwen2.5-1.5B (2 KV heads) would fail on 4 chips and SmolLM2-135M (3 KV heads) on 2 and 4.
  Read from the assertions and the configs, not run.
- `determine_device_name` maps one Blackhole chip to `P150`, two to `P300` and four to `P150x4`.
- `get_base_model_name` reads the last part of `HF_MODEL`, so an `unsloth` copy takes the same configuration path
  as the Meta repository.
- SmolLM2-135M is not listed (only the 1.7B is), has no model_params directory and has the 3-KV-head problem. A
  hand-made Blackhole package exists (`stisiTT/smollm2-135m-instruct-p150`), which suggests it needed changes.
- Llama-3.2-3B has an installed Blackhole container package (`stisiTT/llama-3.2-3b-blackhole-v51`). That makes a 3B
  fine-tune a weights-only case through the container route of issue 60, so the 3B does not test the full-port
  route.

Choice: `unsloth/Llama-3.1-8B-Instruct`. It is not the smallest candidate, and the issue asked for a small model. The
reason to pick it: a pilot that fails on a 1B model could be a harness problem or a tt-transformers-on-Blackhole
problem, and the pilot could not tell which. The 8B is the only candidate with Blackhole targets, so a failure there points at the
harness. It is 16.1 GB, fits one p150 board's memory, and its CPU reference is slower but within what stage 1 does
for the 27B. The smallest fallback is `unsloth/Llama-3.2-1B-Instruct`, which costs less time per run and has the
weaker Blackhole evidence (question 1).

What could not be verified:

- That tt-transformers at v0.80.0 serves either Llama model on this box with no change. The `model_targets.yaml`
  Blackhole rows are targets the CI enforces, as far as the file says. Nothing here confirms CI runs them on a p300
  or that the v0.80.0 tag passes them.
- That the host tree's build, its Python environment and the shared virtual environment can run
  `models/tt_transformers` together from orchard. Not tried.
- Whether the standalone `tenstorrent/tt-transformers` repository, which research also matches by name
  (`llama32_1b`), supports Blackhole. It was not read for this note.
- Quality on this box. The published numbers are teacher-forced top-1 and top-5 on 512 tokens, per board type.

## 3. Dry run of the preflight

`python3 -m orchard.cli bringup <model> --dry-run --style plain` (read only, started nothing, 2026-10-26, against
the machine's `bringup.toml`). Both models gave the same shape:

- Passed: hub (8B: license llama3.1, 16.1 GB; 1B: llama3.2, 2.5 GB), disk, tiers, the CPU stand-in, `bus` (4 chips
  in D0), the reference interpreter. Warned: visible credentials (accepted by default), gozer (other clients hold
  chips), the skills clone (1 commit behind upstream, with local changes).
- Blocked on `port`: something listens on the coder port 8011 (`coder-unusable`). That is the machine's running
  coder; a real run starts it itself or finds it free. Unrelated to the model.
- Blocked on `base` (`nearest-model-missing`): "no installed bundle serves meta-llama/Llama-3.1-8B-Instruct, the
  model this one is based on, so stages 0, 2 and 4 have nothing to compare it with or serve it through". The row
  names the existing code: a model_params config for Llama-3.1-8B-Instruct, the README row "n150 / p100 / p150" and
  "The code names a Blackhole board". For the 1B it says the README row names n150 only and "no Blackhole board
  named in that row", with a warning to check that it runs here. Both list the same three weak bundles
  (`episod/tt-tnt`, `episod/tt-tnt-1024`, `stisiTT/llama-3.2-3b-blackhole-v51`).

Two facts from the block matter for the design. The stated base is the gated Meta repository, which orchard cannot
download. And the preflight already finds the tt-transformers support as text, so the information the new base kind
needs exists; no code consumes it as data yet.

Control: `HuggingFaceTB/SmolLM2-135M-Instruct` also blocks on `nearest-model-missing`, finds no exact
tt-transformers match and lists only the standalone repository's `llama32_1b` and `llama32_3b` directories as name
matches (the weak kind of evidence).

## 4. Question (a): how a supported model passes `nearest-model-missing`

Today the `base` check passes only when an installed bundle serves the base, because stages 2 and 4 swap weights into
that bundle. A model that tt-transformers supports has no bundle, and the stage 2 to 6 table that would apply to it
is the upstream "write a new decoder" pipeline, which is far more than the model needs.

Proposal: a new class, `stock-runtime` ("tt-transformers serves this model as it is"), with these parts.

1. Research already finds the exact match. This branch records it as data (`report["tt_transformers"]`, section 9):
   which source matched (`model_params` or `readme`), which id it matched for (the model itself or a stated base),
   the boards the README row names and whether one is Blackhole.
2. Preflight: when no bundle serves the base and the support record has an exact match, the `base` row is a warning
   that names the match, the boards and the Blackhole fact, and the run goes on. It does not need the base snapshot:
   stage 0 compares the model's `config.json` with the `config.json` in `model_params/<name>` (the host tree has it
   for both Llama models), so a gated Meta repository is never downloaded. Default for a Blackhole fact of `False`
   or `None` (the 1B README row): warn and go on, because the stage 2 measurement decides, and a block would stop
   the pilot before the only test that settles it. `--require-blackhole-named` (not built) would turn that into a block.
3. Stage 0 sets the class when the model's decisive config values equal the `model_params` config and its tensor
   names and shapes match the layout `models/tt_transformers` loads. Otherwise it keeps `full-port`.
4. The supervisor treats `stock-runtime` like `weights-only` for the "an unattended run blocks on a full port"
   rule, because no new model code is written.

Not built, because it needs question 2 answered and a stage 2 skill that does not exist yet (section 6). A
preflight that passes a model the later stages cannot take would start a run that fails at stage 0 or 2, which is
worse than the current block.

Default if the operator does not decide: keep the block. The block text already names the tt-transformers code.

## 5. Question (b): what an unattended run may write

Today an agent writes only inside its stage directory (the runner refuses other writes), and the supervisor never
publishes or pushes. The upstream skills write to `models/autoports/<model>/` in a tt-metal checkout, which orchard
agents do not have.

Proposal (default):

- The host tree `~/tt-metal` stays read only. No stage edits, rebuilds or installs there.
- A stock-runtime run needs no tt-metal change. If a stage needs one (a model entry, a config), the agent writes it
  in `<run>/tt-metal-scratch/`, a copy of `models/` (tt_transformers and common are about 12 MB together) that is
  put in front of the host tree on `PYTHONPATH`. The kernels and the built `ttnn` come from the host build. A
  `git diff --no-index` patch of the scratch directory against the host tree goes in the stage evidence, together
  with the host tree's tag and whether it was dirty.
- The upstream autoports pipeline (`models/autoports/<model>`) is not used in the pilot. If a later model needs it,
  the same scratch directory is the place, which keeps the pilot's rule the same.
- No package, no push, no publish. Stage 7 stays skipped (it is for the weights-only path). The operator bundle
  gets the patch, the commit and dirty flag, and the measured numbers, and says in `RISKS.md` that a card built
  from this tree would come from an unpushed tree (the limit the Coder-Next spike already has).
- The runner needs one change before this works: a write target under `tt-metal-scratch/` for stages 2 to 4 of this
  class. The current rule that a path starting with `stages/` is run relative stays.

This is a decision for the operator (question 3). Nothing here was built.

## 6. Question (c): a measured gate per stage

The weights-only path gates on measurements: a CPU reference from stage 1, top-1 agreement, coherence, the prompt
suite, evidence files recomputed by the gate. The stock-runtime route reuses those and adds nothing new to trust.

Stages, for the `stock-runtime` class:

| Stage | What runs | Gate (proposed) | Basis |
|---|---|---|---|
| 0 | triage against `model_params/<name>` | class `stock-runtime`; config values equal; tensors load in the layout tt-transformers reads | the existing template, extended |
| 1 | CPU reference | unchanged (`gate_reference`, with the template stamp) | exists |
| 2 | one chip: serve through tt-transformers with `HF_MODEL` set to the model's snapshot, then compare with the CPU reference | teacher-forced top-1 at least 0.85 and top-5 at least 0.95 over the stage 1 reference text (8B); coherent free run; prompt suite with no blocking finding; the record shows the device name (`P150`) and the tt-metal tag | see below |
| 3 | skipped: stage 2 served the full model | recorded as skipped, as on the weights-only path | same reason as today |
| 4 | the same check on 2 chips (`P300`) and 4 chips (`P150x4`) | each count passes the stage 2 bars; `n_kv_heads` divides the count (read from the config before any chip is used) | `gate_mesh_swap` pattern, one record per count |
| 5, 6 | skipped as on the weights-only path | | |
| 8 | bundle | unchanged, plus the scratch patch and the tt-metal tag | |

Bars. `model_targets.yaml` gives Llama-3.1-8B at 90 percent top-1 and 98 percent top-5 on `bh_p150` (teacher
forced, 512 tokens, against the Hugging Face model). The proposed 0.85 and 0.95 sit 0.05 and 0.03 below that, so a
result at the published level passes and a clear regression fails; 0.85 also equals the weights-only
`SWAP_TOP1_MIN`. For the 1B fallback the published figure is 79 to 80 percent top-1 (Wormhole only), so its
bars would be 0.75 and 0.95 and the 0.85 bar would not apply. All of these are choices from one published source,
and no Blackhole figure exists for the 1B. The first run uses the accuracy optimization mode, because the
performance mode lowers precision on purpose. A control (the same prompts on a second supported model) is issue
15's subject and is not required for the pilot; the report says no control was used. The free-run 32-token
agreement is recorded and not gated, because it was calibrated on a 27B model.

If the operator prefers the full-port table as written (functional decoder, full model, mesh shrink), its gates
are the existing ones: `gate_decoder` (PCC at least 0.995, argmax match), `gate_full_model` (parity and top-1) and
`gate_mesh` (one record per configuration). They check the shape of a result file the agent writes, with evidence
paths, and do not recompute a measurement. Those would have to move to the template-and-record pattern before a run
is trusted, which is a larger piece of work than the stock-runtime skill.

## 7. Question (d): can the local coder drive these skills

Not known. The evidence so far: Qwen3-Coder-Next drove the weights-only stages with templates, and in the kernel
probe `silu_mul` passed the simulator 8 of 8 times and hardware 3 of 3, while `rmsnorm` (0 of 16) and `linear_scan` (0 of 8) never passed the simulator and never
reached a chip.
The upstream pipeline asks for far more (a functional decoder, a full model, reviews by an independent context,
subagents).

Three tests, each with a prediction written now and a pass criterion:

| Test | What | Pass criterion | Prediction |
|---|---|---|---|
| T1 template-first | the coder drives stage 2 of the stock-runtime route with a shipped script: it writes a small config file, runs the script, writes `hw_test.json` and `handoff.json` | 3 of 3 runs at temperature 0 reach a passing gate within 30 turns each; the shipped script's sha256 is unchanged in each | likely passes; the weights-only stages work this way |
| T2 replay | `python3 -m orchard.rolefit` with turns recorded from the upstream skills' first goal | 50 of 50 well-formed calls, no repeated failing call, decode at least 30 tokens per second at 32K | likely passes (same model, same tool format) |
| T3 one layer | the upstream first goal (`functional-decoder`) for one layer of the pilot model, with `models/tt_transformers` hidden from the agent's reading list so it cannot copy the answer | prefill and decode PCC at least 0.995 against the Hugging Face layer within 3 attempts of 60 turns at temperature 0.7, and a reviewer other than the author (the 27B tier) returns a clean pass | likely fails; the kernel probe's two failed ops are the nearest evidence |

T1 and T2 need no new model code and are cheap. T3 needs a chip, a tt-metal scratch tree and a second model for
the review, and it tests the upstream pipeline and not the stock-runtime route. Run T1 and T2 first.

## 8. Questions for the operator

1. Pilot model. Recommendation: `unsloth/Llama-3.1-8B-Instruct` (the only candidate with Blackhole targets, so a
   failure points at the harness), with `unsloth/Llama-3.2-1B-Instruct` as the cheaper fallback and a Llama-3.2-3B
   fine-tune as a control on the weights-only container route. Default: the 8B alone.
2. A new class `stock-runtime` and a `weights-swap-check`-style skill for it (section 4), or keep using the upstream
   five stages. Recommendation: the new class, because the issue says the port should be mostly configuration and
   the upstream pipeline writes new decoder code for a model that needs none. Default: neither is built; the run
   stays blocked.
3. Scratch writes (section 5): `<run>/tt-metal-scratch/` as an overlay on the read-only host tree, patch kept as
   evidence, no push, no package. Recommendation: yes. Default: no writes outside the stage directory.
4. Gate bars (section 6): top-1 at least 0.85 and top-5 at least 0.95 for the 8B, accuracy mode first.
   Recommendation: yes, and say in the report that they are choices from one published source. Default: not
   applicable until a skill exists.
5. Runtime: run tt-transformers from the host v0.80.0 tree and its build, under a gozer lease. The orchard bundles pin
   another ttnn wheel. Recommendation: yes for the pilot, after an import check that opens no device
   (`importlib.util.find_spec`). Default: no hardware use.
6. Run T1 and T2 of section 7 before building anything else. Recommendation: yes.
7. Reviewer for any future T3: the 27B tier, which means stopping the coder first. Recommendation: do not run T3
   until T1 and T2 pass.

## 9. What this branch changes

Research now records tt-transformers support as data, which section 4 needs and no current code reads:

- `orchard/research/ports.py`: `tt_transformers()` fills `data["supported"]` with one row per exact match. A row has
  `name`, `via` (`model_params` or `readme`), `matched` (the first id in the list that matched; the report passes
  the stated bases before the model itself), `boards` (in README order) and `blackhole` (true when a Blackhole
  board is named, false when only other boards are, null when none is named or the match is a model_params
  directory). `collect()` passes it on as `data["tt_transformers"]`.
- `orchard/research/report.py`: the report gains `tt_transformers` (cleaned again, odd rows dropped, an empty list
  when the collector failed or the run is offline) and the Markdown gains a "tt-transformers support" section.
- No preflight, supervisor, stage table or skill change. The `base` check still blocks, as section 3 shows.
- Not read by research: `models/model_targets.yaml`, which is the stronger Blackhole evidence (section 2). Reading
  it from research is a next step once question 2 is decided.

Tests: 9 new in `tests/test_research_ports.py` and 9 in `tests/test_research_report.py`. The mutation table is in the
CLAUDE.md entry for this branch.

## 10. Decisions and what was built next (2026-10-26, same branch, version 0.9.0)

The operator answered the questions in section 8:

1. Pilot: `unsloth/Llama-3.1-8B-Instruct`, the 1B as the fallback.
2. Build the `stock-runtime` class and its skill, wired in so that it is used with no flag when research finds an exact tt-transformers match.
3. Scratch writes in `<run>/tt-metal-scratch/` are allowed. The host tree stays read only. No package, push or publish.
4. The gate bars are approved (0.85 and 0.95 for the 8B, 0.75 and 0.95 for a model of 3 billion parameters or fewer), still labelled as choices in `orchard/defaults.py`.
5. tt-transformers may run from the host v0.80.0 build under a lease.
6. Run T1 and T2 first. 7. The T3 reviewer is decided later; T3 is not started.

Built, all without a chip:

- `orchard/stock_runtime.py`: stage 0 as supervisor code. The model's config is compared with `model_params/<name>/config.json` on the keys that decide
  the architecture (a context length, a dtype and a library version are left out), its tensor names, shapes and dtypes are compared with the Hugging Face
  Llama and Qwen2 layout, and the measurement decides: `stock-runtime` when both agree, `full-port` with the reasons when they do not. On the real files
  (`unsloth/Llama-3.1-8B-Instruct` against the host tree) it answers `stock-runtime`, 8.03 billion parameters, tag v0.80.0, clean tree.
- `orchard/classes.py`, `orchard/stages.py`: the class and the path `stock-runtime`; stage 0 is `STOCK_STAGE_0` when the run's inputs hold `stock_params`;
  stage 2 is `STOCK_STAGE_2` (skill `stock-runtime-check`, gate `gate_stock_runtime`); stages 3 to 7 are skipped with a reason each (stage 4's 2-chip and
  4-chip checks are not built, and the reason says so). `gate_delta_stock` checks stage 0's file; `gate_stock_runtime` re-reads the template's evidence and
  judges it again.
- `orchard/preflight.py`: when no bundle serves the base and research recorded an exact `model_params` match whose directory exists on this box, the `base`
  row is a warning and the run goes on. `orchard/cli.py` passes `--input stock_params=<dir>` to the supervisor. `ORCHARD_TT_METAL_DIR` names the tree; the
  default is `tt-metal` under the operator's home.
- `orchard/agent.py`, `orchard/supervisor.py`: `write_file` reaches `<run>/tt-metal-scratch/` (and nowhere else outside the stage directory) in stages 2 to 4
  of this path.
- `orchard/skills/stock-runtime-check.md` and `orchard/skills/stock-runtime-templates/stock_check.py`: `prepare` copies the host tree's `models/`, the root
  `conftest.py`, `pytest.ini` and `tests/scripts` into the scratch directory, builds a model directory named after the tt-transformers model, checks that the
  interpreter can import numpy, cv2 and torch, makes a reference file if the tree has none, and writes `hw_test.json`; `run` runs tt-transformers' own token
  accuracy test and a free run from the scratch copy and writes `evidence/stock-check.json`, the logs and `evidence/scratch.patch`.
