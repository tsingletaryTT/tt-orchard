# Kernel attempt: an op inventory and a tt-lang kernel probe

Date: 2026-10-15. Status: implemented on branch kernel-attempt; section 11 records what changed during implementation. Branch: `kernel-attempt`.

Section 11 amends sections 4.3, 5.1, 5.2, 5.3, 5.4 and 7; where they differ, section 11 wins.

## 1. Purpose and scope

Operator prompt: "What would it take for tt-orchard to take the initiative with models it can't find an
equivalent for and attempt to write the kernels (in tt-lang for example, using the present skills) instead?"
The operator then approved the first two steps, asked that this stay in step with the tt-model-manager branches,
and said provenance metadata stays on the orchard side. Nothing goes upstream; any fix to tt-lang or
tt-model-manager is local.

This spec covers two parts and the environment they need:

- **Part A, op inventory.** For a model with no nearest bundle, list the operations its reference code uses and say
  which ones ttnn already offers. No hardware, no model.
- **Part B, kernel probe.** A command that measures whether the local coder can write a tt-lang kernel that passes
  a check, on operations whose right answer is known. Simulator first, one chip second.
- **Environment.** A reproducible tt-lang simulator environment that includes a local fix for a packaging bug.

Out of scope for this branch: kernels for a real model, stage-machine changes for kernel work (a later step, "C"),
the `generic_op` export, any change to tt-model-manager or tt-lang, publishing.

## 2. Facts this design rests on

Each fact names where it was read. "Not run" means the source was read and nothing was executed.

1. **A full-port route exists and has never run.** `orchard/classes.py` maps `full-port` and `unknown` to the
   `full-port` path. Stages 3, 5 and 6 are its stages. An unattended retry of a full port always blocks (log,
   2026-10-06).
2. **tt-model-manager can ship custom code in a v6 thin bundle.** The installed `tt-model package-thin` takes
   `--extra-code` (a pure-Python tree) and `--ops-wheel` (a `generic_op` wheel). The unmerged branch
   `docs/v6-canonical-thin-path` names this the canonical shape: stock ttnn, the model's code in `extra_code/`,
   custom kernels only as `generic_op` wheels. A new registered C++ ttnn op stays on v5 or v5.1, which orchard
   refuses. (`feat/v6-extra-code-127` is not an ancestor of `main` by SHA, but `main` contains the feature.)
3. **A tt-lang kernel has a route to `generic_op`.** tt-lang's `ttl-export` command exports a kernel as TT-Metal C++
   kernels plus a Python runner that calls `ttnn.generic_op`. Not run. Its tools assume an SSH and docker remote,
   so orchard needs its own local version (out of scope here).
4. **ttnn op names cannot be read without importing it.** In the shared venv's installed ttnn there are no `.py` or
   `.pyi` files naming `rms_norm`, `softmax`, `silu` or `add`. Importing ttnn opens the device (global CLAUDE.md).
   The coverage list therefore comes from a recorded dump taken under a lease (section 4.3).
5. **The tt-lang simulator wheel has a packaging bug.** `ttl/sim/diagnostics.py` loads
   `sim_dir.parent / "ttl" / "diagnostics.py"`, a source-tree path. The wheel does not ship that file, so a
   diagnosed kernel error ends as `FileNotFoundError`. Present in `tt-lang-sim` 1.1.6 (PyPI) and in a wheel built
   from main at `9b7ad24` (v1.1.10 plus 23 commits). On upstream's four `examples/errors/*` cases the exit code was
   non-zero before and after the fix, so the exit code cannot tell the bug from correct behaviour. After copying the
   source `python/ttl/diagnostics.py` (stdlib imports only) into `site-packages/ttl/ttl/`, the simulator printed
   the shape mismatch, both copy-lock errors and the deadlock, and `compiler_only_external_call.py` failed with
   both strings upstream's test expects.
6. **The reference venv's transformers (5.19.0) already ships** moshi, rwkv, mamba, mamba2, kyutai_speech_to_text,
   whisper, sam and yolos, so refusing remote code (section 4.2) excludes none of the first candidates.
7. **Versions on this box (2026-10-15).** Shared venv: `tt-lang 1.1.6` (latest on PyPI, published by Tenstorrent AI
   ULC), `ttnn 0.65.1rc17.dev6200`. Whether that tt-lang and that ttnn run a kernel together has not been tested
   (section 5.4 starts with that test).

## 3. Environment: `orchard/ttlang_env.py`

`tt-orchard kernel-env [--check]` builds or checks the simulator environment. It never touches the shared venv.

Steps, each skipped when already true:

1. Shallow clone of `https://github.com/tenstorrent/tt-lang.git` into `<orchard-src>/tt-lang` at the pinned SHA
   (`defaults.TTLANG_PIN`, first value `9b7ad24d1…`), with no submodules. An existing directory at another SHA is
   reported, never reset.
2. A venv (`<venvs>/ttlang`, Python 3.12) with CPU torch from the PyTorch CPU index, installed first as
   `setup_machine.py` does for the reference venv.
3. The `tt-lang-sim` wheel built from `packaging/sim` with `TTLANG_VERSION_OVERRIDE=<tag>+g<sha>`, installed.
4. The diagnostics fix: copy the clone's `python/ttl/diagnostics.py` to `site-packages/ttl/ttl/diagnostics.py`,
   recording the sha256 of source and copy. The copy is a local patch of a vendor package and is labelled that way
   in the output and in every probe report.
5. Verification, which is the instrument check for the whole probe: run upstream's `examples/errors/eltwise_add_error.py`
   and require the text `does not match Block shape`, and run `compiler_only_external_call.py` and require both
   `ttl.call_extern_func is not supported by the Python simulator` and `use tt-lang-sim --backend=emule`. A bare
   non-zero exit does not count.

`--check` runs only step 5 plus a version read and exits 0 or 2. Paths come from `bringup.toml` keys with defaults
under the operator's home, as the reference venv does. The command refuses `pip` into any other interpreter.

## 4. Part A: op inventory

### 4.1 Pieces

- `orchard/skills/ops-inventory-templates/ops_inventory.py`: runs in `reference_python` (needs torch and
  transformers). Reads `config.json`, builds the model on the `meta` device with the layer count reduced
  (default 2), runs `torch.export` on a prefill-sized input (sequence length 8), and writes `raw-ops.json`:
  every aten op with its count, one sample of input shapes and dtypes, and the module paths that call it.
- `orchard/opinventory.py`: standard library only. Loads `raw-ops.json`, assigns statuses, scales counts by the
  real layer count, adds the comparison with a nearest model, and writes `ops-inventory.json` and `.md`.
- `orchard/data/aten_to_ttnn.json` and `orchard/data/decompositions.json`: hand-kept tables with a test per entry
  that the aten name is real (checked against the raw ops of the fixture).
- `tt-orchard inventory MODEL_DIR [--against DIR] [--layers N] [--out DIR]` and
  `tt-orchard inventory refresh-ttnn` (section 4.3).

### 4.2 What it will and will not load

The architecture named in `config.json` must resolve to a class in the installed transformers, and `config.json`
must have no `auto_map`. Anything else ends as `skipped: needs remote code` with exit 0 and an inventory that
holds only that reason. A repo's own `modeling_*.py` is never imported (the same rule the triage template
follows for `.py` files). An opt-in sandboxed mode is a later decision.

A `torch.export` failure (data-dependent control flow, an op with no meta kernel) is recorded per module with
the error text. There is no fallback tracer. The report says how many modules traced and how many did not, so a
reader can see when an inventory is partial.

### 4.3 Statuses and the ttnn dump

(Superseded in part: see section 11, item 8 for a fifth status, `ignored`, and item 10 for what the dump lists.)

`tt-orchard inventory refresh-ttnn` runs `import ttnn; dir(...)` for the public op namespaces under
`gozer run --chips 1 --who "claude:orchard-inventory"`, and writes `orchard/data/ttnn_ops-<ttnn version>.json`
with the version, the dump's sha256 and the names. It is a command the operator runs; the supervisor never
calls it.

| Status | Meaning |
| --- | --- |
| `mapped` | the mapping table names a ttnn op and the dump contains that name |
| `composable` | a decomposition exists and every leaf is `mapped` |
| `unmapped` | neither |
| `unchecked` | no dump for the installed ttnn, so nothing is claimed |

`mapped` means "the name exists in the dump of ttnn X". It does not say the op supports the shape, dtype or layout
the model needs. The report states this in its header, and every row carries the dump's version.

### 4.4 Output

`ops-inventory.json` holds: schema version, model id and architecture, transformers and torch versions, what was
traced (`prefill`, layers traced, layers in the real config, sequence length), the dump used or `null`, the ops
(aten name, count, scaled count, shapes sample, dtypes, modules, status, ttnn name, decomposition leaves),
a summary by status, the trace failures, and a `label` block saying which fields are measured and which are
tables. With `--against`, each op also says whether the nearest model's inventory has it.

Decode-time structure (KV cache update, recurrent state update) is not traced in this version. For RWKV and
Mamba, the slow reference path unrolls the recurrence over the 8 input positions, so the scan's ops appear, and
the report says `traced: prefill, recurrence unrolled over 8 positions`.

### 4.5 Stage 0 hook (last task, additive)

When `reference_python` is set, stage 0's skill runs the template after the triage draft and records the
summary under `inventory` in `delta.json`. No gate changes. When the inventory has `unmapped` ops and stage 0's
measured class is `weights-only`, the finding says so; the measured class still decides, as with the base
research.

## 5. Part B: kernel probe

### 5.1 Command and layout

`tt-orchard kernel-probe --op NAME [--attempts 5] [--phase sim|hw] [--out DIR]`. It is a command with no
stage. Each attempt gets `DIR/<op>/attempt-<n>/` with `kernel.py` (the only file the agent writes),
`check.json`, the agent log, and a copy of the spec. `DIR/kernel-probe-report.json` and `.md` summarise.

Ops (`orchard/skills/kernel-loop-templates/ops/<op>.json`), easiest first (superseded: see section 11, item 1, for a fourth op, `add`):

| Op | Why |
| --- | --- |
| `silu_mul` | elementwise, two inputs; the simplest fused op |
| `rmsnorm` | a reduction plus elementwise; ttnn has the op, so the answer is known |
| `linear_scan` | a first-order recurrence over a sequence (`h[t] = a[t]*h[t-1] + b[t]`), the shape RWKV and Mamba need |

Each spec holds shapes (tile aligned), dtype, the reference function (a name in a fixed table in the template's
`references.py`; the spec cannot name arbitrary code), the correlation bars and the input sets. Inputs are
seeded random values plus a heavy-tailed set (a few scaled outliers), because a norm on unit-normal inputs is
easy. Activations captured from a real model are a later addition and the report says the sets are synthetic.

### 5.2 The check

(Superseded in part: the hash check below is replaced by section 11, item 2, and the PCC-only comparison by item 4.)

`kernel_check.py` (template, run by the agent as a shell command) imports `kernel.py`, runs it on the simulator
through the ttlang venv's `tt-lang-sim`, compares with the torch reference by Pearson correlation (PCC) and max
absolute error, and writes `check.json`. `check.json` records the sha256 of `kernel_check.py` and the spec. The
probe command refuses a `check.json` whose hashes differ from the shipped template's, because the agent's shell
can write any file in its directory. Failure categories come from the simulator's own diagnostic text, which is
valid only because section 3 step 5 passed:

`no_kernel`, `import_error`, `tt_lang_error` (with the diagnostic's first line), `shape_mismatch`,
`numeric_mismatch`, `timeout`, `compile_error`, `hang`, `hw_numeric_mismatch`.

Provisional bars (to be replaced by the first measurements, and labelled provisional in every report):
simulator PCC at least 0.999; hardware PCC at least 0.99 in bf16.

### 5.3 The agent

`AgentStep` on the configured coder tier with a new skill, `orchard/skills/kernel-probe.md`. The skill says:
write `kernel.py` only, run `kernel_check.py`, read the first diagnostic line, change one thing, never read the
ledger or transcripts. The runner's rules apply unchanged (no `pip`, no gozer writes, no resets, no uploads).
Allowed new commands: `kernel_check.py` and `tt-lang-sim` from the ttlang venv. Budget per attempt: 40 turns,
30 minutes wall time, 300 s per simulator run (all provisional).

Attempts are independent draws from a server that may not accept a seed. The report says so and gives counts and no
rates or intervals. (Corrected by amendment 27: at temperature 0, the default, attempts can be one sample.)

### 5.4 Hardware phase

Before the first hardware attempt the probe runs a smoke test: upstream's `examples/eltwise_add.py` with the shared
venv's tt-lang and ttnn, under `gozer run --chips 1 --exact 1 --who "claude:orchard-kernel-probe"`. If it fails,
the report says `hw: unavailable` with the output's last lines, and the probe ends after the simulator phase.
Each hardware run has a 600 s deadline and a process-group kill. A hang ends the probe with a `blocked` outcome
that names the attempt. The probe never resets a chip; the lease release does, as everywhere else in orchard.

### 5.5 Report

Per op and attempt: simulator pass, hardware pass or `null`, PCC values, turns, wall time, failure category and
first diagnostic line. Per op: attempts to first pass. The header lists the tt-lang versions, the source SHA, the
diagnostics patch hashes, the coder's model id and the provisional bars.

## 6. Provenance metadata (orchard side)

Each op in an inventory, and each probe attempt, carries a `provenance` block: `origin` (`existing`, `composed` or
`kernel`), evidence file sha256 values, tt-lang version and source SHA, whether the diagnostics patch was applied,
simulator and hardware PCC, the coder's model id, and the run or probe id. Stage 8's bundle shows it in
`RESULTS.md`. Nothing is written into tt-model-manager's manifest.

## 7. Tests and instrument checks

Test first, one commit per change, a mutation per guard (the repo's practice). The checks that matter most:

- **Environment.** A fake machine with every call logged, as in `test_setup_machine`. Mutations: no diagnostics
  copy, a copy of the wrong file, an exit-code-only verification. The first mutation must make the verification
  fail.
- **Inventory.** The fixture is a recorded `raw-ops.json` from a real trace of an installed transformers
  architecture (no hand-made list), with its hash noted, so the suite needs no torch. An opt-in live test
  (`ORCHARD_LIVE_INVENTORY=1`) re-traces. Mutations: remote code accepted, `unchecked` reported as `mapped`, a dump
  for a different ttnn version accepted, a decomposition with an unmapped leaf counted as `composable`, layer
  scaling off.
- **Probe.** For each op, a deliberately wrong kernel fixture (a norm with the wrong epsilon, a scan with the
  wrong carry) must come back `numeric_mismatch`; a test that has not been seen to fail proves nothing. Mutations:
  template hash not checked (superseded: see section 11, item 2; the probe now re-runs the check itself), category taken from the exit code, hardware phase without a lease (a tripwire gozer
  fails any call), no smoke test, no deadline.
- **No network and no device.** Inventory and probe tests run with sockets blocked and with the no-chip device
  mask.

## 8. Risks and limits

- The simulator may support less than the compiler (tt-lang's functionality matrix says which), so a simulator
  pass does not promise a hardware pass.
- The shared venv's tt-lang 1.1.6 and ttnn 0.65.1rc17 may not work together; section 5.4 finds out.
- The local coder has not been shown to write tt-lang. Part B exists to measure that.
- `mapped` is a name match. The inventory can understate what is missing.
- Prefill-only tracing leaves out cache and state updates.
- Synthetic inputs can pass a kernel that fails on real activations.
- A v6 bundle cannot carry a new registered C++ op.
- The diagnostics patch edits a vendor package in our venv. A tt-lang release that ships the file makes the
  patch redundant; step 5 of the environment check still passes then.

## 9. Success criteria and what comes next

**A:** `tt-orchard inventory` runs, with no hardware, on Moshi, RWKV and Mamba2 from the installed transformers,
produces every field in section 4.4 in minutes, and the Qwen3 fixture comes out with no `unmapped` ops once a
dump exists.

**B:** `tt-orchard kernel-probe` produces a table of attempts per op, on the simulator at least, and on one chip if
the smoke test passes.

**C (not in this branch):** wire a kernel loop into the full-port stages. The operator decides from B's table.
A proposal for the go/no-go: at least 3 of 5 attempts pass on `silu_mul` and `rmsnorm` in the simulator and at
least 1 of 5 on `linear_scan`.

## 10. Appendix: gaps in tt-model-manager's metadata (for the operator to raise)

Read from `docs/thin_packages.md` on `origin/docs/v6-canonical-thin-path` (unmerged) and `src/tt_kernel/manifest.py`.

- `deps.custom_ops` lists only wheels passed with `--ops-wheel` whose ops `model.py` imports statically.
- Pure-Python ops in `extra_code/`, ops called by name through `ttnn.generic_op`, and ops imported from a helper
  module are shipped and not recorded.
- The manifest has no field for how an op was made (composed or written), by what, or what it was checked against.
- The `generic_op` route for a tt-lang kernel (fact 3) is not mentioned in tt-model-manager's docs; a search of `origin/main` and the docs branch
  found no mention of tt-lang.
- Unmerged branches seen on 2026-10-15 that touch this: `docs/v6-canonical-thin-path`, `feat/v6-extra-code-127`,
  `fix/v6-reproducible-install`, `feat/auxiliary-weights`, `raahem/renumber-scoped-devices`.

## 11. Amendments (2026-10-15 and 2026-10-16, during implementation and after the final review)

Each item changes or adds to the approved design above. Reasons come from the implementation ledger and the task reviews.

1. **A fourth probe op, `add`.** It is a calibration op. Its working kernel is upstream's example, so a fault in the harness or
   the environment shows up as the failure of a kernel that must pass.
2. **Section 5.2's hash check on the agent's `check.json` is replaced by a stronger rule.** The probe re-runs the shipped check
   on a probe-owned clean copy of `kernel.py` under `DIR/.verdict/<op>-<n>/`. It records `agent_check_untrusted` when the agent's
   own result disagrees. `kernel.py` is copied only as a regular file of at most 256 KiB, opened with `O_NOFOLLOW`. Reads of
   files the agent can write are capped. A sha256 re-check of the copy comes before the gozer call, so the file that was gated is
   the file that runs on hardware. The report is built from the records the run produced; a resume accepts only `<op>-<int>`
   records with the same probe id. The boundary is best effort against shortcuts and accidents. It does not stop a determined
   attacker. `load_attempts` reads records back for a resume; no command calls it yet. Reason: the first design read its
   verdict from a directory the agent could write, and the review showed a planted symlink or a surviving agent process could
   forge a pass.
3. **Two failure categories were added: `harness_error` and `forbidden_code`.** `harness_error` covers a fault in the templates or
   the environment, including the harness `setup` phase (device open, input building, the reference). `forbidden_code` covers a
   `kernel.py` that the static gate refuses, and one that is a link, a device or larger than 256 KiB.
4. **The check is stronger than PCC alone.** PCC does not change when the output is scaled, so a kernel that returned the wrong
   magnitude passed. The check now also needs a per-set relative L2 error, a least-squares slope and a max-abs-error bar. All
   bars are provisional and live in the op specs. A missing field fails. A best-effort static `ast` gate on `kernel.py` runs
   first. It allows the imports `ttl`, `ttnn` and `math`, refuses `import torch` and host code, requires `OPERATION` to be a plain
   top-level `def` decorated only with `@ttl.operation`, and accepts aliases. It reports every violation with a remedy.
5. **The export call** is `torch.export.export(model, (), {"input_ids": ids, "use_cache": False}, strict=False)`. It was measured
   on 2026-10-15 to trace Qwen3, RWKV, Mamba2 and Moshi's text model. Without `use_cache=False`, three of the four failed. A
   dispatch-level recorder was tried and fails on Moshi.
6. **`orchard/stallwatch.py`** (operator instruction, 2026-10-15: "Instead of pure timers, we need watchers"). Every long child
   of `kernel-probe` (the simulator check, the smoke test and the hardware runs) runs under a watcher that stops it on a stall (no output and no CPU) or on a repeating output line. The stop sends
   SIGTERM to the whole process group first, so `gozer run` can release its lease, and SIGKILL after a grace period. Deadlines
   are a backstop. The budgets in sections 5.3 and 5.4 are backstops. Only `kernel-probe` uses the watcher so far; see the log in
   CLAUDE.md for the places that still use timers.
7. **`scaled_count`** is `count_in_layers x (layers in config / layers traced) + count outside layers`, rounded with `round()`. The report says when a
   count is not scaled. Counts are per static node of the traced graph, so a loop body that runs many times counts once.
8. **A fifth inventory status, `ignored`**, for metadata no-ops. `orchard/data/ignored_ops.json` holds exactly
   `aten._assert_tensor_metadata.default`, `aten.alias.default` and `aten.contiguous.default`. A ruling also named `detach` and
   `lift_fresh_copy`; they were not shipped. Section 4.3 listed four statuses. Without `ignored` the unmapped list was mostly noise.
9. **The inventory traverses higher-order-op bodies.** Python-level nodes (for example `getitem`) stay out of the op list and are
   recorded under `python_nodes`. Higher-order wrappers are recorded under `higher_order_nodes`. Both count nodes (a
   `cond` node has two bodies and counts once). Before this, the rotary embedding's `cos`, `sin` and `to.device` were missed.
10. **The ttnn dump** lists public non-class callables at the top level and one level into `experimental` and `transformer`
    (named `experimental.x`). A real dump for ttnn 0.65.1rc17.dev6200 (821 names) is committed. `mapped` is a name match only:
    `rsub` is exact only for alpha=1, and `conv1d` takes prepared weights.
11. **`run_inventory`'s `data_dir` parameter is now `dump_dir`.** The tables are always the packaged ones.
12. **`tt-orchard kernel-probe` exit codes** are 0 (done), 2 (refused or bad input), 3 (crash), 4 (interrupt) and 5 (hardware
    hang). The report is written whatever happens, from the records so far, and no hardware phase starts after a crash.
13. **`kernel-env` completes an interrupted clone.** A `.git` with no HEAD and the right origin is finished (the remote exists, so only fetch and
    checkout run again). A clone at another commit is never reset. The plan's code deleted the clone instead, which blocked every retry
    after one failed fetch.
14. **The stage 0 hook is step 9 of the delta-triage skill**, after `delta.json` is final, because step 8 rewrites the whole file.
15. **Process kills.** The kill changed from SIGKILL to SIGTERM first, then SIGKILL after a grace period (item 6). Before the
    first SIGTERM the stop checks that the process group id is above 1, is not the caller's own group, and that the child leads
    the group. The SIGKILL after the grace checks the id (above 1), the caller's own group and that the group still has members.
    It does not check leadership, on purpose: the leader may already be reaped. If anything raises during the grace period, the
    SIGKILL still goes out.
16. **Success criterion A (section 9) is not met in full.** It asked for no `unmapped` ops on the Qwen3 fixture once a dump exists.
    The final run leaves 6 unmapped: `__and__`, `index`, `to.dtype_layout`, `diff`, `new_ones` and `to.device`. They stay unmapped
    because their ttnn semantics are not provably identical to the aten ops. Mapping them is a table decision for the operator or a
    later task.
17. **gozer gets a 120 s SIGTERM grace.** `gozer run` releases its lease in a `finally` after its child exits, and the release
    resets the chip, which takes about 42 s on this box. The first 5 s grace killed gozer during that reset. Both gozer calls of the
    probe now pass `GOZER_TERM_GRACE_S` = 120 s; the simulator check keeps 5 s. gozer is still killed at once by a second interrupt
    during the grace, and then the lease can stay behind.
18. **A sweep for stray agent processes.** When the agent step ends (also after a crash or an interrupt) and before any verdict, the
    probe stops every process of the operator's whose environment has `ORCHARD_RUN_DIR` exactly equal to the attempt directory:
    SIGTERM, up to 2 s, a second scan, then SIGKILL. It never signals itself or an ancestor. The record lists each process with
    `stopped`, which is true only when the process is gone (no `/proc` entry, a zombie, or another process under the pid); the
    report counts the others as not confirmed stopped. It cannot see a process whose `/proc/<pid>/environ` no longer holds the
    value (an exec with a new environment, or a program that overwrote that memory; `clearenv()` alone does not change it) or
    one whose `/proc` entry is not readable (a non-dumpable process).
19. **The report gives what the stall watcher saw.** For the smoke test and each hardware run it prints the longest quiet stretch
    and the longest run of one repeated line, beside the limits (60 s and 20 lines; neither is measured). A stopped run's first line is the
    watcher's reason (`backstop`, `stalled` or `repeating`). The table has a status column, says `add` is a calibration op and that
    the simulator computes in f32, and prints a PCC near 1 as `1 - <distance>` or `1 + <distance>`.
20. **The static gate allows only listed names from `typing` and `__future__`, and refuses forbidden names as attributes and
    imports.** `from typing import sys as s` reached `sys` through the earlier gate. A limit stays: `from ttnn import shutil as s`
    passes, because the gate cannot tell a module from another name in `ttl` or `ttnn` without importing ttnn, which opens the device.
21. **`opinventory.main` defaults `--ttnn-python` to `~/.tenstorrent-venv/bin/python` under the passwd home**, and the stage 0 skill
    passes it. In an agent shell HOME points inside the run directory. Without the flag every op was `unchecked`.
22. **The ops-inventory template builds only a subclass of `PreTrainedModel`.** A downloaded config.json could name any public
    callable of transformers. A name that is not a model class, is not a string, or whose lookup raises is written as `skipped`.
23. **The simulator dies with the check.** kernel_check runs tt-lang-sim in its own session, kills that group on a timeout, turns
    SIGTERM, SIGINT and SIGHUP into an exception that kills it too (SIGINT and SIGHUP are left alone when they start ignored;
    SIGTERM is always handled), and starts it with `PR_SET_PDEATHSIG` set to SIGKILL for the case where kernel_check itself is
    killed outright. An interrupted check then ends by the signal that arrived, so its exit status names it.
24. **One probe per output directory.** `kernel-probe` holds an exclusive `flock` on `probe.lock` in `--out` for its whole run. A
    second probe that cannot take it is refused with exit 2 and told the holder's pid (written in the file as information). The
    kernel releases the lock when the holder dies, SIGKILL included; the file may remain then, and it is harmless. A clean exit
    removes the file while it still holds the lock. The first version (a pid file with a staleness check) let two probes that
    both judged a stale lock dead run at once.
25. **The stall watcher keeps at most 64 KiB of an unfinished line.** A longer piece is fed to the repeat detector as a line.
26. **Sampling flags.** `kernel-probe` takes `--temperature` (default 0) or `--temperatures` (a comma-separated list; `--attempts`
    runs at each), `--top-p` and `--seed`. Temperatures are in [0, 2], top-p in (0, 1]; bad values are refused with exit 2. No seed
    is sent at temperature 0. Above 0, attempt k (counting every attempt of the run) sends base + k; without `--seed` the base comes
    from the probe id and the report records it. A seed repeats a sampled run only if the server honours it, which is not verified.
    `AgentStep` takes the same three options; its default request is unchanged (`temperature: 0`, no other sampling key), so the
    supervisor is unchanged. The default stays greedy: the operator asked for determinism where it is useful and an easy way to
    raise the temperature to find wild or ingenious kernels. Greedy decoding makes repeats common; it does not make a kernel
    reproducible (see item 27).
27. **The report counts distinct kernels.** The claim that attempts are independent draws did not hold at temperature 0, where
    the server is asked for greedy decoding. The live probe gave 1 distinct kernel in 3 attempts of `add` and 1 in 5 of
    `linear_scan` (the five `linear_scan` attempts each took 9 turns and 198 s), 3 in 5 of `rmsnorm`, and 5 in 5 of `silu_mul`
    (why `silu_mul` diverged was not measured). Repeats are common at temperature 0 and not guaranteed. Each op is now split by
    temperature, each group says "distinct kernels: N of M attempts" by kernel sha256 (attempts that wrote no kernel are counted
    apart), and a warning follows when more than one attempt wrote a kernel and all wrote the same one.
28. **The latest skills (Task 14).** Operator rule, 2026-10-16: "WE USE THE LATEST tenstorrent/skills and tt-lang skills
    available. We check in preflight whether we have the latest even." The live probe had given the coder only the short
    `kernel-probe.md` and `eltwise_add.py`; `rmsnorm` passed 0 of 10 and `linear_scan` 0 of 6, failing on tt-lang API
    mistakes (element loops over Blocks, an invented `ttl.zeros`, `reduce_sum` without `dims` and `shape`), while a full
    tt-lang skill sat unused on the machine.
    - `orchard/skillsfresh.py` compares three sources with upstream, read only: the tenstorrent/skills clone(s) that
      `skills_dirs` point into (HEAD against `git ls-remote origin refs/heads/main`), the tt-lang checkout (its HEAD is
      the pin, so "behind" means upstream main moved past `TTLANG_PIN`), and the installed tt-lang skill (each file's git
      blob sha against the upstream tree, read with one `gh api` call; `SKILL.md` after tt-warp's `ttm-` rename and
      provenance block are undone). States: current, behind N, ahead, differs, unknown. Unknown never claims current;
      a current result for the same local state from the last 24 hours covers it. Only read-only git subcommands and
      `gh api` GETs can run, `git status` runs with `GIT_OPTIONAL_LOCKS=0`, only github.com is contacted, and the
      network has 15 s per call and 45 s in all.
    - tenstorrent/skills has no tt-lang skill. The installed one (`~/.claude/skills/ttm-tt-lang`) was copied by
      `tt-warp fastmodels sync` from tenstorrent/tt-metal, branch `agentic-research/fast-models-fast`, path
      `.agents/skills/tt-lang`. tt-metal's main removed that path (83bf34e), so the branch is the only upstream copy.
      On 2026-10-16 every file matched it.
    - That skill documents tt-lang specification 0.12. The pinned checkout documents 0.20: `ttl.kernel` became
      `ttl.operation` (0.13), `buffer_factor` became `block_count` (0.15), `broadcast`, `fill` and others moved from
      `ttl.math` to `ttl.block` (0.17), and `reduce_sum` takes `dims` and a result `shape` with no scaler. The check
      warns when the skill's version is older. The probe therefore gives the coder, in each attempt's `reference/`,
      the skill's `guide.md` and `examples.md` and the checkout's `TTLangSpecification.md`, `eltwise_add.py` and
      `eltwise_broadcast_reduce.py`, and `kernel-probe.md` says what to read for each kind of op and which names
      changed. A test reads the simulator's source in the checkout and checks each named API claim.
    - `kernel-probe` refuses (exit 2) when a reference file is missing, a link, not a regular file or over 1 MiB, and
      when the tt-lang checkout or the skill is stale; `--allow-stale-skills` runs anyway and the report records it
      with each source's local and upstream shas. `kernel-env --check` fails the same way and writes no cache. The
      `tt-orchard bringup` preflight has a `skills` row for the tenstorrent/skills clone: a warning when it is not
      current, dirty or on another branch, never a block, because bringup runs unattended. Orchard never updates a
      source; each result names the command a person runs.
    - Not measured: whether the extra reference files raise the probe's pass counts. No probe has run with them.
    - After the independent review (fix wave): the skill says to copy only the operation function (the examples'
      `from sim import ttl, ttnn`, `import torch` and from_torch/to_torch host code are refused; a test runs the gate
      over every shipped reference and requires the skill to name each construct it refuses) and that the reduce
      example is tagged `TTLANG_HARDWARE_CI: xfail-compiler`. An `unknown` result names no update. ls-remote's
      answer is read by the exact ref `refs/heads/main`. A cache file that raises RecursionError is ignored. The
      readers check the file type before the open and refuse a linked directory on the way to a reference file. The
      skill files are read once and the same bytes are hashed and copied. A dry run and the UI preflight write no
      cache. git runs with `http.followRedirects=false`, `core.fsmonitor=false` and `core.hooksPath=/dev/null`, and a
      timed-out call is killed with its process group. The block on a tt-lang checkout behind upstream main stays; its
      message says the fix is to move the pin and names `TTLANG_PIN`, `TTLANG_DIAGNOSTICS_SHA256` and
      `tt-orchard kernel-env`.
    - Task 14b, from the live rmsnorm probe: when kernel.py has exactly one top-level `@ttl.operation` function under
      another name, the `no_kernel` line names it and says to rename it to `OPERATION` (the category stays
      `no_kernel`, and the probe's re-check and the hardware gate give the same line). The skill's step 3 says the
      name is literally `OPERATION`, with a skeleton, and that inputs come in the order of `inputs` in the spec.
    - Task 14c: the skill states general facts on units (spec shapes and `x.shape` count elements; buffer shapes and
      tensor slices count tiles of 32 x 32; inputs are bf16 in tile layout), the simulator's IndexError for a slice
      in elements, and that the result is written into `out`. No op-specific code. The watchdog's TurnRepeat also
      reports a cycle of period 2 or 3 repeated for three full periods (all stages; identical signatures only).
    - Task 14d: the operation's name is free. The gate takes the one top-level def decorated with `ttl.operation`
      (any accepted spelling) as the operation; its name keeps the old protections (bound only by that def, one
      decorator, not a refused name); two such defs are refused; none is `no_kernel` ("defines no @ttl.operation
      function"). The harness calls the function the gate's own reading names, so the agent's check, the probe's
      re-check and the hardware phase agree. This replaces Task 14b's rename message and literal-name instruction.
      Reason: the coder ignored the naming instruction in three live runs (L4 linear_scan; L4c rmsnorm twice, the
      second with the rule and a `def OPERATION` skeleton in its prompt: it wrote `def rmsnorm` six times in turns 10
      to 19). Evidence: the scratchpad archives `live/aborted2/probe-c-rmsnorm/rmsnorm/attempt-1/` and
      `live/aborted3/probe-c-rmsnorm/rmsnorm/attempt-1/` (session scratchpad, not in the repository).
