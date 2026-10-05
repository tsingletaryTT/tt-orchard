# tt-orchard: stage 4 tests each shipped chip configuration (weights-only path) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the weights-only path, stage 4 shows the new model working on the chip configurations its packages will ship for: 2 chips and 4 chips required, 1 chip optional and recorded as supported or not. Each configuration is one hardware test that the supervisor runs under its own lease, and a configuration passes only with the supervisor's own record of its test.

**Architecture:** The stage 4 prepare step writes one directory per configuration (`stages/4/configs/<chips>/`, a copied template and its `swap_config.json`) and a list, `hw_tests.json`. A new module, `orchard/hwtests.py`, validates that list and owns its files: the plan (`tests/plan.json`, the resume marker), one record per test (`tests/<chips>/test-result.json`) and the summary. The supervisor runs the tests in order of chip count, leasing a free board or parking the coder (leasing any further board first), and stops any container a test left behind. Bundles (1 and 2 chips) reuse the stage 2 template. A new template, `serve_and_compare_container.py`, serves the 4-chip container package: it asks `tt-model serve ... --print` for the docker command, edits the argv with exact counts and runs it. The edits give the container a fresh tensor cache, an empty Hugging Face cache, the model directory and its blobs mounted read-only at their own paths, `MODEL_WEIGHTS_DIR` and `HF_MODEL`, and only the leased device nodes. A new gate, `gate_mesh_swap`, adds per-configuration checks to `gate_mesh`.

**Tech Stack:** Python 3.12 standard library only, pytest. Templates are tested against `tests/fake_swap_server.py` (stdlib HTTP, opens no device), a fake `tt-model` and a fake `docker` that are scripts on PATH. The supervisor is tested against the fake two-board machine in `tests/run_fakes.py` and the scripted model server in `tests/fake_model.py`.

**Spec:** `docs/superpowers/specs/2026-10-01-orchard-design.md`, sections 5 (stage 4: "multichip, then shrink to 2 and 1 chips", gate "per-config evidence"), 6 (park and restore, and its branch table) and 10 (resume markers, low disk, denials). Built on main at `1407417` (plan 4, the weights-only path, `--required-chips`, the swap templates with their 3300 s cold-boot timeout, and no gate feedback after a failed hardware test).

**Checked before hand-over:** every code block below was extracted from this document by a script into a fresh clone of the repo (branch `main` at `1407417`), applied task by task in document order, and run on 2026-10-03: 1416 passed and 1 skipped (1326 passed and 1 skipped before, so 90 new tests). Each task's "fails" step was run on that copy with only the task's test blocks applied, and the failure counts below are the observed ones. Every mutation step was run on the finished copy; each named test failed with the mutation and passed after the restore. The implementer still runs every step; this check does not replace them.

## Design

This section answers the five questions of the brief. The tasks implement it.

### 1. How the 4-chip configuration of a weights-only model is tested

**Package.** `changh95/qwen3.8-27b-p300x2`, profile `batch32`: a v5.1 container package, 4 chips (`P150x4`), 262,144 context, 32 sequences, no speculative decoder, accepts sampling. Measured on 2026-10-03: with a fresh tensor cache it answered `42` and `Paris` (run log, 16:20Z). The other installed 4-chip package, `mando2222/qwen3.8-27b-dflash2-p300x2-q4kv`, still has an unrebuilt September cache and uses a speculative decoder, so it is not used here.

**What `tt-model serve --print` prints.** This plan did not run `tt-model` (the brief forbids it). The command below was read from the installed tt-model source: `compose_run` in `~/.tenstorrent-venv/lib/python3.12/site-packages/tt_kernel/container.py` and `serve_container` in `container_cli.py`, with the package's manifest (`~/.cache/tt-model/pulled/changh95__qwen3.8-27b-p300x2/tt_kernel_manifest.json`) and the vllm-plugin launcher (`launchers.py`, `serve_argv`, `serve_env`). `--print` composes with `detach=False` and prints `shlex.join(argv)`:

```text
docker run --name tt-model-qwen3.8-27b-p300x2-batch32 --user <uid>:<gid>
  --label org.tenstorrent.tt-model=qwen3.8-27b-p300x2 --label ...profile=batch32 --label ...devices=0,1,2,3
  --device /dev/tenstorrent/0:/dev/tenstorrent/0  (one per --device-id)
  --ipc host --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G
  --volume $HF_HOME:/hf --env HF_HOME=/hf
  --volume ~/.cache/tt-model/qwen3.8-27b-p300x2/cache:/cache --env TT_METAL_CACHE=/cache
  --volume ~/.cache/tt-model/qwen3.8-27b-p300x2/weights:/weight-cache --env TT_DIT_CACHE_DIR=/weight-cache
  --volume ~/.cache/tt-model/qwen3.8-27b-p300x2/tensors:/tensor-cache --env TT_CACHE_PATH=/tensor-cache
  --publish <port>:<port>
  --env ARCH_NAME=blackhole ... --env HF_MODEL=Qwen/Qwen3.8-27B --env MESH_DEVICE=P150x4 ... (sorted)
  tt-model/qwen3.8-27b-p300x2:0becf4834925
  vllm serve Qwen/Qwen3.8-27B --revision 1d4bf0f... --max-model-len 262144 --max-num-seqs 32
  --block-size 64 --additional-config '{...}' ... --max-num-batched-tokens 262144 --port <port>
```

`tests/container_fakes.py` (Task 2) reproduces this shape. The runbook entry (Task 11) has the operator compare it with a real `--print` before the first run.

**What can be overridden.** `tt-model serve` has `--port`, `--device-id`, `--profile`, `--detach`, `--local-only` and `--no-update-check`, and no option for extra docker arguments or environment (its `--env` belongs to `tt-model package`). `HF_HOME` and `HF_HUB_CACHE` in the caller's environment move the `/hf` mount; nothing moves `/tensor-cache`. So the harness must run the `docker run` itself.

**Edit the parsed argv.** The memory note's sed (`s/^docker run /docker run -d --env ... /`) adds one flag and checks nothing. The template parses the printed line with `shlex.split` and walks the options one by one. Each edit names the token it changes and must apply exactly once (or at most once, where it says so). An option the template does not know exits 2 before anything starts. That is safer: a change in tt-model's output stops the test with a message, where a sed would run a command nobody read.

**How the new weights get in.** prepare_swap.py builds `model-dir/` exactly as for stage 2: the nearest model's `config.json` and preprocessor configs copied, the new model's tokenizer files and every `*.safetensors` linked by absolute resolved paths (into the new model's HF blob directory). The template then:
- mounts `model-dir/` read-only at its own absolute path, and each blob directory its links point into the same way, so the links resolve inside the container and the served model name is the model-dir path (the same as stage 2's requests);
- replaces `vllm serve Qwen/Qwen3.8-27B` with `vllm serve <model-dir>` and deletes `--revision` (a local directory has no revision);
- sets `--env HF_MODEL=<model-dir>` and `--env MODEL_WEIGHTS_DIR=<model-dir>`. The TT runtime in the source build resolves the weights directory as `MODEL_WEIGHTS_DIR`, then `HF_MODEL`, then the config path (`tt/qwen36_vllm.py`, line 227-228). The image was built from tt-metal `0e988de`, which is not on this box, so both are set;
- replaces the `/hf` volume's source with an empty `hf-isolated/` directory. The operator's Hugging Face cache holds the base model. Without it a runtime that ignored both variables could not find `Qwen/Qwen3.8-27B` and would fail at load. This is the structural guard against "a 4-chip container silently serving base weights". The gate's 0.85 bar (base weights measured 0.78 on 2 chips) is the second guard.

**Tensor cache.** The `/tensor-cache` volume's source becomes the configuration's own `tt_cache` (`/mnt/bonus/models/orchard-runs/cache/<model slug>/4chip-qwen3.8-27b-p300x2/tt_cache`). The template refuses a cache inside `~/.cache/tt-model` (exit 3) and applies stage 2's marker guard (a non-empty cache must hold `.orchard-model` naming this model). The `/cache` volume (the package's compiled-kernel cache) stays: kernels do not depend on the weights, and a warm kernel cache saves the cold compile that took more than 26 minutes on 2026-10-03.

**Device nodes.** The supervisor gives the test `TT_VISIBLE_DEVICES` (PCI addresses) and a new variable, `ORCHARD_DEVICE_IDS`, with the `/dev/tenstorrent` indices of the same chips from the lease. The template passes them as `--device-id`, and refuses a printed command that maps the whole `/dev/tenstorrent` directory or any other set of nodes (CLAUDE.md: a container that opens chips it was not leased can crash another tenant).

**Who runs docker.** The command runner refuses `docker run`, `docker stop`, `docker rm` and `tt-model serve` for agent shells, so the agent cannot run this test. It copies the template, writes `swap_config.json` and lists the test; the supervisor runs it as a hardware test, as stage 2 does. The supervisor builds the command itself (`python3 stages/4/configs/<chips>/<script>`), so the agent chooses a script from two names and cannot hand it a shell string. The test runs in the agent environment (HOME inside the run directory), so the template runs `tt-model serve --print` with `HOME` and `HF_HOME` set to the operator's (`operator_home`, `hf_home` in the config), which is where tt-model keeps `installed.json` and the package caches.

**Leftover containers.** The template stops (`docker stop -t 60`, SIGTERM then SIGKILL), saves `docker logs`, removes (`docker rm --force`) and checks its container in a `finally`. A test killed at its deadline runs no `finally`, and a container started with `docker run --detach` is not in the killed session, so the template labels it with `ORCHARD_TEST_LABEL` (`orchard.test=<12 hex digits of sha256(run dir)>`). After each test the supervisor lists containers with that label, stops and removes them, and blocks the stage (no release, no restore) if one is still listed.

### 2. The container template and its config keys

`orchard/skills/weights-swap-templates/serve_and_compare_container.py` (Tasks 2 and 3). It imports `guard_cache`, `load_reference`, `measure`, `write_report` and `ServerHTTPError` from `serve_and_compare.py`, which the skill copies next to it, so both templates measure the same way: 32 greedy tokens free-run, 32 teacher-forced single tokens compared with the stage 1 reference, the same coherence rule and the same `swap-check.json` with a `result_draft`. A `Container` object stands in for the `Popen` that `measure` polls.

Config keys (`swap_config.json` in the configuration directory): `run_dir`, `package`, `profile`, `chips`, `nearest_model_id`, `base_snapshot`, `new_snapshot` (both read by prepare_swap.py), `new_model_id`, `tt_cache`, `hf_home`, `operator_home`, `port`; optional for tests `health_timeout_s` and `test_raise_after_ready`. A bundle configuration keeps stage 2's keys: `bundle_dir` in place of `package`, and no `profile`, `chips` or `operator_home`. prepare_swap.py learns one thing: with `package` and no `bundle_dir` it builds only `model-dir/` (Task 1).

Exit codes: 0 measured; 2 the config, the supervisor's variables or the printed command could not be used (nothing started); 3 the tensor cache was refused; 4 the container did not start, exited or never became healthy (it waits up to `HEALTH_TIMEOUT_S`, 3300 s); 5 the server answered with an HTTP error.

### 3. The stage 4 flow on the weights-only path

- `spec_for(4, "weights-only")` returns `WEIGHTS_ONLY_STAGE_4`: skill `weights-swap-configs`, gate `gate_mesh_swap`, `boards=2` (the most any test needs), resume marker `tests/plan.json`, `tests=True`, disk 110 GB. The full-port path keeps `mesh-shrink`.
- **Configurations.** The run's `--required-chips` (operator: `2,4`) are required. The skill also asks for the 1-chip configuration with the existing bundle `episod/qwen3.8-27b-dflash2-p150` (16K context) and says to record it with `pass` false and a reason when it fails or no package exists. The gate (from `gate_mesh`) accepts an optional failure.
- **Prepare.** For each configuration the agent copies the three templates into `stages/4/configs/<N>/`, writes `swap_config.json` with a new cache per configuration, runs `prepare_swap.py`, and lists the test in `hw_tests.json` (`chips`, `script`, `deadline_s`; 3600 s each).
- **Validation before any lease** (`read_plan`): chips from 1 to 4, each count once, a known script that exists, a positive deadline, an absolute `tt_cache` in each `swap_config.json`, no cache inside `~/.cache/tt-model` or `~/.cache/qwen36-src-build`, no two configurations sharing a cache, a test for every required count, and deadlines adding up to no more than the stage budget. Any problem fails the attempt with the reasons, before a lease is taken.
- **Running.** The supervisor writes `tests/plan.json` (and an `evidence` ledger entry with its sha256), then runs each test with no record, in order of chip count (1, 2, 4). With the coder on 2 chips the 1- and 2-chip tests use the free board and the 4-chip test parks the coder; with the coder on 4 chips every test parks it.
- **Per-configuration evidence.** `tests/<N>/output.txt` (the supervisor's capture), `tests/<N>/test-result.json` (the supervisor's record: config, command, return code, timeout, seconds, chips, device ids, output sha256), and the template's `configs/<N>/evidence/swap-check.json`, `server.log` and, for the container, `docker-argv.json`. The stage's `test-result.json` lists every record for the finish step.
- **`result.json`** (the finish step): `{"configs": [{"chips": 2, "pass": true, "kind", "package", "serves", "server_ready_s", "coherent", "free_run_text", "top1_agreement", "n_tokens", "cache_dir", "evidence": [".../swap-check.json", ".../server.log", "stages/4/tests/2/output.txt"]}, ..., {"chips": 1, "pass": false, "reason": "...", "evidence": ["stages/4/tests/1/output.txt"]}]}`.
- **Gate feedback after a failed test.** Main gives no gate feedback after a failed hardware test. For a list, a failed test is one that must pass and did not: a required configuration's, or any listed one when the run names no required counts. An optional configuration that failed still leaves the one continuation for a malformed `result.json`, and an escalated attempt interrupted by a kill does not resume from a failed required test (Task 10).
- **Gate.** `gate_mesh_swap` = `gate_mesh` (required counts must pass; optional ones may fail) plus, for every entry that claims a pass: the supervisor's record shows exit 0, no timeout and N chips; the stage 2 bar on the entry's swap fields; and the entry's evidence includes the test's `swap-check.json`, whose `result_draft` holds the same `top1_agreement`. The supervisor also refuses any `tests/<N>/test-result.json` whose sha256 differs from the sha256 the ledger recorded when the supervisor wrote it, because the agent's `write_file` can write in `tests/` too.

### 4. The StageSpec and supervisor changes, and a crash in the middle

- `StageSpec` gains `tests: bool = False` (the hardware phase runs a list) and `disk: float | None = None` (a stage-specific free-disk need; `disk_gb` returns it when set). `STAGE4_SWAP_DISK_GB = 110.0` (three caches of about 34 GB: 34 GB measured for 2 chips, 31 GB for 4 chips, 1 chip assumed) and `TEST_DISK_GB = 40.0` (checked again before each test). The stage budget stays 28,800 s: three tests at 3600 s, a park and restore with a cold-boot budget for the coder, and two agent steps fit in about 17,200 s (a test pins this).
- Caches go under `/mnt/bonus/models/orchard-runs/cache/<model slug>/<N>chip-<package name>/tt_cache`, on the same disk as the run directory, so the run directory's disk check covers them.
- **Supervisor.** `_stage_body` sends a stage with `tests` to `_run_test_list`: prepare (unless a resumed stage has `tests/plan.json`), `read_plan`, `write_plan`, then `_run_listed` for each pending test, then `write_summary`; then the finish step and the gate, as for any hardware stage. `_spawn` is the shared test runner (the single-test path now uses it too) and adds `ORCHARD_DEVICE_IDS` and `ORCHARD_TEST_LABEL` to the test's environment.
- **A crash in the middle.** Each record is written whole (temp file and rename) as soon as its test ends. On restart the ledger shows stage 4 open; `open_stage_dir` keeps the directory because `tests/plan.json` exists; `_run_test_list` loads the plan, skips the prepare step and runs the first configuration with no record. A park interrupted by the crash is finished by `_ensure_coder` (recover and restore) before the stage resumes. A test interrupted by the crash runs again. Its cache may be part-written (a conversion reads back without an error), so before running a test whose cache's last test did not exit 0 (killed, timed out or failed), the supervisor moves that cache aside to `<cache>.interrupted-<k>` and records it; nothing is deleted.

### 5. The park decision for a 4-chip test

`decide_park(coder boards, chips, test.boards)` as today. For a test that needs more boards than are free:
1. Lease the further boards the test needs first (`test.boards - coder boards`, by the free board's first chip, or through the queue if another tenant holds it). If that lease cannot be had, the coder has not been touched and the stage blocks.
2. Park the coder with `Handoff.park()` (note, canary, stand-in, `tt-model stop`, confirm, reset). If the park blocks, the further lease is released.
3. Run the test on the coder's chips plus the further boards, in device order, all under this run's leases.
4. Sweep test containers; release the further lease (gozer resets it); `Handoff.restore()` (reset, start, ready, canary compared, stand-in stopped). A test that failed or timed out is recorded and the coder is restored all the same.

Measured costs on this box: `tt-model stop` 1 to 2 s (`TT_MODEL_STOP_S` 1.9), each gozer reset 41.7 s (`GOZER_RESET_S`, twice per park), the 2-chip coder back to ready in 110 to 130 s over four restarts (run log 2026-10-03), a 2-chip cold conversion about 5 minutes. **Not measured:** the 4-chip cold conversion, the 4-chip coder's restart, the 1-chip cache size. With the coder on 2 chips one stage 4 run costs one park (about 2 + 42 + 42 + 120 s, about 3.5 minutes) plus three test boots.

## Global Constraints

- Work in `/home/ttuser/code/tt-orchard`. Task 0 creates the branch `multichip-stage-4` from `main` at `1407417` or later; every later task commits on it. The repo is local only. Never run `git push` and never create a remote.
- Python 3.12, standard library only. Every new module starts with `from __future__ import annotations` and a docstring that says what the module owns. Comments explain why.
- The unit suite uses no hardware and no real model. No test opens `/dev/tenstorrent/*` or runs the real `tt-smi`, `tt-model`, `docker`, `gozer` or a model server. Template tests put fake `tt-model` and `docker` scripts first on PATH; supervisor tests pass `containers=FakeContainers()` to `build`, so the supervisor's label search never reaches a real docker.
- Do not run the supervisor or a template against hardware, and do not run any gozer, docker or tt-model command that changes state on the real box. The operator runs the runbook entry (Task 11).
- Change existing modules only where a task says so. The single-test hardware path (`hw_test.json`) and the full-port table keep their behavior; their tests stay green.
- Every default value lives in `orchard/defaults.py` as one named constant, with a comment that cites where it was measured or says that it is a choice and was not measured.
- Every guard gets a mutation step: change the guard, run the named test, watch it fail, restore the guard. After each restore, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`) before the confirming run.
- Run the whole suite before each commit: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider`. The baseline is 1326 passed and 1 skipped in about 250 s. After this plan: 1416 passed and 1 skipped in about 330 s. The template tests need the `tokenizers` package, which the interpreter at `~/.tenstorrent-venv/bin/python3` has; without it `tests/test_weights_swap_templates.py` skips and the counts differ.
- Skills never name a lease tool (`tests/test_skills.py` checks the word "gozer").
- The run never publishes. Agent shells get no credentials, and the test container gets no `HF_TOKEN`.
- Commit messages are plain English, one idea per commit. Writing rules for docs and comments: state the finding; short sentences; no "X, not Y" framing; no aphoristic closers; no metaphors that stand in for a claim.

## Review Focus

1. A 4-chip container that serves the base weights with no error. Expected: the edited argv names the model directory in `vllm serve`, `HF_MODEL` and `MODEL_WEIGHTS_DIR`, mounts an empty `/hf` and never the operator's Hugging Face cache; a passing configuration needs `top1_agreement` of at least 0.85 (Task 2: `test_the_edit_points_every_weight_path_at_the_new_model`; Task 3: `test_the_container_serves_the_new_weights_and_is_removed_afterwards`; Task 4: `test_the_swap_mesh_gate_holds_each_configuration_to_the_stage_2_bar`).
2. A tensor cache shared across models or configurations. Expected: the container template refuses a cache inside `~/.cache/tt-model` and a cache marked for another model before anything starts; the supervisor refuses a list where two configurations share a cache or one sits in a package cache, before any lease (Task 3: `test_a_cache_inside_the_package_caches_is_refused_before_anything_starts`, `test_a_cache_marked_for_another_model_is_refused`; Task 5: `test_two_configurations_with_one_cache_are_refused`, `test_a_cache_inside_a_package_cache_is_refused`; Task 7: `test_a_list_whose_configurations_share_a_cache_fails_before_any_hardware_is_used`). A cache whose test was killed or timed out is moved aside before it is used again (Task 9: `test_a_cache_whose_test_was_killed_is_moved_aside_before_the_test_runs_again`, `test_a_timed_out_test_leaves_its_cache_to_be_moved_aside_by_the_next_attempt`).
3. A 4-chip test run while another lease holds a board. Expected: the further board is leased through the queue before the park, the coder stays up while it waits, and when the test starts every chip it gets is under one of this run's leases (Task 7: `test_the_4_chip_test_waits_for_a_board_another_tenant_holds_and_runs_only_on_this_runs_leases`). The template maps only the leased device nodes (Task 2: `test_a_printed_command_of_another_shape_is_refused`, first two cases).
4. The coder not restored after a failed 4-chip test, or a container left holding chips. Expected: a test that exits 4 is recorded, the further board is released and the coder is restored before the finish step, through the restore itself (Task 7: `test_a_failed_4_chip_test_still_releases_its_board_and_restores_the_coder`); the template removes its container on every exit path (Task 3: `test_the_container_is_stopped_and_removed_on_every_exit_path`); a labelled container left behind is stopped by the supervisor, and one that survives blocks before any release or restore (Task 7: `test_a_container_a_test_left_running_is_stopped_before_its_lease_goes_back`, `test_a_container_that_will_not_stop_blocks_before_any_lease_is_released_or_the_coder_restored`).
5. A pass recorded for a configuration that was never run. Expected: the gate needs the supervisor's record with exit 0 on N chips, and the supervisor refuses a record whose sha256 differs from the ledger's (Task 4: `test_the_swap_mesh_gate_refuses_a_pass_for_a_configuration_whose_test_never_ran`; Task 7: `test_a_pass_claimed_for_a_configuration_whose_test_never_ran_fails_the_gate`; Task 8: `test_a_test_record_the_agent_wrote_fails_the_gate`). A crash resumes at the first configuration with no record and never re-runs a recorded one (Task 9: `test_a_kill_during_the_list_resumes_at_the_first_configuration_without_a_record[2]` and `[4]`).

## File Structure

| File | Responsibility |
|---|---|
| `orchard/skills/weights-swap-templates/prepare_swap.py` | also builds a model directory alone, for a container package |
| `orchard/skills/weights-swap-templates/serve_and_compare.py` | `load_reference` and `write_report` split out, so the container template measures the same way |
| `orchard/skills/weights-swap-templates/serve_and_compare_container.py` | new: the 4-chip container test (argv edit, docker run, measure, stop) |
| `orchard/skills/weights-swap-configs.md` | new: the stage 4 skill for the weights-only path |
| `orchard/defaults.py` | `STAGE4_SWAP_DISK_GB`, `TEST_DISK_GB` |
| `orchard/stages.py` | `StageSpec.tests` and `.disk`, `_swap_reasons`, `hw_record_path`, `hw_record_problem`, `gate_mesh_swap`, `WEIGHTS_ONLY_STAGE_4`, `spec_for(4, ...)` |
| `orchard/hwtests.py` | new: the list of tests, its validation and files, suspect caches, records only the supervisor wrote |
| `orchard/context.py` | the prepare and finish task texts for a list of tests; the finish step sees `hw_tests.json` |
| `orchard/supervisor.py` | `LabelledContainers`; `_run_test_list`, `_run_listed`, `_spawn`, the sweep, the record check, `_test_failure`; `build(..., containers=)` |
| `tests/container_fakes.py`, `tests/fake_tt_model.py`, `tests/fake_docker.py` | new: the printed command, fake tt-model, fake docker |
| `tests/test_container_template.py`, `tests/test_hwtests.py`, `tests/test_supervisor_stage4.py` | new test files |
| `tests/run_fakes.py` | `FakeContainers`; stage 4's scripted files for the weights-only path and for mesh-shrink |
| `tests/test_*.py` (existing) | updated where stage 4's behavior changes on the weights-only path |
| `docs/runbooks/hardware-validation.md`, `README.md`, `CLAUDE.md` | the operator's runbook entry and the log |

---

### Task 0: Branch from main

**Files:**
- None changed.

**Interfaces:**
- Consumes: `main` at `1407417` or later.
- Produces: the branch `multichip-stage-4`.

- [ ] **Step 1: Confirm a clean tree on main**

Run: `cd /home/ttuser/code/tt-orchard && git status --short --branch && git log --oneline -1`
Expected: `## main` and nothing else; the last commit is `1407417 Merge branch 'no-feedback-after-failed-test'` or a later commit. A later commit can change the code the replace blocks below target, in particular `_stage_body` and `_run_stage` in `orchard/supervisor.py` and `gate_weights_swap` in `orchard/stages.py`. If a replace target is not found exactly once, stop and report it. If the tree is not clean, stop and report it.

- [ ] **Step 2: Run the baseline suite**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `1326 passed, 1 skipped` (more if later commits added tests; note the number).

- [ ] **Step 3: Branch**

Run: `git switch -c multichip-stage-4 && git status --short --branch`
Expected: `## multichip-stage-4` and nothing else.

---

### Task 1: prepare_swap.py builds a model directory alone for a container package

**Files:**
- Modify: `orchard/skills/weights-swap-templates/prepare_swap.py` (docstring, `load_config`, `main`)
- Test: `tests/test_weights_swap_templates.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `swap_config.json` may name `package` (any non-empty string) in place of `bundle_dir`; then `prepare_swap.py` builds `model-dir/`, writes no `run.sh`, prints `run.sh: not built; <package> is a container package, ...` and exits 0. With neither key it exits 2 naming `bundle_dir (or package, for a container package)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_weights_swap_templates.py`:

```python


# ---- prepare_swap.py for a container package (stage 4) ------------------------------------------

def container_config(prep, **change):
    cfg = json.loads((prep["stage"] / "swap_config.json").read_text())
    del cfg["bundle_dir"]
    cfg.update(change)
    (prep["stage"] / "swap_config.json").write_text(json.dumps(cfg))


def test_prepare_for_a_container_package_builds_only_the_model_dir(prep):
    container_config(prep, package="changh95/qwen3.8-27b-p300x2")
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    md = prep["stage"] / "model-dir"
    assert (md / "config.json").read_text() == (prep["base"] / "config.json").read_text()
    assert os.readlink(md / "tokenizer.json") == os.path.realpath(prep["new"] / "tokenizer.json")
    assert not (prep["stage"] / "run.sh").exists()
    assert "changh95/qwen3.8-27b-p300x2 is a container package" in r.stdout


def test_prepare_with_neither_a_bundle_nor_a_package_exits_2(prep):
    container_config(prep)
    r = run_prepare(prep["stage"])
    assert r.returncode == 2, r.stdout + r.stderr
    assert "bundle_dir (or package" in r.stderr
    assert not (prep["stage"] / "model-dir").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_weights_swap_templates.py`
Expected: 2 FAIL, 20 passed. Both fail with `prepare_swap: swap_config.json is missing ['bundle_dir']`.

- [ ] **Step 3: Implement**

In `orchard/skills/weights-swap-templates/prepare_swap.py`, replace:

```python
run.sh is written. Everything else in the script, including its environment lines, stays as it is.

Config keys read here: bundle_dir, nearest_model_id, base_snapshot, new_snapshot. The other keys
in swap_config.json are for serve_and_compare.py.
"""
from __future__ import annotations
```

with:

```python
run.sh is written. Everything else in the script, including its environment lines, stays as it is.

For a container package (stage 4's container configuration) swap_config.json names `package`
and no `bundle_dir`. Then only model-dir/ is built: serve_and_compare_container.py edits the
container's docker command and needs no run.sh.

Config keys read here: nearest_model_id, base_snapshot, new_snapshot, and bundle_dir or package.
The other keys in swap_config.json are for serve_and_compare.py and serve_and_compare_container.py.
"""
from __future__ import annotations
```

In `orchard/skills/weights-swap-templates/prepare_swap.py`, replace:

```python
    except (OSError, ValueError) as exc:
        fail(f"cannot read {path}: {exc}")
    missing = [k for k in ("bundle_dir", "nearest_model_id", "base_snapshot", "new_snapshot")
               if not cfg.get(k)]
    if missing:
        fail(f"swap_config.json is missing {missing}")
```

with:

```python
    except (OSError, ValueError) as exc:
        fail(f"cannot read {path}: {exc}")
    missing = [k for k in ("nearest_model_id", "base_snapshot", "new_snapshot") if not cfg.get(k)]
    if not cfg.get("bundle_dir") and not cfg.get("package"):
        missing.append("bundle_dir (or package, for a container package)")
    if missing:
        fail(f"swap_config.json is missing {missing}")
```

In `orchard/skills/weights-swap-templates/prepare_swap.py`, replace:

```python
def main() -> int:
    cfg = load_config()
    bundle = Path(cfg["bundle_dir"]).resolve()
    base, new = Path(cfg["base_snapshot"]), Path(cfg["new_snapshot"])
    for label, d in (("bundle_dir", bundle), ("base_snapshot", base), ("new_snapshot", new)):
        if not d.is_dir():
            fail(f"{label} {d} is not a directory")
    src_run = bundle / "run.sh"
    if not src_run.is_file():
        fail(f"{src_run} does not exist")
    run_out = HERE_DIR / "run.sh"
    # Edit in memory first, so a failed edit leaves no run.sh and no half-built model-dir.
    model_dir = HERE_DIR / "model-dir"
    text, notes = edit_run_script(src_run.read_text(encoding="utf-8"), bundle,
                                  cfg["nearest_model_id"], model_dir)
    model_dir, copied, linked, absent = build_model_dir(base, new)
    run_out.write_text(text, encoding="utf-8")
    run_out.chmod(run_out.stat().st_mode | 0o111)
    print(f"model-dir: {model_dir}")
    print(f"  copied from base_snapshot: {', '.join(copied) or 'nothing'}")
```

with:

```python
def main() -> int:
    cfg = load_config()
    base, new = Path(cfg["base_snapshot"]), Path(cfg["new_snapshot"])
    bundle = Path(cfg["bundle_dir"]).resolve() if cfg.get("bundle_dir") else None
    for label, d in (("bundle_dir", bundle), ("base_snapshot", base), ("new_snapshot", new)):
        if d is not None and not d.is_dir():
            fail(f"{label} {d} is not a directory")
    model_dir = HERE_DIR / "model-dir"
    if bundle is not None:
        src_run = bundle / "run.sh"
        if not src_run.is_file():
            fail(f"{src_run} does not exist")
        # Edit in memory first, so a failed edit leaves no run.sh and no half-built model-dir.
        text, notes = edit_run_script(src_run.read_text(encoding="utf-8"), bundle,
                                      cfg["nearest_model_id"], model_dir)
    model_dir, copied, linked, absent = build_model_dir(base, new)
    print(f"model-dir: {model_dir}")
    print(f"  copied from base_snapshot: {', '.join(copied) or 'nothing'}")
```

In `orchard/skills/weights-swap-templates/prepare_swap.py`, replace:

```python
    if absent:
        print(f"  absent in new_snapshot, not linked: {', '.join(absent)}")
    print(f"run.sh: {run_out}")
    print(f'  HERE="{bundle}"; --model {model_dir}')
```

with:

```python
    if absent:
        print(f"  absent in new_snapshot, not linked: {', '.join(absent)}")
    if bundle is None:
        print(f"run.sh: not built; {cfg['package']} is a container package, and "
              "serve_and_compare_container.py starts it")
        return 0
    run_out = HERE_DIR / "run.sh"
    run_out.write_text(text, encoding="utf-8")
    run_out.chmod(run_out.stat().st_mode | 0o111)
    print(f"run.sh: {run_out}")
    print(f'  HERE="{bundle}"; --model {model_dir}')
```

In `orchard/skills/weights-swap-templates/prepare_swap.py`, replace:

```python
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

with:

```python
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_weights_swap_templates.py`
Expected: 22 passed.

- [ ] **Step 5: Mutation**

In `orchard/skills/weights-swap-templates/prepare_swap.py`, change the line `    if not cfg.get("bundle_dir") and not cfg.get("package"):` to `    if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_weights_swap_templates.py -k neither`. Expected: 1 failed. Restore the line, delete the bytecode, run it again. Expected: 1 passed.

- [ ] **Step 6: Run the suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -1` (expected `1328 passed, 1 skipped`), then:

```bash
git add orchard/skills/weights-swap-templates/prepare_swap.py tests/test_weights_swap_templates.py
git commit -m "prepare_swap: build only the model directory for a container package"
```

---

### Task 2: The container template's argv edit

**Files:**
- Create: `orchard/skills/weights-swap-templates/serve_and_compare_container.py` (the module docstring, `EditError`, `blob_dirs`, `edit_docker_argv`)
- Create: `tests/container_fakes.py`
- Test: `tests/test_container_template.py` (new)

**Interfaces:**
- Consumes: nothing.
- Produces in the template: `class EditError(Exception)`; `blob_dirs(model_dir: Path) -> list[str]`; `edit_docker_argv(argv: list[str], *, image: str, nearest: str, model_dir, tt_cache, hf_dir, name: str, label: str, device_ids: list[int], port: int, blobs: list[str]) -> list[str]` (raises `EditError`); constants `STAGE_DIR`, `TT_DEVICE`, `VALUE_OPTIONS`, `FLAG_OPTIONS`.
- Produces in `tests/container_fakes.py`: `PACKAGE`, `IMAGE`, `NEAREST`, `REVISION`, `ADDITIONAL`, `printed_argv(*, hf, pkg_cache, port, device_ids, whole_dir=False, extra_options=(), extra_env=()) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/container_fakes.py`:

```python
"""The docker command `tt-model serve <container package> ... --print` prints, for the container
template tests (tests/test_container_template.py, tests/fake_tt_model.py).

`printed_argv` follows tt-model's compose_run (tt_kernel/container.py, read on 2026-10-03) for the
4-chip plain package changh95/qwen3.8-27b-p300x2, profile batch32: name, user and labels; one
--device per leased chip; ipc, hugepages; the /hf, /cache, /weight-cache and /tensor-cache
volumes, each beside its variable; the published port; the serve env sorted by name; the image;
then the vllm-plugin launcher's server argv (`vllm serve <weights id> --revision <sha> ...`).
--print composes with detach=False, so the printed command has no --detach.
"""
from __future__ import annotations

PACKAGE = "changh95/qwen3.8-27b-p300x2"
IMAGE = "tt-model/qwen3.8-27b-p300x2:0becf4834925"
NEAREST = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ADDITIONAL = '{"tt": {"fabric_config": "FABRIC_1D", "trace_region_size": 1073741824}}'


def printed_argv(*, hf, pkg_cache, port, device_ids, whole_dir=False, extra_options=(),
                 extra_env=()) -> list[str]:
    devices = (["--device", "/dev/tenstorrent"] if whole_dir else
               [x for d in device_ids for x in ("--device", f"/dev/tenstorrent/{d}:/dev/tenstorrent/{d}")])
    env = {"ARCH_NAME": "blackhole", "HF_HUB_OFFLINE": "1", "HF_MODEL": NEAREST,
           "MESH_DEVICE": "P150x4", "QWEN36_DRAFTER": "mtp", "VLLM_RPC_TIMEOUT": "900000",
           **dict(extra_env)}
    return (["docker", "run", "--name", "tt-model-qwen3.8-27b-p300x2-batch32", "--user", "1000:1000",
             "--label", "org.tenstorrent.tt-model=qwen3.8-27b-p300x2",
             "--label", "org.tenstorrent.tt-model.profile=batch32",
             "--label", "org.tenstorrent.tt-model.devices=" + ",".join(str(d) for d in device_ids)]
            + devices
            + ["--ipc", "host", "--mount", "type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G",
               "--volume", f"{hf}:/hf", "--env", "HF_HOME=/hf",
               "--volume", f"{pkg_cache}/cache:/cache", "--env", "TT_METAL_CACHE=/cache",
               "--volume", f"{pkg_cache}/weights:/weight-cache", "--env", "TT_DIT_CACHE_DIR=/weight-cache",
               "--volume", f"{pkg_cache}/tensors:/tensor-cache", "--env", "TT_CACHE_PATH=/tensor-cache",
               "--publish", f"{port}:{port}"]
            + list(extra_options)
            + [x for k, v in sorted(env.items()) for x in ("--env", f"{k}={v}")]
            + [IMAGE, "vllm", "serve", NEAREST, "--revision", REVISION, "--max-model-len", "262144",
               "--max-num-seqs", "32", "--block-size", "64", "--additional-config", ADDITIONAL,
               "--max-num-batched-tokens", "262144", "--port", str(port)])
```

Create `tests/test_container_template.py`:

```python
"""serve_and_compare_container.py: the stage 4 template that serves the swapped weights with a
container package. The argv edit is tested in-process; the whole script runs against a fake
`tt-model` and a fake `docker` (tests/fake_tt_model.py, tests/fake_docker.py) that start
tests/fake_swap_server.py, which opens no device."""
import importlib.util
from pathlib import Path

import pytest

from container_fakes import IMAGE, NEAREST, PACKAGE, REVISION, printed_argv

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "orchard" / "skills" / "weights-swap-templates"
LABEL = "orchard.test=0123456789ab"


def load_template():
    spec = importlib.util.spec_from_file_location("serve_and_compare_container",
                                                  TEMPLATES / "serve_and_compare_container.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sac = load_template()

# ---- the argv edit ---------------------------------------------------------------------------------


@pytest.fixture
def paths(tmp_path):
    p = {"hf": tmp_path / "home" / ".cache" / "huggingface",
         "pkg": tmp_path / "home" / ".cache" / "tt-model" / "qwen3.8-27b-p300x2",
         "md": tmp_path / "configs" / "4" / "model-dir", "cache": tmp_path / "cache" / "4chip" / "tt_cache",
         "hf_dir": tmp_path / "configs" / "4" / "hf-isolated",
         "blobs": str(tmp_path / "home" / ".cache" / "huggingface" / "hub" / "models--A--B" / "blobs")}
    return p


def edit(paths, printed=None, ids=(0, 1, 2, 3), port=8100, **kw):
    argv = printed if printed is not None else printed_argv(hf=paths["hf"], pkg_cache=paths["pkg"],
                                                            port=port, device_ids=list(ids))
    return sac.edit_docker_argv(argv, image=IMAGE, nearest=NEAREST, model_dir=paths["md"],
                                tt_cache=paths["cache"], hf_dir=paths["hf_dir"],
                                name="orchard-4chip-8100", label=LABEL, device_ids=list(ids),
                                port=port, blobs=[paths["blobs"]], **kw)


def pairs(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == flag]


def test_the_edit_points_every_weight_path_at_the_new_model(paths):
    out = edit(paths)
    md = str(paths["md"])
    k = out.index(IMAGE)
    assert out[:3] == ["docker", "run", "--detach"]
    vols = pairs(out[:k], "--volume")
    assert f"{paths['cache']}:/tensor-cache" in vols                 # a fresh cache for this model
    assert f"{paths['hf_dir']}:/hf" in vols                          # the base weights are not mounted
    assert f"{md}:{md}:ro" in vols and f"{paths['blobs']}:{paths['blobs']}:ro" in vols
    assert not any(v.startswith(f"{paths['hf']}:") or v.startswith(f"{paths['pkg']}/tensors")
                   for v in vols)
    envs = pairs(out[:k], "--env")
    assert envs.count(f"HF_MODEL={md}") == 1 and envs.count(f"MODEL_WEIGHTS_DIR={md}") == 1
    assert not any(e.startswith("HF_MODEL=Qwen") for e in envs)
    assert "TT_CACHE_PATH=/tensor-cache" in envs and "HF_HOME=/hf" in envs
    assert pairs(out[:k], "--label") == [LABEL] and pairs(out[:k], "--name") == ["orchard-4chip-8100"]
    assert pairs(out[:k], "--device") == [f"/dev/tenstorrent/{d}:/dev/tenstorrent/{d}" for d in range(4)]
    assert out[k + 1:k + 4] == ["vllm", "serve", md]
    assert "--revision" not in out and REVISION not in out
    assert pairs(out[k:], "--port") == ["8100"] and "--max-model-len" in out[k:]


def test_options_the_edit_does_not_touch_are_kept_in_order(paths):
    printed = printed_argv(hf=paths["hf"], pkg_cache=paths["pkg"], port=8100, device_ids=[0, 1, 2, 3])
    kept = ["--user", "--ipc", "--mount", "--publish"]
    before = [(f, v) for f, v in zip(printed, printed[1:]) if f in kept]
    out = edit(paths)
    assert [(f, v) for f, v in zip(out, out[1:]) if f in kept] == before
    assert f"{paths['pkg']}/cache:/cache" in pairs(out, "--volume")   # the JIT kernel cache stays


def test_an_existing_model_weights_dir_is_replaced_and_not_added_twice(paths):
    printed = printed_argv(hf=paths["hf"], pkg_cache=paths["pkg"], port=8100, device_ids=[0, 1, 2, 3],
                           extra_env={"MODEL_WEIGHTS_DIR": "/somewhere/else"})
    out = edit(paths, printed=printed)
    assert [e for e in pairs(out, "--env") if e.startswith("MODEL_WEIGHTS_DIR=")] == [
        f"MODEL_WEIGHTS_DIR={paths['md']}"]


def test_a_token_variable_is_not_passed_to_the_test_container(paths):
    printed = printed_argv(hf=paths["hf"], pkg_cache=paths["pkg"], port=8100, device_ids=[0, 1, 2, 3])
    k = printed.index(IMAGE)
    printed = printed[:k] + ["--env", "HF_TOKEN"] + printed[k:]
    assert "HF_TOKEN" not in edit(paths, printed=printed)


def bad(paths, change):
    printed = printed_argv(hf=paths["hf"], pkg_cache=paths["pkg"], port=8100, device_ids=[0, 1, 2, 3])
    return change(printed, paths)


def swap(old, new, count=1):
    def change(argv, paths):
        text = "\x00".join(argv)
        assert text.count(old) >= 1, old
        return text.replace(old, new, count).split("\x00")
    return change


def insert_before_image(*tokens):
    def change(argv, paths):
        k = argv.index(IMAGE)
        return argv[:k] + list(tokens) + argv[k:]
    return change


@pytest.mark.parametrize("change,words", [
    (swap("/dev/tenstorrent/0:/dev/tenstorrent/0", "/dev/tenstorrent"), "not one /dev/tenstorrent"),
    (swap("/dev/tenstorrent/3:/dev/tenstorrent/3", "/dev/tenstorrent/5:/dev/tenstorrent/5"), "the lease gave"),
    (insert_before_image("--volume", "/x:/tensor-cache"), "/tensor-cache in the printed command, found 2"),
    (swap("\x00--env\x00HF_MODEL=Qwen/Qwen3.8-27B", ""), "HF_MODEL"),
    (insert_before_image("--volume", "/x:/hf-hub"), "/hf-hub"),
    (insert_before_image("--privileged"), "unknown option '--privileged'"),
    (swap("TT_CACHE_PATH=/tensor-cache", "TT_CACHE_PATH=/elsewhere"), "the tensor cache must be"),
    (swap(f"\x00{IMAGE}\x00", "\x00other/image:1\x00"), "image"),
    (insert_before_image(IMAGE), "image"),
    (swap(f"serve\x00{NEAREST}", "serve\x00Other/Model"), NEAREST),
    (swap(f"--revision\x00{REVISION}", f"--revision\x00{REVISION}\x00--revision\x00{REVISION}"),
     "at most one --revision"),
    (swap("--port\x008100", "--port\x008000"), "the config's port is 8100"),
    (lambda argv, p: ["docker", "create"] + argv[2:], "docker run"),
])
def test_a_printed_command_of_another_shape_is_refused(paths, change, words):
    with pytest.raises(sac.EditError) as exc:
        edit(paths, printed=bad(paths, change))
    assert words in str(exc.value)


def test_blob_dirs_lists_the_directories_the_links_point_into(tmp_path):
    md, blobs = tmp_path / "model-dir", tmp_path / "hub" / "models--A--B" / "blobs"
    md.mkdir()
    blobs.mkdir(parents=True)
    for name in ("tokenizer.json", "model-00001.safetensors"):
        (blobs / f"b-{name}").write_text("x")
        (md / name).symlink_to(blobs / f"b-{name}")
    (md / "config.json").write_text("{}")                 # a copied file is inside model-dir already
    assert sac.blob_dirs(md) == [str(blobs)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_container_template.py`
Expected: 1 error during collection: `FileNotFoundError` for `serve_and_compare_container.py` (the module is loaded at import).

- [ ] **Step 3: Implement**

Create `orchard/skills/weights-swap-templates/serve_and_compare_container.py`:

```python
#!/usr/bin/env python3
"""Serve the swapped weights with a container package and compare the chip's tokens with the CPU
reference (stage 4 on the weights-only path).

The weights-swap-configs skill copies this file, serve_and_compare.py and prepare_swap.py into one
configuration directory, stages/4/configs/<chips>/, writes swap_config.json there and runs
prepare_swap.py, which builds model-dir/ (for a container package it builds no run.sh). The
supervisor runs this file as one of the stage's hardware tests, on leased chips. It names the
chips twice: TT_VISIBLE_DEVICES (PCI addresses) and ORCHARD_DEVICE_IDS (the /dev/tenstorrent
indices of the same chips, which docker needs). It also sets ORCHARD_TEST_LABEL, a docker label
for every container a test of this run starts, so the supervisor can find a container this script
failed to stop.

`tt-model serve` takes no extra docker arguments or environment. So this script asks it for the
command it would run (`tt-model serve <package> ... --print`, which prints and starts nothing),
edits that argv, and runs the edited `docker run` itself. Editing the argv with exact counts is
safer than a sed over the printed text: each edit names the token it changes, and a printed
command whose shape this script does not know exits 2 before anything starts.

The edits (each must apply exactly once unless it says otherwise):
- `--detach` is added (the printed command has none), so `docker run` returns the container id.
- `--name` becomes `orchard-<chips>chip-<port>`. The test container then never takes the name a
  `tt-model serve` of the same package would use, and `tt-model stop` never finds it.
- Every `--label` is dropped and ORCHARD_TEST_LABEL is added. The tt-model labels would make the
  container look like a tt-model server of that package to `tt-model stop` and to tt-model's
  free-chip scan.
- Each `--device` must be one `/dev/tenstorrent/<N>` node, and the N must be exactly
  ORCHARD_DEVICE_IDS. A mapping of the whole `/dev/tenstorrent` directory exits 2: a container
  that opens a device it was not leased can crash another tenant's chips.
- `--env TT_CACHE_PATH` must be `/tensor-cache`, and the `/tensor-cache` volume's source becomes
  `tt_cache`, a new directory for this model and this configuration. The package's own cache
  (~/.cache/tt-model/<package>/tensors) is keyed only by layer name and mesh, so it would serve
  the base model's tensors without any error.
- The `/hf` volume's source becomes `hf-isolated/`, an empty directory next to this script. The
  printed command mounts the operator's whole Hugging Face cache there, and that cache holds the
  base model. Without it nothing in the container can load the base weights by their hub id, so a
  runtime that ignores MODEL_WEIGHTS_DIR fails with an error. A `/hf-hub` volume (HF_HUB_CACHE
  outside HF_HOME) exits 2.
- `model-dir/` is mounted read-only at its own absolute path, and so is each directory its links
  point into (the new model's blobs). The links then resolve inside the container as they do on
  the host, and the served model name is the model-dir path, as in serve_and_compare.py.
- `--env HF_MODEL=...` becomes the model-dir path, and `--env MODEL_WEIGHTS_DIR=<model-dir>` is
  set (replaced if present, added if not). The TT runtime takes its weights directory from
  MODEL_WEIGHTS_DIR, then HF_MODEL, then the config path.
- `--env HF_TOKEN` is dropped if present; a test container gets no token.
- In the server's arguments after the image, the nearest model's id becomes the model-dir path,
  and `--revision <sha>` and `--tokenizer-revision <sha>` are deleted (at most one of each),
  because a local directory has no revision. The server's `--port` must equal the config's port.

Any option this script does not know exits 2. A new option in tt-model's output must be read by a
person before a test trusts it.
"""
from __future__ import annotations

import os
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
TT_DEVICE = "/dev/tenstorrent"
VALUE_OPTIONS = {"--name", "--user", "--label", "--device", "--ipc", "--mount", "--volume", "--env",
                 "--publish"}
FLAG_OPTIONS = {"--detach", "--rm"}


class EditError(Exception):
    """The printed command is not the shape this script edits. Nothing has started."""


def blob_dirs(model_dir: Path) -> list[str]:
    """The directories model-dir's links point into, sorted. prepare_swap.py writes absolute,
    fully resolved link targets, so these are the new model's blob directories."""
    return sorted({os.path.dirname(os.path.realpath(p)) for p in model_dir.iterdir() if p.is_symlink()})


def _once(count: int, what: str) -> None:
    if count != 1:
        raise EditError(f"expected exactly one {what} in the printed command, found {count}")


def edit_docker_argv(argv: list[str], *, image: str, nearest: str, model_dir, tt_cache, hf_dir,
                     name: str, label: str, device_ids: list[int], port: int,
                     blobs: list[str]) -> list[str]:
    """The edited `docker run` argv (see the module docstring). Raises EditError."""
    if argv[:2] != ["docker", "run"]:
        raise EditError(f"the printed command does not start with 'docker run': {argv[:2]}")
    hits = [i for i, a in enumerate(argv) if a == image]
    _once(len(hits), f"image {image!r}")
    opts, server = argv[2:hits[0]], list(argv[hits[0] + 1:])
    md = str(model_dir)
    out = ["docker", "run", "--detach"]
    seen = {"--name": 0, "/tensor-cache": 0, "/hf": 0, "HF_MODEL": 0, "MODEL_WEIGHTS_DIR": 0,
            "TT_CACHE_PATH": 0}
    devices = []
    i = 0
    while i < len(opts):
        opt = opts[i]
        if opt in FLAG_OPTIONS:
            if opt == "--rm":
                out.append(opt)
            i += 1                                    # --detach is already in `out`
            continue
        if opt not in VALUE_OPTIONS or i + 1 >= len(opts):
            raise EditError(f"unknown option {opt!r} in the printed command; read it before "
                            "trusting this script with it")
        val = opts[i + 1]
        i += 2
        if opt == "--name":
            seen["--name"] += 1
            out += [opt, name]
        elif opt == "--label":
            continue
        elif opt == "--device":
            src = val.split(":")[0]
            if not src.startswith(TT_DEVICE + "/"):
                raise EditError(f"--device {val} is not one /dev/tenstorrent/<N> node; refusing a "
                                "container that can open chips it was not leased")
            devices.append(src[len(TT_DEVICE) + 1:])
            out += [opt, val]
        elif opt == "--volume":
            rest = val.partition(":")[2]
            dst = rest.split(":")[0]
            if dst == "/hf-hub":
                raise EditError("the printed command mounts /hf-hub (HF_HUB_CACHE is outside "
                                "HF_HOME); unset HF_HUB_CACHE for this test")
            if dst in ("/tensor-cache", "/hf"):
                seen[dst] += 1
                out += [opt, f"{tt_cache if dst == '/tensor-cache' else hf_dir}:{rest}"]
            else:
                out += [opt, val]
        elif opt == "--env":
            key = val.split("=", 1)[0]
            if key in ("HF_MODEL", "MODEL_WEIGHTS_DIR"):
                seen[key] += 1
                out += [opt, f"{key}={md}"]
            elif key == "TT_CACHE_PATH":
                if val != "TT_CACHE_PATH=/tensor-cache":
                    raise EditError(f"--env {val}: the tensor cache must be the /tensor-cache volume")
                seen[key] += 1
                out += [opt, val]
            elif key != "HF_TOKEN":
                out += [opt, val]
        else:
            out += [opt, val]
    _once(seen["--name"], "--name")
    _once(seen["/tensor-cache"], "--volume <dir>:/tensor-cache")
    _once(seen["/hf"], "--volume <dir>:/hf")
    _once(seen["TT_CACHE_PATH"], "--env TT_CACHE_PATH=/tensor-cache")
    _once(seen["HF_MODEL"], "--env HF_MODEL=<id>")
    if seen["MODEL_WEIGHTS_DIR"] > 1:
        raise EditError(f"expected at most one --env MODEL_WEIGHTS_DIR, found {seen['MODEL_WEIGHTS_DIR']}")
    want = sorted(str(d) for d in device_ids)
    if sorted(devices) != want:
        raise EditError(f"the printed command maps devices {sorted(devices)}; the lease gave {want}")
    out += ["--label", label, "--volume", f"{md}:{md}:ro"]
    for d in blobs:
        out += ["--volume", f"{d}:{d}:ro"]
    if seen["MODEL_WEIGHTS_DIR"] == 0:
        out += ["--env", f"MODEL_WEIGHTS_DIR={md}"]
    out.append(image)
    model_hits = [j for j, a in enumerate(server) if a == nearest]
    _once(len(model_hits), f"server argument {nearest!r}")
    server[model_hits[0]] = md
    for flag in ("--revision", "--tokenizer-revision"):
        at = [j for j, a in enumerate(server) if a == flag]
        if len(at) > 1:
            raise EditError(f"expected at most one {flag} in the server arguments, found {len(at)}")
        if at:
            del server[at[0]:at[0] + 2]
    ports = [server[j + 1] for j, a in enumerate(server[:-1]) if a == "--port"]
    if ports != [str(port)]:
        raise EditError(f"the server's --port is {ports}; the config's port is {port}")
    return out + server
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_container_template.py`
Expected: 18 passed.

- [ ] **Step 5: Mutations**

Run each against `python3 -m pytest -q -p no:cacheprovider tests/test_container_template.py -k <selector>`, see it fail, restore, delete the bytecode, see it pass:
1. In `edit_docker_argv`, change `out += [opt, f"{tt_cache if dst == '/tensor-cache' else hf_dir}:{rest}"]` to `out += [opt, val if dst == '/hf' else f"{tt_cache}:{rest}"]` (the operator's HF cache stays mounted). Selector `every_weight_path`: 1 failed.
2. Change `        out += ["--env", f"MODEL_WEIGHTS_DIR={md}"]` to `        pass`. Selector `every_weight_path`: 1 failed.
3. Change `            if not src.startswith(TT_DEVICE + "/"):` to `            if False:`. Selector `another_shape`: 1 failed (the whole-directory case).

- [ ] **Step 6: Run the suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -1` (expected `1346 passed, 1 skipped`), then:

```bash
git add orchard/skills/weights-swap-templates/serve_and_compare_container.py tests/container_fakes.py tests/test_container_template.py
git commit -m "Container template: edit the printed docker command with exact counts"
```

---

### Task 3: The container template runs the test, and the fakes it runs against

**Files:**
- Modify: `orchard/skills/weights-swap-templates/serve_and_compare.py` (docstring; `main` split into `load_reference`, `write_report`, `main`)
- Modify: `orchard/skills/weights-swap-templates/serve_and_compare_container.py` (docstring tail, imports, append the run steps)
- Create: `tests/fake_tt_model.py`, `tests/fake_docker.py`
- Test: `tests/test_container_template.py` (imports, append)

**Interfaces:**
- Consumes: Task 2's `edit_docker_argv`, `blob_dirs`, `EditError`; Task 1's container mode of `prepare_swap.py`.
- Produces in `serve_and_compare.py`: `load_reference(run_dir: Path) -> tuple[list[int], list[int], str | None]` (exits 2 when the reference has fewer than `N_TOKENS` ids); `write_report(cfg: dict, run_dir: Path, cache: Path, weights_env: dict, m: dict, tokenizer, reference, log_path: Path, extra: dict | None = None) -> dict` (writes `swap-check.json` next to `log_path`, prints and returns the draft).
- Produces in the container template: `fail(message, code=2)`, `docker(args, timeout=DOCKER_TIMEOUT_S)`, `class Container` (`poll()`, `returncode`, `save_log()`, `stop() -> bool`), `inside(path, root) -> bool`, `package_image(cfg) -> str`, `printed_command(cfg, ids) -> list[str]`, `main() -> int`; constants `DOCKER_TIMEOUT_S = 120.0`, `STOP_GRACE_S = 60`, `REQUIRED`. The script reads `ORCHARD_DEVICE_IDS` and `ORCHARD_TEST_LABEL` from its environment and writes `evidence/docker-argv.json`, `evidence/server.log`, `evidence/swap-check.json` (extra keys `kind`, `package`, `profile`, `chips`, `device_ids`, `docker_argv`, `hf_isolated`, `container_stopped`).

- [ ] **Step 1: Write the failing tests and the fakes**

Create `tests/fake_docker.py`:

```python
"""A stand-in for `docker`, for tests/test_container_template.py. It opens no device.

State lives in the directory $FAKE_DOCKER_STATE: config.json ({"image", "fake_server",
"server_config"}), one containers/<id>.json per container docker would still list, a log per
container and runs.jsonl (every `docker run` argv). Supported:

- `run --detach ... <image> <server args>`: refuses (exit 125) a volume whose source does not
  exist, and a link in an identically mounted directory whose target lies outside every mounted
  source (it would not resolve inside a real container). Otherwise it starts
  tests/fake_swap_server.py in its own session with only the `--env` variables (plus PATH), the
  published port and the model the server arguments name, and prints the container id.
- `inspect --format ... <id>`: "true 0" while the server runs, "false 1" after; exit 1 if unknown.
- `logs <id>`, `stop -t N <id>` (SIGTERM to the group, then SIGKILL), `rm --force <id>`.
- `ps --all --quiet --filter id=<id>` and `ps --quiet --filter label=<label>`.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

STATE = Path(os.environ.get("FAKE_DOCKER_STATE", "/nonexistent"))
FLAGS = {"--detach", "--rm"}


def records() -> dict:
    d = STATE / "containers"
    d.mkdir(parents=True, exist_ok=True)
    return {p.stem: json.loads(p.read_text()) for p in d.glob("*.json")}


def alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run(args: list[str]) -> int:
    cfg = json.loads((STATE / "config.json").read_text())
    with open(STATE / "runs.jsonl", "a") as f:
        f.write(json.dumps(args) + "\n")
    k = args.index(cfg["image"])
    opts, server = args[:k], args[k + 1:]
    env, volumes, labels, port, i = {}, [], [], None, 0
    while i < len(opts):
        if opts[i] in FLAGS:
            i += 1
            continue
        flag, val = opts[i], opts[i + 1]
        i += 2
        if flag == "--env":
            key, _, value = val.partition("=")
            env[key] = value
        elif flag == "--volume":
            volumes.append(val.split(":"))
        elif flag == "--label":
            labels.append(val)
        elif flag == "--publish":
            port = val.split(":")[0]
    for src, *_ in volumes:
        if not os.path.isdir(src):
            print(f"docker: bind source path does not exist: {src}", file=sys.stderr)
            return 125
    sources = [os.path.realpath(v[0]) for v in volumes]
    for src, dst, *_ in volumes:
        if src != dst:
            continue
        for p in Path(src).iterdir():
            target = os.path.realpath(p)
            if p.is_symlink() and not any(os.path.commonpath([target, s]) == s for s in sources):
                print(f"docker: {p} links to {target}, which no volume mounts", file=sys.stderr)
                return 125
    cid = f"fakecid{len((STATE / 'runs.jsonl').read_text().splitlines()):04d}"
    log = open(STATE / f"{cid}.log", "wb")
    model = server[server.index("serve") + 1]
    proc = subprocess.Popen([sys.executable, cfg["fake_server"], "--config", cfg["server_config"],
                             "--port", port, "--model", model],
                            env={**env, "PATH": os.environ.get("PATH", "")}, stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    (STATE / "containers" / f"{cid}.json").write_text(json.dumps(
        {"pid": proc.pid, "labels": labels, "argv": args}))
    print(cid)
    return 0


def stop_group(pid: int, wait_s: float = 5.0) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.monotonic() + wait_s
    while time.monotonic() < end and alive(pid):
        time.sleep(0.05)
    if alive(pid):
        os.killpg(pid, signal.SIGKILL)


def main() -> int:
    args = sys.argv[1:]
    cmd, rest = args[0], args[1:]
    recs = records()
    if cmd == "run":
        return run(rest)
    if cmd == "ps":
        filt = rest[rest.index("--filter") + 1]
        key, _, value = filt.partition("=")
        for cid, rec in recs.items():
            if (key == "id" and cid == value) or (key == "label" and value in rec["labels"]):
                print(cid)
        return 0
    cid = rest[-1]
    if cid not in recs:
        print(f"Error: No such container: {cid}", file=sys.stderr)
        return 1
    pid = recs[cid]["pid"]
    if cmd == "inspect":
        print("true 0" if alive(pid) else "false 1")
    elif cmd == "logs":
        sys.stdout.write((STATE / f"{cid}.log").read_text(errors="replace"))
    elif cmd == "stop":
        stop_group(pid)
        print(cid)
    elif cmd == "rm":
        stop_group(pid, wait_s=0.0)
        (STATE / "containers" / f"{cid}.json").unlink()
        print(cid)
    else:
        print(f"fake docker: {cmd} is not supported", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Create `tests/fake_tt_model.py`:

```python
"""A stand-in for `tt-model`, for tests/test_container_template.py.

It answers only `tt-model serve <package> ... --print`: it prints a note line and then the docker
command from tests/container_fakes.py, as tt-model's --print does, and starts nothing. Every call
is appended to the "calls" file named in the JSON config at $FAKE_TT_MODEL_CONFIG, with HOME and
HF_HOME, so a test can check what the template asked for. The config's "printed" dict passes
keyword arguments to printed_argv (for example {"whole_dir": true}).
"""
from __future__ import annotations

import json
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from container_fakes import printed_argv  # noqa: E402


def main() -> int:
    args = sys.argv[1:]
    with open(os.environ["FAKE_TT_MODEL_CONFIG"], encoding="utf-8") as f:
        cfg = json.load(f)
    with open(cfg["calls"], "a", encoding="utf-8") as f:
        f.write(json.dumps({"argv": args, "HOME": os.environ.get("HOME"),
                            "HF_HOME": os.environ.get("HF_HOME")}) + "\n")
    if args[:1] != ["serve"] or "--print" not in args:
        print("fake tt-model: only `serve <package> ... --print` is supported", file=sys.stderr)
        return 1
    opt = lambda name: args[args.index(name) + 1]      # noqa: E731
    ids = [int(i) for i in opt("--device-id").split(",")]
    print("• profile 'batch32' (the author's default)")
    print(shlex.join(printed_argv(hf=os.environ["HF_HOME"], pkg_cache=cfg["pkg_cache"],
                                  port=int(opt("--port")), device_ids=ids, **cfg.get("printed", {}))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

In `tests/test_container_template.py`, replace:

```python
tests/fake_swap_server.py, which opens no device."""
import importlib.util
from pathlib import Path

```

with:

```python
tests/fake_swap_server.py, which opens no device."""
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

```

Append to `tests/test_container_template.py`:

```python


# ---- the whole script, against fake tt-model and docker -------------------------------------------

TESTS = Path(__file__).resolve().parent
FAKE_SERVER, FAKE_TT_MODEL, FAKE_DOCKER = (TESTS / "fake_swap_server.py", TESTS / "fake_tt_model.py",
                                           TESTS / "fake_docker.py")


@pytest.fixture
def crig(tmp_path):
    """An operator home with the package installed and both models in its HF cache, a run with a
    stage 1 reference, one configuration directory prepared by prepare_swap.py, and fake tt-model
    and docker first on PATH. The finalizer kills any fake server a test left running."""
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from test_weights_swap_templates import GENERATED, PROMPT_IDS, VOCAB, free_port, make_snapshot

    home = tmp_path / "operator-home"
    hf = home / ".cache" / "huggingface"
    pkg_cache = home / ".cache" / "tt-model" / "qwen3.8-27b-p300x2"
    for d in ("cache", "weights", "tensors"):
        (pkg_cache / d).mkdir(parents=True)
    (home / ".cache" / "tt-model" / "installed.json").write_text(json.dumps(
        {PACKAGE: {"repo_id": PACKAGE, "container": True, "image": IMAGE, "profile": "batch32"}}))
    tok = Tokenizer(WordLevel({w: i for i, w in enumerate(VOCAB)} | {"[UNK]": len(VOCAB)},
                              unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(tmp_path / "tokenizer.json"))
    base = make_snapshot(hf / "hub" / "models--Qwen--Qwen3.8-27B", {"config.json": '{"base": true}'})
    new = make_snapshot(hf / "hub" / "models--Altworld--Hemmingway-1",
                        {"tokenizer.json": (tmp_path / "tokenizer.json").read_text(),
                         "model-00001-of-00001.safetensors": "new weights"})
    run = tmp_path / "run"
    ref = run / "stages" / "1" / "evidence" / "reference"
    ref.mkdir(parents=True)
    (ref / "prompt-ids.json").write_text(json.dumps({"prompt_ids": PROMPT_IDS}))
    (ref / "generated-ids.json").write_text(json.dumps(
        {"generated_ids": GENERATED, "generated_text": " ".join(VOCAB[i] for i in GENERATED)}))
    cdir = run / "stages" / "4" / "configs" / "4"
    cdir.mkdir(parents=True)
    for name in ("prepare_swap.py", "serve_and_compare.py", "serve_and_compare_container.py"):
        shutil.copy(TEMPLATES / name, cdir)
    cfg = {"run_dir": str(run), "nearest_model_id": NEAREST, "base_snapshot": str(base),
           "new_snapshot": str(new), "new_model_id": "Altworld/Hemmingway-1",
           "tt_cache": str(tmp_path / "orchard-cache" / "hemmingway-1" / "4chip-p300x2" / "tt_cache"),
           "hf_home": str(hf), "operator_home": str(home), "port": free_port(), "package": PACKAGE,
           "profile": "batch32", "chips": 4, "health_timeout_s": 30}
    (cdir / "swap_config.json").write_text(json.dumps(cfg))
    prep = subprocess.run([sys.executable, str(cdir / "prepare_swap.py")], capture_output=True,
                          text=True, timeout=60)
    assert prep.returncode == 0, prep.stdout + prep.stderr
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, script in (("tt-model", FAKE_TT_MODEL), ("docker", FAKE_DOCKER)):
        (bin_dir / name).write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        (bin_dir / name).chmod(0o755)
    state = tmp_path / "docker-state"
    state.mkdir()
    pid_file, server_cfg = tmp_path / "server-pid.json", tmp_path / "server.json"
    (state / "config.json").write_text(json.dumps({"image": IMAGE, "fake_server": str(FAKE_SERVER),
                                                   "server_config": str(server_cfg)}))
    tt_cfg, calls = tmp_path / "tt-model.json", tmp_path / "tt-model-calls.jsonl"
    rig = {"cdir": cdir, "cfg": cfg, "home": home, "hf": hf, "pkg_cache": pkg_cache, "state": state,
           "pid_file": pid_file, "calls": calls}

    def start(mode="perfect", printed=None, env_change=None, **overrides):
        (cdir / "swap_config.json").write_text(json.dumps(cfg | overrides))
        server_cfg.write_text(json.dumps({"mode": mode, "vocab": VOCAB, "prompt_ids": PROMPT_IDS,
                                          "generated_ids": GENERATED, "model": str(cdir / "model-dir"),
                                          "pid_file": str(pid_file)}))
        tt_cfg.write_text(json.dumps({"calls": str(calls), "pkg_cache": str(pkg_cache),
                                      "printed": printed or {}}))
        env = {k: v for k, v in os.environ.items()
               if k not in ("MODEL_WEIGHTS_DIR", "HF_MODEL", "HF_HUB_CACHE")}
        env.update(PATH=f"{bin_dir}:{env['PATH']}", FAKE_DOCKER_STATE=str(state),
                   FAKE_TT_MODEL_CONFIG=str(tt_cfg), ORCHARD_DEVICE_IDS="0,1,2,3",
                   ORCHARD_TEST_LABEL=LABEL)
        for k, v in (env_change or {}).items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
        return subprocess.run([sys.executable, str(cdir / "serve_and_compare_container.py")],
                              capture_output=True, text=True, timeout=180, env=env)

    rig["start"] = start
    try:
        yield rig
    finally:
        for rec in (state / "containers").glob("*.json"):
            pid = json.loads(rec.read_text())["pid"]
            try:
                os.killpg(pid, signal.SIGKILL)
                print(f"fixture killed a leaked fake container group {pid}", file=sys.stderr)
            except ProcessLookupError:
                pass


def docker_runs(rig) -> list[list[str]]:
    path = rig["state"] / "runs.jsonl"
    return [json.loads(ln) for ln in path.read_text().splitlines()] if path.exists() else []


def leftover(rig) -> list[str]:
    return sorted(p.stem for p in (rig["state"] / "containers").glob("*.json"))


def server_gone(rig, within=5.0) -> bool:
    pgid = json.loads(rig["pid_file"].read_text())["pgid"]
    end = time.monotonic() + within
    while time.monotonic() < end:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_the_container_serves_the_new_weights_and_is_removed_afterwards(crig):
    r = crig["start"]()
    assert r.returncode == 0, r.stdout + r.stderr
    md = str(crig["cdir"] / "model-dir")
    seen = json.loads(crig["pid_file"].read_text())
    assert seen["model_arg"] == md
    assert seen["env"]["MODEL_WEIGHTS_DIR"] == md and seen["env"]["HF_MODEL"] == md
    assert seen["env"]["TT_CACHE_PATH"] == "/tensor-cache" and seen["env"]["HF_HOME"] == "/hf"
    [argv] = docker_runs(crig)
    vols = pairs(argv, "--volume")
    assert f"{crig['cfg']['tt_cache']}:/tensor-cache" in vols
    assert f"{crig['cdir'] / 'hf-isolated'}:/hf" in vols
    assert not any(v.split(":")[0] == str(crig["hf"]) for v in vols)
    assert pairs(argv, "--label") == [LABEL]
    rep = json.loads((crig["cdir"] / "evidence" / "swap-check.json").read_text())
    assert rep["result_draft"]["top1_agreement"] == 1.0 and rep["result_draft"]["coherent"] is True
    assert rep["result_draft"]["evidence"] == ["stages/4/configs/4/evidence/swap-check.json",
                                               "stages/4/configs/4/evidence/server.log"]
    assert rep["kind"] == "container" and rep["container_stopped"] is True and rep["device_ids"] == [0, 1, 2, 3]
    assert json.loads((crig["cdir"] / "evidence" / "docker-argv.json").read_text()) == ["docker", "run"] + argv
    [call] = [json.loads(ln) for ln in crig["calls"].read_text().splitlines()]
    assert call["HOME"] == str(crig["home"]) and call["HF_HOME"] == str(crig["hf"])
    assert call["argv"][:2] == ["serve", PACKAGE] and call["argv"][-1] == "--print"
    assert call["argv"][call["argv"].index("--device-id") + 1] == "0,1,2,3"
    assert (Path(crig["cfg"]["tt_cache"]) / ".orchard-model").read_text() == "Altworld/Hemmingway-1"
    assert leftover(crig) == [] and server_gone(crig)


@pytest.mark.parametrize("mode,overrides,code", [("http500", {}, 5), ("die", {}, 4),
                                                 ("perfect", {"test_raise_after_ready": True}, None)])
def test_the_container_is_stopped_and_removed_on_every_exit_path(crig, mode, overrides, code):
    r = crig["start"](mode, **overrides)
    if code is None:
        assert r.returncode not in (0, 2, 3, 4, 5), r.stdout + r.stderr
    else:
        assert r.returncode == code, r.stdout + r.stderr
    assert leftover(crig) == [], "docker still lists the test container"
    assert server_gone(crig), "the fake container's server is still running"


def test_a_server_that_exits_shows_the_end_of_its_log(crig):
    r = crig["start"]("die")
    assert "fake server log line 69" in r.stdout and "exited with code 1" in r.stdout


def test_a_cache_inside_the_package_caches_is_refused_before_anything_starts(crig):
    r = crig["start"](tt_cache=str(crig["pkg_cache"] / "tensors"))
    assert r.returncode == 3, r.stdout + r.stderr
    assert "where the packages keep their own tensor caches" in r.stderr
    assert docker_runs(crig) == [] and not crig["calls"].exists()


def test_a_cache_marked_for_another_model_is_refused(crig):
    cache = Path(crig["cfg"]["tt_cache"])
    cache.mkdir(parents=True)
    (cache / ".orchard-model").write_text("Qwen/Qwen3.8-27B")
    (cache / "layer0.bin").write_text("base model tensors")
    assert crig["start"]().returncode == 3
    assert docker_runs(crig) == []


def test_a_printed_command_that_maps_every_device_exits_2_and_starts_nothing(crig):
    r = crig["start"](printed={"whole_dir": True})
    assert r.returncode == 2, r.stdout + r.stderr
    assert "not one /dev/tenstorrent/<N> node" in r.stderr
    assert docker_runs(crig) == []


@pytest.mark.parametrize("env_change", [{"ORCHARD_TEST_LABEL": None}, {"ORCHARD_DEVICE_IDS": "0,1"},
                                        {"ORCHARD_DEVICE_IDS": None}])
def test_without_the_supervisors_variables_it_exits_2(crig, env_change):
    r = crig["start"](env_change=env_change)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "ORCHARD_TEST_LABEL and ORCHARD_DEVICE_IDS" in r.stderr
    assert docker_runs(crig) == [] and not crig["calls"].exists()


def test_a_package_that_is_not_installed_exits_2(crig):
    r = crig["start"](package="someone/else-p300x2")
    assert r.returncode == 2 and "is not installed" in r.stderr
    assert docker_runs(crig) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_container_template.py tests/test_weights_swap_templates.py`
Expected: 12 FAIL, 40 passed. The script has no `main` yet, so it exits 0 having done nothing; each test fails on a missing file (the fake server's pid file or `tt-model-calls.jsonl`) or on the exit code.

- [ ] **Step 3: Implement**

In `orchard/skills/weights-swap-templates/serve_and_compare.py`, replace:

```python
The server rejects logprobs and sampling parameters, so requests carry only model, prompt,
max_tokens and temperature.
"""
from __future__ import annotations
```

with:

```python
The server rejects logprobs and sampling parameters, so requests carry only model, prompt,
max_tokens and temperature.

serve_and_compare_container.py (stage 4) is copied next to this file and imports guard_cache,
load_reference, measure, write_report and ServerHTTPError from it, so both templates measure the
same way.
"""
from __future__ import annotations
```

In `orchard/skills/weights-swap-templates/serve_and_compare.py`, replace:

```python


def main() -> int:
    from tokenizers import Tokenizer      # imported here so a missing package fails before Popen
```

with:

```python


def load_reference(run_dir: Path) -> tuple[list[int], list[int], str | None]:
    """The stage 1 prompt ids, generated ids and generated text. Exits 2 when the reference holds
    fewer than N_TOKENS generated ids."""
    ref = run_dir / "stages" / "1" / "evidence" / "reference"
    prompt_ids = json.loads((ref / "prompt-ids.json").read_text(encoding="utf-8"))["prompt_ids"]
    gen = json.loads((ref / "generated-ids.json").read_text(encoding="utf-8"))
    if len(gen["generated_ids"]) < N_TOKENS:
        print(f"serve_and_compare: the reference has {len(gen['generated_ids'])} generated ids; "
              f"{N_TOKENS} are needed", file=sys.stderr)
        sys.exit(2)
    return prompt_ids, gen["generated_ids"], gen.get("generated_text")


def write_report(cfg: dict, run_dir: Path, cache: Path, weights_env: dict, m: dict, tokenizer,
                 reference, log_path: Path, extra: dict | None = None) -> dict:
    """Step (g). Writes swap-check.json next to log_path and prints the result draft. `extra` adds
    keys to the report; serve_and_compare_container.py records its docker command there."""
    prompt_ids, generated_ids, generated_text = reference
    top1 = m["matches"] / N_TOKENS
    coh = coherence(m["free_run_text"])
    swap_json = log_path.parent / "swap-check.json"
    draft = {"serves": True,
             "server_ready_s": m["server_ready_s"],
             "coherent": coh["coherent"],
             "free_run_text": m["free_run_text"][:200],
             "top1_agreement": top1,
             "n_tokens": N_TOKENS,
             "cache_dir": str(cache),
             "evidence": [os.path.relpath(swap_json, run_dir), os.path.relpath(log_path, run_dir)]}
    report = {"label": "measured", "new_model_id": cfg["new_model_id"],
              "model_dir": weights_env["MODEL_WEIGHTS_DIR"], "port": cfg["port"],
              "weights_dir_env": weights_env, "hf_model_env": weights_env["HF_MODEL"],
              "server_ready_s": m["server_ready_s"], "n_tokens": N_TOKENS,
              "matches": m["matches"], "top1_agreement": top1, **coh,
              "prompt_ids": prompt_ids, "reference_generated_ids": generated_ids[:N_TOKENS],
              "reference_generated_text": generated_text, "free_run_text": m["free_run_text"],
              "free_run_ids": tokenizer.encode(m["free_run_text"], add_special_tokens=False).ids,
              "teacher_forced": m["forced"], "mismatches": m["mismatches"], **(extra or {}),
              "result_draft": draft}
    swap_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(draft, indent=2, ensure_ascii=False))
    return draft


def main() -> int:
    from tokenizers import Tokenizer      # imported here so a missing package fails before Popen
```

In `orchard/skills/weights-swap-templates/serve_and_compare.py`, replace:

```python
    cfg = load_config()
    run_dir = Path(cfg["run_dir"]).resolve()
    ref = run_dir / "stages" / "1" / "evidence" / "reference"
    prompt_ids = json.loads((ref / "prompt-ids.json").read_text(encoding="utf-8"))["prompt_ids"]
    gen = json.loads((ref / "generated-ids.json").read_text(encoding="utf-8"))
    generated_ids, generated_text = gen["generated_ids"], gen.get("generated_text")
    if len(generated_ids) < N_TOKENS:
        print(f"serve_and_compare: the reference has {len(generated_ids)} generated ids; "
              f"{N_TOKENS} are needed", file=sys.stderr)
        return 2
    tokenizer = Tokenizer.from_file(str(STAGE_DIR / "model-dir" / "tokenizer.json"))
    cache = Path(cfg["tt_cache"])
```

with:

```python
    cfg = load_config()
    run_dir = Path(cfg["run_dir"]).resolve()
    reference = load_reference(run_dir)
    tokenizer = Tokenizer.from_file(str(STAGE_DIR / "model-dir" / "tokenizer.json"))
    cache = Path(cfg["tt_cache"])
```

In `orchard/skills/weights-swap-templates/serve_and_compare.py`, replace:

```python
                            stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        m = measure(cfg, proc, log_path, tokenizer, prompt_ids, generated_ids)
    except ServerHTTPError as exc:
        print(f"serve_and_compare: the server returned an error: {exc}")
```

with:

```python
                            stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        m = measure(cfg, proc, log_path, tokenizer, reference[0], reference[1])
    except ServerHTTPError as exc:
        print(f"serve_and_compare: the server returned an error: {exc}")
```

In `orchard/skills/weights-swap-templates/serve_and_compare.py`, replace:

```python
        stop_server(proc)
        log.close()

    top1 = m["matches"] / N_TOKENS
    coh = coherence(m["free_run_text"])
    swap_json = evidence / "swap-check.json"
    draft = {"serves": True,
             "server_ready_s": m["server_ready_s"],
             "coherent": coh["coherent"],
             "free_run_text": m["free_run_text"][:200],
             "top1_agreement": top1,
             "n_tokens": N_TOKENS,
             "cache_dir": str(cache),
             "evidence": [os.path.relpath(swap_json, run_dir), os.path.relpath(log_path, run_dir)]}
    report = {"label": "measured", "new_model_id": cfg["new_model_id"],
              "model_dir": model_dir, "port": cfg["port"],
              "weights_dir_env": weights_env, "hf_model_env": weights_env["HF_MODEL"],
              "server_ready_s": m["server_ready_s"], "n_tokens": N_TOKENS,
              "matches": m["matches"], "top1_agreement": top1, **coh,
              "prompt_ids": prompt_ids, "reference_generated_ids": generated_ids[:N_TOKENS],
              "reference_generated_text": generated_text, "free_run_text": m["free_run_text"],
              "free_run_ids": tokenizer.encode(m["free_run_text"], add_special_tokens=False).ids,
              "teacher_forced": m["forced"], "mismatches": m["mismatches"],
              "result_draft": draft}
    swap_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(draft, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
```

with:

```python
        stop_server(proc)
        log.close()
    write_report(cfg, run_dir, cache, weights_env, m, tokenizer, reference, log_path)
    return 0

if __name__ == "__main__":
```

In `orchard/skills/weights-swap-templates/serve_and_compare_container.py`, replace:

```python
Any option this script does not know exits 2. A new option in tt-model's output must be read by a
person before a test trusts it.
"""
from __future__ import annotations

import os
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
TT_DEVICE = "/dev/tenstorrent"
VALUE_OPTIONS = {"--name", "--user", "--label", "--device", "--ipc", "--mount", "--volume", "--env",
```

with:

```python
Any option this script does not know exits 2. A new option in tt-model's output must be read by a
person before a test trusts it.

Steps when run: read swap_config.json (keys below) and the supervisor's two variables; refuse a
tt_cache inside ~/.cache/tt-model and apply serve_and_compare.py's cache guard (exit 3); look up
the package's image in ~/.cache/tt-model/installed.json; ask `tt-model serve ... --print` with
HOME and HF_HOME set to the operator's (the shell this runs in has its own HOME); edit the argv and
save it as evidence/docker-argv.json; create every volume source as this user (docker would
create a missing one as root); `docker run`; then measure exactly as serve_and_compare.py does
(its `measure`, with the container standing in for the process) and write evidence/swap-check.json
through its `write_report`. The `try` starts on the line after `docker run` returns, and its
`finally` runs `docker stop` (SIGTERM, then SIGKILL after STOP_GRACE_S), saves `docker logs` to
evidence/server.log, runs `docker rm --force` and checks that docker no longer lists the
container.

Config keys: run_dir, nearest_model_id, new_model_id, tt_cache, hf_home, operator_home, port,
package, profile, chips (and health_timeout_s, test_raise_after_ready for tests, as in
serve_and_compare.py).

Exit codes: 0 when the measurements completed, whatever they say. 2 the config, the supervisor's
variables or the printed command could not be used; nothing was started. 3 the tensor cache was
refused. 4 the container did not start, exited, or never became healthy. 5 the server answered a
request with an HTTP error.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE_DIR))       # serve_and_compare.py is copied next to this file
DOCKER_TIMEOUT_S = 120.0
STOP_GRACE_S = 60
REQUIRED = ("run_dir", "nearest_model_id", "new_model_id", "tt_cache", "hf_home", "operator_home",
            "port", "package", "profile", "chips")
TT_DEVICE = "/dev/tenstorrent"
VALUE_OPTIONS = {"--name", "--user", "--label", "--device", "--ipc", "--mount", "--volume", "--env",
```

Append to `orchard/skills/weights-swap-templates/serve_and_compare_container.py`:

```python


# ---- running the test ----------------------------------------------------------------------------

def fail(message: str, code: int = 2) -> None:
    print(f"serve_and_compare_container: {message}", file=sys.stderr)
    sys.exit(code)


def docker(args: list[str], timeout: float = DOCKER_TIMEOUT_S) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


class Container:
    """The started container, with what serve_and_compare.measure asks of a Popen: poll() and
    returncode."""

    def __init__(self, cid: str, log_path: Path):
        self.cid, self.log_path, self.returncode = cid, log_path, None

    def save_log(self) -> None:
        r = docker(["logs", self.cid])
        self.log_path.write_text(r.stdout + r.stderr, encoding="utf-8")

    def poll(self):
        r = docker(["inspect", "--format", "{{.State.Running}} {{.State.ExitCode}}", self.cid])
        parts = r.stdout.split()
        if r.returncode == 0 and parts[:1] == ["true"]:
            return None
        self.save_log()                 # measure prints the log's tail when the server is gone
        self.returncode = int(parts[1]) if r.returncode == 0 and len(parts) == 2 else -1
        return self.returncode

    def stop(self) -> bool:
        """Stop, save the log, remove. True when docker no longer lists the container."""
        docker(["stop", "-t", str(STOP_GRACE_S), self.cid], timeout=STOP_GRACE_S + DOCKER_TIMEOUT_S)
        self.save_log()
        docker(["rm", "--force", self.cid])
        left = docker(["ps", "--all", "--quiet", "--filter", f"id={self.cid}"])
        return left.returncode == 0 and not left.stdout.strip()


def inside(path: Path, root: Path) -> bool:
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def package_image(cfg: dict) -> str:
    path = Path(cfg["operator_home"]) / ".cache" / "tt-model" / "installed.json"
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))[cfg["package"]]
    except (OSError, ValueError, KeyError) as exc:
        fail(f"{cfg['package']} is not installed according to {path}: {exc!r}")
    if not entry.get("container") or not entry.get("image"):
        fail(f"{cfg['package']} is not a container package in {path}")
    return entry["image"]


def printed_command(cfg: dict, ids: list[int]) -> list[str]:
    env = dict(os.environ, HOME=str(cfg["operator_home"]), HF_HOME=str(cfg["hf_home"]))
    for key in ("HF_HUB_CACHE", "HF_TOKEN"):
        env.pop(key, None)
    argv = ["tt-model", "serve", cfg["package"], "--local-only", "--no-update-check", "--port",
            str(cfg["port"]), "--profile", cfg["profile"], "--device-id",
            ",".join(str(i) for i in ids), "--print"]
    r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=DOCKER_TIMEOUT_S)
    if r.returncode != 0:
        fail(f"{shlex.join(argv)} exited {r.returncode}: {(r.stderr or r.stdout)[-2000:]}")
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("docker run ")]
    if len(lines) != 1:
        fail(f"expected one 'docker run' line from {shlex.join(argv)}, found {len(lines)}:\n"
             f"{r.stdout[-2000:]}")
    return shlex.split(lines[0])


def main() -> int:
    from tokenizers import Tokenizer      # imported here so a missing package fails before docker
    from serve_and_compare import (ServerHTTPError, guard_cache, load_reference, measure,
                                   write_report)

    try:
        cfg = json.loads((STAGE_DIR / "swap_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"cannot read swap_config.json: {exc}")
    missing = [k for k in REQUIRED if cfg.get(k) in (None, "")]
    if missing:
        fail(f"swap_config.json is missing {missing}")
    label = os.environ.get("ORCHARD_TEST_LABEL", "")
    try:
        ids = [int(i) for i in os.environ.get("ORCHARD_DEVICE_IDS", "").split(",")]
    except ValueError:
        ids = []
    if not label or len(ids) != int(cfg["chips"]):
        fail(f"the supervisor sets ORCHARD_TEST_LABEL and ORCHARD_DEVICE_IDS ({cfg['chips']} ids) "
             f"for a hardware test; got {label!r} and {os.environ.get('ORCHARD_DEVICE_IDS')!r}")
    run_dir = Path(cfg["run_dir"]).resolve()
    reference = load_reference(run_dir)
    model_dir = STAGE_DIR / "model-dir"
    if not (model_dir / "tokenizer.json").exists():
        fail(f"{model_dir} has no tokenizer.json; run prepare_swap.py first")
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    cache = Path(cfg["tt_cache"])
    shared = Path(cfg["operator_home"]) / ".cache" / "tt-model"
    if inside(cache, shared):
        fail(f"tt_cache {cache} is inside {shared}, where the packages keep their own tensor "
             "caches. Use a new directory for this model and this configuration.", 3)
    guard_cache(cache, cfg["new_model_id"])
    image = package_image(cfg)
    hf_dir = STAGE_DIR / "hf-isolated"
    hf_dir.mkdir(exist_ok=True)
    try:
        argv = edit_docker_argv(printed_command(cfg, ids), image=image,
                                nearest=cfg["nearest_model_id"], model_dir=model_dir,
                                tt_cache=cache, hf_dir=hf_dir,
                                name=f"orchard-{cfg['chips']}chip-{cfg['port']}", label=label,
                                device_ids=ids, port=int(cfg["port"]), blobs=blob_dirs(model_dir))
    except EditError as exc:
        fail(str(exc))
    evidence = STAGE_DIR / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "docker-argv.json").write_text(json.dumps(argv, indent=1), encoding="utf-8")
    for flag, value in zip(argv, argv[1:]):
        if flag == "--volume":
            Path(value.split(":")[0]).mkdir(parents=True, exist_ok=True)
    log_path = evidence / "server.log"
    r = docker(argv[1:])
    if r.returncode != 0:
        print(f"serve_and_compare_container: docker run exited {r.returncode}: {r.stderr[-2000:]}")
        return 4
    container = Container(r.stdout.split()[-1], log_path)
    stopped = False
    try:
        m = measure(cfg, container, log_path, tokenizer, reference[0], reference[1])
    except ServerHTTPError as exc:
        print(f"serve_and_compare_container: the server returned an error: {exc}")
        sys.exit(5)
    finally:
        stopped = container.stop()
        if not stopped:
            print(f"serve_and_compare_container: docker still lists container {container.cid} "
                  "after stop and rm; the supervisor looks for it by its label", file=sys.stderr)
    weights_env = {"MODEL_WEIGHTS_DIR": str(model_dir), "HF_MODEL": str(model_dir)}
    write_report(cfg, run_dir, cache, weights_env, m, tokenizer, reference, log_path,
                 extra={"kind": "container", "package": cfg["package"], "profile": cfg["profile"],
                        "chips": int(cfg["chips"]), "device_ids": ids, "docker_argv": argv,
                        "hf_isolated": str(hf_dir), "container_stopped": stopped})
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_container_template.py tests/test_weights_swap_templates.py`
Expected: 52 passed. The stage 2 template's tests still pass after the split of its `main`.

- [ ] **Step 5: Mutations**

Selector against `tests/test_container_template.py`; fail, restore, delete the bytecode, pass:
1. In the container template's `main`, change `        stopped = container.stop()` to `        stopped = True`. Selector `every_exit_path`: 3 failed (docker still lists the container).
2. Change `    if inside(cache, shared):` to `    if False:`. Selector `package_caches`: 1 failed.
3. In `printed_command`, change `env = dict(os.environ, HOME=str(cfg["operator_home"]), HF_HOME=str(cfg["hf_home"]))` to `env = dict(os.environ)`. Selector `serves_the_new_weights`: 1 failed.

- [ ] **Step 6: Run the suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -1` (expected `1358 passed, 1 skipped`), then:

```bash
git add orchard/skills/weights-swap-templates/ tests/fake_tt_model.py tests/fake_docker.py tests/test_container_template.py
git commit -m "Container template: run the edited docker command, measure, always remove the container"
```

---

### Task 4: The stage 4 spec, disk need and gate for the weights-only path

**Files:**
- Modify: `orchard/defaults.py` (append), `orchard/stages.py` (docstring, imports, `StageSpec`, `gate_weights_swap`, new helpers and gate, `WEIGHTS_ONLY_STAGE_4`)
- Test: `tests/test_defaults.py` (append), `tests/test_stages.py` (append)

**Interfaces:**
- Consumes: `gate_mesh`, `inside`, `_load`, `_done` (existing in `orchard/stages.py`).
- Produces in `orchard.defaults`: `STAGE4_SWAP_DISK_GB = 110.0`, `TEST_DISK_GB = 40.0`.
- Produces in `orchard.stages`: `StageSpec.tests: bool = False`, `StageSpec.disk: float | None = None` (`disk_gb` returns it when set); `_swap_reasons(d: dict, where: str = "") -> list[str]` (the serves reason quotes a `failure` field, as main's `gate_weights_swap` does); `hw_record_path(stage_dir, chips: int) -> Path` (`<stage>/tests/<chips>/test-result.json`); `hw_record_problem(stage_dir, chips: int) -> str | None`; `gate_mesh_swap(stage_dir, run_dir, required=None) -> GateResult`; `WEIGHTS_ONLY_STAGE_4` (`skill="weights-swap-configs"`, `refs=("tt-device-usage",)`, `boards=2`, `marker="tests/plan.json"`, `tests=True`, `disk=STAGE4_SWAP_DISK_GB`). `spec_for` does not return it yet (Task 7). The names avoid a `test_` prefix, because pytest would collect an imported function named `test_*`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_defaults.py`:

```python


def test_the_weights_only_stage_4_disk_holds_three_measured_caches():
    # 34 GB (2-chip bundle) and 31 GB (4-chip container) measured; the 1-chip cache is assumed 34 GB.
    assert d.STAGE4_SWAP_DISK_GB >= 34 + 34 + 31
    assert d.TEST_DISK_GB >= 34


def test_the_stage_4_budget_holds_three_tests_and_a_park():
    # Three tests at the skill's 3600 s each (a first boot with cold caches took more than 26 min
    # on 2026-10-03, and the 4-chip conversion is not measured), one park and restore with a coder
    # boot within the cold-boot budget, and one agent step of TOOL_TIMEOUT_S each for prepare and
    # finish.
    need = (3 * 3600 + d.TT_MODEL_STOP_S + 2 * d.GOZER_RESET_S + d.COLD_BOOT_BUDGET_S
            + 2 * d.TOOL_TIMEOUT_S)
    assert d.STAGE_BUDGET_S[4] >= need
```

Append to `tests/test_stages.py`:

```python


# ---- stage 4 on the weights-only path -------------------------------------------------------------

def swap_config(n, ok=True, **change):
    """One stage 4 configuration entry as the weights-swap-configs skill writes it."""
    if not ok:
        return {"chips": n, "pass": False, "reason": "no package fits", **change}
    swap = f"stages/4/configs/{n}/evidence/swap-check.json"
    entry = {**{k: v for k, v in SWAP.items() if k != "evidence"}, "chips": n, "pass": True,
             "evidence": [swap, f"stages/4/tests/{n}/output.txt"]}
    entry.update(change)
    return entry


def mesh_stage(tmp_path, configs, records=None, drafts=None):
    """A run with stage 4's result.json, the supervisor's test records and each test's
    swap-check.json. `records` maps chips to a test-result override (None leaves it out)."""
    run = tmp_path / "run"
    sd = run / "stages" / "4"
    write(run, "stages/4/result.json", {"configs": configs})
    for c in configs:
        n = c["chips"]
        write(run, f"stages/4/tests/{n}/output.txt", "test output")
        rec = {"returncode": 0, "timed_out": False, "chips": [f"chip{i}" for i in range(n)]}
        override = (records or {}).get(n, {})
        if override is not None:
            write(run, f"stages/4/tests/{n}/test-result.json", {**rec, **override})
        draft = (drafts or {}).get(n, {"top1_agreement": c.get("top1_agreement")})
        write(run, f"stages/4/configs/{n}/evidence/swap-check.json", {"result_draft": draft})
    return sd, run


def test_weights_only_stage_4_is_a_list_of_tests_with_its_own_gate_marker_and_disk():
    from orchard.defaults import STAGE4_SWAP_DISK_GB
    from orchard.stages import WEIGHTS_ONLY_STAGE_4 as s4, gate_mesh_swap
    assert s4.number == 4 and s4.tests and s4.boards == 2
    assert s4.skill == "weights-swap-configs" and s4.gate is gate_mesh_swap
    assert s4.marker == "tests/plan.json" and s4.disk_gb == STAGE4_SWAP_DISK_GB
    assert s4.budget_s == STAGES[4].budget_s
    assert STAGES[4].disk_gb == 80.0 and not STAGES[4].tests       # the plan 4 table is unchanged


def test_the_swap_mesh_gate_passes_required_configurations_with_records_and_an_optional_failure(tmp_path):
    from orchard.stages import gate_mesh_swap
    g = gate_mesh_swap(*mesh_stage(tmp_path, [swap_config(2), swap_config(4), swap_config(1, ok=False)]),
                       required=(2, 4))
    assert g.ok, g.reasons
    assert "stages/4/tests/4/test-result.json" in g.evidence


def test_the_swap_mesh_gate_refuses_a_pass_for_a_configuration_whose_test_never_ran(tmp_path):
    from orchard.stages import gate_mesh_swap
    g = gate_mesh_swap(*mesh_stage(tmp_path, [swap_config(2), swap_config(4)], records={4: None}),
                       required=(2, 4))
    assert g.reasons == ("the 4-chip configuration claims a pass, but the supervisor has no record "
                         "of its test (tests/4/test-result.json)",)


@pytest.mark.parametrize("record,words", [
    ({"returncode": 4}, "its test exited 4"),
    ({"returncode": None, "timed_out": True}, "did not finish before its deadline"),
    ({"chips": ["a", "b"]}, "which is not 4 chips"),
])
def test_the_swap_mesh_gate_refuses_a_pass_whose_test_did_not_finish_on_its_chips(tmp_path, record, words):
    from orchard.stages import gate_mesh_swap
    g = gate_mesh_swap(*mesh_stage(tmp_path, [swap_config(4)], records={4: record}), required=(4,))
    assert len(g.reasons) == 1 and words in g.reasons[0], g.reasons


def test_the_swap_mesh_gate_holds_each_configuration_to_the_stage_2_bar(tmp_path):
    # The base weights standing in agreed 25 of 32 with the Hemmingway-1 reference.
    from orchard.stages import gate_mesh_swap
    g = gate_mesh_swap(*mesh_stage(tmp_path, [swap_config(4, top1_agreement=25 / 32)]), required=(4,))
    assert g.reasons == ("the 4-chip configuration: top1_agreement 0.78125 is below the minimum of 0.85",)


def test_the_swap_mesh_gate_checks_an_optional_configuration_that_claims_a_pass(tmp_path):
    from orchard.stages import gate_mesh_swap
    g = gate_mesh_swap(*mesh_stage(tmp_path, [swap_config(2), swap_config(1)], records={1: None}),
                       required=(2,))
    assert g.reasons == ("the 1-chip configuration claims a pass, but the supervisor has no record "
                         "of its test (tests/1/test-result.json)",)


def test_the_swap_mesh_gate_needs_the_tests_own_report_with_the_same_agreement(tmp_path):
    from orchard.stages import gate_mesh_swap
    missing = swap_config(4, evidence=["stages/4/tests/4/output.txt"])
    g = gate_mesh_swap(*mesh_stage(tmp_path, [missing]), required=(4,))
    assert g.reasons == ("the 4-chip configuration: evidence must include "
                         "stages/4/configs/4/evidence/swap-check.json",)
    g = gate_mesh_swap(*mesh_stage(tmp_path / "b", [swap_config(4)], drafts={4: {"top1_agreement": 0.5}}),
                       required=(4,))
    assert g.reasons == ("the 4-chip configuration: top1_agreement 0.94 is not the result_draft's in "
                         "stages/4/configs/4/evidence/swap-check.json",)


def test_the_swap_mesh_gate_keeps_the_mesh_gates_rules(tmp_path):
    from orchard.stages import gate_mesh_swap
    g = gate_mesh_swap(*mesh_stage(tmp_path, [swap_config(2)]), required=(2, 4))
    assert g.reasons == ("the 4-chip configuration is required and has no entry",)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_stages.py tests/test_defaults.py`
Expected: 11 FAIL, 92 passed: ten in `test_stages.py` with `ImportError: cannot import name 'WEIGHTS_ONLY_STAGE_4'` or `'gate_mesh_swap'`, and `test_the_weights_only_stage_4_disk_holds_three_measured_caches` with `AttributeError: ... 'STAGE4_SWAP_DISK_GB'`. `test_the_stage_4_budget_holds_three_tests_and_a_park` passes already: it pins the existing budget.

- [ ] **Step 3: Implement**

Append to `orchard/defaults.py`:

```python

# ---- stage 4 on the weights-only path: one hardware test per chip configuration -----------------
# Each configuration converts the new weights into its own tensor cache. Measured sizes for a
# Qwen3.8-27B-sized model: 34 GB for the 2-chip bundle (2026-09-30, and the Hemmingway-1 prototype
# on 2026-10-03) and 31 GB for the 4-chip plain container (rebuilt 2026-10-03). The 1-chip cache was
# not measured and is assumed to be the same size.
STAGE4_SWAP_DISK_GB = 110.0     # choice: three caches of about 34 GB (1, 2 and 4 chips) plus margin,
                                # checked on the run directory's disk before stage 4 starts
TEST_DISK_GB = 40.0             # choice: one cache plus margin, checked again before each
                                # configuration's test, so a resumed stage stops before a test that
                                # cannot finish its conversion
```

In `orchard/stages.py`, replace:

```python

The table holds one spec per stage. The path stage 0 chose can replace a spec: on the weights-only
path stage 2 uses the weights-swap-check skill and `gate_weights_swap`, and stage 3 is skipped
(`spec_for`, `run_path`).

```

with:

```python

The table holds one spec per stage. The path stage 0 chose can replace a spec: on the weights-only
path stage 2 uses the weights-swap-check skill and `gate_weights_swap`, stage 3 is skipped, and
stage 4 runs one hardware test per chip configuration (`WEIGHTS_ONLY_STAGE_4`, `gate_mesh_swap`)
(`spec_for`, `run_path`).

```

In `orchard/stages.py`, replace:

```python

from orchard.defaults import (COLD_START_S, LONG_STAGE_S, RUN_COLD_BOOT_CAP, RUN_ESCALATION_CAP,
                              RUN_WALL_CLOCK_S, STAGE2_PCC_MIN, STAGE_BUDGET_S, STAGE_DISK_GB,
                              SWAP_MIN_TOKENS, SWAP_TOP1_MIN)
from orchard.tiers import TierConfig

```

with:

```python

from orchard.defaults import (COLD_START_S, LONG_STAGE_S, RUN_COLD_BOOT_CAP, RUN_ESCALATION_CAP,
                              RUN_WALL_CLOCK_S, STAGE2_PCC_MIN, STAGE4_SWAP_DISK_GB, STAGE_BUDGET_S,
                              STAGE_DISK_GB, SWAP_MIN_TOKENS, SWAP_TOP1_MIN)
from orchard.tiers import TierConfig

```

In `orchard/stages.py`, replace:

```python
    marker: str | None                  # resume marker: a file in the stage directory
    skip: str | None = None             # why plan 4 skips this stage

    @property
```

with:

```python
    marker: str | None                  # resume marker: a file in the stage directory
    skip: str | None = None             # why plan 4 skips this stage
    tests: bool = False                 # the hardware phase runs a list of tests (hw_tests.json,
                                        # orchard/hwtests.py) in place of one hw_test.json
    disk: float | None = None           # free disk the stage needs, when it differs from STAGE_DISK_GB

    @property
```

In `orchard/stages.py`, replace:

```python
    @property
    def disk_gb(self) -> float:
        return STAGE_DISK_GB[self.number]


```

with:

```python
    @property
    def disk_gb(self) -> float:
        return self.disk if self.disk is not None else STAGE_DISK_GB[self.number]


```

In `orchard/stages.py`, replace:

```python
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    if d.get("serves") is not True:
        reason = f"serves must be true (the server started and answered), got {d.get('serves')!r}"
        failure = d.get("failure")
        if isinstance(failure, str) and failure.strip():
```

with:

```python
    if err:
        return GateResult(False, (err,))
    reasons, seen = _swap_reasons(d), []
    _evidence(run_dir, d.get("evidence"), "result.json", reasons, seen)
    return _done(reasons, seen)


def _swap_reasons(d: dict, where: str = "") -> list[str]:
    """The swap fields of one result (stage 2's result.json, or one stage 4 configuration), checked
    against the stage 2 bar. Each failing field gets its own reason, which names the field."""
    reasons = []
    if d.get("serves") is not True:
        reason = f"{where}serves must be true (the server started and answered), got {d.get('serves')!r}"
        failure = d.get("failure")
        if isinstance(failure, str) and failure.strip():
```

In `orchard/stages.py`, replace:

```python
        reasons.append(reason)
    if d.get("coherent") is not True:
        reasons.append(f"coherent must be true (the free-run text is readable), got {d.get('coherent')!r}")
    n = d.get("n_tokens")
    if isinstance(n, bool) or not isinstance(n, int) or n < SWAP_MIN_TOKENS:
        reasons.append(f"n_tokens must be a whole number of at least {SWAP_MIN_TOKENS}, got {n!r}")
    top1 = d.get("top1_agreement")
    if not _number(top1) or not 0 <= top1 <= 1:
        reasons.append(f"top1_agreement must be a fraction from 0 to 1, got {top1!r}")
    elif top1 < SWAP_TOP1_MIN:
        reasons.append(f"top1_agreement {top1} is below the minimum of {SWAP_TOP1_MIN}")
    ready = d.get("server_ready_s")
    if not _number(ready) or ready <= 0:
        reasons.append(f"server_ready_s must be a positive number of seconds, got {ready!r}")
    _evidence(run_dir, d.get("evidence"), "result.json", reasons, seen)
    return _done(reasons, seen)


```

with:

```python
        reasons.append(reason)
    if d.get("coherent") is not True:
        reasons.append(f"{where}coherent must be true (the free-run text is readable), got {d.get('coherent')!r}")
    n = d.get("n_tokens")
    if isinstance(n, bool) or not isinstance(n, int) or n < SWAP_MIN_TOKENS:
        reasons.append(f"{where}n_tokens must be a whole number of at least {SWAP_MIN_TOKENS}, got {n!r}")
    top1 = d.get("top1_agreement")
    if not _number(top1) or not 0 <= top1 <= 1:
        reasons.append(f"{where}top1_agreement must be a fraction from 0 to 1, got {top1!r}")
    elif top1 < SWAP_TOP1_MIN:
        reasons.append(f"{where}top1_agreement {top1} is below the minimum of {SWAP_TOP1_MIN}")
    ready = d.get("server_ready_s")
    if not _number(ready) or ready <= 0:
        reasons.append(f"{where}server_ready_s must be a positive number of seconds, got {ready!r}")
    return reasons


```

In `orchard/stages.py`, replace:

```python
        if chips not in listed:
            reasons.append(f"the {chips}-chip configuration is required and has no entry")
    return _done(reasons, seen)

```

with:

```python
        if chips not in listed:
            reasons.append(f"the {chips}-chip configuration is required and has no entry")
    return _done(reasons, seen)


def hw_record_path(stage_dir, chips: int) -> Path:
    """Where the supervisor records one configuration's hardware test (orchard/hwtests.py). Only
    the supervisor writes this file."""
    return Path(stage_dir) / "tests" / str(chips) / "test-result.json"


def hw_record_problem(stage_dir, chips: int) -> str | None:
    """Why the supervisor's record does not show a finished test on `chips` chips, or None."""
    path = hw_record_path(stage_dir, chips)
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return f"the supervisor has no record of its test (tests/{chips}/test-result.json)"
    except (OSError, ValueError) as exc:
        return f"tests/{chips}/test-result.json is not readable JSON: {exc}"
    if not isinstance(rec, dict):
        return f"tests/{chips}/test-result.json is not a JSON object"
    if rec.get("timed_out") is not False:
        return f"its test did not finish before its deadline (timed_out {rec.get('timed_out')!r})"
    if rec.get("returncode") != 0:
        return f"its test exited {rec.get('returncode')!r}"
    if not isinstance(rec.get("chips"), list) or len(rec["chips"]) != chips:
        return f"its test ran on {rec.get('chips')!r}, which is not {chips} chips"
    return None


def gate_mesh_swap(stage_dir, run_dir, required=None) -> GateResult:
    """Stage 4 on the weights-only path: `gate_mesh`, and for every configuration that claims a
    pass, proof that the supervisor ran its test and that the test measured a passing swap.

    For a passing entry with N chips:
    - tests/N/test-result.json must show an exit code of 0, no timeout and N chips. Only the
      supervisor writes it, so a configuration whose test never ran cannot pass.
    - the entry must hold the stage 2 fields and meet the stage 2 bar (`_swap_reasons`).
    - its evidence must include configs/N/evidence/swap-check.json, and that file's result_draft
      must hold the same top1_agreement.
    An entry that does not claim a pass is judged by `gate_mesh` alone.
    """
    base = gate_mesh(stage_dir, run_dir, required)
    d, err = _load(stage_dir, "result.json")
    if err:
        return base
    reasons, seen = list(base.reasons), list(base.evidence)
    rel = os.path.relpath(os.path.realpath(stage_dir), os.path.realpath(run_dir))
    for c in d.get("configs") if isinstance(d.get("configs"), list) else []:
        if (not isinstance(c, dict) or c.get("pass") is not True or isinstance(c.get("chips"), bool)
                or not isinstance(c.get("chips"), int)):
            continue
        n = c["chips"]
        where = f"the {n}-chip configuration"
        problem = hw_record_problem(stage_dir, n)
        if problem:
            reasons.append(f"{where} claims a pass, but {problem}")
        else:
            seen.append(f"{rel}/tests/{n}/test-result.json")
        reasons += _swap_reasons(c, f"{where}: ")
        swap = f"{rel}/configs/{n}/evidence/swap-check.json"
        if swap not in (c.get("evidence") if isinstance(c.get("evidence"), list) else []):
            reasons.append(f"{where}: evidence must include {swap}")
            continue
        try:
            draft = json.loads(inside(run_dir, swap).read_text(encoding="utf-8"))["result_draft"]
        except (AttributeError, OSError, ValueError, KeyError, TypeError):
            draft = None
        if not isinstance(draft, dict) or draft.get("top1_agreement") != c.get("top1_agreement"):
            reasons.append(f"{where}: top1_agreement {c.get('top1_agreement')!r} is not the "
                           f"result_draft's in {swap}")
    return _done(reasons, seen)

```

In `orchard/stages.py`, replace:

```python
SKIP_3_WEIGHTS_ONLY = "weights-only path: the stage 2 serve-and-compare covers the full model"
WEIGHTS_ONLY_STAGE_3 = dataclasses.replace(STAGES[3], skip=SKIP_3_WEIGHTS_ONLY)


```

with:

```python
SKIP_3_WEIGHTS_ONLY = "weights-only path: the stage 2 serve-and-compare covers the full model"
WEIGHTS_ONLY_STAGE_3 = dataclasses.replace(STAGES[3], skip=SKIP_3_WEIGHTS_ONLY)

# Stage 4 shows the new weights working on each chip configuration the packages will ship for.
# Nothing is parallelised or shrunk: an existing package or bundle serves each configuration, so the
# stage runs one serve-and-compare test per configuration (the weights-swap-configs skill writes
# hw_tests.json; orchard/hwtests.py reads it) and `gate_mesh_swap` checks the result. `boards` is
# the most any one test needs (the 4-chip configuration needs both boards). The resume marker is
# the supervisor's copy of the validated test list, so a resumed stage skips the prepare step and
# goes on at the first configuration without a test record.
WEIGHTS_ONLY_STAGE_4 = dataclasses.replace(
    STAGES[4], name="weights swap on each chip configuration", skill="weights-swap-configs",
    refs=("tt-device-usage",), boards=2, gate=gate_mesh_swap, marker="tests/plan.json", tests=True,
    disk=STAGE4_SWAP_DISK_GB)


```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_stages.py tests/test_defaults.py`
Expected: 103 passed. The existing `gate_weights_swap` tests pass unchanged: its reasons now come from `_swap_reasons` with the same words.

- [ ] **Step 5: Mutations**

Against `tests/test_stages.py -k <selector>`; fail, restore, delete the bytecode, pass:
1. In `gate_mesh_swap`, change `        problem = hw_record_problem(stage_dir, n)` to `        problem = None`. Selector `never_ran`: 1 failed.
2. Change `        reasons += _swap_reasons(c, f"{where}: ")` to `        pass`. Selector `stage_2_bar`: 1 failed.
3. Change `draft.get("top1_agreement") != c.get("top1_agreement"):` to `False:`. Selector `own_report`: 1 failed.

- [ ] **Step 6: Run the suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -1` (expected `1370 passed, 1 skipped`), then:

```bash
git add orchard/defaults.py orchard/stages.py tests/test_defaults.py tests/test_stages.py
git commit -m "Stage 4 on the weights-only path: spec, disk need and gate_mesh_swap"
```

---

### Task 5: The list of hardware tests and its files

**Files:**
- Create: `orchard/hwtests.py`
- Test: `tests/test_hwtests.py` (new)

**Interfaces:**
- Consumes: `hw_record_path` (Task 4); `NOTE_KEYS` (`orchard/handoff.py`); `CHIPS_PER_BOARD` (`orchard/defaults.py`).
- Produces in `orchard.hwtests`: `SCRIPTS = {"serve_and_compare.py": "bundle", "serve_and_compare_container.py": "container"}`; `SHARED_CACHE_DIRS = (".cache/tt-model", ".cache/qwen36-src-build")`; `PLAN = "tests/plan.json"`; `@dataclass(frozen=True) class HwTest(chips: int, script: str, deadline_s: float, cache: str)` with `.boards -> int` and `.command(stage: int) -> str`; `read_plan(stage_dir, *, required, max_chips: int, budget_s: float, home) -> tuple[list[HwTest], list[str]]`; `write_plan(stage_dir, tests) -> Path`; `load_plan(stage_dir) -> list[HwTest] | None`; `pending(stage_dir, tests) -> list[HwTest]`; `write_record(stage_dir, chips: int, record: dict) -> Path`; `write_summary(stage_dir, tests) -> Path`; `suspect_caches(entries, stage: int) -> set[str]` (keys `cache` on the "hardware test started" decision and the `what="hardware test"` evidence entry); `move_aside(cache) -> Path | None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hwtests.py`:

```python
"""The list of hardware tests a weights-only stage 4 runs, and the records it leaves."""
import json
from pathlib import Path

import pytest

from orchard.hwtests import (HwTest, load_plan, move_aside, pending, read_plan, suspect_caches,
                             write_plan, write_record, write_summary)

NOTE = {"goal": "g", "stage": 4, "evidence": ["e"], "next_action": "n", "check_on_return": "c"}


def config(sd, n, script, cache):
    d = sd / "configs" / str(n)
    d.mkdir(parents=True, exist_ok=True)
    (d / script).write_text("# template copy\n")
    (d / "swap_config.json").write_text(json.dumps({"tt_cache": str(cache)}))


@pytest.fixture
def plan(tmp_path):
    """A stage directory with three configurations, each with its own cache, and the list."""
    sd = tmp_path / "run" / "stages" / "4"
    caches = {n: tmp_path / "orchard-cache" / "hemmingway-1" / f"{n}chip" / "tt_cache" for n in (1, 2, 4)}
    config(sd, 1, "serve_and_compare.py", caches[1])
    config(sd, 2, "serve_and_compare.py", caches[2])
    config(sd, 4, "serve_and_compare_container.py", caches[4])
    tests = [{"chips": 4, "script": "serve_and_compare_container.py", "deadline_s": 3600},
             {"chips": 2, "script": "serve_and_compare.py", "deadline_s": 2400},
             {"chips": 1, "script": "serve_and_compare.py", "deadline_s": 2400}]
    (sd / "hw_tests.json").write_text(json.dumps({"tests": tests}))
    (sd / "handoff.json").write_text(json.dumps(NOTE))
    home = tmp_path / "operator-home"

    def read(required=(2, 4), budget=28800.0, edit=None):
        if edit:
            data = json.loads((sd / "hw_tests.json").read_text())
            edit(data["tests"])
            (sd / "hw_tests.json").write_text(json.dumps(data))
        return read_plan(sd, required=required, max_chips=4, budget_s=budget, home=home)
    return {"sd": sd, "caches": caches, "home": home, "read": read, "tmp": tmp_path}


def test_a_good_list_is_read_in_order_of_chips_with_boards_and_commands(plan):
    tests, problems = plan["read"]()
    assert problems == []
    assert [(t.chips, t.boards, t.deadline_s) for t in tests] == [(1, 1, 2400), (2, 1, 2400), (4, 2, 3600)]
    assert tests[2].command(4) == "python3 stages/4/configs/4/serve_and_compare_container.py"
    assert tests[0].cache == str(plan["caches"][1])


def set_key(i, key, value):
    def edit(tests):
        if value is ...:
            tests[i].pop(key)
        else:
            tests[i][key] = value
    return edit


@pytest.mark.parametrize("edit,words", [
    (set_key(0, "chips", 0), "chips from 1 to 4"),
    (set_key(0, "chips", 8), "chips from 1 to 4"),
    (set_key(0, "chips", True), "chips from 1 to 4"),
    (set_key(1, "chips", 4), "listed more than once"),
    (set_key(0, "script", "run.sh"), "script must be one of"),
    (set_key(1, "deadline_s", 0), "positive deadline_s"),
    (set_key(1, "deadline_s", ...), "positive deadline_s"),
])
def test_a_bad_entry_is_refused_with_a_reason_that_names_it(plan, edit, words):
    _, problems = plan["read"](required=(), edit=edit)
    assert len(problems) == 1 and words in problems[0], problems


def test_a_required_configuration_without_a_test_is_refused(plan):
    _, problems = plan["read"](edit=lambda t: t.pop(0))
    assert problems == ["the 4-chip configuration is required and hw_tests.json has no test for it"]


def test_deadlines_longer_than_the_stage_budget_are_refused(plan):
    _, problems = plan["read"](budget=8000.0)
    assert problems == ["the tests' deadlines add up to 8400 s, more than the stage budget of 8000 s"]


def test_a_missing_script_or_config_is_refused(plan):
    (plan["sd"] / "configs" / "2" / "serve_and_compare.py").unlink()
    (plan["sd"] / "configs" / "1" / "swap_config.json").unlink()
    _, problems = plan["read"](required=())
    assert len(problems) == 2
    assert "configs/2/serve_and_compare.py does not exist" in problems[0]
    assert "configs/1/swap_config.json must exist" in problems[1]


def test_a_cache_inside_a_package_cache_is_refused(plan):
    config(plan["sd"], 4, "serve_and_compare_container.py",
           plan["home"] / ".cache" / "tt-model" / "qwen3.8-27b-p300x2" / "tensors")
    _, problems = plan["read"](required=(2,))
    assert len(problems) == 1 and "a cache another model uses" in problems[0], problems


def test_two_configurations_with_one_cache_are_refused(plan):
    config(plan["sd"], 1, "serve_and_compare.py", plan["caches"][2])
    _, problems = plan["read"]()
    assert problems == [f"tests[2]: tt_cache {plan['caches'][2]} is also the 2-chip configuration's; "
                        "each configuration needs its own"]


def test_a_relative_cache_and_a_bad_handoff_note_are_refused(plan):
    config(plan["sd"], 1, "serve_and_compare.py", "cache/tt_cache")
    (plan["sd"] / "handoff.json").write_text(json.dumps({**NOTE, "goal": ""}))
    _, problems = plan["read"]()
    assert problems[0].startswith("handoff.json needs goal")
    assert "needs an absolute tt_cache" in problems[1]


def test_the_plan_round_trips_and_pending_follows_the_records(plan):
    tests, _ = plan["read"]()
    write_plan(plan["sd"], tests)
    assert load_plan(plan["sd"]) == tests
    assert pending(plan["sd"], tests) == tests
    write_record(plan["sd"], 1, {"returncode": 0, "timed_out": False, "chips": ["a"]})
    assert [t.chips for t in pending(plan["sd"], tests)] == [2, 4]
    summary = json.loads(write_summary(plan["sd"], tests).read_text())
    assert summary["tests"] == [{"returncode": 0, "timed_out": False, "chips": ["a"]},
                                {"chips": 2, "missing": True}, {"chips": 4, "missing": True}]


def test_an_unreadable_plan_loads_as_none(plan):
    assert load_plan(plan["sd"]) is None
    (plan["sd"] / "tests").mkdir()
    (plan["sd"] / "tests" / "plan.json").write_text('{"tests": [{"chips": 2}]}')
    assert load_plan(plan["sd"]) is None


def entry(event, **data):
    return {"event": event, "stage": 4, "data": data}


def test_a_cache_is_suspect_while_its_latest_test_has_not_exited_0():
    started = lambda c: entry("decision", decision="hardware test started", cache=c)      # noqa: E731
    ended = lambda c, rc, to=False: entry("evidence", what="hardware test", cache=c,      # noqa: E731
                                           returncode=rc, timed_out=to)
    assert suspect_caches([started("/a")], 4) == {"/a"}                      # killed mid-test
    assert suspect_caches([started("/a"), ended("/a", 0)], 4) == set()
    assert suspect_caches([started("/a"), ended("/a", 4)], 4) == {"/a"}
    assert suspect_caches([started("/a"), ended("/a", None, True)], 4) == {"/a"}
    assert suspect_caches([started("/a"), ended("/a", 4), started("/a"), ended("/a", 0)], 4) == set()
    other = dict(started("/b"), stage=2)
    assert suspect_caches([other], 4) == set()


def test_move_aside_keeps_the_cache_under_a_new_name_and_leaves_an_empty_one(tmp_path):
    cache = tmp_path / "tt_cache"
    assert move_aside(cache) is None
    cache.mkdir()
    assert move_aside(cache) is None and cache.is_dir()
    (cache / "layer0.bin").write_text("half written")
    aside = move_aside(cache)
    assert aside == tmp_path / "tt_cache.interrupted-1" and (aside / "layer0.bin").is_file()
    assert not cache.exists()
    (cache).mkdir()
    (cache / "x").write_text("again")
    assert move_aside(cache) == tmp_path / "tt_cache.interrupted-2"


def test_a_test_needs_whole_boards():
    assert [HwTest(n, "s", 1.0, "/c").boards for n in (1, 2, 3, 4)] == [1, 1, 2, 2]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_hwtests.py`
Expected: 1 error during collection: `ModuleNotFoundError: No module named 'orchard.hwtests'`.

- [ ] **Step 3: Implement**

Create `orchard/hwtests.py`:

```python
"""A hardware phase that runs several tests: stage 4 on the weights-only path (spec section 5).

This module owns the list of tests one stage runs, one per chip configuration, and the files that
record it. The stage's prepare step writes `hw_tests.json`:

    {"tests": [{"chips": 2, "script": "serve_and_compare.py", "deadline_s": 2400},
               {"chips": 4, "script": "serve_and_compare_container.py", "deadline_s": 3600}]}

Each test's files live in `configs/<chips>/` of the stage directory: the template script and its
`swap_config.json`. The supervisor builds the command itself (`python3 stages/<n>/configs/<chips>/
<script>`), so an agent cannot hand it an arbitrary shell string. `read_plan` refuses a list the
supervisor must not run: a bad entry, a repeated chip count, a required count with no test, deadlines
that add up to more than the stage budget, a tensor cache inside a shared cache directory, or two
configurations that share one tensor cache. The tensor cache is keyed only by layer name and mesh,
so a shared one serves another model's or another mesh's tensors without an error.

After the prepare step, the supervisor writes the validated list to `tests/plan.json` (the stage's
resume marker) and runs the tests in order of chip count, each under its own lease. Each test's
record goes to `tests/<chips>/test-result.json` (orchard/stages.py, `hw_record_path`) as soon as the
test ends, so a supervisor that crashes during the list resumes at the first configuration with no
record. When the list is done, `write_summary` writes the stage's `test-result.json` for the finish
step.

A test that did not exit 0 may have stopped while it was converting weights into its tensor cache,
and a part-written cache is read back without an error. `suspect_caches` finds those caches in the
ledger, and the supervisor moves such a cache aside (`move_aside`, never a delete) before the next
test that would use it.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

from orchard.defaults import CHIPS_PER_BOARD
from orchard.handoff import NOTE_KEYS
from orchard.stages import hw_record_path

SCRIPTS = {"serve_and_compare.py": "bundle", "serve_and_compare_container.py": "container"}
# Tensor caches that belong to installed packages or to hand builds, relative to the operator's
# home. A test's tt_cache must not be inside one of them.
SHARED_CACHE_DIRS = (".cache/tt-model", ".cache/qwen36-src-build")
PLAN = "tests/plan.json"


@dataclass(frozen=True)
class HwTest:
    chips: int
    script: str
    deadline_s: float
    cache: str                    # the tt_cache from configs/<chips>/swap_config.json, resolved

    @property
    def boards(self) -> int:
        return -(-self.chips // CHIPS_PER_BOARD)

    def command(self, stage: int) -> str:
        return f"python3 stages/{stage}/configs/{self.chips}/{self.script}"


def _inside(path: str, root: str) -> bool:
    root = os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def read_plan(stage_dir, *, required, max_chips: int, budget_s: float,
              home) -> tuple[list[HwTest], list[str]]:
    """The tests in hw_tests.json sorted by chip count, and every reason not to run them."""
    stage_dir = Path(stage_dir)
    try:
        data = json.loads((stage_dir / "hw_tests.json").read_text(encoding="utf-8"))
        note = json.loads((stage_dir / "handoff.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [], [f"hw_tests.json and handoff.json must exist and be JSON: {exc}"]
    problems = []
    if not isinstance(note, dict) or any(note.get(k) in (None, "") for k in NOTE_KEYS):
        problems.append(f"handoff.json needs {', '.join(NOTE_KEYS)}")
    items = data.get("tests") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        return [], problems + ["hw_tests.json needs a non-empty list under \"tests\""]
    shared = [str(Path(home) / d) for d in SHARED_CACHE_DIRS]
    tests: list[HwTest] = []
    caches: dict[str, int] = {}
    for i, t in enumerate(items):
        where = f"tests[{i}]"
        n = t.get("chips") if isinstance(t, dict) else None
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= max_chips:
            problems.append(f"{where} needs chips from 1 to {max_chips}, got {n!r}")
            continue
        if any(x.chips == n for x in tests):
            problems.append(f"{where}: the {n}-chip configuration is listed more than once")
            continue
        if t.get("script") not in SCRIPTS:
            problems.append(f"{where} script must be one of {sorted(SCRIPTS)}, got {t.get('script')!r}")
            continue
        d = t.get("deadline_s")
        if isinstance(d, bool) or not isinstance(d, (int, float)) or d <= 0:
            problems.append(f"{where} needs a positive deadline_s, got {d!r}")
            continue
        cdir = stage_dir / "configs" / str(n)
        if not (cdir / t["script"]).is_file():
            problems.append(f"{where}: configs/{n}/{t['script']} does not exist")
            continue
        try:
            cache = json.loads((cdir / "swap_config.json").read_text(encoding="utf-8")).get("tt_cache")
        except (OSError, ValueError, AttributeError) as exc:
            problems.append(f"{where}: configs/{n}/swap_config.json must exist and be a JSON object: {exc}")
            continue
        if not isinstance(cache, str) or not os.path.isabs(cache):
            problems.append(f"{where}: configs/{n}/swap_config.json needs an absolute tt_cache, got {cache!r}")
            continue
        real = os.path.realpath(cache)
        hit = next((s for s in shared if _inside(real, s)), None)
        if hit:
            problems.append(f"{where}: tt_cache {cache} is inside {hit}, a cache another model "
                            "uses; each model and configuration needs a new directory")
            continue
        if real in caches:
            problems.append(f"{where}: tt_cache {cache} is also the {caches[real]}-chip "
                            "configuration's; each configuration needs its own")
            continue
        caches[real] = n
        tests.append(HwTest(n, t["script"], float(d), real))
    listed = {t.chips for t in tests}
    for c in required or ():
        if c not in listed:
            problems.append(f"the {c}-chip configuration is required and hw_tests.json has no test for it")
    total = sum(t.deadline_s for t in tests)
    if total > budget_s:
        problems.append(f"the tests' deadlines add up to {total:g} s, more than the stage budget of "
                        f"{budget_s:g} s")
    return sorted(tests, key=lambda t: t.chips), problems


def _write_json(path: Path, data) -> None:
    """Whole or not at all: a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_plan(stage_dir, tests: list[HwTest]) -> Path:
    path = Path(stage_dir) / PLAN
    _write_json(path, {"tests": [dataclasses.asdict(t) for t in tests]})
    return path


def load_plan(stage_dir) -> list[HwTest] | None:
    """The validated list from tests/plan.json, or None when the file is missing or unreadable."""
    try:
        data = json.loads((Path(stage_dir) / PLAN).read_text(encoding="utf-8"))
        return [HwTest(**t) for t in data["tests"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def pending(stage_dir, tests: list[HwTest]) -> list[HwTest]:
    """The tests that have no record yet, in order."""
    return [t for t in tests if not hw_record_path(stage_dir, t.chips).is_file()]


def write_record(stage_dir, chips: int, record: dict) -> Path:
    path = hw_record_path(stage_dir, chips)
    _write_json(path, record)
    return path


def write_summary(stage_dir, tests: list[HwTest]) -> Path:
    """The stage's test-result.json: every configuration's record, for the finish step."""
    out = []
    for t in tests:
        path = hw_record_path(stage_dir, t.chips)
        out.append(json.loads(path.read_text(encoding="utf-8")) if path.is_file()
                   else {"chips": t.chips, "missing": True})
    path = Path(stage_dir) / "test-result.json"
    _write_json(path, {"tests": out})
    return path


def suspect_caches(entries: list[dict], stage: int) -> set[str]:
    """Tensor caches whose latest test started and did not end with exit 0, from the ledger."""
    state: dict[str, bool] = {}
    for e in entries:
        d = e["data"]
        if e["stage"] != stage or not d.get("cache"):
            continue
        if e["event"] == "decision" and d.get("decision") == "hardware test started":
            state[d["cache"]] = False
        elif e["event"] == "evidence" and d.get("what") == "hardware test":
            state[d["cache"]] = d.get("returncode") == 0 and d.get("timed_out") is False
    return {c for c, ok in state.items() if not ok}


def move_aside(cache) -> Path | None:
    """Rename a non-empty cache directory to <cache>.interrupted-<k> and return the new path. An
    absent or empty directory is left as it is (None)."""
    cache = Path(cache)
    if not cache.is_dir() or not any(cache.iterdir()):
        return None
    k = 1
    while (aside := cache.with_name(f"{cache.name}.interrupted-{k}")).exists():
        k += 1
    os.rename(cache, aside)
    return aside
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_hwtests.py`
Expected: 19 passed.

- [ ] **Step 5: Mutations**

Against `tests/test_hwtests.py -k <selector>`; each 1 failed, then restore, delete the bytecode, pass:
1. `        if hit:` to `        if False:`. Selector `package_cache`.
2. `        if real in caches:` to `        if False:`. Selector `one_cache`.
3. `    if total > budget_s:` to `    if False:`. Selector `budget`.
4. `            state[d["cache"]] = d.get("returncode") == 0 and d.get("timed_out") is False` to `            state[d["cache"]] = True`. Selector `suspect`.
5. `        if c not in listed:` to `        if False:`. Selector `required`.

- [ ] **Step 6: Run the suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -1` (expected `1389 passed, 1 skipped`), then:

```bash
git add orchard/hwtests.py tests/test_hwtests.py
git commit -m "hwtests: validate a stage's list of hardware tests and keep its records"
```

---

### Task 6: The weights-swap-configs skill and the context for a list of tests

**Files:**
- Create: `orchard/skills/weights-swap-configs.md`
- Modify: `orchard/context.py` (`PHASE_TASKS`, `build_messages`)
- Test: `tests/test_skills.py` (`LOCAL`, `GATE_FILES`, append), `tests/test_context.py` (append)

**Interfaces:**
- Consumes: `WEIGHTS_ONLY_STAGE_4` (Task 4), `SCRIPTS` (Task 5).
- Produces: `PHASE_TASKS["prepare-tests"]`, used when `phase == "prepare"` and `spec.tests`; a finish step of such a stage shows `hw_tests.json` and `test-result.json`. The skill `weights-swap-configs` (`status: draft.`, gate file `result.json`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_context.py`:

```python


def test_a_stage_with_a_list_of_tests_prepares_hw_tests_json_and_finishes_from_the_list(tmp_path):
    from orchard.stages import WEIGHTS_ONLY_STAGE_4
    run, sp = setup(tmp_path)
    sd = run / "stages" / "4"
    sd.mkdir(parents=True)
    common = dict(spec=WEIGHTS_ONLY_STAGE_4, run_dir=run, stage_dir=sd, skill_path=sp, refs={},
                  entries=[], facts={"model": "m"}, resumed=False)
    _, user = build_messages(phase="prepare", **common)
    assert "write hw_tests.json in your stage directory" in user and "hw_test.json" not in user
    assert "`python3 stages/4/configs/<chips>/<script>`" in user
    (sd / "hw_tests.json").write_text('{"tests": [{"chips": 4}]}')
    (sd / "test-result.json").write_text('{"tests": [{"returncode": 0}]}')
    _, user = build_messages(phase="finish", **common)
    assert "## stages/4/hw_tests.json" in user and "## stages/4/test-result.json" in user
    _, single = build_messages(phase="prepare", **{**common, "spec": STAGES[4]})
    assert "Write hw_test.json in" in single
```

In `tests/test_skills.py`, replace:

```python

SKILLS = Path(__file__).resolve().parent.parent / "orchard" / "skills"
LOCAL = ("delta-triage", "reference-gate", "weights-swap-check", "serving-check", "operator-bundle")
# Spec section 11: existing skills the stages use, referenced by name only.
SPEC_EXISTING = {"model-bringup", "functional-decoder", "full-model", "multichip", "mesh-shrink",
```

with:

```python

SKILLS = Path(__file__).resolve().parent.parent / "orchard" / "skills"
LOCAL = ("delta-triage", "reference-gate", "weights-swap-check", "weights-swap-configs", "serving-check",
         "operator-bundle")
# Spec section 11: existing skills the stages use, referenced by name only.
SPEC_EXISTING = {"model-bringup", "functional-decoder", "full-model", "multichip", "mesh-shrink",
```

In `tests/test_skills.py`, replace:

```python
                 "stage-review", "tti-release"}
GATE_FILES = {"delta-triage": "delta.json", "reference-gate": "reference.json",
              "weights-swap-check": "result.json", "serving-check": "result.json",
              "operator-bundle": "PUBLISH_COMMANDS.txt"}

```

with:

```python
                 "stage-review", "tti-release"}
GATE_FILES = {"delta-triage": "delta.json", "reference-gate": "reference.json",
              "weights-swap-check": "result.json", "weights-swap-configs": "result.json",
              "serving-check": "result.json",
              "operator-bundle": "PUBLISH_COMMANDS.txt"}

```

Append to `tests/test_skills.py`:

```python


def test_the_configs_skill_copies_the_three_templates_that_exist_in_this_repo():
    text = (SKILLS / "weights-swap-configs.md").read_text()
    main = "/home/ttuser/code/tt-orchard/orchard/skills/weights-swap-templates/"
    for name in ("prepare_swap.py", "serve_and_compare.py", "serve_and_compare_container.py"):
        assert main + name in text
        assert (SKILLS / "weights-swap-templates" / name).is_file()


def test_the_configs_skill_writes_the_list_the_supervisor_reads():
    import json
    import re

    from orchard.hwtests import SCRIPTS
    from orchard.stages import WEIGHTS_ONLY_STAGE_4
    text = (SKILLS / "weights-swap-configs.md").read_text()
    block = re.search(r'(\{"tests": \[.*?\]\})', text, re.S).group(1)
    tests = json.loads(" ".join(block.split()))["tests"]
    assert sorted(t["chips"] for t in tests) == [1, 2, 4]
    assert all(t["script"] in SCRIPTS for t in tests)
    assert sum(t["deadline_s"] for t in tests) <= WEIGHTS_ONLY_STAGE_4.budget_s


def test_the_configs_skill_quotes_the_gate_bar_and_the_cache_rule():
    from orchard.defaults import SWAP_TOP1_MIN
    text = " ".join((SKILLS / "weights-swap-configs.md").read_text().split())
    assert f"`top1_agreement` of at least {SWAP_TOP1_MIN}" in text
    assert "Every configuration gets its own new `tt_cache`" in text
    assert "A configuration whose test never ran cannot pass." in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_skills.py tests/test_context.py`
Expected: 6 FAIL, 27 passed: five skill tests, because `weights-swap-configs.md` does not exist, and the context test, because the prepare text still asks for `hw_test.json`.

- [ ] **Step 3: Implement**

In `orchard/context.py`, replace:

```python
                "its output to evidence/hw-test-output.txt and writes test-result.json. Then reply "
                "with a short summary and no tool call."),
    "finish": ("The supervisor ran your hardware test; its record and output are below and in "
               "your stage directory. Write {gate} from that evidence. Do not invent a result. "
```

with:

```python
                "its output to evidence/hw-test-output.txt and writes test-result.json. Then reply "
                "with a short summary and no tool call."),
    # A stage whose spec has `tests` (stage 4 on the weights-only path) prepares a list of tests.
    "prepare-tests": ("Prepare this stage's hardware tests, one per chip configuration. Do not run "
                      "them yourself. Put each configuration's files in configs/<chips>/ of your "
                      "stage directory as the skill describes. Then write hw_tests.json in your "
                      "stage directory as {{\"tests\": [{{\"chips\": <n>, \"script\": "
                      "\"serve_and_compare.py\" or \"serve_and_compare_container.py\", "
                      "\"deadline_s\": <seconds>}}, ...]}} and handoff.json with the keys goal, "
                      "stage, evidence, next_action and check_on_return. The supervisor runs each "
                      "test as `python3 stages/{n}/configs/<chips>/<script>`, in order of chip "
                      "count, on leased chips with TT_VISIBLE_DEVICES, ORCHARD_DEVICE_IDS and "
                      "ORCHARD_TEST_LABEL set. It saves each test's output to "
                      "tests/<chips>/output.txt, writes tests/<chips>/test-result.json, and writes "
                      "test-result.json for the whole list. Then reply with a short summary and no "
                      "tool call."),
    "finish": ("The supervisor ran your hardware test; its record and output are below and in "
               "your stage directory. Write {gate} from that evidence. Do not invent a result. "
```

In `orchard/context.py`, replace:

```python
        system += [f"- {name}: {path or 'not installed on this machine'}" for name, path in refs.items()]

    user = ["## Task", PHASE_TASKS[phase].format(gate=spec.gate_file, n=n), "", "## Run"]
    user += [f"- {k}: {v}" for k, v in facts.items()]
    user += [f"- stage directory: stages/{n}"]
```

with:

```python
        system += [f"- {name}: {path or 'not installed on this machine'}" for name, path in refs.items()]

    task = PHASE_TASKS["prepare-tests" if phase == "prepare" and spec.tests else phase]
    user = ["## Task", task.format(gate=spec.gate_file, n=n), "", "## Run"]
    user += [f"- {k}: {v}" for k, v in facts.items()]
    user += [f"- stage directory: stages/{n}"]
```

In `orchard/context.py`, replace:

```python
        own.append(spec.marker)
    if phase == "finish":
        own += ["hw_test.json", "test-result.json"]
    for name in own:
        f = stage_dir / name
```

with:

```python
        own.append(spec.marker)
    if phase == "finish":
        own += ["hw_tests.json" if spec.tests else "hw_test.json", "test-result.json"]
    for name in own:
        f = stage_dir / name
```

Create `orchard/skills/weights-swap-configs.md`:

```markdown
---
name: weights-swap-configs
description: Stage 4 of a tt-orchard run when stage 0 found a weights-only delta. Serve the new weights on each chip configuration the packages will ship for (2 and 4 chips required, 1 chip optional), using the packages that already serve the nearest model, and compare each with the stage 1 CPU reference.
status: draft. A local tt-orchard copy written on 2026-10-03 from the stage 2 weights-swap-check skill and its templates. No harness run has used it yet. It lives in tt-orchard.
---

# Weights swap on each chip configuration

## When to use this

Stage 0 found a weights-only delta and stage 2 showed the new weights serving on one
configuration. This stage shows them working on every chip configuration the packages will ship
for. No model code is written and nothing is parallelised: for each configuration an existing
package or bundle of the nearest model serves the new weights, and the same two tested template
scripts as stage 2 measure it. You copy the templates and write one small config file per
configuration. You do not write a script.

## Goal

Write `result.json` in your stage directory:

    {"configs": [
      {"chips": 2, "pass": true, "kind": "bundle", "package": "episod/qwen3.8-27b-dflash2-p300",
       "serves": true, "server_ready_s": 120.4, "coherent": true,
       "free_run_text": "first 200 characters", "top1_agreement": 0.94, "n_tokens": 32,
       "cache_dir": "<this configuration's tensor cache>",
       "evidence": ["stages/4/configs/2/evidence/swap-check.json",
                    "stages/4/configs/2/evidence/server.log", "stages/4/tests/2/output.txt"]},
      {"chips": 4, "pass": true, "kind": "container", "package": "changh95/qwen3.8-27b-p300x2", ...},
      {"chips": 1, "pass": false, "reason": "what failed, from the test output",
       "evidence": ["stages/4/tests/1/output.txt"]}]}

The context lists the required chip counts for this run (normally 2 and 4). Each required count
needs an entry with `pass` true. Any other count is optional. Try it, and if it does not work,
record it with `pass` false and the reason. If no package exists for an optional count, record
`"pass": false, "reason": "not run: no installed package serves <N> chips"` and give the
evidence file that shows it (for example the output of `tt-model list` saved under
`stages/4/evidence/`).

The gate checks, for every entry with `pass` true: the supervisor's record
`tests/<chips>/test-result.json` shows that the test ran on that many chips and exited 0; the entry
has `serves` true, `coherent` true, `n_tokens` of at least 16, `top1_agreement` of at least 0.85
and a positive `server_ready_s`; and its evidence includes
`stages/4/configs/<chips>/evidence/swap-check.json`, whose `result_draft` has the same
`top1_agreement`. A configuration whose test never ran cannot pass.

## How to work

- Do the configurations one at a time, in this order: 2 chips, 4 chips, then 1 chip.
- Write each file as soon as you know its content.
- Do not investigate anything this skill does not list.

## The configurations on this machine

| chips | kind | package | context | template script |
|---|---|---|---|---|
| 1 | bundle | `episod/qwen3.8-27b-dflash2-p150` at `~/.cache/tt-model/models/episod/qwen3.8-27b-dflash2-p150` | 16K | `serve_and_compare.py` |
| 2 | bundle | `episod/qwen3.8-27b-dflash2-p300` at `~/.cache/tt-model/models/episod/qwen3.8-27b-dflash2-p300` | 262K | `serve_and_compare.py` |
| 4 | container | `changh95/qwen3.8-27b-p300x2`, profile `batch32` | 262K | `serve_and_compare_container.py` |

To check a row, run `tt-model list` and read `~/.cache/tt-model/installed.json` (a container
package has `"container": true` and an `"image"`). A bundle has `run.sh` and `venv/` in its
directory. If a required package is missing, write that in `result.json` and stop.

## Prepare phase: the steps for one configuration with N chips

1. Make the directory and copy the three templates into it:

       mkdir -p stages/4/configs/N
       cp /home/ttuser/code/tt-orchard/orchard/skills/weights-swap-templates/prepare_swap.py /home/ttuser/code/tt-orchard/orchard/skills/weights-swap-templates/serve_and_compare.py /home/ttuser/code/tt-orchard/orchard/skills/weights-swap-templates/serve_and_compare_container.py stages/4/configs/N/

2. Write `stages/4/configs/N/swap_config.json` (absolute paths). The snapshots, the model ids and
   `run_dir` are the same as in stage 2's `swap_config.json` (`stages/2/swap_config.json`); read it.

   For a bundle (1 or 2 chips):

       {"run_dir": "<the run directory>",
        "bundle_dir": "<bundle directory from the table>",
        "nearest_model_id": "Qwen/Qwen3.8-27B",
        "base_snapshot": "<nearest model snapshot>",
        "new_snapshot": "<new model snapshot>",
        "new_model_id": "<new model id>",
        "tt_cache": "/mnt/bonus/models/orchard-runs/cache/<slug>/<N>chip-<package name>/tt_cache",
        "hf_home": "/home/ttuser/.cache/huggingface",
        "port": <8100 + N>}

   For the container package (4 chips): no `bundle_dir`; add the package, its profile, the chip
   count and the operator's home:

       {"run_dir": "<the run directory>",
        "package": "changh95/qwen3.8-27b-p300x2",
        "profile": "batch32",
        "chips": 4,
        "nearest_model_id": "Qwen/Qwen3.8-27B",
        "base_snapshot": "<nearest model snapshot>",
        "new_snapshot": "<new model snapshot>",
        "new_model_id": "<new model id>",
        "tt_cache": "/mnt/bonus/models/orchard-runs/cache/<slug>/4chip-qwen3.8-27b-p300x2/tt_cache",
        "hf_home": "/home/ttuser/.cache/huggingface",
        "operator_home": "/home/ttuser",
        "port": 8104}

   Every configuration gets its own new `tt_cache`. Never reuse stage 2's cache, another
   configuration's cache, or anything under `~/.cache/tt-model/` or `~/.cache/qwen36-src-build/`.
   The tensor cache is keyed only by layer name and mesh, so a cache made by another model or
   another mesh is read without an error and serves the wrong tensors. The supervisor refuses a
   list in which two configurations share a cache.
3. Run `python3 stages/4/configs/N/prepare_swap.py`. It builds `stages/4/configs/N/model-dir/` and,
   for a bundle, `stages/4/configs/N/run.sh`. Exit 2 means an edit did not apply; its message names
   the fact in `swap_config.json` to fix.
4. Check that `stages/4/configs/N/model-dir` exists (`ls -l`), and for a bundle `run.sh` too.

When every configuration is prepared:

5. Write `hw_tests.json` in your stage directory, one entry per configuration you prepared:

       {"tests": [{"chips": 2, "script": "serve_and_compare.py", "deadline_s": 3600},
                  {"chips": 4, "script": "serve_and_compare_container.py", "deadline_s": 3600},
                  {"chips": 1, "script": "serve_and_compare.py", "deadline_s": 3600}]}

6. Write `handoff.json` and reply with a short summary.

Do not run either serve script yourself. The supervisor runs each one under a lease, in order of
chip count, and records it.

## What the two serve scripts do

`serve_and_compare.py` (bundles) is the stage 2 script: it starts the bundle's edited `run.sh` with
`MODEL_WEIGHTS_DIR` and `HF_MODEL` set to the model directory and a fresh tensor cache, then
measures.

`serve_and_compare_container.py` (the container package) asks `tt-model serve <package> --print`
for the docker command, edits it and runs it: the tensor cache becomes your `tt_cache`, the
operator's Hugging Face cache is replaced by an empty directory so the base weights cannot be
loaded, the model directory and the new model's weight files are mounted read-only at their own
paths, `MODEL_WEIGHTS_DIR` and `HF_MODEL` name the model directory, and only the leased chips are
mapped. It saves the edited command as `evidence/docker-argv.json`, and it always stops and
removes its container.

Both write `stages/4/configs/N/evidence/swap-check.json`, whose `result_draft` holds the fields of
one `configs` entry (without `chips`, `pass`, `kind` and `package`). Exit codes: 0 measured, 2 the
config or the docker command could not be used, 3 the tensor cache was refused, 4 the server did
not become healthy, 5 the server answered with an HTTP error. Both wait up to 3300 s for the
server. A first boot converts the weights into the new cache (about 5 minutes for 2 chips here; not
measured for 4 chips), and a bundle whose kernel compile cache is cold compiles every kernel first
(more than 26 minutes on 2026-10-03). The 2-chip bundle reuses the kernels stage 2 compiled; the
1-chip bundle compiles its own. The container keeps the package's own kernel cache.

## Finish phase

Read the stage's `test-result.json` (every configuration's record), and for each configuration
`stages/4/tests/N/output.txt` and `stages/4/configs/N/evidence/swap-check.json`.

- If the test exited 0, copy `result_draft` into the entry, add `chips`, `kind` and `package`, and
  add `stages/4/tests/N/output.txt` to its evidence. Set `pass` true only if the draft meets the
  bar above; otherwise set `pass` false and give the reason (for example "top1_agreement 0.75 is
  below 0.85").
- If the test did not exit 0, set `pass` false and copy the failure text from `output.txt` into
  `reason`. Do not investigate firmware or cache directories.

Do not invent a number.

## Do not

- Do not modify a bundle, a package, the nearest model's snapshot, or any tensor cache you did not
  create.
- Do not start, stop or remove a container, and do not run `tt-model serve` or `docker` yourself.
- Do not write model code or your own serving script.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_skills.py tests/test_context.py`
Expected: 33 passed.

- [ ] **Step 5: Mutation**

In `orchard/context.py`, change `PHASE_TASKS["prepare-tests" if phase == "prepare" and spec.tests else phase]` to `PHASE_TASKS[phase]`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_context.py -k list_of_tests`. Expected: 1 failed. Restore, delete the bytecode, run again: 1 passed.

- [ ] **Step 6: Run the suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -1` (expected `1395 passed, 1 skipped`), then:

```bash
git add orchard/skills/weights-swap-configs.md orchard/context.py tests/test_skills.py tests/test_context.py
git commit -m "weights-swap-configs skill and the prepare task for a list of hardware tests"
```

---

### Task 7: The supervisor runs the list, leases each test and parks only when needed

**Files:**
- Modify: `orchard/supervisor.py` (docstring, imports, `LabelledContainers`, `Supervisor.__init__`, `_stage_body`, `_run_test`, new `_spawn`, `_run_test_list`, `_run_listed`, `_take_test_lease`, `_give_back_test_lease`, `_run_listed_test`, `_sweep`, `build`)
- Modify: `orchard/stages.py` (`spec_for` returns `WEIGHTS_ONLY_STAGE_4`)
- Modify: `tests/run_fakes.py` (`FakeContainers`, stage 4's scripted files), `tests/test_supervisor.py`, `tests/test_run_e2e.py`, `tests/test_stages.py`
- Test: `tests/test_supervisor_stage4.py` (new)

**Interfaces:**
- Consumes: Task 4's `WEIGHTS_ONLY_STAGE_4` and records; Task 5's `read_plan`, `write_plan`, `load_plan`, `pending`, `write_record`, `write_summary`, `HwTest`; `decide_park`, `reacquire`, `Handoff` (`orchard/handoff.py`).
- Produces in `orchard.supervisor`: `class LabelledContainers(run=run_command, docker="docker")` with `list(label) -> list[str] | None` and `remove(cid) -> None`; `Supervisor(..., home=None, containers=None)` with `.run_label` (`"orchard.test=" + sha256(str(run_dir))[:12]`); `build(..., containers=None)`. The hardware test environment gains `ORCHARD_DEVICE_IDS` and `ORCHARD_TEST_LABEL` (both paths). Ledger: an `evidence` entry `what="hardware test list"` with `configs`; "hardware test started" with `config`, `cache`, `lease_ids`; "test lease taken" with `config`; an `evidence` entry `what="hardware test"` with `config`, `cache`, `returncode`, `timed_out`, `path`, `sha256`.
- Produces in `tests/run_fakes.py`: `FakeContainers(listings=None)` (`.calls`), `FAKE_SWAP_TEST`, `SWAP_SCRIPTS`, `fake_cache(n)`, `swap_prepare(caches=None, deadline=60)`, `swap_entry(n, ok=True)`, `MESH_SHRINK`.

The existing coder-death test kills the coder during stage 5 in place of stage 4: stage 4 now parks the coder for its 4-chip test, and the park asks the coder first, so a death there blocks the park before the next stage would notice it. The full-port fake run keeps `hw_test.json` for stage 4 (`MESH_SHRINK`), chosen by the skill name as stage 2's decoder files are. The two e2e layout tests change: with the coder on 2 chips only stage 4's 4-chip test parks, and with it on 4 chips every stage 4 test parks.

- [ ] **Step 1: Write the failing tests and the fakes**

In `tests/run_fakes.py`, replace:

```python
interface: a lease whose owner pid is dead is reaped at the next acquire once no device is open,
reset and release refuse while a device is open, and a chip of a running container shows
HELD-FOREIGN. `MachineCoder` behaves like a container coder. `bringup` is a model script that walks
stages 0 to 8 by writing the files each gate reads (stage 2's result follows the skill the
supervisor named); it answers from the request alone, as a greedy
```

with:

```python
interface: a lease whose owner pid is dead is reaped at the next acquire once no device is open,
reset and release refuse while a device is open, and a chip of a running container shows
HELD-FOREIGN. `MachineCoder` behaves like a container coder. `FakeContainers` stands in for the
docker label search after each stage 4 test. `bringup` is a model script that walks
stages 0 to 8 by writing the files each gate reads (stage 2's result follows the skill the
supervisor named); it answers from the request alone, as a greedy
```

In `tests/run_fakes.py`, replace:

```python


# ---- the scripted bring-up ------------------------------------------------------------------------

```

with:

```python


class FakeContainers:
    """The supervisor's docker label search (orchard/supervisor.py, LabelledContainers).

    `listings` are the answers to successive list() calls; the last one repeats. The default finds
    nothing. A test that wants a container left behind passes [["c1"], []] (found, then gone after
    the stop) or [["c1"]] (it never goes away)."""

    def __init__(self, listings=None):
        self.listings = [list(x) if x is not None else None for x in (listings or [[]])]
        self.calls: list[tuple] = []

    def list(self, label):
        self.calls.append(("list", label))
        return self.listings.pop(0) if len(self.listings) > 1 else self.listings[0]

    def remove(self, cid):
        self.calls.append(("remove", cid))


# ---- the scripted bring-up ------------------------------------------------------------------------

```

In `tests/run_fakes.py`, replace:

```python


FILES = {
    (0, "run"): {"evidence/notes.txt": "configs, tensors and tokenizers compared", "delta.json": DELTA},
```

with:

```python


# Stage 4 on the weights-only path: one test per chip configuration, each a small Python script in
# configs/<chips>/ that records the chips it was given and writes the swap report the gate reads.
FAKE_SWAP_TEST = """import json, os, pathlib
here = pathlib.Path(__file__).resolve().parent
(here / "evidence").mkdir(exist_ok=True)
keys = ("TT_VISIBLE_DEVICES", "ORCHARD_DEVICE_IDS", "ORCHARD_TEST_LABEL")
(here / "evidence" / "devices.txt").write_text("".join(f"{k}={os.environ.get(k)}\\n" for k in keys))
(here / "evidence" / "swap-check.json").write_text(json.dumps({"result_draft": {"top1_agreement": 0.94}}))
"""
SWAP_SCRIPTS = {1: "serve_and_compare.py", 2: "serve_and_compare.py", 4: "serve_and_compare_container.py"}


def fake_cache(n):
    """An absolute tensor cache path nothing creates (the fake tests never touch it)."""
    return f"/nonexistent/orchard-fake-cache/hemmingway-1/{n}chip/tt_cache"


def swap_prepare(caches=None, deadline=60):
    files = {"evidence/notes.txt": "stage 4 plan", "handoff.json": note(4),
             "hw_tests.json": {"tests": [{"chips": n, "script": s, "deadline_s": deadline}
                                         for n, s in SWAP_SCRIPTS.items()]}}
    for n, script in SWAP_SCRIPTS.items():
        files[f"configs/{n}/{script}"] = FAKE_SWAP_TEST
        files[f"configs/{n}/swap_config.json"] = {"tt_cache": (caches or {}).get(n, fake_cache(n))}
    return files


def swap_entry(n, ok=True):
    """One configs entry of stage 4's result.json, as the weights-swap-configs skill writes it."""
    if not ok:
        return {"chips": n, "pass": False, "reason": "does not fit",
                "evidence": [f"stages/4/tests/{n}/output.txt"]}
    return {"chips": n, "pass": True, "kind": "container" if n == 4 else "bundle", "serves": True,
            "server_ready_s": 120.0, "coherent": True, "free_run_text": "The sea was calm.",
            "top1_agreement": 0.94, "n_tokens": 32, "cache_dir": fake_cache(n),
            "evidence": [f"stages/4/configs/{n}/evidence/swap-check.json", f"stages/4/tests/{n}/output.txt"]}


FILES = {
    (0, "run"): {"evidence/notes.txt": "configs, tensors and tokenizers compared", "delta.json": DELTA},
```

In `tests/run_fakes.py`, replace:

```python
    (3, "prepare"): hw(3),
    (3, "finish"): {"result.json": {"parity": True, "top1": 0.97, "evidence": done(3)}},
    (4, "prepare"): hw(4),
    (4, "finish"): {"result.json": {"configs": [{"chips": 2, "pass": True, "evidence": done(4)},
                                                {"chips": 1, "pass": True, "evidence": done(4)}]}},
    (5, "prepare"): hw(5),
    (5, "finish"): {"result.json": {"checks": {k: {"pass": True, "evidence": done(5)}
```

with:

```python
    (3, "prepare"): hw(3),
    (3, "finish"): {"result.json": {"parity": True, "top1": 0.97, "evidence": done(3)}},
    # The fake run's delta says weights-only, so stage 4 runs one test per chip configuration.
    (4, "prepare"): swap_prepare(),
    (4, "finish"): {"result.json": {"configs": [swap_entry(n) for n in SWAP_SCRIPTS]}},
    (5, "prepare"): hw(5),
    (5, "finish"): {"result.json": {"checks": {k: {"pass": True, "evidence": done(5)}
```

In `tests/run_fakes.py`, replace:

```python
DECODER_FINISH = {"result.json": {"pcc": 0.998, "argmax_match": True, "evidence": done(2)}}


def skill_named(request) -> str:
```

with:

```python
DECODER_FINISH = {"result.json": {"pcc": 0.998, "argmax_match": True, "evidence": done(2)}}

# Stage 4's files when the supervisor names the mesh-shrink skill (a full port): one hw_test.json.
MESH_SHRINK = {"prepare": hw(4),
               "finish": {"result.json": {"configs": [{"chips": 2, "pass": True, "evidence": done(4)},
                                                      {"chips": 1, "pass": True, "evidence": done(4)}]}}}


def skill_named(request) -> str:
```

In `tests/run_fakes.py`, replace:

```python
        return final(CANARY_ANSWER)
    key = where(request)
    default = DECODER_FINISH if key == (2, "finish") and skill_named(request) == "functional-decoder" else FILES[key]
    files = (overrides or {}).get(key, default)
    turns = [call("write_file", path=p, content=c if isinstance(c, str) else json.dumps(c))
```

with:

```python
        return final(CANARY_ANSWER)
    key = where(request)
    default = FILES[key]
    if key == (2, "finish") and skill_named(request) == "functional-decoder":
        default = DECODER_FINISH
    elif key[0] == 4 and skill_named(request) == "mesh-shrink":
        default = MESH_SHRINK[key[1]]
    files = (overrides or {}).get(key, default)
    turns = [call("write_file", path=p, content=c if isinstance(c, str) else json.dumps(c))
```

In `tests/test_run_e2e.py`, replace:

```python
from orchard.stages import run_progress
from orchard.supervisor import EXIT_READY, build, parse
from run_fakes import (BOARDS, SWAP_LOW, CrashingLedger, Machine, MachineAdapter, MachineCoder, argv,
                       bringup, clock, feedback_aware, plenty, test_fails_until_escalated, write_tiers)

FIRST, SECOND = 100, 200          # supervisor pids before and after the kill
```

with:

```python
from orchard.stages import run_progress
from orchard.supervisor import EXIT_READY, build, parse
from run_fakes import (BOARDS, SWAP_LOW, CrashingLedger, FakeContainers, Machine, MachineAdapter,
                       MachineCoder, argv, bringup, clock, feedback_aware, plenty,
                       test_fails_until_escalated, write_tiers)

FIRST, SECOND = 100, 200          # supervisor pids before and after the kill
```

In `tests/test_run_e2e.py`, replace:

```python
        return build(args, led, adapter=MachineAdapter(machine, owner_pid=pid),
                     coder=MachineCoder(machine), versions={"tt_model": "test"}, clock=c,
                     sleep=c.sleep, disk_usage=plenty, home=base / "operator-home").run()


```

with:

```python
        return build(args, led, adapter=MachineAdapter(machine, owner_pid=pid),
                     coder=MachineCoder(machine), versions={"tt_model": "test"}, clock=c,
                     sleep=c.sleep, disk_usage=plenty, home=base / "operator-home",
                     containers=FakeContainers()).run()


```

In `tests/test_run_e2e.py`, replace:

```python
    assert run(tmp_path, servers, m, chips=4, pid=FIRST) == EXIT_READY
    es = entries(tmp_path)
    for n in (2, 4, 5, 6):                  # stage 3 is skipped on the weights-only path
        seq = [(e["event"], e["data"].get("step") or e["data"].get("decision")) for e in es if e["stage"] == n]
        test = seq.index(("decision", "hardware test started"))
```

with:

```python
    assert run(tmp_path, servers, m, chips=4, pid=FIRST) == EXIT_READY
    es = entries(tmp_path)
    for n in (2, 5, 6):                     # stage 3 is skipped on the weights-only path
        seq = [(e["event"], e["data"].get("step") or e["data"].get("decision")) for e in es if e["stage"] == n]
        test = seq.index(("decision", "hardware test started"))
```

In `tests/test_run_e2e.py`, replace:

```python
        env = (tmp_path / "run" / "stages" / str(n) / "evidence" / "devices.txt").read_text()
        assert f"TT_VISIBLE_DEVICES={','.join(BOARDS['B0'])}\n" in env       # one board of the four chips
    s = final_state(tmp_path)
    assert s["finished"] and s["done"] == (0, 1, 2, 3, 4, 5, 6, 7, 8) and not s["parked"]
```

with:

```python
        env = (tmp_path / "run" / "stages" / str(n) / "evidence" / "devices.txt").read_text()
        assert f"TT_VISIBLE_DEVICES={','.join(BOARDS['B0'])}\n" in env       # one board of the four chips
    # Stage 4 runs one test per configuration, each between its own park's reset and restore's reset.
    marks = (("park", "reset"), ("decision", "hardware test started"), ("restore", "reset"))
    seq = [(e["event"], e["data"].get("step") or e["data"].get("decision")) for e in es if e["stage"] == 4]
    assert [x for x in seq if x in marks] == list(marks) * 3
    for n, chips in ((1, BOARDS["B0"][:1]), (2, BOARDS["B0"]), (4, BOARDS["B0"] + BOARDS["B1"])):
        env = (tmp_path / "run" / "stages" / "4" / "configs" / str(n) / "evidence" / "devices.txt").read_text()
        assert f"TT_VISIBLE_DEVICES={','.join(chips)}\n" in env, n
        assert f"ORCHARD_DEVICE_IDS={','.join(str(i) for i in range(n))}\n" in env, n
    s = final_state(tmp_path)
    assert s["finished"] and s["done"] == (0, 1, 2, 3, 4, 5, 6, 7, 8) and not s["parked"]
```

In `tests/test_run_e2e.py`, replace:

```python


def test_with_the_coder_on_two_chips_the_free_board_is_used_and_nothing_parks(tmp_path, servers):
    m = Machine()
    assert run(tmp_path, servers, m, chips=2, pid=FIRST) == EXIT_READY
    es = entries(tmp_path)
    assert not [e for e in es if e["event"] in ("park", "restore")]
    taken = [e["data"]["test_lease"]["chips"] for e in es if e["data"].get("decision") == "test lease taken"]
    assert taken == [list(BOARDS["B1"])] * 4          # stages 2, 4, 5 and 6; stage 3 is skipped
    released = [e for e in es if e["data"].get("decision") == "test lease released"]
    assert len(released) == 4 and m.leases == {}
    assert final_state(tmp_path)["finished"]

```

with:

```python


def test_with_the_coder_on_two_chips_only_the_4_chip_test_parks_it(tmp_path, servers):
    m = Machine()
    assert run(tmp_path, servers, m, chips=2, pid=FIRST) == EXIT_READY
    es = entries(tmp_path)
    handoff = [(e["stage"], e["event"], e["data"]["step"]) for e in es if e["event"] in ("park", "restore")]
    assert {s for s, _, _ in handoff} == {4}
    assert [x for x in handoff if x[2] in ("note", "resumed")] == [(4, "park", "note"), (4, "restore", "resumed")]
    taken = [(e["stage"], e["data"].get("config"), e["data"]["test_lease"]["chips"]) for e in es
             if e["data"].get("decision") == "test lease taken"]
    b1 = list(BOARDS["B1"])
    # Stages 2, 5 and 6 and the 1- and 2-chip configurations use the free board; the 4-chip
    # configuration takes the free board too, then parks the coder for its board.
    assert taken == [(2, None, b1), (4, 1, b1), (4, 2, b1), (4, 4, b1), (5, None, b1), (6, None, b1)]
    seq = [(e["data"].get("decision") or e["data"].get("step"), e["data"].get("config")) for e in es if e["stage"] == 4]
    assert seq.index(("test lease taken", 4)) < seq.index(("note", None))      # the further board first
    env = (tmp_path / "run" / "stages" / "4" / "configs" / "4" / "evidence" / "devices.txt").read_text()
    assert f"TT_VISIBLE_DEVICES={','.join(BOARDS['B0'] + BOARDS['B1'])}\n" in env
    released = [e for e in es if e["data"].get("decision") == "test lease released"]
    assert len(released) == 6 and m.leases == {}
    assert final_state(tmp_path)["finished"]

```

In `tests/test_stages.py`, replace:

```python


@pytest.mark.parametrize("n", [0, 1, 4, 5, 6, 7, 8])
def test_the_path_changes_no_other_stage(n):
    for path in ("weights-only", "full-port", None):
        assert spec_for(n, path) == STAGES[n]


```

with:

```python


@pytest.mark.parametrize("n", [0, 1, 5, 6, 7, 8])
def test_the_path_changes_no_other_stage(n):
    for path in ("weights-only", "full-port", None):
        assert spec_for(n, path) == STAGES[n]


def test_only_the_weights_only_path_runs_stage_4_as_a_list_of_tests():
    from orchard.stages import WEIGHTS_ONLY_STAGE_4
    assert spec_for(4, "weights-only") is WEIGHTS_ONLY_STAGE_4
    assert spec_for(4, "full-port") == STAGES[4] and spec_for(4, None) == STAGES[4]


```

In `tests/test_supervisor.py`, replace:

```python
from orchard.supervisor import (EXIT_ABORTED, EXIT_ERROR, EXIT_READY, EXIT_REFUSED, Control, build,
                                main, parse)
from run_fakes import (BOARDS, DELTA, FEEDBACK_HEAD, FILES, SWAP_LOW, CrashingLedger, Machine,
                       MachineAdapter, MachineCoder, argv, bringup, clock, feedback_aware, plenty,
                       test_fails_until_escalated, where, write_tiers)

# The hardware test and agent shells run real bash here, so stub tools come first on PATH.
```

with:

```python
from orchard.supervisor import (EXIT_ABORTED, EXIT_ERROR, EXIT_READY, EXIT_REFUSED, Control, build,
                                main, parse)
from run_fakes import (BOARDS, DELTA, FEEDBACK_HEAD, FILES, SWAP_LOW, CrashingLedger, FakeContainers,
                       Machine, MachineAdapter, MachineCoder, argv, bringup, clock, feedback_aware,
                       plenty, swap_entry, test_fails_until_escalated, where, write_tiers)

# The hardware test and agent shells run real bash here, so stub tools come first on PATH.
```

In `tests/test_supervisor.py`, replace:

```python
        self.clock, self.usage, self.on_sleep = clock(), plenty, None
        self.adapter_cls, self.coder_cls = MachineAdapter, MachineCoder

    def sleep(self, s):
```

with:

```python
        self.clock, self.usage, self.on_sleep = clock(), plenty, None
        self.adapter_cls, self.coder_cls = MachineAdapter, MachineCoder
        self.containers = FakeContainers()

    def sleep(self, s):
```

In `tests/test_supervisor.py`, replace:

```python
            sup = build(self.args, led, adapter=self.adapter_cls(self.m, owner_pid=pid),
                        coder=self.coder_cls(self.m), versions={"tt_model": "test"}, clock=self.clock,
                        sleep=self.sleep, disk_usage=lambda p: self.usage(p), home=self.home)
            return sup.run()

```

with:

```python
            sup = build(self.args, led, adapter=self.adapter_cls(self.m, owner_pid=pid),
                        coder=self.coder_cls(self.m), versions={"tt_model": "test"}, clock=self.clock,
                        sleep=self.sleep, disk_usage=lambda p: self.usage(p), home=self.home,
                        containers=self.containers)
            return sup.run()

```

In `tests/test_supervisor.py`, replace:

```python
        if "tools" in request:
            n, phase = where(request)
            if n in (1, 4) and turn(request) == 0 and n not in killed:
                killed.add(n)
                rig.m.coder_running = False         # the coder dies while the agent works
```

with:

```python
        if "tools" in request:
            n, phase = where(request)
            # Stages 1 and 5: stage 4 on the weights-only path parks the coder for its 4-chip
            # test, and a park asks the coder first, so a death there blocks the park instead.
            if n in (1, 5) and turn(request) == 0 and n not in killed:
                killed.add(n)
                rig.m.coder_running = False         # the coder dies while the agent works
```

In `tests/test_supervisor.py`, replace:

```python
def mesh_result(*configs):
    """A stage 4 result.json override from (chips, pass) pairs."""
    return {(4, "finish"): {"result.json": {"configs": [
        {"chips": c, "pass": ok, "evidence": ["stages/4/evidence/hw-test-output.txt"],
         **({} if ok else {"reason": "does not fit"})} for c, ok in configs]}}}


```

with:

```python
def mesh_result(*configs):
    """A stage 4 result.json override from (chips, pass) pairs."""
    return {(4, "finish"): {"result.json": {"configs": [swap_entry(c, ok) for c, ok in configs]}}}


```

Create `tests/test_supervisor_stage4.py`:

```python
"""Stage 4 on the weights-only path: one hardware test per chip configuration, each under its own
lease, with the coder parked only when its boards are needed."""
import hashlib
import json

import pytest

from orchard import supervisor
from orchard.adapters import ChipState, Queued
from orchard.supervisor import EXIT_READY
from run_fakes import (BOARDS, FILES, FakeContainers, MachineAdapter, bringup, swap_entry,
                       swap_prepare, where)
from test_supervisor import Rig, Stop, escalation_aware

pytestmark = pytest.mark.usefixtures("stub_tools")


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)                     # the coder on two chips, board B0
    yield r
    r.close()


def stage4(rig, decision):
    return [e["data"] for e in rig.entries() if e["stage"] == 4 and e["event"] == "decision"
            and e["data"].get("decision") == decision]


def stop():
    raise Stop()


def test_a_weights_only_run_gives_stage_4_the_configs_skill_and_runs_each_configuration_once(rig):
    assert rig.run() == EXIT_READY
    skills = [d["skill"].rsplit("/", 1)[1] for d in stage4(rig, "agent step")]
    assert skills == ["weights-swap-configs.md"] * 2                  # prepare, finish
    assert [d["config"] for d in stage4(rig, "hardware test started")] == [1, 2, 4]
    sd = rig.run_dir / "stages" / "4"
    assert json.loads((sd / "tests" / "plan.json").read_text())["tests"][2]["chips"] == 4
    assert [json.loads((sd / "tests" / str(n) / "test-result.json").read_text())["returncode"]
            for n in (1, 2, 4)] == [0, 0, 0]
    summary = json.loads((sd / "test-result.json").read_text())
    assert [t["config"] for t in summary["tests"]] == [1, 2, 4]
    assert [d["result"] for d in rig.ends(4)] == ["pass"]
    listed = [e["data"] for e in rig.entries() if e["event"] == "evidence" and e["stage"] == 4
              and e["data"]["what"] == "hardware test list"]
    assert listed[0]["configs"] == [1, 2, 4] and listed[0]["path"] == "stages/4/tests/plan.json"


def test_the_tests_get_their_chips_device_ids_and_the_runs_container_label(rig):
    assert rig.run() == EXIT_READY
    label = "orchard.test=" + hashlib.sha256(str(rig.run_dir.resolve()).encode()).hexdigest()[:12]
    for n, chips, ids in ((1, BOARDS["B1"][:1], "2"), (2, BOARDS["B1"], "2,3"),
                          (4, BOARDS["B0"] + BOARDS["B1"], "0,1,2,3")):
        env = (rig.run_dir / "stages" / "4" / "configs" / str(n) / "evidence" / "devices.txt").read_text()
        assert env.splitlines() == [f"TT_VISIBLE_DEVICES={','.join(chips)}", f"ORCHARD_DEVICE_IDS={ids}",
                                    f"ORCHARD_TEST_LABEL={label}"], n
    assert ("list", label) in rig.containers.calls


class SnapshotAtSpawn:
    """Wraps spawn_checked: records the machine's state at the moment each test starts."""

    def __init__(self, rig, real):
        self.rig, self.real, self.seen = rig, real, []

    def __call__(self, command, run_dir, env, timeout, out):
        m = self.rig.m
        self.seen.append({"command": command, "chips": env["TT_VISIBLE_DEVICES"].split(","),
                          "coder_running": m.coder_running,
                          "leases": {lid: (set(lease.chips), owner) for lid, (lease, owner) in m.leases.items()}})
        return self.real(command, run_dir, env, timeout, out)


class BoardOneBusy(MachineAdapter):
    """Another tenant holds board B1 until `busy` lease attempts for it have been turned away. While
    it is busy, gozer status shows B1's chips HELD by someone else."""
    busy = 0
    coder_up_while_queued: list = []

    def acquire(self, chips, who, reason, *, queue=False, exact=None):
        if BoardOneBusy.busy and (exact is None or exact in BOARDS["B1"]):
            BoardOneBusy.busy -= 1
            BoardOneBusy.coder_up_while_queued.append(self.m.coder_running)
            raise Queued("t-busy")
        return super().acquire(chips, who, reason, queue=queue, exact=exact)

    def status(self):
        out = super().status()
        if not BoardOneBusy.busy:
            return out
        return [ChipState(c.bdf, "HELD", "someone-else", board=c.board, dev_index=c.dev_index, lease_pid=999)
                if c.board == "B1" else c for c in out]


def test_the_4_chip_test_waits_for_a_board_another_tenant_holds_and_runs_only_on_this_runs_leases(rig, monkeypatch):
    BoardOneBusy.busy, BoardOneBusy.coder_up_while_queued = 0, []
    rig.adapter_cls = BoardOneBusy

    def script(request):
        if "tools" in request and where(request) == (4, "prepare"):
            BoardOneBusy.busy = 3                  # the other tenant takes B1 as stage 4 starts
        return bringup(request)
    rig.script = script
    spy = SnapshotAtSpawn(rig, supervisor.spawn_checked)
    monkeypatch.setattr(supervisor, "spawn_checked", spy)
    assert rig.run() == EXIT_READY
    # While B1 was busy the coder stayed up: the further board is leased before the park.
    assert BoardOneBusy.coder_up_while_queued == [True, True, True]
    four = next(s for s in spy.seen if s["command"].endswith("configs/4/serve_and_compare_container.py"))
    assert four["coder_running"] is False
    assert sorted(four["chips"]) == sorted(BOARDS["B0"] + BOARDS["B1"])
    assert all(owner == rig.m.owner_pid for _, owner in four["leases"].values())
    assert set().union(*(chips for chips, _ in four["leases"].values())) == set(four["chips"])


def failing_4_chip_test(required_ok=True):
    files = swap_prepare()
    files["configs/4/serve_and_compare_container.py"] = "import sys\nprint('server never healthy')\nsys.exit(4)\n"
    return {(4, "prepare"): files,
            (4, "finish"): {"result.json": {"configs": [swap_entry(1), swap_entry(2), swap_entry(4, ok=False)]}}}


def test_a_failed_4_chip_test_still_releases_its_board_and_restores_the_coder(rig):
    rig.args.required_chips = (2,)                 # here the 4-chip configuration is optional
    rig.script = lambda r: bringup(r, overrides=failing_4_chip_test())
    assert rig.run() == EXIT_READY
    es = [e for e in rig.entries() if e["stage"] == 4]
    seq = [(e["event"], e["data"].get("step") or e["data"].get("decision"), e["data"].get("config")) for e in es]
    test = seq.index(("decision", "hardware test started", 4))
    after = seq[test:]
    assert ("decision", "test lease released", None) in after
    # The coder comes back before the finish step, and through the restore itself: a restore left
    # for the next stage's recovery would also bring it back, one stage later.
    assert after.index(("restore", "canary", None)) < after.index(("restore", "resumed", None))
    assert after.index(("restore", "resumed", None)) < after.index(("decision", "agent step", None))
    assert not [x for x in seq if x[1] == "recover after restart; the machine wins"]
    rec = json.loads((rig.run_dir / "stages" / "4" / "tests" / "4" / "test-result.json").read_text())
    assert rec["returncode"] == 4
    assert [d["result"] for d in rig.ends(4)] == ["pass"] and rig.ends(5)[0]["result"] == "pass"


def test_a_pass_claimed_for_a_configuration_whose_test_never_ran_fails_the_gate(rig):
    # The agent drops the 4-chip test from the list and writes the files a passing test would
    # leave. Only the supervisor's record is missing.
    files = swap_prepare()
    files["hw_tests.json"] = {"tests": [t for t in files["hw_tests.json"]["tests"] if t["chips"] != 4]}
    files["configs/4/evidence/swap-check.json"] = {"result_draft": {"top1_agreement": 0.94}}
    files["tests/4/output.txt"] = "a test that never ran"
    rig.script = escalation_aware(4, {(4, "prepare"): files})
    assert rig.run() == EXIT_READY
    first = next(e["data"] for e in rig.entries() if e["event"] == "escalate" and e["stage"] == 4)
    assert first["reasons"] == ["the 4-chip configuration claims a pass, but the supervisor has no "
                                "record of its test (tests/4/test-result.json)"]


def test_a_list_whose_configurations_share_a_cache_fails_before_any_hardware_is_used(rig):
    shared = str(rig.tmp / "orchard-cache" / "one-cache")
    rig.script = escalation_aware(4, {(4, "prepare"): swap_prepare(caches={1: shared, 2: shared})})
    assert rig.run() == EXIT_READY
    first = next(e for e in rig.entries() if e["event"] == "escalate" and e["stage"] == 4)
    assert first["data"]["reasons"] == [f"tests[1]: tt_cache {shared} is also the 1-chip "
                                        "configuration's; each configuration needs its own"]
    before = [e for e in rig.entries() if e["seq"] < first["seq"] and e["stage"] == 4
              and e["data"].get("decision") in ("hardware test started", "test lease taken")]
    assert before == []


def test_a_list_missing_a_required_configuration_fails_before_any_hardware_is_used(rig):
    rig.args.required_chips = (2, 4)
    files = swap_prepare()
    files["hw_tests.json"] = {"tests": [t for t in files["hw_tests.json"]["tests"] if t["chips"] != 4]}
    rig.script = escalation_aware(4, {(4, "prepare"): files})
    assert rig.run() == EXIT_READY
    first = next(e["data"] for e in rig.entries() if e["event"] == "escalate" and e["stage"] == 4)
    assert first["reasons"] == ["the 4-chip configuration is required and hw_tests.json has no test for it"]


def test_a_container_a_test_left_running_is_stopped_before_its_lease_goes_back(rig):
    rig.containers = FakeContainers([[], ["c1"], []])      # found after the 2-chip test, then gone
    assert rig.run() == EXIT_READY
    notes = [e for e in rig.entries() if e["event"] == "notice" and e["stage"] == 4]
    assert [n["data"]["containers"] for n in notes] == [["c1"]]
    assert ("remove", "c1") in rig.containers.calls
    seq = [(e["event"], e["data"].get("decision"), e["data"].get("lease_id")) for e in rig.entries()
           if e["stage"] == 4]
    note_at = next(i for i, x in enumerate(seq) if x[0] == "notice")
    released = [i for i, x in enumerate(seq) if x[1] == "test lease released"]
    assert released[0] < note_at < released[1]           # after the 1-chip release, before the 2-chip one


def test_a_container_that_will_not_stop_blocks_before_any_lease_is_released_or_the_coder_restored(rig):
    rig.containers = FakeContainers([[], [], ["c9"]])       # the 4-chip test's container stays
    at_pause = {}

    def paused():
        at_pause.update(leases=len(rig.m.leases), coder=rig.m.coder_running)
        raise Stop()
    rig.on_sleep = paused
    with pytest.raises(Stop):
        rig.run()
    blocked = [e["data"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked")]
    assert blocked[-1]["reason"].startswith("test containers are still running after docker stop and rm")
    assert blocked[-1]["evidence"]["containers"] == ["c9"]
    es = [e for e in rig.entries() if e["stage"] == 4]
    test = max(i for i, e in enumerate(es) if e["data"].get("decision") == "hardware test started")
    assert not [e for e in es[test:] if e["event"] == "restore"]
    assert not [e for e in es[test:] if e["data"].get("decision") == "test lease released"]
    assert at_pause == {"leases": 2, "coder": False}       # the coder's lease and the further board's
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_stage4.py tests/test_supervisor.py tests/test_run_e2e.py tests/test_stages.py`
Expected: 65 FAIL, 98 passed. Most fail with `TypeError: build() got an unexpected keyword argument 'containers'`; `test_only_the_weights_only_path_runs_stage_4_as_a_list_of_tests` fails because `spec_for(4, "weights-only")` is still `STAGES[4]`.

- [ ] **Step 3: Implement**

In `orchard/stages.py`, replace:

```python
def spec_for(number: int, path: str | None) -> StageSpec:
    """The stage spec for `number` on `path`. Only the weights-only path changes the table: stage 2
    gets the swap skill and gate, and stage 3 is skipped (the supervisor records it as skipped,
    as it does stage 7)."""
    if path == "weights-only" and number == 2:
        return WEIGHTS_ONLY_STAGE_2
    if path == "weights-only" and number == 3:
        return WEIGHTS_ONLY_STAGE_3
    return STAGES[number]

```

with:

```python
def spec_for(number: int, path: str | None) -> StageSpec:
    """The stage spec for `number` on `path`. Only the weights-only path changes the table: stage 2
    gets the swap skill and gate, stage 3 is skipped (the supervisor records it as skipped, as it
    does stage 7), and stage 4 runs one test per chip configuration."""
    if path == "weights-only" and number == 2:
        return WEIGHTS_ONLY_STAGE_2
    if path == "weights-only" and number == 3:
        return WEIGHTS_ONLY_STAGE_3
    if path == "weights-only" and number == 4:
        return WEIGHTS_ONLY_STAGE_4
    return STAGES[number]

```

In `orchard/supervisor.py`, replace:

```python
Plan 4 runs stages 0 to 6 and 8. Stage 7 (package and container build) is recorded as skipped,
and so is stage 3 on the weights-only path.
"""
from __future__ import annotations
```

with:

```python
Plan 4 runs stages 0 to 6 and 8. Stage 7 (package and container build) is recorded as skipped,
and so is stage 3 on the weights-only path.

On the weights-only path stage 4 runs a list of hardware tests, one per chip configuration
(orchard/hwtests.py). The prepare step writes hw_tests.json; the supervisor validates it, writes
tests/plan.json (the resume marker) and runs the tests in order of chip count, each under its own
lease, then the finish step writes result.json. Each test's record is written as soon as it ends,
so a crash resumes at the first configuration with no record. For each test:
- enough free boards: lease them (one board: the free one, by its first chip) and leave the coder
  loaded;
- otherwise the coder's boards are needed: first lease any further boards the test needs (so a
  board that cannot be had leaves the coder untouched), then park the coder, run the test on the
  coder's chips plus the further boards, release the further boards and restore the coder. A
  test that fails is recorded, and the coder is restored all the same. With the coder on two
  chips, the 4-chip test is the only one that parks. Measured costs of one park and restore on
  this box: `tt-model stop` 1 to 2 s, a gozer reset 41.7 s (twice: park and restore), the 2-chip
  coder back to ready in about 120 s. The 4-chip coder's restart was not measured.
After every test the supervisor stops and removes any container that still carries this run's test
label (`run_label`); a container that survives that blocks the stage before any lease is released
or reset.
"""
from __future__ import annotations
```

In `orchard/supervisor.py`, replace:

```python
import contextlib
import functools
import json
import os
```

with:

```python
import contextlib
import functools
import hashlib
import json
import os
```

In `orchard/supervisor.py`, replace:

```python
from orchard.defaults import (AGENT_CONTINUATION_TURNS, CHIPS_PER_BOARD, CMD_TIMEOUT_S,
                              COLD_BOOT_BUDGET_S, CONTROL_POLL_S, FIRST_BOOT_EXPECTED,
                              FIRST_BOOT_PROMPT, RUN_CANARY_PROMPT)
from orchard.handoff import (NOTE_KEYS, Blocked, Budgets, Handoff, decide_park, progress, reacquire,
                             recover, wait_stopped)
from orchard.ledger import Ledger, LedgerCorrupt, LedgerLocked
from orchard.runner import Denied, check_string
```

with:

```python
from orchard.defaults import (AGENT_CONTINUATION_TURNS, CHIPS_PER_BOARD, CMD_TIMEOUT_S,
                              COLD_BOOT_BUDGET_S, CONTROL_POLL_S, FIRST_BOOT_EXPECTED,
                              FIRST_BOOT_PROMPT, RUN_CANARY_PROMPT, STOP_TIMEOUT_S)
from orchard.handoff import (NOTE_KEYS, Blocked, Budgets, Handoff, decide_park, progress, reacquire,
                             recover, wait_stopped)
from orchard.hwtests import load_plan, pending, read_plan, write_plan, write_record, write_summary
from orchard.ledger import Ledger, LedgerCorrupt, LedgerLocked
from orchard.runner import Denied, check_string
```

In `orchard/supervisor.py`, replace:

```python
        self._stop = None
        self._nudges.clear()


```

with:

```python
        self._stop = None
        self._nudges.clear()


class LabelledContainers:
    """Containers this run's hardware tests started, found by their docker label.

    A test script stops its own container in a `finally`. A test killed at its deadline runs no
    `finally`, and a container started with `docker run --detach` is not in the killed session, so
    it keeps the chips. The supervisor looks for the label after every test of a list.
    """

    def __init__(self, run=run_command, docker: str = "docker"):
        self.run, self.docker = run, docker

    def list(self, label: str) -> list[str] | None:
        """The ids of containers with `label`, or None when docker could not be asked."""
        r = self.run([self.docker, "ps", "--all", "--quiet", "--filter", f"label={label}"], CMD_TIMEOUT_S)
        return r.stdout.split() if r.returncode == 0 else None

    def remove(self, cid: str) -> None:
        self.run([self.docker, "stop", "-t", "60", cid], STOP_TIMEOUT_S)
        self.run([self.docker, "rm", "--force", cid], CMD_TIMEOUT_S)


```

In `orchard/supervisor.py`, replace:

```python
                 sleep=time.sleep, budgets: Budgets = Budgets(), disk_usage=shutil.disk_usage,
                 credentials_visible: list[str] | None = None,
                 required_chips: tuple[int, ...] | None = None):
        self.run_dir = Path(run_dir).resolve()
        self.ledger, self.cfg, self.model_id = ledger, cfg, model_id
```

with:

```python
                 sleep=time.sleep, budgets: Budgets = Budgets(), disk_usage=shutil.disk_usage,
                 credentials_visible: list[str] | None = None,
                 required_chips: tuple[int, ...] | None = None, home=None, containers=None):
        self.run_dir = Path(run_dir).resolve()
        self.ledger, self.cfg, self.model_id = ledger, cfg, model_id
```

In `orchard/supervisor.py`, replace:

```python
        self.credentials_visible = list(credentials_visible or [])
        self.required_chips = tuple(required_chips) if required_chips else None   # stage 4's required counts
        self.control = Control(self.run_dir)
        self.actuator = RunActuator(self.control)
```

with:

```python
        self.credentials_visible = list(credentials_visible or [])
        self.required_chips = tuple(required_chips) if required_chips else None   # stage 4's required counts
        self.home = Path(home) if home is not None else operator_home()   # whose shared caches to refuse
        self.containers = containers if containers is not None else LabelledContainers()
        # The docker label every test container of this run carries (ORCHARD_TEST_LABEL).
        self.run_label = "orchard.test=" + hashlib.sha256(str(self.run_dir).encode()).hexdigest()[:12]
        self.control = Control(self.run_dir)
        self.actuator = RunActuator(self.control)
```

In `orchard/supervisor.py`, replace:

```python
                return out.status, [f"the agent step ended: {out.status} {out.detail}".strip()], None
        else:
            if not (resumed and (stage_dir / "test-result.json").is_file()):
                out, _ = self._step(spec, "prepare", stage_dir, escalated, resumed)
                if out.status != "done":
```

with:

```python
                return out.status, [f"the agent step ended: {out.status} {out.detail}".strip()], None
        else:
            if spec.tests:
                ended = self._run_test_list(spec, stage_dir, escalated, resumed)
                if ended:
                    return ended
            elif not (resumed and (stage_dir / "test-result.json").is_file()):
                out, _ = self._step(spec, "prepare", stage_dir, escalated, resumed)
                if out.status != "done":
```

In `orchard/supervisor.py`, replace:

```python
    def _run_test(self, spec, stage_dir: Path, test: dict, lease: Lease) -> dict:
        n = spec.number
        chips = list(lease.chips[:spec.boards * CHIPS_PER_BOARD])
        env = agent_env(self.run_dir, extra=self.extra_env)
        # The leased chips replace the agent shells' no-chip mask. TT_METAL_VISIBLE_DEVICES takes
        # device indices in another form, so the hardware test gets none and the runtime follows
        # TT_VISIBLE_DEVICES.
        env["TT_VISIBLE_DEVICES"] = ",".join(chips)
        env.pop("TT_METAL_VISIBLE_DEVICES", None)
        deadline = min(float(test["deadline_s"]), spec.budget_s)
        out_path = stage_dir / "evidence" / "hw-test-output.txt"
        self.ledger.append("decision", n, decision="hardware test started", command=test["command"],
                           deadline_s=deadline, chips=chips, lease_id=lease.lease_id)
        t0 = self.clock()
        with open(out_path, "wb") as out:
            try:
                code, timed_out = spawn_checked(test["command"], self.run_dir, env, deadline, out)
            except Denied as exc:
                code, timed_out = None, False
                out.write(f"refused: {exc}\n".encode("utf-8"))
        result = {"command": test["command"], "returncode": code, "timed_out": timed_out,
                  "seconds": round(self.clock() - t0, 3), "chips": chips,
                  "output": evidence_record(self.run_dir, out_path)}
        marker = stage_dir / "test-result.json"
        tmp = stage_dir / "test-result.json.tmp"
        tmp.write_text(json.dumps(result, indent=2))
        os.replace(tmp, marker)         # the resume marker appears whole or not at all
        self.ledger.append("evidence", n, what="hardware test", returncode=code, timed_out=timed_out,
                           **evidence_record(self.run_dir, marker))
        return result


```

with:

```python
    def _run_test(self, spec, stage_dir: Path, test: dict, lease: Lease) -> dict:
        n = spec.number
        k = spec.boards * CHIPS_PER_BOARD
        deadline = min(float(test["deadline_s"]), spec.budget_s)
        result = self._spawn(n, test["command"], lease.chips[:k], lease.dev_indices[:k], deadline,
                             stage_dir / "evidence" / "hw-test-output.txt", lease_id=lease.lease_id)
        marker = stage_dir / "test-result.json"
        tmp = stage_dir / "test-result.json.tmp"
        tmp.write_text(json.dumps(result, indent=2))
        os.replace(tmp, marker)         # the resume marker appears whole or not at all
        self.ledger.append("evidence", n, what="hardware test", returncode=result["returncode"],
                           timed_out=result["timed_out"], **evidence_record(self.run_dir, marker))
        return result

    def _spawn(self, n: int, command: str, chips, ids, deadline: float, out_path: Path,
               **record) -> dict:
        """Run one hardware test command on `chips` (device indices `ids`) and return its record.
        `record` adds keys to the "hardware test started" decision."""
        env = agent_env(self.run_dir, extra=self.extra_env)
        # The leased chips replace the agent shells' no-chip mask. TT_METAL_VISIBLE_DEVICES takes
        # device indices in another form, so the hardware test gets none and the runtime follows
        # TT_VISIBLE_DEVICES. A container test needs the /dev/tenstorrent indices of the same
        # chips (ORCHARD_DEVICE_IDS) and the label its containers carry (ORCHARD_TEST_LABEL).
        env["TT_VISIBLE_DEVICES"] = ",".join(chips)
        env.pop("TT_METAL_VISIBLE_DEVICES", None)
        env["ORCHARD_DEVICE_IDS"] = ",".join(str(i) for i in ids)
        env["ORCHARD_TEST_LABEL"] = self.run_label
        self.ledger.append("decision", n, decision="hardware test started", command=command,
                           deadline_s=deadline, chips=list(chips), **record)
        t0 = self.clock()
        with open(out_path, "wb") as out:
            try:
                code, timed_out = spawn_checked(command, self.run_dir, env, deadline, out)
            except Denied as exc:
                code, timed_out = None, False
                out.write(f"refused: {exc}\n".encode("utf-8"))
        return {"command": command, "returncode": code, "timed_out": timed_out,
                "seconds": round(self.clock() - t0, 3), "chips": list(chips),
                "output": evidence_record(self.run_dir, out_path)}

    # ---- a list of hardware tests (stage 4 on the weights-only path) ---------------------------

    def _run_test_list(self, spec, stage_dir: Path, escalated: bool, resumed: bool):
        """Prepare the list (unless a resumed stage has its plan), then run every test that has
        no record. Returns (status, reasons, gate) when the stage ends here, or None when the
        finish step should run."""
        n = spec.number
        tests = load_plan(stage_dir) if resumed else None
        if tests is None:
            out, _ = self._step(spec, "prepare", stage_dir, escalated, resumed)
            if out.status != "done":
                return out.status, [f"the prepare step ended: {out.status} {out.detail}".strip()], None
            tests, problems = read_plan(stage_dir, required=self.required_chips,
                                        max_chips=spec.boards * CHIPS_PER_BOARD,
                                        budget_s=spec.budget_s, home=self.home)
            if problems:
                return "fail", problems, None
            plan = write_plan(stage_dir, tests)
            self.ledger.append("evidence", n, what="hardware test list",
                               configs=[t.chips for t in tests], **evidence_record(self.run_dir, plan))
        # A test lease left from a block (a container that would not stop) goes back first.
        if not self._release_test_lease():
            self._block(n, "releasing a test lease left from an earlier test failed")
        for test in pending(stage_dir, tests):
            self._run_listed(spec, stage_dir, test)
        write_summary(stage_dir, tests)
        return None

    def _run_listed(self, spec, stage_dir: Path, test) -> None:
        n = spec.number
        chips = self.adapter.status()
        d = decide_park(self.coder_lease.units, chips, test.boards)
        self.ledger.append("decision", n, decision="hardware phase", config=test.chips,
                           action=d.action, free_boards=list(d.free_boards),
                           server_boards=list(d.server_boards))
        first_chip = {c.board: c.bdf for c in reversed(chips)}       # each board's first chip
        if d.action == "use_free":
            exact = first_chip[d.free_boards[0]] if test.boards == 1 else None
            lease = self._take_test_lease(spec, test, test.boards, exact)
            self._run_listed_test(spec, stage_dir, test, [lease])
            self._sweep(n)
            self._give_back_test_lease(n, lease)
            return
        # The coder's boards are needed. Further boards first: if one cannot be had, the coder
        # has not been touched.
        extra = None
        more = test.boards - len(self.coder_lease.units)
        if more > 0:
            exact = first_chip[d.free_boards[0]] if d.free_boards else None
            extra = self._take_test_lease(spec, test, more, exact)
        h = self._handoff(n, self.coder_lease)
        try:
            lease = h.park()
        except Blocked:
            self._release_test_lease()
            raise
        self._run_listed_test(spec, stage_dir, test, [lease] + ([extra] if extra else []))
        self._sweep(n)
        if extra is not None:
            self._give_back_test_lease(n, extra)
        h.restore()
        self.coder_lease = h.lease

    def _take_test_lease(self, spec, test, boards: int, exact) -> Lease:
        n = spec.number
        lease = reacquire(self.adapter, chips=boards * CHIPS_PER_BOARD, who=WHO,
                          reason=f"stage {n} {test.chips}-chip test", ledger=self.ledger, stage=n,
                          wait_budget_s=spec.budget_s, clock=self.clock, sleep=self.sleep,
                          exact=exact)
        self.test_lease = lease
        self.ledger.append("decision", n, decision="test lease taken", config=test.chips,
                           test_lease=lease.record())
        return lease

    def _give_back_test_lease(self, n: int, lease: Lease) -> None:
        try:
            self.adapter.release(lease)        # the lease tool resets the board as it releases
        except AdapterError as exc:
            self._block(n, f"releasing the test lease failed: {exc}", lease_id=lease.lease_id)
        self.test_lease = None
        self.ledger.append("decision", n, decision="test lease released", lease_id=lease.lease_id)

    def _run_listed_test(self, spec, stage_dir: Path, test, leases: list[Lease]) -> None:
        """Run one configuration's test on the first `test.chips` chips of `leases`, in device
        order, and write its record."""
        n = spec.number
        held = sorted((i, c) for lease in leases for c, i in zip(lease.chips, lease.dev_indices))
        chosen = held[:test.chips]
        out_dir = stage_dir / "tests" / str(test.chips)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = self._spawn(n, test.command(n), [c for _, c in chosen], [i for i, _ in chosen],
                             min(test.deadline_s, spec.budget_s), out_dir / "output.txt",
                             config=test.chips, cache=test.cache,
                             lease_ids=[lease.lease_id for lease in leases])
        path = write_record(stage_dir, test.chips, {"config": test.chips, **result,
                                                    "device_ids": [i for i, _ in chosen]})
        self.ledger.append("evidence", n, what="hardware test", config=test.chips, cache=test.cache,
                           returncode=result["returncode"], timed_out=result["timed_out"],
                           **evidence_record(self.run_dir, path))

    def _sweep(self, n: int) -> None:
        """Stop and remove any container this run's tests left running. A container that is
        still listed afterwards blocks the stage: releasing or resetting chips under it is
        refused, and the operator decides."""
        found = self.containers.list(self.run_label)
        if found is None:
            self.ledger.append("notice", n, what="docker could not be asked for test containers",
                               label=self.run_label)
            return
        if not found:
            return
        self.ledger.append("notice", n, what="a hardware test left containers running; stopping them",
                           containers=found, label=self.run_label)
        for cid in found:
            self.containers.remove(cid)
        left = self.containers.list(self.run_label)
        if left:
            self._block(n, "test containers are still running after docker stop and rm; no test "
                           "lease was released and the coder was not restored", containers=left)


```

In `orchard/supervisor.py`, replace:

```python
def build(args, ledger, *, adapter=None, coder=None, versions=None, http=post_json,
          probe=probe_model, clock=time.time, sleep=time.sleep, budgets=Budgets(),
          disk_usage=shutil.disk_usage, home=None) -> Supervisor:
    """A Supervisor from parsed `run` arguments. Tests pass fakes for the machine."""
    # Everything that can be refused is checked before any external command runs.
```

with:

```python
def build(args, ledger, *, adapter=None, coder=None, versions=None, http=post_json,
          probe=probe_model, clock=time.time, sleep=time.sleep, budgets=Budgets(),
          disk_usage=shutil.disk_usage, home=None, containers=None) -> Supervisor:
    """A Supervisor from parsed `run` arguments. Tests pass fakes for the machine."""
    # Everything that can be refused is checked before any external command runs.
```

In `orchard/supervisor.py`, replace:

```python
                      extra_env=extra_env, versions=versions, http=http, probe=probe,
                      clock=clock, sleep=sleep, budgets=budgets, disk_usage=disk_usage,
                      credentials_visible=found, required_chips=required)


```

with:

```python
                      extra_env=extra_env, versions=versions, http=http, probe=probe,
                      clock=clock, sleep=sleep, budgets=budgets, disk_usage=disk_usage,
                      credentials_visible=found, required_chips=required,
                      home=operator_home() if home is None else home, containers=containers)


```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_stage4.py tests/test_supervisor.py tests/test_run_e2e.py tests/test_stages.py`
Expected: 163 passed. The e2e kill test (`test_a_kill_after_any_ledger_event_reaches_the_same_final_state`) now walks stage 4's three tests and parks too; it takes about twice as long as before.

- [ ] **Step 5: Mutations**

Against `tests/test_supervisor_stage4.py -k <selector>`; fail, restore, delete the bytecode, pass:
1. In `_run_listed`, delete the line `        h.restore()` (keep `self.coder_lease = h.lease`). Selector `failed_4_chip`: 1 failed (the coder comes back only through the next stage's recovery).
2. In the `use_free` branch of `_run_listed`, delete the line `            self._sweep(n)`. Selector `left_running`: 1 failed.
3. Change `        self._run_listed_test(spec, stage_dir, test, [lease] + ([extra] if extra else []))` to `        self._run_listed_test(spec, stage_dir, test, [lease])`. Selector `device_ids`: 1 failed.
4. Move the park before the further lease: replace the block from `        extra = None` through the `except Blocked:` clause with `h = self._handoff(n, self.coder_lease)`, `lease = h.park()`, then the `extra = None ... extra = self._take_test_lease(...)` lines. Selector `another_tenant`: 1 failed (the coder was stopped while the queue waited).
5. In `_spawn`, delete `        env["ORCHARD_TEST_LABEL"] = self.run_label`. Selector `device_ids`: 1 failed.

- [ ] **Step 6: Run the suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -1` (expected `1404 passed, 1 skipped`), then:

```bash
git add orchard/supervisor.py orchard/stages.py tests/run_fakes.py tests/test_supervisor.py tests/test_run_e2e.py tests/test_stages.py tests/test_supervisor_stage4.py
git commit -m "Supervisor: run stage 4's tests one configuration at a time, each under its own lease"
```

---

### Task 8: A test record counts only if the supervisor wrote it

**Files:**
- Modify: `orchard/hwtests.py` (docstring, import, append `unrecorded`), `orchard/supervisor.py` (imports, `_check_gate`)
- Test: `tests/test_hwtests.py` (append), `tests/test_supervisor_stage4.py` (append)

**Interfaces:**
- Consumes: the `what="hardware test"` evidence entries Task 7 writes (`path`, `sha256`); `evidence_record` (`orchard/stages.py`); `GateResult`.
- Produces: `unrecorded(entries, stage_dir, run_dir, stage: int) -> list[str]`, one reason per `tests/*/test-result.json` whose sha256 is not the latest the ledger holds for its path. `_check_gate` appends those reasons to the gate's for a stage with `tests`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hwtests.py`:

```python


def test_a_record_counts_only_with_the_sha256_the_ledger_recorded(tmp_path):
    from orchard.hwtests import unrecorded
    from orchard.stages import evidence_record
    run = tmp_path / "run"
    sd = run / "stages" / "4"
    path = write_record(sd, 2, {"returncode": 0, "timed_out": False, "chips": ["a", "b"]})
    good = [{"event": "evidence", "stage": 4, "data": {"what": "hardware test", "config": 2,
                                                       **evidence_record(run, path)}}]
    assert unrecorded(good, sd, run, 4) == []
    assert unrecorded([], sd, run, 4) == ["stages/4/tests/2/test-result.json was not written by the "
                                          "supervisor: its sha256 is not the one the ledger recorded"]
    path.write_text(path.read_text().replace('"returncode": 0', '"returncode": 0 '))
    assert len(unrecorded(good, sd, run, 4)) == 1
```

Append to `tests/test_supervisor_stage4.py`:

```python


def test_a_test_record_the_agent_wrote_fails_the_gate(rig):
    # The 4-chip test exits 4. The finish step claims a pass and rewrites the record to match.
    forged = {"config": 4, "returncode": 0, "timed_out": False, "chips": ["a", "b", "c", "d"]}
    bad = failing_4_chip_test()
    bad[(4, "finish")] = {"tests/4/test-result.json": forged,
                          "result.json": {"configs": [swap_entry(1), swap_entry(2), swap_entry(4)]}}
    bad[(4, "prepare")]["configs/4/evidence/swap-check.json"] = {"result_draft": {"top1_agreement": 0.94}}
    rig.script = escalation_aware(4, bad)
    assert rig.run() == EXIT_READY
    first = next(e["data"] for e in rig.entries() if e["event"] == "escalate" and e["stage"] == 4)
    assert first["reasons"] == ["stages/4/tests/4/test-result.json was not written by the supervisor: "
                                "its sha256 is not the one the ledger recorded"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_hwtests.py tests/test_supervisor_stage4.py`
Expected: 2 FAIL, 28 passed: `ImportError: cannot import name 'unrecorded'`, and the supervisor test finds no escalation, because the forged record passed the gate.

- [ ] **Step 3: Implement**

In `orchard/hwtests.py`, replace:

```python
step.

A test that did not exit 0 may have stopped while it was converting weights into its tensor cache,
and a part-written cache is read back without an error. `suspect_caches` finds those caches in the
```

with:

```python
step.

An agent's write_file can write anywhere in its stage directory, tests/ included. `unrecorded`
compares every tests/<chips>/test-result.json with the sha256 the ledger recorded when the
supervisor wrote it, so a record the agent wrote or changed fails the stage.

A test that did not exit 0 may have stopped while it was converting weights into its tensor cache,
and a part-written cache is read back without an error. `suspect_caches` finds those caches in the
```

In `orchard/hwtests.py`, replace:

```python
from orchard.defaults import CHIPS_PER_BOARD
from orchard.handoff import NOTE_KEYS
from orchard.stages import hw_record_path

SCRIPTS = {"serve_and_compare.py": "bundle", "serve_and_compare_container.py": "container"}
```

with:

```python
from orchard.defaults import CHIPS_PER_BOARD
from orchard.handoff import NOTE_KEYS
from orchard.stages import evidence_record, hw_record_path

SCRIPTS = {"serve_and_compare.py": "bundle", "serve_and_compare_container.py": "container"}
```

Append to `orchard/hwtests.py`:

```python


def unrecorded(entries: list[dict], stage_dir, run_dir, stage: int) -> list[str]:
    """A reason for every tests/<chips>/test-result.json whose sha256 is not the one the ledger
    recorded for that path when the supervisor last wrote it."""
    latest: dict[str, str] = {}
    for e in entries:
        d = e["data"]
        if e["event"] == "evidence" and e["stage"] == stage and d.get("what") == "hardware test":
            latest[d.get("path")] = d.get("sha256")
    out = []
    for path in sorted(Path(stage_dir).glob("tests/*/test-result.json")):
        rec = evidence_record(run_dir, path)
        if latest.get(rec["path"]) != rec["sha256"]:
            out.append(f"{rec['path']} was not written by the supervisor: its sha256 is not the one "
                       "the ledger recorded")
    return out
```

In `orchard/supervisor.py`, replace:

```python
from orchard.handoff import (NOTE_KEYS, Blocked, Budgets, Handoff, decide_park, progress, reacquire,
                             recover, wait_stopped)
from orchard.hwtests import load_plan, pending, read_plan, write_plan, write_record, write_summary
from orchard.ledger import Ledger, LedgerCorrupt, LedgerLocked
from orchard.runner import Denied, check_string
from orchard.server import ServerControl, ServerError, ServerSpec, StopCheck
from orchard.stages import (TierUnavailable, attempt_started_ts, budget_cap, check_disk,
                            coder_state, delta_path, evidence_record, open_stage_dir,
                            resolve_endpoint, resolve_skill, run_path, run_progress, spec_for,
```

with:

```python
from orchard.handoff import (NOTE_KEYS, Blocked, Budgets, Handoff, decide_park, progress, reacquire,
                             recover, wait_stopped)
from orchard.hwtests import (load_plan, pending, read_plan, unrecorded, write_plan, write_record,
                             write_summary)
from orchard.ledger import Ledger, LedgerCorrupt, LedgerLocked
from orchard.runner import Denied, check_string
from orchard.server import ServerControl, ServerError, ServerSpec, StopCheck
from orchard.stages import (GateResult, TierUnavailable, attempt_started_ts, budget_cap, check_disk,
                            coder_state, delta_path, evidence_record, open_stage_dir,
                            resolve_endpoint, resolve_skill, run_path, run_progress, spec_for,
```

In `orchard/supervisor.py`, replace:

```python
            bundle.mkdir(exist_ok=True)
            shutil.copyfile(self.ledger.path, bundle / "ledger.jsonl")
        return self._gate(spec)(stage_dir, self.run_dir)

    def _gate_feedback(self, spec, stage_dir: Path, step: AgentStep, gate):
```

with:

```python
            bundle.mkdir(exist_ok=True)
            shutil.copyfile(self.ledger.path, bundle / "ledger.jsonl")
        gate = self._gate(spec)(stage_dir, self.run_dir)
        if spec.tests:
            # The gate reads the test records from the stage directory, where the agent can write
            # too. The ledger holds the sha256 of each record the supervisor wrote.
            forged = unrecorded(self.ledger.read(), stage_dir, self.run_dir, spec.number)
            if forged:
                return GateResult(False, gate.reasons + tuple(forged), gate.evidence)
        return gate

    def _gate_feedback(self, spec, stage_dir: Path, step: AgentStep, gate):
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_hwtests.py tests/test_supervisor_stage4.py`
Expected: 30 passed.

- [ ] **Step 5: Mutations**

1. In `_check_gate`, change `            if forged:` to `            if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_stage4.py -k agent_wrote`: 1 failed.
2. In `unrecorded`, change `        if latest.get(rec["path"]) != rec["sha256"]:` to `        if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_hwtests.py -k sha256`: 1 failed.
Restore each, delete the bytecode, run again: passed.

- [ ] **Step 6: Run the suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -1` (expected `1406 passed, 1 skipped`), then:

```bash
git add orchard/hwtests.py orchard/supervisor.py tests/test_hwtests.py tests/test_supervisor_stage4.py
git commit -m "A stage 4 test record counts only with the sha256 the ledger recorded"
```

---

### Task 9: Resume at the first configuration without a record; free disk and part-written caches

**Files:**
- Modify: `orchard/supervisor.py` (docstring, imports, the start of `_run_listed`)
- Test: `tests/test_supervisor_stage4.py` (append)

**Interfaces:**
- Consumes: `TEST_DISK_GB` (Task 4); `suspect_caches`, `move_aside` (Task 5); `check_disk` (`orchard/stages.py`).
- Produces: before each test, a block "the <N>-chip test needs 40.0 GB free on the run directory's disk; <free> GB is free" when short, and a decision "moved a tensor cache aside" (`cache`, `aside`, `reason`) when the test's cache is suspect and not empty.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_supervisor_stage4.py`:

```python


# ---- resuming the list, free disk, and part-written caches -----------------------------------------

def started(rig, config):
    return [d for d in stage4(rig, "hardware test started") if d["config"] == config]


def test_a_test_short_of_disk_blocks_and_the_resume_goes_on_at_that_test(rig):
    from collections import namedtuple
    from orchard.defaults import TEST_DISK_GB
    U = namedtuple("U", "total used free")
    first = rig.run_dir / "stages" / "4" / "tests" / "1" / "test-result.json"
    state = {"short": True}

    def usage(path):           # enough for the stage, too little once the 1-chip test is done
        return U(0, 0, 30e9 if state["short"] and first.is_file() else 10 ** 15)
    rig.usage = usage

    def freed():
        state["short"] = False
        supervisor.Control(rig.run_dir).write("resume")
    rig.on_sleep = freed
    assert rig.run() == EXIT_READY
    blocked = [e["data"]["reason"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked")]
    assert blocked == [f"the 2-chip test needs {TEST_DISK_GB} GB free on the run directory's disk; 30.0 GB is free"]
    assert [len(started(rig, c)) for c in (1, 2, 4)] == [1, 1, 1]      # the 1-chip test was not run again
    assert [d["result"] for d in rig.ends(4)] == ["pass"]


def kill_points(entries):
    """Stage 4's ledger events after which a kill is most telling: each test's start and record, each
    lease taken, and each reset of the park and the restore."""
    out = []
    for e in entries:
        d = e["data"]
        if e["stage"] != 4:
            continue
        if (e["event"] == "evidence" or d.get("decision") in ("hardware test started", "test lease taken")
                or (e["event"] in ("park", "restore") and d.get("step") == "reset")):
            out.append(e["seq"])
    return out


def fresh_rig(path, chips):
    path.mkdir(parents=True)
    return Rig(path, chips=chips)


@pytest.mark.parametrize("chips", [2, 4])
def test_a_kill_during_the_list_resumes_at_the_first_configuration_without_a_record(tmp_path, chips):
    from fakes import Crash
    ref = fresh_rig(tmp_path / "reference", chips)
    try:
        assert ref.run() == EXIT_READY
        points = kill_points(ref.entries())
    finally:
        ref.close()
    assert len(points) >= 9
    for k in points:
        r = fresh_rig(tmp_path / f"kill-{k}", chips)
        try:
            with pytest.raises(Crash):
                r.run(crash_if=lambda e, k=k: e["seq"] == k)
            recorded = {c for c in (1, 2, 4)
                        if (r.run_dir / "stages" / "4" / "tests" / str(c) / "test-result.json").is_file()}
            planned = (r.run_dir / "stages" / "4" / "tests" / "plan.json").is_file()
            assert r.run(pid=200) == EXIT_READY, k
            prepares = [d for d in stage4(r, "agent step") if d["phase"] == "prepare"]
            assert len(prepares) == 1 if planned else 1 <= len(prepares) <= 2, (k, len(prepares))
            for c in (1, 2, 4):
                runs = len(started(r, c))
                assert runs == 1 if c in recorded else 1 <= runs <= 2, (k, c, runs)
            assert [d["result"] for d in r.ends(4)] == ["pass"], k
            assert not r.m.coder_running and r.m.leases == {}, (k, r.m.leases)
        finally:
            r.close()


def test_a_cache_whose_test_was_killed_is_moved_aside_before_the_test_runs_again(rig):
    from fakes import Crash
    cache = rig.tmp / "orchard-cache" / "4chip" / "tt_cache"
    rig.script = lambda r: bringup(r, overrides={(4, "prepare"): swap_prepare(caches={4: str(cache)})})
    cache.mkdir(parents=True)

    def killed_in_test(e):
        if e["data"].get("decision") == "hardware test started" and e["data"].get("config") == 4:
            (cache / "layer0.bin").write_text("half converted")       # the conversion had begun
            return True
        return False
    with pytest.raises(Crash):
        rig.run(crash_if=killed_in_test)
    assert rig.run(pid=200) == EXIT_READY
    moved = stage4(rig, "moved a tensor cache aside")
    assert [(d["cache"], d["aside"]) for d in moved] == [(str(cache), f"{cache}.interrupted-1")]
    assert (rig.tmp / "orchard-cache" / "4chip" / "tt_cache.interrupted-1" / "layer0.bin").is_file()
    seqs = [e["seq"] for e in rig.entries() if e["data"].get("decision") in ("moved a tensor cache aside",
                                                                            "hardware test started")
            and e["data"].get("config", 4) == 4 and e["stage"] == 4]
    assert len(seqs) == 3                      # started (killed), moved aside, started again


def test_a_timed_out_test_leaves_its_cache_to_be_moved_aside_by_the_next_attempt(rig):
    rig.args.required_chips = (2, 4)
    cache = rig.tmp / "orchard-cache" / "4chip" / "tt_cache"
    cache.mkdir(parents=True)
    (cache / "layer0.bin").write_text("converted by a test that ran out of time")
    slow = swap_prepare(caches={4: str(cache)})
    slow["configs/4/serve_and_compare_container.py"] = "import time\ntime.sleep(30)\n"
    slow["hw_tests.json"]["tests"][2]["deadline_s"] = 1
    good = swap_prepare(caches={4: str(cache)})
    honest = {"result.json": {"configs": [swap_entry(1), swap_entry(2), swap_entry(4, ok=False)]}}
    rig.script = escalation_aware(4, {(4, "prepare"): slow, (4, "finish"): honest})

    def attempt_files(request):          # the escalated attempt writes the working files
        if "tools" in request and where(request) == (4, "prepare") and \
                "stage 4: escalate" in request["messages"][1]["content"]:
            return bringup(request, overrides={(4, "prepare"): good})
        return rig_script(request)
    rig_script, rig.script = rig.script, attempt_files
    assert rig.run() == EXIT_READY
    recs = [e["data"] for e in rig.entries() if e["event"] == "evidence" and e["stage"] == 4
            and e["data"].get("config") == 4]
    assert [(d["returncode"], d["timed_out"]) for d in recs] == [(None, True), (0, False)]
    moved = stage4(rig, "moved a tensor cache aside")
    assert [d["aside"] for d in moved] == [f"{cache}.interrupted-1"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_stage4.py`
Expected: 3 FAIL, 12 passed: the disk test (no block before the 2-chip test) and the two cache tests (no "moved a tensor cache aside" decision). The two kill tests pass already: they pin the resume behavior Tasks 4 and 7 built (`tests/plan.json` as the marker, `load_plan` on resume, one record per test), and the mutation step shows each can fail.

- [ ] **Step 3: Implement**

In `orchard/supervisor.py`, replace:

```python
After every test the supervisor stops and removes any container that still carries this run's test
label (`run_label`); a container that survives that blocks the stage before any lease is released
or reset.
"""
from __future__ import annotations
```

with:

```python
After every test the supervisor stops and removes any container that still carries this run's test
label (`run_label`); a container that survives that blocks the stage before any lease is released
or reset. Before each test it checks the free disk again (TEST_DISK_GB), and it moves aside a
tensor cache whose last test did not exit 0: a conversion that stopped part way leaves a cache
that reads back without an error.
"""
from __future__ import annotations
```

In `orchard/supervisor.py`, replace:

```python
from orchard.defaults import (AGENT_CONTINUATION_TURNS, CHIPS_PER_BOARD, CMD_TIMEOUT_S,
                              COLD_BOOT_BUDGET_S, CONTROL_POLL_S, FIRST_BOOT_EXPECTED,
                              FIRST_BOOT_PROMPT, RUN_CANARY_PROMPT, STOP_TIMEOUT_S)
from orchard.handoff import (NOTE_KEYS, Blocked, Budgets, Handoff, decide_park, progress, reacquire,
                             recover, wait_stopped)
from orchard.hwtests import (load_plan, pending, read_plan, unrecorded, write_plan, write_record,
                             write_summary)
from orchard.ledger import Ledger, LedgerCorrupt, LedgerLocked
from orchard.runner import Denied, check_string
```

with:

```python
from orchard.defaults import (AGENT_CONTINUATION_TURNS, CHIPS_PER_BOARD, CMD_TIMEOUT_S,
                              COLD_BOOT_BUDGET_S, CONTROL_POLL_S, FIRST_BOOT_EXPECTED,
                              FIRST_BOOT_PROMPT, RUN_CANARY_PROMPT, STOP_TIMEOUT_S, TEST_DISK_GB)
from orchard.handoff import (NOTE_KEYS, Blocked, Budgets, Handoff, decide_park, progress, reacquire,
                             recover, wait_stopped)
from orchard.hwtests import (load_plan, move_aside, pending, read_plan, suspect_caches, unrecorded,
                             write_plan, write_record, write_summary)
from orchard.ledger import Ledger, LedgerCorrupt, LedgerLocked
from orchard.runner import Denied, check_string
```

In `orchard/supervisor.py`, replace:

```python
    def _run_listed(self, spec, stage_dir: Path, test) -> None:
        n = spec.number
        chips = self.adapter.status()
        d = decide_park(self.coder_lease.units, chips, test.boards)
```

with:

```python
    def _run_listed(self, spec, stage_dir: Path, test) -> None:
        n = spec.number
        ok, free = check_disk(self.run_dir, TEST_DISK_GB, usage=self.disk_usage)
        if not ok:
            self._block(n, f"the {test.chips}-chip test needs {TEST_DISK_GB} GB free on the run "
                           f"directory's disk; {free} GB is free", need_gb=TEST_DISK_GB, free_gb=free)
        if test.cache in suspect_caches(self.ledger.read(), n):
            aside = move_aside(test.cache)
            if aside is not None:
                self.ledger.append("decision", n, decision="moved a tensor cache aside",
                                   reason="its last test did not exit 0, so it may be part-written",
                                   cache=test.cache, aside=str(aside))
        chips = self.adapter.status()
        d = decide_park(self.coder_lease.units, chips, test.boards)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_stage4.py`
Expected: 15 passed.

- [ ] **Step 5: Mutations**

Against `tests/test_supervisor_stage4.py -k <selector>`; fail, restore, delete the bytecode, pass:
1. In `_run_listed`, change `            aside = move_aside(test.cache)` to `            aside = None`. Selector `moved_aside`: 2 failed.
2. Change the `        if not ok:` line of the per-test disk check to `        if False:`. Selector `short_of_disk`: 1 failed.
3. In `_run_test_list`, change `        tests = load_plan(stage_dir) if resumed else None` to `        tests = None`. Selector `kill_during`: 2 failed (the resumed run prepares again).
4. In `orchard/stages.py`, change `marker="tests/plan.json", tests=True,` to `marker="test-result.json", tests=True,`. Selector `kill_during`: 2 failed (the stage directory is moved aside and recorded tests run again).

- [ ] **Step 6: Run the suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -1` (expected `1411 passed, 1 skipped`), then:

```bash
git add orchard/supervisor.py tests/test_supervisor_stage4.py
git commit -m "Stage 4 tests: check free disk before each, move a part-written cache aside"
```

---

### Task 10: Gate feedback and resume for a list follow the required configurations' records

**Files:**
- Modify: `orchard/hwtests.py` (docstring, import, append `failed_tests`), `orchard/supervisor.py` (imports, `_run_stage`, `_stage_body`, new `_test_failure`), `orchard/context.py` (`PHASE_TASKS`, `build_messages`)
- Test: `tests/test_hwtests.py`, `tests/test_context.py`, `tests/test_supervisor_stage4.py` (append)

**Interfaces:**
- Consumes: `hardware_test_failure` (main, `orchard/supervisor.py`), `load_plan` and `hw_record_problem` (Tasks 4 and 5).
- Produces: `failed_tests(stage_dir, required) -> dict | None` (`{"configs": [...], "problem": "..."}`); `Supervisor._test_failure(spec, stage_dir) -> dict | None`, used by the "no gate feedback: the hardware test failed" check in `_stage_body` and the "not resuming from a failed hardware test" check in `_run_stage`; `PHASE_TASKS["finish-tests"]`.

Main's `hardware_test_failure` reads a top-level `returncode` from the stage's `test-result.json`. A list's summary has none, so before this task every stage 4 gate failure counted as a failed hardware test, and an optional configuration's failure blocked the continuation that could fix a malformed `result.json`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_context.py`:

```python


def test_the_finish_step_of_a_list_records_a_failed_configuration_and_stops(tmp_path):
    from orchard.stages import WEIGHTS_ONLY_STAGE_4
    run, sp = setup(tmp_path)
    sd = run / "stages" / "4"
    sd.mkdir(parents=True)
    _, user = build_messages(spec=WEIGHTS_ONLY_STAGE_4, phase="finish", run_dir=run, stage_dir=sd,
                             skill_path=sp, refs={}, entries=[], facts={"model": "m"}, resumed=False)
    assert "stages/4/tests/<chips>/output.txt" in user and "pass false" in user
    assert "evidence/hw-test-output.txt" not in user
```

Append to `tests/test_hwtests.py`:

```python


def test_failed_tests_names_each_test_that_must_pass_and_did_not(plan):
    from orchard.hwtests import failed_tests
    tests, _ = plan["read"]()
    assert failed_tests(plan["sd"], (2, 4)) == {"configs": [], "problem": "tests/plan.json could not be read"}
    write_plan(plan["sd"], tests)
    ok = {"returncode": 0, "timed_out": False}
    write_record(plan["sd"], 1, {**ok, "returncode": 4, "chips": ["a"]})
    write_record(plan["sd"], 2, {**ok, "chips": ["a", "b"]})
    write_record(plan["sd"], 4, {**ok, "chips": ["a", "b", "c", "d"]})
    assert failed_tests(plan["sd"], (2, 4)) is None                  # only the optional test failed
    assert failed_tests(plan["sd"], None) == {"configs": [1], "problem": "the 1-chip configuration: its test exited 4"}
    write_record(plan["sd"], 4, {"returncode": None, "timed_out": True, "chips": ["a", "b", "c", "d"]})
    got = failed_tests(plan["sd"], (2, 4))
    assert got["configs"] == [4] and "did not finish before its deadline" in got["problem"]
```

Append to `tests/test_supervisor_stage4.py`:

```python


# ---- a failed test in the list: no gate feedback when a required configuration failed --------------

def feedback_then(stage_phase, before, after, base_overrides):
    """A script that writes `before` for stage_phase until the conversation holds the gate
    feedback, then `after` (turns counted from the feedback). Other steps use base_overrides."""
    from run_fakes import FEEDBACK_HEAD

    def script(request):
        if "tools" in request and where(request) == stage_phase:
            msgs = request["messages"]
            fb = [i for i, m in enumerate(msgs) if m["role"] == "user"
                  and (m.get("content") or "").startswith(FEEDBACK_HEAD)]
            if not fb:
                return bringup(request, overrides={stage_phase: before})
            return bringup(dict(request, messages=msgs[:2] + msgs[fb[-1]:]), overrides={stage_phase: after})
        return bringup(request, overrides=base_overrides)
    return script


def test_a_failed_required_configuration_ends_the_stage_without_gate_feedback(rig):
    rig.args.required_chips = (2, 4)
    rig.script = escalation_aware(4, failing_4_chip_test())      # the finish records 4 chips as failed
    assert rig.run() == EXIT_READY
    first = next(e["seq"] for e in rig.entries() if e["event"] == "escalate" and e["stage"] == 4)
    early = [e["data"] for e in rig.entries() if e["stage"] == 4 and e["seq"] < first and e["event"] == "decision"]
    assert not [d for d in early if d["decision"] == "gate feedback"]
    [nofb] = [d for d in early if d["decision"] == "no gate feedback: the hardware test failed"]
    assert nofb["configs"] == [4] and "its test exited 4" in nofb["problem"]
    assert [d["result"] for d in rig.ends(4)] == ["escalate", "pass"]


def test_a_failed_optional_configuration_still_gets_gate_feedback_for_a_malformed_result(rig):
    rig.args.required_chips = (2, 4)
    one_fails = swap_prepare()
    one_fails["configs/1/serve_and_compare.py"] = "import sys\nsys.exit(4)\n"
    honest = {"result.json": {"configs": [swap_entry(1, ok=False), swap_entry(2), swap_entry(4)]}}
    rig.script = feedback_then((4, "finish"), {"result.json": {"configs": "two and four"}}, honest,
                               {(4, "prepare"): one_fails})
    assert rig.run() == EXIT_READY
    decisions = [d["decision"] for d in stage4(rig, "gate feedback")]
    assert decisions == ["gate feedback"]
    assert [d["result"] for d in rig.ends(4)] == ["pass"]


def until_escalated(stage, overrides):
    """escalation_aware, also counting the escalate entry alone ("stage N escalate"), which is all
    the context shows after a kill between the escalate entry and the stage_end."""
    def script(request):
        if "tools" in request and where(request)[0] == stage:
            ctx = request["messages"][1]["content"]
            if f"stage {stage} escalate" not in ctx and f"stage {stage}: escalate" not in ctx:
                return bringup(request, overrides=overrides)
        return bringup(request)
    return script


def test_a_kill_after_the_escalate_entry_tests_the_escalated_attempt_again(rig):
    from fakes import Crash
    rig.args.required_chips = (2, 4)
    rig.script = until_escalated(4, failing_4_chip_test())
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "escalate" and e["stage"] == 4)
    assert rig.run(pid=200) == EXIT_READY
    assert len(stage4(rig, "not resuming from a failed hardware test")) == 1
    assert [d["result"] for d in rig.ends(4)] == ["pass"]
    assert [d["config"] for d in stage4(rig, "hardware test started")] == [1, 2, 4, 1, 2, 4]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_context.py tests/test_hwtests.py tests/test_supervisor_stage4.py`
Expected: 4 FAIL, 47 passed: `failed_tests` cannot be imported; the finish text still names `hw-test-output.txt`; the "no gate feedback" decision has no `configs`; and the optional failure gets no continuation. `test_a_kill_after_the_escalate_entry_tests_the_escalated_attempt_again` passes already, for the wrong reason (a summary with no `returncode` always counts as failed); the mutation step shows it can fail.

- [ ] **Step 3: Implement**

In `orchard/context.py`, replace:

```python
               "to make the result pass. The next attempt runs a fresh test. Reply with a short "
               "summary and no tool call."),
}

```

with:

```python
               "to make the result pass. The next attempt runs a fresh test. Reply with a short "
               "summary and no tool call."),
    # The finish step of a stage whose spec has `tests`.
    "finish-tests": ("The supervisor ran your hardware tests, one per chip configuration. The list "
                     "and every record are below, and each test's output is in "
                     "stages/{n}/tests/<chips>/output.txt. Write {gate} from that evidence. Do not "
                     "invent a result. For a configuration whose test failed (returncode not 0, or "
                     "timed_out true), write its entry with pass false, copy the failure text from "
                     "its output.txt into `reason`, and set every number you did not measure to "
                     "null. Do not investigate a failure and do not try to make it pass. Reply with "
                     "a short summary and no tool call."),
}

```

In `orchard/context.py`, replace:

```python
        system += [f"- {name}: {path or 'not installed on this machine'}" for name, path in refs.items()]

    task = PHASE_TASKS["prepare-tests" if phase == "prepare" and spec.tests else phase]
    user = ["## Task", task.format(gate=spec.gate_file, n=n), "", "## Run"]
    user += [f"- {k}: {v}" for k, v in facts.items()]
```

with:

```python
        system += [f"- {name}: {path or 'not installed on this machine'}" for name, path in refs.items()]

    task = PHASE_TASKS[f"{phase}-tests" if spec.tests and phase in ("prepare", "finish") else phase]
    user = ["## Task", task.format(gate=spec.gate_file, n=n), "", "## Run"]
    user += [f"- {k}: {v}" for k, v in facts.items()]
```

In `orchard/hwtests.py`, replace:

```python
supervisor wrote it, so a record the agent wrote or changed fails the stage.

A test that did not exit 0 may have stopped while it was converting weights into its tensor cache,
and a part-written cache is read back without an error. `suspect_caches` finds those caches in the
```

with:

```python
supervisor wrote it, so a record the agent wrote or changed fails the stage.

`failed_tests` says whether a test that must pass did not: every listed test when the run names
no required chip counts (as `gate_mesh` reads them), else each required count's test. The
supervisor gives no gate feedback after such a failure, and an escalated attempt that a kill
interrupted does not resume from it (orchard/supervisor.py, `_test_failure`).

A test that did not exit 0 may have stopped while it was converting weights into its tensor cache,
and a part-written cache is read back without an error. `suspect_caches` finds those caches in the
```

In `orchard/hwtests.py`, replace:

```python
from orchard.defaults import CHIPS_PER_BOARD
from orchard.handoff import NOTE_KEYS
from orchard.stages import evidence_record, hw_record_path

SCRIPTS = {"serve_and_compare.py": "bundle", "serve_and_compare_container.py": "container"}
```

with:

```python
from orchard.defaults import CHIPS_PER_BOARD
from orchard.handoff import NOTE_KEYS
from orchard.stages import evidence_record, hw_record_path, hw_record_problem

SCRIPTS = {"serve_and_compare.py": "bundle", "serve_and_compare_container.py": "container"}
```

Append to `orchard/hwtests.py`:

```python


def failed_tests(stage_dir, required) -> dict | None:
    """None when every test that must pass has a record showing exit 0 on its chips with no
    timeout. Otherwise {"configs": [chip counts], "problem": one sentence per failed test}. With
    no required counts every listed test must pass. A missing plan counts as a failure."""
    tests = load_plan(stage_dir)
    if tests is None:
        return {"configs": [], "problem": "tests/plan.json could not be read"}
    bad = [(t.chips, hw_record_problem(stage_dir, t.chips)) for t in tests
           if not required or t.chips in required]
    bad = [(n, p) for n, p in bad if p]
    if not bad:
        return None
    return {"configs": [n for n, _ in bad],
            "problem": "; ".join(f"the {n}-chip configuration: {p}" for n, p in bad)}
```

In `orchard/supervisor.py`, replace:

```python
from orchard.handoff import (NOTE_KEYS, Blocked, Budgets, Handoff, decide_park, progress, reacquire,
                             recover, wait_stopped)
from orchard.hwtests import (load_plan, move_aside, pending, read_plan, suspect_caches, unrecorded,
                             write_plan, write_record, write_summary)
from orchard.ledger import Ledger, LedgerCorrupt, LedgerLocked
from orchard.runner import Denied, check_string
```

with:

```python
from orchard.handoff import (NOTE_KEYS, Blocked, Budgets, Handoff, decide_park, progress, reacquire,
                             recover, wait_stopped)
from orchard.hwtests import (failed_tests, load_plan, move_aside, pending, read_plan, suspect_caches,
                             unrecorded, write_plan, write_record, write_summary)
from orchard.ledger import Ledger, LedgerCorrupt, LedgerLocked
from orchard.runner import Denied, check_string
```

In `orchard/supervisor.py`, replace:

```python
                           f"{free} GB is free", need_gb=spec.disk_gb, free_gb=free)
        marker = self.run_dir / "stages" / str(n) / "test-result.json"
        if resuming and escalated and spec.boards and marker.is_file() and hardware_test_failure(marker.parent):
            # A kill came after the escalate entry and before the stage_end. The escalated attempt
            # must run a fresh prepare and hardware test: its finish step cannot pass on a test
```

with:

```python
                           f"{free} GB is free", need_gb=spec.disk_gb, free_gb=free)
        marker = self.run_dir / "stages" / str(n) / "test-result.json"
        if resuming and escalated and spec.boards and marker.is_file() and self._test_failure(spec, marker.parent):
            # A kill came after the escalate entry and before the stage_end. The escalated attempt
            # must run a fresh prepare and hardware test: its finish step cannot pass on a test
```

In `orchard/supervisor.py`, replace:

```python
                return out.status, [f"the finish step ended: {out.status} {out.detail}".strip()], None
        gate = self._check_gate(spec, stage_dir)
        if not gate.ok and spec.boards and (failed := hardware_test_failure(stage_dir)):
            # The finish step was told to record a failed test honestly. Its result cannot pass
            # the gate, and a continuation would ask the agent to make it pass. So the stage ends
```

with:

```python
                return out.status, [f"the finish step ended: {out.status} {out.detail}".strip()], None
        gate = self._check_gate(spec, stage_dir)
        if not gate.ok and spec.boards and (failed := self._test_failure(spec, stage_dir)):
            # The finish step was told to record a failed test honestly. Its result cannot pass
            # the gate, and a continuation would ask the agent to make it pass. So the stage ends
```

In `orchard/supervisor.py`, replace:

```python
            gate = self._check_gate(spec, stage_dir)
        return ("pass", [], gate) if gate.ok else ("fail", list(gate.reasons), gate)

    def _check_gate(self, spec, stage_dir: Path):
```

with:

```python
            gate = self._check_gate(spec, stage_dir)
        return ("pass", [], gate) if gate.ok else ("fail", list(gate.reasons), gate)

    def _test_failure(self, spec, stage_dir: Path) -> dict | None:
        """Why the stage's hardware testing failed, or None. One test: its test-result.json. A list
        of tests: the records of the tests that must pass (`failed_tests`), so an optional
        configuration that failed still leaves the gate feedback for a malformed result file."""
        if spec.tests:
            return failed_tests(stage_dir, self.required_chips)
        return hardware_test_failure(stage_dir)

    def _check_gate(self, spec, stage_dir: Path):
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_context.py tests/test_hwtests.py tests/test_supervisor_stage4.py`
Expected: 51 passed.

- [ ] **Step 5: Mutations**

Fail, restore, delete the bytecode, pass:
1. In `_test_failure`, delete the two lines `if spec.tests:` and `return failed_tests(stage_dir, self.required_chips)`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_stage4.py -k "optional_configuration_still or required_configuration_ends"`: 2 failed.
2. In `failed_tests`, delete the condition `if not required or t.chips in required` (check every test). Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_stage4.py tests/test_hwtests.py -k "optional_configuration_still or failed_tests"`: 2 failed.
3. In `_run_stage`, change the condition of the "not resuming from a failed hardware test" check to `if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_stage4.py -k escalate_entry`: 1 failed.
4. In `build_messages`, change `PHASE_TASKS[f"{phase}-tests" if spec.tests and phase in ("prepare", "finish") else phase]` to `PHASE_TASKS["prepare-tests" if phase == "prepare" and spec.tests else phase]`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_context.py -k list_records`: 1 failed.

- [ ] **Step 6: Run the suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -1` (expected `1416 passed, 1 skipped`), then:

```bash
git add orchard/hwtests.py orchard/supervisor.py orchard/context.py tests/test_hwtests.py tests/test_context.py tests/test_supervisor_stage4.py
git commit -m "Stage 4: gate feedback and resume follow the required configurations' records"
```

---

### Task 11: Runbook entry, README and log

**Files:**
- Modify: `docs/runbooks/hardware-validation.md` (two lines of the Hemmingway-1 entry, a new section at the end), `README.md` (the Supervisor item), `CLAUDE.md` (append a log entry)

**Interfaces:**
- Consumes: everything above.
- Produces: the operator's instructions for stage 4 on the weights-only path.

- [ ] **Step 1: Write the documents**

Append to `CLAUDE.md`:

```markdown

## 2026-10-03: stage 4 tests each chip configuration on the weights-only path
Prompt: stage 4 must show the new model working on the configurations the packages ship for: 2 and 4 chips required, 1
optional (plan `docs/superpowers/plans/2026-10-03-multichip-stage-4.md`). Decisions:
- One hardware test per configuration, listed by the prepare step in `hw_tests.json` and validated by
  `orchard/hwtests.py`; the supervisor builds each command, runs the list in order of chip count, each under its own
  lease, and records each test in `tests/<N>/test-result.json`. `tests/plan.json` is the resume marker. The gate
  (`gate_mesh_swap`) counts a pass only with that record, and the supervisor refuses a record whose sha256 differs from
  the sha256 the ledger holds.
- The 4-chip test (`serve_and_compare_container.py`) runs the printed `docker run` itself with exact-count argv edits:
  fresh tensor cache, empty `/hf`, model directory and blobs mounted read-only at their own paths, `MODEL_WEIGHTS_DIR`
  and `HF_MODEL` set, only the leased device nodes, a run label. `tt-model serve` takes no extra docker arguments.
- A test that needs the coder's boards leases any further board first, then parks; a failed test still restores the
  coder. A container left by a test is stopped by label; one that survives blocks before any release. A cache whose last
  test did not exit 0 is moved aside before its next test.
- After a failed hardware test there is no gate feedback (main, `no-feedback-after-failed-test`). For a list, "failed"
  means a test that must pass did not: a required configuration's, or any listed one when no counts are required. An
  optional configuration that failed still leaves the one continuation for a malformed `result.json`.
Not measured: the 4-chip cold conversion time, the 4-chip coder's restart, the 1-chip cache size.
```

In `README.md`, replace:

```markdown
  a qwen-code transcript into that stream, and the detector thresholds come from a replay of the
  recorded loop.
- **Supervisor.** The supervisor (`python3 -m orchard.supervisor run ...`) runs stages 0 to 6 and 8 of a weights-only bring-up with local models, parks the coder for hardware stages, records every step in the ledger, and stops at ready for operator review. It never publishes. The path stage 0 writes in `delta.json` decides stages 2 and 3. On `weights-only`, stage 2 serves the new weights with the nearest model's existing TT implementation and compares the chip's tokens with the CPU reference (skill `weights-swap-check`, gate `gate_weights_swap`), and stage 3 is recorded as skipped. The agent copies two tested scripts from `orchard/skills/weights-swap-templates/` and fills in one config file; it does not write a serving script. The scripts point the server at the new weights with `MODEL_WEIGHTS_DIR` and `HF_MODEL`; without them the bundle serves the nearest model's weights. The gate needs `top1_agreement` of at least 0.85: the nearest model's weights standing in agreed on 25 of 32 tokens (0.78), the new weights on 30 of 32 (0.94), one prompt each. On `full-port`, or a path the supervisor cannot read, stage 2 runs `functional-decoder` and stage 3 runs as before.

## Try it
```

with:

```markdown
  a qwen-code transcript into that stream, and the detector thresholds come from a replay of the
  recorded loop.
- **Supervisor.** The supervisor (`python3 -m orchard.supervisor run ...`) runs stages 0 to 6 and 8 of a weights-only bring-up with local models, parks the coder for hardware stages, records every step in the ledger, and stops at ready for operator review. It never publishes. The path stage 0 writes in `delta.json` decides stages 2 and 3. On `weights-only`, stage 2 serves the new weights with the nearest model's existing TT implementation and compares the chip's tokens with the CPU reference (skill `weights-swap-check`, gate `gate_weights_swap`), and stage 3 is recorded as skipped. The agent copies two tested scripts from `orchard/skills/weights-swap-templates/` and fills in one config file; it does not write a serving script. The scripts point the server at the new weights with `MODEL_WEIGHTS_DIR` and `HF_MODEL`; without them the bundle serves the nearest model's weights. The gate needs `top1_agreement` of at least 0.85: the nearest model's weights standing in agreed on 25 of 32 tokens (0.78), the new weights on 30 of 32 (0.94), one prompt each. Stage 4 on `weights-only` serves the new weights on each chip configuration the packages ship for (skill `weights-swap-configs`, gate `gate_mesh_swap`): one hardware test per configuration, run by the supervisor in order of chip count, each under its own lease, parking the coder only when its boards are needed. The 4-chip test runs the container package's docker command, edited to use a fresh tensor cache, the model directory and no access to the base weights. A configuration passes only with the supervisor's own record of its test. On `full-port`, or a path the supervisor cannot read, stage 2 runs `functional-decoder` and stage 3 runs as before.

## Try it
```

In `docs/runbooks/hardware-validation.md`, replace:

```markdown
- Nothing listens on port 8000: `ss -ltn "( sport = :8000 )"` prints only its header.
- The run directory is on `/mnt/bonus` (404 GB free on 2026-10-02). The root disk is 99% full,
  and stages 2 to 6 each require 40 GB free (stage 4 requires 80 GB).
- Do not give the harness the reference answer, the nearest supported model or the base model's
  tensor cache. The nearest model (Qwen3.8-27B) and its revision are lines of the reference
```

with:

```markdown
- Nothing listens on port 8000: `ss -ltn "( sport = :8000 )"` prints only its header.
- The run directory is on `/mnt/bonus` (404 GB free on 2026-10-02). The root disk is 99% full,
  and stages 2 to 6 each require 40 GB free (stage 4 requires 80 GB, or 110 GB on the weights-only
  path, plus 40 GB again before each of its tests).
- Do not give the harness the reference answer, the nearest supported model or the base model's
  tensor cache. The nearest model (Qwen3.8-27B) and its revision are lines of the reference
```

In `docs/runbooks/hardware-validation.md`, replace:

```markdown
  `HF_HOME`: check that `/mnt/bonus/models/hemmingway-1/hf` holds no `token` or `stored_tokens`
  file. `HF_TOKEN_PATH` keeps huggingface_hub away from a token there, and it does not stop `cat`.
- Stage 4 needs the `mesh-shrink` skill, which is untracked in the skills repo
  (`plugins/tt-model-bringup/skills/mesh-shrink/`). Do not switch branch or clean that tree
  during the run, or stage 4 blocks with "skill not found".

What stops an agent and what does not. Agent shells run as the same user as the supervisor. The
```

with:

```markdown
  `HF_HOME`: check that `/mnt/bonus/models/hemmingway-1/hf` holds no `token` or `stored_tokens`
  file. `HF_TOKEN_PATH` keeps huggingface_hub away from a token there, and it does not stop `cat`.
- On the full-port path stage 4 needs the `mesh-shrink` skill, which is untracked in the skills
  repo (`plugins/tt-model-bringup/skills/mesh-shrink/`). Do not switch branch or clean that tree
  during the run, or stage 4 blocks with "skill not found". On the weights-only path stage 4 uses
  the local `weights-swap-configs` skill (see "Stage 4 on the weights-only path" below).

What stops an agent and what does not. Agent shells run as the same user as the supervisor. The
```

Append to `docs/runbooks/hardware-validation.md`:

```markdown

## Stage 4 on the weights-only path (2 and 4 chips required, 1 chip optional)

Purpose: show the new weights working on each chip configuration the packages will ship for. Run
with `--required-chips 2,4`. The 1-chip configuration is optional and is recorded as supported or
not. Nothing here was run on hardware when this was written; the first harness run is the test.

Configurations on this box, each with its own tensor cache under
`/mnt/bonus/models/orchard-runs/cache/<model slug>/<N>chip-<package>/tt_cache`:

| chips | package | kind | context | template |
|---|---|---|---|---|
| 1 | `episod/qwen3.8-27b-dflash2-p150` | bundle | 16K | `serve_and_compare.py` |
| 2 | `episod/qwen3.8-27b-dflash2-p300` | bundle | 262K | `serve_and_compare.py` |
| 4 | `changh95/qwen3.8-27b-p300x2`, profile `batch32` | container | 262K | `serve_and_compare_container.py` |

Before running:
- `tt-model list` shows all three installed, and `docker image ls` lists
  `tt-model/qwen3.8-27b-p300x2:0becf4834925`.
- Optional, read-only check of what the image does with the weights directory (starts a container
  with no device):

      docker run --rm --entrypoint grep tt-model/qwen3.8-27b-p300x2:0becf4834925 \
        -n -e MODEL_WEIGHTS_DIR -e HF_MODEL /opt/tt-metal/models/demos/blackhole/qwen36/tt/qwen36_vllm.py

  The source build on this box resolves the weights directory as `MODEL_WEIGHTS_DIR`, then
  `HF_MODEL`, then the config path. The image was built from another tt-metal commit, which is not
  on this box. The template sets both variables, so either order serves the model directory.
- `/mnt/bonus` has at least 110 GB free: three caches of about 34 GB (34 GB measured for 2 chips,
  31 GB for 4 chips, 1 chip not measured).

What happens: the prepare step writes one directory per configuration (`stages/4/configs/<N>/`)
and `hw_tests.json`. The supervisor refuses the list before any lease if a required count has no
test, two configurations share a cache, a cache is inside `~/.cache/tt-model` or
`~/.cache/qwen36-src-build`, or the deadlines add up to more than the stage budget. It then runs
the tests in order of chip count:
- With the coder on 2 chips (board 0): the 1- and 2-chip tests lease board 1 and leave the coder
  up. The 4-chip test first leases board 1, then parks the coder (stand-in canary, `tt-model stop`,
  reset of board 0), runs on all four chips, releases board 1 and restores the coder. Measured:
  `tt-model stop` 1 to 2 s, each reset 41.7 s, the 2-chip coder back to ready in about 120 s.
- With the coder on 4 chips: every test parks the coder. Its restart time was not measured.

The 4-chip test asks `tt-model serve changh95/qwen3.8-27b-p300x2 --print` for the docker command,
edits it and runs it itself, because `tt-model serve` takes no extra docker arguments. The edited
command is saved as `stages/4/configs/4/evidence/docker-argv.json`. The first boot converts the
weights into the new cache. That took about 5 minutes for 2 chips here; for 4 chips it was not
measured. A bundle with a cold kernel compile cache also compiles every kernel first (more than 26
minutes on 2026-10-03): the 1-chip bundle compiles its own, the 2-chip bundle reuses stage 2's, and
the container keeps the package's kernel cache. Each test's deadline is 3,600 s.

Check after the stage, for the 4-chip configuration:
- In `docker-argv.json`: the `/tensor-cache` volume's source is the configuration's own `tt_cache`;
  the `/hf` volume's source is `stages/4/configs/4/hf-isolated`; `MODEL_WEIGHTS_DIR` and
  `HF_MODEL` name `stages/4/configs/4/model-dir`; no volume source is `~/.cache/huggingface`; the
  four `--device` entries match the four chips in `tests/4/test-result.json`.
- `stages/4/configs/4/evidence/server.log` shows the weights loaded from the model directory.
- `top1_agreement` in `stages/4/configs/4/evidence/swap-check.json` is at least 0.85. The base
  weights standing in agreed 25 of 32 (0.78) on 2 chips.
- `docker ps -a --filter label=orchard.test` lists nothing.

If the 4-chip test exits 4 and `server.log` shows the runtime looking up `Qwen/Qwen3.8-27B` in
an empty hub cache, the container needs something from the operator's Hugging Face cache that the
template did not expect. Record it. Do not remove the `/hf` isolation to get a pass: without it a
runtime that ignores the weights variables serves the base model with no error. Bring the log to
the operator.

Stop conditions for this stage, in addition to the run's: a `blocked` notice that names test
containers still running, a cache moved aside that you did not expect, or a 4-chip
`top1_agreement` under 0.85.
```

- [ ] **Step 2: Check the writing rules**

Run: `git diff -U0 -- docs README.md CLAUDE.md | grep '^+' | grep -n -e ', not ' -e 'rather than' -e 'instead of' -e 'reads as' -e 'honest' -e 'the one' -e 'which is why' -e 'This is what'`
Expected: no output, or only lines whose wording is justified (none were left when this plan was checked).

- [ ] **Step 3: Run the whole suite**

Run: `find . -name __pycache__ -prune -exec rm -rf {} + && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -1`
Expected: `1416 passed, 1 skipped` (plus whatever later commits on main added).

- [ ] **Step 4: Commit**

```bash
git add docs/runbooks/hardware-validation.md README.md CLAUDE.md
git commit -m "Docs: stage 4 on the weights-only path, runbook entry and log"
```

---

## Hand-off notes

- Nothing in this plan has run on hardware. The first harness run of stage 4 is the test of the container template's edits against the real `tt-model serve --print` output, of the 4-chip conversion time, and of whether the image needs anything from the operator's Hugging Face cache. The runbook entry says what to check and what to do if the isolated `/hf` breaks the load.
- The single-test path (stages 2, 5 and 6) has the same weaknesses this plan closes for stage 4: an agent can write `test-result.json` itself, and a timed-out test's cache is reused. They were left unchanged.
- Main's finish-step text for one test names `evidence/hw-test-output.txt`. A stage with a list gets its own finish text (Task 10), which names `tests/<chips>/output.txt`.
