# tt-orchard: stage 7 packages a weights-only model (plan 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From a weights-only model the harness has brought up, stage 7 stages a v6 thin tt-model package for each hardware profile we ship that has an installed source bundle (2 chips required; 1 and 4 chips optional), boots an installed copy of the 2-chip package on a leased board, compares it with the stage 1 reference, and hands the operator a card, a record and the publish commands as text. Nothing is published. The first target is `Altworld/Hemmingway-1` (CC BY-NC 4.0), a fine-tune of Qwen3.8-27B.

**Architecture:** Stage 7 becomes supervisor code with no agent and no model (`StageSpec.harness`). It runs only on the weights-only path of a run started with `--package-format v6 --package-namespace <ns>`; every other run skips it as before. `orchard/package.py` reads stages 0, 2 and 4, finds the nearest model's installed v6 bundles, runs `tt-model package-thin --out` (never with a repo id) with `--weights` naming the new model, and edits each staged `run.sh` so that a shipped `prepare_model_dir.py` builds `model-dir/` (the nearest model's config files, the new model's tokenizer and weights) and `--model`, `HF_MODEL` and `MODEL_WEIGHTS_DIR` all name it. Each package is scrubbed (`orchard/scrub.py`, `scrub_package`), carries a card with its license and labelled numbers (`orchard/package_card.py`), and the required profile's copy is installed and served by `orchard/package_templates/verify_bundle.py` as the stage's hardware test, under the lease and park machinery stages 2 to 6 already use. `gate_package` (`orchard/stages.py`) checks everything again from disk. The supervisor copies the record, the cards and `PUBLISH_COMMANDS.txt` into the operator bundle.

**Tech Stack:** Python 3.12 standard library, pytest. Tests run a fake `tt-model` on PATH (`tests/fake_package_thin.py`, which records its argv and writes a minimal bundle), fake installed bundles and Hugging Face caches (`tests/package_fakes.py`), and the existing fake vLLM server (`tests/fake_swap_server.py`). `verify_bundle.py` reuses `serve_and_compare.py`, which needs the `tokenizers` package; the tests that run it skip with a message when it is missing. On this machine it is installed in the interpreter that runs the suite.

**Spec:** `docs/superpowers/specs/2026-10-01-orchard-design.md`, sections 5 (stage 7), 8 (leases), 10 (failure handling, denials, scrub) and 11 (skills). Plan 4 (`docs/superpowers/plans/2026-10-02-stage-machine-and-agents.md`) built the stage machine; the weights-only path and the swap templates came after it (`CLAUDE.md`, 2026-10-03 entries). Facts this plan rests on, each from a source read for it: the hand-made packages in `~/code/qwen3.8-27b-dflash2-2chip-p300c/thin/` (`stage.sh`, `stage2.sh`, `splice_extra_args.py`, `scrub_manifest.py`, `push_bundle.py`, the staged bundle `A/`); `tt-model package-thin --help`, `tt-model package --help`, `tt-model push --help` (read 2026-10-03; `package-thin` and `package` push when given a repo id, `push` publishes v5.1 container packages only); `~/tt-model-manager-prs/docs/publishing.md`, `thin_packages.md` and `container_packages.md` (v5.1 always builds an image; repo tags live in the card's front matter); the run log `docs/run-logs/2026-10-hemmingway-1.md` (the `MODEL_WEIGHTS_DIR` correction, the architecture-registration finding, the license); and the operator's notes on the Qwen3.8 packages (`package-thin` has no flag for fixed vLLM arguments; the build host's name is in the manifest; `install.sh` is not executable; `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0`; tensor caches are keyed by layer name only).

**Checked before hand-over:** every code block below was extracted from this document's source by a script, applied task by task in document order to a fresh clone of the repo (branch `main` at `1407417`), and run on 2026-10-03: 1454 passed and 1 skipped (1326 passed before, so 128 new tests). Each task's tests failed before its implementation and passed after it, with the outcome shown in Step 2 and Step 4. Every mutation step was run on that copy; each named test failed with the mutation and passed after the restore. The whole suite ran after every task. Nothing ran on hardware. The implementer still runs every step; this check does not replace them.

## Global Constraints

- Work in `/home/ttuser/code/tt-orchard`. Task 0 creates the branch `packaging-stage7` from `main`; every later task commits on it. The repo is local only. Never run `git push` and never create a remote.
- Python 3.12, standard library only in `orchard/`. Every new module starts with `from __future__ import annotations` and a docstring that says what the module owns. Comments explain why. `verify_bundle.py` runs as a hardware test under the supervisor's own interpreter and imports `tokenizers`, as `serve_and_compare.py` already does.
- The unit suite uses no hardware, no network and no real model. No test opens `/dev/tenstorrent/*` or runs `tt-smi`, the real `tt-model`, `docker`, `gozer`, `hf` or a model server other than the fakes. The `stub_tools` fixture keeps the real tools off PATH; stage 7 tests put the fake `tt-model` first.
- The run never publishes. Nothing in this plan runs `tt-model push`, `tt-model publish`, `tt-model package` or `package-thin` with a repo id, `hf upload`, `git push` or the Hugging Face hub API. `orchard/package.py` runs `package-thin` only through `assert_no_publish`, and a test shows its only external calls are `package-thin ... --out` and the copy's `install.sh`. The publish commands are text in `PUBLISH_COMMANDS.txt`.
- Do not run the supervisor against hardware, and do not run any gozer command that changes state on the real box. The controller runs the runbook entry (Task 11) with the operator's agreement.
- Every default value lives in `orchard/defaults.py` as one named constant, with a comment that cites where it was measured or says that it is a choice and was not measured.
- Every guard gets a mutation step: change the guard, run the named test, watch it fail, restore the guard (spec section 13). After each restore, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`) before the confirming run.
- Run the whole suite before each commit: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider`. The baseline is 1326 passed and 1 skipped in about 245 s. After this plan: 1454 passed and 1 skipped in about 270 s.
- Change existing modules only where a task says so: `orchard/defaults.py`, `orchard/runner.py`, `orchard/scrub.py` (append), `orchard/stages.py`, `orchard/supervisor.py`, `orchard/skills/operator-bundle.md`, `tests/fake_swap_server.py`, the test files named in each task, and the documents in Task 11.
- Out of scope: building a v5.1 container image; publishing of any kind; running stage 7 on hardware; building a 4-chip v6 source bundle; changes to tt-model-manager, tt-gozer or tenstorrent/skills; a stage 7 agent or skill.
- Commit messages are plain English, one idea per commit. Writing rules for docs, comments and messages: state the finding; short sentences; no "X, not Y" framing; no aphoristic closers; no metaphors that stand in for a claim.

## Review Focus

1. The staged `run.sh` loads the base weights. The TT runtime takes its weights directory from `MODEL_WEIGHTS_DIR`, then `HF_MODEL`, then the config path, and the nearest model's id in `HF_MODEL` serves the nearest model's weights with no error. Expected: every weights setting names the bundle's `model-dir`, the boot check refuses unless the server process's own environment says so, its Hugging Face home holds no snapshot of the nearest model, and the gate checks `run.sh` again (Task 5: `test_wiring_points_every_weights_setting_at_the_model_dir`, `test_a_script_that_names_the_base_model_has_a_wiring_problem`; Task 7: `test_a_copy_whose_run_sh_lost_the_weights_directory_is_caught`, `test_the_operators_own_weights_variables_cannot_stand_in_for_the_bundles`, `test_the_boot_check_hf_home_holds_the_new_model_and_the_drafter_only`; Task 9: `test_a_run_sh_edited_back_to_the_base_weights_fails`).
2. A cache, token or hostname leaks into a package. Expected: the build host's name is replaced in the manifest, a scrub hit stops the stage before anything is installed, the staged directory is never installed into or served, and the gate scrubs again with this machine's hostname (Task 3: `test_caches_weights_and_install_output_are_hits`, `test_the_hostname_tokens_and_home_paths_are_found_in_text_files`; Task 6: `test_a_staged_profile_is_wired_scrubbed_and_built_by_one_package_thin_call`; Task 7: `test_the_installed_copy_serves_the_new_weights_and_agrees`; Task 8: `test_a_scrub_hit_stops_the_stage_before_anything_is_installed`; Task 9: `test_a_hostname_or_a_cache_in_the_staged_package_fails`).
3. The card claims a number that was not measured. Expected: numbers appear only in the card's table; a measured row names evidence inside the run directory and the first file holds the value; a performance number in the prose is refused; an optional profile's card borrows no number from another chip count (Task 4: `test_a_measured_number_needs_its_evidence_and_its_value_in_the_first_file`, `test_a_number_in_the_prose_is_refused`, `test_a_row_labelled_measured_by_hand_without_a_value_is_refused`; Task 8: `test_finish_records_a_passing_boot_check_on_the_required_profile_only`).
4. The license is lost, or the card implies commercial use of a non-commercial model. Expected: the card's front matter carries the license read from the new model's own card, a non-commercial card says "Non-commercial use only.", commercial wording is refused, an unknown license counts as non-commercial, and the gate fails when the license cannot be read (Task 4: `test_a_card_that_implies_commercial_use_of_a_non_commercial_model_is_refused`, `test_a_card_that_lost_its_license_is_refused`, `test_non_commercial_licenses_are_recognised`; Task 9: `test_a_card_whose_license_was_changed_fails`, `test_a_model_whose_license_cannot_be_read_fails`).
5. The boot check passes against the wrong server. Expected: it refuses a port that already answers, a server whose `/v1/models` does not list the copy's model-dir, a server process whose environment names another weights directory, and a model-dir built from other weights; the gate refuses a verify record whose server environment does not match (Task 7: `test_a_port_that_already_answers_is_refused_before_any_server_starts`, `test_a_server_that_serves_another_model_is_refused`, `test_a_model_dir_built_from_other_weights_is_refused`; Task 9: `test_a_boot_check_against_another_server_fails`).

## Decisions this plan makes

The brief left these open. Each is a choice, written here so a reviewer can reject it.

- **Stage 7 is opt-in.** It packages only when the run was started with `--package-format v6` (or was given it on a resume before stage 7 started). A run without the option skips stage 7 as plan 4 did, so the existing end-to-end tests and the Hemmingway-1 run in progress keep working.
- **v5.1 is refused at start.** `tt-model package --container` always builds an image (1.5 to 2.5 h cold, measured on the Audio8 build), and no existing image serves the new weights without a changed launch. `--package-format v5.1` exits 2 with that reason (`PACKAGE_DEFERRED`).
- **The weights pointer is the new model, and `run.sh` builds a model-dir.** `--weights Altworld/Hemmingway-1` alone fails: its config names `Qwen3_5ForCausalLM`, which the bundle's model class does not serve (run log, prototype finding 1). The package ships the nearest model's config files in `base_config/` and the stage 2 recipe as `prepare_model_dir.py`.
- **Profiles come from installed v6 bundles of the nearest model.** Stage 2's bundle gives the required profile (2 chips). Another bundle gives an optional profile when stage 4 passed its chip count. On this machine the 4-chip Qwen3.8 packages are v5.1 containers, so the 4-chip profile is recorded as skipped with that reason, and the 1-chip `episod/qwen3.8-27b-dflash2-p150` gives the optional 1-chip profile.
- **Only the required profile is boot-checked.** Stage 7 needs one board. An optional profile is staged and carded, its numbers are TODO, and its publish line is commented out with "NOT boot-checked".
- **The boot check serves an installed copy.** The staged directory stays clean. The copy converts a fresh tensor cache inside the run directory, so no cache is reused.
- **Publish commands are private `hf upload` lines.** `tt-model push` handles v5.1 only, and `tt-model package-thin <repo>` would rebuild the bundle without the `run.sh` edits. The card's front matter carries the license and the tags that tt-model's own push adds.
- **Stage 7 failures pause and are not escalated.** No model ran, so another model cannot fix it.
- **Agents may not run `tt-model package` or `package-thin` at all.**
- **Wheels are not searched.** Each must be byte-identical to the source bundle's wheel, which the operator has already shipped.
- **package-thin and install.sh run with the agent shells' environment**: no tokens, HOME inside the run directory.

## File Structure

| File | Responsibility |
|---|---|
| `orchard/defaults.py` | stage 7's budget and disk; package formats, the deferred format, timeouts, the tt-model install root |
| `orchard/runner.py` | the `tt-model-package` refusal |
| `orchard/scrub.py` | `scrub_package`: the stricter scrub for a staged package |
| `orchard/package_card.py` | the card (README.md), the license, the number and commercial-wording checks |
| `orchard/package.py` | stage 7's work: run facts, source bundles, profiles, `package-thin --out`, `run.sh` edits, the boot check's copy, cards, `package.json`, publish commands |
| `orchard/package_templates/prepare_model_dir.py` | shipped in each package; builds `model-dir/` before vLLM starts |
| `orchard/package_templates/verify_bundle.py` | the boot check, run by the supervisor as stage 7's hardware test |
| `orchard/stages.py` | `StageSpec.harness`, `PACKAGE_STAGE_7`, `gate_package`, the package options in the ledger |
| `orchard/supervisor.py` | the run options, `_package_body`, no escalation for stage 7, the copy into the operator bundle |
| `orchard/skills/operator-bundle.md` | stage 8 copies stage 7's publish commands word for word |
| `tests/fake_package_thin.py`, `tests/package_fakes.py` | the fake `tt-model`; fake installed bundles, runs and HF caches |
| `tests/fake_swap_server.py` | gains `/v1/models` and takes the model from `--model` |
| `tests/test_package_*.py`, `tests/test_gate_package.py`, `tests/test_supervisor_package.py` | one test file per part |
| `README.md`, `CLAUDE.md`, the spec, `docs/runbooks/hardware-validation.md` | what was built, and the packaging runbook entry |

---

### Task 0: Branch from main

**Files:**
- None changed.

**Interfaces:**
- Consumes: `main` at `1407417` (`Merge branch 'no-feedback-after-failed-test'`) or later.
- Produces: the branch `packaging-stage7`.

- [ ] **Step 1: Confirm a clean tree on main**

Run: `cd /home/ttuser/code/tt-orchard && git status --short --branch && git log --oneline -1`
Expected: `## main`, and at most one untracked file, this plan (`?? docs/superpowers/plans/2026-10-03-packaging-stage-7.md`); the last commit is `1407417 Merge branch 'no-feedback-after-failed-test'` or later. `orchard/skills/weights-swap-templates/serve_and_compare.py` must exist (Task 7 copies it). A later commit can change the interfaces this plan uses: if `gate_weights_swap`, `spec_for`, `Supervisor._hardware_phase`, `_read_test`, `_stage_body`, `_end_stage`, `agent_env` or `tests/fake_swap_server.py` differ from what Tasks 7, 9 and 10 replace, stop and report it. If anything else is modified or untracked, stop and report it.

- [ ] **Step 2: Run the baseline suite**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `1326 passed, 1 skipped`.

- [ ] **Step 3: Branch**

Run: `git switch -c packaging-stage7 && git add docs/superpowers/plans/2026-10-03-packaging-stage-7.md && git commit -m "Plan 5: stage 7 packages a weights-only model" && git status --short --branch`
Expected: `## packaging-stage7` and nothing else. If the plan was already committed, run `git switch -c packaging-stage7` alone.

---

### Task 1: Stage 7 defaults: budget, disk and the package constants

**Files:**
- Modify: `orchard/defaults.py` (stage 7's budget and disk; append the plan 5 block)
- Test: `tests/test_defaults.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces in `orchard.defaults`: `STAGE_BUDGET_S[7] = 14400.0`, `STAGE_DISK_GB[7] = 80.0`, `PACKAGE_FORMATS = ("v6", "v5.1")`, `PACKAGE_DEFERRED: dict[str, str]` (format to reason), `PACKAGE_THIN_TIMEOUT_S`, `PACKAGE_INSTALL_TIMEOUT_S`, `PACKAGE_VERIFY_DEADLINE_S`, `PACKAGE_HEALTH_TIMEOUT_S`, `TT_MODEL_MODELS_ROOT = "~/.cache/tt-model/models"`.

Stage 7 had a budget and a disk need of 0 because plan 4 skipped it. It now installs a copy of the package (a venv built from pip pins, with vLLM compiled from source) and converts a fresh tensor cache for one boot check. The copy starts with cold caches, as stage 2's first boot did, so its health wait is the swap template's 3300 s (weight conversion plus a cold kernel compile of more than 26 min, measured 2026-10-03). The install is not measured on this machine, so its timeout is a named choice with a comment that says so.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_defaults.py`:

```python


def test_the_package_stage_budget_holds_the_install_and_the_boot_check():
    assert (d.PACKAGE_INSTALL_TIMEOUT_S + d.PACKAGE_VERIFY_DEADLINE_S + d.PACKAGE_THIN_TIMEOUT_S
            < d.STAGE_BUDGET_S[7])
    assert d.PACKAGE_HEALTH_TIMEOUT_S + 120 <= d.PACKAGE_VERIFY_DEADLINE_S
    assert d.PACKAGE_VERIFY_DEADLINE_S > d.COLD_BOOT_S


def test_the_boot_check_waits_as_long_as_stage_2s_first_boot():
    # The installed copy starts with cold caches, like stage 2's first boot, so its health wait is
    # at least the swap template's (3300 s: weight conversion plus a cold kernel compile).
    import re
    from pathlib import Path
    script = (Path(d.__file__).with_name("skills") / "weights-swap-templates" / "serve_and_compare.py").read_text()
    health = float(re.search(r"^HEALTH_TIMEOUT_S = ([\d.]+)", script, re.MULTILINE).group(1))
    assert d.PACKAGE_HEALTH_TIMEOUT_S >= health


def test_the_package_stage_disk_need_covers_a_fresh_tensor_cache():
    assert d.STAGE_DISK_GB[7] >= 34 + 40


def test_every_deferred_package_format_is_a_known_format_and_v6_is_not_deferred():
    assert set(d.PACKAGE_DEFERRED) <= set(d.PACKAGE_FORMATS)
    assert "v6" in d.PACKAGE_FORMATS and "v6" not in d.PACKAGE_DEFERRED
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_defaults.py`
Expected: FAIL. 4 tests fail (`4 failed, 15 passed in 0.04s`). The first failure reads `AttributeError: module 'orchard.defaults' has no attribute 'PACKAGE_INSTALL_TIMEOUT_S'`.

- [ ] **Step 3: Implement**

Replace in `orchard/defaults.py`:

```python
# cold boot (COLD_BOOT_S, about 30 min) plus its checks; stage 4 boots two configurations. Stage 7
# is skipped in plan 4, so its budget is 0.
STAGE_BUDGET_S = {0: 7200.0, 1: 14400.0, 2: 14400.0, 3: 21600.0, 4: 28800.0, 5: 10800.0,
                  6: 14400.0, 7: 0.0, 8: 3600.0}
# Free disk each stage needs on the run directory's filesystem before it starts (spec section 10).
# 40 GB covers one converted 2-chip tensor cache: the base Qwen3.8-27B TP=2 cache measured 34 GB
# (2026-09-30). Stage 4 converts a second (1-chip) cache. The other values are choices.
STAGE_DISK_GB = {0: 1.0, 1: 5.0, 2: 40.0, 3: 40.0, 4: 80.0, 5: 40.0, 6: 40.0, 7: 0.0, 8: 1.0}
LONG_STAGE_S = 3600.0           # spec section 10: a stage with a longer budget must declare a resume marker
```

with:

```python
# cold boot (COLD_BOOT_S, about 30 min) plus its checks; stage 4 boots two configurations. Stage 7
# (plan 5) holds the package install (PACKAGE_INSTALL_TIMEOUT_S), one boot check
# (PACKAGE_VERIFY_DEADLINE_S) and a park and restore. When a run builds no package it is skipped.
STAGE_BUDGET_S = {0: 7200.0, 1: 14400.0, 2: 14400.0, 3: 21600.0, 4: 28800.0, 5: 10800.0,
                  6: 14400.0, 7: 14400.0, 8: 3600.0}
# Free disk each stage needs on the run directory's filesystem before it starts (spec section 10).
# 40 GB covers one converted 2-chip tensor cache: the base Qwen3.8-27B TP=2 cache measured 34 GB
# (2026-09-30). Stage 4 converts a second (1-chip) cache. The other values are choices.
# Stage 7 (plan 5) installs a copy of the package (a venv and a uv cache, size not measured) and
# converts a fresh tensor cache for its boot check (34 GB measured for the 2-chip cache).
STAGE_DISK_GB = {0: 1.0, 1: 5.0, 2: 40.0, 3: 40.0, 4: 80.0, 5: 40.0, 6: 40.0, 7: 80.0, 8: 1.0}
LONG_STAGE_S = 3600.0           # spec section 10: a stage with a longer budget must declare a resume marker
```

Replace in `orchard/defaults.py`:

```python
FIRST_BOOT_EXPECTED = "42"      # a first start must answer with text that contains this, or the run blocks
```

with:

```python
FIRST_BOOT_EXPECTED = "42"      # a first start must answer with text that contains this, or the run blocks

# ---- plan 5: stage 7, the package ---------------------------------------------------------------
PACKAGE_FORMATS = ("v6", "v5.1")    # the values --package-format accepts
PACKAGE_DEFERRED = {                # formats that are accepted as names and refused at start
    "v5.1": ("a v5.1 container package needs a new image build (1.5 to 2.5 h cold, measured on the "
             "Audio8 v5.1 build), and tt-model package --container always builds one. Plan 5 "
             "builds v6 thin bundles only"),
}
PACKAGE_THIN_TIMEOUT_S = 1800.0     # choice: `tt-model package-thin --out` copies the wheels and writes
                                    # a few files; not measured
PACKAGE_INSTALL_TIMEOUT_S = 5400.0  # choice: install.sh fetches an interpreter and the pip pins and
                                    # builds vLLM from source; not measured on this machine
PACKAGE_HEALTH_TIMEOUT_S = 3300.0   # how long the boot check waits for /health. The installed copy
                                    # starts with an empty tensor cache and an empty kernel compile
                                    # cache, as stage 2's first boot did: weight conversion about 5 min
                                    # plus a cold compile of more than 26 min (measured 2026-10-03). The
                                    # same value as the swap template's HEALTH_TIMEOUT_S
PACKAGE_VERIFY_DEADLINE_S = 4200.0  # choice: the boot check's hardware deadline: the health wait, 33
                                    # short requests, and the server's stop
TT_MODEL_MODELS_ROOT = "~/.cache/tt-model/models"   # where `tt-model` installs bundles on this machine
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_defaults.py`
Expected: PASS (19 passed).

- [ ] **Step 5: Mutation checks**

1. Stage 7's disk need: in `orchard/defaults.py`, replace `6: 40.0, 7: 80.0, 8: 1.0}` with `6: 40.0, 7: 0.0, 8: 1.0}`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_defaults.py::test_the_package_stage_disk_need_covers_a_fresh_tensor_cache`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
2. The boot check's health wait: in `orchard/defaults.py`, replace `PACKAGE_HEALTH_TIMEOUT_S = 3300.0` with `PACKAGE_HEALTH_TIMEOUT_S = 3000.0`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_defaults.py::test_the_boot_check_waits_as_long_as_stage_2s_first_boot`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: 1330 passed and 1 skipped (the skip is the opt-in replay).

```bash
git add orchard/defaults.py tests/test_defaults.py
git commit -m "Give stage 7 a budget, a disk need and the package constants"
```

---

### Task 2: Agents may not run `tt-model package` or `package-thin`

**Files:**
- Modify: `orchard/runner.py` (one rule, its detail text, the docstring's list)
- Test: `tests/test_runner.py` (one `DENY` entry; append one test)

**Interfaces:**
- Consumes: the runner's `RULES` and `RULE_DETAIL`.
- Produces: the rule `tt-model-package` (denies `tt-model package` and `tt-model package-thin` in any spelling the runner parses).

`tt-model package-thin <repo>` and `tt-model package <repo>` upload when given a repo id, and the runner did not refuse them (`tt-model push` and `publish` were already refused). Stage 7 now packages as supervisor code, so no agent needs either verb. The rule refuses both whatever their arguments, which is simpler than telling a repo id from an option value.

- [ ] **Step 1: Write the failing tests**

Replace in `tests/test_runner.py`:

```python
        "tt-model publish episod/x",
    ],
```

with:

```python
        "tt-model publish episod/x",
    ],
    "tt-model-package": [
        "tt-model package-thin episod/hemmingway-1-p300 --model-py model.py --out x",
        "tt-model package-thin --model-py model.py --out {run}/x",
        "tt-model package episod/x --wheels-dir w",
        "tt-model package --container tt-model.yaml",
        "env A=1 tt-model package-thin --out x",
        "bash -c 'tt-model package-thin --out x'",
    ],
```

Replace in `tests/test_runner.py`:

```python
    assert marker.exists()
```

with:

```python
    assert marker.exists()


def test_a_package_refusal_says_stage_7_packages_and_how_to_proceed(run_dir):
    # `tt-model package` and `package-thin` push when given a repo id, so agents never run them.
    with pytest.raises(Denied) as exc:
        check_string("tt-model package-thin --model-py model.py --out x", run_dir)
    assert exc.value.rule == "tt-model-package"
    assert "stage 7" in str(exc.value) and "To proceed:" in str(exc.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_runner.py`
Expected: FAIL. 8 tests fail (`8 failed, 477 passed in 0.33s`). The first failure reads `Failed: DID NOT RAISE <class 'orchard.runner.Denied'>`.

- [ ] **Step 3: Implement**

Replace in `orchard/runner.py`:

```python
  run, firmware or a reset word; docker with stop, kill, rm, run, start, restart, push, exec, cp
  and similar verbs; tt-model stop, serve, run, rm, unpublish and login; kill, pkill, killall,
  reboot, shutdown and systemctl/service stop or restart; ssh, scp and sftp, and rsync with a
  `host:` or `rsync://` operand; curl with a request body, form, upload or a method other than
```

with:

```python
  run, firmware or a reset word; docker with stop, kill, rm, run, start, restart, push, exec, cp
  and similar verbs; tt-model stop, serve, run, rm, unpublish, login, package and package-thin;
  kill, pkill, killall, reboot, shutdown and systemctl/service stop or restart; ssh, scp and
  sftp, and rsync with a
  `host:` or `rsync://` operand; curl with a request body, form, upload or a method other than
```

Replace in `orchard/runner.py`:

```python

def _git_push(name, argv, ctx):
```

with:

```python

# `tt-model package` and `package-thin` upload when given a repo id, and stage 7 packages the model
# as supervisor code (orchard/package.py), so an agent never needs either verb.
TT_MODEL_PACKAGE = {"package", "package-thin"}


def _tt_model_package(name, argv, ctx):
    return name == "tt-model" and any(a in TT_MODEL_PACKAGE for a in argv[1:])


def _git_push(name, argv, ctx):
```

Replace in `orchard/runner.py`:

```python
    ("tt-model-publish", _tt_model_publish),
    ("git-push", _git_push),
```

with:

```python
    ("tt-model-publish", _tt_model_publish),
    ("tt-model-package", _tt_model_package),
    ("git-push", _git_push),
```

Replace in `orchard/runner.py`:

```python
                      "supervisor starts and stops the coder",
    "tt-model-control": "the supervisor starts and stops model servers, and publishing is for the "
```

with:

```python
                      "supervisor starts and stops the coder",
    "tt-model-package": "stage 7 packages the model as supervisor code, and package or "
                        "package-thin with a repo id uploads. To proceed: write what the package "
                        "needs into your stage's result file; the supervisor stages the package",
    "tt-model-control": "the supervisor starts and stops model servers, and publishing is for the "
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_runner.py`
Expected: PASS (485 passed).

- [ ] **Step 5: Mutation checks**

1. Package-thin is refused: in `orchard/runner.py`, replace `TT_MODEL_PACKAGE = {"package", "package-thin"}` with `TT_MODEL_PACKAGE = {"package"}`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_runner.py::test_a_package_refusal_says_stage_7_packages_and_how_to_proceed`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: 1337 passed and 1 skipped (the skip is the opt-in replay).

```bash
git add orchard/runner.py tests/test_runner.py
git commit -m "Refuse tt-model package and package-thin in agent shells"
```

---

### Task 3: The package scrub

**Files:**
- Modify: `orchard/scrub.py` (append `scrub_package`)
- Test: `tests/test_scrub.py` (import line; append)

**Interfaces:**
- Consumes: `orchard.scrub.scrub_text`.
- Produces in `orchard.scrub`: `FORBIDDEN_DIRS`, `FORBIDDEN_SUFFIXES`, `BINARY_SUFFIXES = (".whl",)`, `NAMESPACE_FILES = frozenset({"README.md"})`, `scrub_package(root, *, hostname: str | None = None, home: str | None = None, namespace: str | None = None) -> list[str]` (each hit as `"relative/path: what was found"`, sorted).

A staged package is uploaded as it is, so it gets a stricter check than the operator bundle. Besides the hostname, tokens and home paths, nothing that installing or serving produces may be in it: a tensor cache (keyed by layer name only, so another model's cache serves that model's weights without an error), weights, a venv, an interpreter, a Hugging Face cache, a built model-dir, or a symbolic link. The operator's namespace may appear only in the card. Wheels are binary and are not read here; Task 6 checks that each is byte-identical to the source bundle's.

- [ ] **Step 1: Write the failing tests**

Replace in `tests/test_scrub.py`:

```python

from orchard.scrub import scrub_bundle, scrub_text

```

with:

```python

from orchard.scrub import scrub_bundle, scrub_package, scrub_text

```

Replace in `tests/test_scrub.py`:

```python
    assert scrub_bundle(tmp_path, hostname=HOST, home=HOME) == ["card.md: the hostname 'quietbox-7'"]
```

with:

```python
    assert scrub_bundle(tmp_path, hostname=HOST, home=HOME) == ["card.md: the hostname 'quietbox-7'"]


# ---- the package scrub (plan 5, stage 7) ---------------------------------------------------------


def package_dir(root):
    """A clean staged thin bundle: text files, a wheel and a card that names the operator's repo."""
    root.mkdir()
    (root / "run.sh").write_text('HERE="$(pwd)"\nexport HF_MODEL="$HERE/model-dir"\n')
    (root / "tt_kernel_manifest.json").write_text('{"producer": {"hostname": "redacted"}}\n')
    (root / "README.md").write_text("Serve it: tt-model serve episod/hemmingway-1-p300\n")
    (root / "wheels").mkdir()
    (root / "wheels" / "ttnn-0.79.0-cp312-cp312-linux_x86_64.whl").write_bytes(
        b"PK\x03\x04 built in /home/alice/tt-metal on quietbox-7")
    return root


def test_a_clean_package_has_no_hits_and_wheel_contents_are_not_read(tmp_path):
    root = package_dir(tmp_path / "pkg")
    assert scrub_package(root, hostname=HOST, home=HOME, namespace="episod") == []


@pytest.mark.parametrize("rel, what", [
    (".tt_cache/layer0.tensorbin", "a tensor cache directory"),
    ("tensors/layer0.bin", "a tensor cache directory"),
    ("venv/bin/python", "an installed venv"),
    (".python/cpython/bin/python3", "an installed interpreter"),
    (".hf/hub/x", "a Hugging Face cache"),
    ("model-dir/config.json", "a built model directory"),
    ("layer0.tensorbin", "a tensor cache file"),
    ("model-00001-of-00002.safetensors", "a weights file"),
    ("pytorch_model.bin", "a weights or cache file"),
])
def test_caches_weights_and_install_output_are_hits(tmp_path, rel, what):
    root = package_dir(tmp_path / "pkg")
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text("x")
    assert scrub_package(root, hostname=HOST, home=HOME, namespace="episod") == [
        f"{rel.split('/')[0]}: {what}"]


def test_a_symbolic_link_is_a_hit(tmp_path):
    root = package_dir(tmp_path / "pkg")
    (root / "config.json").symlink_to(tmp_path / "elsewhere.json")
    assert scrub_package(root, hostname=HOST, home=HOME, namespace="episod") == [
        "config.json: a symbolic link"]


def test_the_hostname_tokens_and_home_paths_are_found_in_text_files(tmp_path):
    root = package_dir(tmp_path / "pkg")
    (root / "tt_kernel_manifest.json").write_text('{"producer": {"hostname": "quietbox-7"}}\n')
    (root / "run.sh").write_text("export HF_HOME=/home/alice/.cache/huggingface\n")
    assert scrub_package(root, hostname=HOST, home=HOME, namespace="episod") == [
        "run.sh: an absolute home path", "tt_kernel_manifest.json: the hostname 'quietbox-7'"]


def test_the_operator_namespace_is_allowed_only_in_the_card(tmp_path):
    root = package_dir(tmp_path / "pkg")
    (root / "run.sh").write_text('export HF_MODEL="episod/hemmingway-1-p300"\n')
    assert scrub_package(root, hostname=HOST, home=HOME, namespace="episod") == [
        "run.sh: the operator's namespace 'episod' outside README.md"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_scrub.py`
Expected: FAIL with a collection error, `ImportError: cannot import name 'scrub_package' from 'orchard.scrub'`.

- [ ] **Step 3: Implement**

Append to `orchard/scrub.py`:

```python


# ---- the package scrub (plan 5, stage 7) --------------------------------------------------------
# A staged package is uploaded as it is, so it gets a stricter check than the operator bundle:
# besides the text search above, nothing produced by installing or serving may be in it. A tensor
# cache is keyed by layer name only, so a cache built for another model would serve that model's
# weights without an error. Weights are a pointer in the manifest and are never shipped. Wheels are
# binary and are not searched; orchard/package.py checks that each one is byte-identical to the
# wheel in the source bundle it came from.

FORBIDDEN_DIRS = {".tt_cache": "a tensor cache directory", "tensors": "a tensor cache directory",
                  "venv": "an installed venv", ".venv": "an installed venv",
                  ".python": "an installed interpreter", ".uv": "an installed uv",
                  ".hf": "a Hugging Face cache", ".cache": "a runtime cache",
                  "model-dir": "a built model directory"}
FORBIDDEN_SUFFIXES = {".tensorbin": "a tensor cache file", ".safetensors": "a weights file",
                      ".bin": "a weights or cache file", ".pt": "a weights file",
                      ".pth": "a weights file", ".gguf": "a weights file"}
BINARY_SUFFIXES = (".whl",)
NAMESPACE_FILES = frozenset({"README.md"})       # where the operator's repo id is meant to appear


def scrub_package(root, *, hostname: str | None = None, home: str | None = None,
                  namespace: str | None = None) -> list[str]:
    """Every hit in a staged package, as 'relative/path: what was found', sorted by path.

    A forbidden directory is reported once and not searched. `namespace` is the operator's Hugging
    Face namespace: it belongs in the card's serve and publish lines and nowhere else."""
    hostname = socket.gethostname() if hostname is None else hostname
    home = os.path.expanduser("~") if home is None else home
    root = Path(root)
    hits: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        for d in sorted(dirnames):
            rel = (here / d).relative_to(root).as_posix()
            if (here / d).is_symlink():
                hits.append((rel, "a symbolic link"))
            elif d in FORBIDDEN_DIRS:
                hits.append((rel, FORBIDDEN_DIRS[d]))
        dirnames[:] = [d for d in dirnames if d not in FORBIDDEN_DIRS and not (here / d).is_symlink()]
        for name in filenames:
            f = here / name
            rel = f.relative_to(root).as_posix()
            if f.is_symlink():
                hits.append((rel, "a symbolic link"))
                continue
            suffix = f.suffix.lower()
            if suffix in FORBIDDEN_SUFFIXES:
                hits.append((rel, FORBIDDEN_SUFFIXES[suffix]))
                continue
            if suffix in BINARY_SUFFIXES:
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            hits += [(rel, what) for what in scrub_text(text, hostname=hostname, home=home)]
            if (namespace and rel not in NAMESPACE_FILES
                    and re.search(rf"(?<![\w.-]){re.escape(namespace)}/", text)):
                hits.append((rel, f"the operator's namespace {namespace!r} outside README.md"))
    return [f"{rel}: {what}" for rel, what in sorted(hits)]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_scrub.py`
Expected: PASS (24 passed).

- [ ] **Step 5: Mutation checks**

1. A tensor cache is a hit: in `orchard/scrub.py`, replace `FORBIDDEN_DIRS = {".tt_cache": "a tensor cache directory",` with `FORBIDDEN_DIRS = {`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_scrub.py::test_caches_weights_and_install_output_are_hits`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
2. The namespace check: in `orchard/scrub.py`, replace `if (namespace and rel not in NAMESPACE_FILES` with `if (False and rel not in NAMESPACE_FILES`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_scrub.py::test_the_operator_namespace_is_allowed_only_in_the_card`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
3. A symbolic link is a hit: in `orchard/scrub.py`, replace

```python
            if f.is_symlink():
                hits.append((rel, "a symbolic link"))
                continue
```

   with nothing (delete these lines).

   Run `python3 -m pytest -q -p no:cacheprovider tests/test_scrub.py::test_a_symbolic_link_is_a_hit`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.


- [ ] **Step 6: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: 1350 passed and 1 skipped (the skip is the opt-in replay).

```bash
git add orchard/scrub.py tests/test_scrub.py
git commit -m "Add the package scrub: caches, weights, install output, links and the namespace"
```

---

### Task 4: The package card and its license

**Files:**
- Create: `orchard/package_card.py`
- Test: `tests/test_package_card.py`

**Interfaces:**
- Consumes: `orchard.stages.inside(run_dir, rel) -> Path | None`.
- Produces in `orchard.package_card`: `LICENSES`, `PERMISSIVE`, `NC_LINE = "Non-commercial use only."`, `COMMERCIAL_CLAIMS`, `read_license(snapshot) -> str | None`, `non_commercial(license_id) -> bool`, `Number(name, value, unit, label, evidence: tuple[str, ...])`, `CardFacts(name, namespace, model_id, revision, nearest_model, source_name, license_id, chips, mesh, arch, max_model_len, max_num_seqs, drafter, verified, numbers, not_measured)`, `render_card(f: CardFacts) -> str`, `card_problems(card: str, *, license_id: str, run_dir) -> list[str]`.

The card is the package's README.md, so it is what a stranger reads on the Hub. Its front matter carries the license and the tags `tt-model search` filters on (the same tags tt-model's own push adds). The license comes from the new model's own card: Altworld/Hemmingway-1 is CC BY-NC 4.0, and the base model's Apache 2.0 does not carry over. A license this module does not know counts as non-commercial. `card_problems` reads a card back from disk, so the gate (Task 9) checks the file as it is, edits included.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_package_card.py`:

```python
"""The package card and its license (plan 5, stage 7)."""
import json

import pytest

from orchard.package_card import (CardFacts, Number, card_problems, non_commercial, read_license,
                                  render_card)

REV = "1a5f363a3dd2d1cc456c28b8abbb403b9555efaf"


def run_with_evidence(tmp_path):
    run = tmp_path / "run"
    (run / "stages" / "2").mkdir(parents=True)
    (run / "stages" / "2" / "result.json").write_text(json.dumps({"top1_agreement": 0.94,
                                                                  "server_ready_s": 280.5}))
    (run / "stages" / "2" / "evidence").mkdir()
    (run / "stages" / "2" / "evidence" / "swap-check.json").write_text("{}")
    return run


def facts(**over):
    base = dict(
        name="hemmingway-1-p300", namespace="episod", model_id="Altworld/Hemmingway-1",
        revision=REV, nearest_model="Qwen/Qwen3.8-27B", source_name="qwen3.8-27b-dflash2-p300",
        license_id="cc-by-nc-4.0", chips=2, mesh="P150x2", arch="blackhole", max_model_len=262144,
        max_num_seqs=4, drafter="incoai/Qwen3.8-27B-DFlash2", verified=True,
        numbers=(Number("top1 agreement with the CPU reference (2 chips, stage 2)", 0.94, "fraction",
                        "measured", ("stages/2/result.json", "stages/2/evidence/swap-check.json")),
                 Number("time to first token", None, "ms", "TODO", ())),
        not_measured=("the drafter's acceptance rate on this model",))
    base.update(over)
    return CardFacts(**base)


def test_a_rendered_card_passes_its_own_check(tmp_path):
    run = run_with_evidence(tmp_path)
    card = render_card(facts())
    assert card_problems(card, license_id="cc-by-nc-4.0", run_dir=run) == []
    assert card.startswith("---\nlicense: cc-by-nc-4.0\nbase_model: Altworld/Hemmingway-1\n")
    for tag in ("tt-model-cache", "blackhole", "vllm", "thin", "p150x2"):
        assert f"\n- {tag}\n" in card
    assert "tt-model serve episod/hemmingway-1-p300" in card
    assert "| time to first token | TODO | TODO | - |" in card


def test_a_non_commercial_license_is_shown_and_says_non_commercial(tmp_path):
    card = render_card(facts())
    assert "CC BY-NC 4.0" in card and "Non-commercial use only." in card
    assert "Commercial use is not permitted by that license." in card


def test_a_permissive_license_card_has_no_non_commercial_line(tmp_path):
    card = render_card(facts(license_id="apache-2.0"))
    assert "Apache 2.0" in card and "Non-commercial" not in card
    assert card_problems(card, license_id="apache-2.0", run_dir=run_with_evidence(tmp_path)) == []


@pytest.mark.parametrize("claim", [
    "Commercial use is permitted.", "Ready for commercial deployments.", "Use it commercially.",
    "Production-ready for your customers.", "Licensed for enterprise use."])
def test_a_card_that_implies_commercial_use_of_a_non_commercial_model_is_refused(tmp_path, claim):
    card = render_card(facts()).replace("## What it runs", claim + "\n\n## What it runs")
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run_with_evidence(tmp_path))
    assert any("implies commercial use" in p for p in problems), problems


def test_a_card_that_lost_its_license_is_refused(tmp_path):
    run = run_with_evidence(tmp_path)
    card = render_card(facts())
    assert any("license" in p for p in card_problems(card.replace("license: cc-by-nc-4.0\n", ""),
                                                     license_id="cc-by-nc-4.0", run_dir=run))
    no_line = card.replace("Non-commercial use only. ", "")
    assert any("Non-commercial use only." in p
               for p in card_problems(no_line, license_id="cc-by-nc-4.0", run_dir=run))
    other = render_card(facts(license_id="apache-2.0"))
    assert any("apache-2.0" in p and "cc-by-nc-4.0" in p
               for p in card_problems(other, license_id="cc-by-nc-4.0", run_dir=run))


def test_a_measured_number_needs_its_evidence_and_its_value_in_the_first_file(tmp_path):
    run = run_with_evidence(tmp_path)
    missing = facts(numbers=(Number("ready", 280.5, "s", "measured", ("stages/2/nope.json",)),))
    assert any("stages/2/nope.json" in p for p in
               card_problems(render_card(missing), license_id="cc-by-nc-4.0", run_dir=run))
    wrong = facts(numbers=(Number("ready", 99.0, "s", "measured", ("stages/2/result.json",)),))
    assert any("99.0" in p for p in
               card_problems(render_card(wrong), license_id="cc-by-nc-4.0", run_dir=run))


def test_a_number_in_the_prose_is_refused(tmp_path):
    card = render_card(facts()).replace("## How to serve", "It decodes at 80 tok/s.\n\n## How to serve")
    assert any("80 tok/s" in p for p in
               card_problems(card, license_id="cc-by-nc-4.0", run_dir=run_with_evidence(tmp_path)))


def test_a_row_labelled_measured_by_hand_without_a_value_is_refused(tmp_path):
    card = render_card(facts()).replace("| time to first token | TODO | TODO | - |",
                                        "| time to first token | fast | measured | - |")
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run_with_evidence(tmp_path))
    assert any("time to first token" in p for p in problems), problems


def test_the_license_is_read_from_the_snapshot_card(tmp_path):
    (tmp_path / "README.md").write_text("---\nlicense: cc-by-nc-4.0\nbase_model:\n- Qwen/Qwen3.8-27B\n"
                                        "---\n# Hemmingway-1\n")
    assert read_license(tmp_path) == "cc-by-nc-4.0"
    (tmp_path / "README.md").write_text("---\nlicense: 'Apache-2.0'\n---\n")
    assert read_license(tmp_path) == "apache-2.0"
    (tmp_path / "README.md").write_text("# no front matter\nlicense: mit\n")
    assert read_license(tmp_path) is None
    (tmp_path / "README.md").unlink()
    assert read_license(tmp_path) is None


@pytest.mark.parametrize("license_id, nc", [("cc-by-nc-4.0", True), ("cc-by-nc-sa-4.0", True),
                                            ("cc-by-nc-nd-4.0", True), ("apache-2.0", False),
                                            ("mit", False), ("cc-by-4.0", False), ("other", True)])
def test_non_commercial_licenses_are_recognised(license_id, nc):
    assert non_commercial(license_id) is nc
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_package_card.py`
Expected: FAIL with a collection error, `ModuleNotFoundError: No module named 'orchard.package_card'`.

- [ ] **Step 3: Implement**

Create `orchard/package_card.py`:

```python
"""The package card and the license it carries (plan 5, stage 7).

This module owns the README.md of a staged package: what it says, and the check that it says only
what the run can back. `render_card` writes it from `CardFacts`; `card_problems` reads a card back
and lists what is wrong. The gate (orchard/stages.py, gate_package) runs the check on the file on
disk, so a card edited after it was rendered is checked as it is.

The license comes from the new model's own Hugging Face card (`read_license`, the `license:` key
in the snapshot's README.md front matter). A license this module cannot name, and `other`, count
as non-commercial: the card then says "Non-commercial use only." and the check refuses wording
that implies commercial use. Altworld/Hemmingway-1 is CC BY-NC 4.0, and the base model's Apache
2.0 license does not carry over to it.

Numbers appear only in the card's Numbers table. Each row is labelled `measured` or `TODO`. A
measured row names its evidence files, relative to the run directory; the first one is the result
file the number was read from, and the check requires the value to appear in it. A number with a
performance unit anywhere else in the card is refused, because nothing ties it to evidence.

What the check does not do: judge whether the prose is true, or catch a commercial claim worded in
a way the patterns below do not list.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from orchard.stages import inside

LICENSES = {"cc-by-nc-4.0": ("CC BY-NC 4.0", "https://creativecommons.org/licenses/by-nc/4.0/"),
            "cc-by-nc-sa-4.0": ("CC BY-NC-SA 4.0", "https://creativecommons.org/licenses/by-nc-sa/4.0/"),
            "cc-by-nc-nd-4.0": ("CC BY-NC-ND 4.0", "https://creativecommons.org/licenses/by-nc-nd/4.0/"),
            "cc-by-4.0": ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"),
            "apache-2.0": ("Apache 2.0", "https://www.apache.org/licenses/LICENSE-2.0"),
            "mit": ("MIT", "https://opensource.org/licenses/MIT")}
PERMISSIVE = frozenset({"apache-2.0", "mit", "cc-by-4.0"})
NC_LINE = "Non-commercial use only."
# Wording that implies commercial use. Checked case-insensitively on cards for non-commercial models.
COMMERCIAL_CLAIMS = (r"(?<!non-)(?<!non )commercial (?:use|deployments?|purposes?|products?|"
                     r"applications?)(?! is not permitted)",
                     r"\bcommercially\b", r"production[- ]ready", r"\bfor (?:your )?customers\b",
                     r"\benterprise\b", r"\bresell", r"\bmonetiz")
PERF_IN_PROSE = re.compile(r"\b\d[\d.,]*\s*(?:tok/s|tokens/s|t/s|tokens per second|ms\b|%)")
NUMBERS_HEAD = "## Numbers"
CARD_TAGS = ("tt-model-cache", "vllm", "thin")       # the tags tt-model's own push adds to a v6 bundle


def read_license(snapshot) -> str | None:
    """The `license:` value of the front matter in `<snapshot>/README.md`, lowercased, or None."""
    try:
        text = (Path(snapshot) / "README.md").read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.startswith("---\n"):
        return None
    front = text[4:].split("\n---", 1)[0]
    m = re.search(r"^license:\s*['\"]?([A-Za-z0-9.+-]+)['\"]?\s*$", front, re.M)
    return m.group(1).lower() if m else None


def non_commercial(license_id: str) -> bool:
    """True for any CC NC license and for anything this module does not know to be permissive."""
    return license_id not in PERMISSIVE


@dataclass(frozen=True)
class Number:
    name: str
    value: float | int | None
    unit: str
    label: str                       # "measured" or "TODO"
    evidence: tuple[str, ...]        # run-relative; the first is the result file holding the value


@dataclass(frozen=True)
class CardFacts:
    name: str                        # the bundle name, also the repo name the publish line uses
    namespace: str                   # the operator's Hugging Face namespace
    model_id: str
    revision: str
    nearest_model: str
    source_name: str                 # the nearest model's bundle this package was built from
    license_id: str
    chips: int
    mesh: str
    arch: str
    max_model_len: int
    max_num_seqs: int
    drafter: str | None
    verified: bool                   # the stage 7 boot check passed for this profile
    numbers: tuple[Number, ...]
    not_measured: tuple[str, ...]


def _value(n: Number) -> str:
    return "TODO" if n.label != "measured" else f"{json.dumps(n.value)} {n.unit}"


def render_card(f: CardFacts) -> str:
    lic_name, lic_link = LICENSES.get(f.license_id, (f.license_id, None))
    lic = f"{lic_name} ({lic_link})" if lic_link else lic_name
    tags = [*CARD_TAGS[:1], f.arch, *CARD_TAGS[1:], f.mesh.lower()]
    lines = ["---", f"license: {f.license_id}", f"base_model: {f.model_id}",
             "pipeline_tag: text-generation", "tags:", *[f"- {t}" for t in tags], "---", "",
             f"# {f.name}", "",
             f"A tt-model v6 thin bundle that serves {f.model_id} (revision `{f.revision}`) on "
             f"{f.chips} Tenstorrent {f.arch} chips (mesh {f.mesh}). The tt-orchard harness staged it "
             "from a run of that model. It is a draft for operator review.", "",
             "## License", ""]
    if non_commercial(f.license_id):
        lines.append(f"The weights this bundle points to are licensed {lic}. {NC_LINE} "
                     "Commercial use is not permitted by that license.")
    else:
        lines.append(f"The weights this bundle points to are licensed {lic}.")
    lines += ["The bundle ships no weights. `tt-model` downloads them from "
              f"{f.model_id} with your own Hugging Face account.", "",
              "## What it runs", "",
              f"- Weights: {f.model_id} at revision `{f.revision}`.",
              f"- Model code: the Tenstorrent implementation of {f.nearest_model}, taken from the "
              f"bundle {f.source_name}. {f.model_id} has the same text architecture, so only the "
              "weights differ.",
              f"- At each start the launcher builds `model-dir/` from the configuration files of "
              f"{f.nearest_model} (shipped in `base_config/`) and the tokenizer and weights of "
              f"{f.model_id}. It sets MODEL_WEIGHTS_DIR and HF_MODEL to that directory, so the chips "
              f"load the weights of {f.model_id}.",
              f"- Context length {f.max_model_len} tokens; up to {f.max_num_seqs} sequences at a time."]
    if f.drafter:
        lines.append(f"- Speculative decoding uses the drafter {f.drafter}, which was trained on "
                     f"{f.nearest_model}. Its acceptance rate on this model is listed under Not "
                     "measured.")
    lines += ["", NUMBERS_HEAD, "",
              "Every number is labelled. A `measured` number names its evidence files, relative to "
              "the run directory; the first file holds the value. `TODO` means not measured.", "",
              "| Number | Value | Label | Evidence |", "|---|---|---|---|"]
    for n in f.numbers:
        ev = ", ".join(f"`{e}`" for e in n.evidence) if n.label == "measured" else "-"
        lines.append(f"| {n.name} | {_value(n)} | {n.label} | {ev} |")
    lines += ["", "## Not measured", "", *[f"- {item}" for item in f.not_measured], "",
              "## Boot check", "",
              ("Stage 7 of the run installed this bundle, served it on a leased board and compared "
               "its tokens with the CPU reference. The Numbers table has the result."
               if f.verified else
               "This profile was not booted on hardware by the run. Boot it before publishing."),
              "", "## How to serve", "", f"    tt-model serve {f.namespace}/{f.name}", ""]
    return "\n".join(lines)


def _front(card: str) -> str:
    return card[4:].split("\n---", 1)[0] if card.startswith("---\n") else ""


def _numbers_section(card: str) -> tuple[str, str]:
    """(the Numbers section, the rest of the card)."""
    if NUMBERS_HEAD not in card:
        return "", card
    before, after = card.split(NUMBERS_HEAD, 1)
    section, sep, rest = after.partition("\n## ")
    return section, before + (sep + rest if sep else "")


def card_problems(card: str, *, license_id: str, run_dir) -> list[str]:
    problems = []
    m = re.search(r"^license:\s*(\S+)\s*$", _front(card), re.M)
    if m is None:
        problems.append("the card's front matter has no license")
    elif m.group(1) != license_id:
        problems.append(f"the card's license is {m.group(1)}; the model's license is {license_id}")
    if non_commercial(license_id):
        if NC_LINE not in card:
            problems.append(f"the card for a {license_id} model must say {NC_LINE!r}")
        name = LICENSES.get(license_id, (license_id, None))[0]
        if name not in card:
            problems.append(f"the card must name the license {name!r}")
        for pattern in COMMERCIAL_CLAIMS:
            hit = re.search(pattern, card, re.I)
            if hit:
                problems.append(f"the card implies commercial use of a {license_id} model: "
                                f"{hit.group(0)!r}")
    section, rest = _numbers_section(card)
    if not section:
        problems.append("the card has no Numbers section")
    for row in (r for r in section.splitlines() if r.startswith("| ") and not r.startswith("| Number ")):
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) != 4:
            problems.append(f"Numbers row {row!r} needs four cells")
            continue
        name, value, label, evidence = cells
        if label == "TODO":
            if value != "TODO":
                problems.append(f"number {name!r} is TODO and shows a value")
            continue
        if label != "measured":
            problems.append(f"number {name!r} must be labelled measured or TODO")
            continue
        shown = value.split(" ", 1)[0]
        try:
            float(shown)
        except ValueError:
            problems.append(f"number {name!r} is labelled measured and has no numeric value")
            continue
        paths = [p.strip().strip("`") for p in evidence.split(",") if p.strip() not in ("", "-")]
        if not paths:
            problems.append(f"number {name!r} is labelled measured and names no evidence")
            continue
        found = [inside(run_dir, p) for p in paths]
        for p, f in zip(paths, found):
            if f is None:
                problems.append(f"number {name!r}: evidence {p} is not a file inside the run directory")
        if found[0] is not None and shown not in found[0].read_text(encoding="utf-8", errors="replace"):
            problems.append(f"number {name!r}: the value {shown} is not in {paths[0]}")
    for hit in PERF_IN_PROSE.finditer(rest):
        problems.append(f"a number outside the Numbers table: {hit.group(0)!r}")
    return problems
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_package_card.py`
Expected: PASS (20 passed).

- [ ] **Step 5: Mutation checks**

1. The non-commercial line: in `orchard/package_card.py`, replace `if NC_LINE not in card:` with `if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_card.py::test_a_card_that_lost_its_license_is_refused`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
2. Commercial wording: in `orchard/package_card.py`, replace `for pattern in COMMERCIAL_CLAIMS:` with `for pattern in ():`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_card.py::test_a_card_that_implies_commercial_use_of_a_non_commercial_model_is_refused`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
3. A measured value must be in its evidence: in `orchard/package_card.py`, replace `if found[0] is not None and shown not in` with `if False and shown not in`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_card.py::test_a_measured_number_needs_its_evidence_and_its_value_in_the_first_file`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
4. Numbers outside the table: in `orchard/package_card.py`, replace `for hit in PERF_IN_PROSE.finditer(rest):` with `for hit in []:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_card.py::test_a_number_in_the_prose_is_refused`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
5. An unknown license counts as non-commercial: in `orchard/package_card.py`, replace `return license_id not in PERMISSIVE` with `return "-nc" in license_id`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_card.py::test_non_commercial_licenses_are_recognised`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: 1370 passed and 1 skipped (the skip is the opt-in replay).

```bash
git add orchard/package_card.py tests/test_package_card.py
git commit -m "Add the package card: license front matter, non-commercial check, labelled numbers"
```

---

### Task 5: run.sh wiring for the new weights, and prepare_model_dir.py

**Files:**
- Create: `orchard/package.py` (the module header and the run.sh edits; Tasks 6 to 8 append the rest)
- Create: `orchard/package_templates/prepare_model_dir.py`
- Test: `tests/test_package_runsh.py`

**Interfaces:**
- Consumes: `orchard.package_card` (Task 4), `orchard.scrub.scrub_package` (Task 3), `orchard.stages.gate_weights_swap`, the plan 5 defaults (Task 1). The header imports all of them now, so later tasks only append.
- Produces in `orchard.package`: `PackageError`, `MODEL_DIR = '"$HERE/model-dir"'`, `WIRING_LINES`, `PREPARE_LINE`, `EXEC_LINE`, `TEMPLATES`, `SWAP_TEMPLATES`, `BASE_CONFIG_FILES`, `extra_args_from(source_run_sh: str) -> str`, `splice_extra_args(text: str, extra: str) -> str`, `wire_weights(text: str, model_id: str) -> str`, `weights_wiring_problems(text: str, *, nearest_model: str) -> list[str]`.
- Produces the template `orchard/package_templates/prepare_model_dir.py` (exit 0 ready, 2 refused).

This task holds the fact the whole package rests on. The TT runtime takes its weights directory from `MODEL_WEIGHTS_DIR`, then `HF_MODEL`, then the config path; a bundle that leaves `HF_MODEL` at the nearest model's id serves the nearest model's weights with no error (run log, 2026-10-03 17:45Z: 25 of 32 against 30 of 32). The bundle's model class is registered for the nearest model's vision-language architecture, and Hemmingway-1's config names `Qwen3_5ForCausalLM`, which vLLM refuses. So the staged `run.sh` runs `prepare_model_dir.py` before vLLM. It builds `model-dir/` from the nearest model's config files (shipped in `base_config/`) and the new model's tokenizer and weights, exactly as stage 2's `prepare_swap.py` does, and `--model`, `HF_MODEL` and `MODEL_WEIGHTS_DIR` all name it. package-thin has no flag for fixed vLLM arguments, so they are copied from the source bundle's `run.sh` and spliced in before `"$@"`. The test's run.sh lines are copied from the staged `qwen3.8-27b-dflash2-p300` bundle of 2026-09-30.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_package_runsh.py`:

```python
"""The run.sh edits for a weights-only package, and the model-dir script it runs (plan 5)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchard.package import (EXEC_LINE, PREPARE_LINE, WIRING_LINES, PackageError, extra_args_from,
                             splice_extra_args, weights_wiring_problems, wire_weights)

NEW, BASE = "Altworld/Hemmingway-1", "Qwen/Qwen3.8-27B"
NEW_REV, BASE_REV = "1a5f363a3dd2d1cc456c28b8abbb403b9555efaf", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
EXTRA = ("--additional-config '{\"tt\": {\"l1_small_size\": 24576, \"fabric_config\": \"FABRIC_1D\"}}' "
         "--max-num-batched-tokens 65536 --enable-auto-tool-choice --tool-call-parser qwen3_coder "
         "--reasoning_parser qwen3 --no-async-scheduling")
TEMPLATE = Path(__file__).resolve().parent.parent / "orchard" / "package_templates" / "prepare_model_dir.py"


def generated_run_sh(weights, rev, extra=""):
    """The lines of a package-thin run.sh that the edits touch, in the generated order
    (copied from the staged qwen3.8-27b-dflash2-p300 bundle, 2026-09-30)."""
    tail = f" {extra}" if extra else ""
    return (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'VENV="${VENV:-$HERE/venv}"\nPYBIN="$VENV/bin/python"\n'
        'export HF_HOME="${HF_HOME:-$HERE/.hf}"\n'
        f'export HF_MODEL="${{HF_MODEL:-{weights}}}"\n'
        f'export TT_MODEL_WEIGHTS_REVISION="${{TT_MODEL_WEIGHTS_REVISION:-{rev}}}"\n'
        'export DFLASH_WEIGHTS="incoai/Qwen3.8-27B-DFlash2@dedf8df68adfb1afeaf7b7480c0a0243108177b4"\n'
        f'CMD=("$PYBIN" -m vllm.entrypoints.openai.api_server --model "{weights}" --max_num_seqs 4 '
        f'--block_size 64 --revision {rev} --tokenizer-revision {rev} --max_model_len 262144{tail} "$@")\n'
        'if [ "${TT_MODEL_PRINT:-0}" = "1" ]; then\n'
        "  printf 'HF_MODEL=%s\\n' \"${HF_MODEL:-}\"\n  exit 0\nfi\n"
        'exec "${CMD[@]}"\n')


def test_the_extra_arguments_are_read_from_the_source_bundle():
    assert extra_args_from(generated_run_sh(BASE, BASE_REV, EXTRA)) == EXTRA
    assert extra_args_from(generated_run_sh(BASE, BASE_REV)) == ""


def test_a_source_run_sh_without_the_expected_command_is_refused():
    with pytest.raises(PackageError, match="--max_model_len"):
        extra_args_from(generated_run_sh(BASE, BASE_REV).replace("--max_model_len 262144", ""))


def test_the_extra_arguments_go_before_the_passed_through_ones():
    text = splice_extra_args(generated_run_sh(NEW, NEW_REV), EXTRA)
    assert f'--max_model_len 262144 {EXTRA} "$@")\n' in text
    assert extra_args_from(text) == EXTRA


def test_wiring_points_every_weights_setting_at_the_model_dir():
    text = wire_weights(generated_run_sh(NEW, NEW_REV, EXTRA), NEW)
    for line in WIRING_LINES:
        assert f"\n{line}\n" in text
    assert '--model "$HERE/model-dir" --max_num_seqs 4' in text
    assert "--revision" not in text and "--tokenizer-revision" not in text
    assert text.index(PREPARE_LINE) < text.index(EXEC_LINE)
    assert NEW not in text.split("CMD=(")[1]          # the command names the model-dir only
    assert weights_wiring_problems(text, nearest_model=BASE) == []


def test_the_unwired_generated_script_has_wiring_problems():
    problems = weights_wiring_problems(generated_run_sh(NEW, NEW_REV), nearest_model=BASE)
    assert any("MODEL_WEIGHTS_DIR" in p for p in problems)
    assert any("prepare_model_dir.py" in p for p in problems)


def test_a_script_that_names_the_base_model_has_a_wiring_problem():
    text = wire_weights(generated_run_sh(NEW, NEW_REV), NEW) + f'export X="{BASE}"\n'
    assert weights_wiring_problems(text, nearest_model=BASE) == [
        f"run.sh names the nearest model {BASE}"]


def test_a_second_weights_setting_has_a_wiring_problem():
    text = wire_weights(generated_run_sh(NEW, NEW_REV), NEW).replace(
        EXEC_LINE, f'export MODEL_WEIGHTS_DIR="/somewhere/else"\n{EXEC_LINE}')
    assert any("MODEL_WEIGHTS_DIR" in p for p in weights_wiring_problems(text, nearest_model=BASE))


@pytest.mark.parametrize("edit, named", [
    (lambda s: s.replace(f'--model "{NEW}"', '--model "Other/Model"'), "--model"),
    (lambda s: s.replace(f"--revision {NEW_REV} ", ""), "--revision"),
    (lambda s: s.replace('export HF_MODEL="${HF_MODEL:-' + NEW + '}"\n', ""), "HF_MODEL"),
    (lambda s: s.replace('exec "${CMD[@]}"\n', ""), "exec"),
])
def test_an_edit_that_does_not_apply_exactly_once_is_refused(edit, named):
    with pytest.raises(PackageError, match=named):
        wire_weights(edit(generated_run_sh(NEW, NEW_REV)), NEW)


# ---- prepare_model_dir.py, the script the staged run.sh runs before vLLM -------------------------

def hf_snapshot(hf_home, repo, rev, files):
    """An HF-cache-shaped snapshot under hf_home/hub: files are links into blobs/."""
    org, name = repo.split("/")
    root = hf_home / "hub" / f"models--{org}--{name}"
    blobs, snap = root / "blobs", root / "snapshots" / rev
    blobs.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True)
    for fname, content in files.items():
        (blobs / f"b-{fname}").write_text(content)
        (snap / fname).symlink_to(os.path.relpath(blobs / f"b-{fname}", snap))
    return snap


@pytest.fixture
def bundle(tmp_path):
    b = tmp_path / "bundle"
    (b / "base_config").mkdir(parents=True)
    (b / "base_config" / "config.json").write_text('{"architectures": ["Qwen3_5ForConditionalGeneration"]}')
    (b / "base_config" / "preprocessor_config.json").write_text("{}")
    (b / "tt_kernel_manifest.json").write_text(json.dumps(
        {"schema_version": "6", "weights": {"repo_id": NEW, "revision": NEW_REV}}))
    (b / "prepare_model_dir.py").write_text(TEMPLATE.read_text())
    return b


def run_prepare(bundle, hf_home, offline=True, pythonpath=None):
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "HF_HUB_CACHE")}
    env["HF_HOME"] = str(hf_home)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    if pythonpath:
        env["PYTHONPATH"] = str(pythonpath)
    return subprocess.run([sys.executable, str(bundle / "prepare_model_dir.py")], env=env,
                          capture_output=True, text=True, timeout=60)


NEW_FILES = {"config.json": '{"architectures": ["Qwen3_5ForCausalLM"]}', "tokenizer.json": "{}",
             "tokenizer_config.json": "{}", "generation_config.json": "{}",
             "model.safetensors.index.json": "{}", "model-00001-of-00002.safetensors": "w1",
             "model-00002-of-00002.safetensors": "w2"}


def test_the_model_dir_has_the_base_config_and_the_new_weights(bundle, tmp_path):
    hf = tmp_path / "hf"
    snap = hf_snapshot(hf, NEW, NEW_REV, NEW_FILES)
    r = run_prepare(bundle, hf)
    assert r.returncode == 0, r.stdout + r.stderr
    md = bundle / "model-dir"
    assert json.loads((md / "config.json").read_text())["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert not (md / "config.json").is_symlink()
    for name in ("tokenizer.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        assert os.readlink(md / name) == os.path.realpath(snap / name)
    assert (md / ".weights").read_text() == f"{NEW}@{NEW_REV}"
    assert run_prepare(bundle, hf).returncode == 0              # a second start rebuilds it


def test_a_missing_snapshot_offline_exits_2_and_names_it(bundle, tmp_path):
    r = run_prepare(bundle, tmp_path / "empty-hf")
    assert r.returncode == 2
    assert NEW in r.stderr and NEW_REV in r.stderr
    assert not (bundle / "model-dir").exists()


def test_a_missing_snapshot_online_is_downloaded_at_the_pinned_revision(bundle, tmp_path):
    hf = tmp_path / "hf"
    fake = tmp_path / "fakepkgs" / "huggingface_hub"
    fake.mkdir(parents=True)
    calls = tmp_path / "calls.json"
    (fake / "__init__.py").write_text(
        "import json, os, pathlib\n"
        "def snapshot_download(repo_id, revision):\n"
        f"    pathlib.Path({str(calls)!r}).write_text(json.dumps([repo_id, revision]))\n"
        "    snap = pathlib.Path(os.environ['HF_HOME']) / 'hub' / 'got' / revision\n"
        "    snap.mkdir(parents=True)\n"
        "    (snap / 'tokenizer.json').write_text('{}')\n"
        "    (snap / 'model-00001-of-00001.safetensors').write_text('w')\n"
        "    return str(snap)\n")
    r = run_prepare(bundle, hf, offline=False, pythonpath=fake.parent)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(calls.read_text()) == [NEW, NEW_REV]
    assert (bundle / "model-dir" / "model-00001-of-00001.safetensors").is_symlink()


def test_a_model_dir_the_script_did_not_build_is_left_alone(bundle, tmp_path):
    hf = tmp_path / "hf"
    hf_snapshot(hf, NEW, NEW_REV, NEW_FILES)
    (bundle / "model-dir").mkdir()
    (bundle / "model-dir" / "mine.txt").write_text("the operator's file")
    r = run_prepare(bundle, hf)
    assert r.returncode == 2 and "not built by this script" in r.stderr
    assert (bundle / "model-dir" / "mine.txt").read_text() == "the operator's file"


def test_an_unpinned_weights_revision_is_refused(bundle, tmp_path):
    (bundle / "tt_kernel_manifest.json").write_text(json.dumps(
        {"weights": {"repo_id": NEW, "revision": "main"}}))
    r = run_prepare(bundle, tmp_path / "hf")
    assert r.returncode == 2 and "40-character" in r.stderr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_package_runsh.py`
Expected: FAIL with a collection error, `ModuleNotFoundError: No module named 'orchard.package'`.

- [ ] **Step 3: Implement**

Create `orchard/package.py`:

```python
"""Stage 7: stage a v6 thin package of a weights-only model, as supervisor code (plan 5).

This module owns the package a run hands to the operator. It never publishes. It runs one external
program on its own, `tt-model package-thin`, always with `--out` and never with a repo id
(`assert_no_publish`), and it runs the staged package's `install.sh` in a separate copy. It never
calls `tt-model push` or `publish`, `hf upload`, `git push` or the Hugging Face hub API. The
publish commands it writes are text for the operator (`publish_commands`).

The package is built from the nearest model's installed v6 bundle (the "source bundle") with the
same model code, wheels, environment and fixed vLLM arguments, and `--weights` naming the new
model at its pinned revision. Three edits follow, each found by failing first on this machine:

1. Fixed vLLM arguments. package-thin has no flag for them, so they are copied from the source
   bundle's run.sh and spliced in before the passed-through `"$@"` (`extra_args_from`,
   `splice_extra_args`).
2. Weights. The TT runtime takes its weights directory from MODEL_WEIGHTS_DIR, then HF_MODEL, then
   the config path. The bundle's model class is registered for the nearest model's architecture
   (a vision-language config), and the new model's config.json names a text-only architecture,
   which vLLM refuses. So run.sh runs `prepare_model_dir.py` before vLLM: it builds `model-dir/`
   from the nearest model's config files (shipped in `base_config/`) and the new model's tokenizer
   and weights. `--model`, HF_MODEL and MODEL_WEIGHTS_DIR all name that directory
   (`wire_weights`). `weights_wiring_problems` checks the result, and the gate runs it again.
3. Provenance. The manifest's `producer.hostname` is replaced, run.sh and install.sh are made
   executable (package-thin leaves them without +x), and every wheel must be byte-identical to
   the source bundle's, because wheels are binary and the scrub does not read them.

Stage 7 ships no tensor cache and no weights (orchard/scrub.py, scrub_package). The boot check
(`prepare_verify`, package_templates/verify_bundle.py) installs and serves a copy, so the staged
directory never gains a venv, a model-dir or a cache. The supervisor passes the agent shells'
environment to every command here (no tokens, HOME inside the run directory), so a call that
tried to upload would also find no credentials.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from orchard.defaults import (PACKAGE_HEALTH_TIMEOUT_S, PACKAGE_INSTALL_TIMEOUT_S,
                              PACKAGE_THIN_TIMEOUT_S, PACKAGE_VERIFY_DEADLINE_S)
from orchard.package_card import CardFacts, Number, non_commercial, read_license, render_card
from orchard.scrub import scrub_package
from orchard.stages import gate_weights_swap

TEMPLATES = Path(__file__).with_name("package_templates")
SWAP_TEMPLATES = Path(__file__).with_name("skills") / "weights-swap-templates"
BASE_CONFIG_FILES = ("config.json", "preprocessor_config.json", "video_preprocessor_config.json")

MODEL_DIR = '"$HERE/model-dir"'
WIRING_LINES = (f"export HF_MODEL={MODEL_DIR}", f"export MODEL_WEIGHTS_DIR={MODEL_DIR}")
PREPARE_LINE = '"$PYBIN" "$HERE/prepare_model_dir.py"'
EXEC_LINE = 'exec "${CMD[@]}"'
PASS_THROUGH = ' "$@")'


class PackageError(Exception):
    """Stage 7 cannot stage or check the package. The message says why."""


# ---- run.sh edits -------------------------------------------------------------------------------

def _cmd_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.startswith("CMD=(")]
    if len(lines) != 1:
        raise PackageError(f"expected exactly one CMD=( line in run.sh, found {len(lines)}")
    return lines[0]


def extra_args_from(source_run_sh: str) -> str:
    """The fixed vLLM arguments a source bundle's run.sh adds after the generated ones.

    package-thin writes `--max_model_len N` last and the passed-through `"$@"` at the end, so
    whatever stands between the two was spliced in by the bundle's author."""
    m = re.search(r'--max_model_len \d+(?P<extra>.*) "\$@"\)\s*$', _cmd_line(source_run_sh))
    if m is None:
        raise PackageError('the source run.sh command has no "--max_model_len N ... "$@")" to read '
                           "the fixed vLLM arguments from")
    return m.group("extra").strip()


def splice_extra_args(text: str, extra: str) -> str:
    """Put `extra` just before the passed-through arguments, so an operator's own arguments still
    come last and win."""
    if not extra:
        return text
    line = _cmd_line(text)
    if not line.endswith(PASS_THROUGH):
        raise PackageError('the run.sh command does not end with "$@")')
    return text.replace(line, line[:-len(PASS_THROUGH)] + " " + extra + PASS_THROUGH)


def _once(pattern: str, repl: str, text: str, what: str, flags=0) -> str:
    out, n = re.subn(pattern, lambda m: repl, text, flags=flags)
    if n != 1:
        raise PackageError(f"expected exactly one {what} in run.sh, found {n}")
    return out


def wire_weights(text: str, model_id: str) -> str:
    """Point every weights setting in a generated run.sh at the bundle's model-dir."""
    q = re.escape(model_id)
    text = _once(rf'--model (?:"{q}"|{q})(?=\s)', f"--model {MODEL_DIR}", text, f'--model "{model_id}"')
    text = _once(r" --revision [0-9a-f]{40}(?=\s)", "", text, "--revision <40 hex>")
    text = _once(r" --tokenizer-revision [0-9a-f]{40}(?=\s)", "", text, "--tokenizer-revision <40 hex>")
    text = _once(r"^export HF_MODEL=.*$", "\n".join(WIRING_LINES), text, "export HF_MODEL= line",
                 flags=re.M)
    return _once(r"^exec \"\$\{CMD\[@\]\}\"$", f"{PREPARE_LINE}\n{EXEC_LINE}", text,
                 'exec "${CMD[@]}" line', flags=re.M)


def weights_wiring_problems(text: str, *, nearest_model: str) -> list[str]:
    """What is wrong with a staged run.sh's weights settings; empty when the chips will load the
    weights in model-dir and nothing names the nearest model."""
    problems = []
    lines = text.splitlines()
    for var, want in (("HF_MODEL", WIRING_LINES[0]), ("MODEL_WEIGHTS_DIR", WIRING_LINES[1])):
        sets = [ln for ln in lines if re.match(rf"\s*(?:export\s+)?{var}=", ln)]
        if sets != [want]:
            problems.append(f"run.sh must set {var} exactly once, as {want!r}; found {sets}")
    cmd = [ln for ln in lines if ln.startswith("CMD=(")]
    if len(cmd) != 1 or cmd[0].count("--model ") != 1 or f"--model {MODEL_DIR} " not in cmd[0]:
        problems.append(f"the run.sh command must pass --model {MODEL_DIR} once")
    if re.search(r"--(?:tokenizer-)?revision\b", text):
        problems.append("run.sh pins a --revision, which a local model-dir does not have")
    if PREPARE_LINE not in lines or EXEC_LINE not in lines or lines.index(PREPARE_LINE) > lines.index(EXEC_LINE):
        problems.append("run.sh must run prepare_model_dir.py before it starts vLLM")
    if nearest_model in text:
        problems.append(f"run.sh names the nearest model {nearest_model}")
    return problems
```

Create `orchard/package_templates/prepare_model_dir.py`:

```python
#!/usr/bin/env python3
"""Build model-dir/ for this bundle before vLLM starts. run.sh runs it with the bundle's python.

Written by tt-orchard (stage 7) for a model whose weights are a fine-tune of a supported model.
The model code in this bundle is registered for the base model's architecture, so vLLM must see
the base model's config files. The tokenizer and weights must be the fine-tune's. This script
builds model-dir/ next to itself from both:

- COPIED from base_config/ (shipped in the bundle): every file there, for example config.json.
- LINKED from the fine-tune's Hugging Face snapshot (absolute links to the resolved files):
  tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json,
  model.safetensors.index.json and every *.safetensors file. tokenizer.json and at least one
  *.safetensors file are required.

The snapshot is the manifest's `weights.repo_id` at `weights.revision` (which must be a pinned
40-character commit) under $HF_HUB_CACHE or $HF_HOME/hub (run.sh sets HF_HOME). When it is missing
and HF_HUB_OFFLINE is not set, it is downloaded with huggingface_hub from the bundle's venv. When it
is missing and HF_HUB_OFFLINE is set, the script exits 2 and names it.

model-dir/.weights records "<repo>@<revision>". A model-dir without that file was not built here,
and the script exits 2 without touching it. Exit 0 means model-dir is ready.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINKED = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
          "model.safetensors.index.json")
MARKER = ".weights"


def fail(message: str) -> None:
    print(f"prepare_model_dir: {message}", file=sys.stderr)
    sys.exit(2)


def snapshot(repo: str, rev: str) -> Path:
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    org, name = repo.split("/", 1)
    snap = Path(hub) / f"models--{org}--{name}" / "snapshots" / rev
    if snap.is_dir():
        return snap
    if os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on"):
        fail(f"the weights snapshot {repo}@{rev} is not in {hub} and HF_HUB_OFFLINE is set. "
             f"Download it first: hf download {repo} --revision {rev}")
    from huggingface_hub import snapshot_download      # in the bundle's venv
    return Path(snapshot_download(repo_id=repo, revision=rev))


def main() -> int:
    weights = json.loads((HERE / "tt_kernel_manifest.json").read_text(encoding="utf-8"))["weights"]
    repo, rev = weights["repo_id"], weights.get("revision") or ""
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        fail(f"the manifest's weights revision {rev!r} is not a 40-character commit")
    md = HERE / "model-dir"
    if md.is_symlink() or (md.exists() and not (md / MARKER).is_file()):
        fail(f"{md} exists and was not built by this script; move it aside")
    snap = snapshot(repo, rev)
    names = list(LINKED) + sorted(p.name for p in snap.glob("*.safetensors"))
    present = [n for n in names if (snap / n).exists()]
    if "tokenizer.json" not in present or not any(n.endswith(".safetensors") for n in present):
        fail(f"{snap} needs tokenizer.json and at least one *.safetensors file")
    if md.exists():
        shutil.rmtree(md)
    md.mkdir()
    for f in sorted((HERE / "base_config").iterdir()):
        shutil.copyfile(f, md / f.name)
    for n in present:
        os.symlink(os.path.realpath(snap / n), md / n)
    (md / MARKER).write_text(f"{repo}@{rev}", encoding="utf-8")
    print(f"prepare_model_dir: {md} holds the base config and {len(present)} files of {repo}@{rev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_package_runsh.py`
Expected: PASS (16 passed).

- [ ] **Step 5: Mutation checks**

1. MODEL_WEIGHTS_DIR is written: in `orchard/package.py`, replace `"\n".join(WIRING_LINES), text, "export HF_MODEL= line"` with `WIRING_LINES[0], text, "export HF_MODEL= line"`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_runsh.py::test_wiring_points_every_weights_setting_at_the_model_dir`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
2. Run.sh may not name the nearest model: in `orchard/package.py`, replace `if nearest_model in text:` with `if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_runsh.py::test_a_script_that_names_the_base_model_has_a_wiring_problem`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
3. A model-dir the script did not build is left alone: in `orchard/package_templates/prepare_model_dir.py`, replace `if md.is_symlink() or (md.exists() and not (md / MARKER).is_file()):` with `if md.is_symlink():`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_runsh.py::test_a_model_dir_the_script_did_not_build_is_left_alone`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
4. The weights revision must be pinned: in `orchard/package_templates/prepare_model_dir.py`, replace `if not re.fullmatch(r"[0-9a-f]{40}", rev):` with `if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_runsh.py::test_an_unpinned_weights_revision_is_refused`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: 1386 passed and 1 skipped (the skip is the opt-in replay).

```bash
git add orchard/package.py orchard/package_templates/prepare_model_dir.py tests/test_package_runsh.py
git commit -m "Wire a staged run.sh to the new weights through a model-dir it builds at start"
```

---

### Task 6: Read the run, find the source bundles, run package-thin with --out only

**Files:**
- Modify: `orchard/package.py` (append)
- Create: `tests/fake_package_thin.py`, `tests/package_fakes.py`
- Test: `tests/test_package_stage.py`

**Interfaces:**
- Consumes: Task 5's run.sh functions; `orchard.stages.gate_weights_swap`; `orchard.package_card.read_license`.
- Produces in `orchard.package`: `sha256(path) -> str`; `Source(path, manifest)` with `name`, `chips`, `weights_repo`, `entry_cls`, `wheels()`; `load_source(path) -> Source`; `find_sources(models_root, *, entry_cls, nearest_model) -> list[Source]`; `RunFacts(run_dir, model_id, revision, nearest_model, new_snapshot, hf_home, source, base_config, passing_chips, license_id)`; `read_run(run_dir) -> RunFacts`; `Profile(chips, source, required)`; `plan_profiles(facts, others) -> tuple[list[Profile], list[dict]]`; `bundle_name(model_id, source_name) -> str`; `THIN_VALUE_OPTIONS`; `assert_no_publish(argv)`; `thin_argv(source, *, model_id, revision, name, out) -> list[str]`; `run_logged(argv, *, log, timeout, env=None) -> int | None`; `stage_profile(profile, facts, out, *, env=None) -> dict` (keys chips, name, required, source, source_manifest_sha256, mesh).
- Produces in tests: `fake_package_thin.py` (a `tt-model` that records argv in `$FAKE_TT_MODEL_LOG` and writes a minimal v6 bundle for `package-thin --out`), `package_fakes.py` (`make_source`, `make_run`, `hf_snapshot`, `fake_bin`, `calls`, `write`, `tokenizer_json`, constants `NEW`, `BASE`, `NEW_REV`, `BASE_REV`, `DRAFTER`, `DRAFTER_REV`, `SOURCE_ENV`, `SOURCE_EXTRA`, `VOCAB`, `PROMPT_IDS`, `GENERATED`, `HAVE_TOKENIZERS`).

The package reuses everything the nearest model's installed v6 bundle (the source bundle) already serves with: model.py, requirements, wheels, environment (including the pinned-memory fix `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0` and the drafter in `DFLASH_WEIGHTS`), resources and fixed vLLM arguments. Stage 2's `swap_config.json` names the bundle that served the new weights; that bundle gives the required profile. Other installed v6 bundles of the nearest model give optional profiles when stage 4 passed their chip count. `assert_no_publish` is the guard at the layer that could upload: package-thin pushes when given a positional repo id, and `--public` and `--publish` change visibility or list a repo, so only known options with values and `--out` pass. `run_logged` runs supervisor commands in their own session and kills the session on a timeout, a signal or an error.

- [ ] **Step 1: Write the failing tests**

Create `tests/fake_package_thin.py`:

```python
"""A stand-in for the `tt-model` CLI in the stage 7 tests. It opens no device and uploads nothing.

Every call appends its arguments as one JSON line to $FAKE_TT_MODEL_LOG. Only
`package-thin ... --out DIR` does anything: it writes a minimal v6 bundle shaped like the real one
(tt_kernel_manifest.json with the build host's name in producer.hostname, run.sh with the generated
command layout, an install.sh that is not executable, the wheels it was given, model.py,
requirements.txt and vllm_models/<name>/vllm_metadata.json). Any other subcommand exits 1.

install.sh makes venv/bin/python a small wrapper, using three variables it reads when it runs (so
the staged install.sh names no path on this machine): $FAKE_TT_MODEL_PYTHON, $FAKE_SWAP_SERVER and
$FAKE_SWAP_CONFIG. Run as `python -m vllm.entrypoints.openai.api_server`, the wrapper execs the fake
server with that config, so the server keeps run.sh's pid and environment; run any other way, it
execs the real interpreter.
$FAKE_TT_MODEL_FAIL makes package-thin print an error and exit 3.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
from pathlib import Path

FAKE_SERVER = Path(__file__).resolve().with_name("fake_swap_server.py")   # for FAKE_SWAP_SERVER


def options(argv):
    opts, positional, i = {}, [], 0
    while i < len(argv):
        if argv[i].startswith("--"):
            opts.setdefault(argv[i], []).append(argv[i + 1])
            i += 2
        else:
            positional.append(argv[i])
            i += 1
    return opts, positional


RUN_SH = """#!/usr/bin/env bash
# Serve this model on TT hardware. Assumes ./install.sh has been run.
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
VENV="${{VENV:-$HERE/venv}}"
PYBIN="$VENV/bin/python"
export PYTHONPATH="$HERE:${{PYTHONPATH:-}}"
export HF_HOME="${{HF_HOME:-$HERE/.hf}}"
export TT_CACHE_PATH="${{TT_CACHE_PATH:-$HERE/.tt_cache}}"
export TT_CACHE_HOME="${{TT_CACHE_HOME:-$HERE/.tt_cache}}"
export HF_MODEL="${{HF_MODEL:-{weights}}}"
export TT_MODEL_WEIGHTS_REVISION="${{TT_MODEL_WEIGHTS_REVISION:-{rev}}}"
{exports}
CMD=("$PYBIN" -m vllm.entrypoints.openai.api_server --model "{weights}" --max_num_seqs {seqs} --block_size {block} --revision {rev} --tokenizer-revision {rev} --max_model_len {ctx} "$@")
if [ "${{TT_MODEL_PRINT:-0}}" = "1" ]; then
  printf 'HF_MODEL=%s\\n  %s\\n' "${{HF_MODEL:-}}" "${{CMD[*]}}"
  exit 0
fi
exec "${{CMD[@]}}"
"""

INSTALL_SH = """#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/venv/bin"
cat > "$HERE/venv/bin/python" <<WRAP
#!/bin/sh
if [ "\\$1" = "-m" ] && [ "\\$2" = "vllm.entrypoints.openai.api_server" ]; then
  shift 2
  exec "${FAKE_TT_MODEL_PYTHON:?}" "${FAKE_SWAP_SERVER:?}" --config "${FAKE_SWAP_CONFIG:?}" "\\$@"
fi
exec "${FAKE_TT_MODEL_PYTHON:?}" "\\$@"
WRAP
chmod +x "$HERE/venv/bin/python"
echo "installed into $HERE/venv"
"""


def package_thin(opts, positional) -> int:
    if os.environ.get("FAKE_TT_MODEL_FAIL"):
        print("fake tt-model: SFPI 7.83.0 does not match the wheel's 7.73.0", file=sys.stderr)
        return 3
    out = Path(opts["--out"][0])
    out.mkdir(parents=True)
    name, weights, rev = opts["--name"][0], opts["--weights"][0], opts["--weights-revision"][0]
    env = dict(e.split("=", 1) for e in opts.get("--env", []))
    (out / "wheels").mkdir()
    plugin = [shutil.copy(w, out / "wheels") for w in opts.get("--plugin-wheel", [])]
    ops = [shutil.copy(w, out / "wheels") for w in opts.get("--ops-wheel", [])]
    models = [shutil.copy(w, out / "wheels") for w in opts.get("--models-wheel", [])]
    shutil.copy(opts["--model-py"][0], out / "model.py")
    shutil.copy(opts["--requirements"][0], out / "requirements.txt")
    meta = json.loads(Path(opts["--metadata"][0]).read_text())
    (out / "vllm_models" / name).mkdir(parents=True)
    (out / "vllm_models" / name / "vllm_metadata.json").write_text(json.dumps(meta, indent=2))
    rel = lambda paths: [f"wheels/{Path(p).name}" for p in paths]        # noqa: E731
    manifest = {
        "schema_version": "6", "name": name, "arch": opts["--arch"][0],
        "device_count": int(opts["--device-count"][0]),
        "producer": {"tt_kernel_version": "0.1.0", "created_at": "2026-10-03T00:00:00+00:00",
                     "hostname": socket.gethostname()},
        "weights": {"repo_id": weights, "revision": rev, "allow_patterns": None,
                    "ignore_patterns": None, "repo_type": "model"},
        "mesh": {"devices": int(opts["--device-count"][0]), "topology": opts["--mesh"][0], "fabric": None},
        "entrypoint": {"cls": meta["main_class"], "arch_name": meta["arch"]},
        "resources": {"max_model_len": int(opts["--max-model-len"][0]),
                      "max_num_seqs": int(opts["--max-num-seqs"][0]),
                      "block_size": int(opts["--block-size"][0]), "extra_args": []},
        "env": env,
        "deps": {"python": opts.get("--python", ["3.12"])[0], "requirements": "requirements.txt",
                 "wheels": rel(plugin + ops), "wheels_dir": "wheels", "models_wheels": rel(models),
                 "vllm": {"version": opts["--vllm-version"][0], "target_device": "empty"},
                 "model_dir": ".", "kind": "vllm"},
    }
    (out / "tt_kernel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    exports = "\n".join(f'export {k}="{v}"' for k, v in env.items())
    (out / "run.sh").write_text(RUN_SH.format(
        weights=weights, rev=rev, exports=exports, seqs=opts["--max-num-seqs"][0],
        block=opts["--block-size"][0], ctx=opts["--max-model-len"][0]))
    (out / "install.sh").write_text(INSTALL_SH)
    for f in ("run.sh", "install.sh"):
        (out / f).chmod(0o600)              # the real package-thin leaves both without +x
    print(f"staged v6 thin bundle {name} at {out}")
    return 0


def main(argv) -> int:
    log = os.environ.get("FAKE_TT_MODEL_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(argv) + "\n")
    if argv[:1] == ["package-thin"]:
        return package_thin(*options(argv[1:]))
    print(f"fake tt-model: {argv[:1]} is not faked", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

Create `tests/package_fakes.py`:

```python
"""Fakes for the stage 7 tests: an installed source bundle, a finished run and its HF caches.

`make_source` writes a v6 thin bundle of the nearest model in the layout `tt-model` installs, with a
run.sh whose command carries fixed extra vLLM arguments, as the operator's hand-made bundles do.
`make_run` writes a run directory whose stages 0, 1, 2, 4 and 6 passed on the weights-only path,
plus two Hugging Face caches: the run's (holding the new model) and the operator's (holding the
drafter and the base model). `fake_bin` puts tests/fake_package_thin.py on PATH as `tt-model`.
Nothing here opens a device or reaches the network.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import fake_package_thin

NEW, BASE = "Altworld/Hemmingway-1", "Qwen/Qwen3.8-27B"
NEW_REV = "1a5f363a3dd2d1cc456c28b8abbb403b9555efaf"
BASE_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DRAFTER, DRAFTER_REV = "incoai/Qwen3.8-27B-DFlash2", "dedf8df68adfb1afeaf7b7480c0a0243108177b4"
ENTRY = "models.demos.blackhole.qwen36.tt.qwen36_vllm_dflash:Qwen36DFlashForCausalLM"
SOURCE_EXTRA = ("--additional-config '{\"tt\": {\"l1_small_size\": 24576}}' "
                "--max-num-batched-tokens 65536 --no-async-scheduling")
SOURCE_ENV = {"ARCH_NAME": "blackhole", "DFLASH_WEIGHTS": f"{DRAFTER}@{DRAFTER_REV}",
              "TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES": "0"}
VOCAB = [a + b for a in ("ba", "de", "ki", "lo", "mu", "ra", "so", "tu") for b in ("n", "l", "r", "s", "t")]
PROMPT_IDS = [1, 2, 3, 4, 5]
GENERATED = list(range(2, 34))
HAVE_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None


def tokenizer_json() -> str:
    """A WordLevel tokenizer over VOCAB when `tokenizers` is importable, else a stub."""
    if not HAVE_TOKENIZERS:
        return "{}"
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tok = Tokenizer(WordLevel({w: i for i, w in enumerate(VOCAB)} | {"[UNK]": len(VOCAB)},
                              unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return tok.to_str()


def make_source(root: Path, *, name="qwen3.8-27b-dflash2-p300", chips=2, mesh="P150x2",
                weights=BASE, env=None) -> Path:
    """An installed v6 bundle at root/<org>/<name>, as `tt-model` lays it out."""
    b = root / "episod" / name
    (b / "wheels").mkdir(parents=True)
    (b / "wheels" / "vllm_tt_plugin-0.1.0-py3-none-any.whl").write_bytes(b"PK plugin wheel")
    (b / "wheels" / "ttnn-0.79.0-cp312-cp312-linux_x86_64.whl").write_bytes(b"PK ttnn wheel")
    (b / "model.py").write_text(f"from {ENTRY.split(':')[0]} import {ENTRY.split(':')[1]}  # noqa\n")
    (b / "requirements.txt").write_text("ttnn==0.79.0\n")
    (b / "vllm_models" / name).mkdir(parents=True)
    (b / "vllm_models" / name / "vllm_metadata.json").write_text(json.dumps(
        {"arch": "Qwen3_5ForConditionalGeneration", "main_class": ENTRY}))
    env = dict(SOURCE_ENV if env is None else env)
    manifest = {
        "schema_version": "6", "name": name, "arch": "blackhole", "device_count": chips,
        "producer": {"hostname": "redacted"},
        "weights": {"repo_id": weights, "revision": BASE_REV},
        "mesh": {"devices": chips, "topology": mesh, "fabric": None},
        "entrypoint": {"cls": ENTRY, "arch_name": "Qwen3_5ForConditionalGeneration"},
        "resources": {"max_model_len": 262144, "max_num_seqs": 4, "block_size": 64, "extra_args": []},
        "env": env,
        "deps": {"python": "3.12", "requirements": "requirements.txt",
                 "wheels": ["wheels/vllm_tt_plugin-0.1.0-py3-none-any.whl"], "wheels_dir": "wheels",
                 "models_wheels": ["wheels/ttnn-0.79.0-cp312-cp312-linux_x86_64.whl"],
                 "vllm": {"version": "0.26.0", "target_device": "empty"}, "kind": "vllm"}}
    (b / "tt_kernel_manifest.json").write_text(json.dumps(manifest, indent=2))
    run_sh = fake_package_thin.RUN_SH.format(
        weights=weights, rev=BASE_REV, seqs=4, block=64, ctx=262144,
        exports="\n".join(f'export {k}="{v}"' for k, v in env.items()))
    (b / "run.sh").write_text(run_sh.replace(' "$@")', f' {SOURCE_EXTRA} "$@")'))
    return b


def hf_snapshot(hf_home: Path, repo: str, rev: str, files: dict) -> Path:
    """An HF-cache-shaped snapshot under hf_home/hub: files are relative links into blobs/."""
    org, name = repo.split("/")
    root = hf_home / "hub" / f"models--{org}--{name}"
    blobs, snap = root / "blobs", root / "snapshots" / rev
    blobs.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True)
    for fname, content in files.items():
        (blobs / f"b-{fname}").write_text(content)
        (snap / fname).symlink_to(os.path.relpath(blobs / f"b-{fname}", snap))
    return snap


def write(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data, indent=2))
    return path


def make_run(tmp: Path, source: Path, *, license_id="cc-by-nc-4.0") -> dict:
    """A run whose stages 0, 1, 2, 4 and 6 passed on the weights-only path, and its two HF caches."""
    run, hf_run, hf_op = tmp / "run", tmp / "hf-run", tmp / "hf-operator"
    front = f"---\nlicense: {license_id}\nbase_model:\n- {BASE}\n---\n" if license_id else ""
    snap = hf_snapshot(hf_run, NEW, NEW_REV, {
        "README.md": front + "# Hemmingway-1\n", "config.json": '{"architectures": ["Qwen3_5ForCausalLM"]}',
        "tokenizer.json": tokenizer_json(), "model-00001-of-00001.safetensors": "new weights"})
    hf_snapshot(hf_op, DRAFTER, DRAFTER_REV, {"model.safetensors": "drafter"})
    hf_snapshot(hf_op, BASE, BASE_REV, {"model-00001-of-00001.safetensors": "base weights"})
    write(run / "stages/0/delta.json", {"model": NEW, "nearest_model": BASE, "path": "weights-only"})
    ref = run / "stages/1/evidence/reference"
    write(ref / "prompt-ids.json", {"prompt_ids": PROMPT_IDS})
    write(ref / "generated-ids.json", {"generated_ids": GENERATED,
                                       "generated_text": " ".join(VOCAB[i] for i in GENERATED)})
    s2 = run / "stages/2"
    write(s2 / "evidence/swap-check.json", {"top1_agreement": 0.94})
    write(s2 / "evidence/server.log", "ready\n")
    write(s2 / "result.json", {"serves": True, "server_ready_s": 280.5, "coherent": True,
                               "free_run_text": "x", "top1_agreement": 0.94, "n_tokens": 32,
                               "cache_dir": "c", "evidence": ["stages/2/evidence/swap-check.json",
                                                              "stages/2/evidence/server.log"]})
    write(s2 / "swap_config.json", {"run_dir": str(run), "bundle_dir": str(source),
                                    "nearest_model_id": BASE, "new_snapshot": str(snap),
                                    "new_model_id": NEW, "hf_home": str(hf_op), "port": 8100})
    write(s2 / "model-dir/config.json", '{"architectures": ["Qwen3_5ForConditionalGeneration"]}')
    write(s2 / "model-dir/preprocessor_config.json", "{}")
    (s2 / "model-dir/tokenizer.json").symlink_to(snap / "tokenizer.json")
    write(run / "stages/4/evidence/hw-test-output.txt", "ok\n")
    write(run / "stages/4/result.json", {"configs": [
        {"chips": 2, "pass": True, "evidence": ["stages/4/evidence/hw-test-output.txt"]},
        {"chips": 1, "pass": True, "evidence": ["stages/4/evidence/hw-test-output.txt"]},
        {"chips": 4, "pass": True, "evidence": ["stages/4/evidence/hw-test-output.txt"]}]})
    write(run / "stages/6/evidence/bench.txt", "decode 80.0 tok/s/user\n")
    write(run / "stages/6/result.json", {"numbers": [
        {"name": "decode", "value": 80.0, "unit": "tok/s/user", "label": "measured",
         "evidence": ["stages/6/evidence/bench.txt"]},
        {"name": "ttft", "value": None, "unit": "ms", "label": "TODO"}],
        "qualitative": {"evidence": ["stages/6/evidence/bench.txt"]}})
    return {"run": run, "hf_run": hf_run, "hf_op": hf_op, "snapshot": snap}


def fake_bin(tmp: Path) -> tuple[Path, Path]:
    """A directory holding `tt-model` (tests/fake_package_thin.py) and the log it writes."""
    d = tmp / "fake-bin"
    d.mkdir()
    log = tmp / "tt-model-calls.jsonl"
    (d / "tt-model").write_text(
        f'#!/bin/sh\nFAKE_TT_MODEL_LOG="{log}" exec "{sys.executable}" "{fake_package_thin.__file__}" "$@"\n')
    (d / "tt-model").chmod(0o755)
    return d, log


def calls(log: Path) -> list[list[str]]:
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

```

Create `tests/test_package_stage.py`:

```python
"""Stage 7, part 1: read the run, find the source bundles, run package-thin with --out only (plan 5)."""
import json
import os
import re
import socket
import subprocess
import time

import pytest

from orchard.package import (PackageError, Profile, assert_no_publish, bundle_name, find_sources,
                             load_source, plan_profiles, read_run, run_logged, stage_profile,
                             thin_argv, weights_wiring_problems)
from orchard.scrub import scrub_package
from package_fakes import (BASE, NEW, NEW_REV, SOURCE_EXTRA, calls, fake_bin, make_run, make_source,
                           write)

pytestmark = pytest.mark.usefixtures("stub_tools")


@pytest.fixture
def world(tmp_path, monkeypatch, stub_tools):
    """Installed bundles (2-chip and 1-chip of the base model, one of another model), a finished
    run, and the fake tt-model first on PATH, ahead of the stubs."""
    models = tmp_path / "models"
    src2 = make_source(models)
    src1 = make_source(models, name="qwen3.8-27b-dflash2-p150", chips=1, mesh="P150")
    make_source(models, name="other-model-p300", weights="Other/Model")
    r = make_run(tmp_path, src2)
    fb, log = fake_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    return {**r, "models": models, "src2": src2, "src1": src1, "log": log, "stubs": stub_tools,
            "tmp": tmp_path}


def test_the_run_facts_come_from_stages_0_2_and_4(world):
    f = read_run(world["run"])
    assert (f.model_id, f.revision, f.nearest_model) == (NEW, NEW_REV, BASE)
    assert f.license_id == "cc-by-nc-4.0"
    assert f.passing_chips == frozenset({1, 2, 4})
    assert f.source.path == world["src2"] and f.source.chips == 2
    assert [p.name for p in f.base_config] == ["config.json", "preprocessor_config.json"]
    assert f.hf_home == world["hf_op"]


@pytest.mark.parametrize("breakage, words", [
    (lambda w: write(w["run"] / "stages/0/delta.json", {"model": NEW, "nearest_model": BASE,
                                                         "path": "full-port"}), "weights-only"),
    (lambda w: write(w["run"] / "stages/2/result.json", {**json.loads(
        (w["run"] / "stages/2/result.json").read_text()), "top1_agreement": 0.5}), "top1_agreement"),
    (lambda w: (w["snapshot"] / "README.md").unlink(), "license"),
    (lambda w: (w["run"] / "stages/2/model-dir/config.json").unlink(), "config.json"),
    (lambda w: write(w["src2"] / "tt_kernel_manifest.json", {**json.loads(
        (w["src2"] / "tt_kernel_manifest.json").read_text()),
        "weights": {"repo_id": "Other/Model", "revision": "0" * 40}}), "Other/Model"),
])
def test_a_run_stage_7_cannot_package_is_refused_with_the_reason(world, breakage, words):
    breakage(world)
    with pytest.raises(PackageError, match=words):
        read_run(world["run"])


def test_only_installed_v6_bundles_of_the_nearest_model_are_sources(world):
    container = world["models"] / "mando" / "x-p300x2"
    write(container / "tt_kernel_manifest.json", {"schema_version": "5.1", "name": "x-p300x2"})
    found = find_sources(world["models"], entry_cls=load_source(world["src2"]).entry_cls,
                         nearest_model=BASE)
    assert [(s.name, s.chips) for s in found] == [("qwen3.8-27b-dflash2-p150", 1),
                                                  ("qwen3.8-27b-dflash2-p300", 2)]


def test_profiles_are_the_stage_2_bundle_plus_others_stage_4_passed(world):
    f = read_run(world["run"])
    profiles, skipped = plan_profiles(f, [load_source(world["src1"]), f.source])
    assert [(p.chips, p.required, p.source.name) for p in profiles] == [
        (1, False, "qwen3.8-27b-dflash2-p150"), (2, True, "qwen3.8-27b-dflash2-p300")]
    assert skipped == [{"chips": 4, "reason": f"no v6 bundle of {BASE} for 4 chips is installed"}]


def test_the_required_profile_needs_a_passing_stage_4_configuration(world):
    write(world["run"] / "stages/4/result.json", {"configs": [{"chips": 1, "pass": True}]})
    with pytest.raises(PackageError, match="no passing 2-chip"):
        plan_profiles(read_run(world["run"]), [])


def test_the_package_thin_call_names_the_new_weights_and_stages_only(world, tmp_path):
    src = load_source(world["src2"])
    argv = thin_argv(src, model_id=NEW, revision=NEW_REV, name="hemmingway-1-p300", out=tmp_path / "o")
    assert argv[:2] == ["tt-model", "package-thin"] and argv[-2:] == ["--out", str(tmp_path / "o")]
    assert argv[argv.index("--weights") + 1] == NEW and argv[argv.index("--weights-revision") + 1] == NEW_REV
    assert "--env" in argv and f"DFLASH_WEIGHTS={src.manifest['env']['DFLASH_WEIGHTS']}" in argv
    assert_no_publish(argv)
    assert bundle_name(NEW, src.name) == "hemmingway-1-p300"


@pytest.mark.parametrize("change, words", [
    (lambda a: a[:2] + ["episod/hemmingway-1-p300"] + a[2:], "positional"),
    (lambda a: a + ["--public"], "--public"),
    (lambda a: a + ["--publish"], "--publish"),
    (lambda a: ["tt-model", "push"] + a[2:], "only `tt-model package-thin`"),
    (lambda a: a[:-2], "--out"),
    (lambda a: a + ["--name"], "--name"),
])
def test_any_call_that_could_upload_is_refused(world, tmp_path, change, words):
    argv = thin_argv(load_source(world["src2"]), model_id=NEW, revision=NEW_REV, name="n",
                     out=tmp_path / "o")
    with pytest.raises(PackageError, match=words):
        assert_no_publish(change(argv))


def staged(world, profile_source=None, required=True):
    f = read_run(world["run"])
    src = profile_source or f.source
    out = world["run"] / "stages/7/package" / bundle_name(NEW, src.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    return f, stage_profile(Profile(src.chips, src, required), f, out), out


def test_a_staged_profile_is_wired_scrubbed_and_built_by_one_package_thin_call(world):
    f, rec, out = staged(world)
    assert rec == {"chips": 2, "name": "hemmingway-1-p300", "required": True,
                   "source": "qwen3.8-27b-dflash2-p300", "mesh": "P150x2",
                   "source_manifest_sha256": rec["source_manifest_sha256"]}
    text = (out / "run.sh").read_text()
    assert weights_wiring_problems(text, nearest_model=BASE) == []
    assert f' {SOURCE_EXTRA} "$@")' in text
    m = json.loads((out / "tt_kernel_manifest.json").read_text())
    assert m["producer"]["hostname"] == "redacted"
    assert m["weights"]["repo_id"] == NEW and m["weights"]["revision"] == NEW_REV
    assert sorted(p.name for p in (out / "base_config").iterdir()) == ["config.json",
                                                                       "preprocessor_config.json"]
    for name in ("run.sh", "install.sh", "prepare_model_dir.py"):
        assert os.access(out / name, os.X_OK), name
    assert scrub_package(out, hostname=socket.gethostname(), namespace="episod") == []
    # One tt-model call, package-thin with --out; no stub (hf, git, gh, docker, curl ...) ran.
    assert [c[:1] + c[-2:] for c in calls(world["log"])] == [["package-thin", "--out", str(out)]]
    assert world["stubs"].calls() == []


def test_an_optional_profile_is_staged_from_its_own_bundle(world):
    _, rec, out = staged(world, load_source(world["src1"]), required=False)
    assert (rec["chips"], rec["name"], rec["required"]) == (1, "hemmingway-1-p150", False)
    assert json.loads((out / "tt_kernel_manifest.json").read_text())["mesh"]["topology"] == "P150"


def test_a_wheel_that_differs_from_the_source_bundle_is_refused(world, monkeypatch, tmp_path):
    other = tmp_path / "other-ttnn"
    other.mkdir()
    (other / "ttnn-0.79.0-cp312-cp312-linux_x86_64.whl").write_bytes(b"PK a different build")
    src = load_source(world["src2"])
    real = thin_argv

    def swapped(*a, **k):
        argv = real(*a, **k)
        i = argv.index("--models-wheel")
        argv[i + 1] = str(other / "ttnn-0.79.0-cp312-cp312-linux_x86_64.whl")
        return argv
    monkeypatch.setattr("orchard.package.thin_argv", swapped)
    with pytest.raises(PackageError, match="not byte-identical"):
        staged(world, src)


def test_a_failed_package_thin_is_reported_with_its_output(world, monkeypatch):
    monkeypatch.setenv("FAKE_TT_MODEL_FAIL", "1")
    with pytest.raises(PackageError, match="exited 3(.|\n)*SFPI"):
        staged(world)


def test_run_logged_kills_the_whole_session_on_a_timeout(tmp_path):
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "slow.sh"
    script.write_text(f"#!/bin/sh\nsleep 60 &\necho $! > {pid_file}\nwait\n")
    script.chmod(0o755)
    t0 = time.monotonic()
    assert run_logged([script], log=tmp_path / "slow.log", timeout=1) is None
    assert time.monotonic() - t0 < 10
    child = int(pid_file.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


def test_run_logged_kills_the_session_when_interrupted(tmp_path, monkeypatch):
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "slow.sh"
    script.write_text(f"#!/bin/sh\nsleep 60 &\necho $! > {pid_file}\nwait\n")
    script.chmod(0o755)
    real_wait = subprocess.Popen.wait

    def interrupted(self, timeout=None):
        if timeout is not None:
            while not pid_file.exists():
                time.sleep(0.05)
            raise KeyboardInterrupt
        return real_wait(self)
    monkeypatch.setattr(subprocess.Popen, "wait", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_logged([script], log=tmp_path / "slow.log", timeout=30)
    monkeypatch.undo()
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_the_packaging_module_has_no_upload_code():
    # The only external program package.py starts on its own is the argv that assert_no_publish
    # accepts; it imports no hub client and names no upload command outside the publish text.
    import orchard.package as pkg
    src = open(pkg.__file__, encoding="utf-8").read()
    assert not re.search(r"^\s*(?:import|from)\s+(?:huggingface_hub|tt_kernel)\b", src, re.M)
    assert "subprocess.run(" not in src and "os.system" not in src
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_package_stage.py`
Expected: FAIL with a collection error, `ImportError: cannot import name 'Profile' from 'orchard.package'`.

- [ ] **Step 3: Implement**

Append to `orchard/package.py`:

```python


# ---- the run's results and the source bundles ---------------------------------------------------

def _json(path: Path, what: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageError(f"{what} ({path}) is missing or not JSON: {exc}") from None
    if not isinstance(data, dict):
        raise PackageError(f"{what} ({path}) is not a JSON object")
    return data


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Source:
    """An installed v6 thin bundle of the nearest model: the model code, wheels and settings the
    package reuses."""
    path: Path
    manifest: dict

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def chips(self) -> int:
        return int(self.manifest["device_count"])

    @property
    def weights_repo(self) -> str:
        return self.manifest["weights"]["repo_id"]

    @property
    def entry_cls(self) -> str:
        return self.manifest["entrypoint"]["cls"]

    def wheels(self) -> list[str]:
        deps = self.manifest["deps"]
        return list(deps.get("wheels") or []) + list(deps.get("models_wheels") or [])


def load_source(path) -> Source:
    path = Path(path)
    m = _json(path / "tt_kernel_manifest.json", "the source bundle's manifest")
    deps = m.get("deps") or {}
    if m.get("schema_version") != "6" or deps.get("kind") != "vllm":
        raise PackageError(f"{path} is not a v6 thin vLLM bundle")
    need = ["run.sh", "model.py", deps.get("requirements") or "requirements.txt",
            f"vllm_models/{m.get('name')}/vllm_metadata.json", *(deps.get("wheels") or []),
            *(deps.get("models_wheels") or [])]
    missing = [rel for rel in need if not (path / rel).is_file()]
    if missing:
        raise PackageError(f"the source bundle {path} lacks {missing}")
    return Source(path, m)


def find_sources(models_root, *, entry_cls: str, nearest_model: str) -> list[Source]:
    """Every installed v6 bundle under `models_root` that serves `nearest_model` with the model
    class `entry_cls`, fewest chips first. A directory that is not such a bundle is skipped."""
    found = []
    for mf in sorted(Path(models_root).expanduser().glob("*/*/tt_kernel_manifest.json")):
        try:
            s = load_source(mf.parent)
            if s.entry_cls == entry_cls and s.weights_repo == nearest_model:
                found.append(s)
        except (PackageError, KeyError, TypeError, ValueError):
            continue
    return sorted(found, key=lambda s: (s.chips, s.name))


@dataclass(frozen=True)
class RunFacts:
    run_dir: Path
    model_id: str
    revision: str
    nearest_model: str
    new_snapshot: Path
    hf_home: Path                    # the HF home stage 2 served with (it holds the drafter)
    source: Source                   # the bundle stage 2 served the new weights with
    base_config: tuple[Path, ...]    # the nearest model's config files stage 2 used
    passing_chips: frozenset[int]    # chip counts with a passing stage 4 configuration
    license_id: str


def read_run(run_dir) -> RunFacts:
    """What stage 7 needs from stages 0, 2 and 4. Raises PackageError naming what is missing."""
    run = Path(run_dir).resolve()
    delta = _json(run / "stages/0/delta.json", "stage 0's delta")
    if delta.get("path") != "weights-only":
        raise PackageError(f"stage 7 packages weights-only runs; stage 0 chose {delta.get('path')!r}")
    gate = gate_weights_swap(run / "stages/2", run)
    if not gate.ok:
        raise PackageError("stage 2's result does not pass its gate: " + "; ".join(gate.reasons))
    cfg = _json(run / "stages/2/swap_config.json", "stage 2's swap_config.json")
    model_id = delta.get("model")
    if cfg.get("new_model_id") != model_id:
        raise PackageError(f"stage 2 served {cfg.get('new_model_id')!r}; stage 0 names {model_id!r}")
    snap = Path(cfg.get("new_snapshot") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", snap.name) or not snap.is_dir():
        raise PackageError(f"stage 2's new_snapshot {str(snap)!r} is not a pinned snapshot directory")
    source = load_source(cfg.get("bundle_dir") or "")
    if source.weights_repo != delta.get("nearest_model"):
        raise PackageError(f"stage 2's bundle serves {source.weights_repo}; the nearest model is "
                           f"{delta.get('nearest_model')}")
    md = run / "stages/2/model-dir"
    base = tuple(md / n for n in BASE_CONFIG_FILES if (md / n).is_file() and not (md / n).is_symlink())
    if md / "config.json" not in base:
        raise PackageError(f"{md}/config.json (the nearest model's config) is missing")
    configs = _json(run / "stages/4/result.json", "stage 4's result").get("configs") or []
    passing = frozenset(c["chips"] for c in configs if isinstance(c, dict) and c.get("pass") is True
                        and isinstance(c.get("chips"), int) and not isinstance(c.get("chips"), bool))
    license_id = read_license(snap)
    if not license_id:
        raise PackageError(f"the new model's license is not in {snap}/README.md; stage 7 stages no "
                           "package without it")
    return RunFacts(run, model_id, snap.name, delta["nearest_model"], snap,
                    Path(cfg.get("hf_home") or ""), source, base, passing, license_id)


@dataclass(frozen=True)
class Profile:
    chips: int
    source: Source
    required: bool                   # the profile stage 2 checked; stage 7 boots it


def plan_profiles(facts: RunFacts, others: list[Source]) -> tuple[list[Profile], list[dict]]:
    """The profiles to stage, and the chip counts stage 4 passed that get no package, with why.

    The required profile uses stage 2's bundle. Another installed bundle of the nearest model is
    an optional profile when stage 4 passed its chip count."""
    req = facts.source
    if req.chips not in facts.passing_chips:
        raise PackageError(f"stage 4 has no passing {req.chips}-chip configuration")
    profiles, skipped = [Profile(req.chips, req, True)], []
    for s in others:
        if any(p.chips == s.chips for p in profiles):
            continue
        if s.chips in facts.passing_chips:
            profiles.append(Profile(s.chips, s, False))
        else:
            skipped.append({"chips": s.chips, "reason": f"stage 4 has no passing {s.chips}-chip "
                                                         f"configuration (bundle {s.name} exists)"})
    for chips in sorted(facts.passing_chips - {p.chips for p in profiles}):
        skipped.append({"chips": chips, "reason": f"no v6 bundle of {facts.nearest_model} for "
                                                  f"{chips} chips is installed"})
    return sorted(profiles, key=lambda p: p.chips), sorted(skipped, key=lambda s: s["chips"])


def bundle_name(model_id: str, source_name: str) -> str:
    """The new model's name plus the source bundle's board suffix: hemmingway-1-p300."""
    return f"{model_id.split('/')[-1].lower()}-{source_name.rsplit('-', 1)[-1]}"


# ---- tt-model package-thin, with --out only ------------------------------------------------------

THIN_VALUE_OPTIONS = frozenset({
    "--model-py", "--kind", "--app", "--requirements", "--plugin-wheel", "--ops-wheel",
    "--models-wheel", "--vllm-wheel", "--vllm-version", "--arch", "--arch-name", "--main-class",
    "--metadata", "--weights", "--weights-revision", "--mesh", "--device-count", "--python",
    "--tt-metal-version", "--max-num-seqs", "--block-size", "--max-model-len", "--env", "--name",
    "--out"})


def assert_no_publish(argv) -> None:
    """Refuse any tt-model call that could upload or list something.

    `package-thin` pushes when it is given a repo id as a positional argument, and `--public` and
    `--publish` change visibility or list the repo. Stage 7 passes only the options it knows, each
    with a value, and `--out`."""
    argv = [str(a) for a in argv]
    if argv[:2] != ["tt-model", "package-thin"]:
        raise PackageError(f"stage 7 runs only `tt-model package-thin`, not {argv[:2]}")
    if "--out" not in argv:
        raise PackageError("tt-model package-thin needs --out, so it stages and does not push")
    i = 2
    while i < len(argv):
        a = argv[i]
        if a in THIN_VALUE_OPTIONS and i + 1 < len(argv):
            i += 2
        elif a.startswith("-"):
            raise PackageError(f"option {a} is not one stage 7 passes (--public and --publish "
                               "change visibility or list the repo)")
        else:
            raise PackageError(f"{a!r} is a positional argument; package-thin pushes to a repo id "
                               "given that way")


def thin_argv(source: Source, *, model_id: str, revision: str, name: str, out: Path) -> list[str]:
    m, s = source.manifest, source.path
    deps, res = m["deps"], m["resources"]
    plugin = [w for w in deps.get("wheels") or [] if Path(w).name.startswith("vllm_tt_plugin")]
    ops = [w for w in deps.get("wheels") or [] if w not in plugin]
    argv = ["tt-model", "package-thin", "--model-py", s / "model.py", "--kind", "vllm",
            "--requirements", s / (deps.get("requirements") or "requirements.txt"),
            *[x for w in plugin for x in ("--plugin-wheel", s / w)],
            *[x for w in ops for x in ("--ops-wheel", s / w)],
            *[x for w in deps.get("models_wheels") or [] for x in ("--models-wheel", s / w)],
            "--metadata", s / "vllm_models" / m["name"] / "vllm_metadata.json",
            "--weights", model_id, "--weights-revision", revision,
            "--arch", m["arch"], "--mesh", m["mesh"]["topology"], "--device-count", m["device_count"],
            "--python", deps.get("python") or "3.12", "--vllm-version", deps["vllm"]["version"],
            "--max-num-seqs", res["max_num_seqs"], "--block-size", res["block_size"],
            "--max-model-len", res["max_model_len"], "--name", name,
            *[x for k, v in (m.get("env") or {}).items() for x in ("--env", f"{k}={v}")],
            "--out", out]
    return [str(a) for a in argv]


def run_logged(argv, *, log: Path, timeout: float, env: dict | None = None) -> int | None:
    """Run one supervisor command with its output appended to `log`. Returns the exit code, or None
    on a timeout. It runs in its own session, and a timeout, a signal or an error kills the whole
    session, so nothing it started outlives the stage."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as out:
        try:
            proc = subprocess.Popen([str(a) for a in argv], stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        except FileNotFoundError as exc:
            out.write(f"not found: {exc}\n".encode())
            return 127
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_session(proc)
            return None
        except BaseException:
            _kill_session(proc)
            raise


def _kill_session(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _tail(log: Path, n: int = 20) -> str:
    try:
        return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def stage_profile(profile: Profile, facts: RunFacts, out: Path, *, env: dict | None = None) -> dict:
    """Run package-thin for one profile into `out` (which must not exist) and apply the edits."""
    src = profile.source
    name = bundle_name(facts.model_id, src.name)
    out = Path(out)
    if out.exists():
        raise PackageError(f"{out} already exists; stage 7 stages into a fresh directory")
    argv = thin_argv(src, model_id=facts.model_id, revision=facts.revision, name=name, out=out)
    assert_no_publish(argv)
    log = out.parent / f"{name}.package-thin.log"
    rc = run_logged(argv, log=log, timeout=PACKAGE_THIN_TIMEOUT_S, env=env)
    if rc != 0:
        raise PackageError(f"tt-model package-thin for {name} exited {rc}; the end of {log.name}:\n"
                           + _tail(log))
    run_sh = out / "run.sh"
    text = splice_extra_args(run_sh.read_text(encoding="utf-8"),
                             extra_args_from((src.path / "run.sh").read_text(encoding="utf-8")))
    text = wire_weights(text, facts.model_id)
    problems = weights_wiring_problems(text, nearest_model=facts.nearest_model)
    if problems:
        raise PackageError(f"{name}/run.sh: " + "; ".join(problems))
    run_sh.write_text(text, encoding="utf-8")
    mpath = out / "tt_kernel_manifest.json"
    m = _json(mpath, "the staged manifest")
    if m.get("weights", {}).get("repo_id") != facts.model_id or m["weights"].get("revision") != facts.revision:
        raise PackageError(f"the staged manifest's weights are {m.get('weights')}; expected "
                           f"{facts.model_id} at {facts.revision}")
    m.setdefault("producer", {})["hostname"] = "redacted"
    mpath.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    (out / "base_config").mkdir()
    for f in facts.base_config:
        shutil.copyfile(f, out / "base_config" / f.name)
    shutil.copyfile(TEMPLATES / "prepare_model_dir.py", out / "prepare_model_dir.py")
    for f in ("run.sh", "install.sh", "prepare_model_dir.py"):
        (out / f).chmod(0o755)
    for w in sorted((out / "wheels").glob("*.whl")):
        theirs = src.path / "wheels" / w.name
        if not theirs.is_file() or sha256(theirs) != sha256(w):
            raise PackageError(f"{name}/wheels/{w.name} is not byte-identical to the source bundle's")
    return {"chips": profile.chips, "name": name, "required": profile.required, "source": src.name,
            "source_manifest_sha256": sha256(src.path / "tt_kernel_manifest.json"),
            "mesh": m["mesh"]["topology"]}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_package_stage.py`
Expected: PASS (23 passed).

- [ ] **Step 5: Mutation checks**

1. A repo id, --public and --publish are refused: in `orchard/package.py`, replace `if a in THIN_VALUE_OPTIONS and i + 1 < len(argv):` with `if True:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_stage.py::test_any_call_that_could_upload_is_refused`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
2. Wheels must match the source bundle: in `orchard/package.py`, replace `if not theirs.is_file() or sha256(theirs) != sha256(w):` with `if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_stage.py::test_a_wheel_that_differs_from_the_source_bundle_is_refused`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
3. A timeout kills the session: in `orchard/package.py`, replace

```python
            _kill_session(proc)
            return None
```

   with

```python
            return None
```

   Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_stage.py::test_run_logged_kills_the_whole_session_on_a_timeout`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

4. An interrupt kills the session: in `orchard/package.py`, replace

```python
            _kill_session(proc)
            raise
```

   with

```python
            raise
```

   Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_stage.py::test_run_logged_kills_the_session_when_interrupted`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

5. The build host's name is replaced: in `orchard/package.py`, replace

```python
    m.setdefault("producer", {})["hostname"] = "redacted"
```

   with nothing (delete these lines).

   Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_stage.py::test_a_staged_profile_is_wired_scrubbed_and_built_by_one_package_thin_call`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

6. The module imports no hub client: in `orchard/package.py`, replace

```python
from orchard.scrub import scrub_package
```

   with

```python
from orchard.scrub import scrub_package
import huggingface_hub  # noqa
```

   Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_stage.py::test_the_packaging_module_has_no_upload_code`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.


- [ ] **Step 6: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: 1409 passed and 1 skipped (the skip is the opt-in replay).

```bash
git add orchard/package.py tests/fake_package_thin.py tests/package_fakes.py tests/test_package_stage.py
git commit -m "Stage one package profile with tt-model package-thin --out from the source bundle"
```

---

### Task 7: The boot check: an installed copy, served on a leased board

**Files:**
- Modify: `orchard/package.py` (append)
- Create: `orchard/package_templates/verify_bundle.py`
- Modify: `tests/fake_swap_server.py` (`/v1/models`, unknown arguments, the model from `--model`)
- Test: `tests/test_package_verify.py`

**Interfaces:**
- Consumes: `RunFacts`, `run_logged`, `_tail`, `_json` (Task 6); `orchard/skills/weights-swap-templates/serve_and_compare.py` (its `wait_healthy`, `complete`, `coherence`, `stop_server`, `ServerHTTPError`, `N_TOKENS`).
- Produces in `orchard.package`: `AUX_REPO`, `aux_repos(env, *, nearest_model) -> list[str]`, `free_port() -> int`, `prepare_verify(staged, verify_dir, facts, *, env=None) -> dict` (keys command, deadline_s, hf_linked, hf_missing).
- Produces the template `verify_bundle.py`: exit 0 measured (writes `evidence/verify.json` with `top1_agreement`, `coherent`, `n_tokens`, `server_ready_s`, `served_model`, `server_weights_env`, `model_dir_weights`, `evidence`), 2 bad reference, 4 server not healthy or died, 5 HTTP error, 6 port busy, 7 wrong served model, 8 weights environment wrong, 9 model-dir built from other weights.

The boot check tests the package as a consumer gets it. A copy of the staged directory is installed (its own install.sh) and served (its own run.sh), so the staged directory never gains a venv, a model-dir or a cache. The check removes `MODEL_WEIGHTS_DIR` and `HF_MODEL` from its environment, so only the bundle can set them, and gives the server a Hugging Face home that links only the new model and the auxiliary repos the manifest names (the drafter), offline. The nearest model is never linked, so a bundle that tried to load its weights fails. Before comparing a token it proves the answers come from this server: the port was free, `/v1/models` lists the copy's model-dir, the server process's own environment (`/proc/<pid>/environ`; run.sh execs the server) names that model-dir, and `model-dir/.weights` names the manifest's weights. The comparison is stage 2's: 32 teacher-forced tokens against the stage 1 reference, with the same server helpers. Like `serve_and_compare.py`, it needs `tokenizers`, and the supervisor runs it with its own interpreter (`sys.executable`), which has it on this machine.

- [ ] **Step 1: Write the failing tests**

Replace in `tests/fake_swap_server.py`:

```python
max_tokens and temperature (the real one rejects logprobs and sampling parameters), and to a
request whose model is not the expected model directory.

```

with:

```python
max_tokens and temperature (the real one rejects logprobs and sampling parameters), and to a
request whose model is not the expected model directory: the config's "model", or else the
`--model` it was started with. GET /v1/models lists that one model, as vLLM does. Arguments it
does not know (the rest of a vLLM command line) are ignored.

```

Replace in `tests/fake_swap_server.py`:

```python
            self._send(200, b"")
        else:
```

with:

```python
            self._send(200, b"")
        elif self.path == "/v1/models":
            self._send(200, json.dumps({"data": [{"id": self.cfg["model"]}]}).encode())
        else:
```

Replace in `tests/fake_swap_server.py`:

```python
    ap.add_argument("--model")            # passed when the test runs a prepare_swap-built run.sh
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    with open(cfg["pid_file"], "w", encoding="utf-8") as f:
```

with:

```python
    ap.add_argument("--model")            # passed when the test runs a prepare_swap-built run.sh
    args, _ = ap.parse_known_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("model", args.model)
    with open(cfg["pid_file"], "w", encoding="utf-8") as f:
```

Create `tests/test_package_verify.py`:

```python
"""Stage 7, part 2: install a copy of the staged package and boot-check it (plan 5).

The copy's install.sh (from tests/fake_package_thin.py) makes venv/bin/python a wrapper that serves
tests/fake_swap_server.py. Everything else is the staged package as stage 7 wrote it: its run.sh,
prepare_model_dir.py and base_config/. Each test that starts a server kills its process group in
a finalizer, so a failing test leaves nothing behind.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from orchard.package import (PackageError, Profile, aux_repos, bundle_name, prepare_verify,
                             read_run, stage_profile)
from package_fakes import (BASE, DRAFTER, GENERATED, HAVE_TOKENIZERS, NEW, NEW_REV, PROMPT_IDS,
                           VOCAB, fake_bin, make_run, make_source)

if not HAVE_TOKENIZERS:
    pytest.skip("SKIPPED: the `tokenizers` package is not importable, so the stage 7 boot check "
                "did not run. verify_bundle.py needs it.", allow_module_level=True)

FAKE_SERVER = Path(__file__).resolve().with_name("fake_swap_server.py")
pytestmark = pytest.mark.usefixtures("stub_tools")


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.fixture
def boot(tmp_path, monkeypatch):
    """A staged 2-chip package, an installed copy under stages/7/verify, and a runner for the check."""
    models = tmp_path / "models"
    r = make_run(tmp_path, make_source(models))
    fb, _ = fake_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    facts = read_run(r["run"])
    out = r["run"] / "stages/7/package" / bundle_name(NEW, facts.source.name)
    out.parent.mkdir(parents=True)
    stage_profile(Profile(2, facts.source, True), facts, out)
    pid_file, server_cfg = tmp_path / "server-pid.json", tmp_path / "server.json"
    env = dict(os.environ, FAKE_TT_MODEL_PYTHON=sys.executable, FAKE_SWAP_SERVER=str(FAKE_SERVER),
               FAKE_SWAP_CONFIG=str(server_cfg))
    verify = r["run"] / "stages/7/verify"
    test = prepare_verify(out, verify, facts, env=env)
    cfg_path = verify / "verify_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["health_timeout_s"] = 30
    cfg_path.write_text(json.dumps(cfg))

    def check(mode="perfect", env_extra=None, **server):
        server_cfg.write_text(json.dumps({"mode": mode, "vocab": VOCAB, "prompt_ids": PROMPT_IDS,
                                          "generated_ids": GENERATED, "pid_file": str(pid_file),
                                          **server}))
        clean = {k: v for k, v in os.environ.items() if k not in ("MODEL_WEIGHTS_DIR", "HF_MODEL")}
        clean.update(env_extra or {})
        return subprocess.run([sys.executable, str(verify / "verify_bundle.py")], capture_output=True,
                              text=True, timeout=180, env=clean, cwd=r["run"])

    state = {**r, "facts": facts, "out": out, "verify": verify, "test": test, "check": check,
             "pid_file": pid_file, "cfg": cfg}
    try:
        yield state
    finally:
        if pid_file.exists():
            pgid = json.loads(pid_file.read_text())["pgid"]
            if group_alive(pgid):
                os.killpg(pgid, signal.SIGKILL)
                print(f"fixture killed a leaked fake server group {pgid}", file=sys.stderr)


def verify_json(state) -> dict:
    return json.loads((state["verify"] / "evidence" / "verify.json").read_text())


def wait_gone(pgid: int, within: float = 5.0) -> bool:
    end = time.monotonic() + within
    while time.monotonic() < end and group_alive(pgid):
        time.sleep(0.1)
    return not group_alive(pgid)


def test_the_installed_copy_serves_the_new_weights_and_agrees(boot):
    r = boot["check"]("perfect")
    assert r.returncode == 0, r.stdout + r.stderr
    v = verify_json(boot)
    md = str(boot["verify"] / "bundle" / "model-dir")
    assert v["top1_agreement"] == 1.0 and v["coherent"] is True and v["n_tokens"] == 32
    assert v["served_model"] == md
    assert v["server_weights_env"] == {"MODEL_WEIGHTS_DIR": md, "HF_MODEL": md}
    assert v["model_dir_weights"] == f"{NEW}@{NEW_REV}"
    assert v["evidence"] == ["stages/7/verify/evidence/verify.json", "stages/7/verify/evidence/server.log"]
    seen = json.loads(boot["pid_file"].read_text())
    assert seen["env"]["HF_HUB_OFFLINE"] == "1" and seen["model_arg"] == md
    assert seen["env"]["TT_CACHE_PATH"] == str(boot["verify"] / "bundle" / ".tt_cache")
    assert wait_gone(seen["pgid"])
    # The staged package itself was never installed into or served.
    assert not (boot["out"] / "venv").exists() and not (boot["out"] / "model-dir").exists()


def test_the_boot_check_hf_home_holds_the_new_model_and_the_drafter_only(boot):
    hub = boot["verify"] / "hf" / "hub"
    assert sorted(p.name for p in hub.iterdir()) == ["models--Altworld--Hemmingway-1",
                                                     "models--incoai--Qwen3.8-27B-DFlash2"]
    assert boot["test"]["hf_linked"] == [NEW, DRAFTER] and boot["test"]["hf_missing"] == []
    assert (boot["hf_op"] / "hub" / "models--Qwen--Qwen3.8-27B").is_dir()     # present, not linked


def test_the_hardware_test_runs_the_copied_script_with_the_supervisors_python(boot):
    assert boot["test"]["command"] == f"{sys.executable} stages/7/verify/verify_bundle.py"
    assert boot["test"]["deadline_s"] > 0
    assert (boot["verify"] / "serve_and_compare.py").is_file()


def test_low_agreement_is_measured_and_reported(boot):
    r = boot["check"]("every4")
    assert r.returncode == 0, r.stdout + r.stderr
    assert verify_json(boot)["top1_agreement"] == 0.75


def test_a_copy_whose_run_sh_lost_the_weights_directory_is_caught(boot):
    run_sh = boot["verify"] / "bundle" / "run.sh"
    run_sh.write_text(run_sh.read_text().replace('export MODEL_WEIGHTS_DIR="$HERE/model-dir"\n', ""))
    r = boot["check"]("perfect")
    assert r.returncode == 8, r.stdout + r.stderr
    assert "MODEL_WEIGHTS_DIR" in r.stdout
    assert wait_gone(json.loads(boot["pid_file"].read_text())["pgid"])


def test_the_operators_own_weights_variables_cannot_stand_in_for_the_bundles(boot):
    # The check removes MODEL_WEIGHTS_DIR and HF_MODEL from its environment, so only the bundle's
    # run.sh can set them. Here they point at the right place and run.sh does not set one of them.
    md = str(boot["verify"] / "bundle" / "model-dir")
    run_sh = boot["verify"] / "bundle" / "run.sh"
    run_sh.write_text(run_sh.read_text().replace('export MODEL_WEIGHTS_DIR="$HERE/model-dir"\n', ""))
    r = boot["check"]("perfect", env_extra={"MODEL_WEIGHTS_DIR": md, "HF_MODEL": md})
    assert r.returncode == 8, r.stdout + r.stderr


def test_a_port_that_already_answers_is_refused_before_any_server_starts(boot):
    with socket.socket() as s:
        s.bind(("127.0.0.1", boot["cfg"]["port"]))
        s.listen()
        r = boot["check"]("perfect")
    assert r.returncode == 6, r.stdout + r.stderr
    assert not boot["pid_file"].exists()


def test_a_server_that_serves_another_model_is_refused(boot):
    r = boot["check"]("perfect", model="/somewhere/else/model-dir")
    assert r.returncode == 7, r.stdout + r.stderr


def test_a_model_dir_built_from_other_weights_is_refused(boot):
    prep = boot["verify"] / "bundle" / "prepare_model_dir.py"
    prep.write_text(prep.read_text().replace('f"{repo}@{rev}", encoding', '"Other/Model@x", encoding'))
    r = boot["check"]("perfect")
    assert r.returncode == 9, r.stdout + r.stderr


def test_a_failed_install_is_reported_with_its_log(tmp_path, monkeypatch):
    r = make_run(tmp_path, make_source(tmp_path / "models"))
    fb, _ = fake_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    facts = read_run(r["run"])
    out = r["run"] / "stages/7/package/x"
    out.parent.mkdir(parents=True)
    stage_profile(Profile(2, facts.source, True), facts, out)
    env = {k: v for k, v in os.environ.items() if not k.startswith("FAKE_")}
    with pytest.raises(PackageError, match="install.sh(.|\n)*FAKE_TT_MODEL_PYTHON"):
        prepare_verify(out, r["run"] / "stages/7/verify", facts, env=env)


def test_auxiliary_repos_never_include_the_nearest_model():
    env = {"DFLASH_WEIGHTS": f"{DRAFTER}@{'d' * 40}", "OTHER": BASE, "ARCH_NAME": "blackhole",
           "X": "a/b/c", "Y": "1"}
    assert aux_repos(env, nearest_model=BASE) == [DRAFTER]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_package_verify.py tests/test_weights_swap_templates.py`
Expected: FAIL with a collection error, `ImportError: cannot import name 'aux_repos' from 'orchard.package'`.

- [ ] **Step 3: Implement**

Append to `orchard/package.py`:

```python


# ---- the boot check: an installed copy, served on a leased board ---------------------------------

AUX_REPO = re.compile(r"^([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?:@[0-9a-f]{40})?$")


def aux_repos(env: dict, *, nearest_model: str) -> list[str]:
    """Hugging Face repos the bundle's environment names (such as the drafter in DFLASH_WEIGHTS),
    without the nearest model, whose weights the package must never load."""
    found = {m.group(1) for v in env.values() if (m := AUX_REPO.match(str(v)))}
    return sorted(found - {nearest_model})


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _repo_dir(hub: Path, repo: str) -> Path:
    org, name = repo.split("/", 1)
    return hub / f"models--{org}--{name}"


def prepare_verify(staged: Path, verify_dir: Path, facts: RunFacts, *, env: dict | None = None) -> dict:
    """Install a copy of a staged package and lay out its boot check. Returns hw_test.json's content
    plus what was linked. The staged directory itself is never installed into or served."""
    verify_dir = Path(verify_dir)
    verify_dir.mkdir(parents=True)
    bundle = verify_dir / "bundle"
    shutil.copytree(staged, bundle, symlinks=True)
    log = verify_dir / "install.log"
    rc = run_logged(["bash", bundle / "install.sh"], log=log, timeout=PACKAGE_INSTALL_TIMEOUT_S,
                    env=env)
    if rc != 0:
        raise PackageError(f"install.sh of the package copy exited {rc}; the end of install.log:\n"
                           + _tail(log))
    hub = verify_dir / "hf" / "hub"
    hub.mkdir(parents=True)
    _repo_dir(hub, facts.model_id).symlink_to(facts.new_snapshot.parents[1])
    linked, missing = [facts.model_id], []
    manifest = _json(bundle / "tt_kernel_manifest.json", "the staged manifest")
    for repo in aux_repos(manifest.get("env") or {}, nearest_model=facts.nearest_model):
        theirs = _repo_dir(facts.hf_home / "hub", repo)
        if theirs.is_dir():
            _repo_dir(hub, repo).symlink_to(theirs)
            linked.append(repo)
        else:
            missing.append(repo)
    (verify_dir / "verify_config.json").write_text(json.dumps({
        "run_dir": str(facts.run_dir), "bundle": str(bundle), "port": free_port(),
        "model_id": facts.model_id, "revision": facts.revision, "hf_home": str(hub.parent),
        "health_timeout_s": PACKAGE_HEALTH_TIMEOUT_S}, indent=2))
    shutil.copyfile(TEMPLATES / "verify_bundle.py", verify_dir / "verify_bundle.py")
    shutil.copyfile(SWAP_TEMPLATES / "serve_and_compare.py", verify_dir / "serve_and_compare.py")
    script = os.path.relpath(verify_dir / "verify_bundle.py", facts.run_dir)
    return {"command": f"{shlex.quote(sys.executable)} {script}",
            "deadline_s": PACKAGE_VERIFY_DEADLINE_S, "hf_linked": linked, "hf_missing": missing}
```

Create `orchard/package_templates/verify_bundle.py`:

```python
#!/usr/bin/env python3
"""Boot an installed copy of the staged package and compare its tokens with the CPU reference.

Stage 7 copies this file and serve_and_compare.py (the weights-swap-check template, whose server
and request helpers it reuses) into stages/7/verify/, next to `verify_config.json`. The supervisor
runs it as the stage's hardware test on a leased board, with that board's chips in
TT_VISIBLE_DEVICES.

It tests the package as a consumer gets it. The server is the copy's own `bash run.sh --port P`.
MODEL_WEIGHTS_DIR, HF_MODEL, TT_CACHE_PATH, TT_CACHE_HOME and HF_HUB_CACHE are removed from the
environment, so the bundle must set the weights directory itself and uses its own empty tensor
cache. HF_HOME is a directory that links only the new model and the auxiliary repos the manifest
names, and HF_HUB_OFFLINE=1, so a bundle that tried to load the nearest model's weights would fail.

Before any token is compared it checks that the answers will come from this server:
- exit 6 if something already answers on the port (the check would read another server);
- exit 7 if /v1/models does not list the copy's model-dir;
- exit 8 if the server process's own environment does not set MODEL_WEIGHTS_DIR and HF_MODEL to
  the copy's model-dir (read from /proc/<pid>/environ; run.sh execs the server, so it keeps the pid);
- exit 9 if model-dir/.weights does not name the manifest's weights and revision.
Then: 32 greedy tokens from the stage 1 prompt, and for each of the 32 reference positions one
token with the reference prefix, re-tokenized and compared with the reference id. Exit 4: the
server exited or never became healthy, or exited during the check. Exit 5: an HTTP error. Exit 0:
measured, whatever the numbers say; evidence/verify.json holds them. The server's process group
is stopped on every exit path.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serve_and_compare as sac  # noqa: E402

SCRUBBED_ENV = ("MODEL_WEIGHTS_DIR", "HF_MODEL", "TT_CACHE_PATH", "TT_CACHE_HOME", "HF_HUB_CACHE")


def port_answers(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def served_models(port: int) -> list[str]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as resp:
        return [m.get("id") for m in json.loads(resp.read().decode("utf-8")).get("data", [])]


def process_env(pid: int) -> dict:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    return dict(kv.split("=", 1) for kv in raw.decode("utf-8", "replace").split("\0") if "=" in kv)


def stop(code: int, message: str) -> None:
    print(f"verify_bundle: {message}")
    sys.exit(code)


def main() -> int:
    from tokenizers import Tokenizer      # imported here so a missing package fails before Popen

    cfg = json.loads((HERE / "verify_config.json").read_text(encoding="utf-8"))
    run_dir, bundle, port = Path(cfg["run_dir"]), Path(cfg["bundle"]), int(cfg["port"])
    model_dir = bundle / "model-dir"
    ref = run_dir / "stages" / "1" / "evidence" / "reference"
    prompt_ids = json.loads((ref / "prompt-ids.json").read_text(encoding="utf-8"))["prompt_ids"]
    generated_ids = json.loads((ref / "generated-ids.json").read_text(encoding="utf-8"))["generated_ids"]
    if len(generated_ids) < sac.N_TOKENS:
        stop(2, f"the reference has {len(generated_ids)} generated ids; {sac.N_TOKENS} are needed")
    if port_answers(port):
        stop(6, f"something already answers on port {port}; refusing, so the check cannot read "
                "another server")
    evidence = HERE / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    log_path = evidence / "server.log"
    env = {k: v for k, v in os.environ.items() if k not in SCRUBBED_ENV}
    env.update(HF_HOME=str(cfg["hf_home"]), HF_HUB_OFFLINE="1")
    log = open(log_path, "wb")
    proc = subprocess.Popen(["bash", str(bundle / "run.sh"), "--port", str(port)], env=env,
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    try:
        ready_s = sac.wait_healthy(proc, port, float(cfg["health_timeout_s"]), log_path)
        models = served_models(port)
        if str(model_dir) not in models:
            stop(7, f"the server on port {port} serves {models}, not {model_dir}")
        seen = process_env(proc.pid)
        weights_env = {k: seen.get(k) for k in ("MODEL_WEIGHTS_DIR", "HF_MODEL")}
        if set(weights_env.values()) != {str(model_dir)}:
            stop(8, f"the server process has {weights_env}; both must be {model_dir}")
        want = f"{cfg['model_id']}@{cfg['revision']}"
        mf = model_dir / ".weights"
        marker = mf.read_text(encoding="utf-8") if mf.is_file() else None
        if marker != want:
            stop(9, f"model-dir/.weights is {marker!r}; expected {want!r}")
        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        free_text = sac.complete(port, str(model_dir), prompt_ids, sac.N_TOKENS)
        matches, mismatches = 0, []
        for k in range(sac.N_TOKENS):
            got = sac.complete(port, str(model_dir), prompt_ids + generated_ids[:k], 1)
            ids = tokenizer.encode(got, add_special_tokens=False).ids
            if ids and ids[0] == generated_ids[k]:
                matches += 1
            else:
                mismatches.append({"position": k, "expected_id": generated_ids[k],
                                   "got_id": ids[0] if ids else None, "got_text": got[:300]})
        if proc.poll() is not None:
            stop(4, f"the server exited with code {proc.returncode} during the check")
    except sac.ServerHTTPError as exc:
        stop(5, f"the server returned an error: {exc}")
    finally:
        sac.stop_server(proc)
        log.close()
    coh = sac.coherence(free_text)
    report = {"label": "measured", "model_id": cfg["model_id"], "revision": cfg["revision"],
              "served_model": str(model_dir), "server_weights_env": weights_env,
              "model_dir_weights": marker, "server_ready_s": round(ready_s, 1),
              "n_tokens": sac.N_TOKENS, "matches": matches,
              "top1_agreement": matches / sac.N_TOKENS, **coh, "free_run_text": free_text,
              "mismatches": mismatches,
              "evidence": [os.path.relpath(evidence / "verify.json", run_dir),
                           os.path.relpath(log_path, run_dir)]}
    (evidence / "verify.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("top1_agreement", "coherent", "server_ready_s")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_package_verify.py tests/test_weights_swap_templates.py`
Expected: PASS (31 passed).

- [ ] **Step 5: Mutation checks**

1. The server's own weights environment: in `orchard/package_templates/verify_bundle.py`, replace `if set(weights_env.values()) != {str(model_dir)}:` with `if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_verify.py::test_a_copy_whose_run_sh_lost_the_weights_directory_is_caught`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
2. The check's environment has no weights variables: in `orchard/package_templates/verify_bundle.py`, replace `SCRUBBED_ENV = ("MODEL_WEIGHTS_DIR", "HF_MODEL",` with `SCRUBBED_ENV = (`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_verify.py::test_the_operators_own_weights_variables_cannot_stand_in_for_the_bundles`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
3. A busy port is refused: in `orchard/package_templates/verify_bundle.py`, replace `if port_answers(port):` with `if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_verify.py::test_a_port_that_already_answers_is_refused_before_any_server_starts`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
4. The served model is the copy's model-dir: in `orchard/package_templates/verify_bundle.py`, replace `if str(model_dir) not in models:` with `if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_verify.py::test_a_server_that_serves_another_model_is_refused`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
5. Model-dir names the manifest's weights: in `orchard/package_templates/verify_bundle.py`, replace `if marker != want:` with `if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_verify.py::test_a_model_dir_built_from_other_weights_is_refused`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
6. The nearest model is never linked: in `orchard/package.py`, replace `return sorted(found - {nearest_model})` with `return sorted(found)`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_verify.py::test_auxiliary_repos_never_include_the_nearest_model`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: 1420 passed and 1 skipped (the skip is the opt-in replay).

```bash
git add orchard/package.py orchard/package_templates/verify_bundle.py tests/fake_swap_server.py tests/test_package_verify.py
git commit -m "Boot-check an installed copy of the package and prove which server answered"
```

---

### Task 8: Stage every profile: cards, package.json and the publish commands as text

**Files:**
- Modify: `orchard/package.py` (append)
- Modify: `tests/package_fakes.py` (append `fake_boot_result`)
- Test: `tests/test_package_publish.py`

**Interfaces:**
- Consumes: Tasks 3 to 7.
- Produces in `orchard.package`: `PUBLISH_FILE = "PUBLISH_COMMANDS.txt"`, `PUBLISH_LINE`, `publish_commands(profiles, *, namespace, license_id) -> str`, `publish_problems(text) -> list[str]`, `card_numbers(facts, profile, verify) -> list[Number]`, `write_card(out, facts, profile, *, namespace, verify)`, `stage_all(run_dir, stage_dir, *, namespace, models_root, env=None, hostname=None) -> dict` (writes `package/<name>/` per profile, `verify/`, `hw_test.json`, `handoff.json`, `package.json`), `finish(run_dir, stage_dir, *, hostname=None) -> dict` (after the boot check: records it, rewrites the cards, scrubs again, writes `PUBLISH_COMMANDS.txt`).
- `package.json`: `{"format": "v6", "namespace", "model", "revision", "nearest_model", "license", "non_commercial", "profiles": [{"chips", "name", "required", "source", "source_manifest_sha256", "mesh", "dir", "verified", "verify", "scrub"}], "skipped_profiles": [{"chips", "reason"}], "hf_linked", "hf_missing", "publish_commands"}`. It holds no absolute path, because the operator bundle copies it.
- Produces in tests: `package_fakes.fake_boot_result(stage, *, returncode=0, top1=0.94)`.

`stage_all` stages every profile and scrubs each one; a hit stops the stage before anything is installed. It installs a copy of the required profile and writes the hardware test and a handoff note in the supervisor's own words (the coder has no conversation in stage 7). `finish` runs after the boot check. Only the boot-checked profile's card shows the run's measurements, because they were taken on that profile's chip count; an optional profile's card shows TODO. The publish commands are private `hf upload` lines. `tt-model push` publishes v5.1 container packages only, and `tt-model package-thin <repo>` would rebuild the bundle without the run.sh edits, so neither is used. The card's front matter carries the license and the tags. A profile that was not boot-checked gets a commented-out line.

- [ ] **Step 1: Write the failing tests**

Append to `tests/package_fakes.py`:

```python

def fake_boot_result(stage: Path, *, returncode=0, top1=0.94) -> None:
    """What the supervisor and verify_bundle.py leave in stages/7 after the boot check: the test's
    output and test-result.json, and with exit 0 the verify.json evidence."""
    write(stage / "evidence/hw-test-output.txt", "verify_bundle: done\n")
    write(stage / "test-result.json", {"returncode": returncode, "timed_out": False,
                                       "output": {"path": "stages/7/evidence/hw-test-output.txt",
                                                  "sha256": "0" * 64}})
    if returncode == 0:
        md = stage / "verify/bundle/model-dir"
        write(stage / "verify/evidence/server.log", "ready\n")
        write(stage / "verify/evidence/verify.json", {
            "label": "measured", "top1_agreement": top1, "coherent": True, "n_tokens": 32,
            "server_ready_s": 301.2, "served_model": str(md),
            "server_weights_env": {"MODEL_WEIGHTS_DIR": str(md), "HF_MODEL": str(md)},
            "evidence": ["stages/7/verify/evidence/verify.json",
                         "stages/7/verify/evidence/server.log"]})
```

Create `tests/test_package_publish.py`:

```python
"""Stage 7, part 3: stage every profile, write the cards, the record and the publish commands (plan 5)."""
import json
import os
import sys
from pathlib import Path

import pytest

from orchard.handoff import NOTE_KEYS
from orchard.package import (PackageError, finish, publish_commands, publish_problems, read_run,
                             stage_all)
from orchard.package_card import card_problems
from orchard.scrub import scrub_package
from package_fakes import (SOURCE_ENV, calls, fake_bin, fake_boot_result, make_run, make_source,
                           write)

HOST = "quietbox-test"
FAKE_SERVER = Path(__file__).resolve().with_name("fake_swap_server.py")
pytestmark = pytest.mark.usefixtures("stub_tools")


@pytest.fixture
def world(tmp_path, monkeypatch, stub_tools):
    models = tmp_path / "models"
    src2 = make_source(models)
    make_source(models, name="qwen3.8-27b-dflash2-p150", chips=1, mesh="P150")
    r = make_run(tmp_path, src2)
    fb, log = fake_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    env = dict(os.environ, FAKE_TT_MODEL_PYTHON=sys.executable, FAKE_SWAP_SERVER=str(FAKE_SERVER),
               FAKE_SWAP_CONFIG=str(tmp_path / "server.json"))
    return {**r, "models": models, "src2": src2, "log": log, "env": env, "stubs": stub_tools,
            "stage": r["run"] / "stages/7"}


def staged(world):
    world["stage"].mkdir(parents=True, exist_ok=True)
    return stage_all(world["run"], world["stage"], namespace="episod", models_root=world["models"],
                     env=world["env"], hostname=HOST)


def test_publish_commands_upload_privately_and_only_boot_checked_packages_get_a_live_line():
    profiles = [{"chips": 2, "name": "hemmingway-1-p300", "mesh": "P150x2", "verified": True},
                {"chips": 1, "name": "hemmingway-1-p150", "mesh": "P150", "verified": False}]
    text = publish_commands(profiles, namespace="episod", license_id="cc-by-nc-4.0")
    live = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert live == ["hf upload --repo-type model --private episod/hemmingway-1-p300 "
                    "stages/7/package/hemmingway-1-p300 ."]
    assert "# hf upload --repo-type model --private episod/hemmingway-1-p150" in text
    assert "NOT boot-checked" in text and "cc-by-nc-4.0 (non-commercial)" in text
    assert publish_problems(text) == []


@pytest.mark.parametrize("line, words", [
    ("hf upload --repo-type model --private episod/x stages/7/package/x . --public",
     "PUBLISH_COMMANDS.txt uses --public"),
    ("tt-model push stages/7/package/x", "not a private hf upload"),
    ("tt-model package-thin episod/x --publish", "PUBLISH_COMMANDS.txt uses --publish"),
    ("hf upload --repo-type model episod/x stages/7/package/x .", "not a private hf upload"),
])
def test_a_publish_line_that_is_not_a_private_upload_is_refused(line, words):
    assert any(words in p for p in publish_problems(f"# header\n{line}\n"))


def test_a_publish_file_with_no_live_line_is_refused():
    assert any("no command" in p for p in publish_problems("# hf upload ...\n"))


def test_stage_all_stages_every_profile_and_lays_out_the_boot_check(world):
    pkg = staged(world)
    assert [(p["chips"], p["name"], p["required"], p["verified"]) for p in pkg["profiles"]] == [
        (1, "hemmingway-1-p150", False, False), (2, "hemmingway-1-p300", True, False)]
    assert pkg["skipped_profiles"] == [
        {"chips": 4, "reason": "no v6 bundle of Qwen/Qwen3.8-27B for 4 chips is installed"}]
    assert (pkg["license"], pkg["non_commercial"], pkg["format"]) == ("cc-by-nc-4.0", True, "v6")
    assert json.loads((world["stage"] / "package.json").read_text()) == pkg
    for p in pkg["profiles"]:
        out = world["run"] / p["dir"]
        card = (out / "README.md").read_text()
        assert card_problems(card, license_id="cc-by-nc-4.0", run_dir=world["run"]) == []
        assert scrub_package(out, hostname=HOST, namespace="episod") == []
    test = json.loads((world["stage"] / "hw_test.json").read_text())
    assert test["command"].endswith("stages/7/verify/verify_bundle.py")
    note = json.loads((world["stage"] / "handoff.json").read_text())
    assert all(note.get(k) not in (None, "") for k in NOTE_KEYS)
    assert (world["stage"] / "verify" / "bundle" / "venv" / "bin" / "python").is_file()
    # Two package-thin calls with --out, nothing else; no stub ran.
    assert [c[0] for c in calls(world["log"])] == ["package-thin", "package-thin"]
    assert all("--out" in c for c in calls(world["log"])) and world["stubs"].calls() == []
    # Every absolute path in package.json lives outside it: the operator bundle copies it.
    assert "/home/" not in (world["stage"] / "package.json").read_text()


def test_a_scrub_hit_stops_the_stage_before_anything_is_installed(world):
    m = json.loads((world["src2"] / "tt_kernel_manifest.json").read_text())
    m["env"] = {**SOURCE_ENV, "BUILD_HOST": HOST}
    write(world["src2"] / "tt_kernel_manifest.json", m)
    with pytest.raises(PackageError, match=f"scrub of hemmingway-1-p300: .*{HOST}"):
        staged(world)
    assert not (world["stage"] / "verify").exists()
    assert not (world["stage"] / "hw_test.json").exists()


def test_finish_records_a_passing_boot_check_on_the_required_profile_only(world):
    staged(world)
    fake_boot_result(world["stage"])
    pkg = finish(world["run"], world["stage"], hostname=HOST)
    req = next(p for p in pkg["profiles"] if p["required"])
    opt = next(p for p in pkg["profiles"] if not p["required"])
    assert req["verified"] is True and opt["verified"] is False
    assert req["verify"]["top1_agreement"] == 0.94 and req["scrub"] == [] and opt["scrub"] == []
    card = (world["run"] / req["dir"] / "README.md").read_text()
    assert ("| top1 agreement with the CPU reference, this package (stage 7) | 0.94 fraction | measured |"
            in card)
    assert card_problems(card, license_id="cc-by-nc-4.0", run_dir=world["run"]) == []
    opt_card = (world["run"] / opt["dir"] / "README.md").read_text()
    assert "stage 2" not in opt_card.split("## Numbers")[1].split("## ")[0]       # no borrowed numbers
    text = (world["stage"] / "PUBLISH_COMMANDS.txt").read_text()
    assert publish_problems(text) == []
    assert "private episod/hemmingway-1-p300 stages/7/package/hemmingway-1-p300 .\n" in text


def test_finish_after_a_failed_boot_check_publishes_nothing(world):
    staged(world)
    fake_boot_result(world["stage"], returncode=4)
    pkg = finish(world["run"], world["stage"], hostname=HOST)
    req = next(p for p in pkg["profiles"] if p["required"])
    assert req["verified"] is False and "exited 4" in req["verify"]["failed"]
    assert any("no command" in p for p in
               publish_problems((world["stage"] / "PUBLISH_COMMANDS.txt").read_text()))


def test_read_run_still_works_after_staging(world):
    staged(world)
    assert read_run(world["run"]).model_id == "Altworld/Hemmingway-1"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_package_publish.py`
Expected: FAIL with a collection error, `ImportError: cannot import name 'finish' from 'orchard.package'`.

- [ ] **Step 3: Implement**

Append to `orchard/package.py`:

```python


# ---- cards, publish commands and the stage's record ----------------------------------------------

PUBLISH_FILE = "PUBLISH_COMMANDS.txt"
PUBLISH_LINE = re.compile(r"^hf upload --repo-type model --private [A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+ "
                          r"stages/7/package/[A-Za-z0-9_.-]+ \.$")


def publish_commands(profiles: list[dict], *, namespace: str, license_id: str) -> str:
    """The operator's publish commands, as text. Only a boot-checked profile gets a live line."""
    nc = " (non-commercial)" if non_commercial(license_id) else ""
    lines = ["# Publish commands for the packages stage 7 staged. The run never ran them.",
             "# Read RESULTS.md, RISKS.md and each package's README.md before running any of them.",
             "# Run them from the run directory. `hf upload --private` creates each repo private.",
             "# The card's front matter sets the repo's license and the tags `tt-model search` uses.",
             f"# License: {license_id}{nc}. Keep the card's license section as it is.",
             "# Making a repo public, or listing it in the tt-model catalog, is a separate decision.",
             "# `tt-model package-thin <repo>` rebuilds the bundle without stage 7's run.sh edits,",
             "# so it is not used to publish these packages.", ""]
    for p in sorted(profiles, key=lambda p: p["chips"]):
        line = (f"hf upload --repo-type model --private {namespace}/{p['name']} "
                f"stages/7/package/{p['name']} .")
        if p.get("verified"):
            lines += [f"# {p['chips']} chips ({p['mesh']}), boot-checked in stage 7:", line, ""]
        else:
            lines += [f"# {p['chips']} chips ({p['mesh']}), NOT boot-checked. Boot it before "
                      "publishing:", f"# {line}", ""]
    return "\n".join(lines)


def publish_problems(text: str) -> list[str]:
    problems = []
    live = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not live:
        problems.append(f"{PUBLISH_FILE} has no command for a boot-checked package")
    for ln in live:
        if not PUBLISH_LINE.match(ln):
            problems.append(f"{PUBLISH_FILE} line {ln!r} is not a private hf upload of a staged package")
    for flag in ("--public", "--publish"):
        if re.search(rf"(?<![\w-]){flag}\b", text):
            problems.append(f"{PUBLISH_FILE} uses {flag}")
    return problems


def _rel(run_dir: Path, path: Path) -> str:
    return os.path.relpath(path, run_dir)


def card_numbers(facts: RunFacts, profile: dict, verify: dict | None) -> list[Number]:
    """The numbers one package's card may show. Only the boot-checked profile shows the run's
    measurements, because they were taken on that profile's chip count."""
    run = facts.run_dir
    if not profile["required"]:
        return [Number("top1 agreement with the CPU reference, this package", None, "fraction",
                       "TODO", ())]
    s2 = _json(run / "stages/2/result.json", "stage 2's result")
    ev2 = ("stages/2/result.json", *s2.get("evidence", []))
    nums = [Number(f"top1 agreement with the CPU reference, stage 2 ({profile['chips']} chips, "
                   f"bundle {profile['source']})", s2["top1_agreement"], "fraction", "measured", ev2),
            Number("server ready after start, stage 2 (empty tensor cache)", s2["server_ready_s"],
                   "s", "measured", ev2)]
    s6_path = run / "stages/6/result.json"
    if s6_path.is_file():
        for n in _json(s6_path, "stage 6's result").get("numbers") or []:
            measured = n.get("label") == "measured"
            nums.append(Number(f"{n['name']} (stage 6)", n.get("value") if measured else None,
                               n["unit"], "measured" if measured else "TODO",
                               ("stages/6/result.json", *n.get("evidence", [])) if measured else ()))
    if verify is None:
        nums.append(Number("top1 agreement with the CPU reference, this package", None, "fraction",
                           "TODO", ()))
    else:
        ev7 = tuple(verify["evidence"])
        nums += [Number("top1 agreement with the CPU reference, this package (stage 7)",
                        verify["top1_agreement"], "fraction", "measured", ev7),
                 Number("server ready after start, this package (stage 7, fresh install, empty "
                        "tensor cache)", verify["server_ready_s"], "s", "measured", ev7)]
    return nums


def write_card(out: Path, facts: RunFacts, profile: dict, *, namespace: str,
               verify: dict | None) -> None:
    m = _json(out / "tt_kernel_manifest.json", "the staged manifest")
    drafters = aux_repos(m.get("env") or {}, nearest_model=facts.nearest_model)
    not_measured = ["a download of the weights through `tt-model pull`, and a boot of the package "
                    "from the Hub"]
    if drafters:
        not_measured.insert(0, "the drafter's acceptance rate on this model")
    facts_card = CardFacts(
        name=profile["name"], namespace=namespace, model_id=facts.model_id, revision=facts.revision,
        nearest_model=facts.nearest_model, source_name=profile["source"],
        license_id=facts.license_id, chips=profile["chips"], mesh=m["mesh"]["topology"],
        arch=m["arch"], max_model_len=m["resources"]["max_model_len"],
        max_num_seqs=m["resources"]["max_num_seqs"], drafter=drafters[0] if drafters else None,
        verified=verify is not None and profile["required"],
        numbers=tuple(card_numbers(facts, profile, verify)), not_measured=tuple(not_measured))
    (out / "README.md").write_text(render_card(facts_card), encoding="utf-8")


def _write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def stage_all(run_dir, stage_dir, *, namespace: str, models_root, env: dict | None = None,
              hostname: str | None = None) -> dict:
    """Stage every profile, scrub each one (a hit stops the stage before anything is installed),
    install a copy of the required profile and write hw_test.json, handoff.json and package.json."""
    facts = read_run(run_dir)
    stage_dir = Path(stage_dir)
    others = find_sources(models_root, entry_cls=facts.source.entry_cls,
                          nearest_model=facts.nearest_model)
    profiles, skipped = plan_profiles(facts, others)
    root = stage_dir / "package"
    root.mkdir(parents=True)
    records = []
    for p in profiles:
        out = root / bundle_name(facts.model_id, p.source.name)
        rec = stage_profile(p, facts, out, env=env)
        write_card(out, facts, rec, namespace=namespace, verify=None)
        hits = scrub_package(out, hostname=hostname, namespace=namespace)
        if hits:
            raise PackageError(f"scrub of {rec['name']}: " + "; ".join(hits))
        records.append({**rec, "dir": _rel(facts.run_dir, out), "verified": False})
    req = next(r for r in records if r["required"])
    test = prepare_verify(facts.run_dir / req["dir"], stage_dir / "verify", facts, env=env)
    _write_json(stage_dir / "hw_test.json",
                {"command": test["command"], "deadline_s": test["deadline_s"]})
    _write_json(stage_dir / "handoff.json", {
        "goal": f"package {facts.model_id} as a v6 thin bundle", "stage": 7,
        "evidence": [_rel(facts.run_dir, stage_dir / "package.json")],
        "next_action": f"boot the installed copy of {req['name']} and compare it with the reference",
        "check_on_return": "stages/7/test-result.json and stages/7/verify/evidence/verify.json"})
    package = {"format": "v6", "namespace": namespace, "model": facts.model_id,
               "revision": facts.revision, "nearest_model": facts.nearest_model,
               "license": facts.license_id, "non_commercial": non_commercial(facts.license_id),
               "profiles": records, "skipped_profiles": skipped, "hf_linked": test["hf_linked"],
               "hf_missing": test["hf_missing"], "publish_commands": f"stages/7/{PUBLISH_FILE}"}
    _write_json(stage_dir / "package.json", package)
    return package


def finish(run_dir, stage_dir, *, hostname: str | None = None) -> dict:
    """After the boot check: record its result, rewrite the cards with it, scrub again and write
    the publish commands."""
    facts = read_run(run_dir)
    stage_dir = Path(stage_dir)
    package = _json(stage_dir / "package.json", "stage 7's package.json")
    test = _json(stage_dir / "test-result.json", "stage 7's test-result.json")
    vpath = stage_dir / "verify" / "evidence" / "verify.json"
    verify = None
    if test.get("returncode") == 0 and vpath.is_file():
        verify = _json(vpath, "the boot check's verify.json")
    for rec in package["profiles"]:
        out = facts.run_dir / rec["dir"]
        rec["verified"] = bool(rec["required"] and verify is not None)
        if rec["required"]:
            if verify is None:
                rec["verify"] = {"failed": f"the boot check exited {test.get('returncode')} "
                                           f"(timed out: {test.get('timed_out')}); see "
                                           f"{test.get('output', {}).get('path')}"}
            else:
                rec["verify"] = {k: verify[k] for k in ("top1_agreement", "coherent", "n_tokens",
                                                        "server_ready_s", "evidence")}
        write_card(out, facts, rec, namespace=package["namespace"],
                   verify=verify if rec["required"] else None)
        rec["scrub"] = scrub_package(out, hostname=hostname, namespace=package["namespace"])
    (stage_dir / PUBLISH_FILE).write_text(
        publish_commands(package["profiles"], namespace=package["namespace"],
                         license_id=facts.license_id), encoding="utf-8")
    _write_json(stage_dir / "package.json", package)
    return package
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_package_publish.py`
Expected: PASS (11 passed).

- [ ] **Step 5: Mutation checks**

1. A failed boot check publishes nothing: in `orchard/package.py`, replace `rec["verified"] = bool(rec["required"] and verify is not None)` with `rec["verified"] = bool(rec["required"])`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_publish.py::test_finish_after_a_failed_boot_check_publishes_nothing`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
2. --public and --publish are refused: in `orchard/package.py`, replace `for flag in ("--public", "--publish"):` with `for flag in ():`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_publish.py::test_a_publish_line_that_is_not_a_private_upload_is_refused`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
3. A scrub hit stops the stage: in `orchard/package.py`, replace

```python
        if hits:
            raise PackageError(f"scrub of
```

   with

```python
        if False:
            raise PackageError(f"scrub of
```

   Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_publish.py::test_a_scrub_hit_stops_the_stage_before_anything_is_installed`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

4. An optional card borrows no numbers: in `orchard/package.py`, replace `if not profile["required"]:` with `if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_package_publish.py::test_finish_records_a_passing_boot_check_on_the_required_profile_only`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: 1431 passed and 1 skipped (the skip is the opt-in replay).

```bash
git add orchard/package.py tests/package_fakes.py tests/test_package_publish.py
git commit -m "Stage every profile, write the cards, package.json and the publish commands as text"
```

---

### Task 9: Stage 7 in the stage table, and gate_package

**Files:**
- Modify: `orchard/stages.py` (docstring, imports, `StageSpec.harness`, `gate_package`, `SKIP_7`, `validate_table`, `PACKAGE_STAGE_7`, `spec_for`, `package_options`, `package_format`)
- Test: `tests/test_gate_package.py`

**Interfaces:**
- Consumes: `orchard.package` (`PUBLISH_FILE`, `publish_problems`, `weights_wiring_problems`), `orchard.package_card` (`card_problems`, `read_license`), `orchard.scrub.scrub_package`, imported inside `gate_package` (the package module imports `orchard.stages`).
- Produces in `orchard.stages`: `StageSpec.harness: bool = False`; `gate_package(stage_dir, run_dir) -> GateResult`; `PACKAGE_STAGE_7` (skill "", boards 1, gate file `package.json`, marker `test-result.json`, `harness=True`); `spec_for(number, path, package_format=None)`; `PACKAGE_OPTIONS_SET = "package options set"`; `package_options(entries) -> dict | None`; `package_format(entries) -> str | None`.

The minimal table change. Stage 7 keeps its skipped spec, with a new reason. On the weights-only path of a run whose options name `v6`, `spec_for` returns `PACKAGE_STAGE_7`: one board, the same resume marker as stages 2 to 6, and `harness=True`, which tells the supervisor that code does the work. `validate_table` accepts a stage with no skill only when it is harness code. `gate_package` trusts `package.json` only to say where to look. It checks again, from the files on disk: the license read from the new model's snapshot, each staged directory's scrub with this machine's hostname, run.sh's weights settings, the manifest's weights, the card, the boot check's numbers and server, and the publish commands.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gate_package.py`:

```python
"""Stage 7's spec on each path, and gate_package checking the staged package from disk (plan 5)."""
import json
import os
import socket
import sys
from pathlib import Path

import pytest

from orchard.defaults import SWAP_TOP1_MIN
from orchard.package import finish, stage_all
from orchard.stages import (PACKAGE_STAGE_7, STAGES, gate_package, package_format,
                            package_options, spec_for, validate_table)
from package_fakes import fake_bin, fake_boot_result, make_run, make_source, write

FAKE_SERVER = Path(__file__).resolve().with_name("fake_swap_server.py")


def test_stage_7_packages_only_on_the_weights_only_path_with_a_format():
    assert spec_for(7, "weights-only", "v6") is PACKAGE_STAGE_7
    for path, fmt in (("weights-only", None), ("full-port", "v6"), (None, "v6"), ("weights-only", "v5.1")):
        assert spec_for(7, path, fmt) is STAGES[7] and STAGES[7].skip
    s = PACKAGE_STAGE_7
    assert (s.harness, s.boards, s.gate_file, s.gate, s.marker, s.skip, s.skill) == (
        True, 1, "package.json", gate_package, "test-result.json", None, "")
    validate_table(tuple(s if t.number == 7 else t for t in STAGES))


def test_a_skill_less_stage_that_is_not_harness_code_is_refused():
    import dataclasses
    bad = dataclasses.replace(PACKAGE_STAGE_7, harness=False)
    with pytest.raises(ValueError, match="stage 7 needs a skill"):
        validate_table(tuple(bad if t.number == 7 else t for t in STAGES))


def test_the_package_options_come_from_run_start_or_a_later_decision():
    start = {"seq": 1, "event": "run_start", "stage": None, "data": {"package": {"format": "v6"}}}
    assert package_format([start]) == "v6"
    assert package_format([{**start, "data": {}}]) is None
    assert package_format([]) is None
    later = {"seq": 2, "event": "decision", "stage": None,
             "data": {"decision": "package options set", "package": {"format": "v6", "namespace": "n"}}}
    assert package_options([{**start, "data": {"package": None}}, later]) == {"format": "v6",
                                                                             "namespace": "n"}


@pytest.fixture
def stage7(tmp_path, monkeypatch, stub_tools):
    """A stage 7 directory whose package was staged, boot-checked (faked) and finished."""
    models = tmp_path / "models"
    r = make_run(tmp_path, make_source(models))
    fb, _ = fake_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    env = dict(os.environ, FAKE_TT_MODEL_PYTHON=sys.executable, FAKE_SWAP_SERVER=str(FAKE_SERVER),
               FAKE_SWAP_CONFIG=str(tmp_path / "server.json"))
    stage = r["run"] / "stages/7"
    stage.mkdir(parents=True)
    stage_all(r["run"], stage, namespace="episod", models_root=models, env=env)
    fake_boot_result(stage)
    finish(r["run"], stage)
    return {**r, "stage": stage, "pkg": stage / "package/hemmingway-1-p300"}


def reasons(s) -> list[str]:
    return list(gate_package(s["stage"], s["run"]).reasons)


def test_a_staged_and_boot_checked_package_passes(stage7):
    g = gate_package(stage7["stage"], stage7["run"])
    assert g.ok, g.reasons
    assert "stages/7/verify/evidence/verify.json" in g.evidence


def test_a_run_sh_edited_back_to_the_base_weights_fails(stage7):
    run_sh = stage7["pkg"] / "run.sh"
    run_sh.write_text(run_sh.read_text().replace('export MODEL_WEIGHTS_DIR="$HERE/model-dir"\n', ""))
    assert any("MODEL_WEIGHTS_DIR" in r for r in reasons(stage7))


def test_a_hostname_or_a_cache_in_the_staged_package_fails(stage7):
    (stage7["pkg"] / "NOTES.txt").write_text(f"staged on {socket.gethostname()}\n")
    (stage7["pkg"] / ".tt_cache").mkdir()
    rs = reasons(stage7)
    assert any("scrub: NOTES.txt: the hostname" in r for r in rs), rs
    assert any("scrub: .tt_cache: a tensor cache directory" in r for r in rs), rs


def test_a_card_whose_license_was_changed_fails(stage7):
    card = stage7["pkg"] / "README.md"
    card.write_text(card.read_text().replace("license: cc-by-nc-4.0", "license: apache-2.0"))
    assert any("card: the card's license is apache-2.0" in r for r in reasons(stage7))


def test_a_model_whose_license_cannot_be_read_fails(stage7):
    (stage7["snapshot"] / "README.md").unlink()
    assert any("license cannot be read" in r for r in reasons(stage7))


def test_a_boot_check_below_the_bar_fails(stage7):
    v = stage7["stage"] / "verify/evidence/verify.json"
    write(v, {**json.loads(v.read_text()), "top1_agreement": 0.78})
    assert any(f"below {SWAP_TOP1_MIN}" in r for r in reasons(stage7))


def test_a_boot_check_against_another_server_fails(stage7):
    v = stage7["stage"] / "verify/evidence/verify.json"
    data = json.loads(v.read_text())
    data["server_weights_env"]["MODEL_WEIGHTS_DIR"] = "/elsewhere/Qwen3.8-27B"
    write(v, data)
    assert any("did not serve its own model-dir" in r for r in reasons(stage7))


def test_a_package_with_no_boot_checked_profile_fails(stage7):
    p = stage7["stage"] / "package.json"
    data = json.loads(p.read_text())
    for prof in data["profiles"]:
        prof["verified"] = False
    write(p, data)
    assert any("no required profile passed its boot check" in r for r in reasons(stage7))


def test_a_public_publish_command_fails(stage7):
    f = stage7["stage"] / "PUBLISH_COMMANDS.txt"
    f.write_text(f.read_text().replace(" .\n", " . --public\n"))
    assert any("--public" in r for r in reasons(stage7))


def test_a_missing_package_json_fails(tmp_path):
    g = gate_package(tmp_path, tmp_path)
    assert not g.ok and g.reasons == ("package.json is missing",)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_gate_package.py tests/test_stages.py tests/test_skills.py`
Expected: FAIL with a collection error, `ImportError: cannot import name 'PACKAGE_STAGE_7' from 'orchard.stages'`.

- [ ] **Step 3: Implement**

Replace in `orchard/stages.py`:

```python
path stage 2 uses the weights-swap-check skill and `gate_weights_swap`, and stage 3 is skipped
(`spec_for`, `run_path`).

```

with:

```python
path stage 2 uses the weights-swap-check skill and `gate_weights_swap`, and stage 3 is skipped
(`spec_for`, `run_path`). On the weights-only path of a run started with --package-format v6,
stage 7 is `PACKAGE_STAGE_7`: supervisor code (orchard/package.py) does its work, with no agent,
and `gate_package` checks the package again from the files on disk (`package_format`).

```

Replace in `orchard/stages.py`:

```python

from orchard.defaults import (COLD_START_S, LONG_STAGE_S, RUN_COLD_BOOT_CAP, RUN_ESCALATION_CAP,
                              RUN_WALL_CLOCK_S, STAGE2_PCC_MIN, STAGE_BUDGET_S, STAGE_DISK_GB,
                              SWAP_MIN_TOKENS, SWAP_TOP1_MIN)
from orchard.tiers import TierConfig
```

with:

```python

from orchard.defaults import (COLD_START_S, LONG_STAGE_S, PACKAGE_FORMATS, RUN_COLD_BOOT_CAP,
                              RUN_ESCALATION_CAP, RUN_WALL_CLOCK_S, STAGE2_PCC_MIN, STAGE_BUDGET_S,
                              STAGE_DISK_GB, SWAP_MIN_TOKENS, SWAP_TOP1_MIN)
from orchard.tiers import TierConfig
```

Replace in `orchard/stages.py`:

```python
    skip: str | None = None             # why plan 4 skips this stage

```

with:

```python
    skip: str | None = None             # why plan 4 skips this stage
    harness: bool = False               # supervisor code does the work; no agent and no model

```

Replace in `orchard/stages.py`:

```python

SKIP_7 = ("plan 4 builds no package or container image; the operator bundle reports the "
          "package as not built")

```

with:

```python

def _inside_dir(run_dir, rel) -> Path | None:
    """The directory `rel` names, if it is a directory inside the run directory (links resolved)."""
    if not isinstance(rel, str) or not rel or os.path.isabs(rel):
        return None
    root = os.path.realpath(run_dir)
    p = os.path.realpath(os.path.join(root, rel))
    return Path(p) if os.path.commonpath([root, p]) == root and os.path.isdir(p) else None


def gate_package(stage_dir, run_dir) -> GateResult:
    """Stage 7: a staged v6 package whose boot check passed.

    Everything that matters is checked again from the files on disk: the license (read from the
    new model's snapshot), each staged directory's scrub, run.sh's weights settings, the manifest's
    weights, the card, the boot check's verify.json and the publish commands. package.json only
    says where to look."""
    from orchard.package import PUBLISH_FILE, publish_problems, weights_wiring_problems
    from orchard.package_card import card_problems, read_license
    from orchard.scrub import scrub_package
    d, err = _load(stage_dir, "package.json")
    if err:
        return GateResult(False, (err,))
    run = Path(run_dir)
    reasons, seen = [], []
    if d.get("format") != "v6":
        reasons.append(f"package.json format must be 'v6', got {d.get('format')!r}")
    delta, _ = _load(run / "stages" / "0", "delta.json")
    swap, _ = _load(run / "stages" / "2", "swap_config.json")
    delta, swap = delta or {}, swap or {}
    license_id = read_license(swap.get("new_snapshot") or "/nonexistent")
    if not license_id:
        reasons.append("the new model's license cannot be read from its snapshot's README.md")
    elif d.get("license") != license_id:
        reasons.append(f"package.json says the license is {d.get('license')!r}; the model's is "
                       f"{license_id!r}")
    profiles = d.get("profiles") if isinstance(d.get("profiles"), list) else []
    if not any(isinstance(p, dict) and p.get("required") and p.get("verified") is True
               for p in profiles):
        failed = [p.get("verify", {}).get("failed") for p in profiles if isinstance(p, dict)]
        reasons.append("no required profile passed its boot check"
                       + (f": {failed[0]}" if failed and failed[0] else ""))
    for p in profiles:
        name = p.get("name") if isinstance(p, dict) else None
        out = _inside_dir(run, p.get("dir")) if isinstance(p, dict) else None
        if out is None:
            reasons.append(f"profile {name!r}: its dir is not a directory inside the run directory")
            continue
        try:
            m = json.loads((out / "tt_kernel_manifest.json").read_text(encoding="utf-8"))
            run_sh = (out / "run.sh").read_text(encoding="utf-8")
            card = (out / "README.md").read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            reasons.append(f"profile {name}: {exc}")
            continue
        w = m.get("weights") or {}
        if (w.get("repo_id"), w.get("revision")) != (delta.get("model"), d.get("revision")):
            reasons.append(f"profile {name}: the manifest's weights are {w.get('repo_id')}@"
                           f"{w.get('revision')}; expected {delta.get('model')}@{d.get('revision')}")
        reasons += [f"profile {name}: {x}" for x in
                    weights_wiring_problems(run_sh, nearest_model=delta.get("nearest_model") or "")]
        reasons += [f"profile {name}: scrub: {x}" for x in
                    scrub_package(out, namespace=d.get("namespace"))]
        if license_id:
            reasons += [f"profile {name}: card: {x}" for x in
                        card_problems(card, license_id=license_id, run_dir=run)]
        if p.get("verified") is True:
            v, verr = _load(Path(stage_dir) / "verify" / "evidence", "verify.json")
            if verr:
                reasons.append(f"profile {name}: {verr}")
                continue
            top1, n = v.get("top1_agreement"), v.get("n_tokens")
            if not _number(top1) or top1 < SWAP_TOP1_MIN:
                reasons.append(f"profile {name}: the boot check's top1_agreement {top1!r} is below "
                               f"{SWAP_TOP1_MIN}")
            if v.get("coherent") is not True:
                reasons.append(f"profile {name}: the boot check's free-run text is not coherent")
            if isinstance(n, bool) or not isinstance(n, int) or n < SWAP_MIN_TOKENS:
                reasons.append(f"profile {name}: the boot check compared {n!r} tokens; at least "
                               f"{SWAP_MIN_TOKENS} are needed")
            served = v.get("served_model") or ""
            if (not served.endswith("/model-dir")
                    or set((v.get("server_weights_env") or {}).values()) != {served}):
                reasons.append(f"profile {name}: the boot check's server did not serve its own "
                               "model-dir with MODEL_WEIGHTS_DIR and HF_MODEL set to it")
            _evidence(run_dir, v.get("evidence"), f"profile {name} boot check", reasons, seen)
    try:
        text = (Path(stage_dir) / PUBLISH_FILE).read_text(encoding="utf-8")
        reasons += publish_problems(text)
    except OSError:
        reasons.append(f"{PUBLISH_FILE} is missing")
    return _done(reasons, seen)


SKIP_7 = ("stage 7 builds a package only on the weights-only path of a run started with "
          "--package-format v6; this run builds none, and the operator bundle reports the package "
          "as not built")

```

Replace in `orchard/stages.py`:

```python
            continue
        if s.gate is None or not s.gate_file or not s.skill:
            raise ValueError(f"stage {s.number} needs a skill, a gate file and a gate")
        if s.budget_s > LONG_STAGE_S and not s.marker:
```

with:

```python
            continue
        if s.gate is None or not s.gate_file or not (s.skill or s.harness):
            raise ValueError(f"stage {s.number} needs a skill (or harness code), a gate file "
                             "and a gate")
        if s.budget_s > LONG_STAGE_S and not s.marker:
```

Replace in `orchard/stages.py`:

```python
WEIGHTS_ONLY_STAGE_3 = dataclasses.replace(STAGES[3], skip=SKIP_3_WEIGHTS_ONLY)

```

with:

```python
WEIGHTS_ONLY_STAGE_3 = dataclasses.replace(STAGES[3], skip=SKIP_3_WEIGHTS_ONLY)

# Stage 7 on the weights-only path of a run started with --package-format v6 (plan 5): supervisor
# code stages a v6 thin package, installs a copy and boots it on one board (orchard/package.py).
# The hardware phase is the same as stages 2 to 6, so the coder is parked when no board is free.
PACKAGE_STAGE_7 = dataclasses.replace(
    STAGES[7], name="package (v6 thin bundle) and boot check", boards=1, gate_file="package.json",
    gate=gate_package, marker="test-result.json", skip=None, harness=True)
validate_table(tuple(PACKAGE_STAGE_7 if s.number == 7 else s for s in STAGES))

```

Replace in `orchard/stages.py`:

```python

def spec_for(number: int, path: str | None) -> StageSpec:
    """The stage spec for `number` on `path`. Only the weights-only path changes the table: stage 2
    gets the swap skill and gate, and stage 3 is skipped (the supervisor records it as skipped,
    as it does stage 7)."""
    if path == "weights-only" and number == 2:
```

with:

```python

def spec_for(number: int, path: str | None, package_format: str | None = None) -> StageSpec:
    """The stage spec for `number` on `path`. Only the weights-only path changes the table: stage 2
    gets the swap skill and gate, stage 3 is skipped (the supervisor records it as skipped), and
    when the run was started with --package-format v6, stage 7 packages the model."""
    if path == "weights-only" and number == 2:
```

Replace in `orchard/stages.py`:

```python
        return WEIGHTS_ONLY_STAGE_3
    return STAGES[number]

```

with:

```python
        return WEIGHTS_ONLY_STAGE_3
    if path == "weights-only" and number == 7 and package_format == "v6":
        return PACKAGE_STAGE_7
    return STAGES[number]


PACKAGE_OPTIONS_SET = "package options set"


def package_options(entries: list[dict]) -> dict | None:
    """The run's stage 7 options: run_start's `package`, or a later "package options set" decision
    (a run started without them may get them on a resume, before stage 7 has started)."""
    opts = None
    for e in entries:
        if e["event"] == "run_start":
            opts = e["data"].get("package")
        elif e["event"] == "decision" and e["data"].get("decision") == PACKAGE_OPTIONS_SET:
            opts = e["data"].get("package")
    return opts if isinstance(opts, dict) else None


def package_format(entries: list[dict]) -> str | None:
    """The run's package format ("v6"), or None when the run builds no package."""
    fmt = (package_options(entries) or {}).get("format")
    return fmt if fmt in PACKAGE_FORMATS else None

```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_gate_package.py tests/test_stages.py tests/test_skills.py`
Expected: PASS (106 passed).

- [ ] **Step 5: Mutation checks**

1. The gate rechecks run.sh: in `orchard/stages.py`, replace `weights_wiring_problems(run_sh, nearest_model=delta.get("nearest_model") or "")]` with `[]]`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_gate_package.py::test_a_run_sh_edited_back_to_the_base_weights_fails`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
2. The gate rescrubs: in `orchard/stages.py`, replace `scrub_package(out, namespace=d.get("namespace"))]` with `[]]`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_gate_package.py::test_a_hostname_or_a_cache_in_the_staged_package_fails`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
3. The boot check's server environment: in `orchard/stages.py`, replace `or set((v.get("server_weights_env") or {}).values()) != {served}):` with `or False):`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_gate_package.py::test_a_boot_check_against_another_server_fails`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
4. The agreement bar: in `orchard/stages.py`, replace `if not _number(top1) or top1 < SWAP_TOP1_MIN:` with `if not _number(top1):`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_gate_package.py::test_a_boot_check_below_the_bar_fails`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
5. An unreadable license: in `orchard/stages.py`, replace

```python
    if not license_id:
        reasons.append("the new model's license cannot
```

   with

```python
    if False:
        reasons.append("the new model's license cannot
```

   Run `python3 -m pytest -q -p no:cacheprovider tests/test_gate_package.py::test_a_model_whose_license_cannot_be_read_fails`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

6. Stage 7 needs the format: in `orchard/stages.py`, replace `if path == "weights-only" and number == 7 and package_format == "v6":` with `if path == "weights-only" and number == 7:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_gate_package.py::test_stage_7_packages_only_on_the_weights_only_path_with_a_format`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
7. Only harness code may have no skill: in `orchard/stages.py`, replace `not (s.skill or s.harness)` with `not (s.skill or True)`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_gate_package.py::test_a_skill_less_stage_that_is_not_harness_code_is_refused`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: 1444 passed and 1 skipped (the skip is the opt-in replay).

```bash
git add orchard/stages.py tests/test_gate_package.py
git commit -m "Add stage 7's harness spec and gate_package, which rechecks the package from disk"
```

---

### Task 10: The supervisor runs stage 7, and the operator bundle carries its publish commands

**Files:**
- Modify: `orchard/supervisor.py` (docstring, imports, run options, `package_option`, `build`, `Supervisor.__init__`, `_run`, `_stage_body`, `_package_body`, `_check_gate`, `_copy_package`, `_end_stage`)
- Modify: `orchard/skills/operator-bundle.md`
- Test: `tests/test_supervisor_package.py`; `tests/test_skills.py` (append)

**Interfaces:**
- Consumes: `stage_all`, `finish`, `PackageError` (Task 8); `spec_for`, `package_format`, `package_options`, `PACKAGE_OPTIONS_SET` (Task 9); `PACKAGE_DEFERRED`, `PACKAGE_FORMATS`, `TT_MODEL_MODELS_ROOT` (Task 1); the existing `_hardware_phase`, `_read_test`, `agent_env`.
- Produces: run options `--package-format {v6,v5.1}`, `--package-namespace NS`, `--package-models-root DIR`; `package_option(args) -> dict | None` (refuses v5.1 and a format without a namespace); `Supervisor(..., package: dict | None = None, package_added: bool = False)`; run_start's `package` field; the decision `package options set`; `bundle/package/` in stage 8 (package.json, PUBLISH_COMMANDS.txt, `<name>-README.md`).

Stage 7 needs no model, so its tier stays `none` and the tier loader is unchanged. `_package_body` stages and scrubs, records an `evidence` entry, then runs the existing hardware phase: with the coder on two chips the free board is leased; with the coder on four chips it parks the coder, runs the boot check and restores it, as for stages 2 to 6. A failure pauses the run without an escalation, because another model cannot fix supervisor code. package-thin and install.sh get the agent shells' environment (no tokens, HOME inside the run directory). The package options are recorded in run_start and kept on resume. A run that started without them, like the Hemmingway-1 run now in progress, can be given them on a resume before stage 7 starts. The test's scripted bring-up writes the files stage 2's skill would leave; stage 7 then runs for real against the fake tt-model and the fake server.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_supervisor_package.py`:

```python
"""The supervisor runs stage 7 on the weights-only path when asked for a package (plan 5).

A scripted bring-up walks stages 0 to 8 against the fake two-board machine. Its stage 1 writes the
reference token ids, and its stage 2 prepare step writes the swap_config.json and model-dir config
that the weights-swap-check skill would leave. Stage 7 then runs as supervisor code: the fake
tt-model stages the package, the copy's install.sh builds a venv whose python serves
tests/fake_swap_server.py, and verify_bundle.py runs as the stage's hardware test under a lease.
"""
import json
import os
import socket
import sys
from pathlib import Path

import pytest

from fake_model import FakeModel
from fakes import Crash
from orchard.ledger import Ledger
from orchard.supervisor import EXIT_READY, EXIT_REFUSED, build, main, parse
from package_fakes import (BASE, BASE_REV, DRAFTER, DRAFTER_REV, GENERATED, HAVE_TOKENIZERS, NEW,
                           NEW_REV, PROMPT_IDS, SOURCE_ENV, VOCAB, calls, fake_bin, hf_snapshot,
                           make_source, tokenizer_json)
from run_fakes import (BOARDS, FILES, CrashingLedger, Machine, MachineAdapter, MachineCoder, argv,
                       bringup, clock, hw, plenty, write_tiers)

if not HAVE_TOKENIZERS:
    pytest.skip("SKIPPED: the `tokenizers` package is not importable, so the stage 7 boot check "
                "did not run. verify_bundle.py needs it.", allow_module_level=True)

FAKE_SERVER = Path(__file__).resolve().with_name("fake_swap_server.py")
pytestmark = pytest.mark.usefixtures("stub_tools")


class Paused(Exception):
    """The run paused for the operator; the test ends here."""


class PackageRig:
    """The machine, the model servers, the installed bundles, the HF caches and the run options."""

    def __init__(self, tmp, monkeypatch, *, chips=2, source_env=None):
        self.tmp, self.m = tmp, Machine()
        models = tmp / "models"
        self.source = make_source(models, env=source_env)
        hf_run, hf_op = tmp / "hf-run", tmp / "hf-operator"
        snap = hf_snapshot(hf_run, NEW, NEW_REV, {
            "README.md": f"---\nlicense: cc-by-nc-4.0\nbase_model:\n- {BASE}\n---\n",
            "tokenizer.json": tokenizer_json(), "model-00001-of-00001.safetensors": "new"})
        hf_snapshot(hf_op, DRAFTER, DRAFTER_REV, {"model.safetensors": "drafter"})
        hf_snapshot(hf_op, BASE, BASE_REV, {"model-00001-of-00001.safetensors": "base"})
        self.run_dir = tmp / "run"
        swap = {"run_dir": str(self.run_dir), "bundle_dir": str(self.source),
                "nearest_model_id": BASE, "new_snapshot": str(snap), "new_model_id": NEW,
                "hf_home": str(hf_op), "port": 8100}
        self.files = {
            (1, "run"): {**FILES[(1, "run")],
                         "evidence/reference/prompt-ids.json": {"prompt_ids": PROMPT_IDS},
                         "evidence/reference/generated-ids.json": {"generated_ids": GENERATED}},
            (2, "prepare"): {**hw(2), "swap_config.json": swap,
                             "model-dir/config.json": '{"architectures": ["Qwen3_5ForConditionalGeneration"]}',
                             "model-dir/preprocessor_config.json": "{}"}}
        self.pid_file, server_cfg = tmp / "server-pid.json", tmp / "server.json"
        server_cfg.write_text(json.dumps({"mode": "perfect", "vocab": VOCAB, "prompt_ids": PROMPT_IDS,
                                          "generated_ids": GENERATED, "pid_file": str(self.pid_file)}))
        fb, self.tt_log = fake_bin(tmp)
        monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
        self.chip_server = FakeModel(lambda r: bringup(r, overrides=self.files), models=["qwen-27b"])
        self.cpu_server = FakeModel(bringup, models=["cpu-model"])
        self.chip_server.__enter__()
        self.cpu_server.__enter__()
        tiers = write_tiers(tmp / "tiers.toml", self.chip_server.endpoint, self.cpu_server.endpoint)
        self.argv = argv(tmp, tiers, self.chip_server.endpoint, chips=chips) + [
            "--package-format", "v6", "--package-namespace", "episod",
            "--package-models-root", str(models),
            "--env", f"FAKE_TT_MODEL_PYTHON={sys.executable}",
            "--env", f"FAKE_SWAP_SERVER={FAKE_SERVER}", "--env", f"FAKE_SWAP_CONFIG={server_cfg}"]
        self.clock = clock()

    def sleep(self, s):
        self.clock.sleep(s)
        if any(e["event"] == "decision" and e["data"].get("decision") == "pause"
               for e in self.led.read()):
            raise Paused()

    def run(self, extra=(), pid=100, crash_if=None):
        path = self.run_dir / "ledger.jsonl"
        self.m.owner_pid = pid
        with (CrashingLedger(path, crash_if=crash_if) if crash_if else Ledger(path)) as led:
            self.led = led
            return build(parse(self.argv + list(extra)), led,
                         adapter=MachineAdapter(self.m, owner_pid=pid),
                         coder=MachineCoder(self.m), versions={"tt_model": "test"},
                         clock=self.clock, sleep=self.sleep, disk_usage=plenty,
                         home=self.tmp / "operator-home").run()

    def entries(self):
        if not (self.run_dir / "ledger.jsonl").exists():
            return []
        with Ledger(self.run_dir / "ledger.jsonl") as led:
            return led.read()

    def close(self):
        self.chip_server.__exit__()
        self.cpu_server.__exit__()
        if self.pid_file.exists():
            try:
                os.killpg(json.loads(self.pid_file.read_text())["pgid"], 9)
            except ProcessLookupError:
                pass


@pytest.fixture
def rig(tmp_path, monkeypatch):
    made = []

    def make(**kw):
        r = PackageRig(tmp_path, monkeypatch, **kw)
        made.append(r)
        return r
    yield make
    for r in made:
        r.close()


def seq(entries, n):
    return [(e["event"], e["data"].get("step") or e["data"].get("decision") or e["data"].get("what"))
            for e in entries if e["stage"] == n]


def test_a_weights_only_run_with_a_package_format_stages_and_boot_checks_in_stage_7(rig):
    r = rig(chips=2)
    assert r.run() == EXIT_READY
    es = r.entries()
    start = next(e for e in es if e["event"] == "run_start")
    assert start["data"]["package"] == {"format": "v6", "namespace": "episod",
                                        "models_root": str(r.tmp / "models")}
    end7 = [e["data"] for e in es if e["event"] == "stage_end" and e["stage"] == 7]
    assert [d["result"] for d in end7] == ["pass"]
    s7 = seq(es, 7)
    assert ("evidence", "package staged") in s7
    assert s7.index(("evidence", "package staged")) < s7.index(("decision", "hardware test started"))
    lease = next(e["data"]["test_lease"] for e in es
                 if e["stage"] == 7 and e["data"].get("decision") == "test lease taken")
    assert lease["chips"] == list(BOARDS["B1"])           # the free board; the coder stays loaded
    assert not [e for e in es if e["event"] == "escalate" and e["stage"] == 7]
    verify = json.loads((r.run_dir / "stages/7/verify/evidence/verify.json").read_text())
    assert verify["top1_agreement"] == 1.0
    assert verify["served_model"] == str(r.run_dir / "stages/7/verify/bundle/model-dir")
    bundle = r.run_dir / "stages/8/bundle/package"
    assert sorted(p.name for p in bundle.iterdir()) == ["PUBLISH_COMMANDS.txt",
                                                       "hemmingway-1-p300-README.md", "package.json"]
    # The only tt-model calls were package-thin with --out; the stub tools (hf, git, gh, docker,
    # curl and the rest) never ran; the fake server was stopped.
    assert calls(r.tt_log) and all(c[0] == "package-thin" and "--out" in c for c in calls(r.tt_log))
    assert not (r.tmp / "stub-bin" / "called.log").exists()
    assert not r.m.leases and not r.m.coder_running


def test_with_the_coder_on_four_chips_stage_7_parks_it_for_the_boot_check(rig):
    r = rig(chips=4)
    assert r.run() == EXIT_READY
    s7 = seq(r.entries(), 7)
    test = s7.index(("decision", "hardware test started"))
    assert s7.index(("park", "reset")) < test < s7.index(("restore", "reset"))
    assert ("restore", "resumed") in s7[test:]


def test_a_scrub_hit_pauses_the_run_at_stage_7_without_escalating_or_leasing(rig):
    r = rig(chips=2, source_env={**SOURCE_ENV, "BUILD_HOST": socket.gethostname()})
    with pytest.raises(Paused):
        r.run()
    es = r.entries()
    end7 = [e["data"] for e in es if e["event"] == "stage_end" and e["stage"] == 7]
    assert [d["result"] for d in end7] == ["fail"]
    assert "scrub" in end7[0]["reasons"][0]
    pause = [e["data"]["reason"] for e in es if e["data"].get("decision") == "pause"]
    assert pause and "not escalated" in pause[-1]
    assert not [e for e in es if e["stage"] == 7 and e["event"] == "escalate"]
    assert not [e for e in es if e["stage"] == 7 and e["data"].get("decision") == "test lease taken"]


def test_without_a_package_format_stage_7_is_skipped(rig):
    r = rig(chips=2)
    r.argv = r.argv[:r.argv.index("--package-format")] + r.argv[r.argv.index("--package-models-root") + 2:]
    assert r.run() == EXIT_READY
    assert [e["data"]["result"] for e in r.entries()
            if e["event"] == "stage_end" and e["stage"] == 7] == ["skipped"]
    assert calls(r.tt_log) == []


def test_v5_1_is_refused_at_start_and_says_why(tmp_path, capsys):
    tiers = write_tiers(tmp_path / "t.toml", "http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1")
    a = argv(tmp_path, tiers, "http://127.0.0.1:8000/v1") + [
        "--package-format", "v5.1", "--package-namespace", "episod", "--gozer", "/nonexistent"]
    assert main(a, home=tmp_path / "operator-home") == EXIT_REFUSED
    err = capsys.readouterr().err
    assert "v5.1 is deferred" in err and "image build" in err
    assert not (tmp_path / "run" / "ledger.jsonl").exists()


def test_a_package_format_needs_a_namespace(tmp_path, capsys):
    tiers = write_tiers(tmp_path / "t.toml", "http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1")
    a = argv(tmp_path, tiers, "http://127.0.0.1:8000/v1") + ["--package-format", "v6",
                                                             "--gozer", "/nonexistent"]
    assert main(a, home=tmp_path / "operator-home") == EXIT_REFUSED
    assert "--package-namespace" in capsys.readouterr().err


def test_a_resumed_run_keeps_its_package_options(rig):
    r = rig(chips=2)
    assert r.run() == EXIT_READY
    with pytest.raises(ValueError, match="has package options"):
        r.run(extra=["--package-namespace", "someone-else"])


def test_a_run_started_without_package_options_can_get_them_before_stage_7(rig):
    r = rig(chips=2)
    plain = r.argv[:r.argv.index("--package-format")] + r.argv[r.argv.index("--package-models-root") + 2:]
    full, r.argv = r.argv, plain
    with pytest.raises(Crash):
        r.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 2)
    r.argv = full
    assert r.run(pid=200) == EXIT_READY
    es = r.entries()
    assert next(e for e in es if e["event"] == "run_start")["data"]["package"] is None
    added = [e for e in es if e["data"].get("decision") == "package options set"]
    assert len(added) == 1 and added[0]["data"]["package"]["namespace"] == "episod"
    assert [e["data"]["result"] for e in es if e["event"] == "stage_end" and e["stage"] == 7] == ["pass"]


def test_package_options_cannot_be_added_once_stage_7_has_started(tmp_path):
    tiers = write_tiers(tmp_path / "t.toml", "http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1")
    base = argv(tmp_path, tiers, "http://127.0.0.1:8000/v1")
    with Ledger(tmp_path / "run" / "ledger.jsonl") as led:
        led.append("run_start", None, model=NEW, versions={}, inputs={}, required_chips=None,
                   package=None)
        led.append("stage_start", 7, skip=True)
        with pytest.raises(ValueError, match="only before stage 7 starts"):
            build(parse(base + ["--package-format", "v6", "--package-namespace", "episod"]), led,
                  adapter=MachineAdapter(Machine()), coder=MachineCoder(Machine()),
                  versions={"tt_model": "test"}, home=tmp_path / "operator-home")
```

Append to `tests/test_skills.py`:

```python


def test_the_bundle_skill_carries_stage_7s_publish_commands_word_for_word():
    text = (SKILLS / "operator-bundle.md").read_text()
    assert "bundle/package/PUBLISH_COMMANDS.txt" in text and "word for word" in text
    assert "non-commercial" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_package.py tests/test_skills.py`
Expected: FAIL. 9 tests fail (`9 failed, 18 passed in 0.83s`). The first failure reads `SystemExit: 2`.

- [ ] **Step 3: Implement**

Replace in `orchard/supervisor.py`:

```python

Plan 4 runs stages 0 to 6 and 8. Stage 7 (package and container build) is recorded as skipped,
and so is stage 3 on the weights-only path.
"""
```

with:

```python

Plan 4 runs stages 0 to 6 and 8; stage 3 is recorded as skipped on the weights-only path. Stage 7
(plan 5) packages the model only on the weights-only path of a run started with
--package-format v6 and --package-namespace; otherwise it is recorded as skipped. It is supervisor
code (orchard/package.py) with no agent: it stages a v6 thin bundle per profile, scrubs each one,
installs a copy of the required profile, boots that copy on a leased board as the stage's hardware
test (parking the coder when no board is free, as stages 2 to 6 do), writes the cards and the
publish commands as text, and runs gate_package. A failure pauses the run and is not escalated,
because no model ran. The run's ledger records the package options in run_start, and a resumed run
keeps them. A run that started without them can be given them on a resume before stage 7 starts;
the ledger then records a "package options set" decision. --package-format v5.1 is refused at
start (orchard/defaults.py, PACKAGE_DEFERRED).
"""
```

Replace in `orchard/supervisor.py`:

```python
                              COLD_BOOT_BUDGET_S, CONTROL_POLL_S, FIRST_BOOT_EXPECTED,
                              FIRST_BOOT_PROMPT, RUN_CANARY_PROMPT)
from orchard.handoff import (NOTE_KEYS, Blocked, Budgets, Handoff, decide_park, progress, reacquire,
                             recover, wait_stopped)
from orchard.ledger import Ledger, LedgerCorrupt, LedgerLocked
from orchard.runner import Denied, check_string
```

with:

```python
                              COLD_BOOT_BUDGET_S, CONTROL_POLL_S, FIRST_BOOT_EXPECTED,
                              FIRST_BOOT_PROMPT, PACKAGE_DEFERRED, PACKAGE_FORMATS,
                              RUN_CANARY_PROMPT, TT_MODEL_MODELS_ROOT)
from orchard.handoff import (NOTE_KEYS, Blocked, Budgets, Handoff, decide_park, progress, reacquire,
                             recover, wait_stopped)
from orchard.ledger import Ledger, LedgerCorrupt, LedgerLocked
from orchard.package import PackageError, stage_all
from orchard.package import finish as package_finish
from orchard.runner import Denied, check_string
```

Replace in `orchard/supervisor.py`:

```python
                            coder_state, delta_path, evidence_record, open_stage_dir,
                            resolve_endpoint, resolve_skill, run_path, run_progress, spec_for,
```

with:

```python
                            coder_state, delta_path, evidence_record, open_stage_dir,
                            PACKAGE_OPTIONS_SET, package_format, package_options,
                            resolve_endpoint, resolve_skill, run_path, run_progress, spec_for,
```

Replace in `orchard/supervisor.py`:

```python
                 credentials_visible: list[str] | None = None,
                 required_chips: tuple[int, ...] | None = None):
        self.run_dir = Path(run_dir).resolve()
```

with:

```python
                 credentials_visible: list[str] | None = None,
                 required_chips: tuple[int, ...] | None = None, package: dict | None = None,
                 package_added: bool = False):
        self.run_dir = Path(run_dir).resolve()
```

Replace in `orchard/supervisor.py`:

```python
        self.required_chips = tuple(required_chips) if required_chips else None   # stage 4's required counts
        self.control = Control(self.run_dir)
```

with:

```python
        self.required_chips = tuple(required_chips) if required_chips else None   # stage 4's required counts
        self.package = dict(package) if package else None    # stage 7: format, namespace, models_root
        self.package_added = package_added    # given on a resume of a run that started without them
        self.control = Control(self.run_dir)
```

Replace in `orchard/supervisor.py`:

```python
                               inputs=self.inputs, required_chips=list(self.required_chips or ()) or None,
                               coder=self.coder.record(),
                               tiers={k: dict(v) for k, v in self.cfg.tiers.items()})
        if self.credentials_visible:
```

with:

```python
                               inputs=self.inputs, required_chips=list(self.required_chips or ()) or None,
                               package=self.package,
                               coder=self.coder.record(),
                               tiers={k: dict(v) for k, v in self.cfg.tiers.items()})
        if self.package_added and package_options(self.ledger.read()) != self.package:
            self.ledger.append("decision", None, decision=PACKAGE_OPTIONS_SET, package=self.package)
        if self.credentials_visible:
```

Replace in `orchard/supervisor.py`:

```python
            # resumed run picks the same skill and gate as the run that crashed.
            spec = spec_for(p.next_stage, run_path(self.ledger.read(), self.run_dir))
            try:
```

with:

```python
            # resumed run picks the same skill and gate as the run that crashed.
            entries = self.ledger.read()
            spec = spec_for(p.next_stage, run_path(entries, self.run_dir), package_format(entries))
            try:
```

Replace in `orchard/supervisor.py`:

```python
    def _stage_body(self, spec, stage_dir: Path, escalated: bool, resumed: bool):
        if spec.boards == 0:
```

with:

```python
    def _stage_body(self, spec, stage_dir: Path, escalated: bool, resumed: bool):
        if spec.harness:
            return self._package_body(spec, stage_dir, resumed)
        if spec.boards == 0:
```

Replace in `orchard/supervisor.py`:

```python

    def _check_gate(self, spec, stage_dir: Path):
```

with:

```python

    def _package_body(self, spec, stage_dir: Path, resumed: bool):
        """Stage 7 as supervisor code (orchard/package.py): stage and scrub every profile, install a
        copy of the required one, boot it under a lease (the same hardware phase as stages 2 to 6),
        then record the result, write the cards and publish commands, and run the gate.

        package-thin and install.sh get the agent shells' environment: no tokens, and HOME inside
        the run directory. So even a call that tried to upload would find no credentials."""
        n = spec.number
        opts = package_options(self.ledger.read())
        if not (resumed and (stage_dir / "test-result.json").is_file()):
            try:
                staged = stage_all(self.run_dir, stage_dir, namespace=opts["namespace"],
                                   models_root=opts["models_root"],
                                   env=agent_env(self.run_dir, extra=self.extra_env))
            except PackageError as exc:
                return "fail", [f"packaging: {exc}"], None
            self.ledger.append("evidence", n, what="package staged",
                               profiles=[p["name"] for p in staged["profiles"]],
                               **evidence_record(self.run_dir, stage_dir / "package.json"))
            test, problems = self._read_test(stage_dir)
            if problems:
                return "fail", problems, None
            self._hardware_phase(spec, stage_dir, test)
        try:
            package_finish(self.run_dir, stage_dir)
        except PackageError as exc:
            return "fail", [f"packaging: {exc}"], None
        gate = self._check_gate(spec, stage_dir)
        return ("pass", [], gate) if gate.ok else ("fail", list(gate.reasons), gate)

    def _check_gate(self, spec, stage_dir: Path):
```

Replace in `orchard/supervisor.py`:

```python
            shutil.copyfile(self.ledger.path, bundle / "ledger.jsonl")
        return self._gate(spec)(stage_dir, self.run_dir)

```

with:

```python
            shutil.copyfile(self.ledger.path, bundle / "ledger.jsonl")
            self._copy_package(bundle)
        return self._gate(spec)(stage_dir, self.run_dir)

    def _copy_package(self, bundle: Path) -> None:
        """Put stage 7's record, publish commands and cards in the operator bundle's package/
        folder, so the operator reads them with the rest. Nothing is copied when no package was
        staged. The bundle scrub then searches these copies too."""
        s7 = self.run_dir / "stages" / "7"
        if not (s7 / "package.json").is_file():
            return
        dest = bundle / "package"
        dest.mkdir(exist_ok=True)
        shutil.copyfile(s7 / "package.json", dest / "package.json")
        if (s7 / "PUBLISH_COMMANDS.txt").is_file():
            shutil.copyfile(s7 / "PUBLISH_COMMANDS.txt", dest / "PUBLISH_COMMANDS.txt")
        for card in sorted((s7 / "package").glob("*/README.md")):
            shutil.copyfile(card, dest / f"{card.parent.name}-README.md")

```

Replace in `orchard/supervisor.py`:

```python
            return "abort"
        if status == "escalate":
```

with:

```python
            return "abort"
        if spec.harness:
            # Supervisor code failed: another model cannot fix it, so it is not escalated.
            self.ledger.append("decision", n, decision="pause",
                               reason=f"stage {n} ({spec.name}) failed; it is supervisor code and "
                                      "is not escalated: " + "; ".join(reasons)[:500])
            self.ledger.append("stage_end", n, result="fail", reasons=reasons)
            return "fail"
        if status == "escalate":
```

Replace in `orchard/supervisor.py`:

```python
                        "ledger records it, and a resumed run keeps it")
    r.add_argument("--gozer", default="gozer")
```

with:

```python
                        "ledger records it, and a resumed run keeps it")
    r.add_argument("--package-format", choices=PACKAGE_FORMATS, default=None,
                   help="stage 7 builds this package on the weights-only path: v6 (a thin bundle). "
                        "v5.1 is refused at start: it needs a container image build. Without this, "
                        "stage 7 is skipped. The ledger records it, and a resumed run keeps it")
    r.add_argument("--package-namespace", default=None,
                   help="the operator's Hugging Face namespace, used in the card and in the "
                        "publish commands (text only; the run never publishes)")
    r.add_argument("--package-models-root", default=TT_MODEL_MODELS_ROOT,
                   help="where tt-model installs bundles; stage 7 looks here for other chip "
                        "counts of the nearest model")
    r.add_argument("--gozer", default="gozer")
```

Replace in `orchard/supervisor.py`:

```python
    return p.parse_args(argv)

```

with:

```python
    return p.parse_args(argv)


def package_option(args) -> dict | None:
    """The stage 7 options as the ledger records them, or None. Refuses a deferred format and a
    format without a namespace."""
    fmt = getattr(args, "package_format", None)
    if fmt is None:
        return None
    if fmt in PACKAGE_DEFERRED:
        raise ValueError(f"--package-format {fmt} is deferred: {PACKAGE_DEFERRED[fmt]}")
    if not args.package_namespace:
        raise ValueError("--package-format needs --package-namespace, the operator's Hugging Face "
                         "namespace for the card and the publish commands")
    return {"format": fmt, "namespace": args.package_namespace,
            "models_root": str(Path(args.package_models_root).expanduser())}

```

Replace in `orchard/supervisor.py`:

```python
    agent_env(run_dir, extra=extra_env)
    found = [str(p) for p in visible_credentials(operator_home() if home is None else home)]
```

with:

```python
    agent_env(run_dir, extra=extra_env)
    package = package_option(args)
    found = [str(p) for p in visible_credentials(operator_home() if home is None else home)]
```

Replace in `orchard/supervisor.py`:

```python
        required = recorded
    return Supervisor(run_dir=run_dir, ledger=ledger, cfg=cfg, model_id=args.model, adapter=adapter,
```

with:

```python
        required = recorded
    added = False
    if progress.started:                        # a resumed run keeps the package it started with
        recorded = package_options(entries)
        stage7_started = any(e["event"] == "stage_start" and e["stage"] == 7 for e in entries)
        if package is not None and package != recorded:
            if recorded is not None or stage7_started:
                raise ValueError(f"this run has package options {recorded}; the options given now "
                                 f"({package}) differ. Leave them out to resume with the recorded "
                                 "ones. Options can be added to a run that has none only before "
                                 "stage 7 starts")
            added = True                        # the run records them before its next stage
        else:
            package = recorded
    return Supervisor(run_dir=run_dir, ledger=ledger, cfg=cfg, model_id=args.model, adapter=adapter,
```

Replace in `orchard/supervisor.py`:

```python
                      clock=clock, sleep=sleep, budgets=budgets, disk_usage=disk_usage,
                      credentials_visible=found, required_chips=required)

```

with:

```python
                      clock=clock, sleep=sleep, budgets=budgets, disk_usage=disk_usage,
                      credentials_visible=found, required_chips=required, package=package,
                      package_added=added)

```

Replace in `orchard/skills/operator-bundle.md`:

```markdown
- `RESULTS.md`: what each stage found, from the stage result files in your context. Every number
  appears with its label (`measured` or `TODO`) and the evidence path behind it. Say that stage 7
  (package and container build) was skipped, so no package exists yet.
- `RISKS.md`: open risks. Include every `TODO` number, every stage 0 hazard and whether a later
  stage dealt with it, anything a gate passed on thin evidence, and what was never tested.
- `PUBLISH_COMMANDS.txt`: the exact commands the operator would run to package and publish, as
  text, one per line, private by default. You never run them. The supervisor refuses publish,
  push and upload commands.
- `card.md` (optional): a draft model card built only from measured numbers.
```

with:

```markdown
- `RESULTS.md`: what each stage found, from the stage result files in your context. Every number
  appears with its label (`measured` or `TODO`) and the evidence path behind it. Say what stage 7
  did. When it staged a package, the supervisor copied its record (`package.json`), its publish
  commands and each package's card into `bundle/package/`: name each package, its chip count, its
  license, and whether its boot check passed. When stage 7 was skipped, say so: no package exists.
- `RISKS.md`: open risks. Include every `TODO` number, every stage 0 hazard and whether a later
  stage dealt with it, anything a gate passed on thin evidence, and what was never tested. When the
  package's license is non-commercial, say so, and say that its card says so.
- `PUBLISH_COMMANDS.txt`: the exact commands the operator would run to package and publish, as
  text, one per line, private by default. You never run them. The supervisor refuses publish,
  push and upload commands. When `bundle/package/PUBLISH_COMMANDS.txt` exists, copy its lines into
  yours word for word, comments included.
- `card.md` (optional): a draft model card built only from measured numbers.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_package.py tests/test_skills.py`
Expected: PASS (27 passed).

- [ ] **Step 5: Mutation checks**

1. Stage 7 is not escalated: in `orchard/supervisor.py`, replace

```python
        if spec.harness:
            # Supervisor code failed
```

   with

```python
        if False:
            # Supervisor code failed
```

   Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_package.py::test_a_scrub_hit_pauses_the_run_at_stage_7_without_escalating_or_leasing`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

2. The run follows its package format: in `orchard/supervisor.py`, replace `run_path(entries, self.run_dir), package_format(entries))` with `run_path(entries, self.run_dir), None)`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_package.py::test_a_weights_only_run_with_a_package_format_stages_and_boot_checks_in_stage_7`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
3. Stage 8 receives the package files: in `orchard/supervisor.py`, replace

```python
            self._copy_package(bundle)
```

   with nothing (delete these lines).

   Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_package.py::test_a_weights_only_run_with_a_package_format_stages_and_boot_checks_in_stage_7`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

4. The v5.1 refusal: in `orchard/supervisor.py`, replace `if fmt in PACKAGE_DEFERRED:` with `if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_package.py::test_v5_1_is_refused_at_start_and_says_why`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
5. Options cannot be added after stage 7 started: in `orchard/supervisor.py`, replace `if recorded is not None or stage7_started:` with `if recorded is not None:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_package.py::test_package_options_cannot_be_added_once_stage_7_has_started`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
6. Added options are recorded: in `orchard/supervisor.py`, replace `if self.package_added and package_options(self.ledger.read()) != self.package:` with `if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_package.py::test_a_run_started_without_package_options_can_get_them_before_stage_7`. Expected: FAIL. Restore the file, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: 1454 passed and 1 skipped (the skip is the opt-in replay).

```bash
git add orchard/supervisor.py orchard/skills/operator-bundle.md tests/test_supervisor_package.py tests/test_skills.py
git commit -m "Run stage 7 in the supervisor and carry its publish commands into the operator bundle"
```

---

### Task 11: Documents: README, CLAUDE.md, the spec and the runbook

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `docs/superpowers/specs/2026-10-01-orchard-design.md`, `docs/runbooks/hardware-validation.md`

**Interfaces:**
- Consumes: everything above.
- Produces: documentation only.

The spec's stage table, denials, scrub paragraph and skills list change to match the code. The runbook gets the packaging entry the controller follows on hardware. The README's test count is the count after Task 10; if the suite reports another number, write that number.

- [ ] **Step 1: Edit the documents**

Replace in `README.md`:

```markdown
| Stage skills (`orchard/skills/`: delta-triage, reference-gate, serving-check, operator-bundle) | local drafts; their home is the tt-model-bringup plugin |
| Model proxy for agents the supervisor did not launch, stage 7 (package and image build) | designed; no code yet |

The suite has 1199 passing tests and 1 skipped, and runs without hardware or network. 854 of them predate plan 4. The skipped test is the opt-in replay of local qwen-code transcripts (`ORCHARD_REPLAY=1 python3 -m pytest tests/test_replay_local.py`). 607 of the tests predate plan 3. The driver's tests use fake gozer roots and a stub child.

```

with:

```markdown
| Stage skills (`orchard/skills/`: delta-triage, reference-gate, serving-check, operator-bundle) | local drafts; their home is the tt-model-bringup plugin |
| Stage 7: a v6 thin package, its boot check, card and publish commands as text (`orchard/package.py`, `orchard/package_card.py`, `orchard/package_templates/`) | built, tested against a fake `tt-model`, fake bundles and a fake server; not yet run on hardware. A v5.1 container package is deferred |
| Model proxy for agents the supervisor did not launch | designed; no code yet |

The suite has 1454 passing tests and 1 skipped, and runs without hardware or network. 1326 of them predate plan 5. 854 of them predate plan 4. The skipped test is the opt-in replay of local qwen-code transcripts (`ORCHARD_REPLAY=1 python3 -m pytest tests/test_replay_local.py`). 607 of the tests predate plan 3. The driver's tests use fake gozer roots and a stub child.

```

Replace in `README.md`:

```markdown

## Try it
```

with:

```markdown

- **Package (stage 7, plan 5).** Started with `--package-format v6 --package-namespace <ns>`, a
  weights-only run packages the model in stage 7 as supervisor code, with no agent. It runs
  `tt-model package-thin --out` from the nearest model's installed v6 bundle with `--weights` naming
  the new model, splices the bundle's fixed vLLM arguments into `run.sh`, and makes `run.sh` build a
  `model-dir/` (the nearest model's config, the new model's tokenizer and weights) and point
  `--model`, `HF_MODEL` and `MODEL_WEIGHTS_DIR` at it. Each staged package is scrubbed (hostname,
  tokens, home paths, caches, weights, the operator's namespace outside the card). A copy is
  installed and booted on a leased board and compared with the stage 1 reference. The card shows
  the license (Hemmingway-1: CC BY-NC 4.0, so it says "Non-commercial use only.") and only numbers
  that name their evidence. The publish commands are text: private `hf upload` lines, live only for
  a boot-checked package. Agents may not run `tt-model package` or `package-thin`. A v5.1 container
  package needs an image build and is refused at start.

## Try it
```

Replace in `CLAUDE.md`:

```markdown
operator. Plan 2 is merged to `main` in tt-gozer. Plan 3 (adapters, server control, park and restore, watchdog) is implemented on branch `plan3-supervisor-behavior`. Plan 4 (stage machine, agent steps, supervisor loop, draft stage skills) is implemented on branch
`plan4-stage-machine`. The Hemmingway-1 run in the runbook has not been run.

```

with:

```markdown
operator. Plan 2 is merged to `main` in tt-gozer. Plan 3 (adapters, server control, park and restore, watchdog) is implemented on branch `plan3-supervisor-behavior`. Plan 4 (stage machine, agent steps, supervisor loop, draft stage skills) is implemented on branch
`plan4-stage-machine`. The Hemmingway-1 run in the runbook has not been run. Plan 5 (stage 7: a v6
thin package, its boot check, card and publish commands as text) is implemented on branch
`packaging-stage7`; its hardware run has not been done.

```

Replace in `CLAUDE.md`:

```markdown
  guard, no failure text in the gate reason, the old finish text.
```

with:

```markdown
  guard, no failure text in the gate reason, the old finish text.
## 2026-10-03: plan 5, stage 7 packages a weights-only model
Prompt (operator): the harness must produce, from a model it brought up, a package that builds as a v5.1 (container) or v6
(thin bundle) tt-model package for the profiles already shipped (2 and 4 chips; 1 optional), and never publish; the run
ends at the operator bundle with the publish commands as text. Plan: `docs/superpowers/plans/2026-10-03-packaging-stage-7.md`.
Key decisions:
- Stage 7 is supervisor code (`orchard/package.py`), with no agent and no model. It runs only on the weights-only path of
  a run started with `--package-format v6` and `--package-namespace`; other runs skip it as before. Options can be added
  on a resume before stage 7 starts. A failure pauses the run and is not escalated.
- The package is built from the nearest model's installed v6 bundle with `tt-model package-thin --out` and `--weights`
  naming the new model. The new model's config names a text-only architecture that the bundle's model class does not
  serve, so `run.sh` runs `prepare_model_dir.py` first: the nearest model's config files (shipped in `base_config/`) plus
  the new model's tokenizer and weights, with `--model`, `HF_MODEL` and `MODEL_WEIGHTS_DIR` all naming that directory.
- The boot check installs a copy, serves it on a leased board (the same hardware phase as stages 2 to 6, parking the
  coder when no board is free), and refuses to compare tokens unless the port was free, `/v1/models` lists the copy's
  model-dir, the server process's own environment names that model-dir, and the model-dir was built from the
  manifest's weights. Its Hugging Face home links only the new model and the drafter, offline.
- Only stage 2's profile (2 chips here) is boot-checked. Other installed v6 bundles of the nearest model give optional
  profiles whose publish lines stay commented out. This machine has no 4-chip v6 bundle (the 4-chip packages are v5.1
  containers), so the 4-chip profile is recorded as skipped. v5.1 is refused at start: it needs an image build.
- The card's license comes from the new model's own card; an unknown license counts as non-commercial. Numbers appear
  only in the card's table, each labelled measured (with evidence holding the value) or TODO.
- Publish commands are private `hf upload` lines; the card's front matter carries the license and the tags. Agents may
  not run `tt-model package` or `package-thin` (with a repo id they upload).
```

Replace in `docs/superpowers/specs/2026-10-01-orchard-design.md`:

```markdown
Status: draft for operator review. Date: 2026-10-01. Author: Claude, with Taylor Singletary.
Plans 1 to 4 are implemented. Not yet done: plan 1 Task 5 (the sizing tool on this machine), plan 3's
park check on hardware, and plan 4's Hemmingway-1 run (docs/runbooks/hardware-validation.md). Every
number below is either cited to a
```

with:

```markdown
Status: draft for operator review. Date: 2026-10-01. Author: Claude, with Taylor Singletary.
Plans 1 to 5 are implemented. Not yet done: plan 1 Task 5 (the sizing tool on this machine), plan 3's
park check on hardware, plan 4's Hemmingway-1 run and plan 5's packaging run
(docs/runbooks/hardware-validation.md). Every
number below is either cited to a
```

Replace in `docs/superpowers/specs/2026-10-01-orchard-design.md`:

```markdown
| 6 | Qualitative check and benchmark | small | measured numbers, each labeled measured or TODO |
| 7 | Package and container build | none; supervisor waits, chips released | image boots and passes the stage 5 checks |
| 8 | Operator bundle | small | results file, ledger, open risks, exact publish commands, scrub check clean |
```

with:

```markdown
| 6 | Qualitative check and benchmark | small | measured numbers, each labeled measured or TODO |
| 7 | Package: a v6 thin bundle (plan 5; a v5.1 container is deferred) | none; supervisor code | each staged package scrubs clean, its run.sh loads the new weights, its card passes the license and number checks, and an installed copy boots on a leased board and agrees with the stage 1 reference (`gate_package`) |
| 8 | Operator bundle | small | results file, ledger, open risks, exact publish commands, scrub check clean |
```

Replace in `docs/superpowers/specs/2026-10-01-orchard-design.md`:

```markdown

- `tt-model push` and `tt-model publish`; `git push` (including `subtree push`, `lfs push`, git
```

with:

```markdown

- `tt-model package` and `tt-model package-thin` in any form (given a repo id, both upload; stage 7
  packages as supervisor code).
- `tt-model push` and `tt-model publish`; `git push` (including `subtree push`, `lfs push`, git
```

Replace in `docs/superpowers/specs/2026-10-01-orchard-design.md`:

```markdown
Bundle scrub check (stage 8): search the package and card for the machine hostname, tokens and
absolute home paths. A hit blocks the bundle.

```

with:

```markdown
Bundle scrub check (stage 8): search the package and card for the machine hostname, tokens and
absolute home paths. A hit blocks the bundle. Stage 7 scrubs each staged package more strictly
(`scrub_package`): no symbolic link, tensor cache, weights file, venv or model-dir may be in it, and
the operator's namespace may appear only in its card. Wheels are binary and are not searched; each
must be byte-identical to the source bundle's.

```

Replace in `docs/superpowers/specs/2026-10-01-orchard-design.md`:

```markdown
- `tt-model-bringup`: `delta-triage`, `reference-gate`, `operator-bundle`. They never name a lease tool.
- `tt-gozer/skills`: `gozer-park` (hold the lease through a swap, reset in place, and release
```

with:

```markdown
- `tt-model-bringup`: `delta-triage`, `reference-gate`, `operator-bundle`. They never name a lease tool.
- Stage 7 has no skill. It is supervisor code (`orchard/package.py`, plan 5).
- `tt-gozer/skills`: `gozer-park` (hold the lease through a swap, reset in place, and release
```

Append to `docs/runbooks/hardware-validation.md`:

```markdown

## Packaging Hemmingway-1 (plan 5)

Purpose: stage 7 packages a weights-only model as a v6 thin bundle and boots an installed copy on
one board. The run publishes nothing. It ends with the publish commands as text.

Who runs it: the controller, with the operator's agreement. It needs one board for the boot check
(or parks the coder when no board is free), the network for the copy's `install.sh`, and 80 GB free
on the run directory's disk (`STAGE_DISK_GB[7]`).

Before running:
- Stages 0 to 6 pass on the weights-only path (the Hemmingway-1 run above).
- The nearest model's v6 bundle is installed: `tt-model list` shows
  `episod/qwen3.8-27b-dflash2-p300`, and stage 2's `swap_config.json` names it as `bundle_dir`.
- `uv` is on PATH, or `install.sh` downloads it.
- The drafter `incoai/Qwen3.8-27B-DFlash2` is in the Hugging Face home stage 2 used (`hf_home` in
  `swap_config.json`).

Run: the Hemmingway-1 command above plus `--package-format v6 --package-namespace episod`. A run
that started without these options can be given them on a resume, as long as stage 7 has not
started; the ledger then records a "package options set" decision. `--package-format v5.1` is
refused at start.

What happens: stage 7 runs `tt-model package-thin --out` once per profile, edits each `run.sh`,
scrubs each package (a hit pauses the run before anything is installed), installs a copy of the
2-chip package under `stages/7/verify/bundle`, and runs `stages/7/verify/verify_bundle.py` as the
hardware test. The copy converts a fresh tensor cache (34 GB measured for the 2-chip cache) and
compiles its kernels into an empty cache (more than 26 min measured on 2026-10-03), so the check
waits up to 3300 s for the server (`PACKAGE_HEALTH_TIMEOUT_S`). The
check refuses to compare tokens unless the port was free, `/v1/models` lists the copy's model-dir,
the server process has `MODEL_WEIGHTS_DIR` and `HF_MODEL` set to it, and `model-dir/.weights` names
`Altworld/Hemmingway-1` at the pinned revision. A failure pauses the run; it is not escalated.

Record afterwards: `stages/7/package.json`, `stages/7/verify/evidence/verify.json` (the top-1
agreement and ready time, labelled measured), the server log's line that names the weights
directory, `stages/7/PUBLISH_COMMANDS.txt`, and the card of each package. Do not run the publish
commands; they are for the operator. A retried stage 7 keeps the earlier attempt's copy, venv and
tensor cache under `stages/7.partial-<k>`; the disk check for the next attempt counts them.
```

- [ ] **Step 2: Sweep the new text and the new code comments**

Run: `git diff -U0 main -- README.md CLAUDE.md docs/superpowers/specs docs/runbooks orchard tests | grep -nE ', not |rather than|instead of|reads as|honest|the one|which is why|This is what'`
Expected: three hits, all in error messages or a comment that state both facts: in `orchard/package.py` (stage 7 runs only ..., not ...), in `verify_bundle.py` (the server on port ... serves ..., not ...) and in `tests/test_package_verify.py` (present, not linked). Rewrite any other hit as a plain statement.

- [ ] **Step 3: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: 1454 passed and 1 skipped (the skip is the opt-in replay).

```bash
git add README.md CLAUDE.md docs/superpowers/specs/2026-10-01-orchard-design.md docs/runbooks/hardware-validation.md
git commit -m "Docs: stage 7 packaging in the README, log, spec and runbook"
```

---

## Hand-off notes

- Nothing here ran on hardware. The real `tt-model package-thin` is beta ("flags and layout may change without notice"); the fake follows the layout of the bundle staged on 2026-09-30. If the real one writes `run.sh` differently, `wire_weights` and `extra_args_from` refuse with a message that names the edit, and stage 7 pauses. Check the first real run's `stages/7/package/<name>/run.sh` by eye.
- The copy's `install.sh` reaches the network (uv, the pip pins, vLLM's `common.txt`) and builds vLLM from source. Its time is not measured; `PACKAGE_INSTALL_TIMEOUT_S` is a choice. Record the install time (`stages/7/verify/install.log` and the ledger timestamps) on the first real run.
- The boot check's Hugging Face home links the new model and the auxiliary repos the manifest's environment names. If the TT runtime fetches something else at start (for example a file of the nearest model), the offline boot fails. That is the check doing its job; record what it needed.
- What the package does not cover, and the card says so under "Not measured": the drafter's acceptance rate on the new model (it was trained on the nearest model), a download through `tt-model pull`, and a boot from the Hub.
- A retried stage 7 moves the earlier attempt aside (`stages/7.partial-<k>`), with its venv and tensor cache. Nothing deletes it. The 80 GB disk check of the next attempt counts it.
- `hf upload --private` makes a repo private only when it creates it; an existing repo keeps its visibility (`hf upload --help`, read 2026-10-03). The publish header says each repo is created private. An operator who reuses an existing repo name checks its visibility first.
- The gate checks shape, evidence and wiring. It cannot tell whether the card's prose is true. The operator reads each card before running a publish command.
- A v5.1 container package remains open. It needs an image build (or an image that serves the new weights with a changed launch) and its own plan.
- Order with the multi-chip plan. `docs/superpowers/plans/2026-10-03-multichip-stage-4.md` is executed first and merges to `main` first. Both plans change `orchard/stages.py`, `orchard/supervisor.py`, `orchard/defaults.py` and `orchard/skills/weights-swap-templates/serve_and_compare.py`. After that merge, the implementer of this plan creates the Task 0 branch from the updated `main` (rebasing it if it already exists) and re-reads those four files before editing them. Where a replacement block no longer matches, the plan's edit is re-applied to the merged text with the same intent; it is not a reason to undo the multi-chip changes. `verify_bundle.py` uses `serve_and_compare.py`'s `wait_healthy`, `complete`, `coherence`, `stop_server`, `ServerHTTPError` and `N_TOKENS`; confirm they still exist. By then `orchard/hwtests.py` and the multi-chip plan file are committed on `main`, so Task 0's check expects a clean tree (apart from this plan file) and a `main` later than `1407417`; the baseline test count is whatever that `main` reports. This plan's fake `tt-model` is `tests/fake_package_thin.py`, so it does not collide with the multi-chip plan's `tests/fake_tt_model.py`.
