# Full port stage 6: quality against the reference and a benchmark (issue 329, stage 6 half)

Branch `full-port-stage-6`, from the local `integration` branch. No PR yet.

## What it does

Stage 6 of a full port has no agent step. The supervisor copies the port that passed (stage 3's when stage 3 ran, else stage 2's) into `stages/6/port/`, copies the shipped
template `port_quality.py`, runs `python3 -P -s stages/6/port_quality.py` under a lease, and writes `result.json` from the template's record. The record holds
teacher-forced perplexity and top-5 logit gaps against the stage 1 reference, a 64-token free run, and a benchmark (one warm-up and three measured runs) with the share of
the decode floor. `result.json` carries a `numbers` list in which every figure is labelled `measured` or `TODO`. The operator bundle prints a section for it and lists the TODO items in RISKS.md.
`--full-port-rehearsal` waives `coherent` and nothing else. Issue 329 still has its stage 5 half (another fork).

Files: `orchard/full_port_quality.py` (gate, derive), `orchard/full_port_quality_stage.py` (supervisor hooks), `orchard/full_port_bundle.py`, `orchard/skills/full-port-templates/port_quality.py`,
`orchard/skills/full-port-quality.md`; small edits in `supervisor.py`, `stages.py`, `build_bundle.py`, `tests/test_skills.py`, README.

## The twelve checklist items

1. Wrong type or shape: `record_problems` checks every array and each benchmark run (type, length, finite, bounds) before any arithmetic; the bundle section prints TODO for a field of the wrong type. Tests: `test_logprobs_that_are_not_log_probabilities_are_refused`, `test_a_decode_speed_that_is_not_steps_over_time_is_refused`, `tests/test_full_port_quality.py` helpers. A parametrized None/NUL/huge-number sweep of every field was not written.
2. Files something else controls: all reads in the gate and the derive go through `safeio` (`full_port._json`, `_sha`, `read_json`, `atomic_write`); the template's own raw reads are marked `# hardening: ok` with reasons; the port copy uses the stage 3 copier (pinned root, no links, caps). `test_a_link_in_the_port_directory_fails_the_template`. The hostile-input registry was not extended.
3. Check, then use: the record names the stage 1 files and the model config by sha256, and the gate compares them with the disk now (`test_a_stage_1_top_five_file_changed_after_the_test_is_refused`, `test_a_model_config_changed_after_the_test_is_refused`). The ledger holds the copied port's hashes before the test runs.
4. Evidence an agent can write: no agent writes any stage 6 file. The marker must be the file the supervisor recorded (`_check_gate`, mutation `supervisor-marker-from-ledger`); the output carries the record's hash once; the template is the shipped file.
5. Verdict stronger than evidence: every number in `result.json` is recomputed from the record (`result-recomputed`); the floor says it is an estimate and names its dtype policy source; things not measured are TODO lines. The limits are in the result and the bundle.
6. Check looks at another object: the port that is measured is tied to the ledger's copy hashes and to the source stage's own record (`test_a_port_that_changed_its_own_source_during_the_test_is_not_the_port_that_passed`, supervisor test with a port that rewrites itself only under `stages/6/`). The chip count comes from `--required-chips`, not a file (`test_with_chips_configured_stage_6_measures_stage_3s_port_on_all_four`).
7. Parser too broad or narrow: the output mark is matched as a whole line, exactly one line may begin with it; the template prefixes every worker line. A port that prints the mark fails the run.
8. Failure paths: a failed test leaves no result.json and the stage fails with the reason; the lease and the coder are given back by the existing hardware phase (`test_the_hardware_test_of_stage_6_runs_under_a_lease_the_supervisor_took`). A kill between the test and the derive is handled by the resume rule (marker present: no second test).
9. Numbers and sizes: positions bounded at 64, free run 64, benchmark steps 32; logprobs at or below zero; share between 0 and 1.5; at least 16 decode steps per run. Not tested at 10**20.
10. Secrets and paths: worker error text goes through the same `clip` as stage 2; the bundle text goes through the bundle's redactor. No new output carries host paths beyond the model snapshot path in the stage's config file.
11. Tests that test the claim: the template runs as a real subprocess on the CPU against a fake `ttnn`; 27 mutations in `scripts/mutations/full_port_stage_6.py` each turn a test red (table below). The fixtures are tiny, so the floor-share bound is tested with a very large config instead.
12. Housekeeping: change fragment `changes/full-port-stage-6.toml`, project-log entry, README note (no new flag), `scripts/version.py check`.

## Tests and mutations

See the final report in the session; the runner refuses to start on a dirty tree and restores in a trap.

## Not tested

No chip run. A port that returns the reference ids without computing them is not detected. The pause before stage 4 still stands until the stage 4 and stage 5 forks merge.
