Three small hardening items left by PR 786 (stage 2 helpers for a model that ships its own code) and PR 791 (per-layer-type rope). One commit each.

## 1. The audit of the model's own code survives a hang (commit 60b5dc07)
When transformers refuses a repository with custom code, stage 1's `allow_remote_code` writes the list of the model's `.py` files with their sha256 and the scrubbed variable names, and only then lets the code import. In the stage 2 construction helper that file went to a temporary directory of the helper process. The supervisor never knew the path, a clean exit removed it, and a model whose code hangs or is killed left no audit.
- `write_effective_construction` makes `stages/N/evidence/construction-remote-code/` beneath the run directory (`mkdir_beneath`, no link followed; an audit from an earlier attempt is removed), passes it to the helper as an absolute last argument, and records the file in the ledger afterwards, also after a timeout, a kill or an error (`remote-code audit recorded`, or `remote-code audit unreadable` with a reason).
- The helper writes it atomically: a new file beside it (`O_EXCL | O_NOFOLLOW`), mode 0o644, `fsync`, `os.replace`.
- `port_construction.read_audit` reads it beneath a pinned directory, limited to 1 MiB and 10,000 files, validates every field, and hashes the bytes it parsed. It is a registered hostile reader (42 cases).

## 2. Stage 1's scrub keeps `TOKENIZERS_PARALLELISM` (commit bd49b2f7)
The credential pattern matches `TOKEN` inside that name, so the route removed a flag of the tokenizers library and listed it as a scrubbed credential. An exact-name allowlist (`NOT_A_CREDENTIAL`, case sensitive) holds it. The pattern is unchanged: `TOKEN(?!IZERS)` would let `MY_TOKENIZERS_PARALLELISM` through.

## 3. No `float()` on a config number that a float cannot hold (commit 50ebb919)
The flat-rope path of the generator called `float(theta)` and `float(scaling)` after `_number`, which lets a 400-digit integer through, and printed `partial_rotary_factor` digit by digit into a stub reason. They now use `_float_of`, refuse with a named reason, and show the factor through `_shown`. The five counted calls are gone and `orchard/port_template.py` leaves `tests/finite_baseline.json`.

## Evidence
- Test first, each seen red for the stated reason. New tests: 27 in `tests/test_construction_remote_audit.py` (the kill test runs the real helper in a bounded child on a model whose configuration module sleeps forever, and checks the audit after SIGKILL), 2 in `tests/test_reference_gate_remote_code.py` (one is a table of 15 scrubbed and 4 kept names), 1 in `tests/test_port_construction_remote.py`, and 24 in `tests/test_port_template_flat_rope_overflow.py`.
- Mutations: item 1, 19 of 20 caught (the survivor, making the audit directory through a link, is covered by the `unlink_beneath` that refuses the same link; removing both is caught). Item 2, 6 of 6. Item 3, 7 of 7.
- Targeted run before this PR: the four files that broke `main` once (`test_cheat_sheet_wiring`, `test_decoder_idioms`, `test_run_state_followups`, `test_run_file_readers`), the PR 786 and 791 test files, the hostile-reader registry, both ratchets and `test_readme`: 5,686 passed, 2 skipped. I did not run the full suite.
- A first version of the supervisor change dropped the `remote_code` field that PR 786 added to the "written" decision; PR 786's own test caught it and it was restored.

## Checklist (docs/pr-hardening-checklist.md)
1. Wrong type or shape in input: `read_audit` checks the type and bounds of every field (a dict, `used` true, names and hashes by pattern, at most 10,000 files, at most 100 variable names). A config number goes through `fits_float`.
2. Files that something else controls: the audit directory is made and cleaned with `mkdir_beneath` and `unlink_beneath` under a pinned run directory; the file is read through a pinned directory, size-limited, no link followed; the helper's temporary file is `O_EXCL | O_NOFOLLOW`. A link where the directory goes is refused (test).
3. Check, then use: one read. `read_audit` hashes the bytes it parsed and the ledger carries that hash.
4. Evidence that an agent can write: the model's code runs with the user's permissions and could rewrite the audit after the helper wrote it. The entry's note says so. The audit feeds no gate.
5. A verdict stronger than its evidence: no verdict depends on it; it is recorded only.
6. The check looks at a different object than the action uses: the hash and the content come from the same bytes.
7. A parser or matcher too broad or too narrow: name and hash patterns are full matches; the scrub allowlist is one exact name and the pattern is unchanged (table test in both directions, mutation for a prefix and for a case-insensitive match).
8. Failure paths that leave state behind: the atomic write leaves no half file, and its temporary file is removed on an error (test). A kill in the instant between the write and `os.replace` leaves a `.tmp` beside the audit; the next attempt does not remove it.
9. Numbers and sizes: 1 MiB, 10,000 files and 50 files shown are labelled choices in the code; the safe JSON reader refuses an integer of more than 100 digits. The generator's config numbers go through `fits_float`.
10. Secrets, paths and host names: the audit holds variable names, file names and hashes, never values (test). The ledger entry holds a path relative to the run directory.
11. Tests that do not test the claim: each guard has a mutation that turns a test red. The 200 second deadline in the kill test bounds only a failure; no clock decides a pass.
12. Housekeeping: change fragment `changes/harden-remote-code-audit.toml`, a dated entry at the end of `docs/project-log.md`, `scripts/version.py check` passes. No new flag, so no README change.

## Not changed, and why
`agent.SECRET_NAME` has the same `TOKEN` match and drops `TOKENIZERS_PARALLELISM` from agent shells; that is outside this change. The excerpts helper parses the model's modeling file and never runs it, so it has no audit.
