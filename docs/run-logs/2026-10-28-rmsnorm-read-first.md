# rmsnorm: does reading the specification's block sections first help the coder? (2026-10-28)

## Pre-registration (written 2026-10-28, before arm A was run)

Question. The coder (Qwen3-Coder-Next) writes a tt-lang `rmsnorm` kernel and has passed the simulator in 0 of 19
earlier attempts (reports rms1 to rms5). A read-only analysis of those 19 attempts found that the coder reads on average
306 of the 1,619 lines of `reference/TTLangSpecification.md` (19%), that the section "Block states" (lines 743 to 764)
was read in 0 of 19, and that `pop(): not a valid next dataflow step` ended 4 of 19, with its explanation in that
section. Is the missing information not read, or read and not found?

Change. One. Arm B's skill (`orchard/skills/kernel-probe.md`) tells the coder to read the sections "Dataflow buffer",
"Block" and "Block states" of the specification in full before it writes a kernel (finding them with `grep -n`, reading
them with `read_file` offset and limit), and to read "Block states" again when a check reports a dataflow error. It names
section titles and contains no worked solution or op-specific fix. Arm A is `origin/main` unchanged.

Library mode (a deviation from the brief, which said to use the default). The default `--library all` puts the
orchard library's verified `rmsnorm` kernel into `reference/` as `library_rmsnorm.py`. With it either arm could pass by
copying that file, which says nothing about reading the specification, and the 19 earlier attempts ran without it (their
system prompt has no library paragraph). Both arms therefore use `--library none`, the condition of the 19 attempts.
The skill's sentence about `reference/library_*.py` stays in both arms and then describes files that are absent; that
costs both arms the same.

Fixed in both arms: op `rmsnorm`; `--library none`; `--phase sim`;
temperature 0.7; the same `--seed` base (the probe adds the attempt number, so attempt k has the same seed in both
arms); 5 attempts; 1800 s wall budget and 40 turns per attempt; the same coder boot, same tt-lang pin and same
simulator environment; arm A first, arm B second. No outside model is used.

Measurements, per attempt (taken from the attempt's report and its `log/probe-*.jsonl` with
`2026-10-28-rmsnorm-read-first-analysis.py`):

| Measurement | Definition |
|---|---|
| passed in the simulator | `passed_sim` in the attempt record, from the unedited check |
| failure class at the end | deadlock, pop invalid step (any error whose text is "not a valid next dataflow step": `pop()`, `store()` or `copy`), shape mismatch, numeric, wall-clock (the attempt ended on the wall budget with no simulator result), other. The class comes from the first line of the unedited check's answer; how the attempt ended (`status`) is listed beside it |
| turns to the first simulator result | the turn whose tool call produced the first `kernel_check.py` result |
| "Block states" read | count of `read_file` calls on the specification whose returned lines cover lines 743 to 764 in full, plus shell calls (grep, sed, head) whose output prints the section's body |
| `read_file` calls | all `read_file` calls in the attempt, any file |
| coder decode speed | tok/s the probe measured before the attempt |

Success criteria (fixed now). Arm B succeeds if either:
1. at least 1 of its 5 attempts passes in the simulator, or
2. its `pop()` / `store()` dataflow-step failures are clearly below arm A's, and its deadlock failures and its numeric
   failures are not higher than arm A's.

If neither holds, the report says that reading the section does not help and the failures are design failures.
Counts only, never rates. Five attempts per arm cannot show a small difference: a change of one attempt in a failure
class is inside what a rerun of the same arm could produce.

Baseline for context only (not a control: earlier skill text, so not comparable): rms5, 5 attempts, 0 passed,
failures deadlock 1, `pop()` 1, numeric 2, shape mismatch 1; rms4: 0 passed, deadlock 3, `pop()` 1, shape mismatch 1.

Clarification added after the pilot and before arm A. The pilot (one arm B attempt, not counted) ended with
`Cannot perform copy (as destination): not a valid next dataflow step`. The class "pop invalid step" is therefore
every "not a valid next dataflow step" error, whatever operation it names, and the success criterion's "`pop()` /
`store()` dataflow-step failures" means that class. The pilot ran with the same seed base and library mode as the arms.
