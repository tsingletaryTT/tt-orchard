# tt-lang pair: a managed venv, the build-info record, and the kernel probe's hardware phase on the pair (2026-10-29)

Operator instruction: "keep working the ttlang things". This continues PR 228, which pinned the verified pair (tt-lang `fec38c6de`,
tt-metal `0e9d200db976`, wheel `tt-lang-light 1.1.10.dev20261026`). Five tasks, in the order below. `<home>` stands for the operator's
home, `<pair-venv>` for the managed venv. Records are in `evidence/ttlang-next/`. Those files are this run's own format and carry the
`probe_id` of each probe, so a later verifications record (the `tt-orchard updates` work in another PR) can be joined to them.

## A. The pair venv as a managed thing

`tt-orchard kernel-env --pair` (orchard/ttlang_pair.py) prints five commands built from the constants in `orchard/ttlang_pin.py`:
download the two wheels, compare their sha256 with the pin, `uv venv`, install the checked wheel files, and record the exports of
`tt-lang-setup-external-tt-metal` for the host tree in `<pair-venv>/orchard-pair-exports.sh`. It then reports what the venv holds, read
through the venv's own python with `importlib.metadata`, with no import of `ttl` or `ttnn`. Nothing runs without `--apply`. The default
venv is `<cache root>/ttlang-pair/venv` (`$ORCHARD_PAIR_VENV` or `--pair-venv-dir` move it). The safetensors note is printed with the plan.

Running `--apply` for real found one fault in the first version: the S3 index is one HTML page of wheel links, so `pip download` needs
`--find-links` and found no version with `--extra-index-url`. That is fixed and has a test. With the fix the two downloaded wheels matched
the pinned sha256 values, the venv was built, and the check printed `ok` for the venv, `tt-lang-light 1.1.10.dev20261026`,
`tt-lang 1.1.10.dev20261026+light` and the exports file. No device was used.

## B. The consistency row

`ttlang-consistency --build-info-file FILE` reads `tt_lang_commit`, `tt_metal_commit` and optionally `version` and `tt_metal_tag`.
`scripts/ttlang_build_info.py` runs inside the pair venv under a lease, calls `ttl.build_info()` and writes that file. The real keys of
`ttl.build_info()` (from `evidence/ttlang-pair/results/pair-chip.jsonl`) are `version`, `ttlang`, `tt_metal_tag` and `tt_metal`, and the
script renames them. It was run once, under `gozer run --chips 1 --exact 2` (board 1, released and reset by gozer when the command ended):

```
{"tt_lang_commit": "fec38c6de4b6f24ce4cb359f04aecc8d4577d6f0", "tt_metal_commit": "0e9d200db976120c129ab0deb13aa3f6d972b723",
 "version": "1.1.10.dev20261026+light", "tt_metal_tag": "v0.79.0-dev20260917", ...}
```

With the file, the row prints `ok   hardware tree: the hardware venv has tt-lang fec38c6de and tt-metal 0e9d200 (v0.79.0-dev20260917), the pinned
pair; the host tree is v0.80.0`. Without it the row is still `todo`. Files: `evidence/ttlang-next/build-info.json`,
`consistency-with-build-info.txt`. The kernel probe's own report still shows the row as `todo`, because the probe does not call
`ttl.build_info()` itself.

## C. The probe's hardware interpreter

The hardware phase now runs in the pair venv when `<pair-venv>/bin/python` and a valid exports file exist. `--shared-python` still wins.
When the venv is missing or has no valid exports file, the probe prints a warning that names `kernel-env --pair --apply` and runs on the
old shared venv (tt-lang 1.1.6). For a pair python the gozer command is `env TT_METAL_HOME=... PYTHONPATH=... LD_LIBRARY_PATH=... <python>
<harness>`, because the light wheel has no ttnn of its own. Each new guard was removed once and a test went red (11 mutations).
The smoke test of the probe (upstream's `eltwise_add.py`) printed `PASSED!` through this path in task D.

## D. Reductions on the pair, hardware phase

Pre-set: the local coder (Qwen3-Coder-Next, port 8011, client use only), temperature 0.7, seed 2026120100 + k, 5 attempts per op, 900 s wall
per attempt, `--library none`, `--no-agent-hw`, `ORCHARD_PROBE_CHIP=2` (board 1), no outside model, one process at a time. Command:
`tt-orchard kernel-probe --op OP --attempts 5 --temperature 0.7 --seed 2026120100 --phase hw --library none --wall-s 900 --no-agent-hw`.
The coder decoded at 85 to 107 tok/s. The attempts ran past 900 s by up to one tool timeout, as the report's budget note says.

| Op | Attempt | Simulator | Failure class (first line) | Chip compile | Chip PCC |
| --- | --- | --- | --- | --- | --- |
| rmsnorm | 1 | fail | numeric_mismatch (PCC 0.9928, bar 0.999) | not run | - |
| rmsnorm | 2 | fail | tt_lang_error (shape mismatch (1, 1) vs (1, 8)) | not run | - |
| rmsnorm | 3 | fail | numeric_mismatch (PCC 0.9928) | not run | - |
| rmsnorm | 4 | fail | tt_lang_error (block-to-block copy requires byte_count) | not run | - |
| rmsnorm | 5 | fail | shape_mismatch (tensor (64, 32) vs block (2, 8)) | not run | - |
| row_sum | 1 | pass | - | compiles and runs | 0.249 (relative L2 3.88; bar 0.99) |
| row_sum | 2 | fail | tt_lang_error (col slice 1:2 outside the tensor) | not run | - |
| row_sum | 3 | fail | tt_lang_error (pop(): not a valid next dataflow step) | not run | - |
| row_sum | 4 | fail | tt_lang_error (DataflowBuffer has no attribute store) | not run | - |
| row_sum | 5 | pass | - | compile error: `TTL subscript base must be a named value` | - |

Totals: rmsnorm 0 of 5 pass the simulator; row_sum 2 of 5. Of the 2 row_sum simulator passes, 1 compiled and ran on the chip and
missed the hardware PCC bar badly, and 1 did not compile. No kernel passed on the chip. rmsnorm reached no chip phase, so the pair's
compiler was not exercised on a coder-written rmsnorm.

What went wrong on the chip, as observed (no cause is claimed):

- `row_sum` attempt 1 (`evidence/ttlang-next/row_sum-attempt-1-kernel.py`): the worst entries are in output columns 1 to 3 of some rows,
  where the reference holds 0.0 and the chip gave a value of the same size as the row sum (for example -53.5 against 53.3 in magnitude).
  The slope against the reference is 1.001 and the PCC is 0.249. The simulator accepted the same file at PCC 1.0.
- `row_sum` attempt 5 reads the grid as `ttl.grid_size(dims=2)[1]` and uses `-(-row_tiles // grid_rows)` inside the threads. The chip's
  error is a different one from the `UnaryMinusOp` error in the chip check log, and the file has both constructs. Which one raised it was
  not isolated.

The probe's own hardware phase for `rmsnorm` and `row_sum` ended with "hardware busy: gozer queued the request": another agent held board
1 when the probe reached the chip phase (`gozer status` showed another lease on chips 2 and 3, later a queue). For `row_sum`, the hardware
phase was therefore rerun on the finished probe directory with `evidence/ttlang-next/hw_only.py` (one smoke test, then each simulator
pass; it retries every 5 minutes while gozer queues). It used the same interpreter choice, chip and harness. `rmsnorm` had no simulator pass,
so it needed no rerun. Raw output: `hw_only-row_sum.log`, reports `kernel-probe-report-*.md`, per-attempt record `task-d-attempts.json`.

Context only, no comparison claimed: the ladder run (2026-10-28, a different coder load, 9 and 3 attempts, simulator only) gave `row_sum` 6 of 9
and `rmsnorm` 0 of 3, and earlier rmsnorm rounds gave 0 of 19. Five attempts per op cannot tell a 0 percent rate from a 40 percent
rate, and the runs differ in the coder's load, seed base and sample size, so this run neither shows nor rules out an effect of the pair.
The simulator does not depend on the pair: the pass counts above are simulator results of the pinned checkout, as before.

New fact from this run: a simulator pass that compiles on the pair can still fail the chip's numeric bar (`row_sum` attempt 1), so the
earlier statement "a simulator pass predicts a compile and a correct run" holds for the 17 kernels of the chip check and not for this coder-written one.

## E. What is left

- Moving the pair venv's build-info call into the probe so its report shows the `hardware tree` row as `ok`.
- The `row_sum` attempt 1 numeric difference between simulator and chip (a reduction's output tile on the chip against the simulator) is
  not explained. It is the next thing to look at and to teach the coder, if it holds for a second kernel.
- Isolating which construct raised `TTL subscript base must be a named value`.
- Issue 205 stays open until the coder-facing reduction guidance is tested on the pair; issue 95 until the hardware row is fed by the probe;
  issue 96 is not closed by 5 attempts.
