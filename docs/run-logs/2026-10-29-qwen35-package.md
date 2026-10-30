# 2026-10-29: package Qwen3.5-35B-A3B for one p300 board

Operator request: package the 40-layer run measured in `2026-10-29-fusion-port.md` (126.8 tok/s, 29.6 percent of the estimated floor).
Nothing is uploaded or published; the operator has not approved publishing this package.

## Decisions

Read first: `orchard/audio_package.py` (stage, card, gate and record pattern), tt-model-manager `main` (`docs/container_packages.md`,
`docs/publishing.md`, `examples/container-example.yaml`) and its open branches (read only, fetched, nothing pushed). The open branch
`fix/card-weights-wording` (2026-10-29, `src/tt_kernel/build.py` and its test) changes where the weights pin is worded on a generated card.
It does not change the manifest schema. Other open branches (`feat/serve-n-chips`, `feat/cohost-kind`, `feat/deploy`) do not touch the fields used here.

Package shape: a recipe package. It is neither a built v5.1 container package nor a v6 thin package.
- A v5.1 package is built by `tt-model package --container`, which needs a tt-metal tree and copies an allowlist of model code into a new
  image. This package changes one file of another author's published bundle and has no tt-metal tree of its own to build from.
- A v6 thin package pins stock `ttnn` and ships model code as `extra_code/`. The fused ops live in the Coder-Next image (a tt-metal fork build),
  so a thin package cannot reproduce the measured setup.
- The recipe keeps the measured image untouched, pins it by image id, mounts a patched copy of the loader at run time, and checks the loader's
  sha256 before and after the patch. `tt-model.yaml` is a draft in the v5.1 vocabulary that records the serve profile (as the audio package's does).
- The weights are a pointer. `fetch_weights.py` reads only the text tensors by byte range at a pinned revision. `hf download` of the same files ran at
  4 to 6 MB/s on this machine and the ranged reads at 35 to 50 MB/s.

License: the model repository's license field is `apache-2.0` (not gated); its `LICENSE` file is the Apache License 2.0 and is copied into the package. Revision
pinned: `59d61f3ce65a6d9863b86d2e96597125219dc754` (read from the Hub API on 2026-10-29). The loader file in the image carries an Apache-2.0 header.

## Progress log

- Worktree `qwen35-package` off `origin/main` (includes PR 236). Weights fetch into a scratch directory started (disk had 233 GB free; 69.3 GB fetch leaves more than 150 GB).
- Module `orchard/qwen35_package.py` (stage, card, gate, publish text) and `tests/test_qwen35_package.py` written; 19 tests pass; 11 mutations of the gate each turned a test red
  (the size limit needed its own test first).
- Fetch: `fetch_weights.py` (the package's own download path, run from the package source) wrote 693 tensors, 69,321,230,976 bytes (the same byte count as the
  earlier slice build) in about 55 minutes at 35 to 50 MB/s, including a 1 GB first batch that held the progress line for several minutes.
- CPU reference: the Hugging Face model in bfloat16 on the CPU, 6 prompts of 32 tokens with a KV cache, under 4 minutes in total.
- Hardware, one ticketed lease (ticket 04c7) on chips 2 and 3, driven from the package source's `run.sh` with an empty `work/` directory:
  cold first boot, quality prompts, chat, long prompt, 5-run benchmark, then a warm restart.

## Results (all from the package's own scripts; raw files in `2026-10-29-qwen35-package-data/`)

| Measurement | Value |
| --- | --- |
| Cold first boot (empty kernel cache and empty tensor cache), container start to first answer | 727 s |
| Warm restart (caches from the cold boot) | 308 s |
| Earlier warm starts in the fusion-port runs (kernel cache seeded from the coder's) | 267 to 272 s |
| Decode, 5 runs of 128 greedy tokens after 1 warm-up | 126.81 / 126.83 / 127.07 tok/s (min, median, max), host load 8.5 to 8.7 |
| Share of the estimated floor (428 tok/s) | 29.6 percent |
| Time to first token, 14-token prompt | 107 to 112 ms |
| Long prompt of 3,817 tokens, code word at the start, question at the end | answered "PELICAN" in 0.84 s |

Cold compile: 727 s covers the kernel compile and the tensor-cache build together; the two were not separated. It was measured once, on a host with other jobs running
(one-minute load average about 8.5), so it is a single sample.

Quality check (6 fixed prompts, 32 tokens, greedy, against the CPU reference):

| Prompt | Leading tokens that agree | First difference (reference gap, device token inside the reference top 5) |
| --- | ---: | --- |
| printing press | 8 | token 8, 0.125 nats, yes |
| Python function | 5 | token 5, 0.375, yes |
| arithmetic | 10 | token 10, 0.375, yes |
| Spanish capitals | 3 | token 3, 0.438, yes |
| photosynthesis | 17 | token 17, 0.125, yes |
| three primary colors | 0 | token 0, 0.375, yes |

Every first difference is a near tie that the device resolved the other way, with the device token inside the reference top 5 and a reference gap of at most 0.44 nats.
After a flip the free-running texts diverge, so the leading counts are short. Read side by side, the device texts are fluent and correct (for example the device wrote
`Red, blue, and yellow.` where the reference wrote `The three primary colors are red, yellow, and blue.`; both answered 391 for 17 x 23). This is not an
accuracy evaluation and a teacher-forced comparison was not run. The 8-token to 17-token prefixes agree with the earlier 16-token fusion-port check.

Chat through `/v1/chat/completions`: with the default template the model opens a thinking section and a 96-token reply ended inside it (finish reason `length`); with
`enable_thinking` false, "Name the three primary colors" gave `Red, blue, and yellow.` and finished normally. The Spanish raw-completion prompt drifted into a thinking block
after its first line on the device (the reference repeated the prompt).

## Gate, scan and the staged package

`orchard/qwen35_package.py` stages into a new directory and `package_problems` is the gate: home paths, host name, tokens, `.tt_ai_key`, `litellm`, weights files and
files over 5 MB, a link, the Apache license text, the pinned revision in the card, manifest, fetch script and notice, the image id in `run.sh`, each measured value in the card,
the cold-start claim against the facts, the executable bit, and a sha256 record of every file. The staged package (24 files, 188 KB) passes with no problems.
Tests: 19 in `tests/test_qwen35_package.py`; each gate check was removed in turn and a test went red (the size limit needed its own test first).

Publish command, as text only (nothing was uploaded; the operator has not approved publishing): 
`hf upload --repo-type model --private NAMESPACE/qwen3.5-35b-a3b-blackhole-p300 ./qwen3.5-35b-a3b-blackhole-p300 .`

## Things to know

- The lease for the boot check was a detached lease (ticket 04c7). Gozer reaped it at 1,078 s, while the warm restart was still to run, because the lease was past its grace window.
  The warm restart (about 5 minutes) therefore ran after the lease had ended and another agent's lease on the same board had started. The log of that agent's run was not read. A future
  run of this check should hold the lease with `--owner-pid` on a process that lives for the whole session.
- The package's `run.sh` has one message edited (the word "one" in an error string) after the boot check; the staged copy was not booted again.
- The 126.83 tok/s median here and the 126.78 in the fusion-port log are two measurements of the same setup; the card uses the package run for the baseline and the earlier run for the
  two ablations (126.84 and 126.78).
- Not tested: more than one sequence, contexts above 3,817 tokens, tool calls, sampling above temperature 0, vision, four dies, a bfp8 or bfloat16 expert policy.

## Review round (PR 246, Copilot): defects fixed

| # | Defect | Fix | Test |
| --- | --- | --- | --- |
| 1 | `run.sh` mounted host devices 0 and 1 whatever the lease granted | Devices come from the inherited `TT_VISIBLE_DEVICES` (a lease exports PCI addresses; each is mapped to its `/dev/tenstorrent/N` node through sysfs, the way `orchard/bus.py` does). `DEVICES` must equal the leased set or the launcher exits 2. With no `TT_VISIBLE_DEVICES` and no `DEVICES` it refuses; with only `DEVICES` it uses them (no lease to check against). Exactly two dies. | 8 launcher tests with a stub `docker`, a fake sysfs and fake device nodes |
| 2 | `max()` of an empty list when every prompt matched fully | The card says all tokens matched and counts the fully matched prompts | all-agree evidence stages and passes the gate |
| 3 | The gate read and hashed files it had already rejected | Links, credential files, weights files and oversized files are found by `stat` only and the gate returns at once | the gate returns exactly one problem when a record-breaking edit is also present |
| 4 | The gate did not compare the card with the evidence | Every card number is recomputed from the raw files in `evidence/` (bench logs, boot lines, both quality files, the model config through `decode_floor`); `facts.json` must equal them and the card must equal `render_card(facts)` | cold start 900 in the card, in `facts.json` and in `boot.txt` each fail; TTFT, warm start, quality rows and median each fail when edited |
| 5 | `compare.py` accepted unequal inputs; `leading_agree` with no difference | It raises on unequal prompt counts, different prompts and unequal or empty token lists; no difference gives the full length | 5 tests |
| 6 | `run.sh` only warned about an image-id mismatch | It exits 2 unless the image resolves to the recorded id; another tag with the same id is accepted | 3 tests |
| 7 | `PATCHED_SHA` was checked only when the patched loader was created | It is checked on every launch | edited leftover loader is refused |

The pinned hashes moved to `pins.env`, which `run.sh` reads and the gate compares with the module's constants. Tests: 44 in `tests/test_qwen35_package.py`. Each new guard was removed in turn and a test went red
(device mapping, lease check, image id, patched-loader check, loader-original check, card-against-evidence, facts check, early return, size and weights check, all-agree branch, pins check).

Boot check of the fixed launcher, from the restaged package, on chips 2 and 3 (ticket 56b6, a lease held through `--owner-pid` on a process that lived for the whole session, released at once):
- With `DEVICES="0 1"` while the lease named chips 2 and 3, the launcher exited 2 with "DEVICES (0 1) is not the leased set (2 3)" before any container started.
- From the lease alone it mounted `/dev/tenstorrent/2` and `/dev/tenstorrent/3` (read back with `docker inspect`), passed the image-id, loader and patch checks, served, and answered a completion.
- This check used a 4-layer slice of the checkpoint fetched with `fetch_weights.py --layers 4` (8.8 GB, a few-GB directory for launcher checks that is not a usable model; its answer was gibberish by construction),
  held on a separate disk. Reason: the root disk had 150 GB free and then 139 GB, and the rule for this task keeps 150 GB free, so the 69 GB full download could not be made. The scratch directory was deleted.
- Not re-measured: the warm start. The 309 s figure is from the earlier boot check and may have overlapped another agent's session. Re-measuring needs the full weights on the fast disk, which the free-space rule did not allow.
  The card now says the warm start was measured once, may have overlapped another job, and was not re-measured. The slice boot took 471 s on the slow disk and is not a timing for the package.
- The cold start (728 s), decode speed (126.83 tok/s median), quality and chat results are unchanged from the first boot check, which ran before the review fixes; they are not re-run. The launcher changes
  (device resolution, checks) do not change the container command that produced them, apart from which host device nodes are mounted.

## Review round 2 (Copilot review of commit eafe0050)

The review resolved the seven earlier findings (device mapping, patched-loader hash on every launch, image id, prompt and token counts, staging against evidence,
early return on oversized files, all-agree runs) and listed four open ones. Each now has a test that was red first.

| Severity | Finding | Fix |
| --- | --- | --- |
| High | The gate executed the package's `check/compare.py` (to recompute the comparison) before checking the record | The comparison is trusted code in `orchard/qwen35_package.py` (`compare_rows`). The package's `check/compare.py` is a standalone copy for users; the gate and staging never import or run it, and it is covered by the record like any other file. A test puts top-level code that writes a marker file into the staged script: the gate does not run it and reports the changed file. A parity test keeps the two copies in step. |
| High | Record keys were used as paths without a containment check | A record path must be relative, made of plain components (no empty, `.` or `..` part, no leading slash, no NUL), must stay inside the package after resolution and must not pass through a link. Otherwise the gate reports it and never opens it. Hashing goes through `safeio.read_bytes` with the 5 MB cap. Tests: `../x`, an absolute path, `a/../../x`, an empty key, `evidence//x`, `evidence/./facts.json`. The resolution check is a second layer: the early link check already rejects any package that holds a link. |
| Medium | A truncated or non-object `facts.json` crashed the gate | Read with `safeio.read_json(expect=dict)`; a failure is one named problem. |
| Medium | A malformed `package-record.json` crashed the gate | The record must be a JSON object with a non-empty `files` object of path to 64-hex sha256; each failure is a named problem. |

Also changed: every package file the gate reads goes through the bounded reader and a non-UTF-8 file is a named problem; the derivation of facts from evidence catches any failure on untrusted evidence
and raises `PackageError`, so a wrong-shaped evidence file is a named problem and a staging error. Tests: 75. Mutations run: the path check, its component rule, the shape check of the record, the dict
requirement on the record, the broad catch on evidence (each red); removing only the resolve-and-contain check leaves the tests green because the link check rejects first.

## Review round 3 (Copilot review of commit f841ed5f): four medium findings, no high

That review resolved the two high and two medium findings of round 2 and listed four new medium ones. Each has tests that were red first (16 failures before the fix, 98 passing after).

| Finding | Fix |
| --- | --- |
| A log with one measured run was labelled "benchmark" | Evidence logs (the package run and both ablation runs) need at least 3 measured runs (`docs/benchmark-rules.md`); fewer is a staging error and a gate problem |
| min, median and max came from the log's summary fields | They are recomputed from `runs[*].decode_tok_s`; a summary that disagrees to two decimals is refused, for the ablation logs too |
| Evidence was reopened with ordinary reads (a FIFO hangs, a large file is read whole) | Every evidence and source file is read once through `safeio.read_bytes` (5 MB cap, non-blocking); the facts and the copies in the package use those same bytes. Tests with a FIFO in the evidence, in the source and in a staged package finish within 10 seconds; an oversize log is refused |
| The gate did not require every support file | The required list is every source file and every generated file, whatever the record says; `NOTICE`, `tt-model.yaml`, `docs/running.md` and `publish-commands.txt` must equal their generator's output |

Mutations run, each red: the minimum repeats, the summary comparison, the capped reader, the generated-file comparison, the required list, and reading evidence outside the reader. The real evidence stages and passes the gate (checked after the changes).
Per the operator's rule, no further Copilot review was requested: the last review listed no High finding.
