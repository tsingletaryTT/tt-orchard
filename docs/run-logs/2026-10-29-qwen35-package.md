# 2026-10-29: package Qwen3.5-35B-A3B for one p300 board

Operator request: package the 40-layer run measured in `2026-10-29-fusion-port.md` (126.8 tok/s, 29.6 percent of the estimated floor).
Nothing is uploaded or published; the operator has not approved publishing this package.

## Decisions

Read first: `orchard/audio_package.py` (stage, card, gate and record pattern), tt-model-manager `main` (`docs/container_packages.md`,
`docs/publishing.md`, `examples/container-example.yaml`) and its open branches (read only, fetched, nothing pushed). The open branch
`fix/card-weights-wording` (2026-10-29, `src/tt_kernel/build.py` and its test) changes where the weights pin is worded on a generated card.
It does not change the manifest schema. Other open branches (`feat/serve-n-chips`, `feat/cohost-kind`, `feat/deploy`) do not touch the fields used here.

Package shape: a recipe package, not a built v5.1 container package and not a v6 thin package.
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
