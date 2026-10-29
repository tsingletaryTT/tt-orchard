# Granite-4.0-H-1B: package, boot check and private upload (2026-10-29)

Operator assignment: make the traced composed path usable as a model, write an OpenAI-compatible chat server, package it like the audio
package, boot-check it on the chip, and upload it as a private repository for review. The generic text-model route of the harness
(issues 61, 90, 214) is built elsewhere and is not part of this work. The runtime keeps a small boundary (`GraniteChip`: `reset`, `step`,
`prefill`, a `max_len` and a vocabulary size; the server's `ChipEngine` adapter is 12 lines), so Granite can plug into that route later.
Code is in `pkgsrc/` (runtime, server, tools), `orchard/granite_package.py` (staging and gate) and the evidence in `evidence/granite-package/`.

## Result

- Private repository `episod/granite-4.0-h-1b-blackhole` holds the package (32 files, no weights). It did not exist before. The first upload
  was commit `007da2b`; the card said `./run.sh`, which fails on a copy pulled from the Hub (the Hub does not keep the execute bit), so the card and manifest
  now say `bash run.sh` and the repository is at commit `1ed61f1`. Nothing was made public.
- The copy pulled back from the Hub is byte-identical to the staged package (sha256 of every file; the Hub adds its own `.gitattributes`), passes the gate, and boots:
  the server printed its ready line and passed all 9 smoke checks (`evidence/granite-package/results/bootP`).
- Decode through the server: 34.4 tok/s. First token for a short streamed chat: 0.63 s. A 1,481-token prompt took 39.6 s to the first token (prefill is a loop of decode steps, 37.4 tok/s).

## The runtime

`pkgsrc/runtime/granite_chip.py` is the earlier traced composed path (see `2026-10-29-granite-repeat-trace.md`) made into a model:

- Context limit is now a setting (`max_len`, a multiple of 32). The key/value cache is that long and the one-hot write and the attention read it all every step.
  Decode speed against the limit with the float32 state (one session, 5 repeats, calm host): 35.4 tok/s at 512, 34.8 at 1,024, 34.3 at 2,048, 32.4 at 4,096.
  With a bf16 state, measured earlier: 37.9 at 128, 37.4 at 512, 36.5 at 1,024, 33.9 at 4,096. The cache itself is small (8 KB per token over the four attention
  layers). **The package default is 2,048**, because the longest sequence checked against the CPU reference is 2,048 tokens and the step time is 3% above the 512 setting.
- Prefill is a loop of decode steps. The logits are read back after the last prompt token only. Measured: 37.4 tok/s for 128, 512 and 1,024-token prompts (34.4 tok/s when
  the logits are read at every step). A chunked scan for the prompt is not built, so a 1,000-token prompt waits about 27 seconds.
- Reset between requests zeroes the recurrent state and leaves the key/value cache alone (positions not written yet are masked, and a masked stale value
  contributes exactly zero).
- Greedy decoding is the measured path. A temperature above 0 samples on the host; sampled output was not evaluated.

### An accuracy problem found on the way, and its fix

The earlier port had only been compared with the reference to 40 positions. On a 1,024-token natural-language sequence (the model's own card, teacher forced against fp32
transformers on the CPU) the bf16-state port had a mean relative L2 of 0.0414 (maximum 0.62) and top-1 0.955, against 0.0084 and 0.973 for the bf16 CPU run, and
the error grew with position (0.013 for positions 0 to 127, 0.084 for 896 to 1,023). The state update multiplies the state by the decay, a value close to 1,
and rounds the product and the state to bf16 at every step; that rounding removes most of the decay. Keeping dt, the decay and the state in float32 on the chip
gives 0.0134 and top-1 0.978 on the same sequence, and the four short prompts stay at the earlier level. The cost is 8% of decode speed
(37.0 to 34.3 tok/s, same host condition). The package default is float32; `--ssd-precision bf16` is the old path. A variant that keeps the state in float32 and
reads it out in bf16 (not shipped) gave the same accuracy (0.0129, top-1 0.982) and no clear speed difference under the host load of that session, so it was dropped.
Evidence: `results/sweep1`, `results/diag1`, `results/diag2`.

## Agreement with the CPU reference (default configuration)

| Run | Relative L2 mean, 4 short prompts | Top-1 | First difference at token |
| --- | --- | --- | --- |
| bf16 CPU run (floor) | 0.0047, 0.0121, 0.0043, 0.0111 | 1.0, 1.0, 0.875, 1.0 | 3, 24, 8, 24 |
| this runtime | 0.0066, 0.0180, 0.0076, 0.0184 | 1.0, 1.0, 0.917, 1.0 | 3, 24, 13, 24 |

2,048-token sequence (the model's card followed by an English project log): relative L2 mean 0.0165 (maximum 0.256), top-1 0.984, against 0.0104, 0.138 and 0.978 for the
bf16 CPU run. The relative L2 is 1.4 to 1.8 times the floor on the short prompts and 1.6 times on the long sequence. The limits: four prompts of 24 tokens, one long sequence of one
kind of text, one run.

## Speed (default configuration, one session of five benches; each cell is 5 repeats after a warm-up, batch 1)

Traced 34.3 tok/s (34.2 to 34.5), which is 19.6% of the one-die bf16 floor of 175.2 tok/s; untraced 24.0 tok/s (23.7 to 24.4). A repeat of the traced run in another session: 34.35.
Optional kernel modes (bf16 state, `state_full_ro`) at the same limit: 38.8 tok/s against 37.1 for composed with a bf16 state, one run each; not the default and not re-checked for agreement
with this runtime. Host load average was 3 to 5 for these sessions (the load of other agents is in each `conditions.txt`).

## The server

`pkgsrc/server/serve.py`, modelled on `scripts/minicpm3/serve.py`: standard library plus `tokenizers` and `jinja2`; `GET /v1/models`, `GET /health`, `POST /v1/chat/completions`
(`messages`, `max_tokens` or `max_completion_tokens`, `stream`, `stop`, `temperature`, `top_p`, `seed`). The prompt is the model's own `chat_template.jinja` (the same ids as
transformers' `apply_chat_template`, checked in `tests/test_granite_serve.py`). Generation ends at the end-of-text token (id 100257, from `config.json`), a stop string, `max_tokens`
(default 256, capped at 4,096) or a full context; `finish_reason` is `stop` for the first two and `length` for the others, taken from what ended the loop. It refuses tool and document inputs,
a prompt longer than the context and malformed bodies with a 400, binds 127.0.0.1 only, and refuses the ports that belong to other work on this machine.
Tests: 18 server tests (no chip; fake engine). Three mutations each turned a test red (the context cap, the end-of-text stop, the finish reason on a stop string).
`tools/smoke.py` is the live check (9 checks).

## The package and the gate

`python -m orchard.granite_package stage ...` writes: `README.md` (the card, front matter `license: apache-2.0`, `base_model`, `pipeline_tag: text-generation`), `LICENSE`
(Apache License 2.0), `NOTICE`, `tt-model.yaml` (a draft v5.1 manifest of kind `http-server`, not built), `run.sh`, `requirements.txt`, `package-record.json` (sha256 of every file),
`runtime/`, `server/`, `tools/`, `runtime/kernels/` (the three coder kernels, unchanged) and `evidence/`. No weights.
The card's numbers come from `evidence/summary.json` (built from the raw result files by `scripts/build_summary.py`), and the gate re-derives every number row and refuses a card that differs.
`python -m orchard.granite_package gate DIR` checks: no weights, link or oversized file; no home path, host name, token, `.tt_ai_key` or `litellm` (the audio package's scrub, reused, plus the
two names); the required files; every Python file compiles and `run.sh` passes `bash -n`; `LICENSE` is the Apache License 2.0; the card's front matter, sections, overclaim words and number rows;
`tt-model.yaml` is v5.1, kind `http-server`, pins the revision, has a `{port}` placeholder, names scripts that exist; the hashes recorded at staging. 20 tests, 16 of them mutations
(each makes the gate fail with a named problem). The staged package, and the copy pulled from the Hub, passed with 0 problems. A manual grep for the host name, the user name, home paths,
`.tt_ai_key`, `litellm` and token patterns found only the public repository name in the card's provenance list.

License: the model repository's metadata says `license: apache-2.0`, it is not gated and ships no license file (checked on the Hub, revision `d18cca4c1…`). The package says that and carries the
Apache License 2.0 text. Conventions of tt-model-manager (its main branch and open card branches) were read, not changed: the card follows its field set (attribution, description, intended use,
performance, limitations, licensing) without a bring-up history.

## Boot check

Under a lease on board 1 (chips 2 and 3, one lease per session, each released right after): the staged package was started with `bash run.sh` in a fresh venv built from the package's
`requirements.txt`. The first attempt failed at `import ttnn`: the requirements lacked ttnn's own Python dependencies (loguru, networkx, graphviz, pyyaml, click, pandas, seaborn, ml_dtypes, pydantic, from
tt-metal's `pyproject.toml`). They are now in `requirements.txt` with a comment. The second attempt (staged) and the third (pulled from the Hub) both printed the ready line (`load_s` 2.5 to 2.6, the weights
being in the local Hugging Face cache) and passed 9 of 9 smoke checks. The boot used the default route `snapshot_download(model, revision)`, which found the cached snapshot; a first download
of the 3 GB weights through that route was not run.

## Mistakes and delays

- A development lease and two measurement lease windows were lost to queueing (other agents were ahead on board 1, about 20 minutes each time), and one session (`f2`, first attempt) failed at device init
  (`Timeout waiting for physical cores`, 5 runs in 5 minutes) and was repeated. No lease was used unleased in this work and none was reaped.
- The `state_full_ro` kernel bench in session `f1` failed because the default kernel directory does not exist in the source tree (it exists in the staged package); it was run from the staged package.

## Not done

Harness `bringup` run (waits for the generic text route); chunked-scan prefill; a first-time weight download test; making the repository public (operator's decision); the tt-model catalog.

# Release 0.2.0 (update after the Copilot review of PR 244 and the operator's release list)

Status line is at the end of this section. Items below follow the operator's numbering. Evidence for this section is in `evidence/granite-package/` (the scripts are
`boot.sh`, `fresh_boot.sh`, `final_boot.sh`, `s1.sh`, `build_summary2.py`, `make_evidence.sh`; results in `results/`).

## Step 1: the Copilot review of PR 244 (all fixed, each with a red test first)

| Defect | Fix | Test |
| --- | --- | --- |
| `boot.sh` left `RPID` empty when the server never became ready, skipped the stop, and `wait` could hang holding the chip | stop the pid the script launched; the wait is bounded (`READY_TRIES`, `READY_SLEEP`) | `test_boot_script_does_not_hang_or_leave_the_server_when_it_never_becomes_ready` |
| `_manifest_problems`: a manifest root that is not a mapping raised `AttributeError` | named problem "document root must be a mapping" | 4 parametrised cases (empty, list, string, number) |
| ready marker fallback accepted a marker the server never prints | the manifest's marker must occur in `server/serve.py` | `ready_line: NEVER_READY` fails the gate; the good manifest passes |
| stop-string holdback slice could go negative and leak part of the stop string | clamp to 0 | generating `abcdef` with stop `abcdef` returns empty |
| oversized prompt with `stream: true` got no JSON 400 | check before any header; one `_started` flag decides whether a 400 or 500 can still be sent | HTTP test |
| the starred expansion `[first, *gen]` ran the whole generation before writing | a lazy loop with a flush per piece | a slow fake engine: first content arrives at 0.3 s or less, the whole stream takes more than 1.4 s |
| the ready line and `evaluate.py` recorded the requested precision | both record `model.f32` (the constructed model); `GraniteChip.effective_precision` | unit test and a card-claim test |

The streamed first-token latency was re-measured. Before the fix the first content chunk arrived with the end of the generation (the smoke test's `first_content_s` equalled `wall_s`,
0.634 and 0.634 s, on a 2-token answer). After the fix the same request gives `first_content_s` 0.560 and `wall_s` 0.619 s (session 3 smoke test): the first content now arrives before the end.
The test `test_streamed_response_arrives_piece_by_piece_not_after_the_whole_generation` is the check that does not depend on a short answer.

## Items of the release list

| # | Item | State | What was done |
| --- | --- | --- | --- |
| 2 | Fresh-machine check, `--self-test` | done | A new venv from `requirements.txt` with `uv pip install --no-cache`: 3 min 46 s, 63 packages, 5.4 GB (the default torch wheel brings the CUDA libraries). An empty `HF_HOME`: `bash run.sh` downloaded the pinned revision (2.8 GB) and served: ready after 627 s from the start of the first run (download and load; the host was loaded, 36 at the start). Smoke test 9 of 9. Nothing broke. The self-test found one defect in itself: the port check failed right after the server stopped (TIME_WAIT), fixed with SO_REUSEADDR and a test that goes red without it. `bash run.sh --self-test` prints one line per check (python, packages, `import ttnn`, chips, pinned weights, free disk, port, tt-metal tag) and opens no device unless `--open-device`. |
| 3 | Server behaviours | done | Tests for: a second request gets 503 with `Retry-After` at once; a client that disconnects mid-stream frees the slot; a body over 1 MiB, malformed JSON, bad fields, tool calls and document inputs get 400; max_tokens cap and context limit (generate tests, stream and non-stream); the same seed repeats sampling and another seed differs; a multi-byte character is never split across pieces; an idle connection is closed after 30 s. Binds 127.0.0.1 by default; `HOST` or `--host` listens elsewhere with a warning; SECURITY.md and the card say there is no sign-in. 30 server tests. |
| 4 | Agreement on three text types | done, with limits | Table below. Float32 state stays the default. |
| 5 | Second session | done (three sessions) | Table below. The sessions differ in host load (1 to 36). A different day-part was not available: all sessions are the same afternoon. |
| 6 | Card | done | The existing table is kept. New rows labelled `measured` (decode, `chat` prompt, 12 tokens, traced and untraced; time to first token; top-1 agreement as a fraction), a Limits section that leads with the requested items, an API section that lists the server behaviours. Ranker result below (in this log only). |
| 7 | Release hygiene | done | Version 0.2.0 in `package-record.json` and the card, `CHANGELOG.md`, `SHA256SUMS` (every file; the gate recomputes it), `SECURITY.md`, pinned revision and the tt-metal tag in the record, LICENSE and NOTICE kept, "community package, not a release of IBM or Tenstorrent" kept, requirements pinned to the tested ranges. |
| 8 | Prefill | done (small win kept) and an issue | A second trace that stops after the last layer for prompt tokens: 38.6 against 37.4 tok/s for a 512-token prompt in one session (+3%); in session 3, 38.6 tok/s at 128, 512 and 1,024 tokens. The logits equal the step-by-step path exactly (max abs difference 0.0, same argmax, 128, 512 and 1,024 tokens). Batching the attention layers or a chunked scan is not small: issue 252 holds the measurements and what it needs. |
| 9 | Gate and boot check on the final copy | done for the staged directory; the Hub pull check is not done | The final staged directory was booted from its own directory in the fresh venv (below). The copy pulled from the Hub will be checked again after the operator pushes. |

### Agreement, three kinds of text (teacher forced against fp32 on the CPU; float32 state; 2,048-token context)

| Text | Run | Relative L2 mean | max | Top-1 |
| --- | --- | --- | --- | --- |
| prose, 2,048 tokens | this package | 0.0165 | 0.256 | 0.984 |
| prose | bf16 CPU run | 0.0104 | 0.138 | 0.978 |
| code, 2,048 tokens | this package | 0.0127 | 0.616 | 0.978 |
| code | bf16 CPU run | 0.0079 | 0.138 | 0.969 |
| non-English passage, 653 tokens | this package | 0.0185 | 0.121 | 0.969 |
| non-English passage | bf16 CPU run | 0.0112 | 0.062 | 0.963 |

The first 1,024 positions of the 2,048-token runs are in `evidence/agree-*-2048.json` (`first_1024_positions`). The four short prompts: the same as the first release (relative L2 0.0066, 0.0180, 0.0076, 0.0184
against floor 0.0047, 0.0121, 0.0043, 0.0111). Limits: one sequence of each kind, one run; the non-English passage is the only short one and was written for this test (French, Spanish, German, Italian); the code
sequence has a single position with relative L2 0.62, which the bf16 CPU run does not have; this does not describe answer quality.

### Sessions (default configuration; each cell 5 repeats after a warm-up)

| | Session 1 | Session 2 | Session 3 |
| --- | --- | --- | --- |
| Start, host load at the bench start | 14:41, 1.0 | 15:46, 17.9 | 16:16, 4.3 (rose to 31 during it) |
| Code | 0.1.0 prefill | 0.2.0 | 0.2.0 |
| Traced decode, tok/s | 34.3 (34.2 to 34.5) | 25.2 | 33.9 (33.7 to 34.2) |
| Untraced decode, tok/s | 24.0 | 13.8 | not valid (see incidents) |
| First token after a 12-token prompt, s | 0.327 | 0.320 | 0.317 |
| Prefill, tok/s (128, 512, 1,024 tokens) | 37.4, 37.4, 37.4 | 38.4, 38.6, 38.6 | 38.6, 38.6, 38.6 |
| Server decode, 64-token completion, tok/s | 34.4 | not run | 34.2 |
| Server first streamed content, s | 0.63 (not lazy yet) | not run | 0.56 |
| Server, 1,481-token prompt to first token, s | 39.6 | not run | 38.4 |

The traced decode spread is 25.2 to 34.3: a loaded host lowers it, because the per-token input writes and the logits read run on the host.

### Ranker (`python -m card_viewer <card> --no-infer`, the final staged card)

Window grade B across p300c. Tiles: `chat` on p300c B, provisional, held back by speed; `coding-agent` on p300c B, provisional, held back by speed. Dimensions: speed, consistency and concurrency
have no grade (the card gives no output length, no tail latency and no two loads on the same lengths); suitability A- (draft threshold: top-1 agreement 97.9% against the 80% bar). The viewer read 2 decode speeds
from the card's `measured` rows and did not attach the time to first token to a speed row beyond the `chat` prompt. Nothing on the card mentions the grades.

## What wiring the coder kernels in by default would get (answer to the operator)

Same session (2,048-position context, traced, one run each, bf16 state in both): composed 37.1 tok/s, `state_full_ro` (coder state-update and readout kernels) 38.8 tok/s, which is +4.6%.
Earlier, at a 128-position context with the 3-round interleaved design (PR 238): composed 37.5, `state_full_ro` 39.3 (+4.8%), compact kernel plus readout kernel 38.8 (+3.5%). Against the package default (composed with a float32 state, 34.3 tok/s) the kernel mode is 13% faster,
but the two do not compute the same thing: the kernels take and return a bf16 state. They have no float32 variant. The kernel path needs: the bf16 state, the tt-lang pair venv (tt-lang-light 1.1.10.dev20261026 with tt-metal 0e9d200; the shared venv's tt-lang 1.1.6 does not compile the reduction),
the kernel directory, and a bf16 readout. Agreement of the kernel mode over long sequences is in the section below (measured in the final lease). Recommendation: keep composed with the float32 state as the default. A kernel mode that matches the float32 state (a float32 kernel,
or a kernel that keeps the decay in float32) would be worth building; today's kernel gains 5% in speed and loses the accuracy that the float32 state buys.

Agreement of the kernel mode over the prose sequence (teacher forced against fp32 on the CPU, 2,048-token context; final lease, host load 30 to 50, which does not change agreement):

| Configuration | 2,048 tokens: relative L2 mean (max), top-1 | first 1,024 tokens: relative L2 mean (max), top-1 |
| --- | --- | --- |
| default: composed, float32 state | 0.0165 (0.256), 0.984 | 0.0134 (0.160), 0.978 |
| composed, bf16 state | 0.0697 (1.53), 0.955 | 0.0414 (0.616), 0.955 |
| kernel mode `state_full_ro`, bf16 state | 0.0733 (2.07), 0.952 | 0.0419 (0.450), 0.949 |

The four short prompts with the kernel mode: relative L2 0.0075, 0.0160, 0.0067, 0.0172, top-1 1.0, 1.0, 0.875, 0.958 (the numbers of the first release's kernel run). The kernel mode has the error of the bf16 state, three to four times the default's,
and a maximum single-position relative L2 of 2.07 over 2,048 tokens. Files: `results/final/agree-kernel.json`, `results/final/agree-composed-bf16.json`, `results/s1/agree-prose.json`.

## Incidents

- The third session's untraced bench (16:24:09 to 16:25:33) ran after gozer reaped the lease (16:24:10 Pacific, "detached lease past its grace window"; `gozer renew` before it had not extended that window).
  Another agent was granted the chips at that moment. The run is not used, and I report it so the other agent's run at that time can be checked. The traced bench ended at 16:24:09, one second before the reap. Later runs were kept under 14 minutes.
- Earlier in this release (the first measurement session of the day) bench-fast took 8 minutes because the host load average reached 41, and the lease was renewed; it finished inside the grace window.
- A first fresh-venv boot failed (missing ttnn dependencies) in the 0.1.0 work; fixed then.
- The final lease (boot of the final staged package, an untraced decode run and the two agreement runs) ran with a host load of 30 to 54, so its untraced decode number (9.9 tok/s) is not used on the card; `evidence/final-bench-untraced-loaded-host.json` holds it.

## Final staged package (0.2.0) and what remains

Staged at `stage/E/granite-4.0-h-1b-blackhole` in the Granite run directory (outside the repository; its path is in the report). The gate passes with 0 problems (the audio package's scrub reused, plus the
checks listed above and the new ones: `SHA256SUMS` recomputed over every file, version in the record, card and changelog, manifest mapping and ready marker). The code files of E are byte-identical to the directory D that the fresh-venv boot check ran on
(`cmp` over the runtime, server, tools, run.sh, requirements, manifest, kernels, license, notice and security files); E differs from D in the card, the evidence files and the hashes only. E itself was then booted from its own directory in the fresh venv: self-test 0 failed, ready line, 9 of 9 smoke checks, first streamed content 0.56 s, `sha256sum -c SHA256SUMS` with 0 files not OK (`results/bootE`). The first `gozer release` after that run was refused ("device still open": the server process was still exiting); a second release a few seconds later succeeded.
Not done: the Hub pull check of the pushed 0.2.0 copy (after the operator pushes); sessions on another day-part (all three are the same afternoon); a chunked-scan prefill (issue 252); an untraced decode number for session 3.
