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
