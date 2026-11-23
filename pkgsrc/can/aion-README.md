---
tags:
- blackhole
- p300x2
- QB2
- tt-dit-server
- thin
- tt-model-cache
- text-generation
license: mit
pipeline_tag: text-generation
base_model:
- microsoft/Aion-1.0-Instruct-2.3B
---

# aion-1.0-instruct-2.3b-blackhole on Tenstorrent p300x2 (QuietBox 2)

> Derived from [microsoft/Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B) by Microsoft (revision `28902cbfd9f4e0347d6c853aa1301f5d559b2fc0`), weights unmodified and not part of this repository. The ttnn port and the server in this bundle are Tenstorrent's, under the Apache License 2.0.

Aion-1.0-Instruct-2.3B is a 2.3 billion parameter instruction-tuned language model for text tasks such as question answering, summarization and rewriting, in eight languages (English, German, Spanish, Portuguese, Italian, French, Chinese, Japanese). It has 28 decoder layers (21 sliding-window attention layers with a 2,048 token window and 7 full attention layers), grouped-query attention with 24 query heads and 8 key-value heads, a vocabulary of 200,029 tokens and a context of 32,768 tokens. This bundle serves it through an OpenAI-style chat and completions API on four Blackhole chips.

Runs on Tenstorrent **p300x2** (QuietBox 2: two p300 boards, four chips), mesh 1x4.

Packaged with tt-model-manager 0.1.0 (manifest schema 6).

## At a glance

| | |
| --- | --- |
| Architecture | decoder-only transformer, 28 layers, 21 sliding-window and 7 full attention, YaRN rotary scaling on the full-attention layers |
| Parameters | 2.3 billion |
| Context | 32,768 tokens in the model; this server accepts up to 8,192 (prompt plus answer) by default; the longest prompt measured was 7,583 tokens |
| Hardware | p300x2 (four Blackhole chips, mesh 1x4) |
| Precision on the chips | bfloat16, one full copy of the model on each chip |
| Decoding | greedy by default; `temperature`, `top_p` and `seed` are honoured |
| License | MIT (model), Apache-2.0 (port and server) |

## Intended use

Question answering, summarization, rewriting and chat in the model's eight languages, on a QuietBox 2 or another machine with four Blackhole chips, behind one OpenAI-style endpoint for one user at a time. It is out of scope for anything the model's own card rules out: generating harmful or misleading content, and automated decisions in critical contexts such as medical or legal judgments. This package is a bring-up port with a modest decode rate (see Expected performance), so it fits evaluation and development more than serving several users.

## Prerequisites

- The [Tenstorrent CLI](https://github.com/tenstorrent/tt-cli/blob/main/docs/prerequisites.md), `tt`: `uv tool install tenstorrent` (or use tt-model alone, see the Quickstart)
- Tenstorrent hardware: **p300x2**, or any machine where four Blackhole chips are free
- 1 GB hugepages mounted at `/dev/hugepages-1G`
- SFPI 7.85.0 or newer on the host (the version `ttnn` 0.80.0 declares). `install.sh` stops with a message when the host's SFPI is older. The bundle was tested on a host with SFPI 7.96.0, tt-kmd 2.12.0 and Blackhole firmware bundle 19.17.1; hosts with SFPI between 7.85.0 and 7.96.0 were not tested
- About 8 GB of free disk: the checkpoint (4.6 GB), the Python environment `install.sh` builds (1.9 GB) and the compiled-kernel cache (1.4 GB after the runs below)

## Quickstart

```bash
tt serve <owner>/aion-1.0-instruct-2.3b-blackhole
```

`tt serve` installs the bundle into its own Python 3.12 environment from pinned requirements (`ttnn==0.80.0` from PyPI) and downloads the [`microsoft/Aion-1.0-Instruct-2.3B`](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B) weights at `28902cbfd9f4e0347d6c853aa1301f5d559b2fc0` into your Hugging Face cache (about 4.6 GB; they are not in this repository). The first start with an empty kernel cache took 18 s to become ready, and later starts took 6 s. The server is ready when `/health` answers and its log shows `"event": "ready"`.

Without tt-cli: `tt-model serve <owner>/aion-1.0-instruct-2.3b-blackhole`. Without tt-model: run `bash install.sh`, then `bash run.sh` (it serves on 127.0.0.1, port 20180; `PORT` and `HOST` change that). The server needs four free chips: on a shared machine, lease both boards first.

## Capabilities

`GET /health`, `GET /v1/models`, `POST /v1/chat/completions` and `POST /v1/completions`.

- The chat route builds the model's own turn format (system, user and assistant turns, each closed by `<|end|>`, with the model's identity line, knowledge cutoff and today's date first), the same text the model's chat template produces. Message roles are `system`, `user` and `assistant`. Message text that contains one of the model's control tokens (`<|user|>`, `<|end|>` and similar) is refused with a `400`, as the model's template refuses it. `/v1/completions` sends the prompt as written.
- Fields: `prompt` or `messages`, `max_tokens` (default 128) or `max_completion_tokens`, `stream`, `stop` (up to four strings of at most 256 characters each), `temperature`, `top_p`, `seed`, and `return_token_ids` (adds the generated token ids). A response carries `usage` and a `timings` object (prefill seconds, decode seconds, decode tokens per second).
- One request runs at a time. A second one gets `503` with `Retry-After: 5`. A body over 1 MiB, malformed JSON, an unsupported field (`n` other than 1, `tools`, `logprobs`, `response_format` and the like) or a prompt longer than the context gets a `400`.

```bash
curl -s http://127.0.0.1:20180/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":16}'
```

| variable | default | meaning |
| --- | --- | --- |
| `AION_MODEL_DIR` | the pinned revision from the Hugging Face cache | a directory with the model files |
| `AION_MAX_LEN` | 8192 | context limit in tokens, prompt plus answer (the model supports 32,768) |
| `AION_CHIPS` | 4 | chips in the mesh |

## Expected performance

Measured through this bundle's own `run.sh` on four Blackhole chips (two p300 boards) with `ttnn` 0.80.0, one request at a time, greedy decoding, wall time through the server, warm kernel cache. Each figure is the median of the runs stated in the N column. The host load average (1, 5 and 15 minutes) was 9.5, 7.0, 3.0 before and 3.6, 4.6, 3.5 after the short-prompt decode runs. The port runs part of its work on the host CPU, so the decode rate falls when other jobs use the host. The load average also rises because of the server itself: during the long-prompt runs below it was 11.8, 3.3, 1.1 before and 14.3, 11.4, 5.4 after, because the host work uses many cores, so a load average above 1 does not by itself mean another job was running. The short-prompt decode, chat-route, accuracy-served and boot rows were measured on SFPI 7.96.0 and tt-kmd 2.12.0. The long-prompt rows, the cold-cache run and the logits records were measured earlier on SFPI 7.85.0 and tt-kmd 2.11.0 (same bundle, same chips); they were not repeated after the upgrade.

| measure | context (prompt tokens) | new tokens | N | value |
| --- | ---: | ---: | ---: | --- |
| decode, batch 1 | 5 to 14 (5 prompts) | 64 | 15 | median 10.91 tok/s (min 10.10, max 12.16) |
| decode, batch 1 | 582 | 9 | 2 | median 0.940 tok/s (min 0.907, max 0.972) |
| decode, batch 1 | 2282 | 9 | 2 | median 0.297 tok/s (min 0.291, max 0.304) |
| decode, batch 1 | 7562 | 9 | 2 | median 0.093 tok/s (min 0.092, max 0.095) |
| prefill (time to the end of the prompt) | 141 | 1 | 3 | median 0.31 s (min 0.31, max 0.38), 455 tok/s |
| prefill (time to the end of the prompt) | 561 | 1 | 3 | median 1.01 s (min 1.01, max 1.17), 557 tok/s |
| prefill (time to the end of the prompt) | 2261 | 1 | 3 | median 3.25 s (min 3.17, max 3.43), 696 tok/s |
| prefill (time to the end of the prompt) | 7541 | 1 | 3 | median 10.70 s (min 10.04, max 10.79), 705 tok/s |

Through the chat route every prompt starts with a system header of about 40 tokens, so the context is longer at the same answer length. Five prompts that ask for a paragraph, 3 runs each, 64 new tokens, prompts of 50 to 52 tokens, warm kernel cache, host load 1.3, 0.9, 0.4 before and 9.5, 7.0, 3.0 after: decode median 6.71 tok/s (min 2.11, max 7.55, 15 runs). The boot check's single 64-token chat answer, on an empty kernel cache, ran at 3.81 tok/s. 

The port has no key-value cache: each new token runs the whole sequence again, so the time per token grows with the length of the prompt and the answer together, as the rows with a longer context show. The in-process decode benchmark of the bring-up harness, which ran on one prompt with 32 steps and no server, gave a median of 8.3 tok/s (min 8.3, max 8.4, 3 runs). A second measurement run, started with an empty kernel cache, gave a served decode median of 11.55 tok/s (min 4.99, max 12.52, 15 runs; host load 1.9, 1.4, 1.5 before and 2.5, 2.0, 1.7 after); its first request took 7.38 s. The two runs ran at different times, so the difference between their medians says nothing about the kernel cache.

Accuracy against the model's own implementation, run on the CPU in bf16 (transformers, custom code of the pinned revision), on exactly the same token ids. For each prompt the reference generated 128 tokens greedily, and both implementations then read the same text one position at a time:

| eval | metric | this package | reference | size |
| --- | --- | ---: | --- | --- |
| 6 prompts, teacher-forced on the reference's own continuation | top-1 agreement with the reference's token | 0.9857 | transformers, CPU, bf16 | 768 positions |
| same | top-1 equals the reference's own top-1 | 0.9870 | same | 768 positions |
| same | positions where the port's top-1 differs | 11 of 768 | same | 10 of them with a reference gap of 0.25 logits or less |

| prompt | positions | top-1 agreement |
| --- | ---: | ---: |
| `The capital of France is` | 128 | 1.0000 |
| `Water boils at a temperature of` | 128 | 0.9609 |
| `def fibonacci(n): """Return the n-th Fibonacci number."""` | 128 | 1.0000 |
| `In 1969, the first humans to walk on the Moon were` | 128 | 0.9766 |
| `<|system|>You are Aion Instruct, a large language model trai` | 128 | 0.9844 |
| `<|system|>You are Aion Instruct, a large language model trai` | 128 | 0.9922 |

These rows show agreement with the reference's own continuation on six prompts (four plain, two in the chat format). They do not measure accuracy on held-out text, and they are not benchmark scores: the model's published benchmark results are on its own model card.

Free-running generation (greedy, through the server) against the reference's 128 tokens:

| prompt | tokens compared | free run |
| --- | ---: | --- |
| `The capital of France is` | 128 | all match |
| `Water boils at a temperature of` | 128 | first difference at token 18 (20 of 128 positions equal) |
| `def fibonacci(n): """Return the n-th` | 128 | all match |
| `In 1969, the first humans to walk on the` | 128 | first difference at token 51 (51 of 128 positions equal) |
| `<|system|>You are Aion Instruct, a large` | 128 | first difference at token 6 (6 of 128 positions equal) |
| `<|system|>You are Aion Instruct, a large` | 128 | first difference at token 27 (127 of 128 positions equal) |

First difference of `Water boils at a temperature of` (token 18): the reference's logits for '50' and '1' are 16.75 and 16.625 (a gap of 0.125), and the port's top two are the same two tokens, '1' at 16.625 and '50' at 16.625 (a gap of 0.0; equal logits, so which of them greedy decoding takes depends on how the tie is broken). A gap that small is within bf16 rounding (neighbouring bf16 values near 20 are 0.125 apart), so this difference is a near tie.

First difference of `In 1969, the first humans to walk on the` (token 51): the reference's logits for ' another' and ' the' are 21.375 and 21.25 (a gap of 0.125), and the port's top two are the same two tokens, ' the' at 21.375 and ' another' at 21.25 (a gap of 0.125). A gap that small is within bf16 rounding (neighbouring bf16 values near 20 are 0.125 apart), so this difference is a near tie.

First difference of `<|system|>You are Aion Instruct, a large` (token 6): the reference's logits for ' and' and ' in' are 21.875 and 21.75 (a gap of 0.125), and the port's top two are the same two tokens, ' and' at 21.375 and ' in' at 21.375 (a gap of 0.0; equal logits, so which of them greedy decoding takes depends on how the tie is broken). A gap that small is within bf16 rounding (neighbouring bf16 values near 20 are 0.125 apart), so this difference is a near tie.

First difference of `<|system|>You are Aion Instruct, a large` (token 27): the reference's logits for ' Return' and ' Returns' are 24.25 and 24.125 (a gap of 0.125), and the port's top two are the same two tokens, ' Returns' at 24.25 and ' Return' at 24.125 (a gap of 0.125). A gap that small is within bf16 rounding (neighbouring bf16 values near 20 are 0.125 apart), so this difference is a near tie.

Served checks (greedy, chat route): a passkey hidden in filler text and five short questions, each checked for the expected fact as a substring.

| passkey in filler text, chat route | prompt tokens | found |
| --- | ---: | --- |
| about 256 tokens | 343 | yes |
| about 2048 tokens | 2303 | yes |
| about 7800 tokens | 7583 | yes |

- `What is the capital of France? Answer in one word.` expected `paris`: found
- `What is 17 plus 25? Reply with just the number.` expected `42`: found
- `Which planet is known as the Red Planet? Answer in one word.` expected `mars`: found
- `What is the chemical symbol for water? Give the formula.` expected `h2o`: not found
- `Name the author of the play Hamlet. Answer with the name only.` expected `shakespeare`: found

Questions answered with the expected fact by the plain substring check: 4 of 5. The miss is the formula for water, which the model wrote with a subscript two (H₂O) where the check looked for H2O; after Unicode compatibility normalisation of the answer the count is 5 of 5. The substring check cannot tell a good answer from a lucky one.

## Limitations

- Four chips are needed. Each chip holds a full copy of the model and runs the same work; this package was measured on four chips only, and it makes no claim that the extra chips make it faster. One request runs at a time; a request that arrives while one is running gets `503`.
- Decode speed is modest. The measured rate is far below the memory-bandwidth limit of the boards (the harness's in-process median is 1.9 percent of an estimated floor of 447 tok/s for decode with one token in flight, assuming bf16 weights), and it falls as the context grows (see the rows above), because the port has no key-value cache and runs the whole sequence again for every token.
- The decode rate depends on host load, because part of the work runs on the host CPU.
- Prompt plus answer is limited to 8,192 tokens by default (`AION_MAX_LEN`). The longest prompt measured was 7,583 tokens. The model supports 32,768 tokens; lengths above 7,583 tokens were not checked on the chips.
- Only the sliding-window mask, rotary scaling and attention paths exercised by the checks above were verified. Quality on other prompts, other languages, sampled decoding, long generations and tool use was not evaluated. Tool calls and the model's tool format are not supported.
- Greedy decoding was used for every number here.
- Message text with control tokens is refused (see Capabilities). The server has no sign-in, no TLS and no rate limit. It binds 127.0.0.1 by default; `HOST=0.0.0.0` listens on every interface, which is only for a network you trust.
- The model is small. Its own card says it can miss fine details, give misstatements and fail at complex multi-step reasoning, and that its safety alignment may not hold against adversarial input. Read the upstream card before using it.
- Tested on one machine with two Blackhole p300 boards, ttnn 0.80.0 (the PyPI wheel), Python 3.12. Other board types and other chip counts were not tested.

## Licensing

The model is released by Microsoft under the MIT License ([`LICENSE`](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B/blob/main/LICENSE) in the upstream repository). The port, the server and the tools in this bundle are Copyright 2026 Tenstorrent USA, Inc., under the Apache License 2.0 (`LICENSE`, `NOTICE` in this bundle).

## Related packages

- Upstream model: [microsoft/Aion-1.0-Instruct-2.3B](https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B).
- Other ttnn ports packaged the same way for four chips: `episod/olmo-hybrid-7b-blackhole`.

## Feedback

Questions and reports: the Discussions tab of this repository, or support@tenstorrent.com.

## Provenance

| component | built from |
| --- | --- |
| weights | `microsoft/Aion-1.0-Instruct-2.3B` at `28902cbfd9f4e0347d6c853aa1301f5d559b2fc0`, downloaded at first start, not in this bundle |
| port | `extra_code/aion_port.py`, generated by the tt-orchard bring-up harness and checked against the CPU reference. One change was made for this bundle: the tensor reader copies each tensor out of the memory-mapped checkpoint (`.clone()`), because with the PyPI `ttnn` 0.80.0 wheel, `ttnn.from_torch` of a memory-mapped tensor (the embedding table) did not return in 120 s, while a copy uploads in 0.2 s |
| server | `extra_code/aion_asgi.py` (ASGI app served by uvicorn) |
| runner | `aion_model.py` at the bundle root, which re-exports the app for tt-model's packager |
| bring-up and measurement records | `docs/run-logs/2026-11-22-aion-can.md` and `docs/run-logs/evidence/aion/` in the tt-orchard repository |
| integrity | `SHA256SUMS` lists the hash of every file |
