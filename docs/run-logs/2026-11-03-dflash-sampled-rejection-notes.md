# How a published bundle serves sampled requests through a speculative drafter (notes for issues 107 and 63)

Read-only study, 2026-11-03. No board was used and nothing was run. Source: the public bundle `mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv-v6-thin-cp4` (card, `PROVENANCE.json`, and the files `extra_code/qwen38/dflash_sampling.py`, `spec_sampling.py` and `decode_sampling/ops.py`). Numbers below are the card's, measured by its author on a shared host (stated variance about 5 percent). We have not reproduced them.

## What changed between the bundle's builds

| Build | Sampled request, one user | Source |
|---|---|---|
| `...-q4kv` (unified v1.1) | plain lane, about 20 to 30 tok/s; its card says DFlash is greedy-only | card of that bundle |
| `...-v6-thin-cp3` | DFlash with exact rejection sampling, about 71 to 73 tok/s (greedy 74 to 77) | card |
| `...-v6-thin-cp4` | about 80 tok/s, 90 to 107 on code-heavy prompts (greedy about 82); plain lane about 47 | card |

Eight users sampled: about 250 tok/s total on both builds.

## How it works (from the code)

1. **The drafter proposes by argmax.** Each draft token is a deterministic function of the committed context. The proposal distribution is therefore a point mass on the drafted token. Rejection sampling then needs no draft distribution: draft j is accepted when a uniform draw u_j is below the target probability of that token, p_j(d_j). On the first rejection at row m, the next token is drawn from the target distribution at row m with the drafted token removed and the rest renormalised. If every draft is accepted, a bonus token is drawn from the last verify row. This is the same rule as vLLM V1's rejection sampler with no draft probabilities, and the committed sequence has the target's sampling distribution exactly (the code comment states this, and the card lists a sampling-distribution check as passing).
2. **All of it runs on the host.** `spec_sampling.py` (`SpecSampler`) and `dflash_sampling.py` import torch and loguru only. The temperature, top-k, top-p and presence-penalty arithmetic is applied per verify row, lazily, so rows after the rejection point are never built. One uniform draw per draft is made up front, so the random stream does not depend on where the rejection lands. The sampler is seeded once per request.
3. **The device keeps only the top 32 candidates per verify row** (`VERIFY_TOPK = 32`, `RowsTopK` in `decode_sampling/ops.py`: scan cores each keep the best K of their tiles, one merge core merges them). A request is "top-k capable" when its top_k is between 1 and 32. For those, the host sees the exact sampling support without reading the full vocabulary row. A presence penalty can push the k-th survivor below the 32nd candidate; `topk_exact` checks that case. A request with no top_k (or one above 32) takes the full-row path.
4. **The commit contract is the greedy one.** `sampled_commit` turns (accepted count, drawn token) into the same `com`, `mi` and next pending token the greedy path builds, so the deferred state commit, the drafter extension and the KV positions follow the sampled result without a second code path.
5. **Which parameters reach it.** `spec_params_for_row` maps the plugin's per-row parameters: temperature at or below 0, top_k of 1 and top_p at or below 0 take the greedy path; a negative presence penalty is clamped to 0 with a warning; seed below 0 or missing draws a fresh seed. The card says frequency and repetition penalties, `min_p`, `min_tokens`, log-probs, `n` above 1, structured output, `logit_bias` and `bad_words` run on the plain lane.
6. **An opt-in gate.** `TT_DFLASH_SAMPLED=1` is read by the model adapter and by the vLLM plugin. With it off, the code paths are the older ones. A second switch, `QWEN36_DFLASH_DRAFT_SAMPLE`, lets the drafter draw its drafts from its own distribution; the accept rule then uses the draft probabilities (accept when u_j times q_j(d_j) is below p_j(d_j); on rejection draw from the positive part of p_j minus q_j). The cp4 card says this raised accepted tokens per step by about 5 percent.

## Other points from the card that bear on tt-orchard

- cp4 changed the default for requests that send no sampling settings from temperature 1.0 to 0.6 with top_k 20 and top_p 0.95. The card says 1.0 occasionally picks an unlikely token, which shows up as typos in identifiers in code.
- The bundle ships its own `vllm_tt_plugin` wheel next to the model code. Issue 107's decision 1 (2026-10-27) allows no change to the plugin and accepts a monkeypatch applied safely from the package. Whether this wheel differs from the upstream plugin was not checked (no diff was made). That check decides whether the bundle's method fits decision 1.
- The drafter weights are bfp8 and the KV cache is Q4 (`q4kv`). The target weights are the unquantized `Qwen/Qwen3.8-27B` checkpoint, converted at load. Nothing in this bundle reads a quantized checkpoint.

## What this means for issue 107

- The "needs a rejection sampler wired into the drafter decoder" blocker in the 2026-10-27 run log has a published working design, and its rule is simple because the drafter proposes by argmax. It fits any drafter that proposes by argmax. A drafter that samples needs the draft probabilities as well (the second switch above).
- What a port to our own models needs: verify rows that expose the top 32 candidates (a device kernel, `RowsTopK`), the host sampler (portable, about 330 lines), a commit mapping for the model's own state, and the plugin side (parameter routing and the gate).
- Of the models in issue 107's list, Clef and CJK-trim ship with the drafter off, so they have nothing to verify against. AREX-2 with the grafted `mtp.*` tensors has an MTP head. The header of `spec_sampling.py` describes exactly that case (an MTP drafter's argmax drafts), but whether the AREX-2 build proposes by argmax was not checked.
- The probe verdicts of PR 109 (`sampled-ignored`, `sampled-full-speed`) give a way to tell a build with this feature from one without it. cp3 and cp4 should read `sampled-full-speed`. We have not run the probe against them.

## What this means for issue 63

The bundle holds no evidence for a quantized checkpoint. Its target is bf16 and the block formats (bfp4, bfp8, Q4) are conversions made at load. The test issue 63 asks for has candidates on the Hub that are quantized copies of models we already have a can or bundle for: `Qwen/Qwen3.5-35B-A3B-FP8` (the can is `episod/qwen3.5-35b-a3b-blackhole-p300`) and `Qwen/Qwen3.8-27B-FP8`, `nvidia/Qwen3.8-27B-NVFP4` and `unsloth/Qwen3.8-27B-NVFP4` (the base of the v6 thin bundles). Running stage 0 on one of them needs no board. Stage 2 on one does.
