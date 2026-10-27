# Greedy-only fast paths: what the Qwen3.8 DFlash2 bundle does with a sampled request (issue 107)

Date: 2026-10-27. Bundle: `episod/qwen3.8-27b-dflash2-p300`, installed with `tt-model`, served on one 2-chip board (mesh 1x2) with its own base weights
`Qwen/Qwen3.8-27B`. File paths below are relative to the bundle's Python environment (`<BUNDLE>/venv/lib/python3.12/site-packages/`). The
tt-metal tree on this machine is v0.80.0 and was read only.

## 1. What the code does (read, not run)

Which sampling controls reach which code, for the bundle as shipped (`--additional-config '{"tt": {..., "sample_on_device_mode": "decode_only"}}'`,
`QWEN36_DRAFTER=dflash2`).

| Question | Answer | Where |
|---|---|---|
| Is there a device sampler on this mesh? | No. The ttnn device sampler is built only for meshes 1x4 and 1x8 whose vocabulary per device fits 64K, and for 1x1 behind `QWEN36_ONDEV_SAMPLING_TP1=1`. A 1x2 mesh has `model.sampling = None`. | `models/demos/blackhole/qwen36/tt/model.py` 111-145 |
| What would the device sampler support? | Temperature, top-p, top-k (at most 32 candidates), presence, frequency and repetition penalties, and a per-request seed. | `models/common/sampling/generator.py` 72-78, `tt_sampling.py` 203 |
| What does the drafter-off path do with `sample_on_device_mode` on 1x2? | It raises `RuntimeError` ("Unset sample_on_device_mode for host sampling"). This is why stage 2 removes the key when it switches the drafter off. | `qwen36_vllm.py` 161-175 |
| What does the drafter-on class do with it? | It accepts the setting without a device sampler, because every decode step's tokens come from "the model-owned greedy speculative outputs"; the prefill anchor is host-sampled. | `qwen36_vllm_dflash.py` 274-289, 6-9, 26 |
| Does the drafter verify sampled tokens? | No. `DFlash2Decoder` asserts `sampling is None` ("greedy only"): accept is the longest prefix where the draft equals the argmax of the verify logits. `decode_forward` reads only tokens, start positions, page table and slot remap, never the request's sampling parameters. | `dflash2_decode.py` 143-156, 10-14; `qwen36_vllm_dflash.py` `decode_forward` |
| What does the plugin do with the request's controls? | For a block-output model it keeps temperature, top-p, top-k and seed (the prefill anchor is sampled by vLLM's host sampler with them) and resets min_p, presence, frequency and repetition penalties to their defaults. It refuses `logprobs`, `n` other than 1, `bad_words` and structured outputs. A decode step that falls back to host sampling raises `ValueError`. | `vllm_tt_plugin/platform.py` 790-836, 2228-2250; `model_runner.py` 2296-2330, 2530-2541 |

So a request with temperature above 0 is accepted. Its first token is sampled. Every later token is the greedy continuation of that first token.

What exists that could verify sampled tokens:

- The MTP drafter's `SpeculativeDecoder` has exact speculative rejection sampling on the host (`spec_decode.py` 18-22, 155-157; `spec_sampling.py` 1-12).
  It treats the draft as a point mass, so it needs only the target's probability of the drafted token and no draft distribution. It is lossless in distribution and
  supports temperature, top-k, top-p and presence penalty. Its header says it was written for P150x4. The DFlash2 decoder's comment says the block drafter "yields tokens, not a draft distribution". For a deterministic draft that is the same case the MTP code already handles, so the greedy-only assertion looks like a missing wiring and not a limit of the method. This was read, not tested.
- The tt-metal tree (v0.80.0) has `vllm_test_utils/spec_test/README.md`, which describes a newer plugin contract (vllm-tt-plugin issue 110): a model returns full verify logits (`TT_SPEC_ACCEPT_MODES=argmax_ids,logits`) and the plugin rejection-samples sampled and penalized requests on the host, with the model's own drafter proposals. The plugin in this bundle (0.1.0) has no `model_owned_drafter` or `VerifyOutput`, so the bundle cannot use it.

Not determined: whether the DFlash2 verify trace can return full logits for 8 rows (the greedy path argmaxes on device and avoids a host read of `[B*T, vocab]`), and what that read would cost; whether the host tt-metal tree has the DFlash2 files (it has `spec_decode.py`, `spec_sampling.py` and `mtp.py`, and no `dflash2*.py`).

## 2. What was measured

One 2-chip board, one lease. Each configuration was served by the stage 2 templates (`prepare_swap.py` with the base as both nearest and new model, then
the run script), then `serve_and_compare.py`'s sampling probe ran three times: one greedy request and two sampled requests (temperature 1.0, top-p 0.9, seeds 1234 and 4321),
96 new tokens, one short plain prompt. Then ten single requests with other controls, and five seeds at temperature 1.0. Raw records:
`docs/run-logs/evidence/sampling-107/drafter-on-measurements.json` and `drafter-off-measurements.json`. One run per configuration. The drafter-on boot was cold for tensor cache (260 s to ready) and the drafter-off boot used the cache the first one made (124 s).

| | Drafter on, `decode_only` (as shipped) | Drafter off, host sampling |
|---|---|---|
| Greedy decode, tokens/s (three probes) | 60.0, 60.4, 60.5 | 19.1, 17.8, 18.2 |
| Sampled decode, tokens/s (mean of two requests, three probes) | 59.2, 60.2, 59.5 | 17.8, 16.5, 17.2 |
| Sampled speed over greedy speed | 0.98 to 1.00 | 0.93 to 0.94 |
| Drafter accepted share (drafted tokens accepted) | 0.586 (greedy and sampled requests alike) | no drafter |
| Sampled text equals the greedy text | yes, 3 of 3 probes, both seeds | no, 0 of 3 |
| Five seeds at temperature 1.0: distinct texts | 1 (all equal the greedy text) | 5 |
| `temperature 0.7`, `top_k 40`, `top_p` alone, no seed, `presence_penalty 0.5`, `frequency_penalty 0.5`, `repetition_penalty 1.1`, `min_p 0.05` | accepted, each returned the greedy text, 59 to 61 tokens/s | accepted, each differed from the greedy text, 17 to 19 tokens/s |
| `temperature 2.0`, seed 7 | accepted; the first token changed to a different one and the rest is the model's text from that start, 39.3 tokens/s | accepted, different text, 17.6 tokens/s |
| `logprobs 1` | refused, HTTP 400: "This block-output model owns its Gumbel sampling and does not support these request parameters" | accepted |

Findings:

1. With the drafter on, sampling is ignored after the first token. All of a request's later tokens are the greedy continuation. This matches the code reading. Seeds, top-k, top-p, min_p and penalties changed nothing except a high temperature, which changed the first token (the `temperature 2.0` request started with a different token). The server returns no warning.
2. Drafter on is about 3.3 times faster than drafter off for greedy requests (60.3 and 18.4 tokens/s, means over three probes) and the same speed for sampled requests, because they are served as greedy ones.
3. Drafter off samples correctly and costs 6 to 7 percent against its own greedy speed. This is the configuration in `episod/clef-p300`, `episod/arex-2-p300` and `episod/qwen3.8-27b-cjk-trim-p300`: the 18 tokens/s figure is the cost of correct non-greedy sampling there.
4. There is no configuration in this bundle that serves sampled requests both correctly and fast on 2 chips.

## 3. What a fast non-greedy path would need

None of these was built or run.

1. Wire the exact rejection sampler into `DFlash2Decoder` for requests with temperature above 0. The accept step needs the verify logits of the drafted rows on the host (a read of 8 rows by the vocabulary per iteration, which the MTP path already does), the request's sampling parameters passed from `decode_forward`, and per-request seeds and penalty history. The drafts stay the drafter's argmax, so acceptance for a sampled request is lower than the 0.586 measured for greedy ones, by an amount nobody has measured. It is a change to bundle model code (in the `tt_metal_models` wheel), outside this repository.
2. Wait for the plugin contract of issue 110 and the model to declare `logits` acceptance. The bundle would need a plugin release that has it.
3. A device sampler for a 1x2 mesh: the ttnn top-k op is limited to 64K per device (the vocabulary is 248,320, so 124,160 per device on two chips). A sampling kernel that works over a larger shard (a two-stage top-k, then temperature, top-p and a draw on the shortlist) is a candidate for `tt-orchard kernel-probe` with the worked-kernel library. It would replace the host sampling of the drafter-off path, which costs only 6 to 7 percent against greedy there, so it would not change that path's 18 tokens/s; the drafter is what makes the fast path fast. A sampling kernel therefore does not help unless the drafter can verify sampled tokens (item 1).
4. For requests that need non-greedy sampling today, serve the drafter-off configuration (a variant chosen by measurement, as the graft trial is) or run two servers and route by request. Neither exists in tt-orchard.

## 4. Limits of the probe and of this note

- The probe decides "sampling ignored" when two seeds give the same text, or when a sampled text equals the greedy text. A server that samples only the first token, as this one does at a high temperature, passes the first rule when the two seeds pick different first tokens, and the second rule when they pick the same one. At temperature 2.0 only the first token changed here. The probe at temperature 1.0 and top-p 0.9 caught all three probes of the drafter-on server because the first token did not change.
- One prompt, 96 tokens, one board, one run per configuration, base model only. Speeds include the prefill of a short prompt. Not measured: 4 chips, requests in parallel, long prompts, the drafter-on configuration with the grafted head of a fine-tune.
- The `tt-orchard` probe was run through the stage 2 template code on a scratch stage directory outside any run, with its own lease; no supervisor stage ran it.
