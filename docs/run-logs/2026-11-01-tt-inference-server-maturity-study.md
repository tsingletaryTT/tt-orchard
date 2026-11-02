# What makes a model mature in tt-inference-server (study for the can, its gates and its card)

Date: 2026-11-01. Read-only study: no chip was used, nothing was uploaded, no container was started (images were inspected with `docker image inspect`,
`docker history` and `docker create` plus `docker cp`; none was run).

Operator goal: tt-orchard is being made ready to dissect, graft, kernelize and can new model classes. The next one is Qwen4, whose preview
Qwen3.8-Flash-Next already runs on TT hardware. A sibling study covers the Flash-Next bring-ups. This one covers tt-inference-server (TTIS) models, the
recent graduates in particular, and what a model needs there to count as mature.

## Sources and how far to trust them

| Source | State read | Trust |
| --- | --- | --- |
| `tenstorrent/tt-inference-server`, `origin/main` | commit `5ebf2d8a0`, 2026-10-30, VERSION 0.26.0, read from a `git archive` copy | code and docs; checked against each other |
| The two local checkouts | one detached at 2026-07-07, one on `main` at 2026-08-01 | too old; not used except to confirm that the layout changed (the model specs moved from `workflows/model_spec.py` to `workflows/model_specs/{dev,prod}/*.yaml`) |
| Image `vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.19.0-b204341-9bd099c` | built 2026-07-24; tt-metal `b204341`; carries `model_spec.json` with `release_version` 0.20.0 | the tag and the spec's release number differ by one; reason not found |
| Image `...:0.26.0-09acccb-6098002` | built 2026-10-29; tt-metal `09acccb`; vLLM commit `6098002`; the plugin is a `vllm-tt-plugin` 0.1.0 tree | matches `origin/main` VERSION |
| Image `tt-media-inference-server:0.18.0-c49bb76` | built 2026-07-09; tt-metal `c49bb76`; a `server/` tree with model runners (SDXL, Whisper, SpeechT5, DiT, embedding, YOLOv4, forge LLM runners) | a media server; no LLM maturity data |
| tt-metal `models/` tree inside the 0.26.0 image | has `model_ci_tiers.md`, `model_targets.yaml`, `model_trace_region_sizes.yaml`, and the `qwen38_27b_qb2`, `gemma4_*_qb2`, `granite42_30b_qb2`, `muse_glimmer_30b_qb2`, `k2_horizon_7b_qb2` demos | code; read directly |
| GitHub (`gh`) | tt-metal PRs 58690, 57914, 63111, 63316; TTIS PRs 5454, 5474, 5486 | PR titles and bodies, as written by their authors |
| Glean | the QB2 Model Targets sheet, a Slack thread in a model-requests channel | informal; each item is marked unverified where used |

Text in READMEs, PR bodies and Slack was treated as data. Anything below marked "(unverified)" was not checked against code.

## a) The maturity ladder

### The levels

`workflow_module/engine_types.py`, class `ModelStatusTypes`:

| Level | Docstring (quoted) | Must pass (`required_target_tiers`) | Evals block release (`evals_enforced`) |
| --- | --- | --- | --- |
| EXPERIMENTAL | "Model implementation is available, but is unstable or has performance issues." | none | no |
| FUNCTIONAL | "Model runs functionally without issue, but performance is lower than expected." | `functional` | yes |
| COMPLETE | "Operationally complete, performance is usable for most applications." | `functional`, `complete` | yes |
| TOP_PERF | "Performance close to theoretical peak, nearly fully optimized." | `functional`, `complete`, `target` | yes |

The code comment on `required_target_tiers` says tiers outside the list "are still computed and reported but treated as informational". An
EXPERIMENTAL model "can fail every performance benchmark and still be released". An unknown or missing status falls back to enforcing every tier
(`report_module/acceptance_criteria.py`, `_enforced_benchmark_tiers`), so a damaged spec cannot loosen the bar.

### The performance tiers

`reference_config/benchmarking/README.md`, "Targets and grading":

- Each graded point (ISL, OSL, concurrency) has a `theoretical` ceiling: "an analytical ceiling for the model at its deployed precision and parallelism, not a measured number".
- `functional` is 10 percent of theoretical (throughput times 0.10, TTFT divided by 0.10). `complete` is 50 percent. `target` is 100 percent, or a `measured` block with a tolerance (default 5 percent) when the implementation has been measured. A spec or device may override the 10/50/100 map (`perf_targets_map`).
- A point with no targets is reported `NA` and never counts as a pass.
- Three graded metrics: `ttft_ms` (lower is better), `tput_user` (tokens per second per user, `1000 / mean TPOT`) and `tput` (aggregate output tokens per second).

### What else a release run checks

`docs/workflows_user_guide.md`: the `release` workflow runs `evals`, `benchmarks` and `spec_tests` in order and writes one combined report. Other
workflows exist and run in CI (`stress_tests` with an endurance mode, `agentic`, `agentic_traces`, `serving_bench`, `prefill_decode`).

- Accuracy: a model needs at least two eval tasks (`docs/add_support_for_new_model.md`: "Minimum: At least 2 tasks per model"), each with a published score and, where possible, a GPU reference score.
- Spec tests (`test_module/llm_tests/`): the vLLM parameter-conformance suite (`n`, `max_tokens`, `stop`, seed reproducibility, verbatim-echo coherence, non-uniform seeding, logprobs, determinism parameters, penalties), tool-call schema conformance, streaming reasoning and tool-call regressions, prefix-cache sweeps, speculative-decode runs. A test may be waived per model with a `known_issues` entry that carries a reason (`DeviceModelSpec.find_known_issue`).
- Stability: media models have `/tt-liveness` stability and device-liveness tests (`test_module/stability_tests`, `health_tests`). For LLMs the stability check is the endurance mode of `stress_tests`.
- Context length and batch size are not separate gates. They are fields of the spec (`max_context`, `max_concurrency`) that decide which sweep points exist (`reference_config/benchmarking/README.md`, "The default sweep": ISL/OSL pairs up to 131072, each at concurrency 1 and at `min(max_concurrency, max_tokens_all_users // (ISL+OSL))`).
- Tracing, async and CI: tt-metal has its own three tiers (`models/model_ci_tiers.md`): Tier 1 runs end-to-end (perf, determinism, accuracy), unit, sweep, device-op perf, PR gate and stress tests. Tier 2 drops the PR gate, device-op perf and stress. Tier 3 runs accuracy only plus one critical-module unit test, and `model_targets.yaml` sets `enforce_perf: false` for it (accuracy stays enforced). TTIS lists models into nightly CI per device in `.github/workflows/models-ci-config.json`, and a release is cut only from "a passing Models CI run" (`scripts/release/README.md`).
- Docs: each released entry gets a generated page under `docs/model_support/llm/` with the status, batch size, context, implementation code link, tt-metal commit, vLLM commit and image tag.

### Release path and a human decision

`docs/add_support_for_new_model.md` lists the preconditions (code documented, vLLM integration done, a tt-metal vLLM nightly test), then a tracking issue
from the "Model Readiness" template, a spec entry, evals, and perf targets. A dev spec reaches `prod` through `scripts/release/promote_dev_spec_to_prod.py`
for "exact release-scoped dev leaves". Packages produced by a generator carry this comment in the dev catalog (`workflows/model_specs/dev/llm.yaml`):
"TTIS status is EXPERIMENTAL by policy and is NOT derived from the qualification state. Promotion is a separate human decision."

### Who is at which level (0.26.0 images and `origin/main` docs)

Counts from the `model_spec.json` inside the images: 0.19.0 image has 225 specs for 69 models (66 EXPERIMENTAL, 79 FUNCTIONAL, 80 COMPLETE). The 0.26.0 image
has 180 specs for 58 models (33 EXPERIMENTAL, 72 FUNCTIONAL, 75 COMPLETE). No spec in the dev or prod YAML sets `TOP_PERF`, and the generated LLM support page has no row for it.

Recent graduates and near graduates (device is a TTIS device name; QB2 is `P300X2`):

| Model | Hardware | Level | Numbers and evidence |
| --- | --- | --- | --- |
| Qwen3.8-27B (hybrid linear attention, GDN) | QB2, TP4 on four Blackhole chips | COMPLETE (release 0.25.0, tt-metal `978f93b`, vLLM `a723030`) | context 262144, batch 16; demo README: 39.4 tok/s/user and TTFT 67.9 ms at 128/128 batch 1; GPQA Diamond 9/10, Terminal-Bench 2.1 4/5, SWE-bench Verified 3/5. Perf target in `model_performance_reference.json`: theoretical TTFT 62 ms, 41 tok/s/user, so `complete` needs TTFT at most 124 ms and at least 20.5 tok/s/user; 128/128 batch 1 is the only graded point (`tests/workflows/test_qwen38_minimal_release.py` asserts exactly one graded point of 24). Largest tested prompt 261,892 tokens. |
| gemma-4-31B-it, gemma-4-12B-it | QB2 and Blackhole Galaxy | COMPLETE | context 262144, batch 32 (QB2), 128 (BH Galaxy), `tt_transformers`; tt-metal CI gate: first 10 of 198 GPQA Diamond questions, minimum 80 percent (`model_targets.yaml`, tolerance 0.2 around 100) |
| gemma-4-26B-A4B-it (MoE, sliding plus full attention) | QB2 | FUNCTIONAL | Shield release: GPQA 9/10, Terminal-Bench 2/5, SWE-bench 2/5; prefix caching at batch 1 only; "Chunked prefill is not supported" (demo README) |
| granite-4.2-30b | QB2 | FUNCTIONAL | CI gate: GPQA subset, "preserve the 6/10 release gate", batch 5 |
| Qwen3.6-27B, Qwen3.5-27B (GDN hybrid) | 4x P150, QB2 | EXPERIMENTAL | context 262144, batch 32; trace region 1 GiB; device sampling decode-only; nightly CI on 4x P150 since #5046; `swe_bench_verified` is disabled in `eval_config.py` ("timeouts from model limitations, re-enable once prefix cache or equivalent is enabled", issue 4675) |
| gpt-oss-120b (MoE) | T3K, Galaxy, QB2, BH Galaxy | EXPERIMENTAL everywhere | context 16384 on T3K, 131072 elsewhere; its eval budget notes record a traced decode of 6.0 tok/s at batch 1 on two chips (`docs/eval_execution_budgets.md`) |
| DeepSeek-R1-0528 | Galaxy, dual, quad | EXPERIMENTAL | context 2048 (Galaxy), 32768 (multi-Galaxy) |
| K2-Horizon-7B, Muse-Glimmer-30B, Kolibri-1, Qwen3.8-27B (Quetzal row) | QB2 | EXPERIMENTAL | new in the 0.26.0 cycle |
| Llama-3.1-8B, Mistral-7B-Instruct-v0.3 | N150, N300, T3K | COMPLETE | the long-running Wormhole entries; Llama-3.1-8B is FUNCTIONAL or EXPERIMENTAL on Blackhole boards |
| Qwen3-32B, Llama-3.3-70B | WH Galaxy | COMPLETE | Qwen3-32B: batch 32, context 131072, `sample_on_device_mode: all`, FABRIC_1D_RING |

What the Qwen, MoE, hybrid-attention and sparse-attention entries needed (from the specs and demo READMEs):

- Qwen3.8-27B (GDN): its own implementation directory in tt-metal (`models/demos/qwen38_27b_qb2`); a `vllm_metadata.json` and `hf-overrides` that rename the architecture (`TTQwen38ForCausalLM`); env vars that switch decode buckets (batch 1, 8, 16), compact decode, batched prefill and a prefill start-up warm-up; `trace_mode: decode_only`; a KV pool decoupled from `max_context` (`max_tokens_all_users_override: 1050592`); prefix caching switched off; a precision file (`config/precision.json`: bfp4 weights with LoFi fidelity, bf8 KV cache, float32 recurrent state, HiFi4 for sensitive ops); a device-side cache zeroing step that cut cache readiness from 5.79 s to 0.018 s on that checkpoint (`docs/cache_initialization.md`); a convolution-state PCC test for the linear-attention layers.
- gemma-4-26B-A4B (MoE with sliding windows): hybrid KV caches with 32-token pages; positions anchor the device sampling seeds so batch rebuilds do not restart a request's random stream.
- Muse-Glimmer-30B: a chunked-prefill configuration (`max_num_batched_tokens: 512`) and `override_generation_config: {"top_k": 32}`, because the device sampler covers top_k 1 to 32 only.
- Flash-Next (hybrid GDN and QSA attention, 512 routed experts, MTP): see section e. It is not in a TTIS spec yet.

## b) How graduation is measured

### What a spec carries

`workflows/model_spec.py` (`ModelSpecTemplate`, `DeviceModelSpec`) and the YAML under `workflows/model_specs/`:

| Field | Meaning (as in the code) |
| --- | --- |
| `weights` | list of Hugging Face repos; `hf_weights_repo` for a shared download repo |
| `impl` | an id that maps to repo URL and code path (for example `models/demos/qwen38_27b_qb2`) |
| `inference_engine` | `VLLM` and others (media, forge) |
| `device_model_specs[]` | one entry per device: `device`, `max_concurrency`, `max_context`, `max_tokens_all_users_override` (the KV pool in tokens), `default_impl`, `perf_targets_map`, `perf_reference[]`, `vllm_args`, `override_tt_config`, `env_vars`, `tensor_cache_timeout` (default 3600 s; 5400 on large models), `docker_shm_size`, `system_requirements` (firmware and KMD specifier with `STRICT` mode), `known_issues[]`, `eval_max_retries`, `supported_modalities` |
| defaults filled in | `block_size 64`, `max_model_len` from `max_context`, `max_num_seqs` from `max_concurrency` (divided by `data_parallel_size`), `max_num_batched_tokens` equal to `max_context`, `seed 9472`, `additional_config` carrying `override_tt_config` |
| `status` | one of the four levels |
| `has_builtin_warmup` | the model warms up inside its own runner; otherwise the container sends one request per padded input length (128, 256, 512, up to the maximum) then writes `/tmp/ready` (`reference_config/benchmarking/README.md`, "Warmup") |
| `metadata` | per weights: `reasoning_parser_name`, `tool_call_parser_name`, and the opt-in `benchmark_protocol: fixed_workload` |
| `docker_image`, `version`, `tt_metal_commit`, `vllm_commit` | pins; a dev entry may name an image only by immutable digest |
| `override_tt_config` keys seen | `sample_on_device_mode` (`all` or `decode_only`), `trace_mode`, `trace_region_size`, `fabric_config` (`FABRIC_1D`, `FABRIC_1D_RING`), `l1_small_size`, `worker_l1_size`, `dispatch_core_axis`, `fabric_max_packet_payload_size_bytes`, `kv_cache_usable_tokens` |
| vLLM args seen | `block_size`, `max_num_seqs`, `no-enable-prefix-caching`, `enable-chunked-prefill`, `max_num_batched_tokens`, `async-scheduling`, `reasoning_parser`, `tool_call_parser`, `enable_auto_tool_choice`, `hf-overrides`, `revision` and `tokenizer_revision`, `default_chat_template_kwargs`, `chat-template-content-format`, `structured-outputs-config`, `max-logprobs` |

### Evals

`reference_config/evals/eval_config.py`: `EvalConfig` per model, `EvalTask` per task. Tasks in use: `r1_gpqa_diamond`, `r1_aime24`, `r1_math500`,
`mmlu_pro`, `ifeval` variants, `humaneval_instruct`, `mbpp_instruct`, `livecodebench`, `ruler`, `mmmu_val`, `chartqa`, `docvqa_val`, and the agentic
`terminal_bench_2_1` and `swe_bench_verified`. Each has a `published_score` with a URL and, where measured, a `gpu_reference_score` with a URL (usually a
comment on a tracking issue).

Grading (`workflow_module/eval_noise.py`, `reference_config/evals/eval_config.py` line 175): a task passes when `score / reference >= 1 - tolerance`
(tolerance default 0.05; the older prose in `docs/add_support_for_new_model.md` says 0.1 and is out of date), or when the shortfall is within 1.96
binomial standard errors of the reference. The noise rule applies only to per-sample correct or incorrect metrics. The comment gives the reason: a task
with a reference in the single digits has a 5 percent bar smaller than its own sampling noise (Llama-3.2-3B IFEval scored 9.43 then 8.32 on the same 541
prompts). An eval block marked `incomplete` blocks release whatever the tier says. A block with no reference score is `NA`, which the acceptance check treats as
not blocking and the strict qualification verdict treats as not measured.

Release-sized runs are expensive, so CI uses reduced modes (`limit_samples_map`): Qwen3.8-27B runs 10 GPQA questions, 10 Terminal-Bench tasks and 5
SWE-bench tasks in nightly. `wall_clock_timeout_seconds` bounds one eval subprocess; a timeout reports an incomplete failed task (`docs/eval_execution_budgets.md`).

### Performance runs

`reference_config/benchmarking/README.md`. Client tools: `vllm bench serve` (default), AIPerf, GenAI-Perf, GuideLLM (`docs/benchmarking_tools.md`). The default
sweep and request counts are in the README; the graded points are the entries in `model_performance_reference.json`. For acceptance it recommends: a warm
server, one discarded pass per point, at least three measured repetitions with medians, at least 32 requests at concurrency 1 and 4 times the concurrency
(minimum 64) at high concurrency, exact output lengths, zero failed requests, and a record of image tag, client version, command and a `tt-smi -s` snapshot.
The opt-in `fixed_workload` protocol enforces most of this: pinned client and tokenizer hashes, fixed seeds, greedy decoding, ignore-EOS, and failure of a
point on partial requests or wrong token counts "even if the client exits 0".

### The report

The release run writes markdown plus JSON (`report_module/`): per-point tables with `functional`, `complete` and `target` checks (PASS, FAIL or NA) and the
measured to target ratio, an acceptance block with named blockers, and categories (Benchmarks, Evals, Spec Tests, Agentic Targets). Acceptance keys are
`acceptance_criteria`, `acceptance_blockers`, `acceptance_criteria_metadata`, `acceptance_summary_markdown`. A model-status mask makes failures
informational and says so in the message ("informational: model status=EXPERIMENTAL"). A `should` priority from a requirements document does the same.
The release README requires "any models with regressions that are being added as default impl" to be "clearly listed in the model waiver section of release notes".

## c) Engineering patterns that matured

Evidence is from the specs, demo READMEs and tt-metal CI files named. "Generic" means it could become a can gate, default or spec field without a model.

| Pattern | Evidence | Generic for tt-orchard |
| --- | --- | --- |
| Weight and kernel caching, cold start separated from warm | benchmarking README: first boot "can take up to an hour on large models"; "Never benchmark a first boot"; `tensor_cache_timeout` per spec | Yes. The can already records a cold start; add a rule that a speed number names whether the kernel cache was warm (the Qwen3.5 card does this by hand). |
| Trace capture and its separate readiness | `/health` returns 200 before trace capture finishes; `/tmp/ready` and the log line "Background trace capture completed successfully" mark the end; `has_builtin_warmup` | Yes. `can boot` waits on `/health` and `ready_contains`; a gate on "ready line seen before the first timed request" is possible. |
| Trace region sizing as data | `models/model_trace_region_sizes.yaml`: "Reserved budget, not measured trace bytes ... measure before shrinking" | Yes. `trace_region_bytes` is already a can field; add a note field saying measured or reserved. |
| Paged KV with an explicit pool budget | `max_tokens_all_users_override`, `kv_cache_usable_tokens`, block size 32 or 64 | Yes. A can field for the KV pool in tokens, checked against `max_model_len * max_num_seqs`. |
| Prefix caching as a per-model switch | most QB2 entries pass `no-enable-prefix-caching`; gemma-4-26B-A4B validates it at batch 1 only; TTIS has a prefix-cache benchmark sweep (`llm_module/prefix_cache`) | Yes as a card line and a spec field; the sweep is a later harness item. |
| Chunked prefill | Muse-Glimmer sets `enable-chunked-prefill` with `max_num_batched_tokens` 512; gemma-4-26B-A4B says it is unsupported; Flash-Next prefills in 2048-row slabs | Yes as a declared capability on the card (supported, unsupported, limit). |
| Speculative decode and MTP | Flash-Next `--mtp 4`: 18.14 ms per token against 26.18 plain; TTIS has `spec_decode` runs and metrics (`llm_module/spec_decode`) | Later. A can field for "draft tokens" and a card row for acceptance rate. |
| Multi-device topology | QB2 is a 1x4 mesh of four Blackhole chips (`MESH_DEVICE (1, 4)`, `p300_x2_mesh_graph_descriptor.textproto`), `FABRIC_1D` or `FABRIC_1D_RING`; Qwen3.6 on P150X4 and P300X2 share an impl | Yes. `mesh`, `device_count` and `mesh_fabric` exist; add the mesh-descriptor path as a checked env entry. |
| Dtype and fidelity policy as a file | `config/precision.json` in Qwen3.8 and Gemma-4-26B-A4B ("fixed by `config/precision.json`"); the repo has a `ttm-datatype-sweep` skill | Yes. A can that ships a precision policy states it on the card (weights dtype, fidelity, KV dtype, recurrent dtype). |
| Sampling on device | `sample_on_device_mode: all` or `decode_only`; device sampler covers top_k up to 32 so `override_generation_config top_k 32` keeps requests on it | Yes. Card capability line: sampling above temperature 0, top_k limit, seed behaviour. |
| Long context | context 262144 across Qwen and Gemma entries; "Largest tested input sequence length: 261,892 tokens"; K2-Horizon validated to 524288 | Yes. The card states the largest tested prompt, separately from the advertised maximum. |
| MoE dispatch limits | Flash-Next: main's `moe_compute` takes at most 128 tokens per call here, so a 2048-row slab needs 16 calls per layer (PR 58690) | Model specific; a can with MoE states the per-call token limit. |
| Linear attention state | Qwen3.8: recurrent state in float32, a conv-state PCC test; PR 63316 fuses the GDN prefill scan; PR 63111 fixed RoPE with transformers 5.17 | Model specific, but the float32 recurrent state belongs on the card's precision line. |
| Tool and reasoning parsers | every chat spec names `reasoning_parser` and `tool_call_parser`; `default_chat_template_kwargs` turns thinking off by default for Qwen3 because "plain requests burn max_tokens inside `<think>`" | Yes. Both are spec fields and a boot-check request option (`request_extra` already covers the template switch). |
| Immutable inputs | `revision` and `tokenizer_revision` pinned to the validated checkpoint ("the hub's main moved to a new checkpoint on 2026-10-01"); Quetzal rows carry `QUETZAL_BUNDLE_MANIFEST_SHA256` and an image digest | Yes. The can pins one revision for weights; add the tokenizer revision when it differs and a bundle digest in `can.json`. |
| Request isolation, determinism | spec tests: seed reproducibility, non-uniform seeding; gemma-4-26B weekly CI: "sampling isolation", cache continuation | Later (needs a served model); a boot-check option that sends the same seeded request twice already exists for audio (`repeat_identical`). |
| Health and watchdog | Flash-Next `--stall-seconds` (exit 1 after 300 s without device progress); TTIS `/tt-liveness` for media | Yes for the can: a stall limit is a run.sh setting a gate can look for. |

## d) Gaps between a can and a graduated entry

### Fields

Left column: what a graduated TTIS entry carries. Right column: what a can has today and what to add. "Matches" means the can already covers it.

| TTIS field | In a can today | Proposed addition |
| --- | --- | --- |
| `weights` and `revision` | matches (`[weights] repo`, 40-character `revision`) | add optional `tokenizer_revision` when it differs |
| `impl` (repo URL, code path) and `tt_metal_commit`, `vllm_commit` | wheels pinned in `requirements`; the card's Provenance table | optional `[provenance]` with the tt-metal commit, vLLM version and plugin ref, written into `can.json` and checked against the card |
| `device` | `mesh`, `device_count`, `arch` | matches; add the mesh-descriptor path as a checked env entry |
| `max_context`, `max_concurrency` | `max_model_len`, `max_num_seqs` | matches |
| `max_tokens_all_users_override` | none | optional `kv_pool_tokens`, validated as at least `max_model_len` and at most what the card says |
| `vllm_args` | `extra_args`, with `OWNED_FLAGS` refused | matches; add known-flag validation for `reasoning_parser` and `tool_call_parser` against the metadata file |
| `override_tt_config` | `tt_config`, `mesh_fabric`, `trace_region_bytes`, reserved keys refused | matches |
| `env_vars` | `[env]` | matches |
| `status` | none | `[readiness] status` is a claim the author makes; the card shows it with the four definitions. The gate checks that the evidence for the claimed level exists (below). TTIS keeps promotion as a human decision, so the can does not derive it |
| `has_builtin_warmup`, `/tmp/ready` | `[check] ready_contains` | add `[check] warm_requests` (discarded requests before any timed one) and record whether `/health` came before the ready line |
| `system_requirements` (firmware, KMD, mode) | none | optional `[requires] firmware`, `kmd`, `tt_metal` specifiers, printed in the card's Prerequisites and in `can.json` |
| `perf_reference` (`isl`, `osl`, `max_concurrency`, `num_prompts`, targets by tier) | one floor, `min_decode_tok_s` | optional `[[perf_point]]` with `isl`, `osl`, `concurrency`, `ttft_ms`, `tput_user` and a `theoretical` ceiling; the card's speed table rows must match these points; a pure function derives the functional (10 percent) and complete (50 percent) tiers exactly as TTIS does |
| `EvalConfig` (task, published score, GPU reference, tolerance, sample counts) | `[check] expect` (one regular expression) | optional `[[eval]]` with `task`, `published`, `reference`, `reference_source`, `n`, `measured`; a stdlib function applies the ratio rule and the 1.96 noise rule; the card gets an accuracy table from it |
| `known_issues` (waivers with a reason) | the card's Limitations section | optional `[[known_issue]]` with `scope` and `reason`; the gate requires each to appear in Limitations |
| spec tests (parameter conformance, tool-call schema) | one chat request, one streamed request, a decode floor | later, with hardware: run TTIS `spec_tests` against the booted can (section f) |
| `docker_image` digest, Quetzal manifest sha256 | `SHA256SUMS` over the bundle | add the digest of `SHA256SUMS` to `can.json` and the card, so a registry entry can pin the exact bundle |
| generated support page and release report | the card | a card section "Readiness" generated from the fields above (section f, item 3) |

### Checks that already match

- Pinned weights revision and exact-pin requirements (TTIS pins `revision`, `tokenizer_revision` and image digests).
- Launch limits in a spec, and refusal of flags the can owns (`OWNED_FLAGS`).
- A warm second request and a decode-rate request in the boot check (`orchard/can.py`, boot check), which matches the TTIS advice to measure after warm-up.
- The label vocabulary of `docs/benchmark-rules.md` (warm-up, three repeats, min/median/max, load average) matches the TTIS acceptance procedure (discarded pass, three repetitions, medians, record of the environment).
- The card's `Number | Value | Label | Evidence` table with `measured`, `single sample` and `estimate` labels matches the TTIS split between measured targets and a theoretical ceiling.
- The CPU reference gate (`orchard/skills/reference-gate.md`) has no TTIS counterpart. TTIS uses published scores and GPU references. The reference gate checks the model against its own CPU run before any chip number counts. This is stricter on correctness and does not cover task quality.

### Checks TTIS has that a can lacks

- A task-quality number against a published or GPU reference (GPQA subset, Terminal-Bench, SWE-bench, IFEval). The can has agreement with a CPU reference and one expected answer.
- Parameter conformance (seeds, stop, penalties, logprobs) and tool-call schema conformance.
- A stated largest tested context, a KV pool, prefix-cache and chunked-prefill status.
- Stress and endurance runs; a stall limit.
- A readiness level with the definition printed beside it.

## e) Qwen4 and Flash-Next

Evidence for the sibling study; nothing below was run.

- tt-metal PR 58690 "Add Qwen3.8 Flash Next model", merged 2026-10-29, branch `codex/publish-qwen38-flash-next`, in `models/demos/blackhole/qwen38_flash_next`. Hybrid GDN and QSA attention, 512 routed experts, an MTP layer and a vision tower on four Blackhole p150 chips as a 1x4 mesh. Routed experts in BF4, dense weights in BF8, one resident request slot, 32,768 tokens by default, served with `--mtp 4`. vLLM serving and CI registration "come in later PRs". Validation in the PR body: JSON acceptance record 96/96; teacher-forced agreement with the Hugging Face model over 8,416 positions: top-1 0.9525, KL 0.0528; MTP decode 18.14 ms per token against 26.18 plain; TTFT at 32K tokens 23.0 s with prefill slabs against 39.7 s chunked; 262K context with MTP, 30 GPQA requests, 369,402 tokens, none garbled; GSM8K 58/64 in the demo.
- Gates in that tree: `tests/pcc_thresholds.json` sets 0.99 (GDN, QSA), 0.98 (layer, MoE) and 0.95 (layer with injection); the launcher's `--require-json-96` refuses to serve unless the recorded greedy stream matches the CPU reference 96/96; reference agreement, task quality and runtime reuse are "separate gates". The checkpoint is hash-pinned (ModelScope revision `2741eec1...`, per-file SHA256), the cache has identity and checksum checks, and Transformers 5.16.1 is needed for the `qwen4_exp` configuration.
- The tree's README states: "No CI pipeline runs the demo or the device tests yet". Status is not in TTIS: a case-insensitive search of `origin/main` for `flash-next` and `flash_next` in every YAML, JSON, Python and Markdown file found no match.
- Related open work in tt-metal (titles only, state as of 2026-10-31): PR 63316 GDN fused prefill (open), 63541 `rotary_embedding` shape at unaligned lengths (open), 63633 `binary_ng` float32 with bf16 operands (open), 60469 `moe_compute` packed routing without a 512-token cap (open), 63440 GLM-5.3-Flash on Blackhole Galaxy (open), 58833 SDPA validation and MLA (open). Merged: 59897 `moe_compute` empty w2 slice, 60705 `moe_compute` xfail on the WH 7x8 grid, 63111 Qwen3.8 RoPE fix.
- TTIS side: PR 5474 and 5486 (merged) add long-context sweeps (256K, 512K, 1M) and AgentX to every release for MiMo-V2.6-Flash. PR 5454 (open) is its Blackhole Galaxy dev spec. A model with 1M context therefore gets its own sweep points and a 6.6 GB KV budget line.
- Glean, unverified: the "QB2 Model Targets" sheet (rev 0.29) lists `Qwen/Qwen4 (TBD)` as rank 0 and Qwen3.8-Flash-Next as rank 1, "First model on the Qwen4 architecture", 180B total parameters, BF16, license `other` (`qwen-community-1.0`) with "legal sign-off" still needed. A model-requests Slack thread says Qwen4 "will be based off Qwen3.8-Flash-Next's architecture just like how Qwen3-Next revealed Qwen3.6/3.8-27B's architecture". A separate prebuilt release repository for QuietBox and LoudBox exists (name only; not read).
- What this means for a can: the license is not Apache-2.0, so the card needs the license section that the Stable Audio 3 can already uses (the agreement quoted word for word). The 360 GB checkpoint and 69 GB expert cache make the weights a pointer and the first start long, as in the Qwen3.5 can. The MoE per-call token limit and the float32 recurrent state belong on the card.

## f) Ranked preparation list

No hardware needed for items 1 to 6.

1. Done in this branch: `[can] card_profile = "catalog"`. The gate requires an `Expected performance` section with a table row that has a number and a `Limitations` section with text. These are the two sections tt-model's `card_publish_gaps()` requires, and match the two things a graduated entry carries. The Qwen3.5 can sets it. Tests failed first, and each guard was mutated (gate wiring, allowed key, fenced headings, digit rule, empty Limitations, unknown profile) to see a test fail.
2. A `[[perf_point]]` table and a pure function that derives TTIS tiers from a theoretical ceiling (10, 50 and 100 percent), with the card's speed table cross-checked against it. Small, pure Python, and it makes a can's speed claims comparable to a TTIS entry.
3. A readiness-report generator: `tt-orchard can report` reads `can.json`, the boot-check evidence and the optional tables and writes a "Readiness" card section in the TTIS vocabulary (status claim, definition, each point with PASS, FAIL or NA, waivers). Depends on 2 and 4.
4. A `[[eval]]` table with the ratio rule plus the 1.96 noise rule (port the 20-line logic of `workflow_module/eval_noise.py`, with its license header and a pointer, or call it from a pinned checkout). The card gets an accuracy table and `NA` when no reference exists.
5. A `[[known_issue]]` table, `[requires]` (firmware, KMD, tt-metal), `kv_pool_tokens`, `tokenizer_revision`, `[provenance]` commits, and the digest of `SHA256SUMS` in `can.json`. Each is an optional field with validation and a card line.
6. A card capabilities block that every vLLM can fills: prefix caching, chunked prefill, sampling on device (top_k limit), speculative decode, largest tested prompt, parsers. The gate checks that each named capability line exists. This keeps cans comparable to each other and to TTIS pages.
7. With hardware, later: harness wiring. TTIS documents that client workflows "only send HTTP requests", so `run.py --workflow spec_tests` (parameter conformance) and `--workflow benchmarks` can run against a booted can through `--server-url`. This needs a model-spec JSON for the can (the `ttm-tti-release` skill says to pass one with `--model-spec-json`; not tried). The 24-point default sweep is too long for a boot check; use two points (128/128 at concurrency 1, and a long-prompt point) as the can's own graded set.
8. With hardware, later: a GPQA 10-question subset and a Terminal-Bench 5-task subset as the optional quality gate, as the tt-metal Tier 3 jobs do (`model_targets.yaml`: `gpqa: 100`, `tolerance 0.1` gives "at least 9/10" for Qwen3.8-27B).
9. With hardware, later: stress and endurance run (TTIS `stress_tests`), and a stall limit in `run.sh`.

## Proposed GitHub issues (not filed)

1. Title: "can: derive TTIS performance tiers from a theoretical ceiling". Body: A graduated tt-inference-server entry grades each point against 10, 50 and 100 percent of an analytical ceiling (`functional`, `complete`, `target`). Add an optional `[[perf_point]]` table to the can spec (`isl`, `osl`, `concurrency`, `ttft_ms`, `tput_user`, `theoretical`) and a pure function for the tiers. Gate: the card's speed table must list the declared points. No hardware needed.
2. Title: "can: accuracy table with the ratio and noise rules". Body: Add `[[eval]]` records (task, published score, reference score and its source, sample count, measured score) and grade them with a ratio rule (`score / reference >= 1 - tolerance`, default 0.05) and a within 1.96 standard errors rule for per-sample metrics. Missing reference gives `NA`, never a pass. Render an accuracy table on the card.
3. Title: "can report: generate a Readiness section for the card". Body: Read `can.json` and the boot-check evidence; write a section with the claimed level (EXPERIMENTAL, FUNCTIONAL, COMPLETE, TOP_PERF), its definition, the evidence per tier and the waivers. The level stays a claim by the author and is not derived. Depends on issues 1 and 2.
4. Title: "can: declare capabilities on the card and check them". Body: Prefix caching, chunked prefill, sampling on device and its top_k limit, speculative decode, largest tested prompt and parsers each get a line on the card of a vLLM can. The gate checks that the lines exist. Reason: TTIS specs and the QB2 demo READMEs state these per model, and they differ between models of one family.
5. Title: "can: run TTIS spec_tests and benchmarks against a booted can". Body: tt-inference-server client workflows run against any OpenAI-compatible server. Try `run.py --workflow spec_tests --server-url ...` with a model-spec JSON for the can after `can boot`, record which tests apply, and keep the two benchmark points the can grades. Needs a lease and a model-spec file; document failures that come from the harness and not the can.
6. Title: "can: pin the tokenizer revision, bundle digest and system requirements". Body: Add optional `tokenizer_revision`, `[requires]` (firmware, KMD, tt-metal) and the digest of `SHA256SUMS` in `can.json` and the card, matching what TTIS specs and Quetzal manifests pin.

## What was not checked

- The reason the 0.19.0 image carries a spec whose `release_version` is 0.20.0.
- The tt-metal Tier 3 job definitions that run the Qwen3.8 weekly tests (only their config and README were read).
- Slack and Glean claims in section e, and the "unusable TTFT at large ISL" note on gpt-oss-120b in a leadership sheet (not used above).
- Accuracy of the TTIS docs where they disagree with code (the eval tolerance default is 0.05 in code and 0.1 in the how-to).
