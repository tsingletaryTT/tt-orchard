# Overnight 2: MiniCPM3-4B (multi-latent attention, dense MLP) on one board

Started 2026-10-28 20:15 Pacific by overnight lead 2. Operator's target: three models that could not be brought up the day before.
My class: an open-weights text model whose blocker is the attention side. This log is updated hourly.

## Choice (first hour)

Model: `openbmb/MiniCPM3-4B`. Licence Apache-2.0, open download with no gate, 62 layers, hidden 2560, 40 heads, about 4B parameters (8 GB in bfloat16).

Why this one:

- The only attention piece with no ttnn equivalent is multi-latent attention (MLA): a low-rank query (`q_a_proj`, norm, `q_b_proj`), a
  compressed key-value latent (`kv_a_proj_with_mqa`, norm, `kv_b_proj`) with a separate 32-wide rotary part shared by all heads, and a value
  head width (64) that differs from the key head width (96). Everything else is Llama-like: RMSNorm, a gated SiLU MLP, rotary embeddings, a tied
  or untied head.
- The MLP is dense. DeepSeek-V2-Lite (the other candidate, 16B, MoE with 64 routed experts) would add a mixture-of-experts path on top of MLA,
  and the licence is a custom one. Two new pieces in one night is worse than one.
- MiniCPM3 adds three scalings that a port must not miss: the embedding is multiplied by `scale_emb` (12), each residual branch by
  `scale_depth / sqrt(num_layers)` (1.4 / sqrt 62), and the logits input is divided by `hidden_size / dim_model_base` (10). It also uses long-rope
  factors, which equal 1 times a table here (short and long factors are the same list, and the length scale is 1).
- Size: 8 GB bfloat16 fits one chip (Blackhole p300 board has two chips with 32 GB each); the decode floor on one chip is about 16 ms per token.
- Reference: the model's own `modeling_minicpm.py` (remote code, written for transformers 4.41; the shared environment has 5.19, so the reference may
  need a shim, and if it does the log says so). A plain-torch reference written from the same file is the fallback.

What tt-metal already has: `models/demos/deepseek_v3/tt/mla/mla1d.py` (2681 lines, written for a multi-chip DeepSeek-V3 mesh with its own weight
loader, so it cannot be reused as it stands). Its structure (q_a, q_b, kv_a, kv_b, a separate rope part) is the same as MiniCPM3's and guides the port; the port
uses the absorbed form with the existing flash MLA op (the op the DeepSeek code calls), which keeps the cache at 288 values per token.

Plan: (a) CPU reference arrays; (b) the whole model on one chip with MLA on the host; (c) MLA as ttnn ops; (d) agreement, then benchmark.
Fallbacks are recorded here with the reason.

## Update 1 (about 20:35 Pacific): the whole model runs on one chip

Environment: one Blackhole p300 chip (gozer lease on one board), shared Tenstorrent venv (torch 2.7.0, transformers 5.19.0, ttnn 0.80.0 from the host tt-metal tree).

CPU reference. `scripts/minicpm3/ref_model.py` is plain torch written from the model's source. transformers 5.19 has a native `minicpm3` class; with a copy of the
config that adds `rope_parameters` (`rope_type` longrope, `factor` 1.0) the two give the same 24 greedy tokens in bfloat16 (`check_ref.py --native`). Remote code
from the model repository was not run (the sandbox refused `trust_remote_code`). The float32 reference text for the prompt is coherent.
Recorded runs (`make_reference.py`): `float32` (reference), `bfloat16` plain torch and `native` transformers in bfloat16 (the environment floor pair).

Probes on the chip (`exp_ops.py`, `exp_mla.py`). Every op needed worked on first call at the model's shapes: matmul with `transpose_b`, `update_cache` into a
`[1,1,T,288]` latent cache, head reshape and transpose, slice and concat on the last dimension, batched matmul over 40 heads, `rms_norm` at 2560 and 256 wide.
`ttnn.transformer.flash_multi_latent_attention_decode` (the existing DeepSeek multi-latent attention op) failed at these shapes with `TT_FATAL: Tree reduction max 6
rounds (64 cores/head), got 110 cores/head`: one key-value head at batch 1 puts all 110 Blackhole cores on one head. Passing an `SDPAProgramConfig` with an 8 by 8
grid fixes it, and the output matches torch (PCC 0.99995, relative L2 0.010 at 40 heads, 256 latent plus 32 rope, 100 cached positions, `cur_pos_tensor`).
So no new kernel is needed: the missing piece was the composition around the existing op (absorbed up-projections, below).

The port (`chip_model.py`): bfloat16 weights, one token per step (the prompt goes through the decode step one token at a time), scalings folded into weights
(`scale_emb` into the embedding, `scale_depth/sqrt(62)` into `o_proj` and `down_proj`, `1/10` into the final norm). Attention in the absorbed form: the query is
projected, its 64-wide no-position part is multiplied by the per-head key up-projection (`kv_b_proj` rows) to 256-wide latent queries, the rope part gets rotary
embedding (rotate-half as a 32 by 32 matmul), the cache holds 288 values per token for the single shared head, the existing flash MLA op attends over it, and the
per-head value up-projection is applied to its output before `o_proj`. The stage-b host-fallback mode exists in the code (`attn="host"`) and was not run, because the
chip op passed.

First hardware numbers (bfloat16 weights, default matmul config, no trace, one host round trip per token):
- Teacher-forced greedy match against the float32 reference: 16 of 16 next tokens (40-token prompt).
- Decode: 18.7, 18.8, 18.75, 18.9 tok/s (repeat 0 is the warm-up) over 48 steps each.
- Floor (`orchard.decode_floor`, one die, 512 GB/s, bfloat16): 15.9 ms per token, 62.8 tok/s. The measured speed is about 30 percent of it.
- Harness fix needed on the way: `orchard.model_config` refused this config ("model config has no v_head_dim"); MiniCPM3's code sets it to hidden size over heads
  (64). Fixed with a type table and a test in this branch.

First-divergence probe, chip against float32, floor pair bfloat16 plain torch and bfloat16 transformers-native. With the default tolerance (3 times the floor) layer 0 already
leaves, at 1.0 percent against a floor of 0.17 percent: the floor holds two CPU runs and does not hold the chip's matmul arithmetic. With a stated tolerance of 5 percent the
first block that leaves is layer 14 (0.028 before, 0.077 after, slope 1.076). Row 0 (the first token, norm 1,460 against about 45 for the others) carries 99.6 percent of the squared
error: channels 48 and 1292 of that row are 8 percent too large. Layers 18 to 61 keep that error (relative L2 0.07, PCC 0.9999). The error enters in layer 14's MLP or at its
add (the attention section is inside tolerance). Next: test HiFi4 with 32-bit accumulation in every matmul (`fidelity="hifi4"`).

## Update 2 (about 20:50 Pacific): the layer 14 error found and fixed; agreement is inside the floor-based bars

Cause of the layer 14 error (`scripts/minicpm3/exp_mlp14.py`, one MLP on the chip with the recorded layer 14 input row for token 0, norm 223 before the norm). The first operation is
already wrong: `ttnn.rms_norm` with its default compute config is 3.7 percent off on that row (torch on the same bfloat16 input is the comparison). The gate, up and
product errors (3.9, 3.8 and 6.0 percent) follow from it. With a compute config that has `fp32_dest_acc_en=True` the norm is 0.08 percent off (HiFi4 and approximate
mode make no difference; LoFi is 2.0 percent off). The matmul fidelity setting I tried first (HiFi4 with 32-bit accumulation on every linear) did not change the
layer 14 error. The operation-level search of the first-divergence probe is what located the cause. The fix is `self._rms` in `chip_model.py`. Layers other than 14 were not
visibly harmed because only the first token of a prompt carries channels that large.

Agreement after the fix (chip with default matmul config, bfloat16 weights, against the float32 reference; floor pair is bfloat16 plain torch and bfloat16 transformers-native):

| block | chip relative L2 | slope | floor relative L2 |
|---|---|---|---|
| layer 0 | 0.0086 | 0.9985 | 0.0017 |
| layer 12 | 0.0177 | 0.9983 | 0.0062 |
| layer 24 | 0.0087 | 0.9927 | 0.0013 |
| layer 48 | 0.0107 | 0.9963 | 0.0029 |
| layer 60 | 0.0387 | 1.0153 | 0.0133 |
| layer 61 | 0.0555 | 1.0107 | 0.0196 |

Prefill last-position logits: PCC 0.9991, relative L2 0.018. Decode step 0 and 15 logits: PCC 0.998 and 0.999, relative L2 0.042. Teacher-forced next-token match is 16 of 16.
The chip is 2.8 to 6 times the CPU floor at the blocks above, as expected: the floor holds two CPU runs and the chip adds bfloat16 matmul arithmetic. The earlier run
(default `rms_norm`, same weights) is the known-bad control: the probe located layer 14 in it, with a 5 percent stated tolerance.

Speed: HiFi4 on every linear costs speed (16.5 against 18.8 tok/s) and does not help agreement, so it is off. Trace capture with a one-hot cache write
(position from a device tensor) gives 18.5 tok/s, the same as no trace: the time is on the device (about 54 ms per token against a 15.9 ms floor) and dispatch is not the cost.
Next: a timing breakdown (attention, MLP, head) and a DRAM-bandwidth probe of the decode matmuls (`exp_matmul.py`).

Probes done for the trace path: `paged_update_cache` works with a height-sharded input and an index tensor; one-hot row write with broadcast multiply and `assign`
works; the rope table by position through `ttnn.embedding` works; trace capture and replay with `copy_host_to_device_tensor` works.

## Update 3 (about 21:00 Pacific): 22.5 tok/s traced; where the time went

Timing method. A traced step is replayed with the input tensors updated by `copy_host_to_device_tensor`; the script (`time_trace.py`) times the input copy, the trace
execution and the logits read back separately, and a `skip` set leaves out parts of the step so each part's cost can be read from the difference. Sub-sections of
the attention were measured the same way. The device is shared, so single numbers move by a few percent.

| part (62 layers, bfloat16, one chip) | ms per token |
|---|---|
| whole step as first written (opt 1, default matmul config) | 51 to 56 execute, plus 8 to 9 for the logits read back |
| MLP (gate, up, down, norm, add) | 21.6 |
| LM head | 1.6 |
| attention and the rest | 32 |
| of that: the two per-head matmuls (`q_nope @ W_UK` with the rope part, and `o @ W_UV`) | 12.6 |
| of that: key path (projection, norm, rope, concat) | 4.6 |
| of that: flash MLA op | 1.1 |
| of that: cache write | 0.3 |

Three changes took the step from about 54 ms execute plus 9 read to about 43 plus 1:
1. The per-head matmuls (40 batches of a single row against a different weight each) ran 85 and 127 microseconds each under the default program config. An explicit
   `MatmulMultiCoreReuseProgramConfig` with one batch per core on an 8 by 5 grid runs them in 16.5 and 9.6 microseconds, with the same error (HiFi2; the same config
   with LoFi is 4 times less accurate: relative L2 0.026 against 0.006). `exp_bmm.py` has the sweep. This saved about 11 ms.
2. The logits come back in row-major layout (147 KB) and not tile layout (32 rows, 4.7 MB): the read back fell from 8 to 9 ms to under 1 ms.
3. Opt 2 of the attention (rotary on the whole 96-wide query with cos 1 and sin 0 on the no-position part, the key up-projection and the rope pass-through in one
   batched matmul with a 288-wide result, the key's rotate-half folded into its projection, and a cache write through `paged_update_cache` with the position
   in a device tensor) cuts about 10 ops per layer. Alone it did not change the time (the cost was in the two batched matmuls); it is what the trace needs.

Result (bfloat16 weights, default matmul config, traced step, host argmax, one token per replay): 22.4, 22.7, 22.6 tok/s over 48 steps after a warm-up run, which is 36 percent of the 62.8 tok/s floor.
Agreement is the same as before the speed changes (first-divergence against float32: layer 0 0.8 percent, layer 61 5.9 percent, last-position prefill logits PCC 0.999,
decode logits PCC 0.998, 16 of 16 teacher-forced tokens). The generated text is the same sentence as the float32 reference.

DRAM bandwidth probe (`exp_matmul.py`, one weight per call, 20 calls per trace): the interleaved bfloat16 matmuls reach 400 GB/s for gate and up (2560 by 6400) and
for the LM head, but only 253 GB/s for down (6400 by 2560) and 198 GB/s for `o_proj`. DRAM-sharded weights with a sharded input reach 479 and 374 GB/s on those two
(down 130 to 68 microseconds, `o_proj` 66 to 35). bfloat8_b weights are not faster at HiFi2 (the matmul is not bandwidth bound there). The DRAM-sharded LM head failed
with a dataflow-buffer error at 8 cores. Not applied yet.

## Update 4 (about 21:15 Pacific): DRAM-sharded MLP and o_proj, a 922-token prompt, 24 tok/s

DRAM-sharded weights (`--shard`) for the three MLP matrices and `o_proj`, with the activations width-sharded across 40 cores (5 by 8) and the gate and up outputs feeding the product and
down projection without leaving the sharded layout. Per MLP layer in isolation (`exp_mlp_block.py`, norm to residual add): 350 microseconds interleaved, 283 sharded. In the model
the traced step goes from 43.3 to 38.5 ms of device time. bfloat8_b weights gave no further gain at HiFi2 (278 microseconds), so weights stay bfloat16.

Breakdown now (62 layers, ms per token of device time, same skip method as above): MLP 17.7, LM head 0.9, attention and the rest 19.8. Inside the attention: key path 4.4, flash MLA op 1.0, the two per-head
matmuls 1.0, rotary on the query 0.7, cache write 0.25, everything else (norms, the query and output projections, reshapes, residual adds) 12.1.

Agreement with the sharded config (first-divergence against float32; 3 times floor is not met by the chip, as before):

| run | prompt tokens | layer 0 | layer 30 | layer 61 | last-position prefill logits PCC | decode logits PCC (step 0, step 15) | teacher-forced tokens |
|---|---|---|---|---|---|---|---|
| short | 40 | 0.0077 | 0.0063 | 0.0634 (floor 0.0196) | 0.9989 | 0.9983, 0.9982 | 16 of 16 |
| long | 922 | 0.0118 | 0.0182 | 0.0764 (floor 0.0346) | 0.9968 | 0.9992, 0.9991 | 16 of 16 |

The long prompt is 3.2 KB of one of this repository's own documents plus an instruction to summarize it; it checks the cache and the flash op over about a thousand positions
(the cache was sized to 1,024). The prefill took 81 seconds (token by token, with a host read of the stream after every layer section, which is the recording cost).

Decode speed from the same traced step: 22.4 to 23.1 tok/s over 48 steps after a 40-token prompt, and 24.2 to 24.6 tok/s after the 922-token prompt (3 measured repeats each, one warm-up each;
the host was loaded by other work, load average about 15, which shows in the spread). The device time of the step is 38.5 ms (26 tok/s), so about 5 ms per token is host work around the trace.

## Update 5 (about 21:35 Pacific): 33.7 tok/s in bfloat16, 40.1 with 8-bit MLP weights; a record per the benchmark rules

Opt 3 of the port (`--opt 3`) keeps the residual stream width-sharded in L1 across 40 cores for the whole step. Two measured per-op facts drive it
(`exp_srms.py`, `exp_layer_ops.py`): `rms_norm` on the interleaved 2560-wide row takes 42 microseconds, and on the sharded row 6.5 microseconds with the same
accuracy (relative L2 0.0036 on a row with channels at 1,000, with 32-bit accumulation); and the three thin projections that start the attention (2560 to 768, 256
and 64) take 37, 33 and 32 microseconds each under the default program config but one fused 2560-to-1088 projection takes 23 under a 1D multicast config.
Opt 3 uses the sharded norm everywhere (the MLP chain then needs no conversion at all), one fused projection (query input, key latent, key rope part and its rotate-half
form), and 32-bit (float32) logits. The last change matters for agreement: with bfloat16 logits (steps of 0.125 between 16 and 32) two of 16 teacher-forced steps flipped at
margins of 0.395 and 0.225; with float32 logits there are no flips.

Result, benchmark record `docs/run-logs/evidence/2026-10-29-minicpm3/bench_opt3_bf16.json` (written by `run_chip.py bench` with `orchard.resident.summarise_runs` and
`orchard.decode_floor`; one chip of a p300 board, traced step, 1 warm-up run and 3 measured runs of 64 tokens after a 40-token prompt, host load average 8.7):
decode 33.65, 33.65, 33.57 tok/s (min 33.57, median 33.65, max 33.65); label "benchmark"; 54 percent of the 62.8 tok/s floor (15.9 ms per token at 512 GB/s for 8.1 GB);
prefill by decode steps 1.19 s for 40 tokens. Device time of the step is 29.3 ms; the host adds under 1 ms.

Variants of the MLP weights and math fidelity (everything else equal; first-divergence against the float32 reference on the same 40-token prompt and 16 teacher-forced steps):

| MLP weights, matmul fidelity | decode tok/s (3 repeats, median) | layer 30 error | layer 61 error (CPU floor 0.0196) | prefill logits PCC | decode step 0 logits PCC, relative L2 | token flips |
|---|---|---|---|---|---|---|
| bfloat16, HiFi2 (default) | 33.7 | 0.0052 | 0.0556 | 0.9992 | 0.9978, 0.050 | 0 of 16 |
| bfloat8_b, HiFi2 | 34.1 | 0.0084 | 0.0694 | 0.9987 | 0.9985, 0.045 | 0 of 16 |
| bfloat8_b, LoFi | 40.1 | 0.0424 | 0.0853 | 0.9981 | 0.9966, 0.062 | 0 of 16 |
| bfloat16, LoFi | 33.7 | 0.0458 | 0.1011 | 0.9973 | 0.9948, 0.077 | 0 of 16 |

bfloat8_b weights alone add nothing at HiFi2; LoFi on top of them reaches 40.1 tok/s (64 percent of the bfloat16 floor, and a higher share of the floor for 8-bit weights,
which is 9.4 ms per token for these weights) at a visibly larger error (layer 30 at 4 percent against 0.5 percent). LoFi with bfloat16 weights costs the error and gives no speed.
The default stays bfloat16 with HiFi2. The 8-bit LoFi variant is a choice for the operator and has not been checked on text quality beyond the 16 tokens.

Noise-floor tool (`python -m orchard.noise_floor`): it refuses the CPU-only floor as unusable for the chip ("the good run fails the bars it was meant to pass";
the damage control on the last layer is not detected). Its floor holds two CPU runs and no chip arithmetic, which the tool states. The reading used instead is
first-divergence against the float32 reference with the CPU floor printed beside every number, plus the known-bad control (the run with the default `rms_norm`, which the
probe located at layer 14 with a 5 percent stated tolerance), plus teacher-forced token and logits agreement.

## Update 6 (about 22:25 Pacific): both chips of the board, text quality and perplexity

Correction to update 5. The benchmark records of the 8-bit MLP variants carry the bfloat16 floor (62.8 tok/s) because the record is written with one fixed policy. For bfloat8_b MLP weights
the floor is 96.8 tok/s (5.3 GB per token), so 40.1 tok/s is 41 percent of it. The bfloat16 result stays at 54 percent. The record of a variant is a tok/s measurement; read its share from this log.

Two chips (`chip_model_tp.py`, `run_chip.py --tp 2`): a 1 by 2 mesh of the board's chips with fabric 1D. The residual stream, the fused 2560-to-1088 input projection and the latent KV cache
(one head) are replicated. Query heads are split 20 and 20 (`qb`, the per-head matmuls and the flash MLA op); `o_proj` is row-parallel; gate and up are column-parallel and down is
row-parallel, with the intermediate width padded from 3,200 to 3,328 per chip so each of the 8 DRAM banks holds whole tiles (the DRAM-sharded matmuls run on 8 cores and reach 430 GB/s on these shapes);
the LM head is split over the vocabulary. Two all-reduces per layer, 19 to 20 microseconds each for one row (`exp_tp.py`, correct to 0.25 percent). It ran correctly on the first attempt:
16 of 16 teacher-forced tokens, layer 61 error 0.058 (single chip 0.056), decode logits PCC 0.9985 (record `bench_tp2_bf16.json` in the evidence folder):
decode 37.96, 38.05, 37.98 tok/s after a warm-up run (single chip: 33.65), prefill 40 tokens in 1.05 s. That is 13 percent faster for twice the chips, and 30 percent of the two-die floor
(125.7 tok/s). The attention side is a fixed cost per layer that two chips do not shrink (about 240 microseconds of small operations per layer against about 160 for the MLP), so the gain is small.
The two-chip path shares the host with other work; the load average during the runs was 15.

Text quality (`qual_check.py`, free-running greedy decode of 64 tokens, single chip opt 3 against the CPU float32 reference, six prompts: seasons, a Python function, a train word problem, a
French translation, a two-sentence summary, a market at dawn): four of six texts are identical token for token over 64 tokens. The other two leave the reference at token 19 (seasons) and
token 26 (Fibonacci) and stay on topic and fluent (both texts are in `qual_chip_opt3.json` and `qual_reference_float32.json`). Teacher-forced perplexity on the first 767 tokens of the
WikiText-2 test split (raw), same window for all runs (`ppl_check.py`; the chip runs the window one token per step through the decode path and reads the logits after each step):

| run | perplexity | mean absolute difference of per-token log probability against the float32 reference | largest difference |
|---|---|---|---|
| CPU float32 reference | 8.057 | | |
| chip, one chip, bfloat16 (opt 3) | 8.007 | 0.051 | 1.25 |
| chip, one chip, bfloat8_b MLP weights, LoFi | 8.047 | 0.078 | 1.81 |
| chip, two chips, bfloat16 | 8.090 | 0.047 | 0.77 |

All three chip runs are within 0.6 percent of the reference perplexity (one window, one text; not a benchmark of model quality).

## Update 7 (about 22:50 Pacific): decode speed against context length, a chat server, the orchard dry run

Decode speed against position (`ctx_bench.py`: one traced loop over 4,096 positions of a real text, the time of a 32-step window at each listed position, bfloat16 weights, opt 3, KV cache
of 288 values per token):

| window starts at position | one chip tok/s | two chips tok/s |
|---|---|---|
| 32 | 33.8 | 37.5 |
| 512 | 32.4 | 37.1 |
| 1,024 | 31.9 | 37.3 |
| 2,048 | 29.9 | 35.4 |
| 4,000 | 30.8 | 36.1 |

The flash MLA op reads every cached row up to the position, and the cost stays small because the cache is 288 values per token for one head: about 9 percent slower at 4,000 tokens on one
chip, 4 percent on two. (Single windows, host load about 8; the two windows at 2,048 and 4,000 differ from the trend by the noise of the shared machine.) Agreement at a 922-token prompt is in update 4.
The chip model sizes the cache to the longest context asked for (a multiple of 128: the flash op needs that); 4,096 was run.

A chat server (`serve.py`, standard library only, 127.0.0.1, port 8210 by default, refuses the ports that belong to other work). `GET /v1/models`, `POST /v1/chat/completions` with
`messages`, `max_tokens` and `stream` (server-sent events); greedy decoding only; one request at a time. Smoke test on one chip: 99 tokens (19 in the prompt, 80 generated) answered in
2.96 s end to end; the streamed reply arrived as pieces; a malformed body got a 400. The first version failed on the tokenizer's return type (`apply_chat_template` returns a batch
object in transformers 5 and no longer a tensor); the failure was returned as an empty reply, so the handler now turns any exception into a logged 500. The server holds the chips while it runs, so it was
stopped (by its pid) and the lease released after the test.

The orchard preflight on this model (`bringup openbmb/MiniCPM3-4B --dry-run`, read only): hub ok (Apache-2.0, 8.2 GB, ships code that the harness would fetch and not run outside the agent sandbox),
disk ok, `base` warn (the model card names no base model, so stage 0 has nothing to compare it with), `port` BLOCK only because another agent's coder holds port 8011. Nothing in the preflight
or in `orchard.hazards` knows about the two things the port had to handle (the flash MLA grid limit and the norm accumulation). `orchard.hazards.scan_model_hazards` reports one hazard for
this config: the vocabulary 73,448 is not a whole number of tiles (it matters for the LM head and the sampler; the port pads to 73,472 and slices before the argmax).

## Update 8 (about 23:15 Pacific): batched prefill

The prompt used to go through the decode step one token at a time (30 ms per token). `ChipMiniCPM3.prefill(ids)` now processes the whole prompt at once with the existing ops on many rows: the same fused
projection and per-head matmuls, `ttnn.fill_cache` to write the 288-wide latent rows, and `ttnn.transformer.flash_mla_prefill` (the DeepSeek prefill op, causal, one key head, value width 256) for the
attention. The prompt is padded to a multiple of 128 tokens (the pad rows sit after the real ones, so the causal attention ignores them, and decode overwrites them before it ever reads them). The weights
that are stored DRAM-sharded for decode (`o_proj`, gate, up, down) get ordinary interleaved copies for prefill, loaded on the first call (about 6 GB more on the chip).

Checked against the recorded float32 reference (`prefill_check.py`): the last prompt token's logits PCC 0.9991 (40-token prompt) and 0.9969 (922-token prompt), the same as the per-token prefill
(0.9991 and 0.9968); the next-token choice matches; 16 teacher-forced decode steps from the filled cache have 0 flips and logits PCC 0.998 to 0.999. Speed:

| prompt tokens | prefill by decode steps | batched prefill (second and later calls) |
|---|---|---|
| 40 | 1.19 s | 0.088 s (460 tok/s) |
| 922 | about 28 s | 0.37 s (2,490 tok/s) |

The first call compiles the kernels (13 to 18 s); the benchmark records treat it as the warm-up run. Benchmark records `bench_opt3_bf16_prefill.json` (40-token prompt) and
`bench_opt3_bf16_prefill_long.json` (922-token prompt): decode 33.85, 33.76, 33.83 tok/s and 32.26, 32.31, 32.27 tok/s (one chip, opt 3, bfloat16, 1 warm-up and 3 measured runs of 64 tokens,
host load about 7.7), which is 54 and 51 percent of the 62.8 tok/s floor. The chat server uses the batched prefill on one chip. The two-chip model still prefills one token per step.

## Update 9 (about 23:20 Pacific): the stage-b host-attention run, a final benchmark set, and hazard row

Stage b (the rest of the port on the chip with the attention core on the host): `run_chip.py agree --attn host --opt 1` runs the projections, norms, the output projection and the MLP on the chip and
the multi-latent attention core in torch on the host (query, latent and rope part read back from the chip every layer, attention output sent back). Result: 16 of 16 teacher-forced tokens, layer 61 error 0.0548,
last-position prefill logits PCC 0.9993, decode step 0 logits PCC 0.9989 and relative L2 0.036. The chip attention (opt 3) has layer 61 error 0.0556 and decode step 0 relative L2 0.050. So the chip's
attention composition adds a little error over the host attention (about 0.036 to 0.050 in the logits), and the rest of the port carries almost all of the 5.5 percent block error. The host mode was run
after the chip attention worked (the method asks for it first; it was not needed to find the flash op's limit).

Final benchmark records (`docs/run-logs/evidence/2026-10-29-minicpm3/bench_*.json`, written by `run_chip.py bench`; 1 warm-up run and 3 measured runs of 64 tokens after a 40-token prompt;
label "benchmark"; the share uses the floor of the weights each run reads, 512 GB/s per chip; host load average 6 to 10 during the runs):

| config | chips | decode tok/s min, median, max | share of floor | prefill, 40 tokens |
|---|---|---|---|---|
| bfloat16, HiFi2 (default) | 1 | 33.61, 33.66, 33.81 | 53.6 percent of 62.8 | 0.088 s batched |
| bfloat16, same, 922-token prompt | 1 | 32.26, 32.27, 32.31 | 51.3 percent | 0.37 s batched |
| MLP bfloat8_b, HiFi2 | 1 | 34.22, 34.23, 34.29 | 35.4 percent of 96.8 | 0.088 s |
| MLP bfloat8_b, LoFi | 1 | 39.84, 39.85, 39.89 | 41.2 percent of 96.8 | 0.088 s |
| bfloat16, HiFi2 | 2 | 39.12, 39.24, 39.31 | 31.2 percent of 125.7 | 1.02 s (one token per step) |
| MLP bfloat8_b, LoFi | 2 | 43.93, 44.07, 44.51 | 22.8 percent of 193.6 | 0.90 s (one token per step) |

The records written before the floor policy was fixed in `run_chip.py` (the first 8-bit runs) were replaced by these. Quality numbers for the 8-bit LoFi variant are in updates 5 and 6
(perplexity 8.047 against 8.057, layer 30 error 4 percent, logits PCC 0.9966 to 0.9981); the two-chip 8-bit variant was not run through the agreement and perplexity checks.

Harness change from this port: `orchard.hazards` has a row `mla_flash_decode_needs_core_grid` for multi-latent attention (the flash MLA decode op needs an explicit core grid at one latent head and
batch 1; issue 213). It is detected from `kv_lora_rank`, needs no nearest bundle, and has a test plus a mutation check (removing the detection turns four tests red).
