# Overnight 2: MiniCPM3-4B (multi-latent attention, dense MLP) on one board

Started 2026-10-28 20:15 Pacific by overnight lead 2. Operator's target: three models that could not be brought up the day before.
My class: an open-weights text model whose blocker is the attention side. This log is updated hourly.

## Choice (first hour)

Model: `openbmb/MiniCPM3-4B`. Licence Apache-2.0, not gated, 62 layers, hidden 2560, 40 heads, about 4B parameters (8 GB in bfloat16).

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
loader, not reusable as is). Its structure (q_a, q_b, kv_a, kv_b, a separate rope part) is the same as MiniCPM3's and guides the port; the port
here uses a plain ttnn composition with the key and value built per head from the latent.

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
embedding (rotate-half as a 32 by 32 matmul), the cache holds 288 values per token for the one shared head, the existing flash MLA op attends over it, and the
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
