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
