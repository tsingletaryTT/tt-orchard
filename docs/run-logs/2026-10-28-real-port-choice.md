# Real port: which model needs a kernel, and why Granite-4.0-H-1B

Date: 2026-10-28. Branch `real-port-granite-mamba2`. Written before any hardware use.

Operator instruction: pick a real open-weights model that tt-orchard cannot run today because tt-metal or ttnn
lacks an op or mixer it needs, bring it up with the harness, let the local coder write the missing kernel through
the kernel loop, and report how far it gets. The outside helper (telegraph) writes no kernel code and is not asked
about kernel work. A stop with a precise reason is a valid result.

## Choice

**`ibm-granite/granite-4.0-h-1b`** (revision `d18cca4c`), with `ibm-granite/granite-4.0-h-350m` as the small
development twin (same architecture and ops, 0.68 GB).

| fact | value | source |
|---|---|---|
| licence | Apache-2.0, not gated | Hub API (2026-10-28, with the account's token; the token is not printed) |
| size | 2.92 GB of safetensors (bf16); about 1.5B parameters | Hub API |
| class | `GraniteMoeHybridForCausalLM`, `model_type granitemoehybrid` | `config.json` |
| layers | 40: 36 `mamba` (Mamba-2 mixer) and 4 `attention` (layers 5, 15, 25, 35) | `layer_types` |
| mixer | hidden 1536; 48 SSM heads of dimension 64; state size 128; 1 group; conv kernel 4; expand 2 | `config.json` |
| attention | 12 heads, 4 KV heads, head dimension 128, no position embedding (`nope`), softmax scale 0.0078125 | `config.json` |
| MLP | dense shared MLP with a SiLU gate (`shared_intermediate_size` 4096); 0 experts | `config.json` |
| scaling | muP multipliers: embedding x12, residual x0.22, logits /6 | `config.json` |
| fit | 3 GB of weights fit one chip; the SSM state is 48 x 64 x 128 values per mamba layer, 36 layers | arithmetic |
| CPU reference | `transformers` 5.19.0 `GraniteMoeHybridForCausalLM` is in the reference venv; `mamba_ssm` and `causal_conv1d` are absent, so its mixer takes the pure torch path (`mamba2_chunk_scan` for prefill, `mamba2_selective_state_update` for one-token decode) | installed source, to be confirmed by running it (step a) |
| not brought up here | no run, issue, package or catalog note names Granite or Mamba-2 | `gh issue list --search`, `grep` |

## What is missing, and where

`tt-orchard inventory` on the model (6 of 40 layers traced, sequence 8, against the 0.80.0 ttnn name dump):
34 distinct ops are `mapped`, 1 `composable`, 3 `verified` on the simulator, 1 `candidate`, 1 `unmapped`
(`aten.einsum`), 3 `ignored`. The op list alone does not show the gap: a Mamba-2 layer written in plain torch
decomposes into matmuls, `cumsum`, `tril`, `exp`, `softplus` and `conv1d`, and each has a ttnn name. The
inventory cannot say that no ttnn op does the Mamba-2 state update.

* ttnn 0.80.0 has `experimental.prefix_scan` and the `ssm` helpers (Mamba-1, one decay per channel), gated delta
  rule ops (`transformer.chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`, used by the Qwen3 family)
  and a Kimi delta attention directory (`kda`). Nothing is named for Mamba-2's scalar decay per head with grouped
  B and C.
* Decode, per mamba layer and token: `state[h,p,n] = state[h,p,n] * exp(dt[h] * A[h]) + dt[h] * x[h,p] * B[n]`,
  then `y[h,p] = sum_n C[n] * state[h,p,n] + D[h] * x[h,p]`. That is an elementwise update and a row reduction
  on a 3072 x 128 state, in 36 layers, for every token.
* Prefill: the chunked scan (`mamba2_chunk_scan`), which carries the state across chunks of 256.
* Also new to the stack: the depthwise causal convolution (kernel 4, 3328 channels) with its rolling state, and
  the gated RMSNorm. `conv1d` and `rms_norm` exist; the gating and the state handling do not.

A version composed from stock ttnn ops is possible: matmul for the readout, broadcast multiplies for the update,
the prefix-scan op with the decay expanded to the full state width. That version is the baseline the kernel has
to beat, and step (b) of the run keeps the recurrence on the host first. The kernel is the missing piece because
no single ttnn op does the Mamba-2 state update, and the run measures whether the local coder can write it.

The kernel target is split into two probe ops, so each failure points at one skill:
`ssd_state_update` (elementwise: `out = decay * s + dtx * b`) and `ssd_readout`
(`out[r, :] = sum over the row of c * s`, a row reduction; `rmsnorm`, the other reduction, passed 0 of 16).

## Expected difficulty

* Model port: moderate. Attention layers are plain GQA without RoPE, the MLP is a normal SiLU MLP, the muP
  scalars are constants. No tt-transformers path serves a Mamba hybrid, so the decoder is new ttnn code and
  stages 3 and 5 of the full-port route are driven by hand here.
* Kernel: hard for the local coder. Evidence from issues 90 and 96: `silu_mul` passed, `rmsnorm` 0 of 16,
  `linear_scan` 0 of 8. The readout is a reduction. A stop with the failure classes counted is a likely result.

## Rejected, and why

| candidate | facts | reason |
|---|---|---|
| `state-spaces/mamba2-370m`, `mamba2-780m` | Apache-2.0, pure Mamba-2 | same op, base models with no instruction tuning; Granite-H puts the op in a model people use. Fallback |
| `ibm-granite/granite-4.0-h-tiny` | Apache-2.0, 13.9 GB, MoE | adds expert routing and is 5 times larger |
| `ibm-granite/granite-4.0-h-350m` | Apache-2.0, 0.68 GB | same architecture; used for development speed |
| `LiquidAI/LFM2-1.2B`, `LFM2-350M` | licence `other` (LFM Open License) | the short convolution exists in tt-metal (`lfm25_vl/tt/short_conv.py`), an LFM2.5-24B package is in the catalog, there is no scan, and the licence needs reading |
| Mamba-1 (`state-spaces/mamba-*`, `falcon-mamba-7b`) | | tt-metal already has a Mamba-1 demo and `experimental.prefix_scan` |
| `fla-hub/rwkv7-1.5B-world`, `RWKV/RWKV7-Goose-World3-1.5B-HF` | Apache-2.0, 3 GB, `RWKV7ForCausalLM` | transformers 5.19.0 ships `rwkv` (RWKV-4) only; RWKV-7 needs remote code from `fla`, which the op inventory refuses to import, so there is no CPU reference without running vendor code |
| `moonshotai/Kimi-Linear-48B-A3B-Instruct` | MIT, 98 GB | far too large; ttnn already has a `kda` directory |
| Qwen3-Coder-Next, Qwen3.8 (delta rule) | | served by bundles; the delta rule ops exist |
| `tiiuae/Falcon-H1-0.5B-Instruct`, `1.5B` | licence `other` (Falcon LLM) | Mamba-2 hybrid too, but the licence needs reading and it adds parallel attention and SSM heads with several muP multipliers |
| `Zyphra/Zamba2-1.2B-instruct` | Apache-2.0 | Mamba-2 plus a shared attention block with LoRA adapters: more new structure than the op under test |
| `microsoft/Phi-4-mini-flash-reasoning` | MIT, 7.7 GB | differential attention and sliding windows: several new pieces |

## Plan and stop rules

(a) CPU reference for the whole model and recorded op-level arrays for the probe ops, the convolution and the
gated norm. (b) Port on a chip with the recurrence on the host; per-layer agreement against the CPU reference.
(c) Kernel loop with the local coder, simulator first, then hardware; attempts, failure classes and passes recorded.
(d) Swap the kernel in, repeat the agreement check, then a first benchmark against the decode floor.
Stop reasons are written down as they happen. Nothing is published. Telegraph is not used for kernel work.
