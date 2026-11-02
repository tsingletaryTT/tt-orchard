# Qwen4-style new ops: what they are and what the trace shows (2026-11-02)

Source: `Qwen4Exp` in transformers 5.19.0 (the reference interpreter), the Flash-Next config fixture, the ttnn 0.80.0 name dump. Everything here ran on the CPU or the meta device. No chip was opened.

## The four ops

| Op | What the reference code does | ttnn 0.80.0 names present |
|---|---|---|
| Gated delta net (GDN) | The class matches Qwen3.5's gated delta net except for the output gate. Qwen3.5 uses silu. This model reads `output_gate_type` (sigmoid in the Flash-Next config). | `transformer.chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`, `gated_delta_attn_seq` |
| Gated residual | Four residual streams. A grouped RMSNorm, a down projection to 320 with silu, an up projection with sigmoid, a gate multiply and a mean over streams, and a 4-wide injection gate `2*sigmoid(x/4)`. | linear, rms_norm, silu, sigmoid, mean |
| Sparse attention indexer (QSA) | Mean-pools 4-token key blocks, scores queries against them (relu, sum over heads, scale), takes the top 512 blocks and always keeps the unfinished tail. | `experimental.indexer_score_dsa`, `topk_large_indices`, `transformer.sparse_sdpa`. Missing: `experimental.deepseek_prefill.update_padded_kv_cache` |
| N-gram embedding | Hashes token ids with an integer multiply, xor and remainder, shifts them with an EOS-aware `cummax`, looks rows up in a 102 GB table, gates, then a dilated depthwise convolution (kernel 4, dilation 3) with silu. | `bitwise_xor`, `remainder`. Missing: `cummax`. Upstream keeps the lookup on the host. |

Two facts that shape a first attempt:

- The indexer budget is 2048 tokens and a block is 4 tokens, so it selects at most 512 blocks. With 2048 or fewer visible tokens every block is selected and the layer is ordinary dense causal attention. A first bring-up can leave the sparse path out and add it for longer contexts.
- The hash is integer arithmetic on a few ids per token. Only the table is large.

## Check 1: the GDN differs from Qwen3.5's only in the output gate

Script run on the CPU with small random weights (hidden 64, 2 key heads, 4 value heads, head dimension 16, 37 positions, prefill path, no cache):

| Comparison | Largest absolute difference |
|---|---|
| Qwen4-style with gate `silu` against Qwen3.5 | 0 |
| Qwen4-style with gate `sigmoid` against Qwen3.5 | 14.4 |
| Qwen4-style with gate `sigmoid` against Qwen3.5 with its gate switched to sigmoid | 0 |

The state dictionaries load into each other with `strict=True`. The decode path (one token, with a cache) was not compared.

## Check 2: a static stand-in for the indexer, so the inventory can trace past it

`tt-orchard inventory` stopped at the first QSA layer: the reference indexer loops over query positions and calls `torch.nonzero`, which `torch.export` cannot trace. The new option `--stub-data-dependent` replaces the indexer's forward with a static version: a reshape and mean for block pooling, a matmul, relu, a sum over heads, a topk over blocks, and a scatter and `index_select` for the mask. It ignores which positions each query can see, so the ops are a guide and the graph is not the real graph. The output lists each replaced module under `stubbed`.

- On a tiny config the stubbed indexer picks the same blocks as the real one for the last query, which sees every token. Three mutations of the stub (reversed topk, no relu, max-pool instead of mean) each turned that test red.
- Full-size Flash-Next config, 4 layers (3 GDN, 1 QSA), 64 positions, meta device:

| Trace | mapped | composable | verified | candidate | unmapped | ignored |
|---|---:|---:|---:|---:|---:|---:|
| 2 layers (no QSA layer) | 36 | 1 | 10 | 25 | 7 | 2 |
| 4 layers with the stub | 39 | 1 | 10 | 27 | 7 | 3 |

The QSA layer adds `relu`, `clamp` and `scaled_dot_product_attention` (mapped), `index_select` and `scatter` (candidate), and `contiguous` (ignored). The unmapped list did not grow: `aten.cummax`, `aten.diff`, `aten.eye`, `aten.histc`, `aten.index_put_`, `aten.unflatten.int` and `transformers.grouped_mm_fallback`., the same seven as the 2-layer trace.

## Not settled

- Whether `ttnn.conv1d` accepts dilation. The fallback is four shifted multiply-adds.
- The signatures of the GDN ops in the wheel against tt-metal main. The upstream port was validated on main.
- The two missing names (the QSA cache write and the BF4 expert packer module).
- The decode-time GDN comparison.
