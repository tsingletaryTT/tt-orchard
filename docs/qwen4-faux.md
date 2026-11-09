# Qwen4-shaped test models ("qwen-faux")

Status: written 2026-11-10 from the code on `main` and the `transformers` 5.19.0 source. Everything here ran on the CPU. No chip was
used. Sections 1 and 5 are written to stand alone, so they can also serve as a model card.

## 1. What this is

These are test checkpoints. They are shaped after public readings of an unreleased model that this page calls "Qwen4". They are not that
model, and no Qwen4 weights or `config.json` exist. The parts that are new in the reading (hyper-connections, the n-gram embedding, the
sparse-attention indexer) are not trained in any of them. Their text is not readable, and no quality figure can come from them.

They exist so that anyone can try a port of the new layer types, on any hardware, before real weights are published. `transformers`
5.19.0 already has the class they load into: `Qwen4ExpForCausalLM` (file `models/qwen4_exp/modeling_qwen4_exp.py`).

| Kind | What it holds | Size | How it is made |
|---|---|---|---|
| **Tiny** | 4 layers, hidden size 64, random weights drawn with `torch.manual_seed(1234)` and stored in bf16. Three delta-net layers, one attention layer with an indexer, one per-layer embedding (n-gram) block, four hyper-connection streams. 590,084 parameters in float32. | 1.3 MB | `scripts/tiny_qwen4/make_checkpoint.py` |
| **Franken** | The first K layers of `Qwen/Qwen3.8-27B` (real weights, hidden size 5120) placed in the same layout, with hyper-connections initialised so that the model computes exactly what those K layers of Qwen3.8 compute. K is a multiple of 4. | K=4 float32 17.2 GB; K=20 bf16 22.5 GB (the embedding and head alone are 5 GB in bf16) | `scripts/tiny_qwen4/make_franken.py` |

Equality with the base, from the records in `docs/run-logs/evidence/2026-11-08-qwen4-franken/` (three seeded prompts of 8, 16 and 33
tokens, every block of the indexer selected):

| Checkpoint | Largest logit difference to Qwen3.8's own class | Top-1 token |
|---|---|---|
| K=4, float32, identity hyper-connections | 0.0 | equal |
| K=20, bf16, identity hyper-connections | 0.25 | equal |
| K=8, bf16, hyper-connections drawn from N(0, 0.02^2) (`--hc-sigma 0.02`) | 5.2 | differs, as intended |

What a franken is good for: a port of the hyper-connections and the gated delta-net can be checked against a model whose correct output
is known from outside. What it cannot do: say anything about how trained Qwen4 weights behave. A K=20 greedy decode of 32 tokens has 14
distinct ids; K=8, 12 and 16 decode to one or two ids, so K=20 is the smallest useful one.

Donor repositories and licenses (the Hub cards at the time of writing): `Qwen/Qwen3.8-27B` is Apache-2.0, and
`higa2515/Qwen3.8-27B-QSA-BF16` (the optional source of the indexer tensors) is Apache-2.0. A franken built with `--indexers` names both
revisions in its manifest and its card. A franken built without it draws the indexer projections at random.

## 2. The layer, step by step

The code is in `transformers/models/qwen4_exp/modeling_qwen4_exp.py` (5.19.0). Class and function names below are from that file.

```
token ids
   |  embed_tokens                                  (vocab, hidden)
   |  repeat 4 times along the feature axis         -> hc_count * hidden   [Qwen4ExpTextModel.forward]
   v
 +-- decoder layer i  [Qwen4ExpTextDecoderLayer.forward] --------------------------------------------+
 |  x  (hc_count * hidden)                                                                           |
 |  (if layer i has a PLE block)  x = x + ple(x, token ids)                                          |
 |  attn_hyper_connection: x -> (block_in, x, inject)      [Qwen4ExpTextGatedResidual]               |
 |  block_out = GatedDeltaNet(block_in)   or   Attention(block_in) with the QSA indexer              |
 |  x = x + inject * block_out   (one weight per stream)                                             |
 |  mlp_hyper_connection:  x -> (block_in, x, inject)                                                |
 |  block_out = MLP(block_in)                                                                        |
 |  x = x + inject * block_out                                                                       |
 +---------------------------------------------------------------------------------------------------+
   v
 hyper_connection_mixer: x -> mean over the 4 streams (hidden)     [GatedResidual, use_combine=False]
   |  lm_head                                       (hidden -> vocab)
   v
logits
```

There is no final norm. `Qwen4ExpTextModel` has none, and the mixer takes its place.

### 2.1 Hyper-connections (`Qwen4ExpTextGatedResidual.forward`)

The state is `hc_count` residual streams side by side. With `n = hc_count`, `x` the state and `rmsnorm_g` an RMS norm applied in groups of
`hidden_size` with scale `(1 + weight)`:

```
x_n     = rmsnorm_g(x)                                      # hc_norm
mix     = sigmoid(up(silu(down(x_n) / n)))                  # input_mix_weight_down: n*hidden -> hc_lowrank, up: back
block_in = mean over streams of ( mix * x_n )               # shape (hidden)
inject  = 2 * sigmoid(block_inject_weight(x_n) / n)         # shape (n), one gate per stream
x_new   = x + block_out[..., None, :] * inject[..., :, None]   # flattened back to n*hidden
```

The residual added to is the un-normed `x`. The final mixer is the same function without `inject`.

**Why zeros make it an identity.** With `down`, `up` and `block_inject_weight` all zero, `mix = sigmoid(0) = 0.5` and `inject =
2 * sigmoid(0) = 1`. The four streams start equal, so they stay equal. `block_in = 0.5 * rmsnorm_g(x) * (1 + hc_norm.weight)`. Setting
`hc_norm.weight = 1 + 2 * w`, with `w` the base model's norm weight, gives `rmsnorm(x) * (1 + w)`, which is what the base model feeds its
block (`scripts/tiny_qwen4/make_franken.py`, `convert_tensors` and the constant `IDENTITY_NOTE`). In float32 every step is a power-of-two
scaling or a mean of equal values, so the result is exact. In bf16 the stored `1 + 2 * w` is rounded.

### 2.2 Gated delta-net, the `linear_attention` layers (`Qwen4ExpTextGatedDeltaNet.forward`, `torch_recurrent_gated_delta_rule`)

```
qkv = silu(depthwise_conv1d(in_proj_qkv(x)))      # kernel linear_conv_kernel_dim, causal; split into q, k (key_dim each) and v (value_dim)
z, b, a = in_proj_z(x), in_proj_b(x), in_proj_a(x)
beta = sigmoid(b)                                 # per value head
g    = -exp(A_log) * softplus(a + dt_bias)        # per value head; the decay is exp(g)
q, k = l2norm(q), l2norm(k);  q = q / sqrt(head_k_dim)         # key heads are repeated to match the value heads
per token t, state S of shape (head_k_dim, head_v_dim):
    S      = S * exp(g_t)
    delta  = (v_t - sum_k(S * k_t)) * beta_t                   # what the state predicts for k_t versus what v_t is
    S      = S + k_t (outer) delta
    out_t  = sum_k(S * q_t)
y = out_proj( rmsnorm(out) * weight * act(z) )     # Qwen4ExpTextRMSNormGated; act = output_gate_type, or hidden_act when unset
```

Prefill uses a chunked version of the same recurrence (`torch_chunk_gated_delta_rule`, chunk size 64).

**The output gate.** `act` is `config.output_gate_type`. When it is unset the code uses `hidden_act` (checked: an unset value gives `silu`
for these configs). The tiny checkpoint writes `sigmoid`, following the public reading of the model's config. The franken writes `silu`,
because the Qwen3.8 base uses silu and the identity needs the same gate. A franken with `sigmoid` differs from the base by about 0.2 on a
small model (`docs/superpowers/specs/2026-11-08-qwen4-franken-design.md`, section 3). The same gate is the only difference from Qwen3.5's
delta-net: a CPU comparison gave a largest difference of 0 with `silu` and 14.4 with `sigmoid` (`docs/run-logs/2026-11-02-qwen4-new-ops.md`;
the single-token decode path was not compared).

### 2.3 Attention with a sparse indexer, the `indexed_attention` layers (`Qwen4ExpTextAttention`, `Qwen4ExpTextQSAIndexer`)

The config lists these layers as `full_attention`; `Qwen4ExpTextConfig.__post_init__` renames them to `indexed_attention`. Attention:

```
q_and_gate = q_proj(x)  -> (query, gate)           # q_proj is twice as wide as the heads
q = rmsnorm_head(query);  k = rmsnorm_head(k_proj(x));  v = v_proj(x)
q, k = rotary(q, k)                                # partial rotary: rope_parameters.partial_rotary_factor (0.25 in both kinds here)
scores = softmax(q k^T / sqrt(head_dim) + mask), mask = causal AND indexer_selection
out = o_proj( (scores v) * sigmoid(gate) )
```

The indexer (`Qwen4ExpTextQSAIndexer.forward`) decides, per query, which keys the mask keeps:

```
qk       = index_qk_proj(x)  ->  indexer_n_heads query heads and one key head, each indexer_head_dim wide
q        = rotary(rmsnorm(q))
blocks   = the visible tokens grouped into blocks of indexer_compress_ratio tokens (complete blocks only)
key_blk  = rotary( rmsnorm( mean of the block's raw keys ) )          # at the block's first position
score    = sum over heads of relu( q . key_blk ) / sqrt(indexer_head_dim)
keep     = the top (indexer_budget // indexer_compress_ratio) blocks, plus every token of the unfinished last block
```

When the number of complete blocks is at most the budget divided by the ratio, every block is kept and the layer is ordinary causal
attention, so the indexer weights cannot change the output. For the tiny model the budget is 16 tokens and the ratio 4, so the selection starts to matter for a query that sees 20 or more
tokens. For the franken the budget is 2048, so it starts at 2052 visible tokens.

### 2.4 The feed-forward block (`Qwen4ExpTextMLP`)

`down_proj( act(gate_proj(x)) * up_proj(x) )` with `act = hidden_act` (silu here). Every checkpoint on this page has this dense block in
every layer. See section 6 for why that needs a shim.

### 2.5 Per-layer embedding, n-gram hash (`Qwen4ExpTextPLELayer`, `Qwen4ExpTextNGramEmbedding`)

Only the tiny model has one (`ple_layer_ids = [2]`, which is one-indexed, so layer index 1). A franken has none (`ple_layer_ids = []`).

```
ids of the last ngram_size tokens, shifted right per position, restarting after an eos token
for each n-gram order n = 2 .. ngram_size, for each of heads_per_ngram heads:
    mixed = (id_0 * m_0) XOR (id_1 * m_1) ... XOR (id_{n-1} * m_{n-1})        # m = layer_multipliers, int64
    row   = (mixed mod prime_h) + offset_h                                     # prime_h: a distinct prime near ngram_vocab_size_base per head (tiny: 1009 to 1021)
emb   = ngram_embedding(rows), the heads concatenated   -> ple_embed_dim
key   = rmsnorm_g(key_proj(emb))  (one key per stream);  value = value_proj(emb)
gate  = sum over features of ( key * rmsnorm_g(x) ) / sqrt(hidden);  gate = sign(gate) * sqrt(max(|gate|, 1e-6))
gv    = sigmoid(gate) * value                      # per stream
out   = gv + silu( dilated depthwise conv1d( rmsnorm_g(gv) ) )          # kernel ple_conv_kernel_size, dilation ngram_size
x     = x + out
```

The multipliers come from `_build_layer_multipliers` (a SplitMix64 hash of `config.seed` and the layer index). In the tiny checkpoint they are three
int64 values between 7.3e15 and 1.6e16, stored as a buffer (`layer_multipliers`). A library that stores integers in 32 bits cannot hold them, so a port has to keep the hash
on the host or use 64-bit integers.

## 3. Configuration keys

Values are from the tiny `config.json` and from a K=20 franken `config.json` made with the current generator. A dash means the key is not
written and the class default applies (`Qwen4ExpTextConfig`).

| Key | Meaning | Tiny | Franken K=20 |
|---|---|---|---|
| `model_type`, `architectures` | The class that loads it | `qwen4_exp_text`, `Qwen4ExpForCausalLM` | same |
| `hidden_size` | Width of one stream | 64 | 5120 |
| `num_hidden_layers` | Layers | 4 | 20 |
| `layer_types` | `linear_attention` (delta-net) or `full_attention` (indexed attention); repeats 3 then 1 | 4 entries | 20 entries |
| `full_attention_interval` | Used only when `layer_types` is missing | 4 | 4 |
| `num_attention_heads`, `num_key_value_heads`, `head_dim` | Attention shape | 8, 2, 32 | 24, 4, 256 |
| `linear_num_key_heads`, `linear_num_value_heads` | Delta-net heads | 2, 6 | 16, 48 |
| `linear_key_head_dim`, `linear_value_head_dim` | Delta-net head width | 32, 32 | 128, 128 |
| `linear_conv_kernel_dim` | Delta-net causal conv kernel | 4 | 4 |
| `intermediate_size` | Dense feed-forward width | 256 | 17408 (the base's own) |
| `hc_count` | Number of residual streams | 4 | 4 |
| `hc_lowrank` | Rank of the hyper-connection mixer | 8 | 640 |
| `output_gate_type` | Gate activation of the delta-net output norm | `sigmoid` | `silu` |
| `partial_rotary_factor` | Share of each head that is rotated | 0.25 | 0.25 (inside `rope_parameters`) |
| `rope_parameters` | RoPE type and base | — (defaults: theta 10000) | theta 10,000,000, interleaved multi-axis sections 11, 11, 10 |
| `indexer_n_heads`, `indexer_kv_heads`, `indexer_head_dim` | Indexer query heads, key heads (must be 1), width | 4, 1, 32 | 4, 1, 128 |
| `indexer_budget`, `indexer_compress_ratio` | Tokens kept per query, tokens per block | 16, 4 | 2048, 4 |
| `ple_layer_ids`, `ple_embed_dim`, `ple_conv_kernel_size` | Per-layer embedding layers (one-indexed), width, conv kernel | [2], 64, 4 | [], —, — |
| `ngram_size`, `heads_per_ngram`, `ngram_vocab_size_base`, `split_ngram_parts`, `seed` | The n-gram table | 3, 2, 1000, 4, 1234 | not used |
| `vocab_size` | Vocabulary | 384 | 248320 (the base's) |
| `max_position_embeddings` | Position limit | 512 | 262144 (the base's; the proofs use at most 33 tokens) |
| `tie_word_embeddings` | `lm_head` tied to the embedding | false | false |

`num_experts` is absent in both. The class default is 512 and its config check refuses a value of 0 or less, which is the reason for the
shim in section 6.

## 4. Where the franken's tensors come from

From `scripts/tiny_qwen4/make_franken.py` (`convert_tensors`, `read_indexers`). `N` is the layer number, `w` a base norm weight.

| Qwen3.8-27B tensor | Franken tensor | Value |
|---|---|---|
| `model.language_model.embed_tokens.weight` | `model.embed_tokens.weight` | copied, cast to the dtype |
| `lm_head.weight` | `lm_head.weight` | copied |
| `model.language_model.norm.weight` | `model.hyper_connection_mixer.hc_norm.weight` | `(1 + 2w)` repeated for the 4 streams; the mixer's `down` and `up` weights are zeros |
| `...layers.N.input_layernorm.weight` | `model.layers.N.attn_hyper_connection.hc_norm.weight` | `(1 + 2w)` repeated 4 times; `input_mix_weight_down`, `input_mix_weight_up` and `block_inject_weight` are zeros (or drawn from N(0, sigma^2) with `--hc-sigma`) |
| `...layers.N.post_attention_layernorm.weight` | `model.layers.N.mlp_hyper_connection.hc_norm.weight` | same construction |
| `...layers.N.linear_attn.*`, `...self_attn.*`, `...mlp.*` | `model.layers.N.linear_attn.*`, `self_attn.*`, `mlp.*` | copied under the new prefix |
| `higa2515` `...self_attn._qsa_indexer.q_proj.weight` and `k_proj.weight` | `model.layers.N.self_attn.indexer.index_qk_proj.weight` | the two joined (query rows first), with `--indexers` |
| `higa2515` `...q_norm`, `...k_block_norm` | `...indexer.q_layernorm.weight`, `...indexer.k_layernorm.weight` | copied, with `--indexers` |
| (no source without `--indexers`) | `...indexer.index_qk_proj.weight`, the two norms | random N(0, 0.02^2), norms zero |
| the release's `k_norm` | not carried | the class does not apply one |

The visual tower and any multi-token-prediction head are left out. `Qwen4ExpForCausalLM` ignores `mtp.*` and `model.visual.*` keys when
loading (`_keys_to_ignore_on_load_unexpected`).

The tiny checkpoint's tensor names follow the same class. For layer 0: `linear_attn.{in_proj_qkv, in_proj_z, in_proj_b, in_proj_a,
conv1d, A_log, dt_bias, norm, out_proj}`, `mlp.{gate_proj, up_proj, down_proj}`, and the two `*_hyper_connection` blocks. Layer 1 adds
`ple.{ple_embedding.*, key_proj, value_proj, norm_key, norm_query, norm_conv, conv1d}`. Layer 3 has `self_attn.{q_proj, k_proj, v_proj,
o_proj, q_norm, k_norm, indexer.*}` in place of `linear_attn`.

## 5. Provisional or verified

"Verified" names the check. "Assumed" means a public reading with no published configuration behind it, and says what would change it.

| Claim | Status | Basis |
|---|---|---|
| The equations in section 2 | Verified | Read from `transformers` 5.19.0 (`modeling_qwen4_exp.py`) on 2026-11-10. |
| Identity hyper-connections reproduce the base: exact in float32 | Verified | `proofs-k4-f32.json`: largest logit difference 0.0, top-1 equal at 155 teacher-forced positions. |
| Same in bf16 within rounding | Verified | `proofs-k20-bf16.json`: 0.25, top-1 equal; K=8, 12, 16: 0.125. |
| A delta-net with `silu` equals Qwen3.5's; with `sigmoid` it does not | Verified, prefill only | `docs/run-logs/2026-11-02-qwen4-new-ops.md`: 0 and 14.4. Decode (one token with a cache) was not compared. |
| The indexer keeps every block when the prompt fits the budget | Verified | Code in `Qwen4ExpTextQSAIndexer.forward`; a budget of 8 tokens moves the logits by 1.11 (K=8), 2.17 (K=20), 0.88 (K=4) against all-selected (franken design note, section 5a). |
| 3 delta-net layers then 1 attention layer | Verified for Qwen3.8-27B; assumed for Qwen4 | `check_layers` in `make_franken.py` refuses a base that breaks the schedule. A different Qwen4 schedule would change `layer_types`. |
| `hc_count` 4, `hc_lowrank` 640 at full size | Assumed | Public readings (`docs/run-logs/2026-11-02-dense-qwen4-prep.md`). 320 in the only public `qwen4_exp` config is a per-rank shard in the reading, so no code may default to it. |
| Hidden size 5120, 24 and 4 attention heads of 256, delta-net 16 and 48 heads of 128 | Assumed for Qwen4 | The Qwen3.8-27B skeleton, which the readings match. |
| Dense feed-forward width | Unknown | The prep note infers about 14336 (the tiny `--width 27b` preset uses it). The franken uses Qwen3.8's own 17408. The two do not agree and no source settles it. |
| Output gate `sigmoid` for Qwen4 | Assumed | The public Flash-Next config sets it. The tiny checkpoint follows it; a franken uses `silu` on purpose (section 2.2). |
| Indexer tensors from `higa2515` map onto `index_qk_proj`, `q_layernorm`, `k_layernorm` | Assumed | Shapes fit and `q_norm` and `k_block_norm` are zero-centred like the class's norms; nothing has run a long prompt through them. The release also applies a `k_norm` to raw keys before pooling and the class does not. |
| Per-layer embedding widths, n-gram table size | Unknown | No public reading. The public Flash-Next config implies a 102 GB table, so the tiny one is only a few kilobytes by design. |
| A multi-token-prediction head | Unknown | `transformers` has none. These checkpoints carry none. |
| Behaviour past 33 tokens or 262,144 positions | Unknown | The proofs use prompts of 8, 16 and 33 tokens. |
| Text quality | None | Random or truncated weights; the text is not readable. |

**One disagreement between a note and the code.** The franken design note (section 3) says the output-gate class default is `sigmoid`. In
5.19.0 an unset `output_gate_type` falls back to `hidden_act` (`silu` here), and Flash-Next's config sets `sigmoid` explicitly. The
franken still writes `silu` by name, so its behaviour is the same.

## 6. Quick start (CPU)

Needs an interpreter with `torch` and `transformers` 5.19.0. Nothing is downloaded and no device is opened.

Build the tiny checkpoint into a scratch directory (3.5 seconds, 1.3 MB; the run also decodes a 32-token reference for three prompts):

```
python3 scripts/tiny_qwen4/make_checkpoint.py --out /tmp/qwen-faux-tiny --variant dense
```

Load it and run it:

```
python3 examples/qwen4_faux_quickstart.py /tmp/qwen-faux-tiny
```

Output of that command on 2026-11-10:

```
layers: ['linear_attention', 'linear_attention', 'linear_attention', 'indexed_attention']
hidden_size 64, hc_count 4, parameters 590,084
input ids [3, 10, 17, 24, 31, 38, 45, 52]
logits shape (1, 8, 384), next-token id 298
greedy continuation of 8: [298, 226, 330, 196, 88, 139, 307, 205]
```

**Why the example goes through a shim.** `transformers` builds a mixture-of-experts block in every `qwen4_exp` decoder layer. Loading the
tiny checkpoint directly with `Qwen4ExpForCausalLM.from_pretrained(path)` does not fail. It builds `Qwen4ExpTextSparseMoeBlock` layers,
reports the checkpoint's dense `gate_proj`, `up_proj` and `down_proj` weights as UNEXPECTED, reports the experts as MISSING, and
initialises them at random, so the model runs and gives wrong answers. The shim
(`orchard/skills/reference-gate-templates/qwen4_dense.py`: `prepare_config`, then the `dense_layers()` context manager) swaps each dense
layer's block for the model's own `Qwen4ExpTextMLP` while the layers are built. The example script shows the three lines that matter
(`load_model` in `examples/qwen4_faux_quickstart.py`).

A franken is built the same way and loaded through the same function. The commands and options are in the docstring of
`scripts/tiny_qwen4/make_franken.py`; K=20 in bf16 needs about 23 GB of disk and runs a CPU proof, which takes a few minutes.

## 7. Reading order, and what this page cannot tell you

1. This page.
2. `docs/superpowers/specs/2026-11-02-tiny-qwen4-rehearsal-design.md`: what each dimension of the tiny model stands for and what each layer exercises.
3. `docs/superpowers/specs/2026-11-08-qwen4-franken-design.md`: how the identity is built, the indexer mapping, and the proof results with their records.
4. `docs/run-logs/2026-11-02-qwen4-new-ops.md`: the four new ops and the delta-net gate comparison.
5. `docs/run-logs/2026-11-02-dense-qwen4-prep.md`: the public sources, and what the dense reading assumes.
6. In `modeling_qwen4_exp.py`: `Qwen4ExpTextDecoderLayer`, `Qwen4ExpTextGatedResidual`, `Qwen4ExpTextGatedDeltaNet`,
   `Qwen4ExpTextAttention` with `Qwen4ExpTextQSAIndexer`, `Qwen4ExpTextPLELayer`, in that order.

Limits of this page:

- It describes the `transformers` 5.19.0 class and two generators. A later `transformers` version may change the class; check section 2
  against it before relying on it.
- It says nothing about a real Qwen4 release. The "provisional" values in section 5 are public readings and may all be wrong.
- It describes what runs on the CPU. Nothing here has shown how any of it behaves on a chip, and nothing here is a serving recipe.
- It does not cover the vision tower, a multi-token-prediction head, or the mixture-of-experts variant of the tiny model (`--variant moe`,
  `--variant mixed` in `make_checkpoint.py`).
