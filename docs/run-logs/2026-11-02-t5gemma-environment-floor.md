# T5Gemma prompt embeddings in three CPU environments (issue 162)

Date: 2026-11-02. CPU only; no device was opened.

## Question

Issue 162 measured a relative difference of about 1.7 percent between the T5Gemma prompt embeddings of two CPU environments that both call themselves
float32. Which setting moves the encoder output?

## Method

One script loads the encoder from the snapshot of `stabilityai/stable-audio-3-small-music` (subfolder `t5gemma-b-b-ul2`, offline, no token), the same way
the library's `T5GemmaConditioner` does (`T5GemmaEncoderModel.from_pretrained` with `is_encoder_decoder = False`), encodes one prompt padded to 256 tokens
with 10 real tokens, and saves the last hidden state. It runs in three interpreters that exist on this machine, once with the library's default loading
and once with `torch_dtype=float32`, and once more with the attention implementation forced to `eager` in the interpreter that defaults to `sdpa`. The
difference is the relative L2 norm over the real tokens (the padding rows are bit-identical and left out).

| environment | torch | transformers | default attention |
|---|---|---|---|
| A (the stable-audio venv, the reference of issue 162) | 2.7.1+cpu | 5.21.0 | eager |
| B (the shared Tenstorrent venv) | 2.7.0+cpu | 5.19.0 | sdpa |
| C (the venv used by the Qwen4 reference tests) | 2.14.1+cpu | 5.19.0 | sdpa |

## Result

The encoder loads in **bfloat16** in all three environments when the library's code runs unchanged: the checkpoint is stored in bfloat16 and
`from_pretrained` keeps the stored dtype. The reference run is labelled float32 (`settings.dtype`), and its text encoder is not.

| comparison (real tokens) | relative L2 |
|---|---|
| A against B, library default (bfloat16) | 0.0174 |
| A against C, library default | 0.0185 |
| B against C, library default | 0.0134 |
| A against B, float32 forced | 0.0044 |
| A against C, float32 forced | 0.0044 |
| B against C, float32 forced | 0.0000 |
| A (float32, eager) against B (float32, eager) | 0.0000 |
| bfloat16 default against float32, same environment | 0.0106 to 0.0143 |

Two settings explain the whole difference:

1. The encoder runs in bfloat16 by default. Two bfloat16 runs that differ only in how attention is computed round differently, and the result differs by
   1.3 to 1.9 percent. A bfloat16 encoder also sits 1.1 to 1.4 percent away from the float32 encoder of its own environment.
2. Transformers 5.19.0 picks `sdpa` for this encoder and 5.21.0 picks `eager`. With float32 forced, A and B differ by 0.44 percent for that reason alone.
   With float32 and `eager` in both, the two environments agree exactly (0.0000), so torch 2.7.0, 2.7.1 and 2.14.1 do not matter here.

transformers 5.21.0 rejects `attn_implementation="sdpa"` for `T5GemmaEncoderModel` ("does not support an attention implementation through
scaled_dot_product_attention yet"), so `eager` is the setting both versions accept.

## What follows

- The 1.7 percent environment floor of issue 162 is a floor for a run whose encoder is bfloat16 with the attention implementation the installed
  transformers chooses. Pinning the encoder to float32 and `eager` attention in the reference run and in the chip run's host encoder gives identical
  embeddings in these environments. That changes the recorded references (their `settings.dtype` says float32 and their encoder was not), so it needs the
  operator's decision and a new reference cache entry; it is not done here.
- Until it is done, the statement next to every chip-versus-reference comparison should say that the text encoder ran in bfloat16 on both sides.
- Whether the sampler's final latent floor (0.086) shrinks with an exact encoder has not been measured. It needs a sampler run in each environment with the
  float32 encoder on the CPU, which takes about a minute per run and no board.

The script is a few lines (the steps above); it is not kept in the repository because it only reads the model and the numbers above are all it produced.
