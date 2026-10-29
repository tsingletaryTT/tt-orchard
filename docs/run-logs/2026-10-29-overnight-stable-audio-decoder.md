# Overnight lead 3 (2026-10-28 evening to 2026-10-29): the Stable Audio 3 autoencoder decoder on a chip

Class given: a non-text pipeline that tt-orchard could not bring up before tonight and that needs new code. Operator asleep until about 05:40 Pacific; no
questions asked, defaults recorded here. Hard stop and final report at 05:20 Pacific.

## Choice (written at about 20:30 Pacific)

Choice: the transformer autoencoder decoder of `stabilityai/stable-audio-3-small-music` (issue 161, design step 2), on one board, with the rest of the pipeline
as PR 165 left it (T5Gemma encoder, sampler and the final waveform unpatching on the host; the diffusion transformer on the chip).

Why this and not a new model:
- Other candidates read: tt-metal already carries Whisper, Qwen3-ASR, Qwen3-TTS, SigLIP, BERT-family and Gemma3 code (so they are "known and not wired", issue 62,
  a different class); SDXL and Z-Image have full pipelines. Kokoro-style TTS is convolution and vocoder heavy with no ttnn pieces to start from.
- The decoder is a real component with a clear missing piece. Its 6 transformer layers use differential attention (two softmax attentions that share values,
  their outputs subtracted) and dynamic-tanh normalisation (`gamma * tanh(alpha * x) + beta`). The design notes say tt-metal v0.80.0 has neither.
- Everything needed to check it exists: the library's own `SAMEDecoder` is the CPU reference, the 56-array reference layout and the first-divergence probe
  from PR 178 exist, and the chip runtime already runs the diffusion transformer, so the clip is end to end and a person can listen to it.
- The benchmark of PR 165 says the host work (sampler plus decoder) is most of the 1.6 s fine-tune clip, so the decoder on the chip is the next speed step.
- Bounded: 6 layers, width 768, 12 heads of 64, attention inside windows of 34 tokens, so no long-sequence attention is needed.

Honest limit of the choice: it adds a component to a model that already runs, not a new model. Every Stable Audio 3 checkpoint shares the architecture, so it
applies to the base control and to other sizes. A different model would have needed a new venv, new reference code and a new runtime in 9 hours with one
shared board.

Licence obligations: the model is under the Stability AI community licence (field `other`, `stable-audio-community`) and redistributes the Gemma terms for the
text encoder; the library is MIT. Nothing is published. Weights stay in the local cache; the run directories hold derived arrays and one clip and are not
committed.

## Decoder structure (read from the library at the pinned commit)

latent [B, 256, T] -> bottleneck decode (host) -> Linear 256 to 768 -> per latent frame, one input token plus 16 learned "new tokens" (noise 0.01 added) -> chunks
of 2 latent frames = 34 tokens, layers 0 to 2 -> shift by 17 tokens with replicate padding, layers 3 to 5 -> keep the last 16 tokens of each 17 -> Conv1d 768 to 512
(kernel 3) -> unpatch (256 samples per patch, 2 channels). Layer: `x = x + attn(DyT(x)); x = x + ff(DyT(x))`, attention has qk DyT per head, partial rotary
(32 of 64 dims, positions 0 to 33 inside the chunk), two softmax attentions with separate q and k and shared v, difference then `to_out`; the feed forward is a
SiLU gated linear unit with inner width 2304 and biases.

## Log

- 20:30 choice made, structure read. Next: host decomposition and a test against the library block, then chip.
