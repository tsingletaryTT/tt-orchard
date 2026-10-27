# A Tenstorrent runtime for Stable Audio 3 small: the smallest path to one clip (design)

Date: 2026-10-28. Status: written for review; nothing is built. Plan slice C of `docs/superpowers/plans/2026-10-28-audio-bringup.md`.
Facts it rests on: `docs/run-logs/2026-10-28-stable-audio-3-facts.md`. Operator rule: models that need new code are welcome and are attempted.

## 1. Goal

One audio clip from `stabilityai/stable-audio-3-small-music` made on a Blackhole chip, checked against a CPU reference, as a measured result. Everything
else (a container package, the bring-up stages, the base control) follows from a runtime that exists. This design says what the smallest such runtime is
and in what order to build it. It changes no code.

## 2. What the model needs at run time

Per clip the pipeline is: encode the prompt with T5Gemma, embed the duration, draw noise of shape `[1, 256, T]` where `T = seconds * 44100 / 4096`
(about 10.8 frames per second, 1,292 frames for the 120 s maximum), run the diffusion transformer for 8 steps (ping-pong sampler, CFG scale 1.0, so
one transformer pass per step and no unconditional pass), then decode the latent with the autoencoder's decoder to stereo audio.

| part | size | what it is (from `model_config.json` and the tensor headers) | runs per clip |
|---|---|---|---|
| T5Gemma encoder | 282 M parameters (12 layers, width 768, 12 heads, head dim 64, MLP 2,048) | bidirectional text encoder, 256 tokens, learned padding | once |
| duration and timestep embedders | under 1 M | Fourier features (`expo`) then small MLPs | once and per step |
| diffusion transformer | 459 M parameters, 20 layers | below | 8 times |
| autoencoder decoder | about half of the 108 M autoencoder parameters | transformer autoencoder, stride 16, patches of 256 samples | once |

Diffusion transformer layer (layer 0 names): `pre_norm` (RMS with gain), `self_attn` (`to_qkv` 3072 by 1024, `to_out`, RMS query and key norm of width 64,
rotary on part of the head because `inv_freq` has 16 entries), `cross_attend_norm` and `cross_attn` (`to_q`, `to_kv` 2048 by 1024, query and key norms), `ff_norm` and a
gated MLP (`ff.0.proj` 8192 by 1024 split into two halves of 4096, `ff.2` 1024 by 4096, with biases), a `to_scale_shift_gate` vector of 6,144 (six adaLN
parameters per layer), and a per-layer `to_local_embed` (257 to 1,024 to 1,024) that adds the inpainting conditioning (a mask and the masked input).
Around the layers: `project_in` and `project_out` (256 to 1,024 and back), `preprocess_conv` and `postprocess_conv` (1 by 1 convolutions), a global-conditioning
embedder producing 6,144 values, and 64 learned memory tokens prepended to the sequence. For text to music the inpaint inputs are zeros, but the
`to_local_embed` bias terms still contribute, so the path has to run.

Sizes are small. The transformer has width 1,024 and a few thousand tokens at most; in bfloat16 its weights are 0.9 GB. One chip is enough, with no tensor
parallelism.

## 3. Host first, chip second

The smallest path moves components to the chip one at a time and keeps each earlier step as the reference for the next.

| step | where each part runs | exit criterion |
|---|---|---|
| 0 | everything on the host CPU with the model's own library (`stable-audio-3` or `stable-audio-tools`) | one clip written; every intermediate saved (prompt embeddings, the 8 noise draws, the latent after each step, the decoded waveform) |
| 1 | T5Gemma encoder and autoencoder decoder on the host; diffusion transformer on the chip | latent after step 1 and after step 8 agree with step 0 (section 5); the clip passes the sound checks |
| 2 | autoencoder decoder on the chip | the waveform agrees with step 0's decode of the same latent |
| 3 | T5Gemma encoder on the chip | prompt embeddings agree with step 0's |
| 4 | the loop packaged as a server with the job API | a container package (section 6); then the audio stages of the spec apply |

Why this order: the transformer is the only part that runs 8 times, so it is the first thing worth moving, and it is the part the harness cannot
otherwise check. The encoder runs once on 256 tokens and the decoder once; on the host both are cheap in tokens but their host time has not been
measured. If the host decode of a 10 s clip turns out to dominate the clip's time, step 2 moves ahead of step 1's tuning. That is a measurement for step 0 to
make (it should record the wall time of each part).

Step 0 has its own risks. The pipeline's code is written for GPUs, the sampler details (the ping-pong update, the length-dependent noise schedule named by
`use_effective_length_for_schedule` and `distribution_shift_options`) live in library code that has to be read, and the library may not install on this
machine without a package the gated repository's terms cover. Step 0 resolves each of those before any chip work starts, and it needs the base model on disk
only for the base control.

## 4. What can be reused

From tt-metal v0.80.0 (`models/tt_dit`, read, not run):
- `blocks/attention.py` and `blocks/transformer_block.py`: attention with self and cross variants and a transformer block that takes adaLN modulation. The
  model's six-parameter adaLN vector per layer matches the pattern those blocks use for the other DiTs, but I did not check parameter order or whether query and
  key RMS norm over head dimension 64 and the partial rotary are supported. Each is a question for the first read of the code.
- `layers/normalization.py` (`RMSNorm`), `layers/linear.py`, `layers/feedforward.py`, `layers/embeddings.py`: the norm, linear, gated MLP and embedding
  layers. The model's norm is RMS with a gain named `gamma`; the autoencoder uses dynamic-tanh normalisation with `alpha`, `beta`, `gamma` per norm, which has no
  layer here (a small elementwise op: `gamma * tanh(alpha * x) + beta`).
- `parallel/config.py` and `parallel/manager.py`: needed only to set up a single-chip configuration.
- `encoders/gemma` and `encoders/t5`: encoder implementations used by LTX, Flux and SD3.5. T5Gemma's encoder is a Gemma-style stack with bidirectional attention;
  whether `encoders/gemma` covers its attention (logit soft-capping 50 in the decoder config, GELU tanh activation, per-layer details) is unread. Step 3 is the
  last step for that reason.
- `pipelines/pipeline_api.py`, `pipelines/cfg.py` and the tests under `tests/models/*`: the shape of a pipeline and of a model-level PCC test.
- The audio layers used by LTX (`layers/audio_*`, `models/audio_vae/*`): a different vocoder and autoencoder family. The tt-dit audio VAE code shows how
  1-D convolution layers are written for this stack; it does not implement this model's transformer autoencoder.

From the MiniMax-Music3 container (hand-run on 2026-10-15; its image has server kind `tt-dit-server`):
- The job API shape (`/v1/music/jobs`, queued jobs, saved audio, per-stage timings) is the shape the spec's stage 2 template expects. A Stable Audio server can
  copy that API. The server's own code lives in the image and was not read on this pass; step 4 reads it first.
- The cache-load hang: `ttnn.as_tensor` with a cache file over about 32 MiB hung on this host. The largest Stable Audio tensor in bfloat16 is the gated MLP
  (8,192 by 1,024, 16 MiB), under that size. The hang may not occur; the probe from the spec (a 64 MiB load under a timeout) tells, and the harness
  already plans to run it and pick the workaround itself.
- The probe, the workaround selection and the per-run container label and sweep rules of the audio spec apply unchanged.

Nothing found covers: the partial rotary with 16 inverse frequencies, the memory tokens, the per-layer local-conditioning MLP, the ping-pong sampler, or the
transformer autoencoder (patching, chunking by 32, differential attention, dynamic-tanh norm).

## 5. The numerical check against the CPU reference

The reference is step 0's saved intermediates for a fixed prompt, duration and seed, with the CPU in float32. The sampler adds fresh noise at each ping-pong
step, so the check feeds the chip path the same noise tensors step 0 drew; with different noise the two latents cannot be compared.

1. A single transformer pass first. Inputs: the step 0 latent, timestep, prompt embeddings and zero inpaint inputs. Compare the chip's output with the CPU's.
2. Per-layer outputs for layer 0, 10 and 19 by hooks, to find where a drift starts. The comparison takes three numbers, as the kernel check does after
   its PCC-only mistake (a scaled output passes PCC): PCC, relative L2 error, and the slope of the best-fit line of chip against CPU. A single number cannot
   catch a scaled output.
3. The latent after 8 steps against step 0's latent, same three numbers, with the same noise draws.
4. The decoded waveform against step 0's decode of the same latent: waveform PCC and a log-mel spectrogram L1 distance. Raw waveform PCC is fragile to small phase
   differences, so the bar goes on the mel distance.
5. The sound checks from the audio spec (duration, level, clipping fraction) on the chip's clip.

The bars are choices until measured. The first measurement is the CPU's own noise floor: run step 0 in bfloat16 and in float32 and take the bfloat16 error as
the best the chip can be expected to do. The bars are then set a margin above that floor, the way the sidecar check's bars were. Each bar is recorded with its
source in `orchard/defaults.py` when it lands.

Controls, as in the prompt-suite design: the base model (`stable-audio-3-small-music-base`) run through the same chip path must pass the same checks against its
own CPU reference, and the fine-tune's clip must differ from the base's for the same seed and prompt. The second check catches serving the wrong weights. Note
the base model needs more steps and a higher CFG scale than the fine-tune (the fine-tune is adversarially post-trained for 8 steps at scale 1.0), so
the base control uses the base's own settings; the base's card names them and step 0 reads them from it.

## 6. Where this meets the harness

- The runtime is delivered as a container package of kind `tt-dit-server` (or whatever step 4 finds that kind requires). With one installed, `serving` for
  Stable Audio 3 models becomes `container`, the A3 stop no longer applies to them, and the audio spec's stage 2 template is the next piece to build.
- `serving: none` models keep stopping after stage 0, so the harness never claims a bring-up it cannot run.
- A package needs the image build the audio spec lists as out of scope (a v5.1 package takes 1.5 to 2.5 hours to build). Until then the runtime runs from the
  tt-metal tree on the host under a lease, which is enough for the numerical result and one clip.

## 7. Open questions

- Whether `stable-audio-3` or `stable-audio-tools` installs in a private virtual environment here, and which is the reference the card calls its own.
- The exact ping-pong sampler and the length-dependent schedule; the library code has to be read.
- Whether tt-dit's attention supports the partial rotary and the query and key norm without a change.
- Host time of the T5Gemma encoder and of the autoencoder decode for a 10 s clip.
- Whether the base's `svd_bases.pt` is needed to serve the fine-tune (the README's example loads the fine-tune alone, so probably not).
- The size of the transformer autoencoder's decoder on the chip (differential attention and chunking by 32 are new code).
- How the clip's quality is judged: a person listens. No metric here says the music is good.
