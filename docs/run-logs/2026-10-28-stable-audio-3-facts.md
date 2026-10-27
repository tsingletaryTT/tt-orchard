# Stable Audio 3 small music: what it is and what exists for it on Tenstorrent (2026-10-28)

Purpose: the facts that decide how the harness handles `stabilityai/stable-audio-3-small-music` and models like it. Plan:
`docs/superpowers/plans/2026-10-28-audio-bringup.md`. Nothing here used a chip, a Hugging Face token or a download of a gated file. Paths are
written with placeholders (`<HF_CACHE>`, `<TT_METAL>`).

## What was read

- The operator's cached copy of `stabilityai/stable-audio-3-small-music` at revision `8bdfb53c6b0f` (`<HF_CACHE>/hub/models--stabilityai--stable-audio-3-small-music`):
  `model_config.json`, the card, `LICENSE.md`, `LICENSE_GEMMA.md`, `NOTICE`, the header of `model.safetensors`, the config and header of
  `t5gemma-b-b-ul2/model.safetensors`.
- The public file lists (names and sizes, no content) of the model and of `stabilityai/stable-audio-3-small-music-base`, through the Hub's file-tree
  API without a token.
- The Hub's search listing for `stable-audio` (names, tags, download counts).
- `tt-model search stable-audio` and `tt-model search audio` (catalog), the installed container registry, and the tt-metal tree at v0.80.0
  (`<TT_METAL>/models/demos`, `models/tt_dit`), read only.
- `tt-orchard bringup stabilityai/stable-audio-3-small-music --dry-run`, which reads the hub row, the research report and the base row.

## What the model is

- A latent diffusion model for text to audio (pipeline tag `text-to-audio`, library `stable-audio-3`), model type `diffusion_cond_inpaint`, stereo, 44,100 Hz,
  `sample_size` 5,292,032 samples (120 s). Licence field `other` (name `stable-audio-community`); the text encoder is redistributed under the Gemma terms
  (`LICENSE_GEMMA.md`). The card lists two ways to run it: the `stable-audio-3` library and `stable-audio-tools`.
- Three components share one `model.safetensors` (685 tensors, 2.27 GB, all F32), with name prefixes:
  - `model.model.*`: the diffusion transformer, 438 tensors, 459,130,896 parameters. Config: depth 20, width 1,024, 16 heads, RMS norm, query-key RMS norm,
    64 memory tokens, adaLN global conditioning, cross attention to 768-wide text tokens, rectified-flow objective (`rf_denoiser`), latent width 256.
  - `pretransform.model.*`: the autoencoder, 244 tensors, 108,244,721 parameters. A transformer autoencoder (`taae_v2`: 6 transformer layers, stride 16,
    patches of 256 samples, differential attention, dynamic-tanh normalisation), downsampling ratio 4,096 (about 10.8 latent frames per second).
  - `conditioner.conditioners.*`: 3 tensors, 198,144 parameters (the seconds-total number embedder and the learned padding embedding).
- The text encoder is `t5gemma-b-b-ul2/`, its own directory with a `config.json` and a 1.18 GB `model.safetensors` (T5Gemma encoder-decoder, 592 M
  parameters; the encoder is 282 M). The config's `repo_id` points at the model's own repository and `subfolder` at that directory.
- The README's example uses 8 steps, CFG scale 1.0 and the `pingpong` sampler. That is the adversarially post-trained setting: the card says the models are trained
  with adversarial post-training to cut the step count.

## Fine-tune against base

- The card's `base_model` is `stabilityai/stable-audio-3-small-music-base`, with the Hub tag `base_model:finetune:`. The two repositories differ in these files
  (names and sizes only; I did not read the base's content):
  - `model.safetensors` has the same size in both (2,270,384,940 bytes), so the tensor layout very likely matches. Not verified: the base's header was not read.
  - `model_config.json`: 10,341 bytes in the model, 8,452 in the base. The model's `training` section holds an `arc` block (an adversarial post-training
    setup with a discriminator); the difference is probably that block. Not verified.
  - The base alone has `svd_bases.pt` (1.27 GB). What it holds is not stated in the files read.
  - The text-encoder directory has the same file sizes in both.
- The operator describes the base as having "2 shards". The base's file list has one `model.safetensors` and one `svd_bases.pt`; no sharded index exists.
- So the likely relation is a same-architecture, different-weights fine-tune (an adversarial post-training of the base). The harness decides this from
  the base's header and config once the base is on disk (see the result section).

## What exists on Tenstorrent

- No bundle, container package or catalog entry serves Stable Audio 3 (`tt-model search stable-audio` returned an empty list; `tt-model search audio` lists
  only `episod/audio8-asr-infinite-p150`, a speech recogniser). The installed container registry holds the MiniMax-Music3 package, TT-OpenVLA, TT-SkyReels and
  TT-Waypoint, and Qwen language-model packages.
- tt-metal v0.80.0: `models/demos/audio` holds `qwen3_asr`, `qwen3_tts` and `whisper`. `models/tt_dit` has pipelines for FLUX, FLUX 2, Ideogram, LTX, Mochi,
  Motif, Qwen-Image, Stable Diffusion 3.5 Large and Wan, plus audio building blocks used by the LTX video pipeline (audio VAE decoder, vocoder, snake and
  resampling layers). A search of both trees for `stable_audio` and `stable-audio` found nothing.
- The pieces a port could start from: `tt_dit` attention and transformer-block layers, RMS norm and linear layers, and the CCL parallel config. The model's
  autoencoder uses differential attention and dynamic-tanh normalisation, which no tt-metal file read here implements. I did not search for either.
- Consequence: nothing in the spec's stage 2 (boot a container, generate clips) can serve this model today. A Stable Audio 3 run can be classified and then
  stops until a runtime exists (plan slice C).

## What the harness did on the real files

- `tt-orchard bringup stabilityai/stable-audio-3-small-music --dry-run` (read only): the hub row blocks (`the model is gated or private`, `credentials-needed`;
  the operator has downloaded it, which is issue 119). The base row blocks with `nearest-model-missing`: no installed bundle serves the base, and
  `tt model search stable` found no bundle. Research read both repositories' weight headers without a token.
- The stage 0 template's audio branch (plan task A2) ran on the cached model as both the model and the nearest model, to show that it reads the real layout
  (this is a self comparison and says nothing about the base): 3 components found, 685 tensors, 3 prefixes with roles, sample rate 44,100,
  2 channels, and the copy hazard fired because the two sides are the same files.

## Not determined

- Whether the base's `model.safetensors` has the same tensor names and shapes (its header was not read), and what differs in its config. Both need the base
  on disk. The harness reads the base from the local cache only.
- What `svd_bases.pt` is used for.
- Whether serving the fine-tune needs anything from the base's `svd_bases.pt`. The README's inference example loads the fine-tune alone.
- How much of the diffusion transformer and autoencoder `tt_dit` layers can be reused.
