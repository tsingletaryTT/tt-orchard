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

## Stage 0 on the real pair (2026-10-28, after the base finished downloading)

The base's files were complete: no `.incomplete` blob, and the sizes equal the public file list (`model.safetensors` 2,270,384,940 bytes, `svd_bases.pt`
1,270,378,123, the text encoder's `model.safetensors` 1,183,022,944, `model_config.json` 8,452). The stage 0 template ran on the fine-tune (revision
`8bdfb53c6b0f`) against the base (revision `4249e9179cd6`), headers and configs only, plus a few KB of sampled values:

- Class `full-port`, path `full-port`, facts `form: full`, `modality: audio`. Two reasons, both in `model_config.json`:
  - `sample_size` is 5,292,032 in the fine-tune and 5,324,800 in the base (120.0 s and 120.75 s at 44,100 Hz).
  - `model.diffusion.diffusion_objective` is `rf_denoiser` in the fine-tune and `rectified_flow` in the base.
- Everything else matches: both have 685 tensors with the same names, shapes and dtypes (F32), the same three components (diffusion transformer 459,130,896
  parameters, autoencoder 108,244,721, conditioner 198,144), and equal text-encoder directories (config and tensor headers).
- The weights are related and not copies. Of 8 sampled tensors, 2 are identical (autoencoder parts), 5 are near (cosine 0.99994 to 1.0, diffusion
  transformer) and 1 is changed (the conditioner's padding embedding, cosine 0.96). That fits a fine-tune that trains the diffusion transformer and leaves
  the autoencoder alone.
- The `training` section differs by an `arc` block (adversarial post-training with a discriminator); the template lists it and ignores it. The base ships
  `svd_bases.pt` (1.27 GB) and the fine-tune does not. What it holds is still not known.
- The library's code (`stable_audio_3/inference/sampling.py`) shows what the two config differences mean at run time: the sampler defaults to `pingpong`
  for `rf_denoiser` and to `euler` for `rectified_flow`, and `generate` uses its own `sample_size` argument (5,292,032). The weights have the same layout, so one
  runtime serves both. The harness's rule (any difference in the `model` section or a top-level key is a full port) files the pair as a full port. Whether that
  rule is too strict for a sampler-parameterisation difference is the operator's decision; the harness does not relax it on its own. For this model it changes
  nothing yet, because the run stops after stage 0 either way (`audio-path-unbuilt`).

## Built since: the CPU reference tooling (step 0 of the runtime design)

`orchard/audio_reference.py` (a recorder and a checker), `orchard/skills/audio-reference-templates/make_reference.py` (runs the library's own `generate` on
the CPU and saves every intermediate) and `tt-orchard audio-reference-env` (the environment). 109, 24 and 42 tests; mutation tables of 89, 47 and 59 mutations
with no survivor at the end. The library checkout at the pinned commit `77e0398c26f7` exists (a read-only `--dry-run` of the environment command confirms it).
The environment itself (CPU torch and the library installed in `~/orchard-venvs/stable-audio`) was not built: the permission system refused the command that
installs external code. No real reference clip has been made and no timing is measured.

Estimate, not measured: the diffusion transformer does about 0.22 TFLOP per pass at 237 tokens (a 10 s clip), so a pass should take seconds on 16 CPU cores
and the fine-tune's 8 steps a minute or so; the base control at 50 steps with CFG scale 7 is about 100 passes. Loading the 2.3 GB checkpoint and decoding 16 s of
audio are the other costs. The pair should fit well inside 30 minutes. The first run measures it.

## Does either repository ship code that the library would run? (checked 2026-10-28, read only)

- Neither snapshot holds a Python file, a pickle or a custom loader. The file lists are: `model.safetensors`, `model_config.json`, the text-encoder directory (`config.json`,
  `generation_config.json`, tokenizer files, `model.safetensors`), licences, the card and a picture. The base also has `svd_bases.pt` (1.27 GB).
- No config names a custom class: `model_config.json` and the text encoder's `config.json` contain no `auto_map` or `custom_pipelines` key, and the library does not
  call `from_pretrained` with `trust_remote_code` anywhere (a search of the library outside its training code).
- The library reads the weights with `safetensors.load_file`, the text encoder through `transformers`, and the config as JSON. It builds the model classes from its own
  code, selected by the `type` strings in the config (`dit`, `taae_v2`, `t5gemma`, `number`), so a config can only choose among classes the library already has.
- `svd_bases.pt` is read by the library only when a LoRA adapter is applied (`models/lora/loader.py`, with `torch.load(..., weights_only=True)`); `generate` never reads it,
  and `make_reference.py` loads no adapter. The same module has one `torch.load(..., weights_only=False)` for a LoRA checkpoint path, which is not reached either.
- Conclusion: running the reference executes the pinned library's code and `transformers`' T5Gemma code, and no code from either model repository.

## The first real CPU reference runs (2026-10-28)

The environment was built with `tt-orchard audio-reference-env` (the command passed every step: source clean at `77e0398c26f7`, CPU torch 2.7.1, the library imported from the
checked tree) and `make_reference.py` ran twice on the 16-core host CPU, float32, Hub offline, no token, `--expect-src` set. Same prompt ("A warm acoustic guitar melody with soft
percussion, 100 BPM"), duration 10 s, seed 42. Runs and clips are under `<HOME>/orchard-runs/stable-audio-ref/`.

| run | settings | wall time (whole process) | `generate` and decode | arrays |
|---|---|---|---|---|
| `small-music` (fine-tune) | 8 steps, CFG 1.0, `pingpong` | 31 s | 24.8 s | 56 |
| `small-music-base` (control) | 50 steps, CFG 7.0, `euler` | 64 s | 57.1 s | 132 |
| `small-music-repeat` (fine-tune again) | as the first | about 45 s | 36.9 s | 56 |

- The two process wall times include loading the 2.3 GB checkpoint and the text encoder; the library's own timer covers sampling and decoding. The pair needed about 95 s, far below the
  45 minute limit, so nothing was cut. The encoder took 0.2 s.
- Reproducibility: the fine-tune run repeated gives a bit-identical result: `python3 -m orchard.audio_reference check` reports PCC 1.0, relative L2 error 0 and slope 1.0 for all 56
  arrays (the noise draws, the layer outputs, every step's latent and denoised estimate, the final latent and the waveform). A CPU float32 run is deterministic here, so a chip
  run can be compared with this reference without a repeat-to-repeat noise term.
- The control passes: `python3 -m orchard.audio_reference control` finds the base's waveform different from the fine-tune's (relative L2 error 1.85, PCC 0.0005, log-spectrum
  distance 0.97). The two use different samplers and step counts, so they are different pieces of music for one seed; that is the control's intent (it catches serving the wrong
  weights), and it does not say anything about quality.
- Clip checks with the spec's sound bars (computed on the saved waveform): both clips are 10.00 s, 44.1 kHz stereo; 97 percent of 100 ms windows are above -40 dBFS in each; the
  fine-tune clips 0.0008 percent of samples at full scale and the base 0.14 percent (the spec's bar is at most 1 percent). Nobody has listened to them.
  - `<HOME>/orchard-runs/stable-audio-ref/small-music/clips/audio.flac` (fine-tune, 0.9 MB)
  - `<HOME>/orchard-runs/stable-audio-ref/small-music-base/clips/audio.flac` (base, 1.0 MB)
- Each run holds `manifest.json` (settings, seeds, model revision and config hash, library commit, per-array sha256; `python3 -m orchard.audio_reference verify` re-hashes them) and
  `arrays/*.npy`: 41 MB for the fine-tune and 73 MB for the base.
- Not done: a bfloat16 CPU run for the noise floor (the script has no dtype option yet), so the checker's bars are still provisional.
- A slip while running it: I ran a helper script from the scratch directory with the venv's python, and a file named `inspect.py` in that shared directory shadowed the standard library
  and broke the numpy import. Running it with `-I` fixed that. Scripts that read model data are run in isolated mode.

## The class rule for serving-only differences (branch audio-class-inference, 2026-10-28)

The operator approved relaxing the strict rule for differences in serving or sampler settings, with a way to let smarter inference decide later. Stage 0's audio branch now sorts each
difference in `model_config.json` into serving, structural or unknown by an allow-list table, and returns a decision object (class, path, the path the strict rule would choose, a
confidence, the evidence for each difference, and any override with its source). `"audio_strict": true` in `triage_config.json` keeps the old rule.

The real pair, re-classified on the same files as above (headers and configs only):
- class `weights-only`, path `weights-only`, alternative path `full-port`, confidence `medium`;
- two differences, both `serving`, both decided by the allow-list: `sample_size` (5,324,800 in the base, 5,292,032 in the fine-tune; the rule cites the library's `generate`) and
  `model.diffusion.diffusion_objective` (`rectified_flow` in the base, `rf_denoiser` in the fine-tune; the rule cites `models/dit.py` lines 452-456, 575-580 and 610-613, which treat
  the two values identically, and `inference/sampling.py` line 434, which uses the objective only to pick the default sampler);
- a hazard says the serving configuration must carry the model's own values for those two keys and that no stage checks it yet.
The audio-path-unbuilt stop still applies whatever the path, so nothing runs on the relaxed class: the run stops after stage 0.
