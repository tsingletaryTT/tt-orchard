# Music and audio generation models to try next: survey and ranking (2026-10-29)

Question from the operator: "are there more music models for us to try?" This is a desk survey. No chip, lease, download of weights or install was used. Facts come from
Hugging Face model cards, `config.json` files and API listings, GitHub repository metadata, one web search, and an internal search whose results are summarised without
links (the internal pages are not public). Every number below is quoted from a source named next to it. Estimates are marked "estimate" and are my own.

What we have already: MiniMax-Music3 (container package from the tt-model-manager catalog, run on one p150, about 3.2 s of compute per second of audio; see
`2026-10-music-blackhole-first.md`) and Stable Audio 3 small-music, small-sfx and medium (diffusion transformer and decoder on the chip, 10 s clips in about 0.8 to 3.8 s;
see `2026-10-29-overnight-stable-audio-decoder.md`).

## Result

Recommendation: try ACE-Step 1.5 first. It has the most permissive licence of the candidates (MIT, with the card stating that generated music may be used commercially),
another team has already put its diffusion transformer and audio decoder on a Blackhole chip, and it contains a Qwen3 language model that the text-model route can serve.
The first day is a run of the existing community bundle through our audio stages. The new work after that is the language-model planner on the chip.

| Rank | Model | Licence | Size | Shape | Estimate to first chip clip (days) | First blocker I expect |
|---|---|---|---|---|---|---|
| 1 | ACE-Step 1.5 (turbo; 5 Hz LM 0.6B/1.7B/4B; XL 4B DiT) | MIT | DiT about 2.4B (4.79 GB bf16 file), LM 0.6B to 4B, VAE 337 MB file | Qwen3 LM planner, then a diffusion transformer, then an Oobleck VAE decoder; 48 kHz stereo, 10 s to 10 min | 1 (existing bundle), 7 to 9 for LM and XL | The community bundle skips the planner LM, and our audio stage 2 expects a runtime we built |
| 2 | HeartMuLa-oss-3B with HeartCodec | Apache 2.0 | LM 3.94B, codec 1.66B (fp32 file counts) | Llama-style LM over 8 codebooks at 12.5 Hz, then a codec decoder to 48 kHz stereo | 8 to 12 | HeartCodec decoder precision: the README says bf16 degrades quality, so it stays fp32 |
| 3 | DiffRhythm 1.2 (bundle exists) and DiffRhythm 2 | Apache 2.0 | 1.2: 2.2 GB checkpoint; 2: 1.14B | Flow-matching transformer on mel-style latents, then a decoder (Oobleck for 1.2, a BigVGAN-style snake vocoder for 2) | 1 for 1.2, 6 to 9 for 2 | DiffRhythm 2 vocoder: strided transposed convolutions with snake activation |
| 4 | YuE2-3B | CC BY-NC 4.0 (weights) | 3.63B plus a 133M VAE | One autoregressive-then-flow-matching transformer writes score and tokens, then latents, then a VAE decodes 48 kHz stereo | 10 to 14 | Custom model code ships as a wheel and has not been read; non-commercial licence needs the gate |
| 5 | MOSS-SoundEffect-v2.0 (sound effects) | Apache 2.0 | DiT 1.3B, Qwen3-1.7B text encoder | WanAudioModel diffusion transformer, flow matching, DAC VAE; 48 kHz, up to 30 s | 5 to 8 | DAC VAE decoder, and 100 solver steps with guidance 4.0 |

All five fit one p150 by weight size on paper. The largest, YuE2, reports a peak of 11 GiB on an RTX 4090 with bf16 weights and an fp32 VAE (YuE2-3B card), against
about 30 GB of device DRAM on a Blackhole chip (the ACE-Step bundle card reports 29.9 GB).

## What already exists on Tenstorrent hardware

This changes the ranking, so it comes first.

- ACE-Step 1.5 turbo has a community container bundle, `sunandbean/acestep-1.5-turbo-p100a` (tt-dit-server, schema 5.1, MIT, updated 2026-10-16). Its card says the diffusion
  transformer and the Oobleck VAE decoder run on the card and the text and lyric encoders, sampler and post-processing stay on the CPU. It reports a 30 s song in 3.25 to
  3.62 s from the published container, against 3.53 to 3.56 s on an RTX 5070 Ti, and an 8-minute song in 19 s end to end in the two-process layout. Its card also says the
  planning LM was not loaded in those runs. Source: https://huggingface.co/sunandbean/acestep-1.5-turbo-p100a and the port repository https://github.com/SunandBean/tt-acestep (Apache 2.0).
- An earlier, larger TTNN port of ACE-Step 1.5 sits in a closed tt-metal pull request (label stale): https://github.com/tenstorrent/tt-metal/pull/46312. Its description lists a TTNN
  5 Hz LM bridge, caption encoder, DiT denoise loop with trace, and an Oobleck VAE with conv and snake blocks, with PCC tests, for the Blackhole QuietBox mesh. I did not read the code
  or run it. Open tt-metal issues 53929, 53932 and 53933 track productizing it. Internal tracking shows another team working on it, currently paused; I do not repeat the details here.
- DiffRhythm 1.2 has a community bundle too, `sunandbean/diffrhythm-1.2-p100a` (Apache 2.0, updated 2026-10-16): a 96 s song in 10.0 s on one p100a, with the flow-matching
  transformer and Oobleck decoder on the card, durations 96 to 285 s. Source: https://huggingface.co/sunandbean/diffrhythm-1.2-p100a
- AudioX has a closed tt-metal bounty issue (https://github.com/tenstorrent/tt-metal/issues/32185). I did not find out whether it was completed.
- The tt-model-manager checkout here has no music entries in its tree. The catalog entries I found are Hugging Face bundles: the two above, our own
  `episod/stable-audio-3-small-music-blackhole`, and a plain copy of the ACE-Step weights (`ayb-bh92/Ace-Step1.5`). Branches read: `feat/card-viewer` (checked out), `docs/v6-canonical-thin-path`,
  `feat/cohost-kind`; none mentions an audio model.
- I found no internal effort on Sonexa, and no `tt-minimax-music-blackhole` repository, by search. The MiniMax-Music3 port is tt-metal pull requests 56043 and 56047 (closed, stale label),
  which match the container we ran.

## Candidate details

Reuse is judged against our code: the diffusion transformer chip runtime (`chip_dit.py`), differential attention, the chip decoder (`chip_decoder.py`), the resident runtime, the audio package builder
(`orchard/audio_package.py`), the licence gate, the pinned tt-lang kernels, and the text-model route (a language model served through the existing text stages). "New" means I
found nothing in our code or in the notes for it. Effort numbers are estimates.

### 1. ACE-Step 1.5

- Sources: https://huggingface.co/ACE-Step/Ace-Step1.5 (card, `acestep-v15-turbo/config.json`, `vae/config.json`), https://huggingface.co/ACE-Step/acestep-v15-xl-turbo, https://github.com/ace-step/ACE-Step-1.5 (MIT, 14,170 stars, last push 2026-10-11), paper https://arxiv.org/abs/2602.00744.
- Release: 2026-01-23 (v1.5), 2026-04-02 (XL 4B DiT). Not gated.
- What it is: a Qwen3-based language model (`acestep-5Hz-lm` 0.6B, 1.7B, 4B) plans the song and writes 5 Hz audio codes; a diffusion transformer turns them into a latent; an Oobleck autoencoder
  (stereo, 48 kHz, strides 2, 4, 4, 6, 10, so 1,920 samples per latent frame, 25 Hz, 64 channels) decodes it. Turbo uses 8 steps without guidance. Base and sft use 50 steps with guidance.
- DiT config (turbo): 24 layers, width 2048, 16 query heads and 8 key-value heads of 128, MLP 6,144, alternating sliding-window (128) and full attention, 8 lyric-encoder and 4 timbre-encoder layers. XL: width 2560, 32 layers, 32 heads, about 4B, 18.8 GB bf16 per its card.
- Output and length: 48 kHz stereo, 10 s to 10 min (GitHub README). Licence: MIT; the card states generated music may be used commercially. The LM variants are initialised from Qwen3 (card table) and the repository also ships a Qwen3-Embedding-0.6B text embedder; the repository declares MIT for everything.
- CPU: the GitHub README says CUDA is recommended and MPS, ROCm, Intel XPU and CPU are also supported. I did not run it.
- Reuse: the LM is an ordinary Qwen3 decoder, so the text-model route applies, and unlike audio, a language model has a token-agreement check against a CPU reference. The audio package builder, licence gate and resident runtime apply. The DiT block shape
  (grouped-query attention, sliding window, adaLN-style timestep conditioning) overlaps our `chip_dit.py` only partly; I did not read its modelling file (96 KB in the repository), so whether it has cross-attention is open.
  The Oobleck decoder is new to us (our Stable Audio 3 decoder is a transformer autoencoder with no convolutional stack), but both community bundles already carry one on the card (VAE decode 0.89 s for 30 s in the container run; PCC 0.99994 against CPU for 512 frames).
- Fit to one p150: the bundle reports 3.2 GB device DRAM in use for the turbo bundle. XL (about 9 GB bf16 by the GitHub README) plus the 4B LM (about 8 GB) fits by weight size.
- Plan and estimate (days, estimate): A) pull the bundle and run our audio stage 2 and the length sweep on it, 1; B) put the 1.7B planner on the chip through the text-model route and check its audio codes by token agreement, 3 to 4; C) XL turbo DiT on the chip, 3 to 4.
- First blocker: our stage 2 expects a runtime built by our package builder. The community bundle is built differently (its image resolved its own torch and numpy). Step A may need a thin adapter before any number is ours.

### 2. HeartMuLa-oss-3B and HeartCodec

- Sources: https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B (config: Llama-3B backbone, Llama-300M decoder, 8 codebooks, audio vocabulary 8,197), https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123, https://github.com/HeartMuLa/heartlib (Apache 2.0, 3,772 stars, last push 2026-10-08), paper https://arxiv.org/abs/2601.10547.
- Release: 2026-01-14, with a reinforcement-learning variant and a "happy-new-year" variant on 2026-01-23 and later; the README recommends the happy-new-year LM and the 20260123 codec. A 7B version is listed as not yet released. Not gated.
- What it is: a lyrics-and-tags song model. A large Llama-style backbone predicts the first codebook per frame and a small decoder predicts the rest, at 12.5 Hz (heartlib README). HeartCodec turns the 8 codebooks (8,192 entries each) into 48 kHz stereo; its config shows
  residual quantizers, a 24-layer transformer with adaLN-single normalisation, and strided convolution up and down sampling (factors 3, 4, 4, 4, 5). The abstract does not say whether the codec decoder uses flow matching; I did not read the codec code.
- Output: the README sets a 240 s default maximum. Languages: Chinese, English, Japanese, Korean, Spanish. Licence: Apache 2.0 for code and weights (the README says the licence was updated to Apache 2.0).
- Speed on a GPU: the README says inference is "around RTF 1.0" and lists acceleration as undone. By dtype: the LM defaults to bf16, the codec to fp32, and the README warns that bf16 for the codec "may result in the degradation of audio quality".
- CPU: the README defaults to CUDA; CPU is not stated.
- Reuse: the text-model route for the 3B Llama backbone; the depth-decoder pattern of the MiniMax-Music3 port (a small per-frame decoder over RVQ codebooks, which our MiniMax log measured at 38 ms per frame); resident runtime, builder, gate (Apache 2.0 is the simple case). New: the HeartCodec decoder, and a CPU or GPU
  reference to compare against, since the codec is the part we cannot yet check.
- Fit to one p150: 3.94B LM parameters in bf16 is about 8 GB, the 1.66B codec in fp32 about 6.6 GB. Fits.
- Estimate: 8 to 12 days. First blocker: the codec stays in fp32 and fp32 matmuls on the chip are slower and larger; if bf16 is audibly worse, the codec may need mixed precision found by listening.

### 3. DiffRhythm 1.2 and DiffRhythm 2

- Sources: https://huggingface.co/ASLP-lab/DiffRhythm-1_2-full (Apache 2.0, 2.2 GB `cfm_model.pt`, config: 16 layers, width 2048, 32 heads, 4 conv layers), https://huggingface.co/ASLP-lab/DiffRhythm2 (Apache 2.0, 1.14B, config: 16 layers, width 2048, 16 heads, block size 10, flex attention, decoder with strides 10, 10, 8, 3, 2, 2 and snakebeta activation at 48 kHz), https://github.com/ASLP-lab/DiffRhythm2 (Apache 2.0, last push 2025-11-09), paper https://arxiv.org/abs/2510.22950.
- Release: DiffRhythm 1.2 2025-09-02; DiffRhythm 2 weights 2025-10-30. Not gated.
- What it is: lyrics-aligned full songs by a flow-matching transformer. DiffRhythm 2 adds block-wise semi-autoregressive generation, so long songs do not need one giant pass.
- Reuse: for 1.2 the community bundle already exists, so the cost is running it through our stages (1 day, estimate). For DiffRhythm 2 the transformer is a standard width-2048 stack (our DiT runtime and resident runtime apply in outline). The flex-attention block mask and the BigVGAN-style vocoder are new. The 1.2 bundle's `OobleckTT` does not cover a snake vocoder.
- Fit to one p150: yes, 1.1B to about 1.2B.
- Estimate: DiffRhythm 2 first chip clip 6 to 9 days. First blocker: the vocoder (transposed convolutions with large strides and snake activation), with the block attention mask second.
- Caveat on value: the 1.2 bundle makes songs of 96 to 285 s and the same community author ported ACE-Step; DiffRhythm 1.2 is the cheapest "another music model" we can show, and a poor use of new-code effort unless we also want DiffRhythm 2.

### 4. YuE2-3B

- Sources: https://huggingface.co/m-a-p/YuE2-3B (card, `config.json`), https://huggingface.co/m-a-p/YuE2-Vae, https://github.com/multimodal-art-projection/YuE (Apache 2.0 code, 12,878 stars, last push 2026-10-06), reports arXiv 2609.33757 and 2610.05336.
- Release: 2026-09-09. Not gated. Weights: CC BY-NC 4.0 (card and its LICENSE file); non-commercial.
- What it is: per the card, one autoregressive-then-non-autoregressive "Mixture-of-Transformers" backbone writes a symbolic score and semantic tokens, then produces acoustic latents by flow matching; the VAE (YuE2-Vae, 132.6M parameters, snake activations, 64-dimension latents)
  makes 48 kHz stereo. Backbone config: 28 layers, width 2048, 16 query and 8 key-value heads of 128, MLP 6,144, vocabulary 184,704, context 24,576. Qwen/Qwen3-1.7B's `config.json` has the same layer count, width, head counts, head dimension and MLP size, so the text-model route's Qwen3 support is the starting point (vocabulary and the extra heads differ).
- Length: 3.6-minute song in 71 s on an RTX 4090 with CUDA graphs and FlashAttention; the card's examples run to 5:00. Peak memory 11.18 GiB. The card asks for a 24 GB NVIDIA GPU. The card claims the highest average on its own WildSongBench against Suno v5 and v6; these are the authors' numbers and I did not check them.
- CPU: not stated.
- Reuse: text-model route for the backbone shape; resident runtime, builder, gate (the gate must refuse commercial use). New: the non-autoregressive flow-matching stage, the AR to NAR switching, and the snake-activation VAE (a convolution-heavy decoder like the ACE-Step one, so work from model 1 transfers).
- Estimate: 10 to 14 days. First blocker: the model code ships as an installable wheel (`yue2_infer-0.1.5`) and has not been read; the mixture-of-transformers routing is the unknown.
- Why it is fourth: highest claimed quality, but the licence blocks any commercial package and the code is the least read.

### 5. MOSS-SoundEffect-v2.0 (sound effects, short musical clips)

- Sources: https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0 (card, `transformer/config.json`, `model_index.json`), https://github.com/OpenMOSS/MOSS-TTS (Apache 2.0, last push 2026-09-06).
- Release: 2026-05-25. Apache 2.0, ungated.
- What it is: a flow-matching diffusion transformer of class `WanAudioModel` (30 layers, width 1,536, 12 heads, MLP 8,960, 128-channel latents), a Qwen3-1.7B text encoder and a DAC VAE, 48 kHz, up to 30 s. Recommended settings: 100 solver steps, guidance 4.0.
- Music scope: the card lists "short musical/percussive clips" under sound effects. It is not a song model.
- Reuse: the transformer is close to our `chip_dit.py` runtime in kind (a plain adaLN-style stack with cross-attention to text; I did not read the model code to confirm). The Qwen3 text encoder uses the text-model route. I did not check whether tt-dit has a block for this class. New: the DAC VAE decoder.
- Fit to one p150: 1.3B DiT plus 1.7B encoder, small.
- Estimate: 5 to 8 days. First blocker: the DAC VAE decoder, then speed: 100 steps with guidance means up to 200 transformer passes per clip (our Stable Audio 3 medium base does 50 passes in 14 s).
- Why it is on the list: it is the closest in kind to what already runs, and it adds a sound-effects model with a permissive licence. It is fifth because it is not music.

## Not worth it now

| Model | Reason (with source) |
|---|---|
| MusicGen small/medium/large and stereo variants (Meta) | Weights CC BY-NC 4.0, code MIT; released 2023 (https://huggingface.co/facebook/musicgen-small). Decoder-only LM (24 layers, width 1,024, 4 codebooks at 50 Hz, 32 kHz EnCodec, T5 text encoder), so about 50 autoregressive steps per second of audio. Easy to run on CPU with `transformers`, but it is older and non-commercial, and the 2026 models above it in this list report more control (lyrics, editing). Useful only as a codec warm-up if HeartMuLa stalls on its codec. |
| SongGeneration v2 (Tencent; `lglg666/SongGeneration-v2-large`) | The licence is unclear: the Hugging Face repository says "unknown", a mirror says "Other", a third-party report says GitHub shows NOASSERTION; the upstream GitHub URL I tried returned 404. A third-party summary quotes a 4B hybrid LLM-diffusion model and RTF 0.82 with 22 to 28 GB of memory; I could not confirm those against a primary source. Revisit once the licence is readable. |
| Magenta RealTime 2 (Google; https://huggingface.co/google/magenta-realtime-2) | CC BY 4.0 weights, Apache 2.0 code, 2.4B and 230M configurations. The checkpoints are listed as `.safetensors` plus `.mlxfn` and the codec is a 64-deep RVQ at 25 Hz built for streaming with 1 s windowed attention. Its streaming shape matches nothing in our harness, and no PyTorch reference is named. A later project, after a streaming runtime exists. |
| Stable Audio Open 1.0 and Open Small (https://huggingface.co/stabilityai/stable-audio-open-small) | Gated, Stability community licence, 11 s maximum for Small. Older family than Stable Audio 3, which we already run. |
| TangoFlux (https://huggingface.co/declare-lab/TangoFlux) | Sound effects, 30 s at 44.1 kHz, 2024, licence files for Stability community terms and GitHub reports NOASSERTION. Stable Audio 3 small-sfx covers the same use. |
| Woosh (Sony; https://huggingface.co/drbaph/Woosh) | Sound effects and video-to-audio; the Hugging Face copy is CC BY-NC 4.0 while the GitHub repository is Apache 2.0, so the weights licence needs a check. Out of scope for music. |
| AudioX (https://huggingface.co/HKUSTAudio/AudioX) | CC BY-NC 4.0, 16 kHz output per the tt-metal bounty issue; a bounty exists. |
| HunyuanVideo-Foley | Video-to-audio; needs a video input. Tencent community licence. |
| MIDI-LLM, Shao-MusicGeneration-v1.0, SongBloom repacks, Riffusion, AudioLDM and AudioLDM2 | MIDI-LLM emits MIDI tokens; Shao's checkpoint is in a distributed-checkpoint format with a non-commercial licence and no architecture I could read; SongBloom has only a third-party safetensors copy; Riffusion (2022) and AudioLDM2 (2023, CC BY-NC-SA 4.0) are older. |
| LFM2.5-Audio, PersonaPlex, Qwen3-TTS-Tokenizer, BigVGAN and RE-USE | Speech dialogue models, speech codecs, a vocoder and speech enhancement. No music generation. |
| MiniMax-Music3 | Already brought up. |

## Searches and gaps

- Hugging Face API, pipeline tags `text-to-audio`, `text-to-music` and `audio-to-audio`, each sorted by downloads, likes and trending score, plus keyword searches (musicgen, ace-step, yue, diffrhythm, songbloom,
  audiocraft, stable-audio, audioldm, tango, sfx, foley and others) and a catalog search for Blackhole bundles. The tag `text-to-music` returned nothing separate from `text-to-audio`.
- Authors: Stability AI (Stable Audio 3, Open, Open Small), Meta (MusicGen and AudioCraft; the AudioCraft repository last pushed 2026-03-03), ACE Studio and StepFun, Tencent (SongGeneration, HunyuanVideo-Foley), HeartMuLa, m-a-p (YuE2), ASLP-lab and Xiaomi (DiffRhythm),
  Google (Magenta RealTime 2; Lyria skipped), Sony (Woosh), OpenMOSS, HKUST (AudioX), declare-lab (TangoFlux). I found no Alibaba or Qwen music-generation model (only Qwen3-TTS speech pieces), and no NVIDIA text-to-music model
  (NVIDIA appears with a speech dialogue model, a vocoder and a speech enhancer).
- The web search found no authoritative list of 2026 releases; it confirmed Magenta RealTime 2, HeartMuLa, YuE2, Stable Audio 3 and MiniMax Music 3 as open-weight releases of 2026 and gave conflicting dates for YuE2. I used the Hugging Face `createdAt` dates in this file.
- Not done: no model code was run, no weights were downloaded, no CPU reference was timed. CPU support is quoted from README text where it exists, and marked "not stated" where it does not. Effort numbers are estimates from the size of the unknown parts, calibrated against the
  Stable Audio 3 decoder work (about one overnight session for a decoder with one missing op) and not against any measurement of these models.

## Proposed order

1. ACE-Step 1.5: step A (existing bundle through our stages), then B (planner LM on the chip), then C (XL).
2. HeartMuLa: start with a CPU or GPU reference clip and the codec decoder read, since the codec is the unknown.
3. DiffRhythm 1.2 bundle as a one-day second data point, DiffRhythm 2 only if the snake vocoder from YuE2 or MOSS is built first.
4. YuE2 and MOSS-SoundEffect share a convolutional snake or DAC decoder with ACE-Step's Oobleck VAE; after model 1 the decoder work is partly paid for.

Public issues, one per top-three candidate, are filed with the plan and the first blocker and linked from the pull request.
