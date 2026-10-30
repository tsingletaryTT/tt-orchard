# ACE-Step 1.5 turbo: moving the CPU pieces of a request onto the chip (2026-10-30, issue 254)

Operator instruction: the ACE-Step 1.5 turbo package is not ready. Put more of it on the Tenstorrent hardware and less on the CPU. Before this work only the diffusion transformer and the VAE decoder ran
on the chip, and the rest of a 30 s request (about 3 of its 3.5 s) ran on the CPU. Paths are placeholders; a scratch directory holds the recorded runs. This log covers steps 1 and 2 (profile and
moves). Step 3 (the package) is in `docs/run-logs/2026-10-30-acestep-package-fast.md`.

Hardware: one chip of board 1 (chips 2 and 3 leased through gozer with `--owner-pid` on a process that lived for the whole session). Prompt "warm lo-fi hip hop, mellow rhodes piano, soft vinyl crackle,
relaxed drums", instrumental, seed 42, 8 steps, no planner, unless a row says otherwise. The host is shared (load average 2 to 7 during the runs); every timing below names its load or comes from a
batch with a stated load.

## Step 1: where a request spends its time

`scripts/acestep/acestep_profile.py` wraps a wall clock (`StageClock`) around each stage of one loaded `Session` and prints inclusive and exclusive time. One unrecorded call per length compiles the kernels,
then one timed call. Exclusive times add up to the call's total. Evidence: `docs/run-logs/evidence/acestep-chip-path/profile-base-30-120.json`.

Three pieces of the 3 s turned out to be repeated or constant work, and one was the garbage collector. The encoders, the pieces the operator named, come after them.

Before any change (chip, bundle code unchanged; this batch ran at load average about 7, so the totals are 0.3 to 0.9 s above the quiet-host sweeps below):

| Stage (exclusive time) | 30 s: calls | seconds | share | 120 s: calls | seconds | share |
|---|---:|---:|---:|---:|---:|---:|
| timbre encoder (4 transformer layers on the CPU, reference audio) | 2 | 1.289 | 33.1 % | 2 | 1.329 | 22.9 % |
| `gc.collect` (called twice per request by the handler) | 2 | 0.836 | 21.5 % | 2 | 0.728 | 12.6 % |
| VAE decode on the chip | 2 | 0.491 | 12.6 % | 4 | 1.882 | 32.4 % |
| transformer steps on the chip (includes the host-to-chip copies of each step) | 8 | 0.399 | 10.2 % | 8 | 0.827 | 14.3 % |
| text encoder (Qwen3-Embedding-0.6B, CPU) | 1 | 0.229 | 5.9 % | 1 | 0.240 | 4.1 % |
| cross-attention keys and values, host projection (inside `prepare`) | 1 | 0.161 | 4.1 % | 1 | 0.145 | 2.5 % |
| lyric encoder (8 layers, CPU) | 2 | 0.152 | 3.9 % | 2 | 0.185 | 3.2 % |
| sampler wavelet correction (DCW) | 8 | 0.058 | 1.5 % | 8 | 0.018 | 0.3 % |
| `prepare` tables, masks and upload | 1 | 0.037 | 0.9 % | 1 | 0.043 | 0.7 % |
| tokenise, masks, silence latents | 1 | 0.030 | 0.8 % | 1 | 0.028 | 0.5 % |
| WAV write | 1 | 0.024 | 0.6 % | 1 | 0.241 | 4.2 % |
| peak normalisation | 1 | 0.004 | 0.1 % | 1 | 0.017 | 0.3 % |
| total of the call | | 3.899 | | | 5.800 | |

Findings:
- The condition encoder ran twice per request. ACE-Step's handler calls `model.prepare_condition`, and `generate_audio` calls it again with the same arguments (2 calls of the timbre encoder, the lyric encoder and the
  text projector in the profile). The first result is only returned to the caller.
- The timbre encoder processes the reference audio. For text to music the reference is always the first 750 frames of the silence latent (`infer_refer_latent`), so its output is a constant of the
  checkpoint.
- The handler calls `gc.collect()` twice per request (`generate_music.py` and `generate_music_decode.py`). With the models loaded each call walks every object in the process (0.4 s).
- The host-to-chip copies are small. `patchify` and `unpatchify` take 2 ms for 8 steps; the transformer's 0.05 s per step is chip time. Keeping the hidden states on the device across the 8 steps would
  save a few milliseconds, so it was not done. The sampler arithmetic (an Euler update on a [750, 64] tensor) is below 1 ms per step.
- A request with a 207-token lyric (60 s, `profile-vocal-60-base.json`) spends 0.555 s in the CPU lyric encoder (2 calls) next to 0.230 s in the text encoder. The lyric encoder is the largest CPU piece of a vocal request after the
  timbre encoder.

## Step 2: the changes, one at a time

Order: largest first. The first three change no arithmetic: they remove repeated or constant work, and the outputs are bit-identical to the unchanged path (maximum absolute difference 0.0 on the noise, the
conditioning, every step's velocity, the final latent and the waveform; compared as arrays of the recorded runs; rechecked for the first three steps together on the final code). The last three put work on the chip in bfloat16, so each has to pass the
per-step bars.

The bars are those of `orchard/step_bars.py`: the teacher-forced bfloat16 CPU floor of each step (committed in `docs/run-logs/evidence/acestep-step-bars/`), bar = 2.0 x floor, wrong-step control must fail. For the
moved encoders the check needed one addition: `acestep_run.py --stepcheck REF --own-conditioning` first makes a clip, then gives the chip's transformer the conditioning that this process made (text encoder, lyric encoder,
cross-attention keys and values, with every chip step on) and compares each step's velocity with the float32 CPU run's velocity on the same recorded input x_k. The conditioning errors then show up in the step errors.
The bars themselves are unchanged.

The ladder below adds one step to the row above it (30 s lo-fi, one warm-up, three timed calls per row, medians; evidence `ladder.json`; load average 2 to 4):

| Row | Step added | Median total (s) | Change from the row above | Output compared with the unchanged chip path |
|---|---|---:|---:|---|
| 0 | none (the bundle as run in the first bring-up) | 3.007 | | |
| 1 | `gc_freeze`: `gc.freeze()` after load | 2.370 | -0.64 | identical |
| 2 | `timbre_cache`: the timbre encoding of the silence reference is computed once | 1.401 | -0.97 | identical |
| 3 | `cond_cache`: the second `prepare_condition` call of a request is answered from the first | 1.363 | -0.04 | identical |
| 4 | `kv_on_chip`: cross-attention keys and values made on the chip | 1.178 | -0.19 | within the bars (below) |
| 5 | `text_on_chip`: the Qwen3 text encoder on the chip | 1.001 | -0.18 | within the bars |
| 6 | `lyric_on_chip`: the lyric encoder on the chip | 0.938 | -0.06 | within the bars |

A 60 s request with a 207-token lyric: 1.93 s with rows 1 to 5, 1.73 s with row 6 added (three timed calls each); the unchanged path took 4.19 s in the profile batch.

What each step is:
- `gc_freeze` (`acestep_fast.freeze_gc`). Moves the loaded objects out of the collector's view. Collection still runs and still frees what a request creates.
- `timbre_cache` (`acestep_fast.cache_timbre`). Cache keyed by the sha256 of the timbre encoder's two input tensors, at most 4 entries. A different reference gets a different key.
- `cond_cache` (`acestep_fast.memoize_prepare_condition`). One entry, compared tensor by tensor, an argument passed as `None` counted as left out, and cleared at the start of every request. The clearing was added
  after a measurement showed the first version also answered an identical next request from the cache, which made a benchmark that repeats one prompt measure the cache. Per request it saves the second encoder call only,
  0.04 s on this prompt and about 0.2 s with long lyrics.
- `kv_on_chip` (`acestep_chip.install_kv_on_chip`). `AceStepDiT.prepare` computed the 24 layers' cross-attention keys and values on the host in float32 (a 2048 by 2048 projection of the encoder states, then per layer a
  projection and a per-head RMS norm of the keys). The version here uploads the padded encoder states once and runs the projections and the norm on the chip in bfloat16, with the checkpoint's weights. The bundle's
  files are not edited: the method is replaced on the loaded worker. On the recorded lo-fi conditioning it took 0.0097 s against 0.139 s on the host; relative L2 against the host's float32 values 0.003 (keys) and 0.006 (values).
- `text_on_chip` (`acestep_chip.install_text_on_chip`, class `ChipEncoderStack`). The 28 layers of Qwen3-Embedding-0.6B run on the chip (grouped-query attention with per-head norm and rotary embedding in the adjacent-pair
  layout, fused SwiGLU, an additive causal mask, the sequence padded with zeros to a multiple of 128 and the padded rows dropped). The weights come from the handler's loaded encoder; the token embedding lookup stays on
  the CPU. 0.011 s against 0.163 s on the CPU for 73 tokens. Relative L2 against the float32 CPU output: 0.0228 with the DiT's matmul fidelity (HiFi2), 0.0187 with HiFi4 (the CPU in bfloat16 gives 0.0182). The stack
  uses HiFi4.
- `lyric_on_chip` (`acestep_chip.install_lyric_on_chip`). The same stack for the lyric encoder (8 layers, width 2048, bidirectional attention, a 128-frame sliding window on alternate layers). It applies to batch 1 with
  every token valid; anything else runs the original CPU path. 0.010 s against 0.252 s for 207 tokens; relative L2 0.0125 with HiFi2 and 0.0316 with HiFi4 (the CPU in bfloat16 gives 0.186). The stack uses HiFi2.
  That the higher fidelity gave the larger error here is a measurement on one lyric; no explanation was found.

### Per-step bars with the moved pieces

Per-step relative L2 against the float32 CPU run on the same input, 30 s, `--own-conditioning`, every chip step on (`bars-fast-*.json`, `stepcheck-fast-*.json`). The bars are the committed ones, 2.0 x the bfloat16 CPU floor.

| Step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | decoder |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lo-fi: chip, before (first bring-up) | 0.078 | 0.054 | 0.023 | 0.020 | 0.023 | 0.020 | 0.022 | 0.028 | 0.0141 |
| lo-fi: chip, all steps on | 0.093 | 0.077 | 0.036 | 0.027 | 0.059 | 0.027 | 0.023 | 0.028 | 0.0141 |
| lo-fi: bar | 0.344 | 0.252 | 0.102 | 0.063 | 0.076 | 0.043 | 0.034 | 0.044 | 0.0151 |
| folk: chip, before | 0.042 | 0.033 | 0.022 | 0.018 | 0.020 | 0.020 | 0.023 | 0.028 | 0.0125 |
| folk: chip, all steps on | 0.061 | 0.071 | 0.022 | 0.018 | 0.020 | 0.021 | 0.023 | 0.028 | 0.0125 |
| folk: bar | 0.265 | 0.197 | 0.081 | 0.052 | 0.034 | 0.040 | 0.037 | 0.045 | 0.0159 |

Every step passes on both prompts and `step_bars` reports "usable: True" (the wrong-step control fails every bar). The closest approach is lo-fi step 4: 0.059 against a bar of 0.076 (1.57 x the floor). The same step
was 0.023 before; it was 0.020 with `kv_on_chip` alone and 0.036 in an earlier run with the text encoder at the lower fidelity, so the text encoder's conditioning error is what moves it. With `kv_on_chip` alone the
steps are within 0.02 of the old values (`bars-kv-only-*.json`).

### End-to-end agreement

Four 30 s clips (lo-fi seeds 42 and 11, folk with a two-line lyric seeds 7 and 3), each against a float32 CPU run of the same prompt and seed (`agreement-end-to-end.json`). Latent and waveform Pearson correlation
(PCC), chip path before and after:

| Clip | latent PCC before | latent PCC after | waveform PCC before | waveform PCC after |
|---|---:|---:|---:|---:|
| lo-fi 42 | 0.9818 | 0.9783 | 0.9389 | 0.9176 |
| folk 7 | 0.9978 | 0.9939 | 0.9798 | 0.9793 |
| lo-fi 11 | 0.9927 | 0.9635 | 0.9535 | 0.8564 |
| folk 3 | 0.9986 | 0.9985 | 0.9847 | 0.9876 |

The after values are lower on three of four clips and the lo-fi 11 clip drops to a waveform PCC of 0.856. That comparison feeds each step's error into the next, so it is sensitive to any change of the
arithmetic. An ablation on the two lo-fi clips (steps `A` = the first three plus `kv_on_chip`, `B` = A plus `lyric_on_chip`, `C` = A plus `text_on_chip`) shows how much:

| Clip | variant | conditioning relative L2 | latent PCC | waveform PCC |
|---|---|---:|---:|---:|
| lo-fi 42 | before | 0 | 0.9818 | 0.9389 |
| lo-fi 42 | A | 0 | 0.9906 | 0.9778 |
| lo-fi 42 | B | 0.0002 | 0.9784 | 0.9283 |
| lo-fi 42 | C | 0.0182 | 0.9787 | 0.9190 |
| lo-fi 11 | before | 0 | 0.9927 | 0.9535 |
| lo-fi 11 | A | 0 | 0.9785 | 0.8855 |
| lo-fi 11 | B | 0.0002 | 0.9820 | 0.9155 |
| lo-fi 11 | C | 0.0182 | 0.9669 | 0.8679 |

A conditioning change of 0.0002 (lyric encoder on the chip) moves lo-fi 42 from 0.9906 to 0.9784 and lo-fi 11 from 0.9785 to 0.9820. A change of the keys and values alone, where the encoder states are identical, moves
lo-fi 11 from 0.9927 to 0.9785. So the end-to-end comparison varies by about 0.01 to 0.03 in latent PCC and 0.05 to 0.07 in waveform PCC under changes far smaller than bfloat16 precision, and it cannot rule the moved
pieces in or out. The per-step bars hold the inputs equal and are the check that applies. The bring-up log (2026-10-29) reached the same conclusion for the CPU bfloat16 run, whose waveform PCC against float32 was 0.892.
Listening is the check that has not been done (see Clips).

## Remaining CPU time

With every step on (`profile-fast-30-120-480.json`, quiet host):

| Length | total (s) | chip VAE | chip transformer | `prepare` | WAV write | tokenise and masks | other CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 s | 0.931 | 0.456 | 0.393 | 0.005 | 0.011 | 0.006 | 0.06 |
| 120 s | 2.868 | 1.819 | 0.825 | 0.014 | 0.041 | 0.022 | 0.15 |
| 480 s | 12.036 | 6.445 | 4.646 | 0.194 | 0.157 | 0.085 | 0.5 |

What is left on the CPU is the tokeniser and masks, the host-side mask and rotary tables of `prepare` (a [S, S] float tensor per song), the sampler update and wavelet correction (0.007 s at 30 s, 0.05 s at 480 s),
the peak normalisation and the WAV write. Together with the chip-side encoders (about 0.03 s) they are the time outside the two chip stages: 9 percent of a 30 s request (0.08 s of 0.93 s) and 8 percent of a 480 s one (0.95 s of 12.04 s). They were not moved: the largest is 0.19 s at 480 s.

## Length sweep (1 warm-up, 3 timed calls per length, same host, same hour)

`sweep-base.json` (bundle path) and `sweep-fast.json` (`--fast all`). Total of one generate call in seconds, min / median / max; "rest" is the median total minus the two chip stages.

| Length | before: total | before: rest | after: total | after: rest | audio seconds per second, before | after |
|---:|---|---:|---|---:|---:|---:|
| 30 s | 2.99 / 3.00 / 3.03 | 2.15 | 0.93 / 0.93 / 0.93 | 0.07 | 10.0 | 32.3 |
| 60 s | 3.73 / 3.76 / 3.78 | 2.24 | 1.63 / 1.63 / 1.64 | 0.11 | 15.9 | 36.7 |
| 120 s | 5.03 / 5.06 / 5.08 | 2.35 | 2.95 / 2.95 / 2.99 | 0.24 | 23.7 | 40.7 |
| 240 s | 7.68 / 7.69 / 7.78 | 2.49 | 5.66 / 5.67 / 5.69 | 0.45 | 31.2 | 42.3 |
| 480 s | 14.39 / 14.57 / 14.59 | 3.31 | 12.20 / 12.26 / 12.27 | 0.94 | 32.9 | 39.2 |

The chip stages did not change (transformer 0.39 to 4.67 s, VAE 0.46 to 6.63 s). The time that the sweep calls "rest" is now 0.07 to 0.94 s. At 480 s the request is bound by the chip's VAE (6.6 s) and transformer
(4.7 s); a faster request from here needs faster chip kernels. More moves would save little. Load: the chip open and VAE warm-up take 7.5 s in both; the whole `Session` takes 13.0 s without the steps and 15.4 s with them (kernel cache warm), so the encoder stacks and the
extra weights add about 2.4 s.

## Clips to listen to (before and after, same prompt and seed)

In `orchard-audio/acestep-2026-10-30-chip-path/` under the home directory of the machine that ran this (48 kHz stereo WAV, peak-normalised by ACE-Step). `before-*` is the bundle path as first run, `after-*` is
`--fast all`. 30 s: `lofi42`, `folk7`, `lofi11`, `folk3` (the folk clips have a two-line lyric); lo-fi 42 also at 60, 120, 240 and 480 s. Each pair is the same prompt and seed through the two paths; the
agreement tables above give each path's distance from the float32 CPU run. The distance between the two clips of a pair was not measured. No one has listened to them. The sound checks and the per-step bars measure level and numerical agreement; they say nothing
about whether the music is good.

## Not done

- The planner language model (issue 272) and the XL transformer (issue 273).
- The sampler update and the wavelet correction on the device, the mask and rotary tables of `prepare` on the device, and a faster WAV write: together under 0.2 s at 480 s.
- The timbre encoder for a non-silent reference: the cache is only a cache; a request with its own reference audio would run the CPU timbre encoder. The server has no reference-audio input.
- Text-encoder lengths above 128 tokens compile a new shape on first use (the 128-token bucket is warmed at load). Lyrics over 128 tokens have the same property for the lyric encoder.
- Per-step bars on long-lyric requests: there is no float32 CPU reference for them. The lyric encoder was checked against the CPU on a 207-token lyric (relative L2 0.0125) and timed on a 60 s clip.
