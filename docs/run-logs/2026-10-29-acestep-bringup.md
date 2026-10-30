# ACE-Step 1.5 turbo on one Blackhole board: the community bundle run, stage 2 style checks, length sweep (2026-10-29, issue 254)

Operator instruction: go ahead with ACE-Step bring-up. Step A: run the community bundle `sunandbean/acestep-1.5-turbo-p100a` through the audio route up to stage 2, then go
further if stage 2 passes. The bundle ran on the chip with the host's tt-metal tree and needed no port. The listenable clips, the stage-2 style checks and a length sweep are below.
Paths are placeholders: `<run>` is a scratch directory with the recorded runs, `<bundle>` is a copy of the bundle's `code/` directory.

## What the bundle is, and what was read before anything ran

The Hub repository is a v5.1 container package (tt-dit-server): an OCI image (about 975 MB, 28 blobs), the Python sources under `code/`, `requirements.lock` and a kernel manifest.
The image was not loaded and no container was started. The sources were enough:

- `code/tt_acestep/` (11 files, 1,849 lines, Apache 2.0 per its card) was read in full. It opens no network connection and starts no process except `remote.py`, which starts a docker
  worker (not used here). `local.py` runs the same worker in the calling process (`LocalWorker`).
- `code/models/` is tt-metal's `models/common` and `models/tt_dit`. `code/acestep/` is ACE-Step itself (MIT, 617 files, commit `ca1e85f` per the card). I did not diff it against upstream.
  I scanned `tt_acestep` with grep for subprocess, eval, exec, pickle, `torch.load` and URLs and read the files in full; I only grepped the vendored ACE-Step tree for third-party imports.
- The weights are `ACE-Step/Ace-Step1.5` at revision `19671f40` (MIT, ungated): the turbo transformer (4.79 GB), the VAE (337 MB) and the Qwen3-Embedding-0.6B text encoder. The snapshot also
  holds the model's Python files (`modeling_acestep_v15_turbo.py` and its configuration) and `silence_latent.pt`. The handler loads the transformer with `trust_remote_code=True`, so those files run.
  I grepped them for the same patterns and found nothing; I did not read all 2,100 lines.
- The planning language model (1.7B) was not downloaded. The bundle does not load it (`thinking=False`).

Environment: a new venv under `orchard-venvs/acestep` (Python 3.12) holding eight pinned packages from the bundle's own `requirements.lock` (transformers 4.57.6, tokenizers 0.22.2,
huggingface-hub 0.36.2, einx, vector-quantize-pytorch, torch-einops-utils, pytorch-wavelets, modelscope, frozendict), installed with `--no-deps`. Everything else (torch 2.7.0 CPU, numpy 2.5.3,
ttnn 0.80.0 from the host tree v0.80.0) comes from the shared TT venv through a `.pth` that is read after the venv's own packages. The bundle was built against tt-metal commit `58648d44`; the host tree
is a different commit (v0.80.0) and the bundle's `models/` directory is used from the bundle. No kernel or op needed a change.

One hazard found: ACE-Step's handler prints "Model code mismatch ... auto-syncing" and copies its own model files into the checkpoint directory. When that directory was a link to the Hugging Face
cache snapshot, the copy went through the link and replaced two cached blobs (the model's `.py` files). I stopped that run, deleted the two blobs and downloaded them again (sizes 13,130 and 96,036 bytes
match the Hub listing). The runner now builds a private checkpoint tree: small files are copies, large weight files are links.

## Scripts

`scripts/acestep/acestep_run.py` runs ACE-Step on the CPU (float32, or bfloat16 for the two stages that go to the chip) or with the transformer and the VAE decoder on the chip through the bundle's
`LocalWorker`, and records the run in the layout of `orchard/audio_reference.py` (noise, conditioning, each step's input and velocity, the final latent, the waveform). `--stepcheck RUN` feeds a recorded
CPU run's inputs to the chip step by step. `--sweep` times lengths with a warm-up and repeats. `scripts/acestep/acestep_report.py` gives sound checks at 48 kHz, agreement, and the sweep table with the
benchmark label of `orchard/resident.py`. These scripts import nothing from the supervisor and are not wired into `orchard/audio_cli.py` yet; the stages expect the Stable Audio runtime layout.
Tests: `tests/test_acestep_scripts.py` (6; three mutations went red: the bfloat16 output cast, the VAE output cast, the 48 kHz rate).

## Results

Hardware: one board (two p300c chips), a single chip opened as a 1x1 mesh, under a gozer lease. The host was shared (load average 2 to 6 during the runs). Prompt "warm lo-fi hip hop, mellow rhodes
piano, soft vinyl crackle, relaxed drums", instrumental, seed 42, 8 steps, no planner.

### Speed (one process, one warm-up, three timed runs per length)

| Length | Total, min / median / max (s) | Transformer (chip) | VAE (chip) | Other, on the CPU | Audio seconds per second |
|---:|---|---:|---:|---:|---:|
| 30 s | 3.11 / 3.47 / 3.62 | 0.39 | 0.46 | 2.62 | 8.6 |
| 60 s | 3.82 / 3.85 / 3.86 | 0.59 | 0.95 | 2.31 | 15.6 |
| 120 s | 5.13 / 5.15 / 5.17 | 0.83 | 1.91 | 2.41 | 23.3 |
| 240 s | 8.02 / 8.08 / 8.15 | 1.91 | 3.33 | 2.83 | 29.7 |
| 480 s | 14.42 / 14.73 / 14.90 | 4.67 | 6.65 | 3.35 | 32.6 |

"Other" is the total minus the two chip stages: text and lyric encoders, the sampler, the host-chip copies and the post-processing. The first call at a new length took 9 to 21 s (kernel compile); with
a warm kernel cache it took about 4 to 6 s. Load: 62 s the first time (56 s of it is the VAE kernels compiling), 7.9 s with the kernel cache warm. Device memory after load: 3.19 GB allocated, 30.99 GB free.
The CPU reference (16 cores, float32) took 41.8 and 42.3 s for the 30 s clip (the VAE is 24.6 to 25.3 s of that); with the two chip stages in bfloat16 on the CPU it took 26.3 s. The card reports 3.25 to 3.62 s for 30 s from
its container; this run, in one process outside a container, took 3.11 to 3.62 s.

### Agreement with the CPU reference (the stage-2 comparison)

`orchard.audio_reference check --cross-precision` against the float32 CPU run, with the provisional bars: 17 of 21 arrays pass. Failures: `step_denoised/v4` to `v7` (PCC 0.975 to 0.965, relative L2 0.223 to 0.263).
Final latent PCC 0.982 (relative L2 0.189); waveform PCC 0.939 (relative L2 0.346, within the waveform bar). Three chip runs in one process produced identical arrays (PCC 1.0, relative L2 0).
Two CPU float32 runs produced identical arrays as well.

That end-to-end comparison measures the 8-step sampler feeding the chip's own errors back in. A CPU run with the transformer and VAE in bfloat16 against the same float32 reference fails the same bars
and does worse than the chip: latent PCC 0.968 (chip 0.982), waveform PCC 0.892 (chip 0.939). So the failing steps reflect bfloat16 precision in a recurrent loop. They do not point to a chip fault.

`orchard.noise_floor` with two float32 CPU runs and the bfloat16 CPU run as the floor, the chip run as the good run, and a 5 percent relative error injected into the latent and waveform as the control: the result is
"the bars are unusable". The floor is so wide (bar relative L2 0.75 on the latent) that the damaged run is not told apart from the good one. The tool refused, as designed.

The check that discriminates is per step, with the inputs held equal. `--stepcheck` gave the chip's transformer the CPU run's recorded input at each of the 8 steps and compared the velocities:

| Step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PCC | 0.9970 | 0.9986 | 0.9997 | 0.9998 | 0.9997 | 0.9998 | 0.9997 | 0.9996 |
| Relative L2 | 0.0775 | 0.0539 | 0.0234 | 0.0198 | 0.0230 | 0.0204 | 0.0225 | 0.0280 |

The chip's VAE decoder on the CPU run's final latent, against the CPU decoder: waveform PCC 0.99995, relative L2 0.0141, slope 1.010.
Controls for these numbers, computed on the CPU run's own velocities: the next step's velocity as a wrong answer gives relative L2 0.60 (step 0), 0.17 (step 3) and 0.26 (step 7), with PCC 0.84, 0.985 and 0.977.
A 10 percent relative noise gives 0.10; a 0.9 scale gives 0.10 with slope 0.90. PCC alone does not separate a wrong step from a good one at steps 3 and 7; relative L2 does. A relative L2 bar of 0.10 per step
rejects the wrong-step controls, sits exactly at the 10 percent noise and the 0.9 scale controls, and the chip passes it at every step. That bar is a choice. A teacher-forced bfloat16 CPU floor was not run, so no measured floor backs it.

Sound checks (`audio_stages.sound_checks` with 48 kHz): all five clips pass (duration, loudness windows, full-scale share). The 30 s CPU and chip clips are similar on every check (loud windows 0.847 and 0.853).

### Clips to listen to

In `orchard-audio/acestep-2026-10-29/` under the home directory of the machine that ran this (48 kHz stereo WAV, after ACE-Step's peak normalisation to -1 dB):

| File | What | Length |
|---|---|---:|
| `chip-instrumental-lofi-30s-seed42.wav` | the prompt above, chip | 30 s |
| `cpu-float32-reference-lofi-30s-seed42.wav` | same prompt and seed, CPU float32 | 30 s |
| `chip-vocal-folk-60s-seed7.wav` | folk, with a lyric of two sections | 60 s |
| `chip-ambient-120s-seed11.wav` | cinematic ambient | 120 s |
| `chip-lofi-480s-seed42.wav` | the same prompt, 8 minutes | 480 s |

No one has listened to them yet. The sound checks measure duration and loudness; they say nothing about whether the music is good or whether the sung words are intelligible.

## What is not done

- The planning language model (the 5 Hz Qwen3 planner) is not loaded, so there are no planner-made audio codes and no thinking mode. That is step 2 of issue 254.
- The XL turbo transformer (step 3 of issue 254) was not tried.
- The stage supervisor was not run on this model: stage 2 and the bars code assume the Stable Audio runtime layout and a 44.1 kHz rate (`audio_stages.SAMPLE_RATE`). The records here use the same reference format,
  so the comparison tools work, and the runner is separate.
- No package was built. `orchard/audio_package.py` copies the Stable Audio runtime files. The community bundle is already a container package and was not changed.
- Seeds do not reproduce across backends (the card says so); here the noise draw is identical on both sides (`noise/x0` PCC 1.0) because the draws come from the CPU generator.
- The per-step bar of 0.10 relative L2 has no measured floor behind it.
- Lyric intelligibility, vocal quality and long-form structure were not evaluated.
- Retries: the first CPU reference attempts failed (the missing module base class for the recorder, and the cache write described above), and the first bfloat16 CPU attempt failed on a dtype mismatch in the encoder path. All numbers above come from the runs that completed.
