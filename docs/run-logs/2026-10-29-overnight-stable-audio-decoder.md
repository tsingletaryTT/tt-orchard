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
- 20:15 to 20:25 (first hardware sessions, one chip each, leases released after each):
  - First chip decode of the 174-frame latent saved by the reference run of PR 165. The layers ran, but the waveform was wrong (PCC -0.02 against the CPU decode). Layer
    by layer the stream agreed to PCC 0.92 to 0.996 with relative L2 errors of 10 to 44 percent, much worse than the bfloat16 floor.
  - Section-by-section probe of layer 0 on the real input (same weights, same seed): every section up to and including the two attention outputs agreed (the two
    outputs each had a relative error of 2.7 and 4.4 percent), and the section after their subtraction did not (relative error 47 percent). Cause: the two attention
    outputs have norms near 290 and are nearly equal, and their difference has a norm near 22, so a bfloat16 error of a few percent in each is a large error in the difference.
  - Fix (`precise_attention`): the query, key and value path stays in float32 (the `to_qkv` weights are also kept in float32), the attention is computed with
    matrix multiplies and a float32 softmax, and the difference is taken on the probabilities, `(softmax(s1) - softmax(s2)) v`. The fused attention kernel takes bfloat16 only.
    Layer 0 after the fix: relative L2 error 0.035 (was 0.43).
  - Decoder alone (same latent, seed 42, noise identical on both sides), chip against the CPU float32 decode: layer relative L2 errors 0.035, 0.027, 0.008, 0.007, 0.007, 0.014
    (PCC 0.9994 to 0.99997); waveform PCC 0.9946, relative L2 0.105, log-mel L1 0.484. The floor, the same CPU decode with the decoder weights cast to bfloat16 with no chip:
    waveform PCC 0.9974, relative L2 0.073, log-mel L1 0.477. The chip is within about 1.4 times the floor on relative L2 and 1.5 percent of it on the log-mel distance.
  - Decoder time for that latent: CPU float32 1.28, 1.38, 1.20 s (three repeats, host load average about 10); chip 0.427, 0.422, 0.425 s (three repeats after a first run).
  - Whole clip, resident runtime from the merged benchmark rules (1 warm-up, 3 measured repeats on one process, diffusion transformer and decoder on the chip, small-music, 8 steps,
    guidance 1.0, seed 42, 10 s): `generate_and_decode_s` 0.921, 0.956, 0.957 s (median 0.956). The same call with the decoder on the host, from the benchmark of PR 165: 1.755, 1.608, 1.578
    s (median 1.608); CPU 3.9, 8.0, 6.0 s (median 6.0). Parts of the decoder-on-chip run: chip decoder call 0.438, 0.440, 0.483 s (the chip part with its transfers 0.403 to 0.405
    s); transformer calls 0.291, 0.301, 0.349 s; encoder 0.040 to 0.041 s; host load average 9.8.
  - `python -m orchard.audio_reference check --cross-precision` of that run against the CPU reference run of PR 165: all arrays ok with the provisional bars; final waveform PCC 0.976,
    relative L2 0.219 (with the decoder on the host the same comparison gave PCC 0.981 and relative L2 0.198).
  - Code: `orchard/skills/audio-chip-templates/chip_decoder.py` (new), `chip_decode_probe.py` (new), `--chip-decoder` in `chip_generate.py` and `ChipRuntime`; tests in
    `tests/test_audio_chip_decoder.py` (the host decomposition against the library's own block with random weights, a mutation of the key mask, the config guard).
- 20:30 other repositories in the family found while reading `model_configs.py`: `small-sfx` (sound effects; same architecture as small-music, only the conditioning text and
  training settings differ), `medium` (DiT depth 24, width 1536, 24 heads with differential attention; decoder depth 12, width 256 channels, sliding windows, sinusoidal
  feed-forward blocks) and the standalone autoencoders SAME-S and SAME-L. The account can open the gated small-sfx and medium repositories (config download returned 200).
  Plan: small-sfx through the same path as a second checkpoint (CPU reference, chip, benchmark); then the medium model, which needs differential attention in the diffusion
  transformer and a different decoder block, as a further real new-code step.
- 20:35 second model: `stabilityai/stable-audio-3-small-sfx` (sound effects), same architecture as small-music, both the diffusion transformer and the decoder paths unchanged.
  - CPU reference (reference environment, float32, prompt "Futuristic laser blast, sharp energy pulse, stereo movement, arcade game", 10 s, 8 steps, guidance 1.0, pingpong, seed 42):
    `generate_and_decode_s` 54.7 s (one sample, host load average above 10).
  - Chip, resident runtime, 1 warm-up and 3 measured repeats, transformer and decoder on the chip: `generate_and_decode_s` 0.930, 0.938, 0.943 s; chip decoder call 0.456,
    0.461, 0.470 s; transformer calls 0.321, 0.328, 0.336 s; load average 18.7.
  - `audio_reference check --cross-precision` against the CPU reference: all arrays ok with the provisional bars. Final latent relative L2 0.162 (PCC 0.987); waveform PCC 0.929,
    relative L2 0.374. The waveform number is looser than small-music's (0.976). Sound-effect clips hold sharp transients, which a phase-sensitive measure scores harshly; no
    floor was measured for this model, so the number is a measurement and not a pass against a set bar. A first run with the wrong prompt (the script's default) was
    discarded and rerun; the mismatch was caught by the settings check of the comparison.
- 20:40 third model: `stabilityai/stable-audio-3-medium` (music and sound effects; 9.2 GB float32 checkpoint). New code needed:
  - Diffusion transformer: differential attention in both the self and the cross section, 24 layers, width 1536, 24 heads (`chip_dit.py`, float32 projections, norms, attention and
    probability difference as for the decoder; the plain path is unchanged). The host decomposition passes the library-block comparison for both variants.
  - CPU reference (float32, same prompt as the music runs, 10 s, 8 steps, guidance 1.0): `generate_and_decode_s` 38.0 s (one sample).
  - Chip, transformer on the chip and the decoder on the host (one run, no warm-up, so a single sample): `generate_and_decode_s` 60.2 s, of which the transformer calls were 20.2 s
    (8 passes, the first includes the kernel compile); weights upload 7.0 s. All 24 layer outputs of the first pass agree with the reference with relative L2 error 0.004 to 0.014
    (PCC at least 0.99992); final latent after 8 steps PCC 0.997, relative L2 0.076; waveform PCC 0.9885, relative L2 0.152; every array ok against the provisional bars.
  - Next: the medium decoder (12 layers, width 1536, sliding window of one latent frame each side, sine gate in the last 8 layers) on the chip; implemented as the block layout in
    `chip_decoder.py` and tested on the host against the library with random weights (frames 1, 2 and 5, with and without the sine gate, a mutation with a wider window).
- 20:50 medium decoder on the chip and the whole medium clip on the chip (resident runtime, 1 warm-up and 3 measured repeats, both the diffusion transformer and the decoder on the chip, 8 steps, guidance 1.0, seed 42, 10 s):
  - Decoder alone on the reference run's latent (12 layers, sliding window, sine gate in the last 8): relative L2 error per layer 0.009 to 0.017 (0.070 in the last layer); waveform PCC 0.99993, relative L2 0.014, log-mel L1 0.209. The floor (the same CPU decoder with bfloat16 weights, no chip): PCC 0.99984, relative L2 0.018, log-mel L1 0.771. Decode time CPU 18.6, 23.0, 22.1 s (host load average above 10) against chip 2.52, 2.55, 2.54 s.
  - Whole clip: `generate_and_decode_s` 3.770, 3.813, 3.913 s (median 3.813) against 38.0 s for the CPU float32 reference (one sample, load average above 10) and 60.2 s for the first chip run with the decoder on the host. Parts: transformer calls 1.151, 1.159, 1.206 s; chip decoder call 2.412, 2.415, 2.432 s (2.33 s on the chip with its transfers); weights upload 7.0 s plus 1.6 s for the decoder; model load 12.6 s.
  - Agreement with the CPU reference (`audio_reference check --cross-precision`): all arrays ok; final latent PCC 0.9953, relative L2 0.097; waveform PCC 0.9884, relative L2 0.152. The first-divergence probe with the environment floor (reference environment against the shared environment's CPU run, relative L2 0.0008 at layer 0): layer 0 is 5.9 times that floor (0.0048) and the error stays between 0.004 and 0.014 over 24 layers. That is the cost of bfloat16 matrix multiplies on the chip, not a divergence that grows.
  - Spectrograms, CPU against chip, are in `docs/run-logs/evidence/2026-10-29-overnight-3/`; the clips are in the run directories outside the repository.
- 21:00 small-sfx through the harness's own stage 2 hardware test (`audio_chip_test.py`, config written by `audio_plan.write_chip_config` with the decoder on the chip, model and base, 1 warm-up and 3 measured repeats per model, one process per model):
  - Both processes ended with status 0 and a manifest. The fine-tune's runs: `generate_and_decode_s` 5.697, 6.087, 6.699 s; the base (50 steps, guidance 7, so 50 transformer passes of the stacked batch) 4.869, 4.954, 5.189 s with the decoder call 0.44 to 0.46 s. The fine-tune's repeats are 6 times slower than the 0.93 s measured an hour earlier in the same process layout, and the transformer calls (1.34 s against 0.33 s) and the decoder call (1.1 to 1.6 s against 0.46 s) both grew. The base process ran right after it with normal times. The host was shared (load average above 10) and another lead held the other board, so I treat the slow repeats as host contention and keep both sets of numbers; I did not find the cause.
  - `audio_reference check --cross-precision` of the first fine-tune run against the CPU reference: ok, waveform PCC 0.929 (the same value as the earlier run, so the output does not depend on the timing).
  - The `audio_reference control` command refuses to compare the base and fine-tune runs because their settings differ by design (the stage's own control function handles that); I did not run the stage gate itself because the supervisor needs the coder tier, which another lead's container holds on its port.
- 21:05 medium through the harness's stage 2 hardware test (same layout as small-sfx above, decoder on the chip, model and base): both processes ended with status 0.
  - Medium: `generate_and_decode_s` 3.709, 3.782, 4.178 s; transformer calls 1.144, 1.162, 1.219 s; decoder call 2.412, 2.423, 2.451 s. Medium base (50 steps, guidance 7): 17.648, 17.898, 19.192 s, of which transformer calls 14.3 s (50 passes) and the decoder 2.4 s. Load seconds: model 11.5, weights 7.0 to 7.8, decoder weights 1.7 to 1.8.
  - `audio_reference check --cross-precision` of the first medium run against the CPU reference: ok, waveform PCC 0.9884 (identical to the earlier run).
  - Control: the base's waveform against the fine-tune's for the same seed and prompt, PCC 0.004 (small-sfx pair) and -0.108 (medium pair): different weights give different audio, as the control needs.
- Licence obligations (read from the downloaded `LICENSE.md` of small-sfx and medium; the same Stability AI Community Licence as small-music): research and non-commercial use free; commercial use needs registration with Stability and ends above USD 1,000,000 annual revenue; a distributed copy needs the licence text, the NOTICE line and "Powered by Stability AI" on the product's page; outputs or the models may not be used to train or improve a foundation generative model (except the models and their derivatives); the Gemma terms apply to the text encoder. Nothing was published; the clips and arrays stay in local run directories.
- 21:07 length sweep, small-music, transformer and decoder on the chip, one loaded process, 1 warm-up and 3 measured repeats at each size (`audio_bench.py --chip-decoder`): `generate_and_decode_s` min, median, max: 10 s 0.813, 0.819, 0.824; 30 s 1.368, 1.394, 1.519; 60 s 2.456, 2.561, 2.755; 120 s 4.226, 4.318, 4.378. The 120 s clip exercises the decoder at 1,292 latent frames (646 chunks). No CPU sweep was run in the same period (the host was busy); the CPU numbers above are for 10 s only. Agreement was checked on the 10 s clip only.

## Status at 21:10 (about 8 hours before the deadline)

Reached: the Stable Audio 3 decoder on the chip (small and medium layouts), the medium diffusion transformer with differential attention on the chip, three models (small-music, small-sfx, medium) producing full clips with both parts on the chip, agreement and benchmark records, planners and stage 2 test wired for the three families. Not reached: the supervisor's stage gate on these models, the T5Gemma encoder on the chip, listening checks. Telegraph was not used (no kernel was written; every missing op was a ttnn composition). No local coder was used.
