# First-divergence probe and measured noise floor, checked on the Stable Audio 3 recordings (2026-10-28)

Issues 172 and 173. Two modules, `orchard/first_divergence.py` and `orchard/noise_floor.py`, each with a command line
(`python -m orchard.first_divergence`, `python -m orchard.noise_floor`) and a function a stage can call. Both work on recorded runs in the
layout of `orchard/audio_reference.py` and read them through `audio_reference.load_run`. Neither depends on the audio chip script. The
numbers below come from run directories that were already on disk. No chip was used, no lease was taken and no run was repeated.

## Runs used

All runs are the Stable Audio 3 small music model, 8 steps, guidance scale 1.0, seed 42, 10 seconds.

| Name | What it is |
|---|---|
| reference | CPU float32, the reference virtual environment |
| reference repeat | the same command run again in the same environment |
| CPU, chip environment | the same CPU code in the chip's own Python environment (torch 2.7.0, transformers 5.19.0 against 2.7.1 and 5.21.0) |
| good chip run | one Blackhole chip, bfloat16 weights, float32 residual stream (final latent PCC 0.9905) |
| bad chip run | one chip, bfloat16 residual stream (the run that failed the provisional bars) |

## What the numbers say

### The reference repeats exactly

The repeat run is bit-identical to the reference on all 56 arrays. The spread inside one environment is zero, so the whole floor comes from
the difference between the two environments.

### The environment floor (CPU in the chip's environment against the reference)

| Group | Floor, relative L2 | Floor, PCC | Worst array |
|---|---|---|---|
| conditioning | 0.0185 | 0.99983 | prompt.0 |
| layer (kind) | 0.0361 | 0.99935 | dit_output |
| layer_NN only | 0.0063 | 0.99998 | layer_18 |
| step latent | 0.0753 | 0.99717 | step 7 |
| latent | 0.0861 | 0.99630 | final |
| waveform | 0.1320 | 0.99146 | audio |

The final latent floor (relative L2 0.0861, PCC 0.9963) matches the figure from issue 162. The T5Gemma embeddings differ by 1.85 percent
between the two environments (issue 162 gave 1.85 percent over the tensor), and the sampler carries that into every later array. The noise
arrays and the `seconds_total` conditioning are bit-identical between the environments.

### The good chip run against the reference

| Number | This tool | Audio lead |
|---|---|---|
| final latent PCC | 0.99051 | 0.9905 |
| final latent relative L2 | 0.1385 | 0.138 |
| final latent slope | 0.9985 | not given |
| final latent against the environment floor | 1.6 times | 0.138 against 0.086 |
| smallest layer-output PCC (20 layers) | 0.99993 | 0.9999 or better |
| layer 0 relative L2 | 0.0170 | 0.01698 (probe, after the feed forward) |
| layer 0 memory rows (rows 0 to 63) | 0.0218 | 0.0218 |
| layer 0 latent rows (rows 64 to 237) | 0.0052 | 0.0052 |

The first block that leaves its tolerance is `layer_00` (tolerance: three times the environment floor of that array, 0.0026; the block is at
0.0170, 19 times its floor). The 64 memory rows hold 58 percent of the energy of that block and carry 96 percent of its squared error. Without
naming any rows, the high-norm search (rows with a norm above twice the median) finds 11 rows that hold 50 percent of the energy and carry 87
percent of the squared error, so a reader sees the concentration without knowing which rows are memory tokens. In layer 19 the memory rows
are at 0.0109 and the latent rows at 0.0642, which is the 6.4 percent of issue 163.

The recorded runs hold no per-operation arrays (the audio lead's probe printed its numbers and did not record them), so on real data the search
stops at the block and says so. The five numbers the probe printed for layer 0 are the same as the block-level numbers above: the probe's last
section (after the feed forward) is the layer output. The operation-level search is tested with planted divergences only. It needs a chip run
whose probe records `layer_00.attn`, `layer_00.cross`, `layer_00.local` and `layer_00.ff` (see the wiring list in the pull request).

### The bad chip run

The bad run passes through the same layer-0 numbers (relative L2 0.0154, memory rows 0.0195) and fails at the final latent: relative L2 0.2600,
which is 3.0 times the floor of 0.0861, so it sits just over a margin of 3 (0.2582). It also shows a slope of 1.037 on the final latent.

## The known-bad control

Bars are `margin` times the floor for each group, with the PCC shortfall scaled by the margin squared (the shortfall of PCC grows with the
square of the error). Margin 3, grouped by kind.

Findings in the order they appeared:

1. **The slope bar cannot come from the CPU floor.** With the slope bar at three times the CPU slope floor (0.0058 on layers), the good chip run
   fails on all 20 layers: every chip layer output is 0.7 to 1.2 percent small (slope 0.988 to 0.993). The CPU environments do not have that scale
   error. This is the good-run check doing its job: the bars were refused as unusable before any damage was tried. The `slope_rule="rel_l2"`
   option sets the slope bar equal to the relative L2 bar (a pure scale error of size s moves the relative L2 error by about s, so a scale error is
   still held to the error budget). The default rule stays `floor`, so the choice is visible in the result.
2. **A single layer rounded to bfloat16 is not detected.** Rounding layer 19 to bfloat16 adds a relative error of about 0.2 percent. The layer
   bar is 0.108 (set by the final projection, which has a floor of 0.036), and extra error below 0.092 stays under the relative L2 bar (see the note
   below on what that number means). With bars grouped by family the layer outputs get their own bar of 0.0189 and extra error below 0.0082 stays under it. Either way bfloat16 rounding of one layer is
   below what these bars can see, and the result reports the bars as unusable for that control.
3. **Damage of the size the bfloat16 residual stream caused is detected.** The bad chip run (a real run, so no synthetic assumption) fails the
   bars at margin 3 (final latent and the last denoised step). A synthetic control of relative size 0.15 on layer 19 fails with kind-level bars,
   and one of 0.05 fails with family-level bars. Both fail where the good run passes.
4. **A control that fails where the good run fails already does not count.** A first version counted any failure. On the real data that reported a
   control as caught only because the good run failed on the slope. The check now requires the control to fail on an array the good run passes,
   and a synthetic control to fail on each array it damaged.

| Setup (margin 3, slope rule relative L2) | Good run | Layer 19 damaged | Bad chip run | Usable |
|---|---|---|---|---|
| kind, rounded to bfloat16 | passes | not detected | detected | no |
| kind, relative error 0.15 | passes | detected | detected | yes |
| kind, relative error 0.10 | passes | not detected | detected | no |
| family, relative error 0.05 | passes | detected | detected | yes |
| kind, margin 2, relative error 0.15 | passes | detected | detected | yes |
| kind, slope rule floor, any damage | fails on 20 layers | not detected | detected | no |

The smallest extra relative L2 error that moves an array past its relative L2 bar (margin 3, kind): conditioning 0.052, latent 0.218, layer
0.092, step latent 0.192, waveform 0.343. This is an estimate from the relative L2 bar alone, and it assumes the extra error is independent of
the error the good run already has, so the two add in quadrature. The PCC and slope bars can reject a smaller error (an error orthogonal to the
signal shows in PCC first), so an error below these numbers is not guaranteed to pass. The numbers say how much room the relative L2 bar leaves,
and the control checks the rest. The result, the summary and the JSON carry the same label.

## What this means for the audio run

* The provisional bars (`DEFAULT_BARS`) passed the good run and failed the bad run, and so do the measured bars at margin 3. The measured bars
  carry their floor and margin in the result. The latent bar moves from the choice 0.25 to 0.258.
* The pass of the good run is 1.6 times the floor on the final latent. A margin of 2 still passes it and fails the bad run on more arrays, so the
  margin can be tightened after the next chip run if the control stays caught.
* The default control (rounding one layer to bfloat16) is too small for any bar set from these floors. A stage should name the control by the
  damage it wants to catch: a real earlier bad run (`--bad-run`), or a relative size taken from a measured dtype error.

## Not done

* No per-operation recording exists on disk, so the operation-level search was not run on real data.
* A hardware run that reproduces the layer-0 probe through the new tool needs the chip path from the open audio pull request and a lease.
* The floor has one CPU run from the chip's environment. More runs in that environment (different thread counts, a different torch build)
  would show whether 0.086 is stable.
