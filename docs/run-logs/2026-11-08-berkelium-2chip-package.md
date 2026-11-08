# Berkelium CadQuery model: a two-chip package, benchmarked and published (2026-11-08)

Task (operator, 2026-11-08): make a two-chip package profile for the CAD model, benchmark it, and publish it to Hugging Face and the tt-model catalog.
The operator's go-ahead to publish covered this model's package only.

## What was built

`Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-merged` (a LoRA merged into Qwen3-32B) had passed the supervisor's stage 2 check on four chips and stopped at
stage 4 with `package-missing`: the base bundle `mando2222/qwen3-32b-blackhole-v51` has only a `p150x4` profile. The new package is a container (v5.1)
package that reuses the base bundle's image unchanged (same digest, 1.2 GB uploaded, layers shared) and changes the manifest and card:

- weights pointer: the fine-tune at revision `73df255658c8b469b0b6081f989b95eff6730892` (not included in the package);
- serve profiles: `p300` (hardware `p300`, mesh `P300`, 8,192 context, 32 sequences, the default) and `p150x4` (the base profile: mesh `P150x4`, 32,768 context, 32 sequences);
- the base manifest's `tt.sample_on_device_mode = "all"` removed from the serve defaults (see Findings);
- card sections from the measurements below.

Published: `episod/berkelium-qwen3-32b-cadquery-p300` on Hugging Face (public, listed in the community catalog; tags `tt-model-container` and `tt-model-catalog` present).
The package is not verified by Tenstorrent.

## Findings

1. The image accepts `MESH_DEVICE=P300` for Qwen3-32B and runs it on one board. Weights are stored as bfloat8_b (attention) and bfloat4_b (feed-forward up and gate),
   which is the runtime's performance setting; the tensor cache for `P300` is about 35 to 43 GB. The KV cache holds 133,120 tokens.
2. With `--max-num-seqs 8` the output on two chips was garbled ("Okay as [The8 Please=# asap ..."). With `--max-num-seqs 32` the same server gave coherent text and valid CadQuery.
   The cause is not known. The package pins 32.
3. The base package's serve default `tt.sample_on_device_mode = "all"` made the first decode fail on two chips (`AssertionError: ttnn_decode_forward got on_device_logits=True but no on-device sampling module exists`).
   The four-chip stage 2 check had run without it. The package leaves it out, so sampling runs on the host. Whether the base package's own `p150x4` profile fails the same way was not tested.
4. `tt-model serve` refused to start with "chip(s) 0, 1 already in use by another container" while no container held the chips and `fuser` showed no holder. The boot check
   called tt-model's own `serve_container` with that one check replaced, only while a gozer lease held the chips (`scripts/can_cad/serve_staged.py`). The cause of the refusal was not found.
   It looks like the same refusal as issue 560.
5. The tensor cache for this package is large. The root disk went to 100% while it was written under `~/.cache/tt-model`; the caches now live on the bonus drive and are linked in.
   Plan for 35 to 45 GB of cache per profile: one profile needs about 96 to 106 GB (61 GB of weights plus the cache) before the Docker image and the kernel cache.

## Measurements

One p300c board, greedy decoding, the package started through tt-model's serve path, warm kernel and tensor caches. Host load average 7.3 to 9.2 during the runs.
First boot with empty caches: 532 s to a ready server (the first attempt, which then failed at decode because of finding 3). Warm boot: 81 s.

| profile | isl | osl | concurrency | n | ttft | prefill | decode | e2el |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p300 | 133 | 220 | 1 | 24 | 0.19 s | n/a | 14.4 tok/s (range 12.06 to 14.98) | 15.7 s |
| p300 | 1975 | 1 | 1 | 3 | 0.61 s | 3248 tok/s | n/a | 0.61 s |
| p300 | 6719 | 1 | 1 | 3 | 2.91 s | 2308 tok/s | n/a | 2.91 s |

An earlier pilot with the serve command of the stage 2 check (no tt-model, same image, `P300`, 32 sequences, same prompts) gave decode 14.7 tok/s median (13.69 to 15.0) and a 1,975-token prefill of 0.144 s.
The prefill figure differs from the package run by a factor of four and the cause was not found. The card uses the package figures.

CadQuery check (the checker was rewritten after the review of PR 629 and the saved replies were checked again; the counts did not change):
8 prompts, greedy, 3 identical repeats, 24 replies. Static checks, no code run: 24 of 24 have a first line that parses as JSON, exactly one python block, a block that
parses, imports of only cadquery and math (every alias is checked), and a last top-level statement that assigns `result`. All 24 passed those checks, so all 24 were run.
Each ran in a child process with an empty environment, resource limits and a timeout; Landlock refused file writes outside its work directory and TCP connections.
No network namespace was used (unprivileged ones are not available here), so UDP and Unix sockets were not blocked. The verdict comes from a STEP file that a trusted epilogue
exports after the generated code ends, loaded in a separate step: 21 of 24 export a CadQuery solid that is valid and has a positive volume (7 of 8 prompts). The failing prompt raised
`No free edges available`. Eight prompts are too few to call this an evaluation. A reply written to attack the sandbox is not what it is built to stop.

## What is not done

- No concurrency above 1, no context above 8,192 on two chips, no benchmark of the `p150x4` profile through the package path.
- Quality was not compared with the four-chip run beyond the stage 2 token agreement, nor with the weights card's own evaluation (different hardware and precision).
- The orchard supervisor's stage 4 was not rerun against the published package.
- The container image is the base bundle's, copied byte for byte. `mando2222/qwen3-32b-blackhole-v51` states no licence; the card says so.
- The weights carry a non-commercial request (the fine-tune used cc-by-nc-4.0 data). The card repeats it.

Scripts: `scripts/can_cad/` (README there).
