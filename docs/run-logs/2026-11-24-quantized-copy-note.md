# A quantized copy of a supported model: what stage 0 sees and what the pause says (2026-11-24, issue 63)

## The part of the issue that needs no board

Issue 63 asks for a hardware test of one quantized copy to learn which formats the runtime accepts, converts or refuses, and then a class or hazard. That test needs an operator-approved board and has not run. The issue's own comment of 2026-11-02 names what can be done without one: the pause for a quantized copy should say that the model is a quantized copy of a supported one. This change does that and nothing more. The class stays `full-port`, and the path, the gate and the stage 2 pause are unchanged.

## What stage 0 saw on a real copy

Stage 0 was run by hand on `Qwen/Qwen3.8-27B-FP8` (revision c3a462cc) against the BF16 `Qwen/Qwen3.8-27B` (revision 1d4bf0f2), both from this machine's Hugging Face cache, with the shipped `delta_triage.py` in a scratch run directory. The template exited 0 in 1.4 s and wrote class `full-port`, path `full-port`. Its path reasons say, for each weight, that the tensor has dtype F8_E4M3 in the new model and BF16 in the nearest model. The tensor comparison shows:

| | BF16 tensors | F8_E4M3 tensors | tensors only in the model |
| --- | --- | --- | --- |
| Qwen3.8-27B-FP8 | 1,199 | 407 | 407, each ending in `weight_scale_inv` |
| Qwen3.8-27B | 1,199 | 0 | 0 |

The model's `config.json` also carries a `quantization_config` with `quant_method` `fp8`, `fmt` `e4m3` and a list of modules left unconverted.

## What changed

- `orchard/quantized_copy.py` reads stage 0's `evidence/tensor-compare.json` and the model's `quantization_config` and names the format: FP8, NVFP4, MXFP8, GPTQ, AWQ or a compressed-tensors method. It looks at the config, then at tensor names that only the model has (`weight_scale_inv`, `weight_scale_2` and `input_scale`, `qweight`, `qzeros` and `g_idx`), then at 8-bit float dtypes that the nearest model does not use. A nearest model that is quantized in the same way gives no note.
- When the full-port pause is written, the supervisor adds one sentence after the fixed text: "The model looks like an FP8 quantized copy of Qwen/Qwen3.8-27B (stage 0 saw: ...). No quantized checkpoint has been served on a Tenstorrent chip by this harness, so the class stays full port until a hardware test of that format says what the runtime accepts, converts or refuses (issue 63)." It also writes a ledger decision, `quantized copy noted`, with the format, the nearest model and what was seen. The pause still classifies as a full port and still maps to the block code `needs-new-model-code`. A resume of a run that paused this way counts as the acknowledged full-port pause.
- A real copy exposed one bug in the first version: the model's `config.json` in the cache is a link into `blobs/`, and the bounded reader refused it, so the `quantization_config` signal was lost. The config is now followed only when the link ends inside the snapshot or its cache's `blobs` directory (`safeio.resolved_inside`).

On the real pair the note now names FP8 and cites the 407 scale tensors, the 407 F8_E4M3 tensors and the config.

## What is left

- The hardware test of one quantized copy of a model that already has a bundle (the issue's candidates: `Qwen/Qwen3.8-27B-FP8`, `nvidia/Qwen3.8-27B-NVFP4`, `Qwen/Qwen3.5-35B-A3B-FP8`), and then a class or a hazard that states what the runtime does with the format. Until then this note is a hint in the pause text and the ledger.
- A full `bringup` run on a quantized copy was not made; the detector was checked on stage 0's real output and on the run-directory fixtures of the tests.
