# How the published Flash-Next ports differ from the generated port (2026-11-12)

This note compares the approaches. It reads the graft study of 2026-11-01 (`2026-11-01-flash-next-graft-study.md`), which read the package cards, READMEs and code of the published ports, and adds what the generator work of 2026-11-12 measured. Nothing here was re-measured on the published ports. Their speeds are the authors' claims.

## The approaches

| Piece | Published ports | Generated port |
|---|---|---|
| Programming style | Route A (`sjettTT`, `DeAIafdeling`, `changh95`, upstream main): about 20 hand-written C++ kernel directories run through `generic_op`. Route B (`tt-hous`): Python with `ttnn.experimental.kda.*` ops. | Plain Python `ttnn` ops emitted by `orchard/port_template.py`. |
| Routed experts | One fused op, `ttnn.experimental.moe_compute`, with BF4 experts packed by a ring-aware packer, expert-parallel over four chips, and the router, sort and combine on the device. | A loop over expert slots built on the host. The router's choices go back to the host every layer. |
| Gated delta net, gated residual, sparse attention | Fused programs (`gdn_step`, `gr_read`, `qsa_block` and others) and the DeepSeek-style sparse ops. | Sequences of small plain ops. |
| Decode | MTP speculative decoding with 3 to 5 drafts, on-device sampling that reads back 8 scalars, a KV cache and a recurrent state. | `decode_step` appends the token and runs the whole sequence again with no cache. |
| Experts at load | Converted to BF4 once and cached (65 to 69 GB). | Streamed one tensor at a time, uploaded in the chosen format. |
| Correctness gate | Checkpoint pinned by digest; acceptance replay against a CPU oracle; JSON 96 of 96. | Fidelity score, stage gates, chip rehearsal. |

## What the numbers say

1. **Op count.** `scripts/port_op_count.py` counts the device ops of the generated tiny Qwen4 layers with the fake ttnn: 116 for a gated-delta-net layer, 140 for a sparse-attention layer, 192 for the layer with the PLE block. Taking Flash-Next's pattern (12 groups of three GDN layers and one attention layer), the mixers and gated residuals alone issue about 5,700 ops per forward, and the MoE blocks add about 3,700 at one token (10 hit slots on four chips) and about 34,500 at 32 tokens (117 hit slots). Probe 16 measured 8 to 14 µs for a small elementwise op and probe 17 about 70 µs for the 6 ops of one expert slot, about 12 µs per op. At 10 µs per op the floor is about 95 ms per token at one token. Upstream main without MTP takes 26 ms per token for the whole model. The count is for the uncached form the generator writes today, so a cached decode would change it; the ratio is large enough that the conclusion holds for any plausible cached form.
2. **The published speed spread.** Three packages run the same weights at the same context on a QB2 at 24, 59 and 74 tok/s. The faster ones use MTP, on-device sampling and a chat server.
3. **The fidelity bar.** Upstream reports a teacher-forced KL of 0.0528 and top-1 agreement of 0.9525 against the Hugging Face model with BF4 experts. By the formula of `scripts/fidelity_score.py` (10 times exp(-KL)) that is about 9.49, below the provisional bfp4 bar of 9.9. The measurements differ in prompts and method, so this shows only that the bar needs a look.

## Hypotheses, most likely first

1. The gap is mostly op count and launch floor. Fused programs for the mixers (about 5,700 ops) and `moe_compute` for the experts (about 3,700 to 34,500 ops) remove most of it. Test: count ops per layer in the published ports' code; time one fused GDN program against the plain-op layer on one chip.
2. `moe_compute` reaches a much higher share of DRAM bandwidth than a batched matmul. Our batched matmul read about 32 GB/s. Test: probe `moe_compute` at Flash widths on two and four chips. Risks: it needs a mesh, the forks carry patches that main lacks, and main limits it to 128 tokens per call.
3. The generated port has no cached decode, so no decode speed exists to compare yet. Until it does, every speed figure for the generated port is a projection.
4. Host round trips (router choices read back 48 times per token) cost a few milliseconds per token. Unmeasured.
5. MTP roughly doubles the speed of the published ports (24 to 59 or 74 tok/s with other differences also present). It needs the 5.2 GB MTP layer and exact acceptance.

## Suggestions

- Add a cached decode to the generated port before comparing speeds (issue filed with this note).
- Probe `moe_compute` on two and four chips, and read its call shape in upstream `moe.py`.
- Count ops of the published ports' layers the same way and compare with 116 and 140.
- Keep the 9.9 bar for now and record that a published BF4 port scores about 9.49 by this formula.
