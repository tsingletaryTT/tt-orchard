# 2026-10-06: bringup spike (Phase 1)

Questions and answers. Every number is one measurement unless it says otherwise.

## (a) Does Coder-Next boot on one board?

Yes, on the `p300` profile (one board, 2 dies, bfp4 routed experts).

- Artifact: `raahemnabeel/qwen3-coder-next-blackhole`, a v5.1 container package, image `bd0ca927a46b`,
  weights `Qwen/Qwen3-Coder-Next@a7fbcb5c` (159 GB on disk, download finished, no `.incomplete` blobs).
- Profiles: default `p300x2` (4 dies, TP4, takes both boards), `p300` (1 board), `p150x4`, `p150x2`
  (the last two marked untested by the author). Only `p300` was run here.
- Provenance: built from a tt-metal checkout that was dirty and unpushed (branch `qwen3_coder_next`,
  `dirty: true`). The package cannot be rebuilt from public sources.
- Cold boot to a ready server: 681 s (`gozer acquire --owner-pid`, then `tt-model serve --profile p300
  --port 20000 --local-only --detach`).
- Canary "What is 7 times 6?": `7 times 6 is **42**.`, finish `stop`, 0.75 s.
- Tool call (one `shell` tool, prompt asking for a directory listing): a parsable call
  `{"command": "ls -la"}` on the first try, 24 tokens, 1.0 s. Server uses the `qwen3_coder` tool parser
  and has no reasoning parser.
- Decode: 512 tokens in 13.1 s, 39.1 tok/s (one prompt of 21 tokens, includes its prefill).
- Prefill: 7,016 tokens in 2.0 s, about 3,560 tok/s (one prompt, repeated text).
- Not measured: 32K context, concurrent requests, the `p300x2` profile, memory use, the replay of
  recorded agent turns, the replay of turns that failed the 27B. Those are steps 2 to 4 of the
  role-fit test in the spec and are still open.
- Shutdown: container stopped, `gozer release febcc9` reset chips 0 and 1, all four chips `FREE`.

## (b) Can hidden states leave the TT stack?

A subagent read the source (reading only, nothing was run). Its verdict is (b): not exposed today, and a
small change to the model code would expose them. I have not read these files myself.

- The vLLM TT plugin has no pooling route: `model_runner.py:323-325` returns an empty list of
  supported pooling tasks, and `input_batch.py:338` asserts that pooling requests are unsupported.
- The Qwen3.8 model code is `models/demos/blackhole/qwen36` (not `qwen3_5`). Its prefill keeps only the last
  row, then applies the final norm and the lm head (`model.py` around lines 579-634). Per-token hidden
  states of shape [T, 5120] are never formed there.
- The DFlash2 package has a tap for residuals after five layers (`_dflash_tap_layers`), which does not
  include the final norm output.
- No script compares final hidden states with Hugging Face. Existing tests compare logits and single ops.

Consequence for the spec: Clef's parity gate needs a small change in the serve-and-compare template or
the model (a `return_hidden` path), or a direct call of the generator. This is the largest open item for
Phase 3. If it cannot be done, Clef ends `blocked` with `unsupported-input-type`, as the spec says.
Next step: read `qwen36/tt/model.py` and the package copy to confirm the cited lines before any design
depends on them.

## (c) Does the triage rule classify Clef as expected?

Checked from the Hub's `model.safetensors.index.json`, with no download and without running `delta_triage.py`:

- Clef backbone tensors: 1,184. Qwen3.8-27B (local copy): 1,199.
- In Clef and not in Qwen3.8-27B: 0. In Qwen3.8-27B and not in Clef: 15, all named `mtp.*`.
- The triage rule already lets `mtp.*` tensors differ. So the backbone should classify `weights-only`.
- Sidecar `joint_head.safetensors`: 122 tensors, all BF16, names start `evidence_layers.*`, 256 MB. It is not
  in the backbone index. Nothing in the current triage knows about it.
- Not checked: text config values (needs `config.json` in the same comparison), tokenizer, vision tensor
  shapes, license text.
- Side effect to record: with `mtp.*` dropped, any package that depends on the nearest model's MTP
  speculative decoding (the DFlash2 packages are a different mechanism) has to be checked separately.

## (d) The stale leases

`gozer status` showed all four chips `STALE`, held by `claude:tt-serve-llm` pid 1329153. That pid did not
exist. `gozer reconcile` (the documented fix for `STALE`) freed all four. Nothing was reset by hand.
Note: while the Coder-Next container ran, its board showed as held by my lease; the `HELD-FOREIGN` state
documented for `tt-model serve` containers did not need a decision.

## Corrections to docs/analysis

- "Runs on 2 chips" is true only for the `p300` profile with bfp4 routed experts. The 4-die precision policy
  is the default profile. Whether bfp4 hurts tool-call reliability over a long run is not measured.
- The success-rate tables and the arbiter's capability scores are still unmeasured.
