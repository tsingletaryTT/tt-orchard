# Cached decode for the generated port of a standard decoder (issue 805)

Branch `generic-cached-decode`, on `origin/main`. Closes #805 (part of #759). Version fragment `changes/generic-cached-decode.toml`.

## What it does

The generated port of a standard decoder (dense attention with grouped keys, per-head norm, sliding-window layers) decoded by appending the token and running the whole sequence
again. The Aion-1.0-Instruct-2.3B port of run 3 decodes at 8.3 tokens per second, 1.9% of its estimated floor. PR 762 had a cache only for the composed Qwen4 layers.

For a plan whose layers are all the standard pre-norm attention and MLP and whose attention the generator writes (`Plan.attention_cached_decode`; no stub, no step before the
first layer, no extra child of the text model, no block family), the generated text now holds:

- `_attention_step`: one row through the attention from kept keys and values. Projections, per-head norm, rotary at the row's own position, the new key and value appended, a
  sliding layer trimmed to its last `window` rows, SDPA of the query row (padded to 32 rows) against all kept rows (padded to a multiple of 32 with an additive mask).
- `_rope_row` and `_rotate_row`: the rotary of one position per layer type, with the same scaling factor the prefill tables carry (Aion: YaRN on the full layers).
- In `_attention`: the prompt keeps the keys after the rotary and the values of its n real rows (a sliding layer: the last `window`), and a step goes to `_attention_step`.
- The cached `prefill` and `decode_step` of PR 762. A step at another position than the next one runs the prompt again. `close` frees the kept tensors.

The whole-sequence run is unchanged. It is the reference of every test and it is what any other generated port still does.

## Evidence

- Tests, written first (seen failing for the missing step function): `tests/test_port_template_generic_decode.py`, 27 tests on the fake ttnn.
  - An Aion-like config (window 8, YaRN original 16, two sliding layers and one full, four query heads on two key heads): 20 steps after a 20-token prompt match the
    whole-sequence run (within 5% of the largest logit, same argmax) and a plain torch forward (2e-3); the sliding layers hold exactly 8 rows and the full layer every row.
  - A plain Qwen-like config, a two-chip mesh, a rewind, a new prompt after a decode; no step calls `_forward`; `close` frees the kept tensors.
  - 16 mutations, each red for the intended reason (the failure message of each was read): window one row too long or too short, rotary at position 0 or at the last row,
    new key or value put before the cached ones, a new row not kept, padding rows kept, keys kept from before the rotary, a sliding layer keeping the first rows of the prompt,
    padding columns not masked, key heads taken as many as the query heads, either per-head norm skipped, the position not advanced, the masks of earlier positions never freed.
- Targeted run of 67 test files (the generator, the stub prompts, the thin repair, the gate, the hardening ratchet, the port checks, the full-port stages): 2,241 passed, 2 skipped. The full suite was not run.
- Two golden pins moved: the hash of the standard small decoder's generated text (two test files). The tiny Qwen4 hash did not move, so the composed layers are byte for byte as before.

## Hardening checklist

1. Wrong type or shape in input: n/a. The generator reads no new outside input; the plan fields it uses are the ones the attention block already validates.
2. Files that something else controls: n/a. No file access.
3. Check, then use: n/a.
4. Evidence an agent can write: n/a. The new functions are marked `# GENERATED (not a stub)`, so a failure inside them is never a repair target (`test_the_step_functions_are_never_a_repair_target`).
5. A verdict stronger than its evidence: the claim is "matches the whole-sequence run on the CPU stand-in". Nothing here says it runs on a board.
6. The check looks at a different object than the action uses: the comparison runs the generated text, the text a board would run, against the same text without a cache and against a torch forward.
7. A matcher too broad or too narrow: `attention_cached_decode` is False for any stub, prelude, epilogue, model block or composed layer (tested with a stub plan: the port keeps the whole-sequence decode).
8. Failure paths that leave state behind: replaced state tensors, padded copies and the rope row tables are freed; `close` frees the kept state (`test_close_frees_the_cached_tensors`).
9. Numbers and sizes: windows of 1 to 40 in the earlier tests of the window; this change uses the window only to trim. The cache grows by one row per step for a full layer (no cap here).
10. Secrets, paths and host names: none.
11. Tests that do not test the claim: 16 mutations above; one mutation first survived because a sliding layer that keeps the whole prompt is trimmed on the first step anyway, so a row-count check after the prompt was added (it fails that mutation). The config uses four query heads on two key heads, because the first config (two on one) let a wrong key-head count pass on the fake.
12. Housekeeping: change fragment, project-log entry, README and CLAUDE.md sentences about the cached decode corrected, `scripts/version.py check`.

## What this does not show

- No board has run it. The ops (`concat`, `slice`, `permute`, an additive mask on SDPA with the query padded to 32 rows) are those the Qwen4 cache already uses, but the fake accepted a reshape of the wrong size in one run, so the fake does not prove the shapes are legal on a board.
- No speed was measured, and the op counts say a speedup is not certain. On the fake, one step of the three-layer test port issues 198 ops (about 66 per layer: 37 reshapes, 28 linears, 15 concats, 13 norms) and one whole-sequence pass issues 83 (about 28 per layer). Where the device is bound by the launch of each op (a short context, eager mode), the step can cost more per token than the whole-sequence pass; it wins when the whole-sequence pass becomes compute-bound by the number of rows, so the gain should grow with the context length (about 8,000 rows at the passkey length) and may be small or negative at a few hundred tokens. I do not give a speed figure. The Aion port decodes at 8.3 tokens per second today, 1.9% of the estimated floor, and the measurement on a board is what decides.
- Cheap savings already taken: the zero padding tensors and the step masks are uploaded once and shared by the layers (from_torch calls per step fell from 14 to 8 on the test port). The remaining ops are mostly the head split and join reshapes and the cache concats. A preallocated, padded cache updated in place, and the on-device head split of `nlp_create_qkv_heads`, would remove most of them; both need a board to check the shapes.
- The kept keys and values live in DRAM and the step concatenates each time (a copy of the cache per layer per step). A preallocated cache with an in-place update is the next step if the concat shows up in a profile.

## Chip run that closes it

The Aion port of `aion-run3` on a leased board with this decode: decode tokens per second over three runs (median and spread, against the 8.3 of the whole-sequence run), then top-1 agreement over the stage 2 reference and the 8,192-token passkey check, neither of which may change.
