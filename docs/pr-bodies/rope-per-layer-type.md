# The port generator follows layer-type rope blocks and the sliding window (issue 790)

Pull request body. Branch `rope-per-layer-type`, on `origin/main`. Fixes #790. Version fragment `changes/rope-per-layer-type.toml`.

## What it does

`microsoft/Aion-1.0-Instruct-2.3B` gives `rope_parameters` keyed by layer type (`full_attention`: yarn, factor 16, original length 2,048, theta 32000;
`sliding_attention`: the default type, theta 32000) and `sliding_window` 2048 for 21 of its 28 layers. The generator took that dict as one flat block. It
found no `rope_type` and no `rope_theta`, wrote the default type with theta 10000.0, and wrote causal attention over the whole prefix in every layer. The
port passed stages 0 to 4 with top-1 0.875 and fidelity 9.249 (the Olmo-Hybrid-7B port gave 1.0 and 9.976) and failed the stage 5 passkey check at 8,192
tokens. Nothing in the run said the generator had not understood the config.

Three commits, in the order the issue asked for.

1. **Refusal.** A layer-keyed `rope_parameters` or a sliding window leaves the attention block as a named stub, and the rotary helpers are no longer
   written with the fallback theta.
2. **Per-layer-type tables.** The plan checks each layer type's block and the generated port writes `_rope_inv_freq` (the default type and yarn, as
   `_compute_yarn_parameters` computes them) and a `_rope_tables` that returns a dict of cos and sin per layer type. `_attention` picks its pair from the
   layer index in its prefix.
3. **Sliding window.** `_window_mask` builds an additive bf16 mask (0 for the query and the window minus one positions before it, -30000 elsewhere, the
   convention of the other generated masks) once per padded length. A sliding layer whose padded sequence is longer than the window passes it with
   `is_causal=False`. Every other layer, and a sliding layer inside the window, keeps the causal call. `close` frees the mask. The window also works with
   a flat rope block.

A config the generator cannot follow still leaves a named stub, and the reason names the layer type and the setting.

## The twelve items of docs/pr-hardening-checklist.md

1. **Wrong type or shape in input.** The rope blocks, `layer_types` and `sliding_window` are values from a model's config. `_layer_rope_plan` reads
   numbers through `_float_of` (`safeio.fits_float`) and `_whole`, and checks `layer_types` as a list of strings as long as the layer count.
   `test_a_hostile_required_value...`, `..._optional_value...`, `..._partial_rotary_factor...` and `..._rope_type_or_layer_type_name...` run `[]`, `{}`,
   `""`, a string with a NUL, `10**400`, `-10**400`, `True`, `nan`, `inf`, `-1`, `0` and `None` through every field; each gives a stub that names the
   setting and never an exception.
2. **Files that something else controls.** n/a. No file is read or written by the new code; it works on a parsed config and tensor headers. The test
   fixture `tests/fixtures/aion-config.json` is a copy of the model's public `config.json`.
3. **Check, then use.** n/a. The plan is a pure function of the config and the headers, and `ATTENTION` is written once into the generated text.
4. **Evidence that an agent can write.** n/a. No evidence file is added or read.
5. **A verdict stronger than its evidence.** A stub reason says what the generator did not follow. The log entry and this body say that the mask and the
   long-sequence call have run on the CPU stand-in only and that no reference run of Aion's passkey check exists.
6. **The check looks at a different object than the action uses.** The comparison runs the generated port text on the CPU stand-in, the text a board
   would run, against a torch forward written from the Aion modeling code. The rotary tables are compared with transformers' own
   `ROPE_INIT_FUNCTIONS['yarn']`. The wiring tests spy on the `scaled_dot_product_attention` call of each layer.
7. **A parser or matcher that is too broad or too narrow.** `rope_keyed_by_layer_type` counts a block as layer-keyed only when every value is a dict;
   tests cover a flat block, an empty block and a flat block with one nested dict. `sliding_window_of` counts a window only when a layer is marked
   `sliding_attention`; tests cover no marker, a null window and a missing window.
8. **Failure paths that leave state behind.** The mask is recorded in `model["window_mask"]` after it is uploaded. A new padded length frees the old
   mask, and `close` frees the last one (`test_a_new_padded_length_frees_the_previous_window_mask`, `test_close_frees_the_window_mask`).
9. **Numbers and sizes.** Windows of `0`, `-3`, `1`, `2`, `31`, `40`, `64`, `10**20`, `10**400`, `inf` and `nan` are covered. A window above the padded
   length uses the causal call, so the n by n mask is built only when it matters (134 MB in bf16 at 8,192). Numbers from the config go through
   `safeio.fits_float` before any conversion: a rope theta of `10**400` gave an `OverflowError` in the first version of this change, and the finite ratchet
   found it.
10. **Secrets, paths and host names.** A config value is never echoed as it came into a stub reason: `_shown` prints a short plain name or a small number,
    and a type word for anything else. Before this, a `sliding_window` of `'"""\nraise SystemExit(7)\n"""'` ended the stub's docstring and the generated port
    did not parse. `test_a_hostile_string_for_sliding_window...`, `..._rope_type...` and `..._layer_type_name...` check that the text parses and holds no
    hostile text.
11. **Tests that do not test the claim.** 44 mutations of the final code, each removing or bending one guard, each turn a test red (the table is below). The
    driver deletes cached bytecode: a mutation of the same length, restored within a second, left a stale `.pyc` and made a mutation look like it survived. The
    tables are checked at 2,048, 4,096 and 8,192 positions with head sizes 16 and 128, where the claim is about scale. The comparison can tell the cases
    apart (`test_the_one_table...` and `test_the_comparison_can_tell_a_window_from_none`). No clock decides a result.
12. **Housekeeping.** The change fragment, a dated entry at the end of `docs/project-log.md`, and `python3 scripts/version.py check`. No new flag. The
    generated text of the standard small decoder and of the tiny Qwen4 model is byte for byte what it was (hashes pinned in the test file).

## Evidence

- Tests: `tests/test_port_template_layer_rope.py`, written first and seen failing for the right reason (at the first run the generator wrote theta 10000.0 and
  a port with no window, and the later tests failed on a stub).
- A tiny port against a torch forward at 48 tokens, above the yarn original length of 16 and above a window of 8, on one and two chips, through two decode
  steps; and windows of 1, 2, 31, 40 and 64 at 40 tokens.
- Mutations (all caught): 3 on the refusals and the matchers, 13 on the tables, 13 on the window, 3 on the generated mark, 6 on hostile values and 6 on echoed
  values.

## What this does not show

- No board has run a generated port with a window. The board's `scaled_dot_product_attention` with an explicit mask at 8,192 tokens (24 heads, 8 key-value
  heads, head size 128 for Aion) is unmeasured for time and memory. The CPU stand-in accepts `sliding_window_size` and the cheat sheet lists it; this change
  does not use it.
- Decode runs the whole sequence again at each step, so a long prompt costs a full pass per token.
- No reference run of Aion's passkey check exists. A failure at 8,192 tokens after this change could still be the model.

## What an Aion rerun on a board should show

With the tables and the window in place the stage 2 record should move toward the Olmo numbers (top-1 near 1.0, fidelity near 9.9), and the stage 5 passkey
check should find the key at 8,192 tokens if the model can.
