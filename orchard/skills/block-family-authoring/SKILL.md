---
name: block-family-authoring
status: draft. For stage 2 of a full port when the generator leaves a block as a stub because its class is new (issue 707). Not run by a stage yet.
---
# Write a block family for the port generator

The port generator (`orchard/port_template.py`) writes a decoder port from the checkpoint's config, construction and tensor list. It writes the
blocks it knows. A block it does not know stays a stub, and a small coder rarely writes a stub well: every stub was a long run of ttnn mistakes
(see "Mistakes seen" at the end). Your job is smaller. You teach the generator one new block, once. You write three things:

1. a recognizer: reads one module of the construction and the tensor shapes, returns settings or `None`;
2. an emitter (and the helpers it calls): returns the Python lines of the block;
3. a test: runs the emitted block on the CPU and compares it with the transformers module at tiny sizes.

All of it runs on the CPU. Nothing in this job opens a chip. Never import the real `ttnn`: importing it opens the device. The tests use the
torch-backed stand-in in `tests/fake_ttnn_compute.py`.

Rules that decide success:

- The test comes first. A block without a passing comparison against the reference is a claim with no check.
- The recognizer returns `None` when it is not sure. A stub is a safe result. A wrong block is not.
- No value is invented. An epsilon, a width, a head count and every tensor name come from the construction, the config or the checkpoint.
- Every `ttnn.<name>` the emitted code calls is in the verified cheat sheet.
- One change at a time. Run the test file of your family after each change.

## The files

- `orchard/port_template.py`: `BlockFamily`, `Site`, `register_block_family`, `BLOCK_FAMILIES`, the four families, `FIXED_HELPERS`, `_compose_layer`.
- `tests/test_port_template_registry.py`: the hyper-connection family and its tests. Copy its shape.
- `tests/fake_ttnn_compute.py`: the fake ttnn. It computes with torch and tracks layouts.
- `orchard/skills/full-port-templates/ttnn_cheat_sheet.json`: the verified ops and their argument names.

## Steps

### 1. Read the reference, line by line

Find the class in the transformers modeling file of the model (the reference python of the run has it). Read `forward` from the first line to the
return. Write down, as a list in your project log entry, every operation in order: each linear layer, norm, activation, reshape, slice,
concatenation, reduction, convolution, gather and any loop. Write the shape after each. Note every tensor the module owns, with its name in the
checkpoint. A bare parameter (`A_log`, `dt_bias`) has no `.weight` suffix.

### 2. Decide what runs where

- Linear layers, norms, activations, matmuls, softmax and element-wise math run on the device.
- A table that is large (the real per-layer embedding table is over 100 GB) stays on the host. The block looks rows up on the host and uploads the
  rows it needs. The load role `host` keeps a tensor in `model["host"]`.
- A tensor that needs a change at load has a load role: `neg_exp` uploads minus exp of the tensor, `conv_taps` uploads each convolution tap as
  its own device tensor, `norm_plus_one` adds one to a norm weight, `tile` uploads a bare parameter in tile layout. The recognizer returns the roles
  in a `placements` dict (tensor name to role).
- Prefer an op the sheet already holds. If an op failed on a board before (see the table below), use the form that worked.

### 3. Write the comparison test first

Copy `test_generated_hyper_connections_match_the_transformers_module` from `tests/test_port_template_registry.py`. It does this:

```python
text, _p = tiny_qwen4()                                  # generate the port for the tiny checkpoint
mod = import_port(monkeypatch, text, tmp_path)           # import it with the fake ttnn installed
model = mod.load(str(snap), fake.MeshDevice(1))          # load random bf16 weights through the generated load
ref = _hf_gated_residual(weights, prefix, True)          # the real transformers module with the same weights
mixed, hyper_input, inj = fn(model, layer, x, ids)       # the generated function on the same input
close(mixed, want_mixed, prefix + " mixed")              # max abs error within 3 percent of the output scale
```

Rules for the test:

- Use tiny sizes (hidden 64, head width 32) and random weights with a fixed seed.
- Check that the reference output is not trivial. If the reference is all zeros or all ones, any block passes. Assert its maximum is large
  enough to mean something.
- Compare every output the function returns.
- Name the test in the family's `reference_test` field as `path::test_name`. `register_block_family` refuses an entry without it.
- The test must fail when the block is wrong. Step 6 shows how to prove that.

### 4. Write the recognizer

`recognize(site)` gets a `Site`: the module path, its class, its attributes, its children, the tensors by name pattern, and `where` (`layer` or
`model`). It returns a settings dict or `None`. Refuse (return `None`) when any of these differ from what the emitter assumes:

- the child set (an extra or a missing child module);
- an attribute that is not a whole number, or a width that does not divide into tiles (a slice must start on a multiple of 32);
- a head count that does not divide as the reference does (value heads over key heads);
- an activation the emitter does not write;
- the shape of any tensor the block reads;
- an epsilon that the construction does not give. Use `_eps(attrs, -1.0)` and refuse when the result is not above zero. No default is invented;
- the class name of a norm (a gated norm is not a plain RMS norm).

Tensor names are checked by the generator before your code sees them: a name must be a dotted identifier chain of at most 200 characters, so a
site path cannot carry a quote into the emitted source. Build every name from `site.path`. Never type a tensor name that you did not read.

### 5. Write the emitter and the helpers

`emit(function, site_path, settings, where)` returns the lines of one function. Rules:

- The first comment line inside the function is `GENERATED_BLOCK_MARK`, followed by the family name. The thin repair never sends a marked
  function to the coder as a repair target.
- Put repeated code in `helpers()`, which returns module-level lines written once. Helpers carry the mark too.
- Use only ops in the cheat sheet. After writing, run `sheet_problems(text)` on the generated port. It lists every `ttnn.<name>` the sheet does
  not cover. The list must be empty.
- Put numbers into the emitted text with `repr`, and tensor names built from `site_path`. Do not put free text from the construction into code.
- Say in the docstring of the function what it takes and returns, with shapes. Activations are `[1, 1, S, features]`; heads are
  `[1, heads, S, head_dim]`.
- Write a `describe(function, settings)` that returns one line: the call and what it returns. The layer stub shows it to the coder.
- A helper that needs a layout raises a named error when it gets another one. A device error such as `Invalid subtile broadcast type` costs a
  board check and says nothing about the cause. `_apply_rotary` is the model.

Register the family:

```python
register_block_family(BlockFamily(
    name="hyper_connection", recognize=_recognize_hyper_connection, emit=_emit_hyper_connection,
    reference_test="tests/test_port_template_registry.py::test_generated_hyper_connections_match_the_transformers_module",
    describe=_describe_hyper_connection, helpers=_hc_helpers, prelude=_emit_hc_prelude, model_level=True))
```

If a layer is made only of generated blocks, `_compose_layer` writes the layer function. Add a layer kind there only with its own comparison test
against the transformers decoder layer.

### 6. Prove each guard with a mutation

A test that has never been seen to fail has not been shown to check anything.

- Emitter: change one line of the emitted text (drop a factor, swap a sign, drop a term). The comparison must fail. The registry test does this
  with `test_the_reference_comparison_fails_when_a_stream_weight_is_dropped`: it replaces the text, reruns the comparison and expects an
  `AssertionError`. Assert that the replacement applied (`assert broken != text`) so the mutation cannot pass by doing nothing.
- Recognizer: remove each guard once and run the refusal tests. One must go red. A guard whose removal leaves every test green has no test.
  Add a case with consistent shapes for it, then check again. A divisibility guard stayed green once because the test shapes did not divide.
- Restore the file after each mutation, and run the file again to see it green.

### 7. What the CPU test cannot show

The fake ttnn computes with torch. It shows that your code asks for the right operations in the right order with the right shapes. It does not
show that a device op accepts those arguments. These were open after each family and each needed a board check:

- a width-1 `ttnn.slice` and a broadcast multiply (hyper-connection injection);
- `ttnn.softplus`, a row-major `ttnn.concat` and the head counts of the delta rule op (gated delta net);
- `ttnn.sum`, `abs`, `sign`, `sqrt` and `clamp` (PLE gate);
- a `[1, 1, S, head_dim]` table multiplied over the heads axis, and `ttnn.linear` on a `[1, 1, heads * S, head_dim]` input (rotary).

List the ops of your family that the fake does by plain torch, and write them in the project log entry under "Not shown by the CPU test". The
board check is not yours. It runs under a lease, for example `gozer run --chips 1 --who <you> --reason <why> -- python <probe>`. Never run a
device command without a lease, and never import `ttnn` in a test.

### 8. Finish

1. Run the plan for the tiny Qwen4 checkpoint and count stubs (`stub_lines` on the generated text). Update the stub counts and stub sets in older
   tests that assert them. Run `tests/test_port_template.py` and every family test file.
2. Run `check_sources` on the generated port: it must parse and hold no stub of your family.
3. Add a change fragment with `python3 scripts/version.py new <slug>`. Never edit a version field.
4. Add an entry at the end of `docs/project-log.md` in plain English: what the family writes, the tests and mutations, and what the CPU test
   cannot show.
5. Run `python3 scripts/version.py check`.
6. Leave the merge to the operator.

## The four families so far

| Family | On the device | On the host | Hazard found |
|---|---|---|---|
| hyper-connection (`_hc_helpers`, prelude, mixer) | grouped RMS norm by reshape, gate and mix by `_linear`, injection by slices and a concat | none | the norm weight is narrower than the stream, so a plain `_rms_norm` fails on padded widths; `_rms_norm` has no eps for this norm and raises a named error (it was a bare `KeyError` before) |
| gated delta net | fused projection, shifted multiply-add convolution, decay and beta, `ttnn.transformer.chunk_gated_delta_rule`, gated norm | none | the tensor names (`in_proj_qkv` and others) were guessed wrong by the coder; `A_log` and `dt_bias` have no `.weight`; `ttnn.conv1d` was wrong for a dilated convolution, so taps are shifted multiply-adds |
| PLE (`_recognize_ple`, `_emit_ple`) and layer composition (`_compose_layer`) | projections, group norms, gate, dilated convolution as shifted multiply-adds | n-gram hashing and the table lookup; the table stays in `model["host"]` | the real table is over 100 GB, so it is never uploaded; the PLE output is added to the streams before the attention connection |
| rotary helpers (`_rope_tables`, `_apply_rotary`) | `x * cos + (x @ R) * sin` with R a constant rotate-half matrix | the tables are built in torch and uploaded | slice, concat and unary minus failed on a board for partial rotary; the tables hold 1 and 0 in the pass-through columns so none is needed; a wrong layout raises a named `ValueError` |

## Mistakes seen from the small coder

From the failure lines of the tiny Qwen4 arms (144 distinct lines over 18 arm directories; a line repeated inside one arm counts once; 39 did not
fit a category). Each line is the symptom and the fix.

1. Wrong tensor name or key in `model["dev"]` or `model["host"]` (19): `KeyError: model.layers.0.linear_attn.in_proj.weight`. Read the names from the
   checkpoint's tensor list. A bare parameter has no `.weight`.
2. A key of `CONFIG` or `model` that does not exist (17): `model.config`, `head_k_dim`, `num_key_value_groups`, `mesh_device` in `model["dev"]`,
   `cos_cached`. `model` is a dict whose keys are `mesh_device`, `n_devices`, `dev`, `host`, `ids`, `ck`. Read config values from `CONFIG` and use
   only its keys.
3. A matmul or reshape with the wrong shape (15): `a_shape[-1] == b_shape[-2]`, `new_volume == old_volume`. Print the shape before the call. An
   activation is `[1, 1, S, F]`; heads are `[1, heads, S, head_dim]`.
4. A ttnn tensor used like a torch tensor (11): `.view`, `.t`, `torch.chunk`, `arange(device=...)`. Use `ttnn.reshape`, `ttnn.permute`, `ttnn.slice`,
   or read back with `_to_host(model, t)` and use torch.
5. A pattern used as a key (9): `KeyError: model.layers.{i}.attn_hyper_connection.hc_norm`. Fill in the layer number. `_pattern(name)` goes
   the other way and is only for the placement table.
6. A stub left unwritten, or a layer that calls back into `LAYERS` (9): `NotImplementedError`, `RecursionError`. Write the function the layer calls. Do
   not call `LAYERS` from inside a layer.
7. A norm weight of another width than the input (8): `padded_shape()[-1] == gamma`. Reshape to `[..., groups, width]`, norm over the last
   dimension, reshape back.
8. A relative import or a module that does not exist (8): `from .helpers import`, `port_helpers`. The port is one file. Import `ttnn`, `torch`, `math`
   and the standard modules, and call the helpers that are in the same file.
9. A helper called with the wrong arguments (7): `_to_host() missing 1 required positional argument`. Every helper takes `model` first. Read the
   signature list in the request.
10. A broadcast or layout the op does not take (2 lines, 5 board checks): `Invalid subtile broadcast type`. Give both operands the same layout and
    the same rank, and check the shape of `x` before the op.

Also seen: replies cut off at the token cap (about one request in five in the thin port), and a reply with two code blocks. Write one function per
reply and keep it short.

```new-names
_recognize_my_block
_emit_my_block
_my_block_helpers
_describe_my_block
```

```other-names
placements
reference_test
neg_exp
conv_taps
norm_plus_one
tile
host
site_path
dt_bias
in_proj_qkv
mesh_device
n_devices
head_k_dim
```

```quoted-names
cos_cached
num_key_value_groups
port_helpers
```
