# Bundle capabilities (`orchard-capabilities.json`)

A tt-model v6 bundle normally serves one model: the model in its manifest's `weights.repo_id`. Stage 0 compares a new model with that
model, and any difference in the text config values that decide the path (layer count, widths, head counts, layer types, vocabulary),
any tensor shape difference and any tensor only one model has makes the run a full port.

Some bundles carry model code that reads the model's configuration at run time and handles a family of models. The Qwen4 bundle
(`mando2222/qwen4-27b-dense-v6-thin`) is the first: its development weights are a 20-layer test checkpoint, and the release it is
built for has 64 layers and may differ in its key-value heads, feed-forward width and other values. Such a bundle ships
`orchard-capabilities.json` in its directory. The file says which differences its code handles, and stage 0 sets aside exactly
those.

## Where it is read

- Stage 0 (`orchard/skills/delta-triage-templates/delta_triage.py`) reads it when `triage_config.json` names `nearest_bundle_dir`.
  The stage 0 skill fills that key from the run's `bundle` input, which the bring-up command records for the bundle the run serves
  the new weights with.
- Stage 0's gate (`orchard/bundle_capabilities.py`, called by the supervisor) fails when the run's bundle ships the file and
  `delta.json` does not record that it was applied with the file's current sha256.
- Stage 2 (`prepare_swap.py`) serves the new model's own `config.json` with only `architectures` replaced by the nearest model's,
  when the bundle ships the file. Without it, stage 2 serves the nearest model's config as before. Stage 7 packages the config
  stage 2 served (`base_config/config.json`), so the package serves the same file.

A bundle without the file behaves as before in every stage.

## The file

```json
{
  "schema": 1,
  "config_keys": {"<text config key>": <rule>, "...": "..."},
  "shapes": [{"tensor": "<regular expression>", "dims": {"<dimension index>": "<expression>"}}],
  "layers": {"pattern": "^model\\.layers\\.(?P<layer>\\d+)\\..+$", "count_key": "num_hidden_layers", "types_key": "layer_types"},
  "optional_tensors": [{"tensor": "<regular expression>", "feature": "<name>", "shape": ["<expression>", "..."]}],
  "features": {"...": "..."},
  "note": "<text>"
}
```

Every key other than `schema` is optional. An unknown key, a malformed rule, a pattern that does not compile, a link in place of the
file, a file larger than 256 KiB, a repeated JSON key or a NaN makes the file unusable, and stage 0 then calls the run a full port
and names the problem.

### Rules

A rule is an object with a `type`: `int`, `number`, `string`, `bool`, `list`, `object` or `any`. It may add:

| Key | Applies to | Meaning |
|---|---|---|
| `one_of` | any type | the value must equal one of the listed values (`true` is not `1`) |
| `min`, `max` | `int`, `number` | bounds, inclusive |
| `multiple_of` | `int` | a whole number above 0 that the value must divide by |
| `items` | `list` | a rule every entry must meet |
| `length` | `list` | a whole number, or a config key whose value (in the new model's config) the length must equal |
| `keys` | `object` | a rule per sub-key; a sub-key the rule names and the value lacks fails unless its rule has `absent_ok` |
| `other_keys` | `object` | `"any"` lets the value hold sub-keys `keys` does not name (the default refuses them) |
| `absent_ok` | any type | the new model may lack the key |

The rule is checked on the new model's value only. The nearest model's value is what the bundle was built with.

### Expressions

`shapes` dimensions and `optional_tensors` shapes are expressions: config keys and whole numbers joined by `*` and `+`, for example
`hc_count*hidden_size` or `indexer_n_heads*indexer_head_dim+indexer_head_dim`. They are evaluated on each model's text config
(`head_dim` falls back to `hidden_size / num_attention_heads`, as stage 0 does everywhere). A key that is missing or not a whole number
gives no value, and the difference then stays a full-port reason.

## What stage 0 sets aside

With the file applied, stage 0 sets aside:

1. A text config key that differs and that `config_keys` declares, when the new value meets the rule.
2. A shape difference of a tensor a `shapes` entry matches, when every differing dimension is declared in that entry and each model's
   size equals the expression on its own config. A change of rank is never set aside.
3. A tensor at a layer index only one model has, when `layers.count_key` is a declared key that differs, the index is below that
   model's own layer count and at or past the other model's, and the other model has the same tensor at a layer of the same type
   (from `types_key`, when both configs list their layer types) with the same dtype and an equal shape, or a shape a `shapes` entry
   explains.
4. A tensor an `optional_tensors` entry matches, in either model. When the entry gives a `shape`, the new model's tensor must have it.

Everything else stays a reason for a full port: an undeclared key, a value outside its rule, an extra or missing tensor no entry
covers, a shape difference along an undeclared dimension, any dtype difference beyond the small float tolerance stage 0 already has.
With the file applied, every differing text config key needs a declaration, including keys stage 0 otherwise ignores (such as
`rope_theta` or `rms_norm_eps`), because stage 2 serves the new model's own config and the bundle's code reads it. Bookkeeping keys
(`architectures`, `transformers_version`, `_name_or_path`) and the keys stage 0 already sets aside as written by a newer transformers
need no declaration. Declare a key the code ignores with `{"type": "any"}`.

`features` is recorded in `delta.json` and not judged.

## What `delta.json` records

```json
"capabilities": {"bundle_dir": "...", "file": "orchard-capabilities.json", "present": true, "applied": true, "sha256": "...",
                 "schema": 1, "problems": [],
                 "exempted": {"config_keys": [{"key": "num_hidden_layers", "model": "64", "base": "20"}],
                              "shapes": [{"tensor": "...", "model": [...], "base": [...]}], "shapes_count": 0,
                              "layer_tensors": {"count": 0, "examples": []},
                              "optional_tensors": {"count": 0, "features": {}, "examples": []}},
                 "features": {}}
```

The record appears only when `nearest_bundle_dir` was given. A bundle without the file gives `present: false` and changes nothing
else. A difference with area `other` says what was set aside, and a hazard says that the path rests on the bundle's declaration.

## An example for the Qwen4 bundle

This is a sketch. The bundle's author writes the real file from the keys the model code reads.

```json
{
  "schema": 1,
  "config_keys": {
    "num_hidden_layers": {"type": "int", "min": 4, "multiple_of": 4},
    "layer_types": {"type": "list", "length": "num_hidden_layers",
                    "items": {"type": "string", "one_of": ["linear_attention", "full_attention"]}},
    "num_key_value_heads": {"type": "int", "one_of": [2, 4]},
    "intermediate_size": {"type": "int", "multiple_of": 128},
    "output_gate_type": {"type": "string", "one_of": ["silu", "sigmoid"]},
    "max_position_embeddings": {"type": "int", "min": 2048},
    "rope_parameters": {"type": "object", "keys": {"rope_theta": {"type": "number", "min": 1},
                                                   "rope_type": {"type": "string", "one_of": ["default"]},
                                                   "partial_rotary_factor": {"type": "number", "min": 0, "max": 1}}},
    "dtype": {"type": "any"}
  },
  "shapes": [
    {"tensor": "^model\\.layers\\.\\d+\\.mlp\\.(gate|up)_proj\\.weight$", "dims": {"0": "intermediate_size"}},
    {"tensor": "^model\\.layers\\.\\d+\\.mlp\\.down_proj\\.weight$", "dims": {"1": "intermediate_size"}},
    {"tensor": "^model\\.layers\\.\\d+\\.self_attn\\.[kv]_proj\\.weight$", "dims": {"0": "num_key_value_heads*head_dim"}}
  ],
  "layers": {"pattern": "^model\\.layers\\.(?P<layer>\\d+)\\..+$", "count_key": "num_hidden_layers", "types_key": "layer_types"},
  "optional_tensors": [{"tensor": "^model\\.layers\\.\\d+\\.self_attn\\.indexer\\..+$", "feature": "qsa_indexer"}],
  "features": {"drafter": "off", "mtp": false, "vision": false}
}
```
