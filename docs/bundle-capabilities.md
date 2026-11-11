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

- The release-day check (`tt-orchard drop-check`, `orchard/drop_check.py`) judges a release's config with
  `orchard.bundle_capabilities.config_reasons(capabilities, release_config, dev_config)`, which runs the stage 0 template's own code
  (`capability_config_reasons`). It takes plain dicts, needs no run directory, and returns named reasons (none when the file covers the
  config). Without `dev_config` it checks the rules, `model_types` and `architectures`; with the config of the bundle's own weights it
  also requires every differing key to be covered, as stage 0 does. The tensor rules need the safetensors headers and are left to stage 0.

A bundle without the file behaves as before in every stage.

## The file

Schema 1 is shared with the release-day check (`tt-orchard drop-check`, `orchard/drop_check.py`). The fields both read come first; the
fields after them are read by stage 0 only, and the release-day check ignores them.

```json
{
  "schema": 1,
  "bundle": "<org>/<name>",
  "model_types": ["..."],
  "architectures": ["..."],
  "keys": {"<text config key or dotted path>": {"equals": "v"}, "...": {"one_of": []}, "...": {"multiple_of": 128},
           "...": {"min": 1, "max": 8}},
  "features": {"indexer": "off | exact-up-to-2051 | full", "mtp": "ignored", "vision": "off"},

  "ignored_keys": ["..."],
  "optional_keys": ["..."],
  "layers": {"pattern": "^model\\.layers\\.(?P<layer>\\d+)\\..+$", "count_key": "num_hidden_layers", "types_key": "layer_types",
             "type_values": ["linear_attention", "full_attention"]},
  "shapes": [{"tensor": "<regular expression>", "dims": {"<dimension index>": "<expression>"}}],
  "optional_tensors": [{"tensor": "<regular expression>", "feature": "<name>", "shape": ["<expression>"]}],
  "note": "<text>"
}
```

`schema`, `model_types`, `architectures` and `keys` are required (the release-day check requires them). Stage 0 also refuses an
unknown top-level field, a malformed rule, a pattern that does not compile, a link in place of the file, a file larger than 256 KiB,
a repeated JSON key and a NaN; it then calls the run a full port and names the problem.

A tensor pattern (`layers.pattern`, `shapes[].tensor`, `optional_tensors[].tensor`) is matched against every tensor name of both models, so stage 0 also
refuses a pattern that can make the match backtrack for a very long time (issue 671). A pattern may use literal text, `\d+`, a character class such as
`[kv]`, an alternation such as `(gate|up)` that is not under a quantifier, and at most two unbounded quantifiers (`*`, `+`, `{n,}`) that are not next to
each other and not inside another quantifier. A quantifier inside a quantifier (`(a+)+`), an alternation under a quantifier (`(a|aa)+`), a lookaround and a
back-reference are refused, and the problem names the pattern. A tensor name over 512 characters takes no part in a match.

### Rules

A rule in `keys` holds one or more of `equals`, `one_of`, `multiple_of`, `min` and `max`, and the new value must meet every one it
holds. `equals` and `one_of` compare JSON values and keep `true` apart from `1`; `multiple_of`, `min` and `max` need a number. A key may
be a dotted path into an object value, for example `rope_parameters.rope_theta`. The rule is checked on the new model's value whenever
the new model has the key, whether or not it differs from the nearest model's, as the release-day check does.

`ignored_keys` names keys the code does not read: any value, or none, is accepted. `optional_keys` names keys the new model may leave
out; a rule in `keys` still applies when the model has the key.

### Expressions

`shapes` dimensions and `optional_tensors` shapes are expressions: config keys and whole numbers joined by `*` and `+`, for example
`hc_count*hidden_size` or `indexer_n_heads*indexer_head_dim+indexer_head_dim`. They are evaluated on each model's text config
(`head_dim` falls back to `hidden_size / num_attention_heads`, as stage 0 does everywhere). A key that is missing or not a whole number
gives no value, and the difference then stays a full-port reason.

## What stage 0 checks and sets aside

With the file applied, the new model's model type (the top-level `model_type`, else the text config's) must be one of `model_types`, each
of its `architectures` must be one of the file's, and every rule in `keys` must hold where the model has the key. Then stage 0 sets aside:

1. A differing text config key that `keys` declares (or whose differing sub-keys `keys` declares as dotted paths), that `ignored_keys`
   names, or that `optional_keys` names when the new model lacks it. The `layers.types_key` list when it has one entry per layer of the
   new model and every entry is in `type_values`. `model_type` when it is one of `model_types`.
2. A shape difference of a tensor a `shapes` entry matches, when every differing dimension is declared in that entry and each model's
   size equals the expression on its own config. A change of rank is never set aside.
3. A tensor at a layer index only one model has, when `layers.count_key` is a declared key that differs, the index is below that
   model's own layer count and at or past the other model's, and the other model has the same tensor at a layer of the same type
   (from `types_key`, when both configs list their layer types) with the same dtype and an equal shape, or a shape a `shapes` entry
   explains.
4. A tensor an `optional_tensors` entry matches, in either model. When the entry gives a `shape`, the new model's tensor must have it.

Everything else stays a reason for a full port: an undeclared key, a value outside its rule, an extra or missing tensor no entry
covers, a shape difference along an undeclared dimension, any dtype difference beyond the small float tolerance stage 0 already has.
With the file applied, every differing text config key needs to be covered, including keys stage 0 otherwise ignores (such as
`rope_theta` or `rms_norm_eps`), because stage 2 serves the new model's own config and the bundle's code reads it. Bookkeeping keys
(`architectures`, `transformers_version`, `_name_or_path`) and the keys stage 0 already sets aside as written by a newer transformers
need nothing.

`features` is recorded in `delta.json`; stage 0 checks only that `features.indexer`, when present, is one of the three values the
release-day check knows.

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

This is a sketch: the release-day check's example file with the fields stage 0 needs to compare the release with a 20-layer
development checkpoint. The bundle's author writes the real file from the keys the model code reads.

```json
{
  "schema": 1,
  "bundle": "mando2222/qwen4-27b-dense-v6-thin",
  "model_types": ["qwen4_exp", "qwen4_exp_text"],
  "architectures": ["Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"],
  "keys": {"hidden_size": {"equals": 5120}, "num_attention_heads": {"equals": 24}, "head_dim": {"equals": 256},
           "num_key_value_heads": {"one_of": [2, 4]}, "intermediate_size": {"multiple_of": 128},
           "num_hidden_layers": {"min": 4, "multiple_of": 4}, "vocab_size": {"equals": 248320}, "hc_count": {"equals": 4},
           "hc_lowrank": {"one_of": [320, 640]}, "output_gate_type": {"one_of": ["silu", "sigmoid"]},
           "hidden_act": {"equals": "silu"}, "rope_parameters.rope_theta": {"min": 1},
           "rope_parameters.partial_rotary_factor": {"min": 0, "max": 1}, "max_position_embeddings": {"min": 2048}},
  "features": {"indexer": "exact-up-to-2051", "mtp": "ignored", "vision": "off"},
  "ignored_keys": ["dtype", "initializer_range", "use_cache", "bos_token_id", "eos_token_id", "pad_token_id"],
  "optional_keys": ["full_attention_interval"],
  "layers": {"pattern": "^model\\.layers\\.(?P<layer>\\d+)\\..+$", "count_key": "num_hidden_layers", "types_key": "layer_types",
             "type_values": ["linear_attention", "full_attention"]},
  "shapes": [
    {"tensor": "^model\\.layers\\.\\d+\\.mlp\\.(gate|up)_proj\\.weight$", "dims": {"0": "intermediate_size"}},
    {"tensor": "^model\\.layers\\.\\d+\\.mlp\\.down_proj\\.weight$", "dims": {"1": "intermediate_size"}},
    {"tensor": "^model\\.layers\\.\\d+\\.self_attn\\.[kv]_proj\\.weight$", "dims": {"0": "num_key_value_heads*head_dim"}}
  ],
  "optional_tensors": [{"tensor": "^model\\.layers\\.\\d+\\.self_attn\\.indexer\\..+$", "feature": "qsa_indexer"}]
}
```
