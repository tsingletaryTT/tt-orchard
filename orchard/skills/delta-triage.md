---
name: delta-triage
description: Stage 0 of a tt-orchard run. Compare a new Hugging Face model with the nearest model that already runs on Tenstorrent hardware, write down every difference with evidence, and name the path the run takes.
status: draft. A local tt-orchard copy. Its home is the tt-model-bringup plugin in tenstorrent/skills (spec section 11). Move it there after a real run has used it.
---

# Delta triage

## Goal

Write `delta.json` in your stage directory. It lists what differs between the new model and the
nearest supported model, gives evidence for each difference, and names the path:

- `weights-only`: the model code that runs the nearest model can run the new one with new weights.
- `full-port`: new model code is needed. Plan 4 of tt-orchard stops the run for the operator here.

## Inputs

The run lists input paths, for example `model=` and `base=` snapshot directories. Read the files
there. Do not download anything. The run has no Hugging Face token, and the weights are on disk.

## What to compare

Save the output of each comparison under `evidence/`.

1. `config.json`: every key that differs. The text-model keys matter most: layers, hidden size,
   attention heads, KV heads, layer types, vocabulary size, context length. Name keys that differ
   only as bookkeeping (`architectures`, `transformers_version`).
2. Tensors: names from each `model.safetensors.index.json`, and shapes and dtypes from the
   safetensors headers. Report the counts, names found in only one model, and every shape or dtype
   difference. A consistent name prefix difference is its own entry (area `tensor_names`).
3. Tokenizer: vocabulary size, merges, added tokens, chat template and pre-tokenizer pattern. If
   any of them differs, encode a set of test strings with both tokenizers and count the strings
   whose ids differ. Save the strings.
4. Files: the weight shards, their total size, and files only one model has (for example `mtp.*`
   or vision weights).
5. `generation_config.json`: the sampling defaults.
6. The license, from the model card or the LICENSE file.

A safetensors header needs no extra package: the first 8 bytes are a little-endian length, and
that many bytes of JSON follow. Write a short Python script into your stage directory with
write_file and run it with `python3`. The supervisor refuses heredocs.

## The file

    {"model": "<org/name>@<revision>", "nearest_model": "<org/name>@<revision>",
     "path": "weights-only",
     "differences": [{"area": "tokenizer", "finding": "...",
                      "evidence": ["stages/0/evidence/tokenizer-diff.txt"]}],
     "hazards": [{"area": "tensor_cache", "finding": "..."}]}

A difference's `area` is one of: config, architecture, tensors, tensor_names, tokenizer,
chat_template, files, generation_config, license, other. Write one entry for each area you
checked, including areas with no difference (the finding says "identical" and the evidence shows
it). A hazard's `area` is one of: tensor_cache, drafter, disk, license, other. Evidence paths are
relative to the run directory and must be files inside it.

## Hazards to look for

- A converted tensor cache of the nearest model that a later stage could reuse. A cache keyed
  only by layer name serves the old weights silently. Name the cache directory that must stay
  unused.
- A draft model for speculative decoding that was trained against the nearest model. Its
  acceptance rate may drop, so it must be measured.
- Free disk where the weights and caches will go.

## Resume notes

Write progress notes to `RESUME.md` in your stage directory as you go. If the run restarts, the
supervisor keeps your stage directory when that file exists and gives you the notes.
