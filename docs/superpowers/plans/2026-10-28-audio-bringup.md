# Audio bring-up: implementation plan

Date: 2026-10-28. Spec: `docs/superpowers/specs/2026-10-15-audio-bringup-design.md` (approved by the operator on 2026-10-28).
Findings that shaped this plan: `docs/run-logs/2026-10-28-stable-audio-3-facts.md`.
First target: `stabilityai/stable-audio-3-small-music` (a full fine-tune of `stabilityai/stable-audio-3-small-music-base`).

## 0. Where this plan differs from the spec

The spec was written for a LoRA on a music model that already runs in a container (MiniMax-Music3). Stable Audio 3 differs in four ways.

1. Nothing serves Stable Audio 3 on a Tenstorrent chip. No bundle, no container package, no tt-metal demo, no `tt_dit` pipeline (read on 2026-10-28; the catalog search for `stable-audio` is empty). The spec's stage 2 boots the base in a container, so for this model there is no container to boot. A run can classify the model and stop. Serving it needs a port of the base first (section 5, slice C).
2. The model is a full fine-tune; it ships no adapter. The spec marks `form: full` as untested and gives it the same path without the merge. Stage 1 has nothing to merge.
3. Its weights are one `model.safetensors` that holds three components (the diffusion transformer, the autoencoder and a small conditioner) under the prefixes `model.model.`, `pretransform.model.` and `conditioner.conditioners.`, next to a text-encoder directory (`t5gemma-b-b-ul2/`) with its own `config.json`. There is no top-level `config.json`; the architecture is in `model_config.json` (the `stable-audio-tools` format). Stage 0's text rules (`config.json` keys such as `hidden_size`) do not apply.
4. The spec's `serving` values are `thin-bundle` and `container`. This plan adds `none` (nothing serves the base).

Where the spec and this plan agree, the spec stands. Section numbers below refer to the spec.

## 1. Order and what needs a chip

| slice | content | chip |
|---|---|---|
| A | the three facts (spec section 3), the audio branch of stage 0, the stop after stage 0, base lookup and preflight messages, the skill text | no |
| B | stage 1 materialize for adapters (spec section 5); stage 2's single-test phase sweep and the container template with the cache-hang probe (spec section 6); `gate_audio_generate`; bundle and web page (spec section 8) | B1 no (fakes), B2 yes |
| C | a Tenstorrent runtime for the model (a port of the diffusion transformer and the autoencoder with `tt_dit` blocks, and a server with the job API the stage 2 template expects) | yes, and it is a project of its own |

Slice A is built first and needs no chip. Slice B needs a model that some container serves; today that is only MiniMax-Music3 (spec first target). Slice C is the work that makes Stable Audio 3 serve at all. Until C exists, a Stable Audio run ends at stage 0 with the block `audio-path-unbuilt`.

## 2. Slice A tasks (each test first, a mutation per guard seen red, one commit per task)

### A1. The facts module and their record

Files: `orchard/audio_facts.py` (new), `orchard/stages.py` (gate and reader), `orchard/supervisor.py` (stage 0's `stage_end`).
- `FORMS = ("adapter", "full")`, `SERVINGS = ("thin-bundle", "container", "none")`, `MODALITIES = ("text", "audio")`; `facts_problems(facts)` returns reasons for a record that is not a dict, has an unknown key, or a value outside its set; `serving_of(base_repo, installed)` returns `thin-bundle` when a schema 6 bundle serves it, else `container`, else `none`.
- `stages.run_facts(entries, run_dir)` reads the facts the way `run_class` reads the class: from stage 0's passing `stage_end`, and `{"form": "full", "serving": "thin-bundle", "modality": "text"}` for a ledger with none.
- The stage 0 gate (`_delta_reasons`) refuses a `facts` record `facts_problems` rejects, and an audio modality with a class other than `weights-only`, `full-port` or `unknown`.
- The supervisor adds `facts` (with `serving` resolved from the installed bundles and containers) to stage 0's passing `stage_end`.
- Mutations: each set, an extra key accepted, a bool as a value, the default for an old ledger, the replay reading `delta.json` instead of the ledger, a schema 5.1 bundle counted as `thin-bundle`.

### A2. The audio branch of stage 0

Files: `orchard/skills/delta-triage-templates/delta_triage.py`, (the audio branch lives in the same file, because the stage copies and hash-stamps one file).
- `detect_audio(snapshot)`: true when the snapshot has a `model_config.json` that is an object with `model_type` and `model` keys, no top-level `config.json`, and at least one top-level safetensors file. Everything is bounded (config 1 MiB, header 8 MiB).
- `components(tensors)`: groups tensors by the first two name parts, with count, parameter count and dtypes. `compare_audio(model_dir, base_dir)`: compares `model_config.json` (leaf by leaf, with the keys that name training only, `training`, listed apart from `model`), the component tables, every shared tensor's shape and dtype, the text-encoder directory's `config.json` and tensor header (hash of the header, not the weights), and each side's file list. It names the files the base has and the model lacks (`svd_bases.pt` in the base).
- Class: `weights-only` when the `model` section of the two configs is equal, every shared tensor has the same shape and dtype, no tensor exists on one side only, and the text encoders have equal headers. A difference in the `model` section, a shape, or a one-sided tensor is `full-port`, with the reasons. A header that cannot be read, or a config that is not an object, is `unknown`. `form` is always `full` in this slice. An adapter repository has no `model_config.json`, so it is not detected as audio; adapter triage belongs to slice B1. `modality` is `audio`.
- `delta.json` gets `facts`, an `audio` record (components, sample rate, channels, sample size, steps and sampler hints from the card's code block only as text, the model's and the base's file lists, the config differences) and hazards in the existing areas (`other`, `license`): the licence is `other` with a Gemma-terms notice; the base's `svd_bases.pt` is not in the model; no check measures how the audio sounds.
- Mutations: the branch off for a text model, the config comparison ignoring a `model` difference, ignoring a training-only difference (must be allowed), a shape difference passing, a one-sided tensor passing, an unreadable header giving `weights-only`, a text model with a stray `model_config.json` entering the branch.

### A3. The stop after stage 0

Files: `orchard/supervisor.py`, `orchard/status.py`, `orchard/blocked.py`, `orchard/narrate.py`.
- A run whose stage 0 facts say `modality: audio` pauses before stage 1 (the way a restricted head does) with a reason that says the audio stages are not built yet and names the facts; unattended it blocks with the new code `audio-path-unbuilt`, with a "how to unblock" text. When slice B lands, this stop applies only to `serving: none`.
- `status.classify_pause` gets the kind `audio_unbuilt`; `blocked.REASONS` and `narrate.UNBLOCK` get the code (a test ties them).
- Mutations: the stop for a text run, no stop for an audio run, an attended resume not acknowledged, an unattended retry passing the stop.

### A4. Base lookup and preflight messages

Files: `orchard/preflight.py` (additive: one `HubInfo` field and the `check_base` message only; the hub-row code belongs to issue 119), `orchard/nearest.py`.
- `HubInfo.pipeline_tag` and `parse_hub` read it. `check_base` appends an audio note to `nearest-model-missing` when the pipeline tag is `text-to-audio`, `text-to-speech`, `automatic-speech-recognition` or `audio-to-audio`: the run needs a base that some bundle or container package serves; none was found; for the cases this repository knows (read on 2026-10-28) tt-metal's `tt_dit` and `models/demos/audio` hold other models and no Stable Audio port.
- `nearest.serving_kind(base_repo, installed)` (used by A1).
- Mutations: the note for a text model, no note for an audio model, a container base missing from the lookup, the note replacing the candidate listing.

### A5. The skill and the README

- `orchard/skills/delta-triage.md`: a paragraph on the audio branch, what it records and what it does not.
- README: the class table row, the preflight text, and the fact that audio runs stop at stage 0 until slice B or C exists.
- A test ties the skill's keys and the facts sets.

## 3. Slice B tasks (not built now)

- B1, no chip: `orchard/skills/lora-merge-templates` wrapper `materialize_adapter.py` (spec section 5); adapter triage (rank, alpha source, component; spec section 5); `spec_for(..., facts)` for the audio variants; the supervisor-owned sweep after every stage 2 test and at resume (spec section 6); `gate_audio_generate` re-deriving every check from the saved files (spec section 6); the stdlib fake container server of spec section 9; the bundle's `audio/` section and the web page's FLAC route (spec section 8). Stage 2 reuses the cache-hang probe of 2026-10-15 (`kv_probe.py` in the music companion repository): a 64 MiB `as_tensor` load under 60 s.
- B2, chip: the first real run on `fiona-crapple`, killed once and resumed (spec section 9).

## 4. Slice C (not built now)

A Stable Audio 3 runtime on a Blackhole board. The parts, from the files and from `tt_dit` (read on 2026-10-28):
- the diffusion transformer (depth 20, width 1024, 16 heads, 459 M parameters, RMS norm, query-key norm, 64 memory tokens, adaLN global conditioning, cross attention to 768-wide text tokens): `tt_dit.blocks` has attention and transformer-block building blocks used by Flux, Wan and others;
- the autoencoder (a transformer autoencoder, 108 M parameters, stride 16 patches of 256, 6 transformer layers, differential attention, dynamic-tanh normalisation, chunked by 32): only its decoder is needed to produce audio;
- the text encoder (T5Gemma encoder, 282 M parameters used) and the 8-step sampler (ping-pong, CFG scale 1.0) can start on the host.
The host-only parts make a useful first measurement: a CPU reference for stage 1 style checks. A port plan needs its own design; this note only lists the parts.

## 5. Hardware order

1. A1 to A5 need no chip.
2. The base `stabilityai/stable-audio-3-small-music-base` must be on disk to run stage 0 on the pair; its headers and config are readable without a token, but the harness does not fetch a gated model.
3. Nothing in slice A opens a device. Slices B2 and C need a lease (`gozer acquire`, then `gozer wait`), one board for B2.

## 6. What this plan does not do

It does not judge how the music sounds, does not publish, and does not use a Hugging Face token for a gated model.
