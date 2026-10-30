# 2026-10-29: the next round of models, and the open issues that unlock them

Prompt (operator): "look for more issues on our list that will unlock another round of models." This is a research note. No hardware was used,
no lease was taken, nothing was installed, no weights were downloaded, nothing was published. The only write is this file and its pull request.

The ranking is by how many more models each open issue would let tt-orchard bring up. Every number that is an estimate says so. Section 9 lists
what I could not verify.

## 1. Short answer

Five issues come first. They are listed in the order to do them (section 8 has the reasons and the first model for each):

1. **#93 (with #62): a first chip result for the stock-runtime path.** One run of Llama-3.1-8B-Instruct. The preflight already routes a model that
   tt-transformers supports (the dry-run below shows it). What is missing is one real stage 2 number. If the path works, it covers the families
   in the host tree's `model_params` list (Llama 3.x, Qwen2.5, Qwen3-32B, Mistral-7B v0.3, phi-4, gemma-3-27b and others).
2. **#60 (with #73): a container (v5.1) package as the nearest model for a weights swap.** 49 of the 50 catalog packages are containers. This opens the
   Qwen3-Next and Coder-Next family (same dimensions as the coder the harness already serves), the Qwen3.5 and Qwen3.6 35B-A3B family, gpt-oss-20b and
   the Gemma-4 packages.
3. **#183: wire the stage 0 hazard scan into stage 0.** The module exists and is not wired. It adds no model by itself. It is the cheap guard for the
   512-expert and wide-vocabulary models in the next two items.
4. **#184: the fusion-port step on a chip.** `fusion-candidates` (PR 235) is merged. The chip step for Qwen3-Next-80B-A3B, and for Qwen3.5 and Qwen3.6
   35B-A3B packages built from a recipe, is the open part.
5. **#63: a hardware test of one quantized copy.** It decides whether FP8, NVFP4 and GPTQ copies of supported models are a class. They are a large
   share of the safetensors repos under every popular base.

Two cheap facts change the plan. First, `Qwen/Qwen3.6-27B` classifies as weights-only against the installed Qwen3.8-27B bundle (stage 0's own
functions, headers only), and the Hub lists 424 fine-tune repos of it. That family needs no issue. It needs a run. Second, PR 245 and PR 247
(the text-model route) are open and will close #206 and #214 when merged. They are the precondition for adding the next hand-ported text model as a
registry entry. That is work for a new issue, listed in section 8.

## 2. What I ran and read

| Source | What I did | Limit |
|---|---|---|
| `gh issue list --state open --limit 300` | read all 62 open issues (titles), and the bodies of about 40 | titles only for the Stable Audio, JSON-reader and updates issues |
| Issues 59 and 28 and the run logs | read | issues 28 and 62 are older than parts of the code (section 7) |
| Run logs `2026-10-19-nemotron-bolt-74b`, `2026-10-19-pplx-decider-v1.2`, `2026-10-28-real-port-choice`, `2026-10-29-fusion-port`, `2026-10-29-ttlang-next` | read | |
| `site/index.html` | read the model stories and the "automatic / needs a person" list | |
| PR 245, 246, 247 | read the PR bodies and, for PR 245, `docs/text-runtime.md` on its branch | not merged; I did not run their code |
| tt-model-manager (read only) | `git fetch`, listed remote branches and their commit subjects; `tt-model search --catalog` | no pushes; I did not read branch diffs |
| `tt-orchard bringup MODEL --dry-run` on 8 models | the base row and the verdict line | section 4 says why no device was touched |
| `python3 -m orchard.classify_bench collect` | stage 0's class from Hub headers for 18 candidate repos, written to a scratch directory | the base snapshot decides the result; see the instrument note |
| Hugging Face Hub metadata (token from the environment, never printed) | trending text and multimodal lists, config and size fields, fine-tune counts by base | counts are a ceiling (section 3) |

Instrument note on the classification bench. It compares each candidate with one local base snapshot. Run against the Qwen3.8-27B snapshot it
classified `Qwen/Qwen3.6-27B` and `sensenova/SenseNova-6.8-Code2Video` as weights-only with complete evidence. Run with the Qwen3.5-35B-A3B snapshot
as base it returned `unknown` for every candidate, including `Qwen/Qwen3.5-35B-A3B` itself ("the two models share no tensor name"). A model
compared with itself must come out weights-only, so that run measured a base snapshot with no readable weight headers and said nothing about the
candidates. I discarded it. A self-control against the Qwen3.8-27B snapshot also came back `unknown`, with `evidence_complete: false` and connection
resets on several shard headers, so that control failed on the network and left the classifier untested. The weights-only results above had
`evidence_complete: true`. I did not repeat them. Every class below that is not from issue 59 is a single reading.

## 3. Where candidate models stop today

The stop classes are stage 0's classes (weights-only, weights plus sidecar, vocab-trim, restricted-head, stock-runtime, full-port, unknown) and the
preflight blocks that arrive before stage 0 (`nearest-model-missing`, `coder-unusable`).

Issue 59's replay of the 50 catalog models (2026-10-24, headers only, base Qwen3.8-27B): of the 29 weights repos with a `config.json`, 24 were
full-port, 4 unknown, 1 weights-only (RAG-Gate-27B, already brought up). The other 21 packages (perception and ComfyUI repos) have no `config.json`.
By kind: 22 language, 13 perception, 5 speech-out, 4 image, 3 music, 1 video, 1 speech-in, 1 embedding. All 49 container packages were brought up
by hand elsewhere. A run started from their Hub repo ends at stage 0 or the preflight.

Hub counts for the same bases (2026-10-29, `base_model:finetune:` relation, 12 pages of 100 at most, so 1200+ means "1200 or more").
"With safetensors" counts repos whose metadata lists safetensors. The count is a ceiling on reach. It includes LoRA merges, repos with a different
layout and repos nobody would serve. It says how large a family is. It does not say how many would pass.

| Base | Fine-tune repos | With safetensors | Route today |
|---|---:|---:|---|
| Qwen/Qwen3.8-27B | 777 | 580 | weights-only, installed bundles (already works) |
| Qwen/Qwen3.6-27B | 424 | 248 | weights-only against the 27B bundle (bench reading); also in tt-transformers `model_params` |
| Qwen/Qwen3.5-35B-A3B | 187 | 160 | `nearest-model-missing` (dry-run) |
| Qwen/Qwen3.6-35B-A3B | 415 | 314 | same family, same stop |
| Qwen/Qwen3-Coder-Next | 40 | 23 | container package only (#60) |
| Qwen/Qwen3-Next-80B-A3B-Instruct | 41 | 27 | no base named (dry-run warn); container only |
| meta-llama/Llama-3.1-8B-Instruct | 1200+ | 722 | stock-runtime route (dry-run warn) |
| meta-llama/Llama-3.2-3B-Instruct | 1200+ | 634 | in `model_params`; not dry-run |
| Qwen/Qwen2.5-7B-Instruct | 1200+ | 949 | in `model_params`; not dry-run |
| Qwen/Qwen3-32B | 595 | 516 | in `model_params`; not dry-run |
| mistralai/Mistral-7B-Instruct-v0.3 | 542 | 222 | in `model_params`; not dry-run |
| google/gemma-3-27b-it | 455 | 297 | in `model_params`; license is gated (manual) |
| microsoft/phi-4 | 129 | 77 | in `model_params`; not dry-run |
| Qwen/Qwen3-8B | 1200+ | 975 | `nearest-model-missing` (dry-run); `model_params` has Qwen3-32B only |
| Qwen/Qwen3-14B | 370 | 275 | same as Qwen3-8B (not dry-run) |
| openai/gpt-oss-20b | 573 | 147 | dry-run warn (no base named); container packages in the catalog |
| google/gemma-4-31B-it | 334 | 231 | needs the v0.81 tree per the text of issue 95; not dry-run |
| ibm-granite/granite-4.0-h-1b | 10 | 7 | dry-run block; hand port exists (PR 244) |
| openbmb/MiniCPM3-4B | 3 | 1 | hand port exists; text route (PR 245) |

Quantized copies (relation `base_model:quantized:`, first 1200 repos at most, split by Hub tags): Qwen3.8-27B has 340 safetensors copies next to 478
GGUF and 349 MLX copies; Qwen3.5-35B-A3B has 65 safetensors copies (316 in all); Qwen3.6-35B-A3B has 192 (936 in all). GGUF and MLX repos are not
safetensors and stage 0 has nothing to read in them. I did not run a GGUF-only repo through stage 0, so I do not know how it reports one.

Trending text and multimodal models on 2026-10-29 (Hub trending score, first 40 of each list; metadata only):

| Model | Facts from the Hub | First stop today |
|---|---|---|
| Nanbeige/Nanbeige4.3-33B | llama arch, 33.1B, apache-2.0, 52 layers, hidden 7168 | full-port against the 27B base. A stock-runtime reading was not taken |
| HuggingFaceBio/Carbon-X | qwen3, 14.8B, apache-2.0, fine-tune of Qwen3-14B | `nearest-model-missing`; the stock route would need to accept the Qwen3 dense sizes |
| InternScience/Sage-A1.5 | qwen3_5_moe, 36.0B, apache-2.0 | needs a 35B-A3B base; its experts are stored one tensor per expert (31,333 text tensors), so a loader must fuse them |
| GrootTeam/Groot-19B-A3B-261023 | qwen3_5_moe, 19.4B, apache-2.0 | same family; fewer parameters than the 35B model, dimensions not read |
| MemTensor/MemT-1.0-Nano, Diffuspace/Clover-35B-A3B-Preview | custom model types on Qwen3.6-35B-A3B | same family; custom tensors (Clover has `bd_softmax_history_bias`) |
| sensenova/SenseNova-6.8-Code2Video | qwen3_5, 27.4B, apache-2.0 | weights-only against the 27B bundle (bench reading); needs a run only |
| LiquidAI/LFM3-1.7B | lfm3, 1.7B, license `other` | full-port; hybrid with convolution layers; no port exists |
| nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16 | gemma4 arch, 8.0B, license `other` | depends on the v0.81 tree if Gemma 4 support is the route (issue 95 text) |
| google/gemma-4-31B-it | gemma4, 31.3B, apache-2.0 | same |
| Qwen/Qwen3.8-Flash-Next | qwen4_exp, 180B, license `other` | full-port; a catalog package for it exists (container) |
| Large mixtures (DeepSeek-V4.1-Flash 763B, Kimi-K3 2.78T, Ling-3.1-flash 572B, MiniMax-M3.1-Flash 220B, GLM-5.3 753B) | sizes from the Hub | do not fit one QuietBox 2 (4 chips) at the sizes listed; I did not compute quantized sizes |
| Speech and audio trending: stable-audio-open-1.0, YuE2-3B, ACE-Step1.5, musicgen-small, Nemotron-3.5-Transcribe-30B-A4B, Kokoro-82M, drama-3 | text-to-audio, speech | the audio route exists for Stable Audio 3 only (section 6) |

## 4. The dry-runs, and why none of them touched a device

I read `orchard/preflight.py` before running anything. The checks it runs are the Hub lookup, disk, credentials, tiers, the coder port, an HTTP probe of
the CPU stand-in, `gozer status` and `gozer topology` (status listings), an optional `tt-smi -s` snapshot, a reference-venv import check
(torch, transformers, tokenizers, safetensors), skills freshness and the updates hint. It imports no ttnn or ttml and takes no lease. `--dry-run`
returns before any stage. While the dry-runs ran, `gozer status` showed two chips held by another lease, as before. I started and stopped nothing.

| Model | Verdict line | Base row |
|---|---|---|
| Qwen/Qwen3-8B | `coder-unusable`, `nearest-model-missing` | no installed bundle; lists the Qwen3.8-27B, Coder-Next, Coder-30B and 122B packages as options |
| meta-llama/Llama-3.1-8B-Instruct | `coder-unusable` only | warn: no installed bundle, but tt-transformers supports it; "the run serves it through tt-transformers as it is"; the README row names Blackhole boards |
| Qwen/Qwen3-30B-A3B | `coder-unusable`, `nearest-model-missing` | base `Qwen3-30B-A3B-Base`; points at code in tt-transformers and tt-metal, which is not a bundle |
| Qwen/Qwen3.5-35B-A3B | same | base `Qwen3.5-35B-A3B-Base`; no bundle |
| ibm-granite/granite-4.0-h-1b | same | no bundle; `tt model search granite` found none |
| openai/gpt-oss-20b | `coder-unusable` only | warn: no `base_model` in the card; closest bundles are the 120B ones; names `models/demos/gpt_oss` |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | `coder-unusable` only | warn: no `base_model`; closest bundles are Qwen3.8-27B ones |
| Qwen/Qwen3-Next-80B-A3B-Instruct | `coder-unusable` only | warn: no `base_model`; closest bundles are Qwen3.8-27B ones |

`coder-unusable` in every row is the check on port 8011: the local coder is running and the dry-run found the port in use. I take it to be an effect
of the dry-run happening while the coder is up. I did not stop the coder to see the row change.

Two details the issues do not say. The three "warn" rows for models with no `base_model` name the nearest bundles as Qwen3.8-27B ones with "no
estimate, weak". A real run that continued from there would classify against the wrong family. And the Llama-3.1-8B row shows the stock route is
built into the preflight, so the title of #62 ("has no path") describes an earlier state (section 7).

## 5. The ranking

Effort is in working days, and every effort figure is my estimate, written before any of the work. "Hub ceiling" is the sum of the fine-tune counts in
section 3 for the families the issue reaches. The ceiling is the size of the families. The number that pass will be smaller, and I cannot say by how
much.

| Rank | Issue | Stop classes it removes | Families reached (Hub ceiling) | Concrete models | Effort (est.) | Risk |
|---:|---|---|---|---|---:|---|
| 1 | #93, #62 | stock-runtime class has no chip result; no chip-count claim; no stage 8 section | Llama 3.x, Qwen2.5-7B, Qwen3-32B, Mistral-7B v0.3, gemma-3-27b, phi-4: about 5,300 repos, 3,700 with safetensors | Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct, Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, Qwen3-32B | 3 to 4 for the first model and stage 4; 2 more for the stage 8 section | Medium. The demo's log lines were matched to the template's parsers from source, and no log was read. A gated license may block the first download |
| 2 | #60, #73 | `nearest-model-missing` when the base is a container package; a swap check that passes on the base model because the container ignores the weights | Qwen3.5 and Qwen3.6 35B-A3B (602), Qwen3-Next and Coder-Next (81), gpt-oss-20b (573), Gemma-4 packages (the issue names jialinyyzz/humanizer) | Huihui-Qwen3-Coder-Next-abliterated, Qwen3-Next-80B-A3B-Instruct and -Thinking, Huihui-Qwen3.5-35B-A3B-abliterated, Holo3-35B-A3B | 5 to 7 (one probe per container, a stage 2 container record, a preflight row) | Medium. Not every container reads `MODEL_WEIGHTS_DIR` (#73's own example hard-codes its repository) |
| 3 | #183 | silent wrong output or a hang from a dimension (512 experts, vocabulary per die above 65,536, a 3-tile hidden size) | none directly; guards the models in rows 2 and 4 | Qwen3-Next-80B-A3B (512 experts), Qwen3.5-35B-A3B (248,320 vocabulary) | 2 (no chip) | Low |
| 4 | #184 | `fusion-candidates` lists ops; nothing ports or measures one on a chip | Qwen3-Next and Coder-Next (81); the 35B-A3B families (602) once built from a recipe | Qwen3-Next-80B-A3B-Instruct (6 ops apply as is, 1 needs new code on two dies); Qwen3.6-35B-A3B | 5 to 8 | Medium. The layer boundary op runs on four dies only; two-die results were already measured for Qwen3.5-35B-A3B |
| 5 | #63 | a quantized copy is classed full-port; no class and no block message | quantized safetensors copies: 340 of Qwen3.8-27B, 65 of Qwen3.5-35B-A3B, 192 of Qwen3.6-35B-A3B | an FP8 or NVFP4 copy of a model that has a bundle; the catalog's Nemotron-3.5-Lightning-30B-A3B-NVFP4 is also a Mamba hybrid, so it stays a full port | 3 (one chip test) plus 1 for the class | High. Whether the runtime accepts block formats is not known. A refusal is a valid result and still removes the guess |
| 6 | #95 | the host tree is held at v0.80.0, so models whose support landed in v0.81 are out of reach; the kernel probe cannot build there | Gemma-4 family (334 on 31B-it) and Nemotron-3.5-Bolt-E4B (a Gemma 4 base); catalog packages for Gemma-4-12B and 26B-A4B | gemma-4-31B-it, Gemma-4-12B-it, Nemotron-3.5-Bolt-E4B-BF16 | 2 to read the v0.81 support list, more for an upgrade | Medium. The issue says upstream tt-lang still emits forms v0.81 removed |
| 7 | #28, #143, #160 to #163, #168, #212, #214, #220 (audio) | audio runs boot the coder before stage 0; a second audio pipeline needs its own port | text-to-audio: stable-audio-open-1.0, ACE-Step1.5, musicgen-small, YuE2-3B | stable-audio-open-1.0 (1.21B, license `other`), ACE-Step1.5 (mit) | 2 for #143; 7 to 15 per new pipeline | High for a new pipeline: each model brought its own blocks |
| 8 | #61, #90, #94 | an unattended run refuses a full port and ends `needs-new-model-code` | every model whose port has not been written | Nanbeige4.3-33B, LFM3-1.7B, Carbon-X, Nemotron Bolt | 20 or more (my estimate; no full-port stage has run on a real model) | High. The coder passed `row_sum` 6 of 9 and `rmsnorm` 0 of 3 on the ladder |
| 9 | #97 | no training path | new student models | none named | not estimated | High |
| 10 | #14, #107, #83, #181, #182, #15, #17, #18, #19, #21, #23, #24, #251 | none of these adds a model. They raise speed, confidence or package coverage for models that already come up | 0 new | | | |

The order follows reach and effort together. Rows 1 and 2 come first because each removes a block that sits before stage 0 for a whole family,
and each has an existing path that was built and has not yet run on a chip. Row 3 has no reach of its own and costs two days. Row 4 needs the guard
from row 3 and the baseline from row 2. Row 5 is a single chip test whose result changes the label on several hundred repos in either direction.
Row 8 is the largest piece of work, and its own issue text names the stock-runtime model in row 1 as the pilot.

### Detail for the top five

**#93 and #62 (stock runtime).** Removes the missing chip result for a model that tt-transformers supports. The host tree's `model_params` lists 23
entries (v0.80.0): EXAONE-4.5-33B, gemma-3-27b-it, LFM2.5-VL-1.6B, Llama-3.1-70B and 8B, Llama-3.2 (1B, 3B, 11B vision, 90B), Meta-Llama-3-8B,
Mistral-7B-Instruct-v0.3, phi-4, Qwen2.5 (7B, 32B, 72B), Qwen2.5-VL (3B, 7B, 32B, 72B), Qwen3-32B, Qwen3.6-27B, Qwen3-VL-32B. Models to run, with facts
from the Hub:

| Model | Size | License | Disk (bf16, 2 bytes per parameter) | Fit |
|---|---:|---|---:|---|
| meta-llama/Llama-3.1-8B-Instruct | 8.0B | llama3.1, gated (manual) | 16 GB | one chip, assuming 32 GB per chip |
| meta-llama/Llama-3.2-3B-Instruct | 3.2B | llama3.2, gated (manual) | 6.4 GB | one chip |
| Qwen/Qwen2.5-7B-Instruct | 7.6B | apache-2.0 | 15.2 GB | one chip |
| mistralai/Mistral-7B-Instruct-v0.3 | 7.2B | apache-2.0 | 14.4 GB | one chip |
| Qwen/Qwen3-32B | 32.8B | apache-2.0 | 65.6 GB | does not fit two chips in bf16 under the same assumption; four chips |

Choose the first run with the license in mind: if the account's token has no access to the Llama repository, start with Qwen2.5-7B-Instruct.
The issue names Llama-3.1-8B-Instruct as the pilot (#61, #93). Effort: stage 2 on one chip, then #93's stage 4 chip-count checks, then the stage 8
section. Risk: whether the Blackhole boards run the demo is untested (#62 says so for `qwen25_vl`, whose README names Wormhole boards).

**#60 and #73 (container as nearest model).** Removes the base check that refuses an installed container. Stage 4 already runs a container with
swapped weights (`serve_and_compare_container.py`), and a hand test of the music container with a merged model served on one chip. The missing
parts are stage 2 drafting a container configuration, the gate accepting the record, and the preflight row. Do #73's one-chip probe first on each
container, so that a package that hard-codes its repository blocks before a coder boot. Models:

| Model | Size | License | Disk (Hub storage) | Family and fit |
|---|---:|---|---:|---|
| huihui-ai/Huihui-Qwen3-Coder-Next-abliterated | 79.7B | apache-2.0 | about 160 GB bf16 | Coder-Next (`qwen3_next`); the coder runs on one p300 board, so a swap should fit the same way (estimate; I did not check how the coder's weights are stored on device) |
| Qwen/Qwen3-Next-80B-A3B-Instruct, -Thinking | 81.3B | apache-2.0 | 162.7 GB | same dimensions as Coder-Next in every number the fused ops read (fusion-port log); fit as above |
| huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated | 36.0B | apache-2.0 | about 72 GB | ran on two dies in the 35B-A3B work (PR 236, 246) with 4-bit routed experts; a bf16 copy of 72 GB does not fit two chips under the 32 GB assumption |
| Hcompany/Holo3-35B-A3B | 35.1B | apache-2.0 | about 70 GB | same |
| InternScience/Sage-A1.5 | 36.0B | apache-2.0 | about 72 GB | same, plus a loader change for per-expert tensors |

The disk figures come from the Hub. At the time of the check the root filesystem had 257 GB free and the second drive 32 GB. I did not read which
one holds the Hugging Face cache, so the 162.7 GB repo may not fit without the preflight's prune options.

**#183 (hazard scan).** `orchard/hazards.py` exists with tests, and the log of 2026-10-28 says it is not wired into a stage. Wire it into stage 0 and
add the stage 2 targeted test for each hit. The issue's own "done when" asks for a run over the issue 59 corpus (24 full-port repos). Effort 2.

**#184 (fusion port on a chip).** Phase 1 is merged (PR 235). The log of 2026-10-29 states the Phase 2 chip port was not run. For Qwen3.5-35B-A3B the
result already exists by hand: 126.8 tok/s median on two dies, 29.6 percent of the estimated floor, with the first 8 tokens equal to the CPU
reference (PR 246 packages it as a local recipe). The open work is to take that result through the harness for a second member of the family, and to
port one changed constant with its measured effect. The log notes the route-k constant has no effect on one-row decode. Qwen3-Next-80B-A3B has no
changed constants, so it tests the swap and leaves the port untested.

**#63 (quantized copy).** The check is one chip test of one quantized copy of a model that already has a bundle. The result has three forms:
accepted, dequantized, refused. Any of them lets stage 0 say "a quantized copy of a supported model" in the block text, which the issue asks for.

## 6. What already exists that may be overlooked

- **Weights-only on the Qwen3.8 and Qwen3.6 27B family.** 580 and 248 safetensors fine-tune repos by Hub count. The bench reading of
  `Qwen/Qwen3.6-27B` and `sensenova/SenseNova-6.8-Code2Video` is weights-only. A run needs no new code. `tt-hous/qwen3.6-27b` appears in the preflight's
  list of base options. Not tried: whether the Qwen3.6-27B config matches the bundle closely enough for stage 2 to pass at 31 of 32 tokens.
- **The Qwen3-Next family sibling path.** The fusion-port log compared Qwen3-Next-80B-A3B with Coder-Next and found identical numbers in every
  dimension the fused ops read (48 layers, 512 experts, top-k 10, hidden 2048). It was rejected as a port pair because nothing differs to port, which
  makes it the simplest weights swap onto the Coder-Next bundle. The hand run of Qwen3.5-35B-A3B used the same bundle's code with a 41-line loader patch
  (PR 246) on two dies.
- **The weights swap with Coder-Next.** The coder itself is `raahemnabeel/qwen3-coder-next`, a container (v5.1) package. A fine-tune of it is the
  nearest model the box already runs. #60 and #73 block it. The Hub lists 23 safetensors fine-tune repos of Coder-Next and 27 of Next-80B.
- **The text-model route (PR 245, 247).** An adapter boundary (`orchard/text_runtime.py`, `REGISTRY`): a new text runtime supplies facts, a weights path,
  five commands and a serving command, and the harness runs them as child processes. The stages recompute every number from files and no stage boots the
  coder, so the coder keeps its board. MiniCPM3-4B is the first adapter. `docs/text-runtime.md` says a hybrid with recurrent blocks records
  `layer_NN.mixer` next to `layer_NN`, which describes Granite-4.0-H-1B. A Granite entry is not in the registry yet.
- **The audio route for diffusion models.** `orchard/audio_*.py` and the audio stage table (stage 1 references, stage 2 chip test and bars, 3 to 7
  skipped) brought three Stable Audio 3 checkpoints to the chip with the text encoder on the host. The code is not specific to one model. The three
  checkpoints share a lineage, and I did not check whether `stable-audio-open-1.0` shares their blocks.
- **`fusion-candidates` (PR 235).** Read only. Before a chip is used, it says which ops of a fused bundle apply to a new model.
- **The kernel probe and the tt-lang pair.** The pinned simulator and compiler wheel agree on the chip for 17 of 20 kernels, including the
  reductions that failed on tt-lang 1.1.6 (issue 205's 0 of 7). On the ladder the coder passed `row_sum` 6 of 9, `center` 1 of 6 and `rmsnorm` 0 of 3.
  A model that needs only elementwise kernels is within reach (Granite's state update compiled and ran). A model that needs a reduction the coder
  cannot write is out of reach.
- **Models that stopped on purpose.** Nemotron-3.5-Bolt-74B (`nearest-model-missing`; Q8_0 ggml-packed weights; omni modality) and pplx-decider-v1.2
  (blocked at stage 2 on 2026-10-19; a later restricted-head parity run by hand measured 14 of 14 answers equal on chips). Bolt-74B needs three things at
  once: a bundle, stage 0 reading the ggml block layout, and a modality. No open issue covers all three, so it is not on the ranked list. Its newer
  sibling `NVIDIA-Nemotron-3.5-Bolt-E4B-BF16` (8.0B, gemma4 arch) is a smaller target that depends on row 6.
- **tt-model-manager branches.** Read only. Branches that bear on packaging: `feat/cohost-runtime`, `feat/cohost-kind`, `feat/deploy`,
  `feat/serve-n-chips`, `feat/board-aware-chips`, `fix/ready-card-non-chat-kinds`, `feat/card-profile-kv-capacity` (two had a merge of main on
  2026-10-29). The cohost branches and `serve --chips N` bear on #23 (1-chip and 4-chip profiles). I did not read their diffs.

## 7. Issue texts that are older than the code

- **#62** says a supported model "has no path". The Llama-3.1-8B dry-run shows a stock-runtime route in the preflight and the README documents the
  `stock-runtime` class. The open part is #93's chip result and the question the issue asks last (which models run on Blackhole).
- **#28** lists "A base served only by a container package" and an audio design "not built". The audio route exists and section 6 describes it.
  The checklist should be updated when #60 closes.
- **#143** (audio runs boot the coder before stage 0) is open. The project log of 2026-10-29 says it is not fixed and that a strict xfail test states
  the wanted behaviour. Every audio run still holds a board it does not use.
- **#205** (0 of 7 reductions compile) has a tested answer in the pinned pair (17 of 20 on the chip). The issue stays open for the 3 that fail.

## 8. Recommended sequence for the next round

| Step | Issue | First model | Why this model | Stop and report if |
|---:|---|---|---|---|
| 1 | #93 with #62 | `meta-llama/Llama-3.1-8B-Instruct` (fallback `Qwen/Qwen2.5-7B-Instruct`) | the pilot both #61 and #93 name; one chip; the dry-run shows the route | stage 2 does not run the demo on the Blackhole board, or the server's log lines do not match the parsers |
| 2 | #60 with #73 | `huihui-ai/Huihui-Qwen3-Coder-Next-abliterated` | a fine-tune of the coder's own model, so no new code; the nearest container is installed | a one-chip probe shows the container reads its own repository (the #73 failure) |
| 3 | #183 | `Qwen/Qwen3-Next-80B-A3B-Instruct` (config only) | its 512 experts sit at the scan's second limit; no chip needed | none; two days of software |
| 4 | #184 | `Qwen/Qwen3-Next-80B-A3B-Instruct` through the Coder-Next bundle, then `huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated` | the first is a pure swap; the second repeats the 35B-A3B result through the harness | disk (162.7 GB of weights) or agreement below the bar |
| 5 | #63 | an FP8 copy of a model that already has a bundle (to be chosen from the 340 safetensors copies of Qwen3.8-27B; I did not choose one) | the runtime's answer decides the class | the runtime refuses the format; record the refusal and its message |

After step 2 passes, PR 246's Qwen3.5-35B-A3B recipe package can serve as the nearest model for the 35B-A3B family, which turns step 4 into a family
sweep.

Suggested new issues (not filed; this task allows comments only): add a Granite-4.0-H-1B adapter to the text runtime registry (10 Hub fine-tunes, 7
with safetensors, and the model is already ported); a bring-up of `Qwen/Qwen3.6-27B` through the existing weights-swap path to check the claim in
section 6; and a Qwen3 dense size check for the stock route (Qwen3-8B, 14B and 4B reach 1200 or more, 370 and 1200 or more fine-tune repos).

## 9. What I could not verify

- Whether any model in the ranked table passes. No chip ran. Every "reach" figure is a count of Hub repos, with LoRA merges, repos in different
  layouts and repos nobody would serve inside it.
- Per-chip DRAM. The repository states per-die bandwidth (512 GB/s) in `docs/run-logs/2026-10-28-decode-floor-break-even-hazards.md` and does not
  state capacity. The "fits one chip" statements assume 32 GB per chip and are arithmetic on parameter counts.
- Which drive holds the Hugging Face cache. Free space was read from the filesystem listing on 2026-10-29 for two mounts only.
- Candidate classes other than the two weights-only readings (SenseNova, Qwen3.6-27B). The rest came out full-port against a dense 27B base, which
  is expected for any other architecture, and a MoE model compared with a dense base says only "different". The comparison with the right base
  failed (section 2).
- The gated licenses: whether the account's token can read the Llama and Gemma repositories.
- That the catalog listing (about 50 packages on 2026-10-29) matches issue 59's list of 2026-10-24 name for name. I did not recount.
- Whether `Qwen3.6-27B` and its fine-tunes pass stage 2 against the Qwen3.8-27B bundle. Stage 0's reading is necessary and does not suffice.
- Effort figures. They are estimates written before any of the work, from the sizes of comparable work in the project log.
- PR 245, 246 and 247 are open. I read their descriptions and, for 245, its docs page. I did not run their tests.
