# Cards for canned models: the tt-model conventions

Operator instruction (2026-10-30), for the card of every can: "note things on the cards. benchmarks. use the tt-model-manager vernacular for model cards.
tt-model-ranker helps us understand what good is."

This page records what the tt-model-manager conventions are, how the Stable Audio 3 can applied them, and what the card viewer (tt-model-ranker) reads. The
other cans (MiniCPM3, ACE-Step, minimax-music3-fiona-crapple, Qwen3.5) can adopt it. Sources, all read only: the `tt-model-manager` checkout (the `main`
branch and the open branches `stisi/feat/model-card-standardizer`, `docs/model-card-skill-cleanup`, `stisi/docs/model-card-field-set`), the installed
`tt_kernel` package (`manifest.py` `CardSpec`, `container_manifest.py` `REQUIRED_CARD_SECTIONS` and `card_publish_gaps`, `build.py` `render_model_card`),
the published cards `episod/audio8-asr-infinite-p150` and `episod/clef-p300` and the card viewer `tt-model-ranker` (its README and `card_viewer/`).

## What `tt-model package-thin --readme` does with a card

It stages the file verbatim as `README.md`. For a v6 thin bundle nobody renders the card, so the can's author writes the text that `render_model_card()` would
render for a container package. The conventions below come from that renderer and from the `model-card-standardizer` skill in tt-model-manager.

## Front matter

The Hub reads these keys. `tt-model search` filters on the tags.

```yaml
---
tags:
- blackhole            # the arch
- p150                 # one tag per board the card claims (p150, p150x2, p300c, p300cx2, QB2)
- tt-dit-server        # the kind (or vllm)
- thin                 # a v6 thin bundle (package-thin adds it at push; clef-p300 carries it)
- tt-model-cache       # the tag tt-model's `search` filters on
- text-to-audio        # domain tags are optional
license: other         # an SPDX id, or "other" with the next two keys
license_name: stability-ai-community-license   # a lowercase slug (the Hub rejects capitals and spaces)
license_link: LICENSE.md
pipeline_tag: text-to-audio
base_model:
- stabilityai/stable-audio-3-small-music
---
```

Do not write `tt-model-catalog`: `tt-model publish` adds it, and it means the repository is listed in the catalog. Do not write a tag or a sentence that says the
model is verified: tt-cli decides that (a curated list, or a copy in the Tenstorrent organisation made by `tt-model verify`).

## Sections, in this order (names as the renderer writes them)

| Heading | What goes in it |
| --- | --- |
| `# <name> on Tenstorrent <boards>` | the title |
| a block quote | attribution: the upstream model, who made it, whether it was changed |
| a paragraph | what the model is (size, architecture) and what it does |
| `Runs on Tenstorrent **<board>** (mesh <mesh>)` | one line |
| `Packaged with tt-model-manager <version> (manifest schema 6)` | one line |
| `## At a glance` | a two-column table: architecture, hardware, context or output limits, license |
| `## Intended use` | what it is for, and what it is out of scope for |
| `## Prerequisites` | the tt CLI (or tt-model alone), the board, and what the user must bring (an account, a token) |
| `## Quickstart` | `tt serve <repo>`, the line that says what it downloads, how long the first start takes, the ready line, and `Without tt-cli: tt-model serve <repo>` |
| `## Serving profiles` | a table, only when there is more than one profile |
| `## Capabilities` (older cards: `## Using it`) | the API. A model that is not an OpenAI chat server says so and gives its routes, request fields and an example |
| `## Expected performance` | required for the catalog (see below) |
| `## Limitations` | required for the catalog: what is not implemented or not tested, what the model should not be used for, failure modes |
| `## Licensing` | when the weights carry a restriction beyond the code's license |
| `## Related packages` | sibling packages and the upstream model |
| `## Feedback` | the repository's Discussions page, `tt report issue` (for the `tt` tool only), support@tenstorrent.com |
| `## Provenance` | a `component | built from` table |

`card_publish_gaps()` names `performance` and `limitations` as the required sections. A card without them renders "Not provided by the package author." The
retired fields `architecture`, `status`, `out_of_scope_use` and `risks` fold into the description and the limitations (the renderer still reads them, with a
warning). Older published cards still carry `## Risks and safety considerations`; the current rule puts risks inside Limitations.

## Expected performance

- A table of accuracy against the reference (PCC, exact match or the modality's equivalent), with the reference's own figure and the prompt set.
- For a model that serves text, a speed table with the columns `profile`, `ISL`, `OSL`, `concurrency`, `N`, `TTFT`, `prefill tok/s/u`, `decode tok/s/u`, `E2EL`
  (`PERFORMANCE_COLUMNS` in `container_manifest.py`; the packager warns when one is missing). `N` is the number of runs behind each row.
- One sentence on how it was measured: the hardware, the prompt set, the tool.

tt-orchard cards also use a `Number | Value | Label | Evidence` table (the Granite card and the audio package card do). The card viewer reads it: only rows labelled
`measured` count as measurements, and the evidence column points at files in the bundle.

## What never goes on a card

From the standardizer skill: no bring-up history (how the port was made, what was tried, earlier versions), no opinion on how good the result is, no explaining away
a gap, and no implementation detail a user does not need to run the model or trust its output. A limitation a user can hit stays, stated as current behaviour.

## What tt-model-ranker reads

`python -m card_viewer CARD.md -o card.html --no-infer` (run it from the `tt-model-ranker` checkout; it reads local files only with `--no-infer`) writes one page.
What it takes from a card:

- the modality from `pipeline_tag` (`text-to-audio` gives `audio`). Audio jobs have no grader yet, so an audio card has no letter grade; the page shows the card's own
  numbers under "Measured on the card" and lists what is missing under "Not measured yet";
- from a section whose heading names a measurement (`Expected performance`, `Measured`, `Benchmarks`): every table row that holds a digit, keyed by its first cell, valued by
  the column headed `value`, `result`, `this package`, `this checkpoint` or `measured`. Columns headed `evidence`, `label`, `note`, `prompts` and `reference` are left out;
- the hardware from the card's board names (`p150`, `p300c`, `QB2`), the precision and sampling words in sentences that contain a measuring word (`measured`, `benchmark`, `latency`, `rows`).

A card that states its conditions in a sentence next to the table (date, host load, board, software version, warm or cold kernel cache) gives the viewer and a human reader the
same information. The Stable Audio 3 card does that under its numbers table.

## What the Stable Audio 3 can did with it

The card is `pkgsrc/can/stable-audio-3-README.md` (a template; `scripts/can_stable_audio_base.py` fills in the license paragraphs, quoted word for word from the model's own
`LICENSE.md`, and the numbers table). The run log is `docs/run-logs/2026-10-30-stable-audio-3-thin-bundle.md`. Points another can can reuse:

- Benchmarks are measured on this machine, through the served bundle where possible, and each row names its conditions: date, host load average, chip, ttnn version, and whether the kernel cache was empty.
- The numbers that depend on the machine (a CPU time on a loaded host) are labelled as such, with the load.
- A model whose license is `other` quotes the agreement's paragraphs word for word and does not summarise obligations. The base-directory script refuses a card whose quotation is not in the license copy.
- The card says nothing about quality, because the measurements do not cover it.

## A reference card

`raahemnabeel/qwen3-reranker-4b` (https://huggingface.co/raahemnabeel/qwen3-reranker-4b) is a public card for a text reranker that serves through `tt serve`. This
section describes what it contains and why each part helps a reader, as a target for a can's card. It is a description. The card was read on 2026-11-02 with
`huggingface_hub` from an isolated interpreter (`python3 -I`), text files only, and nothing was run. The repository's `code/` folder holds the author's notes, and
the precision table below is there and not in the README.

What the card contains, in the order a reader needs it:

- **Parity against a reference on the same token ids.** The `Expected performance` section opens with a table whose columns are `eval`, `metric`, `this package`,
  `HF reference` and `size`. The reference is named (the transformers implementation on CPU, in fp32 for one set and bf16 for the others). The rows give a
  correlation of scores, the mean and the largest score difference in logits, how often the top document matches, and a ranking metric on two public benchmark subsets. The
  `size` column says how many pairs or groups stand behind each row. A sentence under the table says the reference ran on exactly the same token ids as the
  device, so a difference comes from the device's arithmetic and not from tokenization. It says the benchmark rows show parity and are not leaderboard numbers. It
  also records where the package is worse (one group's top-1 flips on the multilingual subset). A reader can tell how far the device result is from the reference and how many
  samples that rests on.
- **Throughput per profile, with the error count.** A second table lists one row per serving profile and client count, with requests per second, tokens per second and the
  50th and 99th percentile latency. A sentence under it states the tool (the real server, closed-loop clients), the length of each run (45 seconds per row), the
  error count (zero), and the dies the numbers came from. A single-document latency table sits above it for the p150 profile.
- **Where the numbers do not apply.** The same sentence says the p150 rows ran on one die of a p300c, and the Limitations section says the p150, p150x2 and p150x4 boards
  run the same per-chip code and were not measured. The card separates what was measured from what the code implies.
- **A precision policy with the choice named.** The README gives the chosen policy in one Limitations bullet (bf16 activations, bfp8 attention, bf16 MLP weights) and the
  score error it costs. The comparison of four policies is in `doc/accuracy.md` in the repository: for each policy the weight formats, the mean and largest score
  difference, the correlation, the top-1 agreement and the run time, with the selected row in bold and one policy marked as the choice of the sibling 8B model.
  Under the table, a tuning that was measured and not adopted (autotuned matmul blocking) is recorded with its reason: it changed the end-to-end time by almost nothing
  and raised the mean score error. A reader of that file learns why the shipped policy is not the fastest one, and that the author tried the obvious alternatives.
- **A specific Limitations section.** Each bullet is something a user can hit: the build is a bring-up build, the score error means candidates within about 0.25 of each
  other can swap places, the input is cut at 8192 tokens from the document side, more chips add throughput and not single-request speed, a failed chip is skipped until restart,
  and which boards were validated. None of them is a general disclaimer.

Two further things the card does that other cans can copy. It states what one chip holds ("every chip holds a full copy of the model") in the description and in the
profile table, so the reader can see that the profiles are data parallel (see the data-parallel block of `tt-orchard residency`). It also ends with a provenance table
that names the digest of the code in the image and says the tree was dirty.

### What the `card_profile = "catalog"` gate checks today

`orchard/can.py` (`card_problems`) checks two things and nothing else about the card's content:

| Item on the reference card | Checked by the gate today? |
| --- | --- |
| An `## Expected performance` section with a table row whose cell has a number | Yes |
| A non-empty `## Limitations` section | Yes |
| A parity table that names its reference | No |
| The size of the sample behind each row | No |
| An error count with the throughput rows | No |
| The tool, run length and dies a throughput table came from | No |
| A precision policy with the chosen one named | No |
| A rejected tuning recorded with its reason | No |
| Limitations that are specific (the gate accepts any text) | No |

So a card with a single row such as `| decode | 12.5 |` and a one-line Limitations section passes. The gate does what `card_publish_gaps()` does and no more. Whether a
number is a measurement, and what it was compared with, is left to the author and the reviewer. One cheap check that fits the gate is that an Expected performance
table with a `reference` column fills that column with text in each row that has a number; it is proposed in a public issue and not built here.
