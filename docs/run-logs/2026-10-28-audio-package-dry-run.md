# Audio package builder: a dry run on the Stable Audio 3 small-music run (2026-10-28, issue 169)

The operator heard the clips of the run `stabilityai--stable-audio-3-small-music-pr165c` and asked for a package staged locally, so that publishing is one step
later. This log says what was staged, what the gate checked, what was not run, and what the operator has to decide before anything is published.
Paths are placeholders (`<run>` is the run directory). Nothing was uploaded, published, pushed or listed, and no chip, lease or port was used.

## What was built

| Piece | Where |
|---|---|
| The builder, the card renderer and the gate | `orchard/audio_package.py` |
| The command | `tt-orchard audio-package RUN_DIR --namespace ORG --bench FILE --demo-repo DIR [--snapshot DIR] [--check]` |
| Tests (105, with fake runs) | `tests/test_audio_package.py` |
| Hostile-input readers (4 registered) | `tests/hostile_inputs.py` |

## What was staged

`<run>/stages/7/package/stable-audio-3-small-music-blackhole/` (20 files, 532 KB, no weights). `package-record.json` has the sha256 of every file.

| File | What it is |
|---|---|
| `README.md` | The card, rendered from the run's numbers. |
| `LICENSE.md` | The Stability AI Community License Agreement, byte for byte from the pinned snapshot (revision `8bdfb53c`). |
| `NOTICE` | The attribution sentence from the agreement, with the agreement's own characters (a copyright sign followed by two spaces), then one line saying what the package does not change. |
| `tt-model.yaml` | A draft v5.1 container manifest, kind `http-server`. It loads in the installed tt-model's loader. It names a tt-metal checkout it cannot supply, so it has not been built. |
| `run.sh` | A launcher that starts the job server and the page without tt-model. |
| `runtime/` | `chip_generate.py`, `chip_dit.py`, `make_reference.py`, `audio_reference.py`, `pinned_files.py`, copied unmodified into the directory layout they expect, plus a one-line `orchard/__init__.py`. |
| `demo/` | `serve_audio.py` (the `POST /v1/audio/jobs` server), `audio_client.py`, `audio_app.py` (the Gradio page) and `requirements.txt`, copied from the separate demo repository, which was only read. |
| `docs/running.md` | How to start it. |
| `evidence/` | `audio_result.json` (the run's stage 2 result) and `bench-chip.json` (the sweep the timings come from). |

The only edit to a copied file: the page's text gets the words "Powered by Stability AI." in front of its first sentence (`demo/audio_app.py`). The record has the hash
of the file before and after. The builder refuses to stage if the demo page no longer has the line it edits.

## Package kind: v5.1 container manifest of kind `http-server`

I read the tt-model-manager checkout (read only; nothing was changed or pushed there) and its branch list. The kinds a package can have are:

- v5.1 container: `vllm-plugin`, `vllm-fork`, `tt-dit-server` (an ASGI app run with uvicorn), and `http-server` (the model's own HTTP server, started by its own command line).
- v6 thin: `vllm` and `tt-dit-server`. The branch `feat/v6-thin-non-vllm-kind` adds the non-vLLM thin kind. It serves an ASGI app with uvicorn.

The job server in the demo repository is a standard-library `ThreadingHTTPServer` with its own launch line, which is what `http-server` is described for. A v6 thin
bundle would need an ASGI wrapper around it, which would be code this builder would have to write and nobody has run. The sibling music package
(`jashansinghTT/minimax-music3-blackhole` and the companion repository) also ships a draft manifest next to a launcher, so `run.sh` is the way the package runs today and
the manifest is for a later build. The catalog reads a built package, so listing this one needs an image built from a tt-metal checkout first.

## The benchmark source

The run's own stage 2 repeats (3 per model) were made before the warm-up rule: the first repeat includes kernel compilation and none has a warm-up. Under
`docs/benchmark-rules.md` (PR 187, not merged when this was written) that is "unwarmed repeats", and the builder refuses to call it a benchmark. The card's timings
come from the sweep file `bench-chip-small-music.json` of PR 187's evidence (one loaded process, one warm-up and 3 measured runs at 10, 30, 60 and 120 s, so the
maximum length is recorded). The builder checked that it is the same model and the same prompt, steps, guidance scale, sampler, seed and sample size as the run. It is not the
same run: the sweep does not record the library commit or the model revision, so the card says the timings are from one process that loaded the model once.
`label_for` in `orchard/audio_package.py` repeats the rule (3 measured runs after at least one warm-up); when PR 187 merges, it can import `resident.label_for` instead.

The card's agreement rows are the run's chip against the CPU reference (final-latent PCC 0.9905, relative L2 0.1385; waveform PCC 0.9807; log-spectrum distance 0.1168) and two
environment-floor rows, each labelled `measured (provisional bars)`.

## What the gate checked

`tt-orchard audio-package ... --check` returned: the package passes the gate. The gate has these checks, each with a test that fails without it (a mutation table is below).

- License, each element on its own: the license named and linked to the package's `LICENSE.md`; research and non-commercial use permitted; registration with Stability AI and the
  USD $1,000,000 revenue term, with the agreement's own sentences quoted and looked up again in the package's `LICENSE.md`; "Powered by Stability AI" on the card, the page and every file in `docs/`;
  the weights stated as not included and as gated; the Gemma Terms of Use named with its link; the runtime code stated as Tenstorrent's; the reader told to read the license; the front matter `license: other`
  with the license's name and link.
- A copy of the agreement is present and is the Stability AI Community License Agreement; `NOTICE` contains the attribution sentence exactly as the agreement writes it. The
  builder reads the sentence from the agreement, so the copyright sign and the two spaces after it are kept.
- Refused wording: open or commercial license claims, quality claims, "official", "endorsed" and logo images.
- No weights file by suffix, no file over 20 MB, no home path, host name or token (`orchard/scrub.py`).
- Numbers: only in the Numbers table; each row labelled `benchmark`, `single sample`, `unwarmed repeats`, `measured (provisional bars)` or `TODO`; a `benchmark` row only when the evidence's runs earn it
  (recomputed); each value is a number in its evidence file, rounded to the digits shown.
- The builder refuses a run with no benchmark-labelled size, a label or spread the runs do not support, no 40-character model revision or library commit, an agreement that failed, a control that did
  not tell the fine-tune from the base model, a snapshot of another revision, and a package directory that already holds weights or anything else.

Mutation table: 62 guards, each removed in turn with the whole test file run. First pass: 5 survived. Four were given a sharper test (the front-matter `license: other` check, the two
number-in-prose checks, which were hidden behind each other, and the provisional-bars check, which the label text satisfied by itself). The fifth, an explicit link refusal for source files,
was redundant with `safeio` (which refuses links) and was removed. Second pass: all red.

One bug the tests found: a revision made only of digits would be read by YAML as a number, so the manifest quotes it.

## What was not run

- A boot of the package on a chip. All boards were in use. The command that runs it (written as text; do not run it without a lease):
  `gozer run --chips 1 --who "operator" --reason "boot check of the staged audio package" -- env ORCHARD_TTNN_PYTHON=<python with ttnn> ORCHARD_AUDIO_SRC=<stable-audio-3 checkout at the commit on the card> <run>/stages/7/package/stable-audio-3-small-music-blackhole/run.sh`,
  then `curl -s localhost:20100/v1/health`, and a `POST /v1/audio/jobs` with `{"prompt": "...", "duration": 10}` followed by `GET /v1/audio/jobs/<id>/audio`.
- A pull from the Hub. The package is not on the Hub, so a pull cannot be tried before it is published. After a private upload, pull it into an empty directory and run the boot check there.
- The shipped server against a loaded model. Its generator starts a new process for each clip, so each clip pays the model load. The card says so.
- Quality. No quality evaluation was run. The operator listened to the clips; the run's recorded limits still say nobody had, and this card makes no quality claim either way.
  If the operator wants "informally listened to" on the card, it needs the operator's wording.
- `build_bundle.py`'s audio section (the review bundle) does not list the demo files. PR 187 changes that file, so the edit is left until it merges.

## The commands that would publish it (text; not run)

These are in `<run>/stages/7/AUDIO_PUBLISH_COMMANDS.txt`. Each needs the operator's explicit go-ahead.

```
hf upload --repo-type model --private episod/stable-audio-3-small-music-blackhole stages/7/package/stable-audio-3-small-music-blackhole .
hf repos settings episod/stable-audio-3-small-music-blackhole --no-private          # a separate decision, after a boot check and a pull
tt-model publish episod/stable-audio-3-small-music-blackhole                        # a separate decision; the catalog reads a built package
```

## For the operator: decisions before anything is public

1. **Whether Tenstorrent can publish under this license at all.** The Community License lets a licensee use the model for research and non-commercial purposes, and for commercial purposes
   only below USD $1,000,000 in annual revenue, counting the licensee and its affiliates "regardless of whether that revenue is generated directly or indirectly" from the model. Above that, "any licenses granted to You under this Agreement shall terminate", and the
   licensee must request a license from Stability AI. Tenstorrent's revenue is above that amount. Publishing a package that carries the model's license text and a runtime for it from a Tenstorrent account
   (or a personal account that Tenstorrent is affiliated with) may need the commercial license from Stability AI. That is a company-level legal question. The card is written so it does not
   claim an answer. Review it with counsel before the repository is made public. Keeping the repository private, or publishing only the runtime without the model's name and license text, are the
   other options.
2. **The model's name.** The license (section 4(c)(i)) says no name or mark owned by Stability AI is used "except as required". The repository and the card use the model's own id
   (`stable-audio-3-small-music`) to say which model the package runs. The gate refuses logos and words such as "official" but cannot decide whether naming the model is allowed. The same question
   applies to the page's heading, which the demo repository wrote ("Stable Audio 3 small music on Tenstorrent Blackhole").
3. **A mismatch in the copied page.** The page's note still says "Only 10 second clips have been checked on the chip", while the card has timings for clips up to 120 s. Both are true (the timings are
   not a check of the clip), and the page was not edited beyond the attribution. Edit it in the demo repository if the operator wants them to agree.
4. **The namespace.** The dry run used `episod`, the account of the earlier publications. The commands take any namespace.
5. **Whether the catalog should list it.** It needs a built container image, and a boot of that image on a chip.
