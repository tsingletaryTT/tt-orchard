# Audio package rebuilt from current main: decoder on the chip (2026-10-29, issue 161)

Operator instruction: "rebuild from current main" for the Stable Audio 3 small-music package. The private Hugging Face repository `episod/stable-audio-3-small-music-blackhole`
was built before PR 211 and PR 224 (issue 187). It carries the diffusion transformer on the chip and the autoencoder decoder on the host. This log says what the builder now
copies and checks, what was staged, the gate output, how the staged package differs from the uploaded one, and what is not done.

Paths are placeholders: `<run>` is the finished run directory `stabilityai--stable-audio-3-small-music-pr165c`, `<overnight>` holds the 2026-10-28 evening chip runs
(see `2026-10-29-overnight-stable-audio-decoder.md`), `<stage>` is a new directory beside them, and `<pkg>` is `<stage>/stable-audio-3-small-music-blackhole`.
Nothing was uploaded, published, pushed to the Hub or listed. No chip, lease or port was used, and the coder container was not touched.

## What changed in the builder

| Change | Where |
|---|---|
| The runtime copy now includes `chip_decoder.py`, `audio_runtimes.py` and `orchard/resident.py`. `chip_generate.py` loads `audio_runtimes.py`, which loads `chip_dit.py` and `chip_decoder.py` and imports `orchard.resident`. The uploaded package has none of the three, and its `chip_generate.py` is an older version than main's. | `RUNTIME_FILES` in `orchard/audio_package.py` |
| A test reads every shipped runtime file for the files it imports or loads by name and requires each one to be in the package. | `tests/test_audio_package_rebuild.py` |
| The packaged generate command (`pinned_generate.py`, the script the server starts for each clip) adds `--chip-decoder` by default. `--host-decoder`, or `AUDIO_HOST_DECODER=1` in the environment, keeps the decoder on the host. The server cannot pass flags of its own, so the variable is how `run.sh` users select it. Both flags together exit 2. `run.sh`, `docs/running.md` and the card document the fallback, and the gate checks that `run.sh` and `docs/running.md` name it. | `orchard/skills/audio-chip-templates/pinned_generate.py`, `run_script`, `running_doc` |
| `chip_generate.py` itself is unchanged (its `--chip-decoder` stays an opt-in flag for the harness). Only the packaged wrapper changes the default. One existing test of the wrapper now passes `--host-decoder` to keep its old argument check. | `tests/test_audio_package.py` |
| A sweep file made with the decoder on the chip carries `chip_decoder_call_s` in every run. The card then has a decoder row per size (median of the runs) and a decoder weight-upload row. The part is all or none: a sweep where some runs lack it is refused. The spread is recomputed from the runs, as for the other parts. | `bench_sizes`, `number_rows` |
| The benchmark and the agreement result have to agree about the decoder. A sweep with the decoder on the chip needs a result whose `runs.chip.chip_decoder` is true, and the reverse. Otherwise the card's timings and agreement would describe different paths. | `facts_from_evidence` |
| CPU float32 timings: a result's `reference_runs.reference` and `reference_runs.host_same_env` become single-sample rows, only when they ran on the CPU in float32 with the chip run's own prompt, steps, guidance, sampler, seed, sample size and duration. | `cpu_timings` |
| The card's "What runs where", the Limitations and the limits paragraph follow the evidence. The gate refuses a card that says the decoder runs on the chip without that evidence, and one that says only the diffusion transformer runs on the chip beside evidence of the decoder on the chip. | `render_card`, `card_problems` |
| `--listening-note TEXT`: the operator's dated listening judgment, printed under its own heading. It needs a date (YYYY-MM-DD) and is one line. The card keeps "No quality evaluation was run". The gate's quality-claim and timing patterns still scan the note, and the gate refuses a note section with no date. | `check_note`, `render_card` |
| `--result FILE`: use that audio result in place of the run's `stages/2/audio_result.json`. Run files are never edited. | `read_facts`, `stage_package`, `cli.py` |
| `python -m orchard.audio_result_derive RUN_DIR CHIP_RUN_DIR OUT_JSON`: writes the result file for a later chip run. It replaces `runs.chip`, `agreement.chip_vs_reference`, `runtime` and `limits`, drops `agreement.chip_vs_host_same_env` and `repeats.chip` (they belong to the earlier chip run), and keeps the CPU reference, the environment floor and the control. `derived_from` records the base file's sha256 and what was replaced, dropped and kept. It refuses another revision, library commit or setting, a run with no decoder timing, and an agreement that did not pass. | `orchard/audio_result_derive.py` |
| `--probe FILE` and `--agreement-report FILE`: optional evidence files (JSON objects), copied to `evidence/decode-probe.json` and `evidence/audio-agreement-report.json` with the home directory replaced by `<home>`. The gate compares them with the sha256 in `package-record.json`. | `extra_evidence`, `RECORDED` |
| `--out DIR`: stage into a new directory, and write `AUDIO_PUBLISH_COMMANDS.txt` beside it. Without it nothing changes: the package goes under the run and the file under `stages/7`. | `main` |

Pinned files and hashes: `pinned_files.py` and `check_revision.py` are copied unmodified, as before, and the gate's `run.sh` checks (revision, `check_revision.py`, private cache,
`AUDIO_PIN_REVISION`, `pinned_generate.py` as the script) are unchanged and pass.

## What was staged

`<pkg>`: 28 files, no weights.

- Numbers: `sweep-music.json` from the overnight lead (small-music, transformer and decoder on the chip, 1 warm-up and 3 measured runs at each of 10, 30, 60 and 120 s;
  `generate_and_decode_s` medians 0.819, 1.394, 2.561 and 4.318 s; host load average about 10 at the start and end).
- Agreement: the run `e2e1` of the overnight lead (10 s, one clip, its own process, both parts on the chip), compared with the CPU reference of the pr165c run by
  `audio_stages.compare`. Final latent PCC 0.9905 (relative L2 0.1385), waveform PCC 0.9758 (relative L2 0.2190, log-spectrum distance 0.1941). These match the numbers the overnight log
  gave from `audio_reference check --cross-precision` (waveform PCC 0.976, relative L2 0.219). The latent figures equal the earlier DiT-on-chip run's, as expected: the decoder does not change the latent.
- CPU timings on the card: 24.795 s (reference environment) and 4.087 s (the chip run's own environment), both one 10 s clip in float32, so single samples. Neither is a repeated measurement. The overnight log's
  3.9, 8.0 and 6.0 s CPU runs come from a separate benchmark directory and are not in the result file, so they are not on the card.
- Evidence files: `evidence/audio_result.json` (derived, see below), `evidence/bench-chip.json` (the sweep, byte for byte), `evidence/audio-agreement-report.json` (all 56 arrays of the comparison),
  `evidence/decode-probe.json` (the decoder alone on the 174-frame latent: per-layer error against CPU float32, the bfloat16 floor, CPU and chip decode times). The first probe run, which gave waveform PCC -0.02 before the precise-attention fix, is not included.
- The result file is derived because the run's own `stages/2/audio_result.json` is the PR 165 result (diffusion transformer on the chip, decoder on the host, agreement PCC 0.9807).
  It is not edited. `audio_result.derived.json` was made from it and from `e2e1`; its `derived_from` entry says so, and the card says so.
- The control (fine-tune against base model) is the PR 165 control, measured with the decoder on the host. It was not repeated with the decoder on the chip, and the card states that.
- The listening note on the card is the operator's: dated 2026-10-29, "the operator listened to the clips from the decoder-on-chip runs of this model and judged them good". It is not a measured quality metric. The wording of which clips was taken from the instruction and has not been checked against a list of clip files.

The license section, the NOTICE sentence, "Powered by Stability AI", the Gemma terms and the Community License statements are produced by the same builder code as before and are checked by the same gate.

## Tests and mutations

`tests/test_audio_package_rebuild.py` has 49 tests, and `tests/test_audio_package.py` now runs 249 (246 before; the three added runtime files add three parametrized cases; one test was edited). Each new guard was removed once and a test went red: the three
runtime files, the default `--chip-decoder`, the removal of `--host-decoder`, the environment variable, both flags together, the result file, the decoder consistency check, the
two card claims, the listening-note date (builder and gate), the CPU dtype check, the fallback documentation check, the comparison of the extra evidence files, the home-path
replacement, the location of the commands text, the all-or-none decoder part, and six checks of the derived result (revision, settings, decoder timing, agreement, the dropped stale
agreement, the copy of the base). Two gaps were found by the table or by reading the output: the gate's "card says it was derived" check was green until it got a test, and
`repeats.chip` of the base result was carried into the derived file until a test and a mutation covered it.

Wider run (xdist, the audio, README, hostile-reader, scrub, path, pinned-file and version test files): 2,793 passed, 2 skipped, 214 expected failures and 2 failed before the last fixes. Both
failures that remained are on `origin/main` as well and do not involve this change: `updates.py` is missing from the hostile-reader backlog list and `--updates-config` is missing from the
README (both from the updates-watch PR). This change's own README flags, the `--host-decoder` entry in the README test's table of other tools' flags, and the derive command's file read
through `safeio` came out of that run.

## Gate output

    $ tt-orchard audio-package <run> --namespace episod --bench <overnight>/sweep-music.json --demo-repo <demo checkout> --result <stage>/audio_result.derived.json \
        --probe <overnight>/chip2/decode_probe.json --agreement-report <stage>/audio-agreement-report.raw.json --listening-note "2026-10-29: ..." --out <pkg>
    staged 27 files in <pkg>
    the package passes the gate. Nothing was uploaded or published; the commands that would publish it are in <stage>/AUDIO_PUBLISH_COMMANDS.txt (text only).

The same command with `--check` passes after the last file change. The gate scrub found no home path, host name, token-shaped string, `.tt_ai_key` or `litellm` in the package
(a separate grep for each, case-insensitive). While checking the package without a chip, importing from it created two `__pycache__` directories inside it. They were removed
and the gate was run again.

## Checks without a chip

- `check_revision.py` from the package, run with the stable-audio venv and the library checkout at the pinned commit, resolved small-music to revision `8bdfb53c...` through a private cache (exit 0).
- `pinned_generate.py` from the package, then `chip_generate.py --help`, printed the `--chip-decoder` option: the wrapper, the real library and the pinned cache work together, and no model ran.
- `audio_runtimes.py` and `chip_decoder.py` import from the package copy, and `orchard.resident` resolves to the package's own `runtime/orchard/resident.py`.
- `bash -n run.sh` passes.
- A boot on a chip was not run. The exact commands and what they check are in `<stage>/BOOT_CHECK.txt`; another agent runs them on board 1.

## Difference from the uploaded private package

Read with `hf download` into a scratch directory (read only). File by file, staged against uploaded:

| File | Status |
|---|---|
| `README.md` | changed: decoder on the chip, 10 to 120 s sweep, decoder rows, CPU rows, new agreement numbers, limits, derived-result and listening-note sections |
| `LICENSE.md`, `LICENSE-runtime`, `NOTICE`, `requirements.txt`, `tt-model.yaml` | unchanged |
| `run.sh` | changed: two comment lines about the host-decoder fallback |
| `docs/running.md` | changed: step 4, the decoder and its fallback |
| `demo/audio_app.py`, `demo/audio_client.py`, `demo/serve_audio.py`, `demo/requirements.txt` | unchanged (the demo checkout is the same) |
| `runtime/orchard/__init__.py`, `audio_reference.py`, `pinned_files.py`, `skills/audio-chip-templates/check_revision.py`, `skills/audio-reference-templates/make_reference.py` | unchanged |
| `runtime/orchard/skills/audio-chip-templates/chip_generate.py` | changed (the decoder flag and the resident runtime of PR 211) |
| `runtime/orchard/skills/audio-chip-templates/chip_dit.py` | changed (float32 attention path and differential attention from the overnight lead) |
| `runtime/orchard/skills/audio-chip-templates/pinned_generate.py` | changed: the decoder default and fallback |
| `runtime/orchard/skills/audio-chip-templates/audio_runtimes.py`, `chip_decoder.py`, `runtime/orchard/resident.py` | new |
| `evidence/audio_result.json`, `evidence/bench-chip.json` | changed (new files) |
| `evidence/audio-agreement-report.json`, `evidence/decode-probe.json` | new |
| `package-record.json` | changed (hashes, the listening note, the new files) |
| `.gitattributes` | only on the Hub. `hf upload` without `--delete` leaves it |

The upload command is in `<stage>/UPLOAD_COMMAND.txt` as text: `hf upload --repo-type model episod/stable-audio-3-small-music-blackhole <pkg> .`. It was not run.

## Not done

- No upload, no change to the Hub repository, no publication. The package is private on the Hub today and the repository's visibility is the operator's decision (the license note in `2026-10-28-audio-package-dry-run.md` still applies).
- No boot on a chip, and no download from the Hub. The card's two TODO rows stay TODO.
- The agreement run (`e2e1`) and the sweep are different processes. Agreement was measured on one 10 s clip, one prompt and one seed. The 30, 60 and 120 s clips were timed and not compared with the CPU reference.
- The host was shared during the sweep (load average about 10), so the timings include contention.
- The control was not repeated with the decoder on the chip, and the decoder-on-host fallback has no numbers in this card.
- `audio_bench.py` and `chip_decode_probe.py` are not in the package. The package alone does not reproduce the card's numbers; the repository does.
- The listening judgment is the operator's, stated as an instruction to the agent. No one checked it against specific clip files.
