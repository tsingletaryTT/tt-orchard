# The Cans view: a read-only page for the canning step

Date: 2026-11-02. Operator request: "scope and build the canning tab".

## What a can is

A can is three things that live in three places:

1. A spec file, `config/can.*.toml`, read with the can module's own loader (`orchard.can.load_spec`).
2. A staged bundle directory, written by `tt-orchard can stage SPEC --out DIR`. The directory name is the spec's `[can] name`.
3. A Hugging Face repository. The spec names the weights repository and the pinned revision. The repository the bundle would be published to does not exist until the operator publishes it, and the canning step never publishes.

`can gate` and `can boot` print their results and exit. Today they record nothing a page could read, except `boot-check.json` in a `--work` directory that can be anywhere, and `stages/7/can.json` when `--run-dir` is given.

## What the view does

`GET /api/cans` answers with one row per spec. The page shows the rows in a new top-level view, "Cans", next to Orchard, Machine, Models and Telegraph. The view only reads. It sends no request to the Hub, opens no chip, runs no subprocess, writes no file and returns no path.

## Where each fact comes from

| Shown | Source | How it is read |
| --- | --- | --- |
| name, kind, mesh, device count, weights repo, pinned revision, card profile | the spec | `can.load_spec(path, base=checkout)`; the spec files are listed from `config/` (at most 60, names `can.<slug>.toml` only, links skipped) |
| link to the weights repository | the spec | built only when the repo id matches `ORG/NAME` with plain characters; otherwise the id is text with no link |
| Hub revision check | not available | the text "not checked here": the page makes no network request, so it cannot say that the pinned revision exists or that a published copy matches |
| staged bundle present | the cans folder | `<cans folder>/<spec name>`, opened beneath a pinned descriptor with no link followed; states: staged, not staged, not a directory, a link (not read), unreadable |
| SHA256SUMS check | the bundle | recomputed here with bounds (see below); states: matches, differs (with the first problems), missing, partly checked |
| gate result | recorded state file | the gate is not run by the page (see "Why the gate is not run") |
| last boot result, decode tok/s, start time, cache state | recorded state file | read when present; "no boot recorded" when not |
| card rows | `README.md` in the bundle | the `## Expected performance` table rows and the `## Limitations` lines, found with the gate's own section reader (`can._sections`), so a heading in a code block or a comment does not count |
| card check | `README.md` in the bundle | `can.card_problems(text, spec.card_profile)`, a pure function of the text |

The cans folder is `ORCHARD_CANS_DIR` when set in the page's environment, and `<runs folder>/cans` otherwise. It holds `<name>/` (the staged bundle) and, optionally, `<name>.can-state.json`.

## Why the gate is not run

`can.gate` is not a pure read of the directory with bounds:

- it runs `scrub_package`, which walks the whole tree and reads every text file with no total limit;
- it hashes every file twice (the sums check and the wheel comparison) and hashes the spec's wheel sources outside the bundle;
- it compiles every Python file in the bundle;
- it imports `tt_kernel` (tt-model's reader), which may be absent on the page's host, or heavy to import;
- it follows links when it reads (`rglob`, `read_text`).

The page runs the parts of it that can be bounded and cannot follow a link (the sums check and the card check), and shows the result of the full gate from a recorded file. A recorded result carries the digest of `SHA256SUMS` at the time. If the bundle's current `SHA256SUMS` has another digest, the page says the record is for an earlier bundle.

## The optional recorded-state file

`tt-orchard can gate`, `can boot` and `can stage` take an optional `--state-dir DIR`. When given, the command writes `DIR/<spec name>.can-state.json` after it finishes. Without the flag nothing is written, and the view works without the file. `can boot` also takes `--cache-state empty|warm`, because the boot check cannot tell whether the kernel cache it used was empty; the operator knows.

The file sits beside the bundle, not inside it. A file inside the bundle would be listed as unlisted by `SHA256SUMS` and fail the gate.

Format (schema 1, JSON, under 64 KB):

```json
{
  "schema": 1,
  "name": "granite-4.0-h-1b-blackhole",
  "gate": {"passed": true, "problem_count": 0, "problems": [], "bundle_sums_sha256": "<64 hex digits>", "at": "2026-11-02T10:00:00Z"},
  "boot": {"passed": true, "decode_tok_s": 31.4, "ready_s": 212.5, "cache_state": "empty",
           "failed": [], "bundle_sums_sha256": "<64 hex digits>", "at": "2026-11-02T10:30:00Z"}
}
```

`bundle_sums_sha256` inside each part is the sha256 of the bundle's `SHA256SUMS` file when that part was written (or null when the file could not be read). It is per part because a later `can gate` keeps an older `boot`, and the two can be for different versions of the bundle. `gate` and `boot` are each optional. A later `can gate` replaces `gate` and keeps `boot`. `problems` and `failed` hold at most 10 entries of at most 300 characters. The page checks every field (types, finite numbers, bounded strings) and treats a file with any bad field as unreadable: it shows "recorded state unreadable" for the piece and nothing from the file.

## Bounds

- Spec files: 60 per page, each read by the loader with its existing 256 KB limit. A spec that the loader refuses is a row named from the file name, with the loader's message (home paths and the checkout path removed).
- `SHA256SUMS`: at most 1 MB and 5,000 lines. Each listed path must be a plain relative path (no `..`, no empty part, no NUL). A path that is refused counts as a problem.
- Hashing: files are opened with no link followed and no wait (`O_NOFOLLOW | O_NONBLOCK`), and a file that is not a regular file is a problem. At most 1 GiB is hashed per bundle. When the limit is reached the check says how many files were not hashed and does not say "matches".
- Unlisted files: a walk of the bundle with `safeio.scan_beneath` (at most 5,000 entries, depth 8). A link or a special file in the bundle is a problem. A walk that was cut short is reported.
- Card: `README.md` is read up to 256 KB. At most 12 table rows and 12 limitation lines are shown, each cut to 200 characters.
- State file: 64 KB.
- Cache: the whole answer is rebuilt at most once every 30 seconds. One rebuild runs at a time.

## Shown when a piece is missing or broken

Nothing becomes an error page. Each piece has its own state and a plain sentence: "spec refused: ...", "no staged bundle", "bundle is a link and was not read", "SHA256SUMS is missing", "no gate recorded", "no boot recorded", "recorded for an earlier bundle", "recorded state unreadable", "card has no Expected performance section". If the whole cans folder is absent, every can shows "no staged bundle" and a note names `ORCHARD_CANS_DIR`.

A row's tone follows its worst piece: red for a failed gate, a sums mismatch, a failed boot or a refused spec; amber for any piece that is missing, partly checked or recorded for another bundle; green only when the bundle is staged, the sums match and a gate and a boot are recorded as passed for this bundle; grey when nothing is staged. Rows have no background tint: the tone colours text on the page paper, so one tone's text never sits on another tone's background, and each status is a shape plus a word.

## What is deliberately not shown

- Any path, host name or home directory. Messages are redacted with the page's `redact`.
- A live Hub state: whether the pinned revision exists, whether a published repository exists, its visibility, its files.
- The result of running the gate on the page's host (see above).
- The full problem list of a gate (at most 10 are shown; the rest are counted).
- Port, requirements, extra code and environment of the spec. The view is a status page, not a spec viewer.
- Publish commands. `can commands` prints them as text for the operator; the page never shows a command that publishes.
- Any control. There is no button that stages, gates, boots or publishes.

## The page

- A top-level view button "Cans" after Telegraph, hash `#cans`, an `aria-pressed` toggle like the others.
- A table with one row per can: a status mark plus the word (colour is never the only signal), name, kind, mesh and devices, weights repo with revision, card profile, bundle, sums, gate, boot. A row opens a detail row with a real button in the Can column (`aria-expanded`, `aria-controls`; Enter and Space work, and keyboard focus stays on the button after a redraw) and shows the card rows, the first problems of each piece, the recorded times and the "not checked here" Hub line.
- Playful and operator modes share the markup. Colours come from the existing tone variables, and the operator mode keeps the plain-type look of the other panes.
- The view polls `/api/cans` while it is the visible view, every 15 seconds, and says when the last answer is old.

## Tests

- `tests/test_cans_view.py`: the view module against a temporary cans folder and temporary specs.
- `tests/test_cans_view_hostile.py`: links, huge and malformed files, FIFOs, NUL in names, `..` in SHA256SUMS.
- `tests/test_can_state.py`: the writer and the `--state-dir` flags.
- `tests/test_webui_cans.py`: the `/api/cans` route (read only, no path in the answer, POST refused).
- `tests/test_webui_cans_layout.py`: the markup, script and styles.
- `tests/test_webui_cans_browser.py`: a browser test, skipped when Playwright or Chrome is missing.
