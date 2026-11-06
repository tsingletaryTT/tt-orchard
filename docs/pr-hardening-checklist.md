# PR hardening checklist

Copilot's review of a pull request found the same kinds of defect again and again. This page lists them, from 190 inline comments on
42 pull requests (PRs 282 to 348). Each item gives the rule and the test that shows the rule holds. Run through the list before you
request a review, and put the answer for each item in the pull request body (the template asks for it). An item that does not apply
gets "n/a" and one sentence on why.

Two items rest on existing tooling: [safe-reading.md](safe-reading.md) for reading and writing files that something else controls, and
`tests/hostile_inputs.py` (the registry `READERS`) for the hostile-input corpus.

## 1. Wrong type or shape in input

Seen in PRs 282, 292, 304, 331, 342, 343: `{"tt": null}` where a dict was expected, `[check] expect = 1`, `"dtype": []`,
`assumptions: [{}]`, `"chips": 2`, a NUL byte in a path, a `PermissionError` that escaped a handler. Each one raised an exception
through a gate, a page or a supervisor stage.

- Check the type of every field you take from JSON, TOML or an evidence record before you use it. Use `safeio.finite_number`,
  `whole_number` and `bounded_str`, and pass `expect=` for nested values too.
- Wrap `os.path.realpath`, `open`, `stat` and similar calls for `ValueError` and `OSError` (including `PermissionError`).
- One bad item becomes one refused row or one named reason. It does not stop the page, the gate or the run.
- Test: parametrize each field with `None`, a list, a dict, a number, `""`, a string with a NUL, and a very large number.

## 2. Files that something else controls

Seen in PRs 331, 342, 345, 347: a FIFO that blocks a read or the write that follows a refused read, a symlinked helper that escapes a
hash, an oversized file read without a bound, a file that grows while it is hashed.

- Read through `safeio` and write with `safeio.atomic_write`. Never open a path for writing after a reader refused it.
- Charge a size budget before the read, so a file that fails or changes still uses its share.
- Walk directories with `pinned_root` and `scan_beneath`. Include symlinks in any manifest or hash.
- Test: register the reader in `tests/hostile_inputs.py` and run FIFO, link to `/dev/zero`, oversize, link out of the directory, and
  a file that grows during the read.

### The symlink and special-file recipe

Copilot raised the same five mistakes on several pull requests: `O_NOFOLLOW` that protects only the last part of a path, a path that is
pinned and then read by name, a FIFO that passes a size check, a lock or temporary file opened with `"w"` by path, and a Hugging Face
cache link that is refused when it should be accepted or measured when it leaves the cache. Before you request a review of code that opens a path or walks a tree:

- Register the reader in `tests/hostile_fs.py` (`PATH_READERS`, with a `PathReader` of one line) and run
  `python3 -m pytest tests/test_hostile_fs_readers.py`. Each reader runs against: a plain file, a link as the last part, a link
  higher up, a root that is a link, a root swapped for a link after it was pinned, a FIFO, a directory, a sparse file of 1 GiB, a
  deep tree, long names, a tree with links, a lock file that is a link, and a Hugging Face cache link inside and outside the cache.
- Open every part of a path with `O_NOFOLLOW` (`safeio.open_beneath`). `O_NOFOLLOW` on the last part alone is not enough.
- After a root is pinned, read and write through the descriptor. A path built from the root's name can reach something else by then.
- Check `S_ISREG` on the descriptor you read, open with `O_NONBLOCK`, and cap the bytes. A size check by name does not stop a FIFO.
- Open a lock or temporary file with `os.open(..., O_CREAT | O_NOFOLLOW)`, never `open(path, "w")`. The ratchet counts both.
- A Hugging Face cache link is accepted when it resolves inside the pinned cache root and refused otherwise. Measure the target's
  size from the descriptor. The size of the link itself is the length of its target name.
- A reader that fails a case today gets an `xfail(strict=True)` entry in `KNOWN_DEFECTS` that names a public issue.

## 3. Check, then use

Seen in PRs 294, 303, 304, 342: the digest taken after the boot, the spec fingerprint taken after the boot, the cache state taken
after the server loaded, an old log line that satisfies a new run, a cached blob reused on length alone.

- Capture the identity of an input (digest, spec, config) before the operation. Bind the result to that identity. Do not read it
  again afterwards.
- Give each run its own output file. Compare the content of a file. A length match is not enough.
- Test: change the input while a fake long operation runs, then assert the record names the original input and the viewer shows the
  record as stale.

## 4. Evidence that an agent can write

Seen in PRs 294, 313, 331, 341: a marker rewritten after the stage, a record deleted so a check skips, a link into `evidence/`.

- Trust an evidence file only through a supervisor ledger entry that holds its path and hash. A missing file fails the check. It does
  not skip it.
- Test: rewrite the file, delete it, and replace it with a link after the stage ends. Each must be refused.

## 5. A verdict stronger than its evidence

Seen in PRs 294, 296, 310, 343, 346: "Dealt with: yes" with no record behind it, a final-code result taken from an earlier session, a
rate gate that read the wrong field, a card line that gives an inference as a measurement.

- Every pass, label and number names what it rests on. A missing input gives "not checked" or a refusal.
- An inference carries its basis in the record (for example "judged by file age").
- A card or run log that states a performance number answers the review questions in `docs/can-cards.md` ("Review questions for a
  performance claim", from `common/tt-perf-claim-review` in tenstorrent/skills): warm or cold, how many runs, the host load, device
  time or wall time, the same harness on both sides, absolute figures, and a fixed precision policy.
- Test: remove the input and assert the output is "not checked". The output must not be a pass.

## 6. The check looks at a different object than the action uses

Seen in PRs 284, 294, 324, 343, 346: every chip in the box against the single board the stage gets, the old lease's chips against the new
lease, the original model directory against the selected variant, the source bundle against the copy that was booted, one
dilation value of a multi-dimensional convolution.

- Write down the object the check inspects and the object the action uses. Test that they are the same in at least two setups where
  they could differ (a second board, a lab machine, a chosen variant).

## 7. A parser or matcher that is too broad or too narrow

Seen in PRs 284, 294, 320, 343, 346: `aten.convolution` that also covers 2D and transposed forms, `"ple" in parts` that catches a
projection weight, only assistant role headers, only 40-character hex pins, CRLF in a hashed template, Markdown fences and
escaped pipes.

- For each classifier, list cases from the neighbouring categories that must not match, and unusual forms that must match (CRLF,
  `:-default` shell forms, tilde fences, indentation).
- Test: a table of positive and negative cases, taken from the real specification where one exists.

## 8. Failure paths that leave state behind

Seen in PRs 296, 305, 345, 346, 347: a trace that is half re-captured, a queue of jobs left waiting after a device fault, a lease held
because a write blocked, a worker that outlives its timeout, a lease record written after the check that can block.

- After each acquire (a lease, a trace, a process, a queue entry), ask what holds if the next step raises or the process is killed.
  Use `try`/`finally`, a process group, and record before you act.
- Test: raise at each step after the acquire and assert the state is recoverable.

## 9. Numbers and sizes

Seen in PRs 286, 291, 296, 343: a block size below the minimum that silently runs two blocks, one more decode step than the context
allows, `n=0` returning one token, a tensor shape of `10**20` that hangs a loop, a conversion that counts one buffer when two exist.

- Bound sizes before arithmetic (`safeio.MAX_NUMEL`). Reject counts of zero or below. Count each buffer at its larger size.
- Test: 0, -1, 1, the largest allowed value, and `10**20`.

## 10. Secrets, paths and host names

Seen in PR 342 and in the redaction findings: a repository label that was redacted while its link held the token.

- Redact before any output, link, log line or card. Build a link only from a value that matches a plain pattern in full.
- Test: an `hf_` token, a home path and a host name in every field that reaches output.

## 11. Tests that do not test the claim

Seen in PRs 320, 336, 342: an environment gate that checks that `transformers` imports but not that the model class exists, a
fixture that is too small to show a property of scale, timing that decides a pass.

- A skip names the exact feature it needs. A guard has a mutation that turns a test red. Use real shapes when the property is about
  size. Do not let a clock decide a result.

## 12. Housekeeping Copilot asks for each time

- A change fragment (`python3 scripts/version.py new ...`) and a dated entry at the end of `docs/project-log.md`.
- Every new command-line flag in the README (`tests/test_readme.py` checks this).
- Docs that state what the code does now. If a sentence names a number or a behaviour, check it against the code.
- `python3 scripts/version.py check` before the push.

## The ratchet

`tests/test_hardening_ratchet.py` counts raw file access in `orchard/` (`.read_text(`, `.read_bytes(`, `.write_text(`,
`.write_bytes(` and `open(`) per file and fails when a count goes up. A line that is right to keep (a read of a file this program
wrote itself, with a reason) carries a `# hardening: ok <reason>` comment and is not counted. When you remove raw access, lower the
baseline with `python3 scripts/hardening_baseline.py --update` in the same change. `--update` refuses to write when any count rose. When a
merge from main brought in raw access from another change, run it with `--accept-growth` and say why in the project log. A marker
goes on the first or last line of the call it marks.

A second count in the same script (`tests/hardening_hazards_baseline.json`) holds link hazards: an `os.open` whose flags never name
`O_NOFOLLOW` (a `dir_fd` anchors the lookup but a link in the last part is still followed, so it does not exempt the call), and the builtin `open` with a literal write mode (`"w"`, `"a"`, `"x"` or `"+"`). It works like the first count, uses
the same marker, and `--update` and `--accept-growth` write both files.
