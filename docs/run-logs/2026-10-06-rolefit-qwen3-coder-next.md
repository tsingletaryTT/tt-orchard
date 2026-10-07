# 2026-10-06: role-fit test of Qwen3-Coder-Next

`python3 -m orchard.rolefit` (orchard/rolefit.py) replays recorded agent turns against a candidate server.
Server: `raahemnabeel/qwen3-coder-next-blackhole`, the `p300` profile (one board, 2 dies, bfp4 routed
experts), started with `tt-model serve --profile p300 --device-id 2,3` on a board leased through gozer.
Cold boot to ready: 473 s (compile caches were warm). The recorded turns come from the five earlier runs'
logs under `/mnt/bonus/models/orchard-runs` (35 log files). Results are in two JSON files next to this one.

## First run: failed the format bar, for one reason

`2026-10-06-rolefit-qwen3-coder-next-p300-first-no-read-file.json`

- Canary ("What is 7 times 6?"): passed, answer "42".
- Format: 46 of 50 turns well formed (0.92, bar 0.95). All 4 misses were `unknown_tool`: the model called
  `read_file` (for example `{"path": "stages/2/evidence/hw-test-output.txt"}`), a tool the loop did not have.
  The 27B used `shell` with `cat` in the same four turns.
- Recorded failure turns: 7 turns that failed in live runs (cut off at max_tokens, or empty with no tool call),
  each replayed 5 times at temperature 0.7. None of the 35 replays repeated a failure.
- Speed (one request at a time): decode 41.6 tok/s at an 8K prompt and 40.2 at 32K; prefill about 3,440 and
  3,700 tok/s.

## The change: a read-only `read_file` tool

Qwen3-Coder models are trained with a `read_file` tool, so the habit will recur. The loop now offers one
(`orchard/agent.py`, `Tools.read_file`). It reads a text file inside the run directory, tries the run
directory and then the stage directory (models write both forms), resolves links before checking the path
(a link out of the run directory is refused), does not expand `~`, cuts long files in the middle, and writes
nothing. It is stricter than `cat` in `shell`, which can read anything the user can. The agent's rules text
names it. Tests: `tests/test_read_file_tool.py` (21), each guard seen red under a mutation.

## Second run: passed

`2026-10-06-rolefit-qwen3-coder-next-p300.json`

- Canary: passed.
- Format: 50 of 50 turns well formed.
- Recorded failure turns: 7 replayed, none repeated a failure (35 replays).
- Speed: decode 41.8 tok/s at 8K and 38.4 at 32K; prefill about 3,440 and 3,700 tok/s.

## What this does and does not show

- It shows that the model produces well-formed tool calls for 50 real prompts from the 27B's runs, and that it
  does not repeat the empty or cut-off replies that stopped the 27B. It does not show that the model's
  commands are correct: a replay has no ground truth for the reply's content. That is measured only by a run.
- The 27B's failure turns came from a reasoning model decoding greedily. Coder-Next has no reasoning mode in
  this package (`reasoning_parser: null`), so a loop of reasoning cannot occur in the same way. A repeated
  tool call or a command loop still can, and the watchdog's detectors are the guard.
- One sample each at temperature 0 (format) and 0.7 (failures); one prompt per size for speed.
- The package was built from a dirty, unpushed tt-metal tree and runs the `p300` profile's bfp4 experts.
- Not tested: the `p300x2` profile (four chips), concurrent requests, a run of hours.

## Decision

`config/tiers.toml` and `config/bringup.toml` on this machine now name Coder-Next on one board (port 8001,
`--coder-chips 2`). The previous 27B configs are kept as `config/local-27b.*.toml`. The first Clef run uses
Coder-Next. If it stalls in ways the 27B did not, the 27B configs go back.
