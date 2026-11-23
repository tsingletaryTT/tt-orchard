# OpenCode v2 against orchard's agent loop, on the tt-lang and tenstorrent skills (2026-11-21)

Branch `opencode-probe-eval`. Tools: `scripts/opencode_probe.py`, `scripts/landlock_exec.py`, `scripts/oc_request_proxy.py`. No file under
`orchard/` changed. Pre-registration first; results follow in this file.

## Summary

**Question.** Is OpenCode v2 (2.3.2) a better coding harness than orchard's own agent loop for tt-lang kernel work and for tenstorrent/skills? Where OpenCode
runs slower, are its results better?

**Answer: not shown, in either direction.** Same model (Qwen3-Coder-Next on four chips, one server), same task text, skill text, attempt directory and unedited
simulator check; the agent loop is the only change. 28 attempts, 14 per harness: `add` 2, `center` 6, `row_sum` 6.

- **Simulator passes:** OpenCode 5 of 14, orchard 4 of 14. By op: `add` 2 of 2 each, `center` 0 of 6 against 1 of 6, `row_sum` 3 of 6 against 1 of 6. No comparison
  gives a Fisher exact p below 0.48.
- **Passes a chip compiler accepts** (compile step only, on the CPU, no device): the pair's compiler (tt-lang 1.1.10.dev) accepts 4 of OpenCode's 5 passes and 2 of
  orchard's 4; tt-lang 1.1.6 accepts 2 and 2. A simulator pass is not a chip pass, and no kernel was built with tt-metal or run on a chip.
- **Quality of the passing kernels:** all nine are at PCC 0.9999999 or better with relative L2 of 1.7e-07 or less, and all nine pass seeds the agent never saw. The
  failures are structural (a simulator error or a shape error) or far from the bar (PCC 0.18 at best); none is a near miss.
- **Time and cost:** OpenCode sends 1.8 to 6.4 times the prompt tokens (its context grows to about 100,000 tokens) and decodes at 28 to 36 tokens per second against
  33 to 45 for orchard. Decode is 78% to 97% of wall time; prefill is 3% to 17%. Among attempts with no load flag, OpenCode was slower only on `add` (median 100.5 s
  against 79 s); on `center` and `row_sum` its clean medians were lower (298 s against 542.5 s; 532 s against 624.5 s). Where OpenCode was slower it did not buy a
  better kernel: the two `add` kernels of each harness are equal on every measure, and its slowest attempt (`center` 3, 1184 s) failed where orchard's pair partner passed.
- **Skills:** OpenCode finds `SKILL.md` skills by itself in four layouts. It loaded the skill in 9 of 12 fact questions and every one of those answers held the facts; in 3 it
  answered from the skill's one-line description and missed facts. No bring-up stage was run through OpenCode, so tenstorrent/skills performance in a stage is not measured.

**What bounds all of it.** (1) Every attempt got the old plain cheat sheet, whose chip claims the chip compilers refuse (issue 801, fix in PR 803), so the pass
counts belong to that sheet. (2) Simulator passes are not chip passes. (3) 10 of the 28 attempts ran inside load windows, and 3 of those also shared the server with another client.
(4) Two to six attempts per cell. (5) The recipes A/B of 2026-11-21 was retracted (1 of 7 against 0 of 8 on chip-acceptable passes); nothing here uses recipes. Details below.

## Pre-registration (written 2026-11-21 before any measured attempt)

### Questions

1. **tt-lang skill.** The kernel probe gives a coder the tt-lang skill text and asks for a kernel that the unedited simulator check passes.
   With the same model, server, task, skill text and check, does OpenCode v2 (2.3.2) pass more or fewer attempts than orchard's own agent
   loop? Orchard's loop is the control.
2. **tenstorrent/skills.** Does OpenCode v2 find `SKILL.md` skills by itself, and does the model use them? This is a capability test with
   counts and short transcripts. It measures no pass rate. Running a bring-up stage through OpenCode is not tested here: a stage needs
   the supervisor's prompt, files and gates, and the supervisor is not changed by this work.

### What is fixed, for both harnesses

- Model and server: `Qwen/Qwen3-Coder-Next`, the 4-chip `p300x2` profile on `http://127.0.0.1:8011/v1`, already running under another lease. It is
  not stopped, restarted or reset, and no attempt takes a lease or opens a device. The 2026-10-28 ladder used the 2-chip `p300` profile, so
  its numbers are context only; the control is run now.
- Command base: `kernel-probe --op OP --attempts 1 --temperature 0.7 --phase sim --library none --no-agent-hw --endpoint http://127.0.0.1:8011/v1
  --model Qwen/Qwen3-Coder-Next --allow-stale-skills`, one invocation per attempt so the harnesses can alternate.
- `--allow-stale-skills`: the probe's skills check fails here for two reasons outside the harness. The tt-lang clone is at the pinned commit
  and not at upstream `main`, and the tt-lang skill cannot be compared with upstream because the GitHub token has no access to that
  organization (SAML). Both harnesses use the flag. The check also reports that the installed tt-lang skill documents specification 0.12 while
  the checkout and simulator are at 0.20, and the probe builds its reference files from the checkout; so "the tt-lang skill" in the measured
  attempts is the probe's reference set and `kernel-probe.md`, and the installed skill is what the skill-discovery test uses.
- Sampling: temperature 0.7, no top_p; each pair of attempts (one per harness) uses the same seed base. The server's counter of decode
  steps "forced by seeded rows" rose during an OpenCode request that carried a seed, so a seed reaches the server. Whether the server makes a
  seeded draw repeat is not checked (the 2026-10-28 ladder says the same).
- Limits: 40 turns (orchard) and `steps: 40` (OpenCode); 4096 tokens per reply (orchard `max_tokens`; OpenCode `limit.output`, sent as
  `max_completion_tokens`); 1800 s of wall time per attempt; the probe's unedited authoritative check decides every pass (`passed_sim`).
- One client at a time against the server. Harnesses alternate in ABBA order.
- Skill text: `orchard/skills/kernel-probe.md`, byte for byte. Orchard sends it as the system prompt. OpenCode gets it as an ambient
  instruction file with a fixed mapping paragraph in front (below).
- Task text: `attempt_user_message(op, adir)`. Attempt directory: `prepare_attempt`. Both are the probe's own code in both harnesses.

### Differences that the settings cannot remove (reported with the results)

- OpenCode adds its own system prompt (about 10 KB in the first request), ten tools (edit, glob, grep, question, read, shell, skill,
  subagent, write, execute) and context compaction. Orchard's agent has three tools (shell, read_file, write_file) and a retry and nudge layer
  (watchdog nudges, retry guard).
- Switched off for OpenCode so an attempt is one agent with file and shell tools: `question`, `subagent`, `execute`, `skill`, `webfetch`,
  `websearch`, the Claude Code configuration loader, model-list fetching, the file watcher, updates and sharing. Left on: `read`, `write`,
  `edit`, `shell`, `grep`, `glob`.
- Orchard runs the shipped check after every write of `kernel.py` and appends the output to the write's result. OpenCode does not, so the
  agent has to run `python3 kernel_check.py` itself. This favours orchard and is part of what the comparison measures.
- The mapping paragraph put in front of the skill for OpenCode, in full: "Tool names in this environment differ from the names in the
  instructions below: `read_file` is the `read` tool (offset and limit count lines); `grep` and `glob` are available too. `write_file` is the
  `write` tool; `edit` changes part of a file. `shell` is the `shell` tool. Writing kernel.py does not run the check for you here. After each
  write, run `python3 kernel_check.py` yourself and read its output. There is no `hardware_check` tool in this environment. The instructions
  below are otherwise the same."
- Shell environment: orchard's agent inherits the operator's `PATH`; the OpenCode sandbox has the simulator venv first, then `/usr/local/bin`,
  `/usr/bin`, `/bin`. Both get `kernelprobe.child_env` (private `HOME`, no token variables, no device).

### Sandbox for OpenCode (a deviation from the plan, with the reason)

The plan named bubblewrap. `bwrap` fails here with `setting up uid map: Permission denied`: `kernel.apparmor_restrict_unprivileged_userns` is 1 and
no AppArmor profile allows bwrap. Changing that is a system security change and was not made. The sandbox is Landlock (Linux security
module, ABI 8 here, no root, no user namespace) through `scripts/landlock_exec.py`:

- Read and execute: `/usr` and the system links, a few `/etc` files, `~/.opencode`, the simulator venv, its interpreter and the tt-lang clone.
  Read and write (no execute): the attempt directory, a private `HOME` and a private `TMPDIR` under `oc-meta/`. Single device files:
  `/dev/null` (read, write), `/dev/urandom`, `/dev/random`, `/dev/zero` (read).
- TCP: connect to the model port only (for the shell commands the model runs) and bind one port, the port OpenCode's server listens on.
- Checked with `open()` and `listdir()` from inside the sandbox before any attempt: `~/.ssh/id_ed25519`, `~/.cache/huggingface/token`,
  `~/.config/gh/hosts.yml`, `~/.docker/config.json`, `~/.config/opencode/opencode.json`, `~/code`, `~/orchard-runs`, `/tmp`, `/etc/shadow`,
  `/dev/tenstorrent` and `/dev/tenstorrent/0` and `/3`, and another process's `/proc/1/environ` all give EACCES. A write outside the attempt
  directory fails. A file written into the attempt directory cannot be executed. A connect to port 22 and a bind to a free port fail. A connect
  to 8011 works. The simulator runs. `HF_TOKEN`, `GH_TOKEN` and `SSH_AUTH_SOCK` are unset.
- `scripts/landlock_selfcheck.py` repeats these checks with real `open`, `listdir`, `connect` and `bind` calls and no model (21 of 21 probes as expected on 2026-11-22; the only environment
  name it lists is `HF_TOKEN_PATH`, orchard's own variable, which points at a file that does not exist in the private home).
- Landlock denies access and does not hide names, so `stat` on `~/.ssh` still succeeds. The plan said "not visible"; the measured fact is
  "access denied".
- OpenCode's own `run --standalone` cannot be used: its private server needs a TCP bind that the sandbox denies. The runner starts
  `opencode serve` as the first sandboxed process on one chosen port with a per-attempt random password (`OPENCODE_PASSWORD`), and runs
  `opencode run --server` as a second sandboxed process that may connect to that port only. The server listens on 127.0.0.1 for the length of
  an attempt, so another local process could send it requests; the password limits that, and the server itself is inside the same sandbox.
- Orchard's own agent shell is not sandboxed this way (it has the fail-closed command lexer and the allow-listed environment). The operator
  accepted visible credentials for it (preflight "credentials ... accepted"). So the OpenCode runs are more restricted than the control.

### Setup traps found while building the runner (before any measured attempt)

- OpenCode sent no temperature. `agent.build.temperature` and the model's `temperature: true` flag did not change the request body. A logging
  proxy (`scripts/oc_request_proxy.py`) showed it. Model `options` go into the body unchanged, so `temperature`, `top_p` and `seed` are set there; the
  proxy then showed `temperature: 0.7`, `top_p`, `seed` and `max_completion_tokens: 4096` in the request.
- With no model set, OpenCode 2 picked a hosted free model of its own (`tidepool-alpha-free`) and answered a test prompt (the prompt was
  "Reply with exactly: ready"). The runner enables only the Tenstorrent provider (`enabled_providers`) and sets the model explicitly. An attempt
  whose event stream names `tidepool` is treated as invalid.
- The v2 tool for the shell is called `shell`. An early mapping paragraph said `bash`; it was corrected before any measured attempt.

### Load conditions

A full orchard test suite (`pytest -n 6`, `nice 10`, `CPUWeight=20`, unit `suite-fixstubs`) ran on this host while the runner was built. systemd gives
its start as 19:25:35Z; the coordinator's note said about 19:34Z, which was later than the clock at the time (19:27Z), so systemd's value is used. The
suite uses fakes and does not use the model server. Every attempt records `started_utc` and `ended_utc`. Attempts that overlap the suite's window are
flagged, and wall-time differences between harnesses are not read from them. The load average was 19.3 at 19:27:47Z. Other windows are in
`~/orchard-runs/ocprobe/load-conditions.txt`.

### Plan

**Pilot** (seed base 2026112000; the pilot attempts are left out of the batch counts): `add` and `center`, one attempt per harness, order: OpenCode `add`, orchard `add`,
orchard `center`, OpenCode `center`. Read the OpenCode transcript (`--format json`, kept under `oc-meta/`) and fix harness traps before the
batch. Model behaviour is left as it is. Anything changed is written below.

**Batch** (seed base 2026112100 plus the pair's index; both harnesses of a pair use the same seed): `add` x2, `center` x4, `row_sum` x4 per
harness, 20 attempts, pairs in this order, harness order flipped on every pair (ABBA): add1 O R, add2 R O, center1 O R, center2 R O,
center3 O R, center4 R O, row_sum1 O R, row_sum2 R O, row_sum3 O R, row_sum4 R O (O is OpenCode, R is orchard).

**Readout**, counts only. Per op and harness: attempts, simulator passes, failure classes from the probe's classifier, turns or steps, agent wall
seconds (with the load flag), tokens where reported, attempts ended by the wall or the step limit, and for OpenCode the tool calls by tool.
Also noted: behaviour at the 4096-token reply cap, tool-call parse failures from this coder, compaction, the server process, how easy the output
is to parse. The 2026-10-28 ladder counts (2-chip coder: `row_sum` 6 of 9, `center` 1 of 6, `rmsnorm` 0 of 3; rungs 1 to 5 passed in single
pilots) are context.

**What the counts cannot show.** With 2 to 4 attempts per op and harness, a pass count of 3 against 1 is inside what chance gives. One model,
one server profile, one task family. The harnesses differ in system prompt, tools, compaction and the auto-run of the check, so a difference
in passes is a difference between the whole setups and says nothing separately about the skill text or about the harness code.

**Stop rules.** Stop and report if the server errors twice in a row, if an attempt escapes the sandbox, or after 5 hours.

### Addendum 1: operator instruction on quality (written 2026-11-21, after the pilot began and before any batch attempt)

The coordinator passed on this instruction from the operator: "continue comparing; when opencode is slower, evaluate if its results are better even
so". The pilot had two of its four attempts done when it arrived (OpenCode `add` and orchard `add`, both passed); no batch attempt had started. The
plan above stays as it is, with these additions.

**Attempts per op.** `center` and `row_sum` may go from 4 to 6 attempts per harness (pairs 5 and 6 of each, harness order `O R` then `R O`,
seed base 2026112100 plus the pair index, continuing the numbering after the 10 pairs of the plan). Rule, fixed now: after the pilot, take the
expected seconds of one attempt as the larger of 600 and the mean of the pilot's `center` attempts, for `center` and for `row_sum` (the 2026-10-28
ladder shows 650 to 1800 s on the 2-chip coder). Projected batch seconds = 2 harnesses x (2 x 150 for `add` + 6 x expected for `center` + 6 x expected
for `row_sum`). If the projection is at most 4.5 hours (the 5-hour cap less a margin for the discovery test and overruns), run 6 per op and harness;
otherwise run 4. The decision and the numbers go in the Pilot section before the batch starts.

**Quality readout** (raw numbers; derived figures are labelled as arithmetic on them), for both harnesses, per attempt and per op:

1. *Time to a pass.* Agent wall seconds of each passing attempt. Passes per hour of agent wall time over all of a harness's attempts of an op
   (passes divided by the sum of agent wall seconds of every attempt, failures included, times 3600). A same-budget view: for each op, with T the
   longest agent wall time among the other harness's passing attempts, how many of this harness's attempts passed within T (undefined when the other
   harness has no pass). Attempts that overlapped the `suite-fixstubs` window are flagged, and wall-time differences are not read from them.
2. *Margin of passing kernels,* from the unedited check's own measures (`set_measures`): `pcc_min`, relative L2, slope, max absolute error, and the
   outlier input sets separately. Worst and median over a harness's passing attempts, with the distance to each bar (PCC 0.999, relative L2 0.01,
   slope 0.98 to 1.02, max absolute error within 0.01 + 0.01 |expected|).
3. *Closeness of failures.* The same measures when the check produced them, and the failure class. A failure is a near miss when it is numeric and
   within 10 times of the bar: PCC at least 0.99, relative L2 at most 0.1, slope between 0.8 and 1.2. Deadlock, dataflow step, shape mismatch, import
   error, tt-lang error, forbidden code and no kernel are structural.
4. *Kernel quality the simulator pass does not show.* (a) A static check of each passing `kernel.py` against the cheat sheet the probe writes into the
   attempt directory (`ttlang_cheat_sheet.json`, the signatures of the package that compiles for the chip): the tt-lang calls the kernel uses, with any
   call that is not in the sheet flagged. (b) Lines of code, and whether the kernel follows the reference files' worked pattern. (c) A held-out run on
   an input set the agent never saw, only if the shipped check can take another seed or shape without being edited; if it cannot, the log says so.
5. *How the work got done.* Number of writes of `kernel.py`, whether the agent ran `kernel_check.py` itself before it finished, tool calls by tool,
   and whether it ended on its own or on a limit; for OpenCode the step count against the 40-step setting. Orchard runs the check after every write
   by itself; those automatic runs are counted apart from runs the agent asked for.
6. *The question for the verdict:* where OpenCode took longer, did it buy a better kernel (wider margin, fewer unsupported calls, passes the held-out
   input set) or only a slower pass; and the same question the other way round for orchard.

**What these numbers cannot show.** Two to six attempts per cell, one model, one server profile. The simulator is the only judge: no kernel is compiled
for the chip, because the coder holds all four chips and the task rules forbid taking a lease. A static check against the cheat sheet finds calls the
compiling package does not list; it does not prove that a kernel compiles.

**Load windows, corrected (2026-11-21 19:49Z).** The coordinator corrected the first suite's window to 19:25:37Z to 19:42:30Z, which agrees with the
journal (19:25:35Z to 19:42:31Z); the "about 19:34Z to about 19:51Z" figure given earlier was withdrawn, and the paragraph under Load conditions above
that uses 19:34Z is superseded by this one. A second suite (`suite-main-final`, nice 10, CPUWeight 20) started about 19:49Z and should end near 20:06Z; its
exact start and end are taken from the journal when it finishes and stored with the results (`suite-window.json`). Rule unchanged: an attempt inside a
window is flagged and is left out of every wall-time comparison between harnesses.

**Runner edit during the pilot.** After the pilot began, optional parameters (`skill_tool`, `skill_paths`) were added to `run_opencode` and
`write_config` for the skill-discovery test. Their defaults reproduce the earlier behaviour, so the pilot's attempts are unaffected.

## Pilot (2026-11-21, 19:34Z to 19:47Z; not counted in the batch)

Seed base 2026112000, one attempt per harness per op, in the registered order. Three attempts overlap the window of the first suite (`suite-fixstubs`, journal 19:25:35Z to 19:42:31Z, the
coordinator's corrected figure 19:25:37Z to 19:42:30Z): OpenCode `add`, orchard `add` and orchard `center`. The OpenCode `center` attempt (19:43:09Z to
19:47:17Z) began after that window ended and ended before the second suite started (about 19:49Z), so it overlaps neither. The load average at its start was not
recorded (the driver took that reading from the batch on); the host's 1-minute load average at 19:47:41Z was 10.9 with no suite active. Wall times from the
pilot are not compared.

| order | harness | op | simulator result | class | turns or steps | agent wall s | tool calls | ended on |
|---|---|---|---|---|---|---|---|---|
| 1 | OpenCode | add | pass | - | 11 | 103.8 | read 9, glob 3, write 1, shell 2 | its own final answer |
| 2 | orchard | add | pass | - | 9 | 71.9 | read_file 6, write_file 2, shell 1 | its own final answer |
| 3 | orchard | center | fail | tt_lang_error | 40 | 307.1 | read_file 15, shell 23, write_file 2 | the 40-turn limit |
| 4 | OpenCode | center | fail | tt_lang_error | 40 | 238.8 | read 13, glob 4, grep 2, write 1, shell 22 | the 40-step limit (a final text was forced at step 40) |

Prompt and completion tokens counted by the harnesses: `add`: OpenCode 539,514 and 2,112, orchard 79,402 and 2,110. `center`: OpenCode 2,225,212 and
5,008, orchard 1,002,839 and 5,546. The completion token counts are close; the prompt token counts differ by a factor of 6.8 on `add` and 2.2 on `center`
(each step resends the context; OpenCode's system prompt and ten tool schemas make its first request much larger, and its file reads come back whole). The
server's own counters (prefix-cache share, prefill and decode seconds) start with the batch.

What the transcripts show (read for harness traps; model behaviour is left alone):

- OpenCode `add`: 11 steps, 15 tool calls, all completed, no error event, no model named other than the Tenstorrent one, nothing on stderr. The agent
  read `kernel_check.py`, `kernel_harness.py` and `references.py`, which the probe's skill never names. The largest input of any step was 66,341 tokens.
- OpenCode `center`: 40 steps, 42 tool calls, all completed, no error event; it ran `kernel_check.py` 22 times itself and read the cheat sheet and the error
  more than once. Its last reasoning names the blocker: the simulator's `reduce_sum` requires a `shape` argument (the simulator error is
  `reduce_sum() missing 1 required positional argument: 'shape'`), while the cheat sheet says the chip's package (1.1.6) has no `shape` argument. The
  probe's `kernel-probe.md` tells the coder to pass `shape`, the unedited check is the simulator, and the cheat sheet says to leave it out. Both harnesses'
  `center` pilot attempts ended on the same simulator error. The conflict belongs to the task and the skill texts and not to either harness. It has to be
  kept in mind when reading `center` and `row_sum` pass counts as a measure of a harness.
- The orchard `center` attempt (the control) is the same failure. The orchard `add` attempt wrote `kernel.py` twice (the second write after the
  automatic check output) and ran the check once itself.
- The two transcripts showed no trap. One was found afterwards, from the tool list in the request (next section). The OpenCode step limit forces a final
  text at step 40, so a 40-step run always ends with `stop`; the runner labels it `turns` (the count is compared with the limit).

Changes after the pilot and before the batch, none touching what an attempt does: the driver takes the model server's counters (`/metrics`: prompt tokens
from cache and computed, generated tokens, request prefill, decode and queue seconds, replies cut at the 4096 cap) and `/proc/loadavg` before and after every
attempt, and stops starting attempts 5 hours after its start. `scripts/oc_quality.py` was added (it reads results and runs nothing against the server).

**Decision under addendum 1.** Expected seconds of one `center` or `row_sum` attempt = max(600, mean of the two pilot `center` attempts, 273) = 600.
Projection = 2 x (2 x 150 + 6 x 600 + 6 x 600) = 15,000 s = 4.17 h, which is at most 4.5 h. So `center` and `row_sum` run 6 attempts per harness: 28
attempts (14 pairs), the 10 pairs of the plan plus `center` 5 and 6 and `row_sum` 5 and 6, harness order flipping on every pair as before. The pilot
attempts took 4 to 5 minutes each, so the projection is more than the likely time. Start: a second suite (`suite-main-final`, started about 19:49Z, expected
to end near 20:06Z) is running, so the batch waits for its end and no batch attempt overlaps it. The wait is used for the skill-discovery test, which
compares no wall times. If another suite starts during the batch, its window is flagged on the attempts it overlaps.

## Before the batch: the tool list OpenCode offered (found 2026-11-21 19:58Z, after the pilot, before any batch attempt)

The plan says OpenCode's extra tools are switched off. The skill-discovery runs showed calls to tools I had not expected, so the request's tool list was
read again through the logging proxy under the pilot's configuration. It held **15 tools**: `edit`, `glob`, `grep`, `read`, `shell`, `write`, `skill`
and eight `opencode_*` tools (`opencode_list_mcp_resources`, `opencode_models`, `opencode_read_mcp_resource`, `opencode_session_move`,
`opencode_session_rename`, `opencode_worktree_create`, `opencode_worktree_list`, `opencode_worktree_remove`). The earlier check of the tool list (10 tools:
the six plus `question`, `subagent`, `skill`, `execute`) had been made before the switches were added, and the list was not read again after that. The two OpenCode
pilot attempts ran with the 15; they used `read`, `glob`, `grep`, `write` and `shell` only.

What each setting does to the list (one request each, proxy log, 2026-11-21 19:57Z to 20:00Z):

| setting | tools sent |
|---|---|
| no `tools` key | the six, `skill`, `question`, `subagent`, `execute`, `webfetch`, `websearch` (12) |
| `webfetch`, `websearch` off | the six, `skill`, `question`, `subagent`, `execute` (10) |
| plus `skill` off | unchanged: `tools.skill: false` does nothing |
| plus `question` off | `question` removed |
| plus `question`, `subagent`, `execute` off | those removed, and the eight `opencode_*` tools appear (15 with `skill`) |
| the line above plus a `permission` deny of `opencode_*`, `skill` and the two worktree tools | seven `opencode_*` tools removed; `skill` and `opencode_worktree_remove` remain (8) |
| agent-level `tools` map or agent-level `permission`, or naming each tool in `permission` | no further change |

With `execute` on, the `opencode_*` tools seem to be reached through it; with it off they are listed one by one. The batch setting is the last-but-one row
(`webfetch`, `websearch`, `question`, `subagent`, `execute` off; `permission` deny of `opencode_*`, `skill`, `opencode_worktree_remove`,
`opencode_worktree_create`): **8 tools offered**, `edit`, `glob`, `grep`, `read`, `shell`, `write`, `skill` and `opencode_worktree_remove`. Called on purpose, `skill`
returns "Unable to load skill default" (the project has no skills) and `opencode_worktree_remove` returns "not a worktree of this project"; the permission deny
does not stop the call from running, and neither tool can do anything in an attempt directory. `write` and `shell` work under this setting. Orchard's agent has three
tools; OpenCode's has six working ones plus those two that only return errors. This replaces the "ten tools, four switched off" statement under Differences. The
discovery test used its own setting (`skill` tool on); its tool counts include a few calls to tools that are not offered, which return errors.

## Skill discovery (capability test, 2026-11-21 19:50Z to 19:57Z)

Question 2. `scripts/oc_skill_discovery.py`: copies (not links) of three skills in each layout, OpenCode in the sandbox with the `skill` tool on, four
questions, one fresh run per cell (20 runs, 425 s of wall time in all; no wall time is compared). The skills: the installed tt-lang skill (`ttm-tt-lang`) and
two tenstorrent/skills (`tt-device-usage`, `functional-decoder`). The answer check is a substring match on the facts listed in the script. Layouts: `.claude/skills`,
`.opencode/skills`, `.opencode/skill`, a folder named in the config's `skills` list, and none (the control).

| layout | list: names found | tt-lang fact | tt-device-usage facts | functional-decoder path | `skill` tool calls over the 3 fact questions |
|---|---|---|---|---|---|
| `.claude/skills` | 3 of 3 | found, 1 skill call | 2 of 2, 1 skill call | 2 of 2, 1 skill call | 3 |
| `.opencode/skills` | 3 of 3 | found, 1 skill call | 2 of 2, 1 skill call | 1 of 2, no skill call, 11 steps | 2 |
| `.opencode/skill` | 3 of 3 | found, 1 skill call | 2 of 2, 1 skill call | 2 of 2, 1 skill call | 3 |
| config `skills` folder | 3 of 3 | found, 1 skill call | 0 of 2, no skill call, 1 step | 1 of 2, no skill call, 1 step | 1 |
| none (control) | 0 of 3 | not found (1 skill call, nothing to load, 6 steps) | 0 of 2 (1 skill call) | 0 of 2 (1 skill call, 12 steps) | 3 (all returned an error) |

Counts. OpenCode v2 found the skills in all four layouts without being told where they are (the list answer named all three each time; the control named none).
Of the 12 fact questions in the four layouts with skills, the model called the `skill` tool in 9, and all 9 answers held every expected fact. The other 3 had no
skill call, and each of them missed at least one fact. For the tt-lang skill the fact (`buffer_factor`, the installed skill's older keyword) was found in 4 of 4
layouts and in 0 of 1 in the control, so the answer came from the skill. For the two tenstorrent/skills, the facts were found in 5 of 8 runs (control: 0 of 4). In the config-folder
layout the model answered the two tenstorrent/skills questions in one step with no skill call. The `device` answer opened "According to the tt-device-usage skill ..." and
gave general statements with none of the skill's own facts, which is a wrong attribution a reader would trust. The `decoder` answer said it was working from the
skill's description and guessed a path pattern.

What this does not show: one run per cell at temperature 0.7, so a 9 of 12 against 3 of 12 split between "called the skill" and "did not" says only that the choice
varies. The questions look up one fact each and none needs a skill's procedure. No bring-up stage was run through OpenCode.

How orchard hands skills over today (read from `orchard/stages.py` and `orchard/supervisor.py`): a stage names its skill and its reference skills; the supervisor
finds `<dir>/<name>.md` or `<dir>/<name>/SKILL.md` in the configured `skills_dirs`, reads the files, and builds the stage's messages from them
(`build_messages(spec, phase, ..., skill_path, refs)`). The model gets the text and makes no choice. OpenCode puts each skill's name and description in the `skill`
tool's description and leaves loading to the model; the runs above show it loads the skill most of the time and sometimes answers from the description alone.

## Batch (2026-11-21 20:05:45Z to 23:53:58Z)

28 attempts, 14 pairs, 3 h 48 min of clock time and 3.73 h of summed agent wall time (OpenCode 7,475 s, orchard 5,946 s). Command and settings as registered, with
`center` and `row_sum` at 6 attempts per harness (addendum 1), the tool list of the section "Before the batch", seeds 2026112101 and up (the CLI sends the base plus 1), one
attempt at a time, harness order flipped on every pair. Every row of the compile probe says `recipes: off`: no attempt used the recipes switch.

### Load and the other client on the server

184 load intervals were recorded (`load-windows.json`): the first suite (`suite-fixstubs`, 19:25:35Z to 19:42:31Z by the journal), the second (`suite-main-final`,
19:49:06Z to 20:05:45Z), the third (`suite-integ2`, 22:06:53Z to about 22:25:33Z), the coordinator's own simulator check (21:26:30Z to 21:27:20Z), and niced pytest runs and
simulator verifications of three forks (sample-hook 21:29:39Z to 21:46:25Z in 38 intervals, audit-hardening 21:28:41Z to 22:05:52Z in 30 intervals, recipes 22:38:21Z to
23:00:10Z in 112 intervals). The host's 1-minute load average at attempt starts ran between 4.0 and 35.5 (median about 9); other work shares the box, so no attempt ran on an idle host.
After the batch I ran two light CPU jobs at nice 10: the compile probe (2026-11-22 22:49:57Z to 22:50:48Z) and the quality readout (22:50:58Z to 22:51:18Z); no attempt was running.

**Another client used the coder** during three attempts. The server's own counters show more requests than the harness made (beyond the probe's one speed check per
attempt): OpenCode `row_sum` 3 (22:21:18Z to 22:43:59Z) +15, orchard `row_sum` 3 (22:43:59Z to 22:52:22Z) +31, orchard `row_sum` 4 (22:52:22Z to 23:08:56Z) +13; in each the server's
summed decode seconds exceed the attempt's wall time, so requests overlapped. The coordinator's forks did not call the coder. The likely source is the operator's own
OpenCode session (outside this sandbox): its database has message rows created 22:30Z to 23:10Z (64 rows; 14 in the 22:30 ten-minute bucket, 28 at 22:40, 18 at 22:50, 4 at 23:00),
none after 23:10Z, and a busy period 22:53:14Z to 23:00:55Z. The rows are message events and not request counts, so the match is by time and not by number. It is
not proven from the server's counters (the server log shows only the docker bridge address). OpenCode `row_sum` 4 (23:08:56Z to 23:20:01Z) overlaps none of that activity and
shows +0. Pass or fail of the three attempts stands; their wall time, decode speed and cache counters are not comparable and are left out of the server columns below.

Attempts with no load window and no other client: **18 of 28**. A `*` marks a load window and a `+` another client in the tables; flagged wall times are listed but not compared.

### Results

#### Results by pair (O is OpenCode, R is orchard; `*` load window, `+` another client on the server)

| pair | O | R |
|---|---|---|
| add1 | PASS, 12 steps, 75 s, ended done | PASS, 9 turns, 51 s, ended done |
| add2 | PASS, 18 steps, 126 s, ended done | PASS, 13 turns, 107 s, ended done |
| center1 | fail tt_lang_error, 40 steps, 176 s, ended turns | fail tt_lang_error, 37 turns, 198 s, ended escalate |
| center2 | fail shape_mismatch, 40 steps, 298 s, ended turns | fail tt_lang_error, 39 turns, 524 s, ended escalate |
| center3 | fail tt_lang_error, 42 steps, 1184 s, ended turns | PASS, 35 turns, 561 s, ended done |
| center4 | fail tt_lang_error, 40 steps, 324 s, ended turns | fail tt_lang_error, 40 turns, 720 s, ended turns |
| center5 | fail tt_lang_error, 40 steps, 184 s, ended turns | fail tt_lang_error*, 40 turns, 372 s, ended turns |
| center6 | fail shape_mismatch*, 40 steps, 510 s, ended turns | fail tt_lang_error*, 11 turns, 143 s, ended escalate |
| row_sum1 | fail numeric_mismatch*, 44 steps, 1690 s, ended error | fail shape_mismatch*, 13 turns, 127 s, ended escalate |
| row_sum2 | fail no_kernel*, 40 steps, 145 s, ended turns | fail numeric_mismatch*, 38 turns, 416 s, ended pause |
| row_sum3 | fail numeric_mismatch*+, 43 steps, 1351 s, ended turns | PASS*+, 40 turns, 492 s, ended turns |
| row_sum4 | PASS, 40 steps, 657 s, ended turns | fail tt_lang_error*+, 34 turns, 986 s, ended error |
| row_sum5 | PASS, 28 steps, 223 s, ended done | fail numeric_mismatch, 40 turns, 853 s, ended turns |
| row_sum6 | PASS, 41 steps, 532 s, ended turns | fail tt_lang_error, 40 turns, 396 s, ended turns |

#### Counts per op and harness

| op | harness | attempts | simulator passes | passes the pair's compiler accepts (1.1.10.dev20261026+light) | passes tt-lang 1.1.6 accepts | failure classes | ended on |
|---|---|---|---|---|---|---|---|
| add | opencode | 2 | 2 | 2 | 2 | - | done 2 |
| add | orchard | 2 | 2 | 2 | 2 | - | done 2 |
| center | opencode | 6 | 0 | 0 | 0 | shape_mismatch 2, tt_lang_error 4 | turns 6 |
| center | orchard | 6 | 1 | 0 | 0 | tt_lang_error 5 | done 1, escalate 3, turns 2 |
| row_sum | opencode | 6 | 3 | 2 | 0 | no_kernel 1, numeric_mismatch 2 | done 1, error 1, turns 4 |
| row_sum | orchard | 6 | 1 | 0 | 0 | numeric_mismatch 2, shape_mismatch 1, tt_lang_error 2 | error 1, escalate 1, pause 1, turns 3 |
| all | opencode | 14 | 5 | 4 | 2 | | |
| all | orchard | 14 | 4 | 2 | 2 | | |

Fisher exact two-sided p for the pass counts (arithmetic on the counts above): all simulator passes 5 of 14 against 4 of 14, p = 1.00; passes the pair's compiler accepts 4 of 14 against 2 of 14, p = 0.65; `row_sum` simulator passes 3 of 6 against 1 of 6, p = 0.55; `center` 0 of 6 against 1 of 6, p = 1.00.

#### Time to a pass (agent wall seconds of passing attempts; `*` or `+` attempts are listed but not compared)

| op | harness | pass walls, s | all attempts' agent wall s | passes per agent hour (arithmetic, all attempts) | passes within the other harness's longest clean pass time |
|---|---|---|---|---|---|
| add | opencode | 75, 126 | 201 | 35.82 | 1 of 2 clean passes (T = 107 s) |
| add | orchard | 51, 107 | 158 | 45.63 | 2 of 2 clean passes (T = 126 s) |
| center | opencode | - | 2676 | 0.00 | 0 of 0 clean passes (T = 561 s) |
| center | orchard | 561 | 2518 | 1.43 | undefined (no clean pass by the other harness) |
| row_sum | opencode | 657, 223, 532 | 4598 | 2.35 | undefined (no clean pass by the other harness) |
| row_sum | orchard | 492* | 3270 | 1.10 | 0 of 0 clean passes (T = 657 s); a flagged pass is not counted |

#### Margins of the passing kernels (the unedited check; bars: PCC at least 0.999, relative L2 at most 0.01, slope 0.98 to 1.02)

| harness | op | n | PCC, worst set | PCC, outlier sets | relative L2, worst | slope range | held-out seeds 101 to 103 |
|---|---|---|---|---|---|---|---|
| opencode | add | 1 | 0.9999999 | 0.9999999 | 0 | 1 to 1 | pass |
| orchard | add | 1 | 0.9999999 | 0.9999999 | 0 | 1 to 1 | pass |
| orchard | add | 2 | 0.9999999 | 0.9999999 | 0 | 1 to 1 | pass |
| opencode | add | 2 | 0.9999999 | 0.9999999 | 0 | 1 to 1 | pass |
| orchard | center | 3 | 1 | 1 | 0 | 1 to 1 | pass |
| orchard | row_sum | 3 | 0.9999999 | 0.9999999 | 1.7e-07 | 1 to 1 | pass |
| opencode | row_sum | 4 | 0.9999999 | 0.9999999 | 0 | 1 to 1 | pass |
| opencode | row_sum | 5 | 0.9999999 | 0.9999999 | 0 | 1 to 1 | pass |
| opencode | row_sum | 6 | 0.9999999 | 0.9999999 | 0 | 1 to 1 | pass |

#### Failed attempts: how near the bar

| harness | op | n | class | near the bar | first line of the check's output |
|---|---|---|---|---|---|
| opencode | center | 1 | tt_lang_error | structural | error: TypeError: reduce_sum() missing 1 required positional argument: 'shape' |
| orchard | center | 1 | tt_lang_error | structural | error: TypeError: reduce_sum() missing 1 required positional argument: 'shape' |
| orchard | center | 2 | tt_lang_error | structural | Deadlock detected: all generators blocked |
| opencode | center | 2 | shape_mismatch | structural | error: ValueError: Tensor shape (32, 32) does not match Block shape (1, 8) (tile counts: 1 vs 8) |
| opencode | center | 3 | tt_lang_error | structural | error: ValidationError: 1 validation error for Block.__getitem__ |
| orchard | center | 4 | tt_lang_error | structural | error: ValueError: Shape mismatch in binary operation: left shape (1, 1) does not match right shape  |
| opencode | center | 4 | tt_lang_error | structural | error: ValueError: reduce shape (1, 8) does not match expected result shape (1, 1) (block shape (1,  |
| opencode | center | 5 | tt_lang_error | structural | error: TypeError: reduce_sum() missing 1 required positional argument: 'shape' |
| orchard | center | 5 | tt_lang_error | structural | error: ValueError: Shape mismatch in binary operation: left shape (1, 8) does not match right shape  |
| orchard | center | 6 | tt_lang_error | structural | error: IndexError: row slice 0:32 is outside the tensor, which has 2 tile(s) along it |
| opencode | center | 6 | shape_mismatch | structural | error: ValueError: Tensor shape (32, 32) does not match Block shape (2, 8) (tile counts: 1 vs 16) |
| opencode | row_sum | 1 | numeric_mismatch | numeric, far from the bar | set seed0: pcc 0.1757563203573227, finite True, bar 0.999 |
| orchard | row_sum | 1 | shape_mismatch | structural | error: ValueError: Tensor shape (64, 32) does not match Block shape (2, 8) (tile counts: 2 vs 16) |
| orchard | row_sum | 2 | numeric_mismatch | numeric, far from the bar | set seed0: pcc -0.0016424593050032854, finite True, bar 0.999 |
| opencode | row_sum | 2 | no_kernel | structural | kernel.py does not exist |
| opencode | row_sum | 3 | numeric_mismatch | numeric, far from the bar | set seed0: pcc 0.03266708925366402, finite True, bar 0.999 |
| orchard | row_sum | 4 | tt_lang_error | structural | error: RuntimeError: Cannot perform pop(): not a valid next dataflow step for this block. |
| orchard | row_sum | 5 | numeric_mismatch | numeric, far from the bar | set seed0: pcc 0.04401608183979988, finite True, bar 0.999 |
| orchard | row_sum | 6 | tt_lang_error | structural | error: ValidationError: 1 validation error for Block.__getitem__ |

#### Passing kernels against the chip compilers (1.1.10.dev20261026+light is the pair, the package the probe's hardware phase compiles with)

| harness | op | n | lines | 1.1.10.dev20261026+light | 1.1.6 | calls the old cheat sheet flags |
|---|---|---|---|---|---|---|
| opencode | add | 1 | 57 | compiles | compiles | none |
| orchard | add | 1 | 57 | compiles | compiles | none |
| orchard | add | 2 | 54 | compiles | compiles | none |
| opencode | add | 2 | 54 | compiles | compiles | none |
| orchard | center | 3 | 68 | refused: ttl.diagnostics.TTLangCompileError: error: <ast.Div object at 0x...> n | refused: ttl.diagnostics.TTLangCompileError: error: reduce_sum() got an unexpec | ttl.math.reduce_sum(shape), ttl.block.broadcast, ttl.block.fill |
| orchard | row_sum | 3 | 47 | refused: ttl.diagnostics.TTLangCompileError: error: constant type NoneType not  | refused: ttl.diagnostics.TTLangCompileError: error: constant type NoneType not  | ttl.math.reduce_sum(shape) |
| opencode | row_sum | 4 | 46 | compiles | refused: ttl.diagnostics.TTLangCompileError: error: reduce_sum() got an unexpec | ttl.math.reduce_sum(shape) |
| opencode | row_sum | 5 | 39 | compiles | refused: ttl.diagnostics.TTLangCompileError: error: reduce_sum() got an unexpec | ttl.math.reduce_sum(shape) |
| opencode | row_sum | 6 | 56 | refused: AttributeError: 'UnaryMinusOp' object has no attribute 'type' | refused: ttl.diagnostics.TTLangCompileError: error: Unknown function: ttl.grid_ | ttl.math.reduce_sum(shape) |

#### How the work got done (all attempts of an op and harness)

| op | harness | steps or turns (min-median-max) | writes of kernel.py | checks the agent ran itself | automatic checks after writes (orchard only) | attempts with no kernel | tool calls |
|---|---|---|---|---|---|---|---|
| add | opencode | 12-15-18 | 3 | 9 | 0 | 0 | glob 3, list 1, read 18, shell 10, write 3 |
| add | orchard | 9-11-13 | 6 | 6 | 6 | 0 | read_file 9, shell 6, write_file 6 |
| center | opencode | 40-40-42 | 25 | 20 | 0 | 0 | edit 2, glob 21, grep 7, list 4, read 105, shell 98, write 24 |
| center | orchard | 11-38-40 | 49 | 0 | 49 | 0 | read_file 93, run_shell_command 1, shell 68, write_file 55 |
| row_sum | opencode | 28-40-44 | 29 | 39 | 0 | 1 | glob 18, grep 10, list 1, read 105, shell 74, write 30 |
| row_sum | orchard | 13-39-40 | 47 | 3 | 47 | 0 | list_directory 1, read_file 139, shell 31, write_file 52 |

#### Tokens and server time (server counters are the change across each attempt; attempts with another client on the server are left out of the server columns)

| op | harness | prompt tokens, harness count | completion tokens, harness count | attempts with clean server counters | prompt tokens from the prefix cache | prefill s | decode s | generated tokens | decode tokens per s (arithmetic) | replies cut at 4096 (server count minus the probe's one speed-check request per attempt) |
|---|---|---|---|---|---|---|---|---|---|---|
| add | opencode | 1,269,006 | 5,204 | 2 of 2 | 90.2% of 1,269,058 | 30 | 162 | 5,804 | 35.7 | 0 |
| add | orchard | 198,120 | 5,824 | 2 of 2 | 79.6% of 198,172 | 10 | 143 | 6,424 | 45.1 | 0 |
| center | opencode | 11,301,921 | 69,935 | 6 of 6 | 94.1% of 11,302,077 | 182 | 2456 | 71,735 | 29.2 | 2 |
| center | orchard | 5,205,036 | 82,734 | 6 of 6 | 91.7% of 5,205,192 | 112 | 2350 | 84,534 | 36.0 | 3 |
| row_sum | opencode | 11,090,670 | 121,597 | 5 of 6 | 93.7% of 8,909,511 | 154 | 3064 | 86,633 | 28.3 | 10 |
| row_sum | orchard | 6,206,444 | 93,722 | 4 of 6 | 92.8% of 4,131,217 | 80 | 1688 | 55,335 | 32.8 | 4 |

Over the 25 attempts with clean server counters, decode seconds are 78% to 97% of agent wall time (median 92%) and prefill seconds are 3% to 17%.


Reading the tables:

- **Counts.** Simulator passes 5 of 14 (OpenCode) against 4 of 14 (orchard). `add` is solved by both, equally fast to within 24 s (OpenCode 75 s and 126 s, orchard 51 s and 107 s; clean attempts). `center` is solved once, by orchard
  (pair 3, 561 s). `row_sum` is solved three times by OpenCode (pairs 4, 5, 6: 657 s, 223 s, 532 s) and once by orchard (pair 3, 492 s, an attempt that shared the server). With 6 attempts per cell,
  3 of 6 against 1 of 6 is inside what chance gives (p = 0.55).
- **Failures.** All 19 failures are structural or numerically far from the bar (PCC 0.18 at best, others near 0): no near miss. The recurring simulator errors are the same in both harnesses:
  `reduce_sum() missing 1 required positional argument: 'shape'` (OpenCode `center` 1 and 5, orchard `center` 1: the old sheet told the coder to leave `shape` out, and the simulator requires it),
  shape and tile-count errors, one deadlock, one `pop()` that was not a valid dataflow step. OpenCode had one attempt with no kernel (`row_sum` 2: 40 steps of reading and shell commands in the simulator
  venv, no write).
- **Time to a pass.** Among clean passes OpenCode took 75, 126, 657, 223 and 532 s; orchard took 51, 107 and 561 s (its 492 s `row_sum` pass shared the server). Passes per agent hour over all attempts
  (arithmetic): `add` 35.8 against 45.6; `center` 0 against 1.43; `row_sum` 2.35 against 1.10. A long OpenCode attempt is mostly decode at a large context: its mean prompt per request is 26,000 to 66,000 tokens
  against 6,000 to 35,000 for orchard.
- **How the work got done.** The OpenCode agent ran `kernel_check.py` itself 68 times in 14 attempts and has no automatic check; orchard's loop runs the check after every write of `kernel.py` (102 automatic runs) and the agent asked for it only 9 times.
  OpenCode's attempts ended on its own answer 3 times, on the step limit 10 times and on an error once (two continuations after a reply cut at 4096 tokens). Orchard's ended on its own answer 3 times, on the turn limit 5 times,
  by the watchdog (escalate or pause) 5 times and on an error once (two replies in a row cut at 4096 tokens). OpenCode's step limit is soft: 42 steps on `center` 3 and 44 on `row_sum` 1 against a setting of 40.
  OpenCode reads whole files and explores (`read` 228 times and `shell` 182 in total against 241 and 105); it opened `kernel_check.py`, `kernel_harness.py` and `references.py`, which the probe's skill never names.
- **Tokens.** OpenCode sent 1,269,006 prompt tokens on `add` against 198,120 (6.4 times), 11,301,921 against 5,205,036 on `center` (2.2 times) and 11,090,670 against 6,206,444 on `row_sum` (1.8 times); completion tokens
  were close (5,204 and 5,824; 69,935 and 82,734; 121,597 and 93,722). The prefix cache served 90% to 94% of OpenCode's prompt tokens and 80% to 93% of orchard's, so the server computed few of them (prefill 3% to 17% of wall time).
  Replies cut at the 4096-token cap (server count minus the probe's one speed-check request per attempt, over attempts with clean counters): `add` 0 and 0, `center` 2 and 3, `row_sum` 10 and 4. The harnesses' own counts agree; OpenCode `row_sum` 1 has one more at the server because of a continuation.

### Kernel quality that the simulator pass does not show

- **Margins and held-out seeds.** No difference: all nine passing kernels are within rounding of perfect on the simulator's measures and pass the three held-out seeds (101 to 103; the shipped check, byte-identical, with only
  the spec's seed list changed; the shape of the inputs was not varied).
- **Chip compilers (compile step on the CPU, no device; `scripts/hwcompile_probe_passes.py` of PR 802).** The pair's compiler accepts 4 of OpenCode's 5 passes (`add` 1 and 2, `row_sum` 4 and 5) and refuses `row_sum` 6
  (`AttributeError: 'UnaryMinusOp' object has no attribute 'type'`). It accepts 2 of orchard's 4 (`add` 1 and 2) and refuses `center` 3 (`<ast.Div> not implemented`: a division inside a thread) and `row_sum` 3
  (`constant type NoneType not implemented`). tt-lang 1.1.6 accepts the four `add` kernels and refuses the others (`reduce_sum() got an unexpected keyword argument 'shape'` for OpenCode `row_sum` 4 and 5). The compile step is
  the first check only: nothing was built with tt-metal and nothing ran on a chip.
- **The old cheat sheet's static check.** It flags `ttl.math.reduce_sum(shape)` in five passing kernels (orchard `center` 3 and `row_sum` 3, OpenCode `row_sum` 4, 5 and 6). The pair's compiler does not refuse any of the five for `shape`: two compile on the pair
  (OpenCode `row_sum` 4 and 5) and three are refused for other reasons. The flag comes from the sheet's wrong claim about `shape` (caveat 1), so it is not evidence about the kernels.
- **Shape of the kernels.** All nine have 39 to 68 lines and follow the reference example's structure (operation, compute and two data-movement threads, dataflow buffers, wait, reserve, store).

### Answer to the question in addendum 1

Where OpenCode took longer, did it buy a better kernel? On `add`, the only op where it was slower among clean attempts (about 24 s and 19 s), the kernels are equal: same 54 and 57 lines, same calls, same margins, same compile result. On `center` 3 it took 1184 s and
failed, where orchard's pair partner passed in 561 s. Where it passed `row_sum` (pairs 4 to 6) it was not slower than orchard's clean attempts, and 2 of those 3 kernels are accepted by the pair's compiler, against 0 of orchard's 1
`row_sum` pass; with this many attempts that is a hint and not a result (p = 0.48 for 2 of 12 against 0 of 12 over `center` and `row_sum`). The other way round: where orchard was slower (`center`, clean median 542.5 s against 298 s;
`row_sum` 624.5 s against 532 s) it did not buy a better kernel either: its two non-`add` passes are both refused by the chip compilers. Neither harness bought quality with time in this data.

### What the counts cannot show, and what is not verified

1. **Both harnesses got a plain cheat sheet with wrong chip claims** (issue 801, fix in PR 803). It calls `ttl.math.fill` and `ttl.math.broadcast` the hardware names; both chip compilers refuse them (`ttl.block.*` is
   the accepted form). It also says the chip's package has no `shape` argument on `reduce_sum`; that holds for tt-lang 1.1.6 and not for the pair, the package the probe's hardware phase compiles with. Every attempt here had a wrong claim, so the pass counts are
   counts for that sheet, and the `center` and `row_sum` failures that end on the `shape` message follow from it. Earlier in this log (pilot section) I read that conflict as the simulator against the chip; it is the sheet's. Runs with the corrected
   sheet are not comparable with these.
2. **Simulator passes are not chip passes.** The coder holds all four chips, so no kernel ran on a chip and none was built with tt-metal. The compile probe is the nearest check available on the CPU; its acceptance is the first
   step only.
3. **The recipes A/B of 2026-11-21 was retracted** (on passes a chip compiler accepts it is 1 of 7 for recipes against 0 of 8 for plain, Fisher exact p = 0.47). Nothing in this log uses recipes, and the same caution applies here:
   every comparison above has p of 0.48 or more.
4. **Size.** Two to six attempts per cell, one model, one server profile, one task family. A pass count of 3 against 1 is inside chance.
5. **Conditions.** 10 attempts ran inside load windows, and 3 of those 10 also shared the server with another client (listed above). The OpenCode attempts ran in a stricter sandbox than orchard's agent shell (Landlock, no
   secrets, no devices, connect to the model port only), with six working tools and two that only return errors, a longer system prompt and context compaction, and without the automatic check after each write. The two OpenCode pilot attempts had a 15-tool list
   (none of the extras was used). The seed reaches the server through the model options; whether the server repeats a seeded draw is not checked.
6. **tenstorrent/skills in a stage is not measured.** The skill-discovery test shows discovery and loading of single skills for fact questions; it does not show that a stage would go better or worse with them.

### Conclusion

OpenCode v2 is not shown to be a better or a worse harness than orchard's loop for tt-lang kernel work on this data: 5 against 4 simulator passes in 14 attempts each, and 4 against 2 passes that the pair's compiler accepts, are both inside chance.
It costs 1.8 to 6.4 times the prompt tokens and, at its long contexts, decodes more slowly; its clean attempts were slower only on the easy op. Its passing kernels are as good as orchard's on every measure the simulator gives, and more of them
(4 of 5 against 2 of 4) are accepted by the chip compiler, which is worth testing with more attempts. It finds and loads skills by itself most of the time. It does not replace the orchard loop's watchdog, automatic check and
fail-closed shell; for a kernel-exploration role it is a candidate for a larger, quieter run. This data does not justify a spec to replace the orchard loop.

**Next experiment.** After PR 803 lands: the same three ops plus `rmsnorm`, at least 10 attempts per cell and harness, on a quiet host with nobody else on the coder, the corrected sheet, the compile probe on every simulator pass as the
primary result, and the same ABBA order and seeds. Then a stage-level test: one bring-up stage (stage 0 or 1) through OpenCode with the supervisor's messages given as instructions, comparing injected skill text with native discovery.

### Data

The tables above come from `scripts/oc_report.py` over the files in `docs/run-logs/2026-11-21-opencode-v2-vs-orchard-harness-data/` (timeline, per-attempt quality records, compile probe output, load windows, load conditions, skill-discovery results, and
the nine passing kernels; home paths and the host name are replaced). The full transcripts (the OpenCode event streams, 5 MB to 30 MB each) stay on the host under `~/orchard-runs/ocprobe/`.
