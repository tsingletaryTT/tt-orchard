# OpenCode v2 against orchard's agent loop, on the tt-lang and tenstorrent skills (2026-11-21)

Branch `opencode-probe-eval`. Tools: `scripts/opencode_probe.py`, `scripts/landlock_exec.py`, `scripts/oc_request_proxy.py`. No file under
`orchard/` changed. Pre-registration first; results follow in this file.

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

**Runner edit during the pilot.** After the pilot began, optional parameters (`skill_tool`, `skill_paths`) were added to `run_opencode` and
`write_config` for the skill-discovery test. Their defaults reproduce the earlier behaviour, so the pilot's attempts are unaffected.

## Pilot

(to be written)
