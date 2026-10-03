# Hardware validation runbook

Status: not run. Nothing here has touched a device. Use it when the operator says a board is ready.

## Purpose

The supervisor design relies on facts that were read from gozer's code and never checked on this
machine (spec section 14, item 2). Check them under a lease, in one session, before plan 3 builds on
them. Each check records its result in a run ledger.

| # | Question | Why it matters |
|---|---|---|
| H1 | Does a lease taken with `--owner-pid` survive more than 15 minutes with no device open? | The swap leaves the board with no device open for minutes. |
| H2 | Does a model server started as a child of the owner show as `HELD` and never as `HELD-FOREIGN`? | The supervisor starts the servers itself. |
| H3 | Does `gozer reset <lease>` refuse while a server holds the device, and reset the chips in place when none does? | It is the reset step of the swap. |
| H4 | Can a process open the device after an in-place reset? | A reset that leaves the chips unusable is worse than none. |
| H5 | How does a container server look to gozer, and how long does a stop, reset and restart take? | gozer cannot see containers. The supervisor's budgets need real numbers. |
| H6 | Does `gozer release` free the board and leave nothing behind? | The end of every run. |

## Rules for the session

- Start only after the operator says the board is ready. The training run in the tt-tnt agent must be
  finished and its lease gone. Confirm with `gozer status` and `docker ps` first.
- Use only the board the operator names. Take it with `--exact`, one board, `--no-queue`. Never touch
  the other board.
- Do not run `tt-smi -r` by hand. The only resets are `gozer reset` and the reset inside
  `gozer release`.
- Do not run `gozer reconcile` while another agent's lease is on the box. It reaps every stale lease,
  including other agents' leases. Use `gozer status`, which never changes state.
- One long-lived process owns the lease: a driver script. A series of one-off shells would give the lease a different pid each time. Its own pid
  (`$$` in bash, `os.getpid()` in Python) is the `--owner-pid`. Every server and test runs as its
  child. A release in a `trap ... EXIT` handler frees the board if the driver dies.
- Record each result in a ledger (`orchard.ledger.Ledger`, event `measurement` or `notice`), with the
  exact command and its exit code. A surprise is a result. Stop and report it. Do not work around it.
- Time budget: about two hours, of which H1 is 16 minutes of waiting.

## Before the session (no hardware)

1. Write the driver and run it against fake gozer roots (`GOZER_ROOT`, `GOZER_SYSFS_ROOT`,
   `GOZER_PROC_ROOT`, `GOZER_RESET_CMD=/bin/true`), as the tests in `tt-gozer` do. A driver that has
   never run is the wrong thing to meet a real board with.
2. Confirm the environment for ttnn code: `~/code/audio8-asr/dev_env.sh` is the known-good example
   (source-build tree, `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0`, and `TT_MESH_GRAPH_DESC_PATH` set
   to the p150 descriptor when only one chip of a two-chip board is used). The board-1
   `ttnn.get_num_devices()` IndexError is known. If a device open fails, that failure is a result.
3. Decide the board and its device indices from `gozer topology`, then check that the indices you
   pass to `tt-model serve --device-id` are the board's two chips.
4. Check that the Audio8 container package can be served locally (`tt-model list` does not list it;
   it was built from a local directory, see `~/code/audio8-asr/NOTES.md`), or choose an installed
   container package whose profile fits one board. H5 needs one.

## The checks

Take the lease once, at the start, and keep it:

```bash
gozer acquire --exact <board BDF or unit> --chips 1 --who "orchard:hardware-check" \
    --reason "validate hold-through-swap" --owner-pid $$ --no-queue --json
```

Take `lease_id` from the JSON. `eval "$(gozer env <lease-id>)"` sets `TT_VISIBLE_DEVICES`.
Record the grant: the chips, the dev indices, `detached` (expect false) and `pid`.

**H1. The lease survives idle time.** With nothing running on the board, wait 16 minutes (900 s is
the detached window). Then run `gozer status`. Expect the board's chips to show `CLAIMED`. `STALE` would mean the lease is no longer judged by the owner pid.
This tests that the lease is judged by the owner pid. It cannot show that a reap would spare it,
because `status` never reaps. If no other agent's lease is on the box, `gozer reconcile` does show
that, and it is worth running then.

**H2. A child of the owner shows as HELD.** Start a small process as a child of the driver that opens
one device and waits (for example `ttnn.open_device(device_id=0)` for 60 s under the dev environment).
While it holds the device, `gozer status` must show the chip `HELD`. `HELD-FOREIGN` here is a failure
of the descendant rule on a real `/proc`; record the process tree (`ps -eo pid,ppid,args`).

**H3. Reset refuses with a holder, resets without one.** While the child holds the device, run
`gozer reset <lease-id>`. Expect exit 15 and `still open`. Stop the child, confirm with `ps` that it
is gone, then run `gozer reset <lease-id>`. Expect exit 0. Record how long it took, and the output of
`tt-smi -s` afterwards (a read-only snapshot).

**H4. The device opens after an in-place reset.** Start the same child again. Expect a normal open.
Then stop it. Record any error text exactly.

**H5. A container server.** Serve a container package on the board with
`tt-model serve <package> --device-id <the board's indices> --detach --port <free port>`. Record the
time to ready. While it runs, `gozer status` is expected to show the chips `CLAIMED` and not `HELD`,
because gozer cannot see the container. While the container is up, run the stop-confirmation
checks from the `gozer-park` skill ("Before every reset") and confirm that each one **shows** the
container: the `docker ps` line with its image id and port, and the `docker inspect` line with its
device paths. These commands have never seen a running container, so a check that cannot show the
container cannot show that it is gone. Record which checks work and which show nothing (a container
started with `--privileged` may list no device paths). Only observe. Do not run `gozer reset` while the
container is up: a reset under a live server can wedge the board, and the code reading and the
tests already establish that gozer would not refuse. Stop with `tt-model stop <package>` and record
the stop time and whether it reported a mesh reset. Confirm with `docker ps` that the container is
gone, run `gozer reset <lease-id>`, serve again, and record the warm time to ready. Earlier
measurements on this machine for the Audio8 package: clean stop 1.6 to 3.9 s, warm boot about 20 s,
cold boot 70.8 s.

**H6. Release.** Run `gozer release <lease-id>`. Expect exit 0, a reset, the board `FREE` in
`gozer status`, no container in `docker ps`, no server process, and no stray files under
`$GOZER_ROOT` for our lease.

## Stop conditions

Stop and report, without trying a workaround, if: the board is a different one than the operator named; any
command touches the other board; `gozer status` shows an unexpected lease on our board; a device open
hangs for more than five minutes; `gozer reset` exits 17 or 18; the lease disappears from
`gozer status` while the driver is alive.

## The first real session

Decided on 2026-10-02 with the operator: one board, alone, supervised, with a short idle wait. The
full 960 second idle wait and two drivers at once come only after one clean run on each board.

1. Before starting: `env | grep GOZER` prints nothing in the shell that runs the driver (the driver
   refuses to run otherwise), and the shell is inside tmux so a closed terminal cannot cut the
   cleanup output.
2. The tt-tnt training agent has been told to yield when asked. Ask it to stop at its next
   checkpoint and release lease `ed5f29` (or whatever lease holds the board). Wait for `gozer status`
   to show the board `FREE`. Do not run `gozer reconcile` and do not touch that lease. The queue may
   hold a ticket from another agent (ticket `062d`, `claude:tt-bio-0.12.0-upgrade`, one chip): when
   the lease is released, the head ticket gets a 90 second claim window and the driver's
   `--no-queue --exact` acquire is refused until that window ends. That refusal is safe. Check
   `gozer status` and `gozer queue`, and start the driver when board 1 is `FREE` and the window has
   passed or the ticket holder has taken board 0. Never jump the queue.
3. Run the driver from the tmux terminal on board 1 (chips 2 and 3, first chip `0000:03:00.0`),
   with the branch gozer named explicitly:
   `python3 -m orchard.hardware_check --board 0000:03:00.0 --gozer /home/ttuser/code/tt-gozer-orchard/bin/gozer --idle-seconds 60 --pause-after-open 20`.
   `--tt-smi` stays off (its default). `tt-smi -s` opens every device on the box. The first open
   starts with an empty per-run JIT cache, so it can take longer than later opens; the driver
   waits up to 300 seconds for it.
4. Keep a second terminal on read-only `gozer status`. During the 20 second pause after the first
   OPENED, check by hand that `gozer status` shows the board's first chip `HELD`, and that
   `ls -l /proc/<child pid>/fd` lists only `/dev/tenstorrent/2` and `/dev/tenstorrent/3`. If it
   lists anything on board 0, send SIGTERM to the driver (it cleans up) and stop.
5. Recovery if the driver dies or hangs: wait for any child holding the device to exit, then
   `gozer release <lease-id>` without `--force`. Never run `tt-smi -r` by hand. Do not send the
   driver a signal while a `gozer reset` or `gozer release` is starting.
6. When the run ends, confirm that `gozer status` shows the board `FREE`, that
   `/tmp/tt-gozer/history.jsonl` has a `released` event for our lease with `reset_ran` and
   `reset_ok` true, that the out-dir has `summary.md`, and that no process is left in the child's
   group. Tell the tt-tnt agent that the hardware is free, and put the ledger and `summary.md`
   paths in the report.

## After the session

Write the results into the run ledger and into `docs/superpowers/specs/2026-10-01-orchard-design.md`
section 14, item 2. If H1, H2 or H3 failed, the fallback in section 8 (a reservation with `yield` and
`redeem`) comes back into the plan. If everything passed, plan 3 can rely on hold-through-swap, and
the measured stop, reset and boot times replace the estimates in sections 3 and 6.

## Park check (plan 3)

Purpose: run the park and restore code (`orchard/handoff.py`) through the gozer adapter on one
real board, with fake servers in place of the coder and the CPU stand-in. It checks the adapter
against the real gozer and a real board reset, the stop checks against a real process, and the
ledger steps. It does not check a device held open (the hardware-check driver above does), a
container server, a real stand-in model, or a real restart time.

Who runs it: the controller. An implementer does not. It takes its own gozer lease under its own pid.

Before running:
- `gozer status` shows every chip of the target board `FREE`, and the other board is not
  `BUSY-UNTRACKED` (another reset is running). The driver refuses otherwise (exit 2).
- No `GOZER_*` variable is set.
- Ports 20990 and 20991 are free (`ss -ltn "( sport = :20990 or sport = :20991 )"` prints only
  its header), or pass `--port` and `--standin-port`.

Run, from the repo root:

    python3 -m orchard.park_check --board 0000:03:00.0

Expected, measured on the hardware-check runs: three resets of the board, about 42 s each (the
park reset, the restore reset, and the reset inside the release), about 2.5 min in all. The
ledger under `runs/park-check/<UTC time>-<BDF>/` shows the park steps `note`, `canary_before`,
`standin_started`, `standin`, `stop_sent`, `stopped`, `reset`, then the restore steps `reset`,
`serve`, `ready`, `canary`, `resumed`, and three measurements (`park_reset_seconds`,
`restore_reset_seconds`, `coder_ready_wait_seconds`). Exit 0. Afterwards `gozer status` shows the board `FREE`.

Stop conditions: any exit other than 0. Read `summary.md` and the last `notice` in the ledger.
If the summary says the release failed, wait until nothing holds the board, then run
`gozer release <lease-id>` from the ledger's `lease taken` decision. Never use `--force`, never
run `tt-smi -r` by hand.

Record afterwards: the run directory, the exit code, the three reset times, and anything in the
summary marked STOP.

## Hemmingway-1 run (plan 4)

Purpose: the first bring-up driven by the harness and not by Claude by hand. The supervisor takes
`Altworld/Hemmingway-1` (a creative-writing fine-tune of Qwen3.8-27B with the same architecture)
from stage 0 to the operator bundle. It parks the large coder for every hardware stage, records
everything in a ledger, and stops at "ready for operator review". It publishes nothing. The run
also checks stage 0 against the hand-written reference answer.

Who runs it: the controller, with the operator's agreement. An implementer does not. It takes all
four chips for hours.

Before running:
- The operator has answered the open decision about the command runner (spec section 10 and
  section 14, item 5). Do not start without that answer.
- The operator confirms which 4-chip package serves the large tier. `config/tiers.toml` names the
  model (`Qwen/Qwen3.8-27B` on port 8000) and no package. The installed 4-chip candidate is
  `mando2222/qwen3.8-27b-dflash2-p300x2-q4kv` (container, profile `batch8-dflash2`). The model id
  that package serves must equal the large tier's `model`, because the supervisor checks
  `/v1/models` for it. If it differs, the operator edits `config/tiers.toml`.
- `gozer status` shows all four chips `FREE` and no other lease. No `GOZER_*` variable is set.
- ollama serves the CPU tier: `curl -s http://127.0.0.1:11434/v1/models` lists `qwen3-coder:30b`.
- Nothing listens on port 8000: `ss -ltn "( sport = :8000 )"` prints only its header.
- The run directory is on `/mnt/bonus` (404 GB free on 2026-10-02). The root disk is 99% full,
  and stages 2 to 6 each require 40 GB free (stage 4 requires 80 GB).
- Do not give the harness the reference answer, the nearest supported model or the base model's
  tensor cache. The nearest model (Qwen3.8-27B) and its revision are lines of the reference
  answer, and every `--input` appears in every agent prompt, so the run command below has no
  `--input base=`. Stage 0 must find the nearest model, its revision and the tensor cache on its
  own. Record in the results that the nearest model was NOT given.
- The run directory is `/mnt/bonus/models/orchard-runs/hemmingway-1-run1`, outside the
  `/mnt/bonus/models/hemmingway-1/` tree. That tree holds `work/stage0-reference.md` and possibly
  notes from the earlier hand bring-up, and agent shells start in the run directory. The model
  input and `HF_HOME` are still inside that tree, so an agent that lists it can find `work/`.
- Agents can still read any absolute path the user can read. The reference answer exists at
  `/mnt/bonus/models/hemmingway-1/work/stage0-reference.md` and inside this repo at
  `tests/fixtures/hemmingway_stage0_reference.md`, and all of `/mnt/bonus` is readable. Nothing in
  the harness stops an agent from reading either file. The check comes after the run (below).
- Credential files. The supervisor refuses to start (exit 2, naming each path) while any of these
  exist in the operator's home: `~/.cache/huggingface/token`, `~/.config/gh/hosts.yml`,
  `~/.ssh/id_*` private keys, `~/.netrc` and `~/.docker/config.json`. On 2026-10-02 the
  Hugging Face token, the gh hosts file, an ssh key and the docker config existed on this box
  (the final review checked existence only). Move them aside for the run and put them back afterwards, or
  pass `--accept-credentials-visible`; the ledger then records that the operator accepted it.
  Agent shells run as the same user, so code an agent runs can read any file that stays. Agent
  shells get `GIT_SSH_COMMAND` set so git over ssh uses no key, and the runner refuses ssh, scp,
  sftp and rsync to a host. If the ssh key stays in place, confirm it needs a passphrase:
  `ssh-keygen -y -P '' -f ~/.ssh/id_ed25519` fails when it does. The preflight does not look in
  `HF_HOME`: check that `/mnt/bonus/models/hemmingway-1/hf` holds no `token` or `stored_tokens`
  file. `HF_TOKEN_PATH` keeps huggingface_hub away from a token there, and it does not stop `cat`.
- Stage 4 needs the `mesh-shrink` skill, which is untracked in the skills repo
  (`plugins/tt-model-bringup/skills/mesh-shrink/`). Do not switch branch or clean that tree
  during the run, or stage 4 blocks with "skill not found".

What stops an agent and what does not. Agent shells run as the same user as the supervisor. The
command runner (`orchard/runner.py`) refuses these spellings: tt-smi resets; every gozer command
except status, env, queue, history, --help and --version; the tt CLI's reset, firmware and
serving verbs; docker and tt-model commands that start, stop, remove or push; kill, pkill,
killall and systemctl stop; ssh, scp, sftp and rsync to a host; Hugging Face uploads; curl and
wget requests that send data; and git push. Agent shells also get `TT_VISIBLE_DEVICES` and
`TT_METAL_VISIBLE_DEVICES` set to `0000:ff:00.0`, which matches no chip. Checked on hardware on
2026-10-02: a device open with that mask fails with `RuntimeError: BDF pattern 0000:ff:00.0 did not match any devices`. These are NOT stopped: `python3 -c` and `perl
-e` code, a script the agent writes, `mv`, `cp` and `tee` (which can overwrite any file the user
can write, the ledger included), and a direct device open from Python. A script that opens a mesh
from an agent shell opens chips the supervisor holds. The stage prompts tell the agent not to do
this, and that is a request only. The operator must accept this risk, in writing, before the run
starts.

Run, from the repo root:

    python3 -m orchard.supervisor run \
      --model Altworld/Hemmingway-1 \
      --run-dir /mnt/bonus/models/orchard-runs/hemmingway-1-run1 \
      --tiers config/tiers.toml \
      --coder-target mando2222/qwen3.8-27b-dflash2-p300x2-q4kv --coder-kind container \
      --coder-profile batch8-dflash2 --coder-port 8000 --coder-chips 4 \
      --required-chips 2,4 \
      --skills-dir /home/ttuser/code/skills/plugins/tt-model-bringup/skills \
      --input model=/mnt/bonus/models/hemmingway-1/hf/hub/models--Altworld--Hemmingway-1/snapshots/1a5f363a3dd2d1cc456c28b8abbb403b9555efaf \
      --env HF_HOME=/mnt/bonus/models/hemmingway-1/hf --env HF_HUB_OFFLINE=1

`--required-chips 2,4` names the chip counts stage 4 must pass for this model. The 1-chip
configuration is optional: the agent should try it and record it with pass false and a reason if it
does not work, and that does not fail the stage. A required count that is missing from stage 4's
result.json or did not pass fails the gate. The ledger records the option in `run_start`, and a
resumed run keeps it (a resume that names different counts is refused). Without the option, every
configuration stage 4 lists must pass.

What happens: the supervisor records the run and the versions it can read (`tt-model --version`,
the firmware from `tt-smi -s`; tt-metal and vLLM are recorded as TODO). It takes a 4-chip lease
under its own pid and starts the coder (a cold boot is about 30 min, budget 45 min). Stages 0 and 1
run on the large server; the small tier is not serving, and the ledger records each substitution.
An escalated step goes to the same large server with a different context, because it is the only
chip tier serving.
On its first start in a run the coder is asked "What is 7 times 6? Reply with only the number." and
must answer with text containing 42. If it does not, the run pauses with a reason that quotes the
answer and says the server answers wrongly, and the answer is saved under `evidence/coder-sanity-*.txt`.
Nothing is retried. Fix the package or the server, then send `control resume`; the question is asked
again. Later starts in the same run skip it, because they compare against the first canary answer.
Each of stages 2 to 6 runs a prepare step, parks the coder (stand-in canary on ollama, stop, reset),
runs the agent's test command on board 0's two chips, restores the coder (reset, start, canary
compared with the pre-park answer) and runs a finish step. Each reset measured 41.7 s and a warm
restart 2 to 3 min. The total run time is not measured. Stage 7 is recorded as skipped. Stage 8
writes the bundle, the supervisor scrubs it, and then it stops the coder and releases the four
chips. Restart the operator's own coder afterwards if it is wanted.

The path stage 0 chooses changes stages 2 and 3. With `"path": "weights-only"` in
`stages/0/delta.json` (the expected answer for Hemmingway-1), stage 2 runs the
`weights-swap-check` skill (`orchard/skills/weights-swap-check.md`) in place of
`functional-decoder`. Its hardware test converts the new weights into a fresh tensor cache, starts
the nearest model's bundle on the new weights and compares the chip's greedy tokens with the
stage 1 reference. The gate (`gate_weights_swap`) needs `serves` and `coherent` true, at least 16
compared tokens, `top1_agreement` of at least 0.85 (`SWAP_TOP1_MIN`) and every evidence file. The
test sets `MODEL_WEIGHTS_DIR` and `HF_MODEL` to the model directory; without them the bundle serves
the nearest model's weights, which agreed on 25 of 32 tokens and passed the old bar of 0.6. Stage 3 is then recorded
as skipped, with the reason "weights-only path: the stage 2 serve-and-compare covers the full
model", and the run goes on to stage 4. A `full-port` path pauses before stage 2 as before and,
after `control resume`, runs `functional-decoder` and stage 3. A path the supervisor cannot read
keeps `functional-decoder` and stage 3; it never guesses weights-only.
Stage 0's passing `stage_end` records the path, and the supervisor reads it from there before each
stage, so an agent's later edit to `delta.json` does not change it. A run whose stage 0 passed under
older code has no path in that entry; on resume the supervisor reads `delta.json` instead. The
stage 2 budget (14,400 s) is longer than the skill's test deadline (2,400 s), so the test gets its
full deadline. The skill's tensor cache goes under `/mnt/bonus/models/orchard-runs/cache/`, on the
same disk as the run directory, and the 40 GB free-disk check covers one converted cache (34 GB
measured for the base model).

Watch and steer, from another shell:

    tail -n 5 /mnt/bonus/models/orchard-runs/hemmingway-1-run1/ledger.jsonl
    python3 -m orchard.supervisor control --run-dir /mnt/bonus/models/orchard-runs/hemmingway-1-run1 pause
    python3 -m orchard.supervisor control --run-dir /mnt/bonus/models/orchard-runs/hemmingway-1-run1 resume
    python3 -m orchard.supervisor control --run-dir /mnt/bonus/models/orchard-runs/hemmingway-1-run1 abort

A paused run holds its leases and waits. Read the last `notice` and `decision` entries before
resuming. `control abort` is the normal way to stop: it stops the coder, releases the lease and
ends the run, which then cannot be resumed.

Ctrl-C (SIGINT) and `kill <pid>` (SIGTERM) on the supervisor take the same abort path. The
supervisor kills the agent's running command or the hardware test, stops the coder, releases
every lease it holds, records the abort and exits 4. This can take a few minutes, and a second
Ctrl-C during it is ignored. Any other error that ends the run (exit 3, printed as `error:`) also
stops the coder and releases the leases. That run is not aborted: fix the cause and run the same
`run` command again to resume. Exit 2 (`refused:`) means nothing was started.

Recovery after a SIGKILL, a reboot or a Python crash, where no handler runs: the coder keeps the
four chips under a dead pid's lease. Run the same `run` command again. It replays the ledger,
stops that coder, takes a new lease under its own pid, starts the coder again (a cold boot of up
to 45 min) and resumes. To end the run instead, send `control abort` once it is running; it stops
the coder and releases the lease at its next check.

Compare stage 0 with the reference answer once stage 0 has passed:

    python3 -m orchard.stages compare-delta \
      /mnt/bonus/models/orchard-runs/hemmingway-1-run1/stages/0/delta.json \
      /mnt/bonus/models/hemmingway-1/work/stage0-reference.md

Exit 0 means every difference area and hazard in the reference has an entry and the path agrees
(`weights-only`). It shows that the shape of the answer is complete. It does not show that the
findings are right: the delta-triage skill lists the same hazard areas and requires an entry for
every area, so a run that follows the skill will nearly always exit 0. A person reads each finding
against the reference, in particular the tokenizer's combining-mark difference and the
tensor-cache hazard, and records the verdict.

Check that no agent read the reference answer. After the run, search the ledger, the transcripts
and the evidence for either name of the reference file:

    grep -rn -e stage0-reference -e hemmingway_stage0_reference \
      /mnt/bonus/models/orchard-runs/hemmingway-1-run1

Report every hit with its file and line. A hit in an agent transcript or evidence file means the
stage 0 result cannot count as found on its own. The compare-delta command above names the
reference path, and its output is not saved in the run directory, so it does not cause a hit.
On 2026-10-02 neither name appeared in `orchard/`, `config/` or the tt-model-bringup skills, so
the prompts themselves do not cause a hit either.

Stop conditions: a pause whose reason you cannot explain, a `blocked` notice from a park or
restore, a canary that changed, or any sign that something was published. Never use `--force`,
never run `tt-smi -r` by hand, and never run the commands in `PUBLISH_COMMANDS.txt`; they are for
the operator.

Record afterwards: the run directory, the exit code, each stage's result, the compare-delta output
and the person's reading of the findings, the reference grep and every hit, that the nearest
model was not given, every number in stage 6 with its label, every pause with its
reason, and the wall time.
