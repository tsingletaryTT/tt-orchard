# tt-orchard

tt-orchard is a supervisor program that brings a new model up on a Tenstorrent machine. It drives
the work with open-source models served on the same machine, and it calls no remote model service.
A run starts from a Hugging Face model id and moves through nine stages, from comparing the new
model with a model that already runs on the hardware to serving and measuring it on the chips.
Every step, decision and evidence file is recorded in an append-only ledger, so a run can be
paused, resumed and recovered after a crash. A run ends at a bundle of results, risks and draft
publish commands for a person to review, and the harness never publishes, pushes or uploads
anything.

## Contents

1. [Status](#1-status)
2. [What you need](#2-what-you-need)
3. [Install](#3-install)
4. [Run a bring-up](#4-run-a-bring-up)
5. [Operate a run](#5-operate-a-run)
6. [Lessons that will bite you](#6-lessons-that-will-bite-you)
7. [Where things are](#7-where-things-are)
8. [Safety and license notes](#8-safety-and-license-notes)

## 1. Status

tt-orchard is early. It has driven part of one bring-up on one machine. The machine is a
Tenstorrent QuietBox 2 with two p300c boards (four Blackhole chips). The model was
`Altworld/Hemmingway-1`, a fine-tune of `Qwen/Qwen3.8-27B` with the same architecture. That is one
model from one family. Nothing has been run on any other machine.

The harness has reached "ready for operator review" once, on one model, with a person watching: stages 0, 1, 2, 4, 7 and 8 passed and
stages 3, 5 and 6 were skipped by design. It has never run unattended from start to finish. During that run a person paused
it, fixed the code, and restarted it many times. Each fix is described in
[docs/run-logs/2026-10-hemmingway-1.md](docs/run-logs/2026-10-hemmingway-1.md).

### Stages

| # | Stage | On real hardware |
|---|---|---|
| 0 | Intake and delta triage against the nearest supported model | Passed on Hemmingway-1 (27 min, 59 agent turns, path `weights-only`) with an open-ended skill. Failed twice on the second model (iapp/openthai2.0-qwen3.8-27b): the agent explored and never wrote `delta.json`. Since 2026-10-04 a template script (`delta_triage.py`) measures and drafts `delta.json` and the agent reviews it; tested with fakes, not yet run on hardware |
| 1 | Environment and CPU reference | Passed on Hemmingway-1 on the fifth attempt (8 min, 25 turns), after fixes to the agent loop and with agent thinking turned off. Since 2026-10-04 a template script (`reference_gate.py`) loads the model, runs the four checks and drafts `reference.json`; tested on a tiny random model, not yet run on hardware |
| 2 | Functional decoder on one chip; on the weights-only path, a weights swap on one board | Passed on Hemmingway-1 (weights-only path): the chip agreed with the CPU reference on 30 of 32 tokens (0.9375, measured, one prompt) |
| 3 | Full model | Skipped by design on the weights-only path. Never run on the full-port path |
| 4 | Multichip, then shrink to fewer chips; on the weights-only path, one test per chip configuration | In progress on Hemmingway-1 when this was written. No result yet |
| 5 | Serving integration | Skipped by design on the weights-only path. Stage 4's serve-and-compare tests boot each configuration, compare the output with the CPU reference and check that it is coherent. Never run on the full-port path |
| 6 | Qualitative check and benchmark | Skipped on the weights-only path, because the operator deferred it. Stage 4's coherence check is the only quality evidence there. Never run on the full-port path |
| 7 | Package and container build | Passed once, on Hemmingway-1 (weights-only path, `--package-format v6`): staged a 2-chip and a 1-chip v6 thin bundle with the real `tt-model package-thin`, installed a copy of the 2-chip bundle and booted it from empty caches on a leased board (ready in 2034 s, top-1 agreement 0.9375 with the CPU reference). The 1-chip bundle is staged and not boot-checked. No 4-chip package is built, because the 4-chip packages on the development machine are container packages. Every other run records the stage as skipped |
| 8 | Operator bundle | Passed once, on Hemmingway-1, with an open-ended skill: wrote `RESULTS.md`, `RISKS.md`, `card.md`, the packages and the publish commands as text, and the run ended at "ready for operator review". Nothing was published. Failed twice on the second model: the agent read stage 7's files with 33 `cat` commands and wrote nothing for 20 turns. Since 2026-10-04 a template script (`build_bundle.py`) builds the whole bundle from the run's files and the agent reviews it; tested with fakes, not yet run on a real run |

The full-port path (a model that needs new model code) has never run. When stage 0 chooses it, the
supervisor pauses before stage 2 for the operator.

### Parts

| Part | Files | State |
|---|---|---|
| Run ledger | `orchard/ledger.py` | Built and tested. Used on the real run |
| Command runner (the shell guard for agents) | `orchard/runner.py` | Built and tested. Best effort; see [Safety](#8-safety-and-license-notes) |
| Tier config loader | `orchard/tiers.py` | Built and tested. Used on the real run |
| Lease adapters (gozer, single tenant) | `orchard/adapters/` | Built and tested. The gozer adapter was used on the real run. The single-tenant adapter is tested with fakes only and the command line does not offer it |
| Server control and canary | `orchard/server.py`, `orchard/canary.py` | Built and tested. Used on the real run |
| Park and restore | `orchard/handoff.py` | Built and tested. The park check ran once on one real board with fake model servers. No real model has been parked during a run yet |
| Watchdog | `orchard/watchdog.py`, `orchard/transcripts.py` | Built and tested. Fired on a real agent during the run. The near-duplicate command check (command shapes, 4 turns) was added after the second-model run and has not fired on a real agent yet |
| Stage machine, agent loop, supervisor | `orchard/stages.py`, `orchard/agent.py`, `orchard/context.py`, `orchard/supervisor.py` | Built and tested. Ran stages 0 to 2 on the real run |
| Crash recovery | `orchard/supervisor.py`, `orchard/handoff.py` | Tested with fakes by killing the supervisor after every ledger event. On hardware: `kill -9`, restart, resume worked at least four times on the real run |
| Stage skills | `orchard/skills/` | Drafts. `delta-triage`, `reference-gate` and `weights-swap-check` have been used on the real run. `weights-swap-configs` and `operator-bundle` have been used on the real run. `serving-check` has never run. `delta-triage`, `reference-gate` and `operator-bundle` were rewritten on 2026-10-04 to run template scripts (`delta-triage-templates/`, `reference-gate-templates/`, `operator-bundle-templates/`); the rewritten versions have not run on a real run |
| Packaging (stage 7) | `orchard/package.py`, `orchard/package_card.py`, `orchard/package_templates/` | Built and tested with fakes, then run once on hardware (Hemmingway-1; see stage 7 above). Wired into the stage table as opt-in supervisor code. A v5.1 container package is refused at start |
| Bundle and package scrub | `orchard/scrub.py` | Built and tested with fakes. The stage 8 gate calls it; it ran once on the real run |
| CPU sizing tool | `orchard/sizing.py` | Built and tested against a fake server. It has not been run against a real ollama. The CPU numbers in this README come from the run log |
| Hardware-check driver | `orchard/hardware_check.py` | Ran on both boards of the development machine, 22 to 24 checks passed per run |
| Park-check driver | `orchard/park_check.py` | Ran once, on one board, with fake model servers. Exit 0, two resets of 41.7 s each (measured) |

### Tests

The suite has 1600 passing tests and 1 skipped test (measured with
`python3 -m pytest -q -p no:cacheprovider`). It needs no hardware and no network. The skipped test
replays local agent transcripts and runs only when `ORCHARD_REPLAY=1` is set and those transcripts
exist.

## 2. What you need

Everything below was checked on the development machine only. Items marked
**not verified outside the development machine** have no evidence from any other host.

### Hardware

- Tenstorrent Blackhole boards. Development used a QuietBox 2 with two p300c boards, which is four
  chips. Other Blackhole systems are **not verified outside the development machine**.
- The chip counts the harness tests (1, 2 and 4) come from the packages that exist for the
  development machine. A machine with a different board count needs a different
  `--required-chips` value and packages for its own chip counts.

### Host software

| Software | What it is for |
|---|---|
| Linux | The only operating system used |
| Python 3.12 or newer and pytest | Runs tt-orchard and its tests. tt-orchard has no runtime pip dependencies |
| [tt-gozer](https://github.com/tsingletaryTT/tt-gozer) with the `reset` command (version 0.3.2 or newer) | Chip leases. The supervisor holds every board under a gozer lease |
| `tt-model` ([tt-model-manager](https://github.com/tenstorrent/tt-model-manager)) | Serves the models the agents use and the models under test |
| `tt-smi`, the Tenstorrent kernel driver and firmware | Device access. The supervisor reads `tt-smi -s` once at the start of a run to record firmware versions |
| docker | Runs container packages |
| ollama | Serves the CPU tier |
| git | Getting the code |
| Hugging Face access | Downloading the new model's weights before the run. The run itself works offline |

### Disk

All numbers are for a Qwen3.8-27B-sized model.

| Item | Size |
|---|---|
| The new model's weights | About 51 GB (measured, Hemmingway-1, 13 safetensors files) |
| One converted tensor cache, 2-chip bundle | 34 GB (measured) |
| One converted tensor cache, 4-chip container | 31 GB (measured) |
| One converted tensor cache, 1 chip | Not measured |
| Free space the supervisor requires before stages 2, 3, 5 and 6 | 40 GB each (choice) |
| Free space before stage 4 on the weights-only path | 110 GB (choice), plus 40 GB again before each configuration's test (choice) |
| Free space before stage 7 (packaging) | 80 GB (choice): an installed copy of the package, its venv and a fresh tensor cache |

Each chip configuration needs its own tensor cache. The supervisor checks free space only on the
filesystem that holds the run directory. On the development machine the root disk was 99% full, so
the run directory, the tensor caches and the model weights all lived on a second, larger disk. Put
them on a disk with room before you start.

### Memory

The CPU tier on the development machine is `qwen3-coder:30b` on ollama (a mixture-of-experts model,
about 3 billion active parameters, 4-bit). Measured with the chips idle: 14.1 tokens/s decode and
93 tokens/s prefill on a 1,605-token prompt. Its speed while a chip model is serving, and how
reliably it makes tool calls, are not measured. The host had 249 GB of RAM, and 133 GB was
available with that day's workload. The model needs about 18 GB at 4-bit (an estimate from its
size). A dense 27B model on the same CPU measured 0.70 tokens/s decode, which is too slow for agent
steps. These numbers are **not verified outside the development machine**.

## 3. Install

### 3.1 Get the code and run the tests

tt-orchard has no public remote yet. Get a copy from whoever shared it with you, then:

```bash
git clone <REPO_URL> tt-orchard
cd tt-orchard
python3 -m pytest -q
```

Expect every test to pass except 1 skipped. The suite took about 6 minutes on the development
machine (measured). The tests use fake hardware,
fake model servers and fake gozer state. They touch no device and no network. If a test fails
here, stop and find out why before going further.

### 3.2 Install tt-gozer

Follow the install section of the [tt-gozer README](https://github.com/tsingletaryTT/tt-gozer). Then
check that `gozer status` lists your boards and that `gozer reset --help` works.

### 3.3 Install tt-model and a bundle for the nearest supported model

Install `tt-model` by following the
[tt-model-manager README](https://github.com/tenstorrent/tt-model-manager). It is not on PyPI.

The harness does not write new model code on the weights-only path. It loads the new model's
weights into an existing tt-model package of a model with the same architecture (the "nearest
supported model"). So you need an installed bundle or container package for that architecture,
for each chip count you want to test. You also need one to serve the agents' own model (the
"coder").

Examples from the development machine. These are examples only; pick the packages that match your
model and your hardware:

- 2 chips: the bundle `episod/qwen3.8-27b-dflash2-p300`.
- 4 chips: the container package `changh95/qwen3.8-27b-p300x2` with profile `batch32`.

```bash
tt-model pull <PACKAGE_ID>
tt-model list
```

Serve each package once by hand under a gozer lease, and check that it answers sensibly, before a
run depends on it. See [lessons](#6-lessons-that-will-bite-you) for why.

### 3.4 Install ollama and pull the CPU-tier model

The CPU tier keeps a model answering while the chips are busy with a hardware test. Install ollama
from its own instructions, start it, and pull the model:

```bash
ollama pull qwen3-coder:30b
curl -s http://127.0.0.1:11434/v1/models
```

The second command must list `qwen3-coder:30b`. The supervisor never starts or stops ollama.

### 3.5 Download the new model's weights

Download the model before the run. The agents get no Hugging Face token and should run offline.

```bash
hf download <MODEL_ID> --cache-dir <HF_HOME_DIR>/hub
```

Note the snapshot directory it prints. You pass it to the run as `--input model=...`.

### 3.6 Write the tier config

```bash
cp config/tiers.example.toml config/tiers.toml
```

`config/tiers.toml` is ignored by git. Edit it. It has three parts.

**`[tiers.<name>]`**, one table per model. Each needs:

| Key | Meaning |
|---|---|
| `role` | A short description, for people |
| `endpoint` | The model server's OpenAI-compatible URL, for example `http://127.0.0.1:8000/v1` |
| `model` | The model id the server lists at `/v1/models` |
| `placement` | `chips` or `cpu` |
| `context_tokens` | Optional. The context length, a positive integer |

**`[stages.<n>]`**, one table for each stage 0 to 8. Each names tiers by name:

| Key | Meaning |
|---|---|
| `run` | The tier that does the stage's work. Stage 7 must say `"none"` |
| `diagnose` | The tier that takes over when the stage escalates. Required for stages 2, 3 and 4 |
| `plan` | The tier that plans. Required for stage 4 and refused on every other stage |

**`[escalation]`** with one key, `default`: the tier that takes over when a stage without a
`diagnose` tier (0, 1, 5, 6 and 8) escalates.

The loader refuses, with a message, any of the following:

- a value that still contains `CHANGE-ME`;
- an endpoint whose host is not `127.0.0.1`, `localhost` or `::1`, or whose scheme is not http or
  https;
- an unknown key or table anywhere (it suggests the closest known key);
- a missing stage, a stage that names an undefined tier, or a stage 7 with a model;
- a config with no `cpu` tier (the CPU tier stands in while a chip model is stopped);
- a tier named `none`.

The development machine used three tiers. Its tier tables were:

```toml
[tiers.large]
role = "plan and diagnose"
endpoint = "http://127.0.0.1:8000/v1"
model = "Qwen/Qwen3.8-27B"
placement = "chips"
context_tokens = 262144

[tiers.small]
role = "routine steps while one board is free"
endpoint = "http://127.0.0.1:8001/v1"
model = "Qwen/Qwen3.8-27B"
placement = "chips"
context_tokens = 262144

[tiers.cpu]
role = "stand-in while the large model is parked, and short steps"
endpoint = "http://127.0.0.1:11434/v1"
model = "qwen3-coder:30b"
placement = "cpu"
```

Its `[stages]` and `[escalation]` tables were the same as in `config/tiers.example.toml`. When a
stage's tier is not serving and another chip tier with the same model id is, that tier serves the
step and the ledger records the substitution. On that machine only one chip server ran at a time
(the coder on port 8000), so it served every step.

### 3.7 Machine paths

You do not edit any skill for your machine. Where a skill needs a path on the machine, it holds a
placeholder, and the supervisor fills it in before the agent reads the skill:

| Placeholder | Value | Override |
|---|---|---|
| `{{ORCHARD_DIR}}` | The tt-orchard checkout the supervisor runs from | none |
| `{{HF_HOME}}` | `$HF_HOME`, else `<operator home>/.cache/huggingface` | `--hf-home` |
| `{{OPERATOR_HOME}}` | Your home directory from the passwd entry | `--operator-home` |
| `{{TT_MODEL_ROOT}}` | `<operator home>/.cache/tt-model/models`, where tt-model installs bundles | follows `--operator-home` |
| `{{CACHE_ROOT}}` | `<parent of the run directory>/cache`. Each chip configuration's tensor cache goes in `<CACHE_ROOT>/<model slug>/<N>chip-<package>/tt_cache` | `--cache-root` |

Put the run directory on a disk with room for the tensor caches (see [Disk](#disk)), or pass
`--cache-root`. The ledger's `run_start` entry records all five values. A resumed run uses the
recorded values, and a resume that passes a different `--cache-root`, `--hf-home` or
`--operator-home` is refused. A skill that holds an unknown placeholder blocks its stage with a
message that names the placeholder and the file. The free-space check looks only at the run directory's disk. A
`--cache-root` on another disk is not checked, so check its free space yourself.

`orchard/hardware_check.py` has no built-in machine paths. Pass `--gozer` (a gozer that has
`reset`), and, unless you give `--child-cmd`, `--env-script` and `--python`. The environment
variables `ORCHARD_GOZER`, `ORCHARD_ENV_SCRIPT` and `ORCHARD_CHILD_PYTHON` work in their place.

## 4. Run a bring-up

### 4.1 Before you start

- `gozer status` shows every chip `FREE` and no other lease.
- ollama is serving the CPU tier (see 3.4).
- Nothing listens on the coder's port: `ss -ltn "( sport = :8000 )"` prints only its header.
- The run directory's disk has the free space listed in [Disk](#disk).
- Decide what to do about credential files (see [5.6](#56-the-credentials-preflight)).

### 4.2 The `run` command and its flags

```text
python3 -m orchard.supervisor run --model MODEL --run-dir RUN_DIR --tiers TIERS
    --coder-target CODER_TARGET [--coder-kind {container,bundle}] [--coder-profile CODER_PROFILE]
    --coder-port CODER_PORT --coder-chips CODER_CHIPS [--coder-image-id CODER_IMAGE_ID]
    [--skills-dir SKILLS_DIR] [--input NAME=PATH] [--env NAME=VALUE]
    [--required-chips N,N] [--cache-root DIR] [--hf-home DIR] [--operator-home DIR]
    [--package-format {v6,v5.1}] [--package-namespace NS] [--package-models-root DIR]
    [--gozer GOZER] [--accept-credentials-visible]
```

| Flag | Required | Meaning |
|---|---|---|
| `--model` | yes | The Hugging Face id of the model to bring up |
| `--run-dir` | yes | The run directory. A new directory starts a run. An existing one resumes that run |
| `--tiers` | yes | The tier config, normally `config/tiers.toml` |
| `--coder-target` | yes | The tt-model package or bundle that serves the agents' model on the chips |
| `--coder-kind` | no | `container` (default) for a v5.1 container package, or `bundle` for a v5 or v6 bundle |
| `--coder-profile` | no | The package's serve profile. Default `default` |
| `--coder-port` | yes | The port the coder serves on. Exactly one `chips` tier in the config must use this port, and the coder must list that tier's `model` at `/v1/models` |
| `--coder-chips` | yes | How many chips the coder uses. gozer leases whole boards of two chips |
| `--coder-image-id` | no | The container image id `tt-model list` prints. It helps the supervisor recognise the coder's container in `docker ps` when it confirms a stop |
| `--skills-dir` | no, repeatable | More skill directories, searched after `orchard/skills`. Stages on the full-port path use skills from the `tt-model-bringup` plugin (for example `functional-decoder`, `full-model`, `mesh-shrink`), so pass that plugin's `skills` directory. A stage whose skill cannot be found blocks |
| `--input` | no, repeatable | `NAME=PATH` facts every agent prompt lists, for example `model=<MODEL_SNAPSHOT_DIR>`. Anything you pass here, every agent sees |
| `--env` | no, repeatable | `NAME=VALUE` variables for agent shells, for example `HF_HOME` and `HF_HUB_OFFLINE=1`. Names that look like credentials are refused |
| `--required-chips` | no | The chip counts stage 4 must pass, such as `2,4`. Other counts are optional. Without it, every configuration stage 4 lists must pass. The ledger records it, and a resume with a different value is refused |
| `--cache-root` | no | Where the per-model tensor caches go (`{{CACHE_ROOT}}` in the skills). Default `<parent of --run-dir>/cache`. The ledger records it, and a resume with a different value is refused |
| `--hf-home` | no | Your Hugging Face cache (`{{HF_HOME}}`). Default `$HF_HOME`, else `<operator home>/.cache/huggingface`. Recorded and kept like `--cache-root` |
| `--operator-home` | no | Your home directory (`{{OPERATOR_HOME}}`), where tt-model keeps its packages. Default your home from the passwd entry. Recorded and kept like `--cache-root` |
| `--package-format` | no | `v6` makes stage 7 build a v6 thin package on the weights-only path (see [5.3](#53-what-each-stage-does)). `v5.1` is refused at start, because it needs a container image build. Without this flag stage 7 is skipped. The ledger records it, and a resume with different options is refused. A run that started without it can be given it on a resume, as long as stage 7 has not started |
| `--package-namespace` | with `--package-format` | Your Hugging Face namespace. Stage 7 writes it into the package card and the publish commands. The run never publishes |
| `--package-models-root` | no | Where tt-model installs bundles. Stage 7 looks here for other chip counts of the nearest model. Default `{{TT_MODEL_ROOT}}`, which is `<operator home>/.cache/tt-model/models` |
| `--gozer` | no | The gozer executable. Default `gozer` from `PATH` |
| `--accept-credentials-visible` | no | Start even though credential files exist in your home directory. **Warning:** agent shells run as your user, so code an agent runs can read those files. The ledger records that you accepted this |

Stage 7 runs only when `--package-format v6` and `--package-namespace` are given and stage 0 chose
the weights-only path. Otherwise it is recorded as skipped.

Exit codes: 0 ready for operator review, 2 refused (nothing was started), 3 stopped on an error
(the hardware was released; run the same command again to resume), 4 aborted.

### 4.3 An example

```bash
python3 -m orchard.supervisor run \
  --model <MODEL_ID> \
  --run-dir <RUN_DIR> \
  --tiers config/tiers.toml \
  --coder-target <CODER_PACKAGE> --coder-kind bundle \
  --coder-port 8000 --coder-chips 2 \
  --required-chips 2,4 \
  --skills-dir <TT_MODEL_BRINGUP_SKILLS_DIR> \
  --input model=<MODEL_SNAPSHOT_DIR> \
  --env HF_HOME=<HF_HOME_DIR> --env HF_HUB_OFFLINE=1
```

Leave the run in a terminal you can come back to, or under `nohup`. It runs for hours. On the
development machine stages 0 to 2 took about 7 hours of wall time including the pauses for fixes.
A clean run's total time is not measured.

Do not pass the nearest supported model as an `--input` if you want to check that stage 0 finds it
on its own.

### 4.4 What the supervisor does

1. Records the run in the ledger, with the versions it can read (`tt-model --version` and the
   firmware from `tt-smi -s`).
2. Takes a gozer lease under its own process id and starts the coder.
3. On the coder's first start, asks "What is 7 times 6?" and blocks the run unless the answer
   contains 42. This catches a server that boots but returns noise.
4. Runs each stage: a fresh, short agent context, the stage's skill, the agent's commands through
   the command runner, and the stage's exit gate. A hardware stage has a prepare step that writes
   the test command, the hardware test itself (run by the supervisor under a lease), and a finish
   step that writes the result.
5. When a test needs the coder's board and no other board is free, it parks the coder: the CPU
   tier answers a canary question, the coder stops, the board is reset in place, the test runs,
   the board is reset again, the coder restarts and must give the same canary answer as before.
6. An agent step that uses up its turns with evidence files on disk and its output file not
   written gets one wrap-up: the same conversation is told which file is missing and which
   evidence exists, and has 12 turns to write it with no new investigation.
7. A failed stage is escalated once to its `diagnose` tier (or the `[escalation]` default). A
   second failure pauses the run for you.
8. At the end it stops the coder, releases every lease, and prints `ready for operator review`.

### 4.5 The run directory

| Path | What it holds |
|---|---|
| `ledger.jsonl` | The append-only record of the run. The state of the run is computed from it, and no other state file exists |
| `ledger.jsonl.lock` | The single-writer lock. A second supervisor on the same run is refused |
| `supervisor.pid` | The pid of the supervisor that last started on this run. `status` reads it. Runs from before this file existed do not have one |
| `*.torn-<time>` | A partial last line cut off by a crash, kept beside the ledger |
| `control`, `control.done-<k>` | The operator command file and the commands already acted on |
| `coder.log` | The coder server's output |
| `evidence/` | Evidence the supervisor writes, for example the coder's first-boot answer |
| `handoff/` | Evidence from parks and restores |
| `home/` | The agent shells' home directory |
| `stages/<n>/` | One directory per stage: the agent's result file (`delta.json`, `reference.json`, `result.json`), `evidence/`, `log/` (each agent step's transcript), and for hardware stages `hw_test.json`, `handoff.json` and `test-result.json` |
| `stages/<n>.partial-<k>/` | An earlier attempt at the stage, moved aside. Nothing is deleted |
| `stages/4/configs/<N>/`, `stages/4/tests/<N>/` | Stage 4 on the weights-only path: each chip configuration's files and the supervisor's record of its test |
| `stages/8/bundle/` | The operator bundle: `RESULTS.md`, `RISKS.md`, `card.md`, `PUBLISH_COMMANDS.txt`, `package/` (stage 7's packages and cards) and a copy of the ledger. `stages/8/evidence/bundle-build.json` lists each bundle file with its sha256 |

## 5. Operate a run

### 5.0 Operating a run

To see where a run stands, run this from the checkout. It reads the run directory and changes
nothing. It takes no lock, so it is safe while the supervisor runs.

```bash
python3 -m orchard.supervisor status --run-dir <RUN_DIR>
python3 -m orchard.supervisor status --run-dir <RUN_DIR> --json
```

It prints about 30 lines: the state of the run (`running`, `paused`, `ready-for-operator-review`,
`aborted`, `stopped-or-crashed` or `not-started`) and how that was decided, each stage's status and
wall time, counts of retries, escalations, nudges, pauses and operator commands, the reason for a
pause, the last five ledger events, free disk, the chip leases, and a final `next:` line with the
usual action for that situation. It exits 0 when it produced a status, and 2 for a bad run
directory or a ledger that fails its hash check. The JSON keys are listed in the docstring of
[`orchard/status.py`](orchard/status.py).

A small local model can act as the operator. [`orchard/skills/operator-runbook.md`](orchard/skills/operator-runbook.md)
tells it to run `status`, pick one action from a table, log it, wait five minutes and repeat. To use
it with qwen-code, add one line to the `QWEN.md` or `AGENTS.md` in the directory where the operator
model starts: "Read `<ORCHARD_DIR>/orchard/skills/operator-runbook.md` before you touch a run." The
runbook never publishes, never aborts and never changes the run script. When the run is ready,
`python3 -m orchard.operator_checks --run-dir <RUN_DIR>` runs the read-only post-run checks (publish-like
tool calls, repos that already exist on the hub, secrets and home paths in the bundle).

### 5.1 Pause, resume, abort

From another shell, in the tt-orchard checkout:

```bash
python3 -m orchard.supervisor control --run-dir <RUN_DIR> pause
python3 -m orchard.supervisor control --run-dir <RUN_DIR> resume
python3 -m orchard.supervisor control --run-dir <RUN_DIR> abort
```

- `pause` stops at the next check. The supervisor keeps its leases and the coder, and waits.
- `resume` continues a paused run. Read the last `notice` and `decision` entries in the ledger
  first, so you know why it paused.
- `abort` stops the coder, releases every lease, and ends the run. **An aborted run cannot be
  resumed.** Running the same command again on that run directory only finishes the release.
- Ctrl-C (SIGINT) and `kill <pid>` (SIGTERM) take the abort path too. The shutdown can take a few
  minutes, and a second Ctrl-C during it is ignored.

### 5.2 Read the ledger

Each line is one JSON object with `seq`, `ts` (UTC), `prev` (the hash of the previous line),
`event`, `stage` and `data`. Events include `run_start`, `stage_start`, `stage_end`, `park`,
`restore`, `retry`, `escalate`, `notice`, `measurement`, `decision` and `evidence`.

```python
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        e = json.loads(line)
        print(e["seq"], e["ts"], e["event"], e["stage"], json.dumps(e["data"])[:160])
```

Save it as `ledger_view.py` and run `python3 ledger_view.py <RUN_DIR>/ledger.jsonl | tail -n 20`.
A pause is a `decision` entry with `"decision": "pause"` and a `reason`. A block is a `notice` with
`"blocked": true`. Do not edit the ledger. Each line carries the previous line's hash, and the
supervisor refuses a ledger whose chain is broken.

### 5.3 What each stage does

| # | Stage | Skill | What it does |
|---|---|---|---|
| 0 | Delta triage | `delta-triage` | The agent writes `triage_config.json` and runs `delta_triage.py`, which compares configs, safetensors headers, tokenizers (with an encode test of 206 strings), files, generation config, license and free disk, and drafts `delta.json` with every difference and the path: `weights-only` or `full-port`. The agent reviews the findings and adds hazards |
| 1 | CPU reference | `reference-gate` | The agent writes `reference_config.json` and runs `reference_gate.py`, which loads the model on CPU in bf16, reports missing and unexpected keys, round-trips 14 strings, decodes 32 greedy tokens with the first-token and append-at-end checks, and drafts `reference.json`. The card check is a form check only. The agent reviews the notes |
| 2 | Weights swap check (weights-only) or functional decoder (full-port) | `weights-swap-check` or `functional-decoder` | Serves the new weights on one board through the existing implementation and compares the chip's tokens with the CPU reference |
| 3 | Full model | `full-model` | Full-port path only. Skipped on the weights-only path |
| 4 | Each chip configuration (weights-only) or multichip and shrink (full-port) | `weights-swap-configs` or `mesh-shrink` | Runs one serve-and-compare test per chip configuration, each under its own lease |
| 5 | Serving integration | `serving-check` | Full-port path only. Checks a served model from outside: boot, a passkey test at two lengths, a repeated canary. Skipped on the weights-only path |
| 6 | Qualitative check and benchmark | `serving-check` | Full-port path only. Five prompts read by the agent, and decode speed and time to first token, each labelled measured or TODO. Skipped on the weights-only path |
| 7 | Package and container build | none (supervisor code) | Weights-only path with `--package-format v6` only. First checks that stage 2 served the weights stage 0 names: the label must name stage 0's repo, and the weight files in the served `model-dir` must link to stage 0's revision in the Hugging Face cache. Then runs `tt-model package-thin --out` from the nearest model's installed v6 bundle, points every weights setting in `run.sh` at a `model-dir` built from the new weights, scrubs each package, installs a copy of the required profile and boots it on a leased board against the stage 1 reference, then writes a card and the publish commands as text. A failure pauses the run and is not escalated. Every other run records it as skipped |
| 8 | Operator bundle | `operator-bundle` | The agent writes `bundle_config.json` and runs `build_bundle.py`, which builds the bundle from the ledger and each stage's files: results with labelled numbers and evidence paths, risks with a computed `Dealt with:` line for the tensor-cache hazard, the model card, stage 7's packages and publish commands, and a copy of the ledger. It prints the scrub's findings. The agent rewrites the summary paragraph and adds risks it can back with evidence. The supervisor copies the ledger in again and scrubs the bundle |

### 5.4 Recover after a crash

If the supervisor dies without running its handlers (SIGKILL, a reboot, a Python crash), the coder
keeps its chips under a lease whose owner is dead. To recover, run the same `run` command again
with the same flags. You can leave out `--required-chips` (the recorded value is kept). The
supervisor replays the ledger, stops the old coder, takes a new lease, starts the coder again and
goes on. If the run was paused, send `resume`.

`kill -9`, then restart with the same command, then `resume` is the tested path. The run log
records it working four times on the development machine. The 2-chip coder came back in 110 to 130
seconds (measured, warm) and gave the same canary answer each time. After a restart the coder can land on the other board if the
old lease has not been cleared yet; that is expected.

`kill -9` is also how to load new code into a paused run. `abort`, Ctrl-C and `kill` all end the
run for good.

A stage that was interrupted restarts from its beginning, or from its resume marker if it has one.
Its old directory is moved aside to `stages/<n>.partial-<k>`.

### 5.5 Exit 3 and blocks

Exit 3 (`error:`) means the supervisor stopped on an error it did not expect. It stops the coder and
releases the leases first. Fix the cause and run the same command to resume. A block (`blocked` in a
`notice`) pauses the run and keeps the hardware. Read the reason, fix the cause, then `resume`.
If the supervisor says releasing the hardware failed, read the last `notice` and `gozer status`
before doing anything. Never run `tt-smi -r` by hand and never pass `--force` to gozer.

### 5.6 The credentials preflight

Agent shells run as your user. The supervisor gives them an environment with no tokens and a home
directory inside the run, but code an agent runs can still read any file your user can read. So
before it starts, the supervisor looks for these files in your home directory:
`.cache/huggingface/token`, `.config/gh/hosts.yml`, `.netrc`, `.docker/config.json`, and SSH
private keys (`.ssh/id_*` without `.pub`). It checks only that they exist. If any exist, it
refuses to start (exit 2) and names them.

You can move them aside for the run and put them back afterwards. Or you can pass
`--accept-credentials-visible`, and the ledger records that you accepted the risk. The check does
not look inside `HF_HOME` or other places, so check those yourself.

### 5.7 What the command runner does not stop

Every agent command goes through `orchard/runner.py`. It reads a small subset of shell and refuses
the rest. It refuses publish, push and upload commands, chip resets, gozer commands that take or
release leases, docker and tt-model commands that start or stop servers, `kill`, `ssh`, curl and
wget requests that send data, changes to the device mask, and deletion outside the run directory.

It is a best-effort guard. It does **not** stop:

- `python3 -c`, `perl -e`, or any script, which can delete files, open a device, upload or signal a
  process;
- `mv`, `cp`, `tee`, `truncate` or `rsync --delete`, which can overwrite any file your user can
  write, the ledger included;
- a direct open of `/dev/tenstorrent/*`. Agent shells get a device mask that matches no chip, but
  that is a request to the runtime, and code can clear it;
- wrappers it does not list, git aliases or hooks in config files, or a denied command run from
  inside a script.

No adversarial search for ways around the runner has been done. The real limits are the user
account the run uses, the lease tool, and not giving the agents credentials.

### 5.8 Review the bundle

When the run prints `ready for operator review`, read `stages/8/bundle/`:

1. `RISKS.md` first: every `TODO` number (including what the package cards list under Not
   measured), every stage 0 hazard with a `Dealt with:` line, the numbers that rest on thin
   evidence, and the license. `Dealt with: Not shown` means the run's files cannot show it.
2. `RESULTS.md`: each number has a label (`measured` or `TODO`) and an evidence path. Open the
   evidence for the numbers you rely on. A gate checks that the files exist and have the right
   shape. It cannot tell whether a claim is true.
3. The agent transcripts under `stages/<n>/log/`. On the first run an agent wrote that its findings
   matched a reference file it had never opened. Check that each claim of a check matches a
   command that ran.
4. `PUBLISH_COMMANDS.txt`: the commands you would run to package and publish, as text. The harness
   never runs them. You decide whether to run them, whether the result is private or public, and
   whether it is listed anywhere.
5. `package/`, when stage 7 staged a package: a copy of each package folder, its record
   (`package.json`), its publish commands and each package's card (`<name>-README.md`). Read each
   card before you run a publish command. The gate checks the card's license and that each number
   names its evidence. It cannot tell whether the prose is true. A publish line for a package
   whose boot check did not run is commented out. `card.md` collects the package cards in one
   file.

`bundle/ledger.jsonl` holds absolute paths from your machine. It is your record and is not for
publishing. The supervisor scrubs the other bundle files for the hostname, tokens and home paths,
and a hit blocks the bundle. The scrub does not find a hostname written another way, a token
format it does not know, or anything inside a binary file.

## 6. Lessons that will bite you

Each of these happened on the development machine. Details are in the
[run log](docs/run-logs/2026-10-hemmingway-1.md).

- **A server boots and returns gibberish.** Symptom: the first-boot check fails, and the answers
  are repeated nonsense tokens. Cause seen: a stale tensor cache built from an older snapshot of
  the weights. Fix: stop the server, move the package's tensor cache directory aside (do not delete
  it until you are sure), and boot again so it rebuilds. Converting weights into an empty cache
  took about 5 minutes on 2 chips (measured, with a warm kernel compile cache).
- **Never share a tensor cache between models or chip configurations.** The cache is keyed only by
  layer name and mesh, so another model's cache serves that model's weights with no error. The
  templates refuse a cache that is not empty and lacks a marker naming the model, and the
  supervisor refuses two configurations that share one.
- **A weights swap must set both `MODEL_WEIGHTS_DIR` and `HF_MODEL`.** The TT runtime takes its
  weights directory from `MODEL_WEIGHTS_DIR`, then `HF_MODEL`, then the path given to `--model`.
  With only `--model` changed, the chip loaded the base model's weights and still agreed with the
  new model's CPU reference on 25 of 32 tokens. With both set, it agreed on 30 of 32.
- **A cold first boot takes about 30 minutes.** Most of it is kernel compilation. A warm restart
  takes about 2 minutes. The harness's hardware tests run with the run's own home directory, so
  their compile cache starts cold. A 1,500-second health timeout was too short; the templates now
  wait 3,300 seconds.
- **Agents run with thinking off.** In thinking mode the Qwen3.8 model spent its whole token
  budget reasoning and never issued a command, with greedy decoding and with sampling. With
  thinking off, the same turn produced a correct command in a few hundred tokens. This was checked
  on one replayed turn, so it is an indication. `AGENT_THINKING` in `orchard/defaults.py` controls
  it.
- **Leases are per board.** gozer leases whole boards of two chips on a p300c, because the runtime
  expands `TT_VISIBLE_DEVICES` to the whole board. Asking for 1 chip gets 2. A 1-chip profile on a
  board lease can fail to start, because tt-model wants the device count to match the profile.
- **`gozer status` can be unreadable for about a minute while chips reset.** A reset opens every
  device on the box for about 42 seconds (measured), so the other board shows `BUSY-UNTRACKED`
  meanwhile. Wait, then read it again. Do not start another reset during that time.
- **One lease holder per board.** Only the supervisor should hold the boards it uses. Do not serve
  another model on those chips by hand during a run. If you need the chips, pause or abort the run
  first.
- **A stage skill that describes a script is too much for a local model.** Asked to write a
  400-line script, the agent explored for 60 turns and wrote nothing. Given tested template scripts
  and one small config file, the same step took 3.5 minutes. Write skills as close to scripts as
  you can.

## 7. Where things are

| Path | What |
|---|---|
| [`orchard/`](orchard/) | The supervisor package. Each module's docstring says what it owns and what it does not do |
| [`orchard/defaults.py`](orchard/defaults.py) | Every timing, budget and threshold, each labelled measured or choice |
| [`orchard/skills/`](orchard/skills/) | The stage skills and the template scripts they copy (delta triage, reference gate, weights swap, operator bundle). These skills live in this repository. It also holds [`operator-runbook.md`](orchard/skills/operator-runbook.md), which is for whoever watches a run and is not a stage skill |
| [`orchard/status.py`](orchard/status.py), [`orchard/operator_checks.py`](orchard/operator_checks.py) | The read-only `status` command and the post-run checks |
| [`orchard/package_templates/`](orchard/package_templates/) | The script each stage 7 package carries (`prepare_model_dir.py`) and stage 7's boot check (`verify_bundle.py`) |
| [`config/tiers.example.toml`](config/tiers.example.toml) | The example tier config |
| [`tests/`](tests/) | The test suite and its fakes |
| [`docs/superpowers/specs/2026-10-01-orchard-design.md`](docs/superpowers/specs/2026-10-01-orchard-design.md) | The design spec |
| [`docs/superpowers/plans/`](docs/superpowers/plans/) | The implementation plans |
| [`docs/runbooks/hardware-validation.md`](docs/runbooks/hardware-validation.md) | Runbook for the hardware checks, the park check, the first run and stage 4 |
| [`docs/runbooks/driver-build-report.md`](docs/runbooks/driver-build-report.md), [`docs/runbooks/driver-review-findings.md`](docs/runbooks/driver-review-findings.md) | How the hardware-check driver was built and what its safety review found |
| [`docs/run-logs/2026-10-hemmingway-1.md`](docs/run-logs/2026-10-hemmingway-1.md) | The log of the first real run |
| [`CLAUDE.md`](CLAUDE.md) | The project log: decisions and notable moments |

### Add or change a stage skill

A skill is a Markdown file found by name: `<dir>/<name>.md` or `<dir>/<name>/SKILL.md`. The
supervisor searches `orchard/skills/` first, then each `--skills-dir` in order. The stage table in
`orchard/stages.py` names the skill each stage uses. To change what a stage asks of the agent, edit
its skill. To change what passes, edit the stage's gate function in `orchard/stages.py` and its
tests.

Keep skills short and concrete. Give the agent tested scripts to copy and a small config file to
fill in. A skill longer than 40,000 characters is cut (`SKILL_CHARS`). A running supervisor keeps
the code and skills it loaded at start; to load a change into a paused run, use `kill -9` and
restart (see [5.4](#54-recover-after-a-crash)).

### Run the tests

```bash
python3 -m pytest -q
```

Run one file with `python3 -m pytest -q tests/test_stages.py`. New guards in this repository are
checked by mutation: remove the guard, watch a test fail, then restore it.

## 8. Safety and license notes

- The harness never pushes, uploads or publishes. A run ends at "ready for operator review". The
  packaging code runs `tt-model package-thin` only with `--out` and writes publish commands as
  text. Agents may not run `tt-model package` or `package-thin` at all, because given a repo id
  both upload.
- Every model endpoint must be on the same machine. The tier loader refuses any other host.
- The license of a packaged model comes from the new model's own Hugging Face card and is carried
  into the package card. A license the code cannot name counts as non-commercial. A fine-tune can
  carry a different license from its base model: Hemmingway-1 is CC BY-NC 4.0 and its base is
  Apache 2.0.
- The command runner is a best-effort guard (see [5.7](#57-what-the-command-runner-does-not-stop)).
  No adversarial search for ways around it has been done.
- gozer cannot see device handles held by another user's processes, such as a root-owned
  container. The supervisor confirms with docker, ps and the server's port that a server has
  stopped before it resets a board.
