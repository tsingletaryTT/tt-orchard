# Qwen4-27B release day

The plan is `docs/superpowers/plans/2026-11-09-qwen4-27b-drop-in.md`. On release day nobody has to start anything: a scheduled
watcher sees the release, checks it, tells you, and starts the bring-up when every chip is free. This page says how to arm the
watcher and what to do with each message it sends.

## 1. Arm the watcher (once)

```bash
tt-orchard drop-watch init --watch-dir ~/orchard-runs/qwen4-watch --target Qwen/Qwen4-27B \
  --run-dir ~/orchard-runs/qwen--qwen4-27b
```

Add `--bundle-dir <tt-model models directory>/mando2222/qwen4-27b-dense-v6-thin` once the Qwen4 package passes milestone M5: the
watcher then reads the package's `orchard-capabilities.json` and the config of its own weights, as stage 0 does. Until then
every Qwen4 release is "not covered" and takes the full-port fallback. Add `--bringup-arg --accept-credentials-visible` only if you accept that the
agents can see this home directory's credential files (README section 5.6); without it the run blocks at the preflight and the
watcher reports that.

A scheduler runs one tick every 30 minutes:

```bash
tt-orchard drop-watch tick --watch-dir ~/orchard-runs/qwen4-watch --json
```

The scheduled Claude task wraps this command and sends you a push notification for every event the tick prints. Check by hand
with `tt-orchard drop-watch show --watch-dir ~/orchard-runs/qwen4-watch`; the full history is in `events.jsonl` there.

To rehearse without a release, lay a release out in a directory (`info.json` with `sha`, `lastModified` and `siblings`, plus
`config.json` and the weight index) and run `tick --hub-dir <dir>` against a second watch directory.

## 2. The messages and what to do

| Event | What it means | What you do |
|---|---|---|
| `other-release` | A `Qwen/Qwen4-27B*` repository other than the target appeared (FP8, Base). | Nothing; it is not run. |
| `hub-unreachable` | The Hub did not answer. Sent once per outage; the state does not change and the next tick tries again. | Nothing, unless it lasts; then check the network. |
| `release-seen` | The target exists. The watcher waits until config, index and every shard are uploaded and the last commit is 30 minutes old. | Nothing. |
| `check` | `drop-check` ran. The text gives the verdict, the route and every reason. | Read it. A not-covered route starts the full-port fallback when the chips are free. |
| `waiting-for-chips` | gozer shows a chip held. The watcher never stops a holder. | Free the chips, for example stop the Qwen3.8 server through its own supervisor (it releases its lease). The next tick starts the run. |
| `run-started` | `tt-orchard bringup` started once, with the command shown. | Follow it with `tt-orchard status` or `tt-orchard watch`. |
| `stage` | A stage finished (pass, fail, escalated, skipped). | Nothing. |
| `run-ready` | The run is ready for operator review. | Section 4. |
| `run-blocked` | The run blocked; the text has the code and reason. | Read `BLOCKED.md` in the run directory. Running the same `bringup` again resumes. |
| `run-failed` | The run aborted or crashed, or bringup exited before the run finished. | Read the bringup log named in the event. |
| `needs-operator` | The check needs a reference interpreter, the Hub record cannot be read, or the watcher cannot trust its own state (a state file it cannot read or that lacks what its state needs is moved aside as `state.json.corrupt-<time>`). | Section 3. |

## 3. When the watcher needs you

- **A new model type** (the release's `model_type` is not one transformers 5.19.0 has, for example `qwen4`): build a second
  reference interpreter beside the current one and point bringup at it. The next tick checks again by itself.

  ```bash
  python3 scripts/reference_venv.py 'transformers==<version with the class>' --arch <ArchitectureClass>
  ```

  Then set the `reference_python` line it prints in `config/bringup.toml`.
- **A corrupt state file**: it was moved aside as `state.json.corrupt-<time>`. Read it, then write the state you want or remove
  `state.json` (the watcher starts again from `waiting`).
- **"A run may have started"**: a tick died after it recorded "starting" and before it recorded the process. Look for a
  `tt-orchard bringup` process and the run directory. If a run is going, set `"state": "running"` and the process id in
  `state.json`; if not, remove `state.json`.

## 4. Host memory and disk

Stage 1 runs the 27B model on the CPU in bf16 (about 54 GB of host memory). Check `free -g` on release day and free memory if
less than about 70 GB is available. Stage 4 needs about 110 GB on the run directory's disk.

## 5. After the run: review, push, publish

The run ends at the review bundle (`stages/8/bundle`). Nothing is pushed or published by the harness. After you review it:

1. Push stage 7's package to the private repository with tt-model (the tt-model-manager thin-package recipe), then pull it and
   serve the pulled copy once.
2. Publish to the catalog only on your yes.

## 6. The full-port fallback and the outside helper

A release the Qwen4 package does not cover starts `bringup ... --full-port`. The coder alone has not yet written a passing port;
the passing full-port runs had the outside helper on. The helper is off on this machine. To turn it on, add this table to
`config/bringup.toml` with your endpoint, models and key (README section 5.9g):

```toml
[telegraph]
enabled = true
endpoint = "https://<your endpoint>/v1"
models = ["<cheaper model>", "<stronger model>"]     # cheapest first
key_env = "<the variable that holds the key>"
max_calls = 40
max_calls_per_problem = 3
```

Rules for when the supervisor may ask the helper (draft, for the operator to approve; the code already enforces the counts in
the first three lines):

- Never on a stage's first attempt. One draft ask per attempt, at most two per stage.
- Repair rounds: at most six asks per stage; when the helper writes the port (`--helper-drives-port`), at most ten rounds and
  sixteen asks per stage.
- Every ask and its cost is in the run's ledger and the Telegraph view; an unhelpful answer moves the next ask up the model list.
- What is sent: the problem, the failure text and code excerpts. Never weights, tokens, home paths or host names (the request is
  redacted before it is sent).
- No spend cap for the Qwen4 bring-up (operator, 2026-11-08); `max_calls` stops a loop.
