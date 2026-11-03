# Tiny Qwen4 rehearsal through `tt-orchard bringup`, first unattended run (2026-11-03)

Command: `tt-orchard bringup local/tiny-qwen4-dense --full-port --full-port-rehearsal --no-fetch --offline --no-github`, with the
Qwen3-Coder-Next coder on board 0 (port 8011) and board 1 free for hardware tests. The model is the random-weight checkpoint from
`scripts/tiny_qwen4/make_checkpoint.py` (4 layers, one PLE layer, one dense FFN). The base is `Qwen/Qwen3.8-27B`.

## Result

| Stage | Result | Wall time | Attempts |
|---|---|---|---|
| 0 intake and delta triage | pass | 76 s | 1 |
| 1 environment and CPU reference | pass | 26 s | 1 |
| 2 functional decoder on one chip | fail (blocked, `stage-failed`) | 478 s | 4 |

The coder boot took about 12 minutes on a cold start. The run never reached stage 3.

## What stage 2 did

1. First attempt: the agent used 20 turns reading examples and wrote no file. The watchdog blocked it (`agent-stuck`) and the supervisor retried the run once.
2. After the retry the agent wrote `port_model.py`. The first hardware test ran on board 1 and took 9 seconds. It failed at `load`:
   `KeyError: 'model.layers.2.ple.conv1d.weight'`.
3. The config says `ple_layer_ids: [2]`. The reference class in `transformers` (`qwen4_exp`) documents these as one-indexed ids, so id 2 is layer index 1, and the checkpoint has its PLE tensors on layer index 1. The checkpoint is consistent. The agent's port read the ids as zero-indexed.
4. The agent's own diagnosis after the failure named the wrong cause ("the port assumes every id in PLE_LAYER_IDS has PLE components"). The supervisor gave no repair round. `orchard/supervisor.py` (`_test_failure`) ends the stage after a failed hardware test, on purpose, and the next attempt starts with a fresh prepare step. Three earlier attempts are in `stages/2.partial-N`.

## What this shows

- Stages 0 and 1, the coder boot, the lease handling, the park-free hardware test on the free board, and the clean stop all worked unattended.
- A port that fails in 9 seconds on a mistake the failure text points at costs a whole attempt. A human porter would fix the line and rerun the check.
- A first-run misread of an index convention is the kind of error a rehearsal is meant to find, and it was found on the chip path.

## Things that went wrong around the run (all mine, none in the shipped checkpoint)

- The first launch used a scratch `HF_HOME`. The supervisor mounts that directory into the coder container, so the coder downloaded its 80 GB base model into it and filled the root disk to 0 bytes. Issue 417.
- A coder container killed mid-boot stays on disk as `Exited (137)` under the same name. `tt-model serve` then refuses to start (`chip(s) 0, 1 already in use by another container`) until the container is removed. Issue 417.
