# Benchmarks: iapp/openthai2.0-qwen3.8-27b, tt-model v6 packages (p300 and p150)

Measured 2026-10-04. Every number below was measured in this run unless the label says otherwise.
Paths are relative to `/mnt/bonus/models/orchard-runs/openthai2-run2/publish/`. Nothing was published, pushed or uploaded.

## Setup

* p300 package: the stage 7 verification install, `stages/7/verify/bundle` (own venv), booted by `raw/start_server.py`, which reproduces `verify_bundle.py`: `bash run.sh --port 51105`, environment scrubbed of `MODEL_WEIGHTS_DIR HF_MODEL TT_CACHE_PATH TT_CACHE_HOME HF_HUB_CACHE`, then `HF_HOME=stages/7/verify/hf`, `HF_HUB_OFFLINE=1`, and `TT_CACHE_PATH=TT_CACHE_HOME=/mnt/bonus/models/orchard-runs/cache/openthai2/2chip-qwen3.8-27b-dflash2-p300/tt_cache` (marker file names this model and revision `a64f2b12...`). run.sh sets `MODEL_WEIGHTS_DIR` and `HF_MODEL` to the bundle's `model-dir`, which links the openthai2 weights. The server reports the served model as that `model-dir` path. Lease: gozer, chips 2,3 (`claude:publish-bench-thai`), released after the run.
* Server settings come from the package's run.sh: max_model_len 262144, max_num_seqs 4, DFlash2 drafter, K=7 draft tokens per iteration.
* Requests are greedy (`temperature 0`, no logprobs). Chat requests use `enable_thinking: false`.
* Speed definitions (same as the earlier qwen3.8-27b-dflash2-2chip-p300c card): per-user tokens/s = 1000 / mean TPOT; TPOT = (end - first token) / (completion tokens - 1); TTFT = time to first streamed content; aggregate tokens/s = total output tokens / wall time of the whole run.
* Coding workload: the first 16 of the 80 `coding` prompts of nvidia/SPEED-Bench (found in the local HF cache, `~/.cache/huggingface/hub/datasets--nvidia--SPEED-Bench`, evaluation licence). Output cap 512 tokens (the earlier card used 2048), so 3 of 16 answers hit the cap. The prompts are not copied into this folder; the result files hold only prompt ids and timings. This is a smaller sample than the earlier card, so do not compare it to that card's numbers as like for like.

## Results

| # | What | Value | Label | Command / script | Evidence |
|---|------|-------|-------|------------------|----------|
| 1 | p300 launch to `/health` 200, drafter and kernel caches as found | about 33.9 min (launch 15:18:40, healthy 15:52:33) | measured | `raw/start_server.py`, `raw/wait_ready.py 51105 raw/p300.pid` | `raw/p300_server.log`, `raw/p300_start_epoch.txt`, `raw/p300_ready.txt` (its 510 s is only the last of three polls) |
| 2 | p300 first answer, "What is 7 times 6?" | "...$$7 \times 6 = 42$$ **Answer:** 42" (34 tokens) | measured, pass | curl to `/v1/chat/completions` | `raw/first_answer_p300.json` |
| 3 | p300 coding, 1 user: per-user speed | 80.5 tok/s (mean TPOT 12.42 ms, median 11.76 ms) | measured | `raw/coding_bench.py 51105 1 ... 512 16` | `raw/coding_u1.json`, `raw/coding_u1.log` |
| 4 | p300 coding, 1 user: TTFT | mean 952 ms (includes prompt processing; prompts differ in length) | measured | same | same |
| 5 | p300 coding, 1 user: aggregate | 65.3 tok/s, 5516 output tokens in 84.5 s (includes prefill and gaps) | measured | same | same |
| 6 | p300 coding, 4 users: per-user speed | 54.2 tok/s (mean TPOT 18.46 ms, median 18.31 ms) | measured | `raw/coding_bench.py 51105 4 ... 512 16` | `raw/coding_u4.json`, `raw/coding_u4.log` |
| 7 | p300 coding, 4 users: TTFT | mean 676 ms | measured | same | same |
| 8 | p300 coding, 4 users: aggregate | 179.3 tok/s, 5516 tokens in 30.8 s | measured | same | same |
| 9 | Length sweep, 1 user, OSL 128, ISL 1026 | TTFT 0.527 s, TPOT 30.19 ms (33.1 tok/s). First-ever 1K request: TTFT 4.576 s (first use of that prompt shape) | measured (second pass; first pass in `sweep.json`) | `raw/sweep.py 51105 raw/sweep_repeat.json 1024 4096 16384` | `raw/sweep_repeat.json`, `raw/sweep.json` |
| 10 | Same, ISL 4079 | TTFT 0.976 s, TPOT 29.51 ms (33.9 tok/s); first pass 0.984 s / 30.08 ms | measured | same | same |
| 11 | Same, ISL 16365 | TTFT 3.859 s, TPOT 26.56 ms (37.7 tok/s); first pass 3.805 s / 25.95 ms | measured | same | same |
| 12 | DFlash2 acceptance, 10 Thai prompts (1 user, 1504 tokens) | 1.03 of 7 draft tokens accepted per iteration (iteration-weighted), 2.01 tokens committed per iteration | measured | `raw/thai_run.py`, then `raw/accept.py raw/p300_server.log` | `raw/acceptance.json`, lines `[dflash2-serve] slot 0 session` in `raw/p300_server.log` |
| 13 | Same, per Thai prompt (accept /7): T1 0.36, T2 1.27, T3 2.86, T4 0.04, T5 0.44, T6 2.94, T7 6.00, T8 1.28, T9 3.00, T10 0.46 | range 0.04 to 6.00 | measured | same | same |
| 14 | DFlash2 acceptance, 16 coding prompts (1 user, 5507 committed tokens) | 5.57 of 7 per iteration, 6.55 tokens committed per iteration | measured | `raw/accept.py` | `raw/acceptance.json` |
| 15 | Same, 4 users | 5.57 of 7, 6.55 per iteration (identical to row 14: greedy decoding gave the same tokens, and the same draft sessions) | measured | `raw/accept.py` | `raw/acceptance.json` |
| 16 | Thai decode speed, 1 user (10 prompts, 1504 tokens, 61 s total wall including prompt processing) | about 24.7 tok/s overall; T8 alone: 700 tokens in 24.1 s (29 tok/s) | measured, derived from client timings | `raw/thai_run.py` | `raw/thai_answers.json`, `raw/thai_run.log` |
| 17 | Thai quality, 10 prompts | 6 correct, 4 partly, 0 wrong, 0 not Thai (grader: the agent running the check) | measured, subjective | `raw/thai_run.py` | `thai-qualitative.md`, `raw/thai_answers.json` |
| 18 | Passkey at 1063 prompt tokens (target "2K"; the prompt-length calibration undershoots) | correct | measured, 1 trial | `raw/needle.py 51105 raw/needle.jsonl 2048:0.5` | `raw/needle.jsonl` |
| 19 | Passkey at 3062 tokens (run to cover the 2K to 4K range) | correct | measured, 1 trial | `... 4000:0.5` | `raw/needle.jsonl` |
| 20 | Passkey at 15783 tokens ("16K") | correct, 8.1 s wall | measured, 1 trial | `... 16384:0.5` | `raw/needle.jsonl` |
| 21 | Passkey at 133778 tokens ("128K") | correct, 48.7 s wall | measured, 1 trial | `... 131072:0.5` | `raw/needle.jsonl` |
| 22 | Teacher-forced agreement with CPU bf16 reference, 8 prompts x up to 48 tokens | 361 of 379 positions, 95.3 percent. Per prompt: 43/48, 45/48, 46/48, 48/48, 44/48, 41/43, 48/48, 46/48 | measured | `raw/agree_chip.py` (chip generates), `raw/agree_cpu.py` (CPU scores each chip token given the chip's prefix) | `raw/agree_chip.json`, `raw/agree_cpu.json`, `raw/agree_cpu.log` |
| 23 | p150 package install | no new install: copied the package folder to `p150-verify/bundle` and ran it with `VENV=` stage 7's venv. The p150 and p300 packages have byte-identical `requirements.txt`, `vllm-overrides.txt`, `install.sh`, `prepare_model_dir.py`, `model.py`, `base_config/` and wheels (same sha256). They differ in `run.sh`, `tt_kernel_manifest.json` and the `vllm_models/<name>/` metadata folder | measured (file comparison) | `cmp`/`diff`/`sha256sum` on `stages/7/package/*` | `p150-verify/bundle` |
| 24 | p150 launch to healthy (1-chip cache) | about 30.8 min (launch 16:00:48, healthy 16:31:36) | measured | `raw/start_server.py` with `TT_VISIBLE_DEVICES` of the leased board | `raw/p150_server.log`, `raw/p150_start_epoch.txt` |
| 25 | p150 first answer, "What is 7 times 6?" | same text as p300, "... = 42" | measured, pass | curl | `raw/first_answer_p150.json` |
| 26 | p150 coding prompt (first SPEED-Bench coding prompt, 1 user) | 158 tokens, TPOT 21.71 ms (46.1 tok/s), TTFT 260 ms | measured, one prompt only | `raw/coding_bench.py 51106 1 ... 512 1` | `raw/p150_coding_u1.json` |
| 27 | p150 DFlash2 acceptance on that prompt | 6.18 of 7 per iteration (the p300 run on the same prompt: 6.18) | measured, one prompt | server log | `raw/p150_server.log` (`slot 0 session: 22 iters, 157 committed tokens`) |
| 28 | p150 passkey at 4109 tokens | correct, 2.6 s wall | measured, 1 trial | `raw/needle.py 51106 raw/p150_needle.jsonl 5000:0.5` | `raw/p150_needle.jsonl` |

## What the numbers show

* Both packages boot from their own warm weights caches and answer the arithmetic check with 42. The p150 package, which stage 7 had not booted, boots and works for the three checks run (first answer, one coding prompt, a 4K passkey). No more p150 testing was done: one coding prompt gives a speed for one prompt, not a distribution, and the p150 max length (16384) was not stress-tested.
* On the SPEED-Bench coding prompts, the 2-chip package decodes at about 80 tok/s for one user and about 54 tok/s per user with four users (179 tok/s in aggregate). These depend on the drafter: acceptance on code was 5.57 of 7 draft tokens.
* The drafter accepts far fewer tokens on Thai text: 1.03 of 7 over the 10 Thai prompts, against 5.57 on code. The drafter was trained on base Qwen3.8-27B, and this run shows that on Thai prompts it agrees with the fine-tune only about one token in each iteration. The run does not separate how much of that gap comes from the language and how much from this fine-tune; testing that would need base-model weights and the same prompts. With that low acceptance, Thai decode ran at about 25 to 29 tok/s, roughly a third of the coding speed. Thai tokens are also shorter units of text than English ones, so tokens/s is not comparable across the two languages as a measure of reading speed.
* The sweep text (random fact sentences) gave TPOT near 30 ms at 1K and 4K and 26 ms at 16K. Acceptance on that text was not parsed from the log, so the cause of the slower decode than on code was not measured.
* The first request of a new prompt shape was slower (4.6 s TTFT for the first 1K prompt against 0.5 s on the second try), so a first request after boot includes some compile time.
* Passkey retrieval worked in all four trials on the 2-chip package, up to 133778 tokens, and in one 4109-token trial on the p150 package. One trial per length is not a reliability figure. The needle position was the middle of the text only.
* Against the CPU bf16 reference, the chip agreed on 95.3 percent of positions when the CPU was given the chip's own prefix. Mismatches were scattered single tokens (for example "simple" vs "user", or "." vs ":"). Whether they are near-ties in the reference logits was not checked. Both sides run in bf16 with different kernels, so 100 percent was not expected. The check used the model's default chat template (reasoning on, as stage 1 and stage 7 did), so the 48 tokens are reasoning text, not final answers.
* The Thai answers are fluent and the numeric and structured tasks (arithmetic, JSON, summary, translation to English) were right. Three answers have a garbled phrase and one translation changed "Wednesday" to "Tuesday". Ten prompts is a smoke test and says nothing about benchmark-level Thai ability, and no base-vs-fine-tune comparison was run.

## What the numbers do not show

* Boot time: the 2 to 5 minute warm boot in the task description was not reproduced. Both boots took about 31 to 34 minutes. The log shows the tensor cache for the main model was read, not regenerated (all 109 "Generating cache" lines are for the dflash drafter files under the cache's `dflash/` folder), and most of the time is the decode warm-up at buckets 1, 2, ... where tt-metal compiles kernels (for example 15:23:43 to 15:27:04 for bucket width 1). The kernel cache under `~/.cache/tt-metal-cache` is shared with other jobs; why it did not hit is not established. A second boot was not timed.
* No logprobs or sampling-based quality was measured (the server rejects them).
* Per-user speeds are from 16 prompts with a 512-token cap, one run each, so run-to-run variation is not measured (the repeated sweep differed by under 3 percent).
* The 4-user run used the same 16 prompts as the 1-user run; with 4 users, 4 requests are in flight and the numbers are for that concurrency only. Not run: 2 users, 80 prompts, 2048-token cap, HumanEval, 256K context.
* The p150 package was run once, with stage 7's venv rather than one built from the p150 package (the inputs are identical, see row 23). Its own `install.sh` was not run.
* The lease for the p150 run (78f6e1) was no longer listed when `gozer release` was called after the run; the chips showed FREE. The server had stopped before the release call.

## Housekeeping

* Servers: both stopped (process groups killed via the pid files `raw/p300.pid`, `raw/p150.pid`; `raw/stop_server.py`). Leases: b8583d released; 78f6e1 not found at release time, chips FREE. The CPU scoring job finished.
* Disk: `/mnt/bonus` had 148 GB free at the end (30 GB at the start of the session, 151 GB when the p150 copy was made). The p150 copy added about 100 MB. Nothing was deleted.
* Not used: `~/.cache/qwen36-src-build` and any other model's tensor cache.
* `raw/generated/` is a scratch folder tt-metal created in the working directory; it holds no results.
