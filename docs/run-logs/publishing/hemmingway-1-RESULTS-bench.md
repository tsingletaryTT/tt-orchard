# Hemmingway-1 package measurements (2026-10-04)

Model: Altworld/Hemmingway-1 (revision 1a5f363a), packages `hemmingway-1-p300` (2 chips) and `hemmingway-1-p150` (1 chip).
Nothing was published, pushed or deleted. All paths below are under
`/mnt/bonus/models/orchard-runs/hemmingway-1-run3/publish/` (called `publish/`). Every number is labelled **measured** and was
measured on this box on 2026-10-04 with the packages as staged.

## How the server was started

Stage 7's `serve_and_compare.py` was reproduced by `scripts/start_server.py`: `bash run.sh --port N` in its own session with
`TT_CACHE_PATH` and `TT_CACHE_HOME` set to the model's own cache, `HF_HOME` = `stages/7/verify/hf`, `HF_HUB_OFFLINE=1`,
`MODEL_WEIGHTS_DIR` and `HF_MODEL` = the bundle's `model-dir`. Server: vllm 0.26.0 with the TT plugin, DFlash2 drafter, greedy only.

* p300: bundle `stages/7/verify/bundle` (the installed copy stage 7 booted), cache
  `/mnt/bonus/models/orchard-runs/cache/hemmingway-1/2chip-qwen3.8-27b-dflash2-p300/tt_cache` (marker `.orchard-model` =
  `Altworld/Hemmingway-1`), port 48271, lease a4ce89 (chips 0000:01:00.0 and 0000:02:00.0), 68 minutes held. Released.
* p150: see section 7.
* Chat requests use `enable_thinking: false`. Stage 7's teacher-forced check used the default template (the model starts with a
  thinking block); section 6 uses the default template too, so it is comparable with stage 7, not with the chat tests.

## Results table

| # | Item | Value | Label | Command or script | Evidence |
|---|---|---|---|---|---|
| 1a | p300 ready time, tensor cache present | 2025 s (33.8 min) from launching `run.sh` to `/health` 200. Stage 7 measured 2034.4 s. | measured | `scripts/start_server.py` then a poll on `/health` | `raw/p300_ready.txt`, `raw/p300_boot_start.txt`, `raw/p300_server.log` |
| 1b | Of that, "phase-1 warmup (alloc + compile)" | 1261.7 s | measured | read from the server log | `raw/p300_server.log` (line with `phase-1 warmup`) |
| 1c | First answer "What is 7 times 6?" | `42` (3 tokens, 4.3 s wall for the first request) | measured | `curl /v1/chat/completions` | `raw/p300_first_answer.json` |
| 2a | Coding, 1 user, per-user speed | 54.1 tok/s (1000 / mean TPOT 18.50 ms); median TPOT 13.61 ms | measured | `scripts/coding_bench.py 48271 1 ... 2048 80` | `raw/p300_coding_u1.json`, `.out` |
| 2b | Coding, 1 user, token-weighted decode speed | 61.8 tok/s over all 80 requests; 69.1 tok/s (mean TPOT 14.46 ms) over the 71 requests with at least 64 output tokens | measured | computed from the same JSON | `raw/p300_coding_u1.json` |
| 2c | Coding, 1 user, TTFT | mean 232 ms, median 185 ms | measured | same | same |
| 2d | Coding, 1 user, aggregate | 59.3 tok/s, 24,941 output tokens, 420.5 s wall, 6 of 80 hit the 2048 cap | measured | same | same |
| 2e | Coding, 4 users, per-user speed | 28.5 tok/s (mean TPOT 35.13 ms, median 23.89 ms); token-weighted 42.2 tok/s; 42.0 tok/s over the 71 requests of at least 64 tokens (mean TPOT 23.83 ms) | measured | `scripts/coding_bench.py 48271 4 ... 2048 80` | `raw/p300_coding_u4.json` |
| 2f | Coding, 4 users, TTFT | mean 749 ms, median 728 ms | measured | same | same |
| 2g | Coding, 4 users, aggregate | 130.4 tok/s, 191.3 s wall | measured | same | same |
| 2h | Sweep ISL 1024 / OSL 128, 1 user (6 requests) | 51.5 tok/s per user including TTFT; TTFT mean 318 ms; TPOT mean 17.1 ms (58 tok/s decode) | measured | `scripts/sweep.sh` (`vllm bench serve`, random prompts, `--ignore-eos`, temperature 0) | `raw/sweep/1024_128_u1.json`, `.log` |
| 2i | Sweep ISL 16384 / OSL 128, 1 user (4 requests) | 21.8 tok/s per user including TTFT; TTFT mean 3788 ms; TPOT mean 16.3 ms (61 tok/s decode) | measured | same | `raw/sweep/16384_128_u1.json` |
| 2j | Extra: ISL 1024 / OSL 128, 4 users (8 requests) | 120.9 tok/s aggregate (30.2 per user including TTFT); TTFT mean 969 ms; TPOT mean 25.7 ms | measured | same | `raw/sweep/1024_128_u4.json` |
| 3a | DFlash acceptance, coding prompts (80 prompts, 1 user) | 4.38 of 7 drafted tokens accepted per verify step (62.6%), 5.36 tokens committed per step; iteration-weighted over 4,644 steps | measured | `scripts/accept.py` on the server log range of the run | `raw/p300_acceptance.jsonl`, `raw/mark_coding_u1_*.txt`, `raw/p300_server.log` |
| 3b | DFlash acceptance, coding prompts at 4 users | Identical to 3a (4,644 steps, 4.38 / 7). Greedy decoding gave the same per-request steps. | measured | same, range of the 4-user run | `raw/p300_acceptance.jsonl` |
| 3c | DFlash acceptance, the 10 qualitative prompts | 1.72 of 7 (24.6%), 2.69 tokens committed per step; 356 steps. Unweighted mean over the 10 sessions 1.87 of 7. | measured | same, range of the qualitative run | `raw/p300_acceptance.jsonl` |
| 3d | DFlash acceptance, random-token sweep prompts | 4.52 of 7 (64.5%) over 422 steps. Random prompts with ignored EOS; not representative of real use. | measured | same | `raw/p300_acceptance.jsonl` |
| 4 | Qualitative check | 7 good, 3 partly, 0 poor of 10, graded by me on one greedy answer each | measured (my judgement) | `scripts/qual.py` | `qualitative.md`, `raw/p300_qual.json` |
| 5a | Passkey, 2,075-token prompt, depth 0.5 | retrieved (answer 625964 = key) | measured, 1 trial | `scripts/needle.py 48271 ... 3000:0.5` | `raw/p300_needle.jsonl` |
| 5b | Passkey, 15,783-token prompt | retrieved, 8.0 s wall | measured, 1 trial | `needle.py ... 16384:0.5` | same |
| 5c | Passkey, 135,567-token prompt | retrieved, 44.5 s wall | measured, 1 trial | `needle.py ... 133000:0.5` | same |
| 5d | Two further short prompts, 1,063 and 2,944 tokens | both retrieved | measured, 1 trial each | `needle.py` | same |
| 6 | Teacher-forced agreement with the CPU reference, 8 prompts x 48 tokens | 365 of 384 = 95.1% first-token agreement. Per prompt 45, 42, 45, 47, 47, 46, 46, 47 of 48. Stage 7 on its single prompt: 30 of 32 (93.75%). | measured | `scripts/cpu_ref.py` (CPU, bf16) then `scripts/tf_compare.py` | `raw/cpu_ref.json`, `raw/p300_teacher_forced.json`, `.out` |
| 6b | Free-run (no forcing) first divergence from the CPU reference, per prompt | token 24, 8, 12, 38, 26, 35, 12, 13 of 48 | measured | `scripts/tf_compare.py` | `raw/p300_teacher_forced.json` |
| 7a | p150 package ready time, tensor cache present | 1665 s (27.8 min); phase-1 warmup 1139.7 s | measured | `scripts/start_server.py` on `p150-bundle` | `raw/p150_ready.txt`, `raw/p150_server.log` |
| 7b | p150 first answer "What is 7 times 6?" | `42` | measured | `curl /v1/chat/completions`, port 48272 | `raw/p150_first_answer.json` |
| 7c | p150, 1 coding prompt (first SPEED-Bench coding prompt) | 159 tokens, TPOT 22.89 ms (43.7 tok/s), TTFT 249 ms. One request only. | measured | `scripts/coding_bench.py 48272 1 ... 2048 1` | `raw/p150_coding_1prompt.json`, `.out` |
| 7d | p150 draft acceptance on that one prompt | 5.91 of 7 (84.4%), 23 steps. One request; not an average. | measured | `scripts/accept.py` | `raw/p150_acceptance.jsonl` |
| 7e | p150 passkey, 4,293-token prompt | retrieved, 2.9 s wall | measured, 1 trial | `scripts/needle.py 48272 ... 5200:0.5` | `raw/p150_needle.jsonl` |

## Workloads and definitions

* Coding prompts: the 80 first-turn prompts of the `coding` category of nvidia/SPEED-Bench, read from the local Hugging Face
  cache (`~/.cache/huggingface/hub/datasets--nvidia--SPEED-Bench`, available offline). The prompts are HumanEval-pack style code
  stubs sent as the user message with no instruction. They are not stored in `publish/` (evaluation licence); the result files hold
  only prompt ids. Answers end at the end-of-turn token or at 2048 tokens.
* Per-user speed = 1000 / mean TPOT, where each request's TPOT = (end - first token) / (completion tokens - 1). This is the
  definition in the earlier Qwen3.8-27B p300c card. The mean is pulled up by very short answers (some requests produce 2 to 4
  tokens and have a noisy TPOT), so rows 2b and 2e also give a token-weighted figure and one over requests of at least 64 tokens.
* Sweep rows are `vllm bench serve` (v0.26.0 from the same venv) on `/v1/completions`. "Per user including TTFT" = OSL / mean
  end-to-end latency, as in the earlier card's tables. Each cell used 4 to 8 requests.
* Acceptance: the server logs one line per finished request, `accept A/7 -> T tok/iter`, where A is the mean number of drafted
  tokens accepted per verify step out of 7. Rows 3a to 3d weight each request by its number of steps. Log line ranges were taken
  with `wc -l` before and after each run (`raw/mark_*.txt`).

## Things that went wrong or need care

1. **Warm boot was not 2 to 5 minutes.** Both servers loaded the tensor cache (log lines `Loaded cache for ...`) and still took
   28 to 34 minutes, of which about 19 to 21 minutes were the drafter-and-verify "phase-1 warmup (alloc + compile)". The cause was
   not investigated. The tensor cache evidently does not remove the compile step in this layout. Stage 7's recorded 2034 s was
   a normal figure for this setup, not a cold-cache outlier. Anything that states a 2 to 5 minute warm boot for these packages
   should not use that number.
2. **One benchmark run was overlapped and discarded.** The coordinator sent 2 chat requests to the server between 15:59:28 and
   16:00:27 local. My first 1-user coding run (15:53:32 to 16:00:43) overlapped that window. It is kept as
   `raw/p300_coding_u1_OVERLAPPED_by_coordinator_requests.json` and is not used (it reported per-user 58.8 tok/s and mean TTFT
   428 ms, with the last requests slowed). The run in the table started at 16:00:57 and is clean. The qualitative run (ended
   15:53), the 4-user run (after 16:08), the sweep and the passkeys all ran outside the window.
3. **Passkey sizes.** The needle script's token calibration is nonlinear at small sizes, so asking for 2,048 tokens gave 1,063.
   The table reports the actual prompt token counts. The "128K" prompt is 135,567 tokens, above 131,072.
4. **Teacher-forced method limits.** The CPU reference decodes with the default chat template, so its 48 tokens are mostly the
   model's thinking text ("Let me work through this carefully ..."), not the final message. The 19 disagreements are near ties
   between plausible next words (for example ` short` vs ` practical`). The comparison re-tokenises the server's text and compares
   the first token id, the same way stage 7 did.
5. **gozer release of the p150 lease.** `gozer release 55109e` reported `reset failed -- unit released but NOT marked clean`
   because `gozer` could not read the board serial of the other agent's chips while it was resetting them. `gozer status` then
   showed chips 0 and 1 as `BUSY-UNTRACKED` with the pid of a `tt-smi -r 0000:03:00.0,0000:04:00.0` that belongs to the other
   agent's release. A later `gozer status` showed all four chips FREE, so the transient state cleared by itself. My servers
   were confirmed stopped (no vllm process remains). I did not reset anything by hand.
6. **Lease grain.** Asking for 1 chip granted both chips of board 0000046131924062; the p150 server used the first chip
   (`TT_VISIBLE_DEVICES` trimmed by `run.sh`).

## 7. p150 package: how it was installed

Disk before starting: 148 GB free on `/mnt/bonus` (30 GB when the task began; the other agent freed space meanwhile). Nothing was
deleted. The p150 `requirements.txt`, the `wheels/` directory, `install.sh`, `prepare_model_dir.py` and `vllm-overrides.txt` are
byte-identical to the p300 bundle's (checked with `diff`; only `run.sh`, `README.md`, `tt_kernel_manifest.json` and the
`vllm_models` folder name differ). I therefore did not re-run `install.sh`. I copied the staged package (96 MB) to
`publish/p150-bundle` and ran its own `run.sh` with `VENV` pointing at the stage 7 venv
(`stages/7/verify/bundle/venv`). `run.sh` then built `p150-bundle/model-dir` from the Hemmingway-1 weights in
`stages/7/verify/hf`. This shows the p150 profile (its `run.sh`, 16K context, single chip, MESH_DEVICE P150) boots and serves
with the staged files. It does not show that a fresh `install.sh` on a clean machine works for the p150 package, because that was
not run. Cache: `/mnt/bonus/models/orchard-runs/cache/hemmingway-1/1chip-qwen3.8-27b-dflash2-p150/tt_cache`
(marker `Altworld/Hemmingway-1`). p150 `run.sh` limits the context to 16,384 tokens, so only the 4K passkey was run.

## What the numbers show and do not show

* The 2-chip package boots from the model's own cache, answers the first check correctly (42) and produces coherent text on
  every qualitative prompt. The same holds for the 1-chip package on one arithmetic question, one coding prompt and one 4K
  passkey. The 1-chip results are a boot check, not a benchmark.
* Speed on 2 chips for coding prompts is about 54 tok/s per user at one user and about 28 tok/s per user at four users by the
  card's definition (1000 / mean TPOT), with an aggregate of 59 and 130 tok/s. Short answers distort the mean TPOT; the
  token-weighted figures are 62 and 42 tok/s. Time to first token is 0.2 to 0.75 s on these short prompts and 3.8 s at a
  16K prompt.
* The drafter, trained on the base Qwen3.8-27B, accepts about 4.4 of 7 drafted tokens (63%) on coding prompts with this
  fine-tune. On everyday-writing prompts it accepts about 1.7 of 7 (25%) and commits 2.7 tokens per step. The writing figure
  comes from 10 prompts and is the relevant one for this model's purpose; speed on writing prompts was not measured at a
  larger scale. Expect lower speed on prose than the coding numbers suggest. No comparison with the base model's acceptance
  was run, so these numbers do not say whether the fine-tune lowered acceptance.
* The passkey tests show retrieval of an easy 6-digit key from synthetic text at 2K, 16K and about 135K tokens, one trial each.
  They are not a long-context quality evaluation.
* The qualitative grades are one reviewer's reading of ten greedy answers. They support "it writes the requested message
  directly without preamble or options". They do not support any claim about quality relative to another model, because no other
  model was run.
* The 95.1% teacher-forced agreement says the chip's next-token choice matches the CPU bf16 reference on 365 of 384 positions in
  thinking-style text. It does not measure end-to-end text quality, and the 19 mismatches were not examined beyond their first
  tokens.
* Not measured: first-boot time with an empty cache and an empty kernel cache, speed above 4 users, throughput on writing
  prompts at 1 and 4 users, p150 speed beyond one request, a clean-machine p150 install, and any base-vs-fine-tune comparison.
