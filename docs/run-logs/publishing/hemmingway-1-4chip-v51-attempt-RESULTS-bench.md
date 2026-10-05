# hemmingway-1-p300x2 (4 chips, v5.1 container) measurements, 2026-10-05

Model: Altworld/Hemmingway-1 @ 1a5f363a3dd2d1cc456c28b8abbb403b9555efaf. Package: `staged/hemmingway-1-p300x2`
(image `tt-model/hemmingway-1-p300x2:9517c364454f`). Nothing was pushed. Paths below are relative to
`/mnt/bonus/models/orchard-runs/hemmingway-1-p300x2-v51/`. Every number is labelled **measured** and was measured on
this box (2 x P300c, chips 0-3, gozer leases f39f06, 8fe179, e73f1b, 6e9bbb).

## Summary

**The package as staged does not serve this model correctly. Do not publish it as is.**

* It builds from public sources, loads its image from the staged layout, boots on 4 chips from an empty tensor cache
  (727 s with a cold kernel cache) and answers the first checks correctly (42, Paris, a correct Fibonacci function).
* With the recipe it follows (changh95's plain-decode settings), later requests in the same server session came back as
  loops ("!!!!", "HeyHeyHey", "# # #"). Turning off the traced prefill (`QWEN36_PREFILL_BUCKET_TRACE=0`) made short
  repeated prompts stable in one test (8 of 8), but the full benchmark then showed: passkey prompts of 2K, 16K and 135K
  tokens decode to garbage after the first (correct) digit; "What is 7 times 6?" answered "4" on one repeat and "421"
  under 10 concurrent requests; concurrent answers never matched the single-request answers; per-user speed at 4 users
  fell to 3.3 tok/s.
* Controls (section C) say whether the fault belongs to this image build, to the stack with this fine-tune, or to the
  stack in general.

## A. Boot and correctness

| # | Item | Value | Label | Command or script | Evidence |
|---|---|---|---|---|---|
| A1 | Cold boot, final package (empty tensor cache, no kernel cache, image loaded from staged `image/`) | 727 s command to ready (tt-model "ready 12m 6s": weights converted 7m 58s, warm-up 185 s) | measured | `tt-model serve --port 8000 --device-id 0,1,2,3 --local-only --no-update-check staged/hemmingway-1-p300x2/tt_kernel_manifest.json` with `HF_HOME` = the local Hemmingway-1 copy, `HF_TOKEN` unset | `bench/raw/serve5b.out`, `bench/raw/boot5b_wrapper_lines.txt` (cache held only the marker at start), `bench/raw/server_boot5b_final.log` |
| A2 | Same, first build (before the two env fixes) | 800 s (boot 1), 761 s (boot 3) | measured | same | `bench/raw/serve.out`, `bench/raw/serve3.out` |
| A3 | Warm boot (tensor cache and kernel cache present) | 440 s to `/health` 200, with `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0` | measured, 1 boot (boot 4, image 3c927f5 + bucket trace off by `--env`) | `bench/raw/serve4-cmd.sh` | `bench/raw/boot4_ready.txt`, `bench/raw/server_boot4.log` |
| A4 | Warm boot without the pinned-memory fix | hung at weight load for 19 min, EngineCore 100% CPU in `ttnn as_tensor` from `shard_w` | measured, 1 boot | `tt-model serve ... --no-async-scheduling` | `bench/raw/server_boot2_hang.log` |
| A5 | One failed cold boot | boot 5: container SIGKILLed (exit 137) 5 min in, while transformers loaded the model on CPU; no error printed; not reproduced in boot 5b (lowest MemAvailable 81.6 GB); cause unknown | measured | same as A1 | `bench/raw/server_boot5_killed.log`, `bench/raw/mem_boot5b.txt` |
| A6 | First answers, final package | "42"; "Paris"; correct iterative `fib`; landlord text coherent; building sample 900 tokens coherent; with no `chat_template_kwargs` (package default thinking off) "42" and empty reasoning | measured | `bench/scripts/checks.py 8000 bench/raw/checks_final.json default-template` | `bench/raw/checks_final.json` |
| A7 | Repeat probes, final package, right after A6 | landlord x3 coherent (wording differs a little between repeats); neighbour x8 identical and coherent | measured | `rep.py` (copy in NOTES) | `bench/raw/probes_final.txt` |
| A8 | Repeat probes, first build (bucket trace on, async on) | 42, Paris, fib good; then "!" (token 0), "2", "1!!!!...", looping refusals | measured | `checks.py`, `probe.py` | `bench/raw/checks.json`, `bench/raw/server_boot1_async.log` |
| A9 | Same, async off only (boot 3) | 42, Paris, fib, building good; landlord "#\n\n!\n\n!..."; neighbour x8 all loops | measured | same | `bench/raw/checks_boot3.json`, `bench/raw/server_boot3.log` |
| A10 | After the benchmark suite, final package | "7 times 6" -> "42" then "4" | measured | `rep.py` | NOTES.md |
| A11 | Concurrent vs single (10 writing prompts, 300-token cap) | 0/10 identical at 4 at a time and at 10 at a time; at 10: "421" for 7 x 6, "Hi" alone for the landlord note | measured | `bench/scripts/batch_check.py` | `bench/raw/batch_check.json` |

## B. Benchmarks on the final package (boot 5b, one session)

Same scripts, prompts and caps as the 2-chip run (`hemmingway-1-run3/publish/scripts`). Greedy, thinking off.

| # | Item | 4 chips (this package) | 2 chips (`episod/hemmingway-1-p300`, RESULTS-bench of run 3) | Label | Evidence |
|---|---|---|---|---|---|
| B1 | Coding, 1 user, per-user (1000 / mean TPOT) | 30.4 tok/s (mean TPOT 32.89 ms, median 32.08). 31.0 over 79 requests excluding the request that overlapped my extra requests | 54.1 tok/s (DFlash2) | measured | `bench/raw/coding_u1.json` |
| B2 | Coding, 1 user, token-weighted | 29.1 tok/s | 61.8 tok/s | measured | same |
| B3 | Coding, 1 user, TTFT / aggregate | mean 160 ms / 28.7 tok/s, 27,134 tokens, 943.8 s, 5 hit 2048 cap | 232 ms / 59.3 tok/s, 6 hit cap | measured | same |
| B4 | Coding, 4 users, per-user | 3.3 tok/s (mean TPOT 306 ms, median 122 ms); token-weighted 10.2 | 28.5 tok/s; token-weighted 42.2 | measured | `bench/raw/coding_u4.json` |
| B5 | Coding, 4 users, TTFT / aggregate | 583 ms / 38.5 tok/s, 10 hit cap, mean output 412 tokens | 749 ms / 130.4 tok/s | measured | same |
| B6 | Coding, 32 users | per-user 1.1 tok/s (mean TPOT 876 ms), token-weighted 3.6, TTFT 2344 ms, aggregate 99.7 tok/s, 14 hit cap, mean output 482 tokens | not supported (max 4 sequences) | measured | `bench/raw/coding_u32.json` |
| B7 | Sweep 1024/128, 1 user (6 requests) | 29.1 tok/s incl. TTFT, TTFT 185 ms, TPOT 33.2 ms | 51.5, 318 ms, 17.1 ms | measured | `bench/raw/sweep/1024_128_u1.json` |
| B8 | Sweep 16384/128, 1 user (4) | 20.5 tok/s incl. TTFT, TTFT 1964 ms, TPOT 33.8 ms | 21.8, 3788 ms, 16.3 ms | measured | `bench/raw/sweep/16384_128_u1.json` |
| B9 | Sweep 1024/128, 4 users (8) | 43.3 aggregate (10.8 per user), TTFT 581 ms, TPOT 88.4 ms | 120.9 (30.2), 969 ms, 25.7 ms | measured | `bench/raw/sweep/1024_128_u4.json` |
| B10 | Passkey 2,075 / 15,783 / 135,567 tokens | all three FAILED: right first digit, then garbage | all retrieved | measured | `bench/raw/needle.jsonl` |
| B11 | Teacher-forced agreement, 8 x 48 | 364/384 = 94.8% (45, 44, 45, 47, 46, 46, 44, 47) | 365/384 = 95.1% | measured | `bench/raw/teacher_forced.json` |
| B12 | Free-run first divergence | tokens 8, 4, 4, 37, 8, 4, 4, 4 ("the nine steps" vs "this carefully") | 24, 8, 12, 38, 26, 35, 12, 13 | measured | `bench/raw/free_run_ids.json` |
| B13 | 10 writing prompts | 6 good, 4 partly (all four for bracket placeholders), 0 poor; graded by the agent | 7 good, 3 partly | measured (agent's judgement) | `bench/raw/qual.json` |
| B14 | Building sample (900 tokens) | coherent to the cap in boots 3, 4 and 5b | 830 tokens, coherent | measured | `bench/raw/checks_final.json` |
| B15 | DFlash acceptance | not applicable (plain decode) | 4.38/7 coding, 1.72/7 writing | - | - |

Conditions that differ from the 2-chip run: different serving stack (plain decode vs DFlash2, other tt-metal branch,
other plugin build); `--no-async-scheduling` and bucket trace off on both; the 4-chip coding u1 run overlapped 8 of my own
requests for about 16 s (one request affected, figures with and without it are given); 4-chip numbers come from a
server whose output was being corrupted (section A), so the speed figures describe a broken configuration.

## C. Controls

Each control used the final package's exact docker argv (same env and vLLM arguments: bucket trace off,
`--no-async-scheduling`, pinned-memory limit 0), an empty tensor cache and an empty kernel cache under `controls/<X>/`,
and the same probe script (`controls/probe.sh`: landlord x3, neighbour x8, "7 times 6" x4, passkey ~2K and ~16K, "7
times 6" x2 again), in one server session each, greedy, thinking off.

| Combination | Cold boot | "7 times 6" x4 | Passkey 2,075 / 15,783 | Result | Evidence |
|---|---|---|---|---|---|
| Our image + Hemmingway-1 (final package, boot 5b) | 727 s | 42 once in the checks; "42" then "4" after the suite | "6" / "9峪986 **>> 98..." | FAIL | `bench/raw/needle.jsonl`, NOTES.md |
| A: changh95's image `tt-model/qwen3.8-27b-p300x2:0becf4834925` + Hemmingway-1 (our wrapper mounted) | 755 s | 42, 4, 4, 4 | "6" / "9 ourselves" | FAIL | `controls/A/probes.txt`, `controls/A/server.log` |
| B: our image + base Qwen/Qwen3.8-27B @ 1d4bf0f | 444 s | 42, 4, 4, 4 | "6" / "98isode并不需要" | FAIL | `controls/B/probes.txt`, `controls/B/server.log` |

All three combinations fail the same way. The first token after prefill is right (the passkey's first digit, the "4"
of 42) and the following decode steps are wrong. The fault therefore belongs to the serving stack (tt-metal 0e988de
qwen36 code with plugin 751ec335 under these settings). The fine-tune and this image build do not cause it. Not tested:
changh95's image with its own published settings and the base weights. Run 3's stage 4 log records that package
"returned noise too" with the base model, which fits. The 2-chip package (tt-metal 9f6b02d, branch qwen36-p150x2,
DFlash2) retrieved the same passkeys, so a working 4-chip package probably needs that branch or a fixed prefill-opt
branch.

## What is and is not shown

* Shown: the v5.1 interface can build this package from public sources in about 8 minutes on this box (warm ccache);
  the staged image loads and boots on 4 chips from an empty, per-package tensor cache; the start-up wrapper builds
  the base-shaped model directory and refuses a foreign cache; first answers are correct.
* Shown: the serving stack this recipe uses corrupts decoding after the first request(s) of a session, with this
  fine-tune and with the base model, in our image and in changh95's image.
* Not shown: any speed figure for a correctly working 4-chip configuration. The speed numbers in section B were
  measured on the failing configuration and should not be published.
* Not measured: warm boot of the final image (an equivalent image took 440 s), sampling above temperature 0, tool calls,
  a Hub pull.

## Disk and caches I created

See NOTES.md, section "Cleanup".
