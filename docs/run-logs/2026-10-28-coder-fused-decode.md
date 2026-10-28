# 2026-10-28: the fused-decode Qwen3-Coder-Next bundle as the default coder (issue 180)

Question: is `raahemnabeel/qwen3-coder-next` (decode step rebuilt around fused kernels) a better coder than the
current default, `raahemnabeel/qwen3-coder-next-blackhole`, on the `p300` profile (one board, two dies), and does
it pass the role-fit test?

Answer: it passed the role-fit test and decoded faster than the earlier package's recorded numbers in all 28 speed
samples, by 1.1 to 2.8 times, with a median of 2.1 times. The default is now the new bundle. The earlier package is
the documented fallback (`scripts/setup.sh --coder coder-next-v1`).

The earlier package was not measured in the same hour. Its numbers below are recorded ones, and the comparison
inherits that limit (see "What could not be determined").

## The two packages

From `tt-model info` for each (manifest schema 5.1) and the public card of the new one.

| | `raahemnabeel/qwen3-coder-next-blackhole` (earlier) | `raahemnabeel/qwen3-coder-next` (new) |
| --- | --- | --- |
| Kind | container (`vllm-plugin`) | container (`vllm-plugin`) |
| Profiles | `p300x2` (default), `p150x4`, `p300`, `p150x2` | the same four |
| Port in the manifest | 8000 (the supervisor passes `--port`) | 8000 (the supervisor passes `--port`) |
| Model id the API expects | `Qwen/Qwen3-Coder-Next` | `Qwen/Qwen3-Coder-Next` (the card says to pass the weights id as the model) |
| Container name on `p300` | `tt-model-qwen3-coder-next-blackhole-p300` | `tt-model-qwen3-coder-next-p300` |
| Image | `tt-model/qwen3-coder-next-blackhole:bd0ca927a46b` | `tt-model/qwen3-coder-next:0c170421ca1b` |
| Args | auto tool choice, chunked prefill, 2048 batched tokens | the same, plus `--enable-prefix-caching` |
| Env | chunked prefill, program guard, trace region | the same, plus fused MoE decode, replicated decode residual, GDN and attention decode ops, rows top-k |
| tt-metal | `d59904e3`, dirty, not pushed | `de8e0fb2`, clean, pushed |
| Tool parser, reasoning parser | `qwen3_coder`, none | `qwen3_coder`, none |

The weights are the same checkpoint (`Qwen/Qwen3-Coder-Next` at revision `a7fbcb5c`) and were already in the local
Hugging Face cache. The card says the fabric layer boundary runs only on four dies, so the p300 profile does not
get it. The card's p300 single-user figure (117.5 tok/s against 39.4) comes from a quiet box and a different method.

What this means for the code:

- The supervisor's start path (`orchard/server.py`) is unchanged. It runs `tt-model serve <package> --detach
  --local-only --no-update-check --port <port> --profile p300 --device-id <leased chips>`, and the container name
  comes from the package name and profile. The stop check matches the container by exact name, by image id
  or by published port, so a running container of the other package is ignored. `tests/test_coder_package.py`
  pins that in both directions.
- `--local-only` needs the package installed first. `tt-model serve raahemnabeel/qwen3-coder-next --local-only` on a
  machine without it printed `is not installed` and treated the package as a bundle (it ignored `--profile`
  and `--device-id`). `tt-model pull raahemnabeel/qwen3-coder-next` (1 min 44 s here for the image load) fixes it.
  `scripts/setup.sh` already pulls the coder package.
- The tier config needs no change: the weights id is the same, so `model = "Qwen/Qwen3-Coder-Next"` stays.

## Conditions

- Machine: one QuietBox 2, four Blackhole p300c chips on two boards. This run held board `0000046131924062`
  (chips 0 and 1) under gozer lease `e1328b` (`claude:coder-upgrade`), granted 19:37:57Z. The lease was detached,
  so gozer reaped it when the container stopped at 20:14:47Z.
- Server: `tt-model serve ... --detach --local-only --no-update-check --port 8013 --profile p300 --device-id 0,1`.
  Started 19:40:34Z, answered `/v1/models` at about 19:56Z. That is about 15.5 minutes with cold compile caches for this
  bundle. The 2026-10-06 role-fit note gives 473 s for the earlier package with warm caches.
- Host: 16 cores shared with other agents. One other coder container (the earlier package, serving the kernel ladder on
  the other board) was running the whole time (its engine process used about 330 percent of a core). One-minute load
  average was 6.3 to 12.7 during the measurements. The other board's lease belonged to the kernel-ladder agent;
  another agent queued for this board at 19:52Z and got it only after this run's container stopped, so no other
  workload used these chips.
- Nobody sent a request to the other coder container. Every request in this log went to port 8013.

## Decode speed at one user

The function is `orchard.speedwatch.measure`, which the kernel probe calls before each attempt: one fixed greedy
prompt, 300 tokens, `usage.completion_tokens` divided by the client's seconds for the whole request.
It counts the prompt's prefill in the time. `others` (requests of other clients on the server) was 0 in every sample.

| Series | Samples | tok/s | Median | Host load (1 min) |
| --- | ---: | --- | ---: | --- |
| A, right after boot | 8 | 97.3, 64.1, 69.6, then 107.5, 107.4, 107.6, 107.5, 107.4 | 107.4 | 10.5 |
| B | 10 | 46.8, 73.7, 89.2, 66.6, 99.0, 56.3, 42.7, 68.8, 87.6, 72.2 | 70.5 | 8.3 to 11.4 |
| C | 10 | 75.1, 93.6, 75.0, 83.6, 87.1, 56.9, 96.0, 48.3, 67.3, 93.6 | 79.3 | 11.3 to 12.7 |
| All | 28 | minimum 42.7, maximum 107.6 | 79.3 | |

Recorded for the earlier package on the same profile:

| Source | tok/s |
| --- | ---: |
| Kernel-probe reports, `docs/run-logs/2026-10-28-rmsnorm-read-first.md` (same function) | 38.5 to 38.6 in most attempts, 27.7 to 35.7 in the first three of one arm |
| Role-fit run, 2026-10-06 (8K prompt, 32K prompt; different method) | 41.8, 38.4 |
| Bring-up spike, 2026-10-06 (512 tokens, short prompt) | 39.1 |

Against 38.55 tok/s (the middle of the kernel-probe reports): the new bundle's median is 2.06 times, its slowest sample 1.11 times
and its fastest 2.79 times. Series A shows a steady value of 107.4 to 107.6 tok/s once the first three samples had warmed it. Series B
and C, taken minutes later with other jobs busy on the host, spread from 42.7 to 99.0 and did not return to that value. The
earlier package's own reports also spread (27.7 to 38.6).

## Long prompts

Streamed requests, a unique first line in each (so prefix caching cannot answer them), a reply of 256 tokens
(the prompt ends with a request for a long explanation). TTFT is the time from sending the request to the first
content chunk. Decode is (completion tokens minus one) divided by the time from the first to the last chunk.

| Prompt tokens (server count) | TTFT, three runs | Decode tok/s, three runs |
| ---: | --- | --- |
| 8,254 | 2.94 s, 2.49 s, 3.59 s | 100.6, 29.2, 37.4 |
| 32,823 | 11.96 s, 11.92 s, 14.06 s | 50.5, 102.9, 76.2 |

The earlier package, from the 2026-10-06 role-fit record: prefill 3,460 tok/s at 8K and 3,706 tok/s at 32K, which is
about 2.4 s and 8.9 s for the same prompt sizes (non-streamed, one-token request). The new bundle's prefill here was
2,740 to 2,810 tok/s (prompt tokens over TTFT). The card says prefill is unchanged. This run cannot say whether the
lower number comes from the bundle or from the busy host. TTFT for long prompts is therefore not shown to be equal,
and was 20 to 35 percent longer here (2.9 s against 2.4 s at 8K, 12.0 s against 8.9 s at 32K).

## Four users at once

Four simultaneous non-streamed requests (distinct short prompts, a unique tag in each, 256 tokens at most), three
rounds. Replies end on their own, so token counts differ. Per user is completion tokens over that request's seconds. Total is all tokens over the
round's wall time.

| Round | Per user, tok/s | Total tok/s |
| ---: | --- | ---: |
| 1 | 49.3, 48.2, 49.3, 49.8 | 171.3 |
| 2 | 48.3, 43.5, 48.3, 48.3 | 170.6 |
| 3 | 52.0, 44.6, 52.0, 49.9 | 181.4 |

The earlier package was not run at four users. Its card gives 24.1 tok/s per user and 82.3 in total for 128 input
and 128 output tokens (a different method, a quiet box); for the new bundle the card gives 78.2 and 199.1. This
run's 43 to 52 per user and 171 to 181 in total are below the card's new-bundle figures, like the single-user samples.

## Role-fit test

`python3 -m orchard.rolefit --endpoint http://127.0.0.1:8013/v1 --model Qwen/Qwen3-Coder-Next --logs <the run logs
the 2026-10-06 test used> --out <file>`. Result file:
`2026-10-28-rolefit-qwen3-coder-next-fused-p300.json`, the same shape as
`2026-10-06-rolefit-qwen3-coder-next-p300.json`.

The same turns were replayed. The selection function was run against the saved logs before the test, and it returned the same 50 source
files and the same 7 failure turns as the 2026-10-06 file lists.

| Part | 2026-10-06, earlier package | 2026-10-28, new bundle |
| --- | --- | --- |
| Canary ("What is 7 times 6?") | 42 | 42 |
| Format, 50 recorded turns at temperature 0 (bar 0.95) | 50 of 50 | 50 of 50 |
| Recorded failure turns, 5 replays each at temperature 0.7 | 7 turns, 0 of 35 repeated | 7 turns, 0 of 35 repeated |
| Verdict | pass | pass |

The replay shows that the new bundle produces well-formed tool calls for 50 real prompts and does not repeat the
empty or cut-off replies of the 27B. It does not show that the commands are correct. The new bundle's numerics
differ in places (expert weights stored at 5 significant bits, a different summation order); the card reports no
measurable change on HumanEval, MBPP, GSM8K and IFEval on p300x2.

### The speed part of the role-fit test was unreliable, and is fixed

`orchard.rolefit.throughput` took three faults to the surface on this server (`tests/test_rolefit.py`, each seen
red first):

1. Both of a size's requests sent the same prompt. With prefix caching on, the second was answered from the first's
   cache, and the difference of the two times is the decode time. Each request now opens with its own line.
2. The prompt ended with "Say hello." The new bundle answered in 3 tokens, and (3 - 1) divided by a difference
   of two timings of noise gave 18.8 tok/s. The earlier package happened to answer at length. The prompt now asks for
   at least 400 words, and a reply shorter than 90 percent of the budget gives no decode number and an error that says so.
3. When the 1-token request took longer than the full one (slow first prefill), the decode number was an empty value with an empty error list.
   It now says so.

Even with these fixes the method subtracts two timings, and on a busy host the 1-token timing varies. Six
runs of it on this server (two of them before the fixes) gave decode between 19 and 391 tok/s. Do not read a speed from that part. The result file's
`throughput` entry (31.38 at 8K, 51.71 at 32K) is kept as the tool wrote it. Use the streamed measurements above for speed.

## What could not be determined

- The earlier package's speed in the same hour, on the same board, under the same load. Its numbers are recorded ones. The kernel-ladder agent's
  timing experiment is running on that package, so it was not sent requests. The 2.1 times median is a comparison of
  this run with the recorded reports. A same-hour comparison needs both packages served one after the other on one board.
- Why the speed was 107.5 tok/s in series A and 43 to 99 later. The host was busy throughout (load 6 to 13), and the
  same fluctuation shows in the earlier package's reports. This log does not separate host load from the bundle.
- Whether long-prompt TTFT matches the earlier package (see above).
- The second boot time. The first boot here compiled kernels from scratch (about 15.5 minutes). A boot with warm caches was not measured.
- A whole bring-up with the new coder. The earlier package ran the Clef bring-up to the operator bundle; the new one has run
  the role-fit test and the speed samples only.
- Whether bfp4 routed experts on two dies change tool-call reliability over many runs (the earlier package has the same
  setting, so this is not new, and is still not measured).
- Prefix caching in the agent loop. The loop resends a growing conversation, which the card's measurements suggest
  helps first-token time. It was not measured with the loop here, and a cached state is a reason to watch for
  differences between a first and a repeated prompt.

## Decision

Role-fit passed and every speed sample was above the earlier package's recorded numbers, so the default coder in
`config/*.qb2-coder-next.toml` and `orchard/setup_machine.py` is now `raahemnabeel/qwen3-coder-next`. The earlier package
stays as `--coder coder-next-v1` with its own config templates. A machine that already has `config/bringup.toml`
keeps what it has: setup never overwrites a config.

## Clean-up

The container `tt-model-qwen3-coder-next-p300` was stopped with `tt-model stop raahemnabeel/qwen3-coder-next --profile p300`
(clean shutdown, 1.4 s). Lease `e1328b` was reaped by gozer at the same moment, and the board went to the queued
job. The other board's container and lease were not touched.
