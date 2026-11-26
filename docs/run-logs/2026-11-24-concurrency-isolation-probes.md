# Concurrency and isolation probes for a served model (2026-11-24)

The Lottolabs releases for Blackhole list, for each batched model, checks that `orchard/served_probes.py` did not have. The cards' own words are
the source here; nothing below was measured on a chip.

| Check on the cards | What the card reports | Probe added |
| --- | --- | --- |
| Each concurrent request equals the same request alone | Qwen3.8-27B, LFM3, LFM2.5 and Inkling-2-Mini: every one of 16 concurrent requests returns the tokens it returns alone (LFM3: 362 of 362 replies). Qwen3.6 batched says its greedy output at 4, 8 and 16 users differs from the same request alone (kernels round differently by batch composition). | `probe_concurrent_equals_alone` |
| A-B-A with 15 running streams | The Inkling and LFM3 cards run a request alone, between running streams, and alone again. | `probe_aba_isolation` |
| Cancel mid-decode | One of 16 streams is cut and the other 15 stay exact; a cancel mid-prefill and mid-decode leaves the next request exact. | `probe_cancel_mid_decode` |
| Speculation on equals off | Gemma 4, Qwen3.6, Qwen3.8-27B, Inkling and LFM3 state that speculative and plain greedy output are token-identical (an Inkling sentence about the Qwen3.8-27B card contradicts the Qwen3.8-27B card's own statement). | `probe_spec_equals_plain` |
| A streamed tool call never arrives with `{}` arguments | LFM3 found that with speculation a whole tool call arrives in one to three events and the incremental parser sent 492 of 523 streamed calls with `{}` arguments. | `probe_streamed_tool_call` |
| vLLM answers 400 to `min_p`, `logit_bias` and `min_tokens` under speculation | Qwen3.6, Qwen3.8-27B and Gemma 4 cards. | `probe_sampling_params_under_speculation` (lists each status; a 400 is the documented answer) |

`served_probes.py` already had `probe_batching`, which compares the aggregate rate of a burst with the single-stream rate. It sends concurrent requests
and does not compare what they return, so a server that mixes up two requests' replies passes it.

## How each probe avoids a false verdict

Every probe has a control. A server that does not repeat itself when one prompt is asked alone twice cannot be judged under load, so the verdict is
`not_checked`. A server that gives every prompt the same reply cannot show crosstalk, so `concurrent_equals_alone` is `not_checked` for it. The
speculation probe refuses two identical request options, because it would compare the server with itself. A fail names the requests that changed.

## What is wired and what is not

`tt-orchard can boot` runs `concurrent_equals_alone`, `aba_isolation` and `cancel_mid_decode` when the can declares `max_num_seqs` above 1, as advisory
rows. The other three need something the can does not declare (a speculation switch, a tool schema the model was trained for, a speculative server),
so they are library functions for the first chip run and for a caller that knows what to send. All six send greedy requests, bounded in count and in tokens;
the stream probes cut their own connection.

## Owed

No probe has run against a real server. The first chip run should serve one batched bundle under a lease and run all six, record the replies that
differ (if any) and say which kernels the card names for that model.
