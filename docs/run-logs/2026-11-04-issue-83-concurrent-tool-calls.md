# Issue 83: tool calls damaged when requests overlap, reproduced on the safe configuration (2026-11-04)

Issue 83 reports that a model served with the drafter off and sampling on the host (the safe configuration) returns damaged tool calls when four requests run at the same time, and not when they run one at a time. This run repeats the observation on a different model in the same serving configuration, finds the trigger, and then isolates the cause (see "Isolation experiments"). The full numbers are also in a comment on issue 83.

## Setup

- One 2-chip Blackhole board, leased through gozer for the length of the run and released afterwards (all four chips free at the end).
- The published package `episod/sensenova-6.8-code2video-p300`: a 27B model, drafter off, sampling on the host, vLLM with the `qwen3_coder` tool-call parser, `max_num_seqs` 4, `--no-async-scheduling`, context 262,144. It serves with the TT model code that the AREX-2 bundle uses, so the serving path is the same. The weights are a different fine-tune.
- Thinking off (`chat_template_kwargs: {"enable_thinking": false}`), 4,096-token limit, one request per prompt, no streaming.
- The server was started from the warm tensor cache and answered after 3 minutes.

## Test

Eight tasks, each with two or three tool definitions and a request for exactly one call with arguments that can be checked exactly: weather with a unit, the area of a rectangle (12.5 by 8), an email with a recipient, subject and body, a flight search (two airports, a date, a passenger count), a table booking, a thermostat value (21.5), a funds transfer (two account ids, 250.75, a currency) and a task with a title, a due date, a priority and a list of two tags. Each condition sent 40 requests (each task five times), with the same prompts in the same order. Every request and reply was kept.

A reply is **damaged** when it hit the token limit, has no tool call, names a function that is not a defined tool, has arguments that are not valid JSON, has a marker such as `<tool_call` or `<parameter` in the arguments or the text, or leaves a required argument out. A reply is **not exact** when the structure is sound but the function or a value is wrong.

## Results

| Condition | Requests in flight | Temperature | Exact | Damaged | Not exact only |
|---|---|---|---|---|---|
| one at a time | 1 | 0 | 40 | 0 | 0 |
| one at a time | 1 | 0.7, seeded | 40 | 0 | 0 |
| continuous | 2 | 0 | 29 | 4 | 7 |
| continuous | 3 | 0 | 17 | 16 | 7 |
| continuous | 4 | 0 | 17 | 19 | 4 |
| continuous | 4 | 0.7, seeded | 17 | 20 | 3 |
| groups of four started together, each group finished before the next | 4 | 0 | 40 | 0 | 0 |

Kinds of damage at four in flight, temperature 0 (a reply can have more than one): arguments left out 15, a marker inside a string 4, a function name with junk pasted on 2. At temperature 0.7: arguments left out 15, markers 8.

Paired by prompt, one at a time against four in flight: 23 prompts exact only in the one-at-a-time run, none exact only at four in flight (exact sign test p = 2.4e-7), at temperature 0 and again at 0.7 with seeds. At two against three in flight: 14 against 2 (p = 0.004).

Examples of damaged replies at four in flight, each with the good reply for the same prompt at one in flight:

| Task | One in flight | Four in flight |
|---|---|---|
| email | `{"to": "ana.ruiz@example.com", "subject": "Q3 budget review", "body": "Numbers attached, please confirm by Friday."}` | `{"to": "ana.ruiz@example.com", "subject": "Q3 budget review"}` (body left out) |
| flights | `{"origin": "SFO", "destination": "JFK", "date": "2026-12-03", "passengers": 2}` | `{"origin": "SFO", "destination": "JFK", "date": "202026-12-03\n</parameter name=\"passengers\""}` |
| transfer | all four arguments | `{"from_account": "CHK-1042"}` |

## What the runs show

1. **It is the same effect as the issue describes**, on a second model, with the same kinds of damage (cut arguments, a stray `<parameter` marker, a name with junk pasted on).
2. **It grows with concurrency:** 0, 4, 16 and 19 damaged replies at 1, 2, 3 and 4 requests in flight.
3. **Four sequences decoding together is not enough.** Groups of four started together and finished together gave 40 of 40 exact and call text identical to the one-at-a-time run.
4. **The trigger tracks a request arriving while others decode.** Counting the requests that started during a given request's decode (after a 1 s allowance for its own prefill, measured from the client side, so approximate): at temperature 0, requests with none were damaged 0 of 25 times across 2, 3 and 4 in flight; requests with one or more were damaged 39 of 95 times. At temperature 0.7 and four in flight: 1 of 5 against 19 of 35.
5. **No cross-request contamination.** None of the 29 wrong values equals a value that belongs to another task. Each damaged reply is its own request's output, cut or garbled.
6. **The server log shows no errors.** One start-up warning: "TT model Qwen36DFlashForCausalLM does not advertise decode_input_update_contract >= 1; preserving its legacy reset_batch reload behavior. Async decode correctness is not guaranteed until the model adapter implements the explicit contract." Async scheduling is off in this package. The isolation experiments below found the cause elsewhere (the fused conv history rebuild), so this warning is not the cause.
7. **An observation (no finding follows from it):** at one in flight, temperature 0.7 with seeds gave exactly the same call text as temperature 0 in all 40 requests. The tool-call tokens are probably high-confidence. The run did not check whether the temperature is applied on the host sampling path.

## Isolation experiments (second session, same day)

The first session found the trigger (a request arriving while others decode). This session changed one thing at a time on the same package, one 2-chip board, port 8014 only, to find the cause. Each row is a fresh server start (5 starts). Every probe is the same eight tasks, 40 requests, at temperature 0 unless stated. A second probe sends the same prompts to the raw completions endpoint with the model's own chat template and records the streamed text, so the tool-call parser is out of the path.

| # | Change from the default | Tool-call probe, 4 in flight (exact / damaged / not exact only) | Raw text at 4 in flight identical to 1 in flight |
|---|---|---|---|
| baseline | none (first session) | 17 / 19 / 4 | not measured |
| E1 | `QWEN36_PLAIN_GDN_SLOT_FAST=0` (host round trip for the slot write) | 17 / 19 / 4, all 40 call texts identical to the baseline | 12 of 40 |
| E2 | `QWEN36_GDN_DECODE_FUSED=0` (composite decode) | 40 / 0 / 0, and again 40 / 0 / 0 | 40 of 40 |
| E3 | `QWEN36_GDN_DECODE_FUSED=1` (fused recurrence, composite conv) | 40 / 0 / 0 | 40 of 40 |
| E4 | none (control, with `QWEN36_GDN_HIST_DEBUG=1` logging only) | 17 / 19 / 4 | 12 of 40 |
| E5 | test patch, default `QWEN36_GDN_DECODE_FUSED=2`: an admission repacks only its own slot of the conv history | 40 / 0 / 0, and again 40 / 0 / 0 | 40 of 40, then 80 of 80 on a second run |

More E5 numbers: three in flight 40 / 0 / 0; four in flight at temperature 0.7 40 / 0 / 0; one in flight at 0.7 40 / 0 / 0. E5 with one request in flight gave the same raw text as E2 with one request in flight in 40 of 40, and E2's text with one in flight equals E1's for all eight tasks, so the composite and fused decode agree when nothing overlaps.

What the experiments show:

1. **The damage is in the model's token stream.** At four in flight, 28 of 40 raw texts differ from the same prompt at one in flight (16 with inserted text, 9 with deleted text, 3 with replaced text). The parser only reports what it is given. One in flight gave one text per task across five repetitions.
2. **At temperature 0 the damage is repeatable.** E1 and E4 are different server starts and gave the same 17 / 19 / 4 split. All 40 call texts in E1 equal the baseline's.
3. **The damage begins 0 to 3 tokens after the prefill step of an arrival** (median 0, 18 victims, from the server log's "Prefilling N user(s) into slots" lines joined with each victim's own stream stalls). A victim that has produced fewer than 12 tokens was hurt in 3 of 12 cases, from 12 to 40 tokens in 3 of 7, and from 40 tokens on in 13 of 15. The damaged text is fluent: it restarts a phrase, repeats or skips a few tokens, or closes a string early.
4. **The slot-write path is not the cause.** The fast device write and the host round trip (E1) give identical output.
5. **The fused decode with a packed conv history is the cause.** Turning it off (E2) or keeping the fused recurrence but using the composite conv (E3) removes the damage in 120 of 120 requests across three probes. `QWEN36_GDN_DECODE_FUSED=2` is the model code's default.
6. **The mechanism, from the code and the log.** With `QWEN36_GDN_DECODE_FUSED=2` the decode op advances the conv window in a packed history buffer (`conv_hist_packed`) and does not update the per-tap `conv_states` (a comment in `_remap_conv_hist_packed` in `models/demos/blackhole/qwen36/tt/gdn/tp.py` says so: "conv_states go stale after the first step"). An admission runs its prefill in a B=1 scratch whose capture path marks the packed history invalid, then `write_slot` finds the history invalid and rebuilds the packed history of every slot from `conv_states` (`_ensure_conv_hist_packed`). For each live row the taps still hold the window from that row's own admission, so the rebuild puts the row's conv window back to that older state. The next few decode steps read a wrong window and the recurrent state absorbs the change. E4's log shows one full rebuild (48 layer writes) for each of the 72 prefill events during the probes. The per-slot repack ran only in addition, for the second and later users of the multi-user events (6 groups of 48 layer writes). This fits the earlier observations: groups of four started together have no live row to roll back (40 of 40 exact), the damage grows with the number of arrivals, and a row that has generated more tokens has a window that differs more from the admission-time one.
7. **E5 tests the mechanism directly.** The patch (a `sitecustomize.py` on the server's `PYTHONPATH`, loaded only when `I83_PATCH=1`; nothing in the model bundle or its virtual environment is edited) marks the history valid before `write_slot` when the buffer was built before, so the existing per-slot repack runs and live rows keep their history. Damage fell from 19 of 40 to 0 of 40 with the fused decode still on, at 3 and 4 in flight and at temperature 0.7, and the raw text at four in flight matched one in flight in 120 of 120 requests.

What stays unknown:

- The patch is a test and is not offered as the fix. It treats "built once" as "valid for live rows". The cases where the invalid flag is set for a good reason (the speculative join with `sync_hist=False`, a reset, a restore) were not exercised. The real fix belongs in the model code that the bundle ships.
- The decode speed with the patch was not measured separately. Median request latency at four in flight was 6.0 s with the patch and 5.1 s in the damaged control, and the damaged runs produced 3,471 completion tokens against 2,475, so the numbers do not compare.
- The original issue's model (AREX-2 weights) was not run. It uses the same model code and the same serving configuration, so the same cause is likely, but that is an inference.
- With the drafter on, the speculative path leaves the packed history invalid on purpose. Whether the same rollback appears there was not tested.
- Why the hurt rate rises with the number of tokens a victim has produced (item 3) is only partly explained: the stale window differs more from the true one as generation moves away from the prompt, but this was not measured.
- `QWEN36_PLAIN_GDN_SLOT_FAST=0` (E1) was set the same way as the other variables. The engine's `/proc` environment view does not list every variable the launcher exports, so its presence in the engine was not read directly. Behaviour confirms the settings for E2 and E3 (the damage disappears, and the composite decode is slower); E1 changed nothing visible.

Evidence: the experiment scripts, probe outputs and server logs were kept in the session's scratch directory (not committed). The test patch is outlined below.

```python
# sitecustomize.py (loaded with I83_PATCH=1): per-slot history repack on admission
# wraps TPGatedDeltaNet.write_slot: when sync_hist is true, the fused conv is on, the packed buffer exists and was
# built before, set _hist_packed_valid = True first, so write_slot takes the per-slot repack path.
# wraps TPGatedDeltaNet._ensure_conv_hist_packed: record that the buffer has been built.
```

## Limits

- One model, one board, 40 requests per condition, eight tasks with short answers (40 to 100 tokens). The original evidence used 300 BFCL items with longer answers; the rate here is not comparable.
- The arrival measure comes from client timestamps and a fixed 1 s allowance for prefill.
- The damage definition above was written before the runs and not changed after.
- Not tested: the graft configuration (drafter on, device sampling) at one and four in flight; the base model without the fine-tune; the original issue's AREX-2 weights.

## What to do next

- Add the concurrent probe to the harness: eight tasks, four in flight against one at a time, compared by prompt. It takes about 5 minutes on a warm server and would show this in a stage 2 or stage 4 run, which send one request at a time today.
- Fix the cause in the TT model code: keep the packed conv history of live rows valid across an admission (or write it back to `conv_states` before a rebuild). Re-run this probe against the fix: the target is 40 of 40 exact at four in flight, with raw text identical to one in flight.
- Until the fix ships, a server can run with `QWEN36_GDN_DECODE_FUSED=0` or `1`, which removed the damage here. Its decode speed cost was not measured.
- Run the untested cases above (drafter on, AREX-2 weights); each is a restart of the server and about 10 minutes.
- Until the cause is fixed, a card for a model served this way should say that tool-call quality drops when requests overlap.
