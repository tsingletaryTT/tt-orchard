# Sampling for the coder

The supervisor sends every coder request with `temperature` 0 unless the tier sets one. That is greedy decoding. This page records a
paired test of greedy decoding against sampling on the coder, what the model's own settings say, and the tier keys that set sampling.

## The test

Model: Qwen3-Coder-Next on one board, as the coder tier. Task: the tiny Qwen4 full-port rehearsal (`--full-port --full-port-rehearsal`),
stage 2, where the coder writes `port_model.py` from the facts file and the reference excerpts. The outside helper was off in both arms, the
code and the checkpoint were the same, and the auto-retry was off. The arms differ in one setting: the tier's sampling. There was one run per arm.

| | Greedy (tier sets nothing) | Sampled (`temperature = 0.7`, `repetition_penalty = 1.05`) |
|---|---|---|
| Agent steps in stage 2 | 2 | 7 |
| Turns | 37 | 103 |
| Tool calls | 27 `read_file`, 12 `shell`, 0 `write_file` | 48 `read_file`, 38 `shell`, 19 `write_file` |
| First write | none | turn 11 in the two prepare steps, turn 2 to 11 in the others |
| A port checked on the chips | never | within about 6 minutes of stage 2 starting |
| How it ended | blocked by the watchdog 64 seconds after stage 2 started (20 turns, no file written) | aborted at the 30-minute limit, port not passing |

The sampled run did not pass stage 2. Its port was checked on the chips three times, and the failures changed from one round to the next (a
missing tensor name, then two different matrix shape errors), so the coder was fixing real faults. It also produced four replies that ran to
the 16,384-token cap and were flagged as repetition, so a repetition penalty of 1.05 does not remove that failure.

Greedy runs 8 to 13 of the same rehearsal agree with the greedy arm: the coder never wrote a first port in any of them. In the runs where a port
was checked on the chips, an outside helper's draft had put it there.

One run per arm is a small test. It shows that the harness default can stop the coder from writing at all. It does not say that 0.7 and 1.05
are the best values.

## What the model ships

The model's `generation_config.json` says `do_sample` true, `temperature` 1.0, `top_k` 40 and `top_p` 0.95. A client that sends no sampling fields
gets those from a server that reads that file. The harness sent `temperature: 0` on every request, which replaces the model's own default.

## Tier keys

All keys are optional and are sent on every request of an agent step only when the tier sets them. A tier that sets none of them sends exactly
what it sent before (`temperature` 0 and no other sampling field).

| Key | Range | Note |
|---|---|---|
| `temperature` | 0 to 2 | The shipped Coder-Next examples set 0.7 |
| `repetition_penalty` | 0.5 to 2 | The shipped Coder-Next examples set 1.05 |
| `frequency_penalty` | -2 to 2 | |
| `top_p` | 0 to 1 | New with this page |
| `top_k` | 0 to 200 | New with this page. 0 means no limit and is sent when set |

`config/tiers.example.toml`, `config/tiers.qb2-coder-next.toml` and `config/tiers.qb2-coder-next-v1.toml` set `temperature` and
`repetition_penalty` on the `large` and `small` tiers. The CPU stand-in tier keeps the defaults. The ledger's `agent step` entry records the
sampling of each step, including `top_p` and `top_k` when set, and each transcript line records the sampling keys it was sent.

## Limits

- The test used the Coder-Next bundle behind the TT vLLM plugin. The request reached the server without an error with `repetition_penalty`.
  Whether the server applies `repetition_penalty`, and whether it accepts `top_k` as an extra body field, was not checked.
- `top_p` and `top_k` were not part of the test.
- A model that is not Coder-Next may need other values. `config/tiers.qb2-27b.toml` is unchanged.
