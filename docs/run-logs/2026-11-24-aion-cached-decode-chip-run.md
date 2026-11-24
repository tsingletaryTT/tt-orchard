# Cached decode of the Aion generated port on a chip (2026-11-24, issue 805)

The cached decode of the generated port (PR 806, tested before only on the CPU stand-in of ttnn) ran on Blackhole chips for the first time. This is a small measurement: 24 tokens per run, one run per prompt and port, on a two-chip mesh.

## What ran

| Item | Value |
| --- | --- |
| Model | Aion-1.0-Instruct-2.3B, the same checkpoint revision as the published bundle |
| Baseline port | the port shipped in the published Aion bundle (whole-sequence decode: every token runs the full prompt again) |
| Cached port | regenerated from main with `python3 -m orchard.port_template generate`, using the effective construction printed by `effective_construction.py` for the checkpoint. The plan reports `attention_cached_decode` true, and the file has `decode_step`, `_attention_step` and `_rope_row`. |
| Runtime | the bundle's own venv (PyPI `ttnn` 0.80.0) and its environment, one two-chip board, a 1x2 mesh with fabric 1D. The bundle serves on four chips; this run did not. The port replicates every weight on each chip, so the per-token work is the same on two chips. |
| Method | `port.prefill(prompt)`, then 24 greedy tokens with `port.decode_step`, timed in one process per port. Both ports ran against the same kernel cache directory, the baseline first. |
| Prompts | `short`: 10 tokens (a one-sentence question). `medium`: 238 tokens (a paragraph repeated three times, then a request to summarise). |

## Result

| Prompt | Port | Prefill | Decode, 23 steps | Decode rate |
| --- | --- | --- | --- | --- |
| short, 10 tokens | whole-sequence | 7.21 s | 4.84 s | 4.75 tok/s |
| short, 10 tokens | cached | 0.40 s | 7.31 s | 3.15 tok/s |
| medium, 238 tokens | whole-sequence | 5.04 s | 29.83 s | 0.77 tok/s |
| medium, 238 tokens | cached | 0.27 s | 3.59 s | 6.40 tok/s |

The 24 greedy token ids were identical between the two ports on both prompts (24 of 24 each), so on these two prompts the cached path gives the same text as the whole-sequence path. The text of both prompts: "... The sky appears blue because molecules in Earth's atmosphere scatter shorter wavelengths of sunlight more than longer ones, and blue light is" and "The printing press, invented by Johannes Gutenberg in the 15th century, revolutionized communication by making books cheaper, more".

On the 238-token prompt the cached port decodes 8.3 times faster. On the 10-token prompt it measured slower (3.15 against 4.75 tok/s).

## Caveats

- One run per cell and 24 tokens. There is no spread, and the figures are not medians. Differences of 20 percent between rows should not be read as real.
- The first call of each kernel shape includes its compile. The baseline ran first, with a cold kernel cache (the log reports no cache hits). The cached port's `decode_step` uses new kernel shapes, and its first step on the short prompt compiled them, inside the 7.31 s. That is the most likely reason the short prompt reads slower with the cache. The medium prompt ran second and its 6.40 tok/s is the better estimate of the cached steady state. A second pass over the short prompt, or a warm-up step before the timer, would settle it; this run did not do either.
- The prefill times (7.21 s against 0.40 s on the short prompt) also differ mostly because the baseline compiled its kernels in that window.
- The bundle's card reports 10.91 tok/s median for short prompts on four chips with a warm cache, and 0.94 tok/s at 582 tokens. Those rows used a different harness and are not comparable with these.
- Top-1 agreement with the CPU reference was not measured here, and neither was the 8,192-token passkey check. The only agreement shown is between the two ports.

## A hang on the first attempt, and the fix (issue 813)

The first cached run did not return. The process spent 32 minutes at 100 percent CPU with 461 MB resident, with its last log line "Fabric initialized on 2 devices", so it never finished loading the weights. The generated port reads each tensor with `f.get_tensor(name)`, which returns a tensor backed by the memory-mapped file, and `ttnn.from_torch` on such a tensor does not return on the PyPI `ttnn` 0.80.0 wheel (issue 813). The Aion bundle's port has `.clone()` there (its packaging change). The scratch copy used for the second run changed only that line (line 86):

```
return f.get_tensor(name).clone()
```

With the clone the load took 4.4 s and the run completed.

## What this means for the merged Aion port

- The generator in `orchard/port_template.py` still writes `return f.get_tensor(name)` in its loader. Any port it regenerates for the Aion bundle, cached or not, hangs on this wheel until the clone is added by hand. The better place for the fix is the generator's loader template, so that a regenerated port carries it.
- The published Aion bundle still holds the whole-sequence port. Putting the cached path into the bundle needs the regenerated port with the clone fix, a repeat of this measurement with a warm cache and more runs, a top-1 and passkey check on the bundle's four chips, and the operator's go-ahead for any republish. Nothing was published or changed in the bundle.
