# The text-model runtime route

This route brings up a text model whose port is a set of scripts in this repository (a "text full port"). It is the harness route for a model that no installed
bundle, container package or tt-transformers entry serves. MiniCPM3-4B (`scripts/minicpm3`) is the first model on it. Issues 61, 90, 206 and 214 describe the
need; issue 143 is the reason no stage boots the coder.

## Starting a run

```
tt-orchard bringup openbmb/MiniCPM3-4B
```

The preflight base row finds the model in the registry (`orchard/text_runtime.py`, `REGISTRY`) and passes the input `text_runtime=minicpm3` to the supervisor.
The input is recorded in the ledger's `run_start`, so a resume keeps the choice. `--input text_decode_tokens=N`, `--input text_ppl_tokens=N` and
`--input text_ppl_parquet=FILE` change the decode length, the perplexity window and the WikiText-2 test file. Without the last one the supervisor looks for
`datasets--Salesforce--wikitext` in the Hugging Face cache. `ORCHARD_TEXT_PYTHON` names the interpreter (default `<operator home>/.tenstorrent-venv/bin/python`);
it needs torch, transformers, safetensors, numpy and ttnn.

## The stages

| Stage | What runs | Hardware | Gate |
|---|---|---|---|
| 0 | triage from the model's own files: configuration check by the adapter, card license, weight files | none | `gate_text_triage` |
| 1 | CPU reference: float32 run (chooses the greedy path), two bfloat16 runs on that path (the floor), float32 perplexity of a fixed window | none | `text_stages.reference_problems` |
| 2 | for each variant, a teacher-forced chip run and the chip's perplexity on the same window, compared under the rule below | one lease, one board | `text_stages.chip_problems` |
| 3 to 5 | skipped with a reason | | |
| 6 | for each variant, one warm-up run and three measured runs in one process (`docs/benchmark-rules.md`), and the share of the decode floor | one lease | `text_stages.bench_problems` |
| 7 | skipped; the package is built from the run's results by the package builder | | |
| 8 | the operator bundle, with the section "Text model on a chip" | none | `gate_bundle` |

None of these stages starts the local coder (`needs_coder=False`), so the coder keeps its board. Stages 2 and 6 take a lease for one board through the
supervisor's own hardware phase and give it back. Each gate recomputes every number from the recorded arrays and files and refuses a result file that
disagrees.

## The bar for a chip run set from CPU floors

`orchard/text_bars.py` states the rule in its module text. In short:

- The floor at an array is the bfloat16 CPU run's relative L2 error against the float32 reference at that array.
- The tolerance is 8 times the floor (at least 1e-6). An array leaves when the chip run's error is above its tolerance.
- The stage needs no array to leave, teacher-forced agreement of at least 90 percent, and a perplexity within 2 percent of the reference's.

The margin of 8 is a choice. MiniCPM3-4B's chip error was 2.8 to 6 times the difference between two bfloat16 CPU implementations (issue 214), and Granite-4.0-H-1B's
was 1.0 to 2.7 times the bfloat16 CPU error (issue 206). The rule cannot show what the chip's own arithmetic adds, a defect smaller than the tolerance, or the
cause of an array that leaves. It says where the first one is.

## Plugging in another text model (the adapter boundary)

Add a subclass of `TextRuntime` in `orchard/text_runtime.py` and register it in `REGISTRY`. The module text lists what it supplies: facts, the weights path,
five commands (reference, reference perplexity, chip agreement, chip perplexity, benchmark), and the serving command for a package. The harness never imports the
runtime's scripts; it starts them as child processes. The contract for each command's output files (array names, JSON keys) is in `orchard/text_reference.py`
and `orchard/text_runtime.py`. `text_runtime.adapter_problems(adapter, context)` checks the boundary without running anything; a new adapter's tests should call
it, as `tests/test_text_runtime.py` does for MiniCPM3.

A runtime needs these pieces in its scripts:

1. A reference script that records `layer_NN` (every block output, all prompt positions), `prefill_logits`, `dec_logits_K` and `tokens` for a float32 run and
   for bfloat16 runs on the same token path, through `orchard.text_reference.make_recorder`.
2. A chip script that records the same names while following the reference's token path, and a benchmark mode that writes the `orchard.resident` record.
3. A perplexity script with a CPU mode and a chip mode over the same text window.

A model whose layers are not a plain stack of `layer_NN` outputs (for example a hybrid with recurrent blocks) records the same names for each block and adds
`layer_NN.mixer` for the operation inside it; the rule applies to every array both runs hold.
