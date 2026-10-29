# MiniCPM3-4B on one Blackhole chip (overnight 2, 2026-10-29)

Run log with all numbers: `docs/run-logs/2026-10-29-overnight-minicpm3-4b.md`. Evidence: `docs/run-logs/evidence/2026-10-29-minicpm3/`.

Environment variables used by the scripts (no machine paths are in the code):

- `MINICPM3_DIR`: a directory with the model's `config.json` and tokenizer files (a download of `openbmb/MiniCPM3-4B`).
- `MINICPM3_WEIGHTS`: the weights as one safetensors file. The repository ships `pytorch_model.bin`; convert it once with
  `torch.load(..., weights_only=True)` and `safetensors.torch.save_file`.
- `MINICPM3_REF`: for `exp_mlp14.py`, a recorded float32 reference run directory.

| file | what it does | needs a chip |
|---|---|---|
| `ref_model.py` | plain-torch CPU reference written from the model's source | no |
| `check_ref.py` | greedy text from the reference; `--native` compares with the transformers `minicpm3` class (needs a copy of the config with `rope_parameters`) | no |
| `make_reference.py` | records float32, bfloat16 and native runs in the `audio_reference` layout for `orchard.first_divergence` | no |
| `chip_model.py` | the chip port: absorbed multi-latent attention with `flash_multi_latent_attention_decode`, trace, sharded options | yes |
| `run_chip.py` | `agree` records a teacher-forced chip run; `bench` times decode (warm-up plus repeats) and writes a record with the floor share | yes |
| `summarize.py` | one line per recorded chip run against the reference | no |
| `exp_*.py`, `time_trace.py`, `prof_step.py` | the single-op and timing probes behind the numbers in the run log | yes |

Typical use, under a gozer lease: `python run_chip.py agree --opt 3 --upd paged --ref REF/float32 --out RUN` then `python run_chip.py bench --opt 3 --upd paged --trace --out record.json`.
