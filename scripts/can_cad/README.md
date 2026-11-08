# Two-chip package for the Berkelium CadQuery model

These scripts built, benchmarked and staged a container (v5.1) package of `Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-merged` with a
two-chip `p300` serve profile. The run log is `docs/run-logs/2026-11-08-berkelium-2chip-package.md`. The scripts take every path from the
environment or the command line; none assumes a machine layout.

| Script | What it does |
| --- | --- |
| `serve-2chip.py` | Serves the fine-tune on one board with the base bundle's container (`MESH_DEVICE=P300`), from the recorded 4-chip docker command (`BERKELIUM_SERVE_DIR`). Take the lease first. |
| `bench.py` | Decode speed at batch size one over eight CadQuery prompts, repeated; TTFT; host load; saves every reply. Standard library only. |
| `bench_prefill.py` | Time to the first token for prompts of about 2,000 and 6,700 tokens with no shared prefix. |
| `check_cq.py` | Runs the saved replies through CadQuery (use a venv that has `cadquery`): header line, one python block, imports, `result = ...`, executes, valid solid. |
| `gen_card.py` | Writes `profile.json`, `card.json` and `README.md` from the measured result files. Every number in the card comes from those files. |
| `stage_package.py` | Copies a download of the base bundle (`base-clone/`) to `staged/` and edits the manifest and card. The image is not rebuilt: it stays the base bundle's, byte for byte. Run it with tt-model's interpreter. |
| `serve_staged.py` | Serves `staged/` through tt-model's own `serve_container`, for the boot check. |
| `boot_check.sh` | Cold boot of the staged package, then one benchmark pass, then stop. Run under a lease. |
| `lease_run.sh` | Takes a two-chip gozer lease (waits in the queue when busy), runs one command, releases. |

Set `CAN_CAD_WORK` to the work directory (it holds `staged/`, `base-clone/`, the caches and `logs/`).

Nothing here uploads anything. Publishing is `tt-model push STAGED --repo OWNER/NAME --public` followed by `tt-model publish OWNER/NAME`, run by a person.
