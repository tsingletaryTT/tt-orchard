# 2026-10-29: restage the Qwen3.5-35B-A3B package from main and boot it

Operator request: the earlier staged copy had one `run.sh` message edited after its boot check. Restage from current main, run the gate, and boot the restaged copy cold on board 1.
Nothing was uploaded or published, and nothing was merged.

## Restage and gate

- Source: `package_src/` and `docs/run-logs/2026-10-29-qwen35-package-data/` on main at `d4efd3e` (PR 246 merged).
- Staged with `orchard.qwen35_package.stage_package` into a new directory under the operator's package folder (`qwen35-35b-a3b-recheck`): 24 files.
- `package_problems` returned an empty list (no problems).
- `tests/test_qwen35_package.py`: 98 passed.

## Boot check (board 1, chips 2 and 3)

- Weights: `fetch_weights.py` from the staged package, 693 tensors, 69,321,331,008 bytes in `model.safetensors`, revision pinned in the package.
- Lease: queued behind another agent's lease on board 1. `gozer wait` granted a lease that belongs to the waiting process, which does not live for the session, so that lease was released and a new one taken with `gozer acquire --owner-pid` on the driver script's pid. The driver held the lease through the boot and released it at the end (release reset the chips).
  The lease granted `TT_VISIBLE_DEVICES` for the two PCI addresses of board 1; `run.sh` mapped them to `/dev/tenstorrent/2` and `/dev/tenstorrent/3`. Chips 0 and 1 (the coder's lease) were not touched.
- Port: 8791. Port 8790 was already bound by another process on the machine, so the first attempt exited at once with "address already in use" before the model loaded; that attempt used no chip time beyond the lease and its release. Ports 8000, 8100 to 8103, 8780, 8011 and 8012 were not touched.
- Cold boot: the work directory was empty (no kernel cache, no tensor cache). Container start to `/v1/models` answering: 471 s. The image-id, loader and patch checks in `run.sh` passed. The first cold boot in `2026-10-29-qwen35-package.md` took 727 s; this one is one more single sample, taken on a different host load, and the two are not a controlled comparison.
- One chat completion through `/v1/chat/completions` with `chat_template_kwargs {"enable_thinking": false}`, temperature 0, `max_tokens` 96, prompt "Name the three primary colors.": answered in 0.87 s with a normal, fluent answer (red, yellow and blue for pigment, then the additive set); `finish_reason` was `length` because the answer was longer than 96 tokens. It did not open a thinking section.
- Not measured here: decode speed, quality prompts, warm restart. Those figures remain the ones in `2026-10-29-qwen35-package.md`.

## Things to know

- Nothing was published. The staged directory and the 65 GB model directory stay local.
