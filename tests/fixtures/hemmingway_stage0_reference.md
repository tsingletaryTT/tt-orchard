# Stage 0 reference answer, written by hand on 2026-10-02 (not by the harness)

Purpose: a gold answer. When the orchard harness runs stage 0 on this model, compare its delta list with this file.
A harness result that misses an item here, or states one that is wrong, is a stage 0 defect.

Model: `Altworld/Hemmingway-1` (spelled with two m's; the one-m name does not exist), revision
`1a5f363a3dd2d1cc456c28b8abbb403b9555efaf`, license CC BY-NC 4.0 (commercial use by agreement), not gated.
Nearest supported model: `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (already running on
this box as the DFlash2 package, 1 chip and 2 chips).

Differences from the nearest supported model:
1. Text config: identical (64 layers, hidden 5120, 24 heads / 4 kv heads, linear attention every 4th layer is full
   attention, vocab 248320, context 262144). Only `architectures` (`Qwen3_5ForCausalLM`) and `transformers_version` differ.
2. Tensors: 866 text tensors, identical names (after dropping the base's `model.language_model.` prefix), shapes and
   dtype (all BF16). The base has 333 more tensors, all vision. The fine-tune has no vision tower.
3. Tensor name prefix: the fine-tune uses `model.*`, the base uses `model.language_model.*`. The runtime's
   `remap_qwen36_state_dict` accepts both.
4. Tokenizer: same vocab (248044), same merges, same 33 added tokens, same chat template. The pre-tokenizer pattern
   drops `\p{M}` (combining marks) from the letter classes, so ids differ only around combining marks (decomposed accents,
   Indic and Thai vowel signs). 5 of 212 test strings differed, all random Unicode with combining marks.
5. Files: 12 weight shards (54.7 GB total with `model-mtp.safetensors`, 0.85 GB). The index lists 15 `mtp.*` keys.
6. Generation config: same sampling defaults as the base (temperature 1.0, top_k 20, top_p 0.95).

Expected path: weights-only swap. No new model code. Run the existing DFlash2 recipe with `HF_MODEL` pointed at this
repo, 2 chips first, then 1 chip.

Hazards:
- The base's converted tensor cache (`~/.cache/qwen36-src-build/cache/tensors-tp2`) must not be reused. Use an empty cache.
  A cache keyed only by layer name would serve the base weights silently.
- The DFlash2 drafter (`incoai/Qwen3.8-27B-DFlash2`) was trained against the base, so acceptance rate may drop. Measure it.
- Root disk is 98% full; keep weights and caches under `/mnt/bonus/models/hemmingway-1`.
