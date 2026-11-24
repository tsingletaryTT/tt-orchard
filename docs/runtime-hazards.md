# Runtime hazards the Lottolabs cards report

`orchard/hazards.py` predicts hazards from a model's config. The Lottolabs releases for Blackhole (read on 2026-11-24) report four failures that no config
number predicts. They come from running a port, so they are listed here and in `RUNTIME_HAZARD_TABLE` (`python3 -m orchard.hazards --runtime`). Every
row is a card claim that this repository has not reproduced. Each row says which card and section it comes from, what a stage should run to look for it,
and what orchard checks today.

| Row | What the card reports | Card and section |
| --- | --- | --- |
| `trace_inputs_overwritten_by_later_capture` | A tensor allocated after a trace was captured can sit inside the released intermediates of the earlier trace, so each replay overwrites it. LFM3: a page table read garbage block ids (wrong tokens, then NaN logits). Qwen3.8-27B: vision rows were overwritten and a 16,384-token image produced garbage. | LFM3-1.7B, "s13 trace page-table bug"; Qwen3.8-27B, "Fixed: vision rows" |
| `lm_head_noc_barrier_hang` | A device hang about once in 20,000 decode steps; the LM head waits in a NoC atomic barrier. The card infers the cause and does not prove it. The fix drains NoC transactions before kernel exit. | Qwen3.8-27B, "A rare device hang is addressed" |
| `host_read_during_trace_capture` | `TT_FATAL: Reads are not supported during trace capture` when a lookup table that the traced step reads is on the host. The fix keeps the table on the device. | Qwen3.6-35B-A3B, "MTP-off fix" |
| `tensor_cache_identity_is_name_and_shape` | The converted-tensor cache is keyed by tensor name and shape, so a cache from another checkpoint serves wrong weights. The launcher writes an identity file (checkpoint manifest hash and image id) and refuses a mismatch. | Qwen3.6-35B-A3B, "Cache safety"; Qwen3.8-27B |

## What orchard has today

- Trace captures and replays are counted by the stage 2 port check (issue 547). Nothing checks tensors that are allocated after a capture, and no stage runs a
  decode of 20,000 steps.
- The tensor cache carries a `.orchard-model` marker that names the model (`orchard/caches.py`). It does not record the checkpoint revision or the serving
  image, so a cache from an older checkpoint of the same model passes it. The card's identity (manifest hash and image id) is stronger.

## The sampler row

The existing row `vocab_per_die_above_65536` says a sampler index wraps at 65,536 columns per die. Two Lottolabs cards (Inkling-2-Mini, 201,024 columns;
Qwen3.6-35B-A3B, 248,320) run on one die with a custom top-k scan. The row's text now says it describes the stock sampler op and names those two cards as
the reason it is not a hardware limit. That is also a card claim.
