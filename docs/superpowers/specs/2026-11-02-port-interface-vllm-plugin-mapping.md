# Full-port interface and the TT vLLM plugin's generator contract

Date: 2026-11-02. Issue 363. Status: mapping written, one check added to stage 2.

Sources read: the decode reload contract in tenstorrent/skills PR 34 (open on 2026-11-02, branch `tcheda/bringup-decode-reload-contract`,
files `plugins/tt-model-bringup/skills/vllm-integration/references/decode-reload-contract.md` and
`plugins/tt-model-bringup/runtime/readiness_check/contract_vllm.py`). The plugin's own specification
(`docs/DECODE_RELOAD_CONTRACT.md` in tenstorrent/vllm-tt-plugin) is the source of truth and was not read here. Re-read this note against it
when PR 34 merges.

## 1. The two interfaces

The stage 2 port (`stages/2/port/port_model.py`, checked by `orchard/skills/full-port-templates/port_check.py`) has four functions:
`load(model_dir, mesh_device)`, `prefill(model, ids)`, `decode_step(model, token_id, position)` and `close(model)`. It serves one request,
owns its KV cache, returns logits to the host, and gets the token and the position on every decode step.

The plugin's adapter (`VllmGeneratorAdapter` in `contract_vllm.py`) has `initialize_vllm_model`, `get_max_tokens_all_users`,
`allocate_kv_cache`, `warmup_model_prefill`, `warmup_model_decode`, `prefill_forward`, `decode_forward`, `read_decode_output` and
`process_decode_output_host`. It serves a batch, uses a KV cache that vLLM allocates and passes in with a page table, and can sample on
the device.

## 2. Mapping

| Port interface | Plugin contract | What a stage 2 port already satisfies | What serving adds |
| --- | --- | --- | --- |
| `load(model_dir, mesh_device)` | `initialize_vllm_model(hf_config, mesh_device, max_batch_size, ...)` | The weights load onto an open device. The mesh is opened by the caller, as the plugin does. | The adapter reads `hf_config` and the batch size and builds the same model. |
| `prefill(model, ids)` | `prefill_forward(tokens, page_table, kv_cache, prompt_lens, start_pos, empty_slots, ...)` | One prompt in, logits for the next token out. Since issue 363 the check refuses a port whose second prefill of the same prompt differs from its first (section 3). | Batches, `empty_slots`, and a vLLM-owned paged KV cache in place of the port's own cache. |
| `decode_step(model, token_id, position)` | `decode_forward(tokens, start_pos, page_table, kv_cache, reload_inputs, reload_page_table, reload_sampling_params, reset_sampling_state, slot_remap, ...)` | Token and absolute position arrive on every call, so the port keeps no resident token or position state. The teacher-forced run feeds reference ids, not the port's own, so a port that ignored `token_id` and used its own previous output would disagree with the reference after its first wrong token. That equals `reload_inputs=True` on every step, which is the contract's host-sampling and teacher-forcing mode. | The four commands, `slot_remap`, and the page table. A port with full reload on every step can leave `supports_async_decode` false and the plugin then reloads on every decode. |
| logits returned to the host | `read_decode_output`, `process_decode_output_host`, `supports_sample_on_device` | Logits on the host are what host sampling needs. | Device sampling and async readback are optional. They are speed work (stage 6), not a rewrite. |
| `close(model)` | none in the protocol | Resources are released. | None. |

## 3. What a stage 2 port must already satisfy

These are properties the port needs now. A port that lacks one has to be rewritten at serving time.

1. Prefill starts a request afresh. The plugin reuses a decode slot for a new request and expects prefill to initialise that request's
   model state (the contract's `empty_slots` text). The check calls `prefill` twice on the same prompt and the gate refuses a record whose
   two first ids differ (`orchard/full_port.py`, test `tests/test_full_port_prefill_reset.py`). Equal first ids do not show that every
   cache entry was cleared. They catch a port that carries state from one request into the next, which is the failure that costs a rewrite.
2. Decode takes the token and the absolute position as arguments and keeps no copy of either as its own state. The teacher-forced
   run feeds reference ids and the free run feeds the port's own ids, so both input sources are exercised. A port that stashes its own
   state and ignores the arguments is not detected directly; the agreement figure is the only evidence against it.
3. Logits come back to the host. Already required by the check.

## 4. What the check does not look at, and why

- Batch size above one, `empty_slots`, partial prefill beside live requests, and slot remaps. They need the plugin's scheduler. A
  single-request port has no way to show them, and a check that invented them would test the check's own batching. They belong to the
  serving adapter, which is stage 5 of the full-port design (slice 6).
- A paged KV cache that vLLM allocates. The port allocates its own. Moving to the plugin's cache is adapter work whose size depends on the
  model's attention. For a hybrid model (recurrent state beside attention) the contract requires `get_kv_cache_spec`. The port's design note
  should say which layers hold recurrent state, because remapping slot-bound recurrent state is the part of the contract that a port written
  for one request is most likely to get wrong.
- Trace capture, warmup, async readback, device sampling and the sampler's seed and penalty state. These are optional capabilities of a
  version-1 adapter, and they are stage 6 work.
- Whether the tt-metal base class a model inherits implements the contract. The contract text says a merged plugin change does not prove it.
  That check needs the plugin and tt-metal commits of the run and belongs in the serving stage.

## 5. Next steps (not done here)

- When PR 34 merges, re-read the contract text and the plugin's `docs/DECODE_RELOAD_CONTRACT.md` against section 2.
- Slice 6 (issue 329) should boot the port behind the plugin's adapter, not only the harness's chat server, once a port has passed stage 2 on a
  board. The first adapter written for a port is the real test of this mapping.
