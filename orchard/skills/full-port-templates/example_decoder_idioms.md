# Decoder port idioms that passed the stage 2 check

IDIOMS TO ADAPT, NOT CODE TO COPY. They come from a port that passed the supervisor's check on a tiny test model (the check compares top-1
tokens with the CPU reference and counts device calls). Your model has its own layers, names and sizes: take every number from
`port_facts.json` and `effective_construction.txt`, and take each op from the excerpts. Each `ttnn.` name below has an entry in
`cheat_sheet.md`; check the arguments there.

## What the supervisor's check wants from the module

`load(model_dir, mesh_device)`, `prefill(model, ids)`, `decode_step(model, token_id, position)` and `close(model)`, as in the skeleton.
- After `load`, the check scans everything reachable from the object `load` returns. Weight elements that are host torch or numpy tensors
  must be at most half of the total. A second host copy of an uploaded weight counts against you.
- A decode step must make at least one matmul-family call per layer. A call in a function that `prefill` and `decode_step`
  never reach does not count, and `model(input_ids)` on a Hugging Face model runs host torch.
- `decode_step`'s `position` is the index of `token_id` in the whole sequence, prompt included.

## 1. Choose weights by shape, upload once, transpose in `load`

```python
for name, t in read_tensors(model_dir).items():             # read the checkpoint tensors; a model built from the config has no weights
    if t.dim() == 2 and name.endswith(".weight") and "embed" not in name:
        dev[name] = upload(mesh, t.to(torch.bfloat16).t().contiguous())    # x @ W, so W is [in, out]
    else:
        host[name] = t.float() if t.is_floating_point() else t   # norms, conv kernels, embedding tables as float32; integer tables stay as they are
```
- Select by tensor shape. A list of layer names misses any projection that is not on the list.
- Keep each tensor in one dict only. Do not keep the `read_tensors` result in the model object.
- `ttnn.linear` computes `x @ w`. A torch Linear weight is `[out, in]`.

## 2. One upload helper and one download helper

```python
def upload(mesh, t):
    kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if n_devices(mesh) > 1:
        kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(mesh)
    return ttnn.from_torch(t, **kw)

def download(mesh, t, rows):
    if n_devices(mesh) > 1:
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:rows]   # one copy per chip: keep the first
    return ttnn.to_torch(t)
```
`n_devices` is `int(mesh.get_num_devices())` inside a try block that returns 1 on failure. Use the same pair for weights, activations and
logits. Every `to_torch` of a mesh tensor needs a composer.

## 3. One projection helper that every layer calls

```python
def lin(m, x, name):                                        # x: torch [..., in] -> torch [..., out]
    shape = x.shape
    xt = upload(m.mesh, x.reshape(1, 1, -1, shape[-1]).to(torch.bfloat16).contiguous())
    yt = ttnn.linear(xt, m.dev[name])
    y = download(m.mesh, yt, 1).float()
    ttnn.deallocate(xt); ttnn.deallocate(yt)               # in a try block; a failed free must not stop the run
    return y.reshape(*shape[:-1], -1)
```
Every projection of the model (attention, MLP, gates, mixers, `lm_head`) calls `lin`. The glue between projections (norms, rotary,
activations, softmax, recurrences, indexers) can stay in torch on the host for a first port that passes. Moving glue to the device is
a later optimisation, one op at a time, with the check run after each.

## 4. Prefill and decode share one forward

```python
def prefill(model, ids):
    model.ids = [int(t) for t in ids]
    return forward(model, model.ids)                        # last-position logits, a torch tensor of vocab size

def decode_step(model, token_id, position):
    model.ids = model.ids[:position] + [int(token_id)]
    return forward(model, model.ids)                        # the whole sequence again: no cache, so no cache drift
```
`forward` is a plain function over token ids that calls `lin` for each projection. Re-running the whole sequence is slow and correct,
and it makes the same `ttnn.linear` calls in decode as in prefill. Add a KV cache only after the check passes.

## 5. Activation and norm settings come from the constructor

Where the reference resolves an activation, eps, group size, dilation or padding differently from the class default, the value in
`effective_construction.txt` is the value to use. A gated norm that uses the class default activation gives a wrong output with no error.

## 6. `close`

Deallocate each device weight in a try block, then drop the dicts. The check closes the device itself.

## What is not shown here

A KV cache, a paged cache, tracing, sharding across chips, a router for a mixture-of-experts layer, and a device implementation of the
recurrent or sparse layers. Each of them needs its own evidence.

## 7. The generated port already has these functions: call them, do not rewrite them

<!-- generated-port-functions: _linear _rms_norm _group_rms_norm _upload _upload_ids _to_host _rope_tables _apply_rotary _gated_delta_net _ple _hyper_connection _hyper_connection_inject _C_self_attn -->

A port the generator wrote already holds the blocks it knows, with these exact arguments. The model comes first in each. Names such as `lin` and
`upload` in sections 1 to 4 belong to the passed port those sections come from; in your port the names are the ones below.
- `_linear(model, prefix, x)`: x [1, 1, S, in] to [1, 1, S, out]. `prefix` is the path of the projection, for example
  `model.layers.3.self_attn.q_proj`; it reads `model['dev'][prefix + '.weight']`, so the block's own path fails with a KeyError.
- `_rms_norm(model, prefix, x)`: RMS norm over the last dimension with the weight at `prefix`. It raises for a prefix that is not a plain RMS
  norm; the function that calls it must compute that norm itself. `_group_rms_norm(model, name, x, hc, hidden, eps)` norms each of hc groups.
- `_to_host(model, t)` returns a torch tensor; `_upload(model, t, layout)` puts a torch tensor on the device; `_upload_ids(model, ids)` takes a
  list of ints. A ttnn tensor is not a torch tensor: read it back with `_to_host` before any torch call.
- `_rope_tables(model, n)` returns `(cos, sin)`, each [1, 1, n, head_dim]; `_apply_rotary(model, x, cos, sin)` rotates x [1, heads, S, head_dim]
  (partial rotary included). The model holds no `cos_cached`, no `rotary_emb` and no `config`.
- `_gated_delta_net`, `_ple`, `_hyper_connection` and `_hyper_connection_inject` are written for you. The layer function calls the parts of a layer
  in the reference's order; the one function the coder writes is the attention: `_C_self_attn(model, i, x, ids)`. The layer passes the attention
  connection's output as `x` and adds the returned block output itself, so return the block output with no residual add.

An attention written with these calls (names and sizes come from port_facts.json and CONFIG):

```python
def _C_self_attn(model, i, x, ids):                              # x [1, 1, S, hidden]
    p = f"model.layers.{i}.self_attn"
    S, heads, hd = x.shape[2], CONFIG["num_attention_heads"], CONFIG["head_dim"]
    q = _linear(model, p + ".q_proj", x)                          # [1, 1, S, heads * hd]
    q = ttnn.reshape(_rms_norm(model, p + ".q_norm", ttnn.reshape(q, (1, 1, S * heads, hd))), (1, S, heads, hd))
    q = ttnn.permute(q, (0, 2, 1, 3))                             # [1, heads, S, hd]
    cos, sin = _rope_tables(model, S)
    q = _apply_rotary(model, q, cos, sin)
    # k and v the same way with the key-value head count; scores, softmax and the value mix; join the heads; then
    return _linear(model, p + ".o_proj", joined)                  # joined [1, 1, S, heads * hd]
```
Keep every ttnn activation 4-D, and remember that a ttnn tensor has no `.view` or `.t()`. The head split by reshape on a tile-layout tensor has not been
checked on a board here; if it complains, read the tensor back with `_to_host`, split in torch and upload with `_upload`.

## 8. Checked on a board (probes 6 to 9, issue 711)

These forms ran on a chip and matched the CPU reference (PCC 0.9998 or better):
- `_apply_rotary(model, x, cos, sin)` with `x [1, heads, S, head_dim]` and tables `[1, 1, S, head_dim]`. The multiply that broadcasts the table over the heads axis works. It raises a named error when `x` has another layout, so reshape and permute `x` to that layout first.
- A per-head norm by reshape to `[1, 1, S * heads, head_dim]`, `_rms_norm` with the head norm weight, and reshape back; the head permute `(0, 2, 1, 3)`; both at `head_dim` 32.
- Attention with a per-query selection mask through `ttnn.transformer.scaled_dot_product_attention`, with grouped key-value heads (native or expanded with `repeat_kv`), and the sigmoid output gate.
- `ttnn.topk` is deterministic. With `stable=True` it picks the lowest index among tied scores. torch's `topk` does not follow that rule on ties, so a selection that has ties (relu scores tie at 0) can differ from torch in which tied entries it takes. Use `stable=True` and say so in the block's comment.
