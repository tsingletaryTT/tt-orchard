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
        host[name] = t.float()                              # norms, conv kernels, embedding tables, integer tables
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
