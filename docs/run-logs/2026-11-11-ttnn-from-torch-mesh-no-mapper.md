# ttnn.from_torch on a mesh device with no mesh_mapper (issue 563)

Question. The cheat sheet said "on a mesh pass a mesh_mapper or the tensor lands on one device". Run 20 of the tiny Qwen4 rehearsal called `ttnn.from_torch(tensor, dtype=..., layout=..., device=mesh_device)` with no mapper. Does that replicate, shard or put the tensor on one chip?

Measurement (2026-11-11, one gozer lease of a two-chip board, `ttnn.open_mesh_device(ttnn.MeshShape(1, 2))`, a `[1, 1, 32, 64]` bf16 tensor, `TILE_LAYOUT`):

| call | device tensors | shape of each | each equals the input |
|---|---|---|---|
| `from_torch(t, ..., device=mesh)` (no mapper) | 2 | `[1, 1, 32, 64]` | yes |
| `from_torch(t, ..., device=mesh, mesh_mapper=ReplicateTensorToMesh(mesh))` | 2 | `[1, 1, 32, 64]` | yes |

Result. With no mapper the tensor is replicated: both chips hold a full copy, the same as the replicate mapper. The earlier note was wrong that the tensor lands on one device. The mapper still states the intent in the port, and a sharded layout needs `ShardTensorToMesh`.

Limits. One mesh shape (1 x 2), one dtype and layout, and the ttnn build in the ttnn virtual environment of this machine. Reading a replicated tensor back with `ttnn.to_torch` still needs a mesh composer (run 20's error).
