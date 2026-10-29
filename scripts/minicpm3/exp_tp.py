# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Is a two-chip tensor-parallel decode worth building? Opens the two chips of one board as a 1 by 2 mesh and times an all-reduce of one decode row."""
import time

import torch
import ttnn

REPS = 20
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), trace_region_size=64 * 1024 * 1024)
try:
    print("mesh", mesh.shape, flush=True)
    x = torch.randn(2, 1, 32, 2560)
    xt = ttnn.from_torch(x, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    for links in (1, 2):
        for topo in (ttnn.Topology.Linear, ttnn.Topology.Ring):
            try:
                y = ttnn.all_reduce(xt, cluster_axis=1, num_links=links, topology=topo)
                ttnn.synchronize_device(mesh)
                got = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
                ref = x.sum(0, keepdim=True)
                ok = ((got[0:1].float() - ref).norm() / ref.norm()).item()
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                for _ in range(REPS):
                    y = ttnn.all_reduce(xt, cluster_axis=1, num_links=links, topology=topo)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.synchronize_device(mesh)
                t0 = time.time()
                for _ in range(5):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                dt = (time.time() - t0) / (5 * REPS)
                ttnn.release_trace(mesh, tid)
                print(f"all_reduce links {links} {topo}: {dt * 1e6:.1f} us per call, relL2 vs sum {ok:.4f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"all_reduce links {links} {topo}: FAILED {str(e).splitlines()[0][:300]}", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
