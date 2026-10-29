"""Per-op check of the two ops the fusion port changes, against a CPU reference, on one die (issue 184).
Router: RouterTopK at E=256, k=8 (the 35B) and E=512, k=10 (the bundle's own). Sampler: RowsTopK at 124160 columns per die
with several scan grids. Prints JSON lines. Never edits the bundle; the grid is set on the imported module for the run."""
import json, os, sys, time
import torch
import ttnn
from models.demos.blackhole.qwen36.tt.moe_decode import topk as rt
from models.demos.blackhole.qwen36.tt.decode_sampling import ops as so

def up(x):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=ttnn.ReplicateTensorToMesh(dev))
def down(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])
def emit(**kw): print("RESULT " + json.dumps(kw), flush=True)

dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
torch.manual_seed(0)

def timed(fn, n=200):
    fn(); ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(n): fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / n * 1e6

# ---- router ------------------------------------------------------------------------------------------------------------
for E, k in ((512, 10), (256, 8)):
    op = rt.RouterTopK(dev, E, k)
    ids, scores = op.alloc_outputs()
    bad_ids = bad_w = 0
    trials = 40
    for i in range(trials):
        x = torch.randn(1, 1, 32, E).bfloat16()
        if i == 0: x[0, 0, 0, E - 1] = 50.0; x[0, 0, 0, 200] = 49.0       # top experts at high indices (mask check)
        tt = up(x)
        op(tt, ids, scores)
        gi = down(ids).reshape(-1)[:k].int(); gs = down(scores).reshape(-1)[:k].float()
        rv, ri = torch.topk(x[0, 0, 0].float(), k)          # sorted
        if i == 1: emit(debug="router", raw_ids=down(ids).reshape(-1)[:16].tolist(), ref_ids=ri.tolist(), raw_scores=down(scores).reshape(-1)[:k].float().tolist(), ref_w=torch.softmax(rv, 0).tolist())
        rw = torch.softmax(rv, 0)
        if set(gi.tolist()) != set(ri.tolist()): bad_ids += 1
        else:
            order = {int(e): j for j, e in enumerate(gi.tolist())}
            gw = torch.tensor([gs[order[int(e)]] for e in ri.tolist()])
            if (gw - rw).abs().max().item() > 2e-2: bad_w += 1
        ttnn.deallocate(tt)
    x = torch.randn(1, 1, 32, E).bfloat16(); tt = up(x)
    us = timed(lambda: op(tt, ids, scores))
    emit(op="router_topk", experts=E, k=k, trials=trials, id_set_mismatches=bad_ids, weight_mismatches=bad_w, us_per_call_untraced=round(us, 1))

# ---- sampler -----------------------------------------------------------------------------------------------------------
for width in (75968, 124160):
    for grid in ((10, 10), (8, 8), (10, 5), (9, 10), (10, 8)):
        so.GRID = grid
        try:
            op = so.RowsTopK(dev, width)
        except Exception as exc:
            emit(op="sampler_rows_topk", width=width, grid=list(grid), error=str(exc)[:200]); continue
        x = torch.randn(1, 1, 32, width).bfloat16()
        planted = {0: width - 3, 1: 70000 if width > 70000 else 5}
        for r, c in planted.items(): x[0, 0, r, c] = 40.0
        tt = up(x)
        v, ix = op(tt, 1)
        gv = down(v)[0, 0, 0].float(); gi = down(ix)[0, 0, 0].long()
        rv, ri = torch.topk(x[0, 0, 0].float(), 32)
        same_set = set(gi.tolist()) == set(ri.tolist())
        emit(op="sampler_rows_topk", width=width, grid=list(grid), scan_cores=len(op.work), index_set_equal=same_set, values_equal=bool((gv == rv).all()),
             top1_equal=int(gi[0]) == int(ri[0]), values_max_abs_diff=float((gv - rv).abs().max()), us_per_call_untraced=round(timed(lambda: op(tt, 1), 100), 1))
        ttnn.deallocate(tt)
ttnn.close_mesh_device(dev)
emit(done=True)
