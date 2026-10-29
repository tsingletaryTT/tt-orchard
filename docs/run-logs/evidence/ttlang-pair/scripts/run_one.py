"""Run one kernel from k/manifest.json, on the chip (python) or in the simulator (tt-lang-sim), and print one RESULT json line."""
import importlib.util, json, sys, traceback
from pathlib import Path
import torch
import ttnn
import ttl

K = Path(__file__).resolve().parent.parent / "k"
name = sys.argv[1]
entry = next(e for e in json.load(open(K / "manifest.json")) if e["name"] == name)

def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()

res = dict(name=name, results=[])
try:
    if hasattr(ttl, "build_info"):
        res["build_info"] = str(ttl.build_info())
except Exception as e:
    res["build_info_error"] = repr(e)
dev = None
try:
    dev = ttnn.open_device(device_id=0)
    spec = importlib.util.spec_from_file_location("k_" + name, K / entry["file"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, entry["fn"])
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    for rows, cols in entry["shapes"]:
        r = dict(shape=[rows, cols])
        try:
            g = torch.Generator().manual_seed(0)
            if entry["kind"] == "add":
                a = torch.rand(rows, cols, generator=g).to(torch.bfloat16); b = torch.rand(rows, cols, generator=g).to(torch.bfloat16)
                ins = [a, b]; exp = a.float() + b.float()
            elif entry["kind"] == "readout":
                c = torch.randn(rows, cols, generator=g).to(torch.bfloat16); s = torch.randn(rows, cols, generator=g).to(torch.bfloat16)
                ins = [c, s]; exp = (c.float() * s.float()).sum(-1, keepdim=True).expand(rows, cols)
            elif entry["kind"] == "rmsnorm":
                x = torch.randn(rows, cols, generator=g).to(torch.bfloat16); w = torch.rand(rows, cols, generator=g).to(torch.bfloat16)
                ins = [x, w]; xf = x.float()
                exp = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
            out = up(torch.zeros(rows, cols, dtype=torch.bfloat16))
            fn(*[up(t) for t in ins], out)
            got = ttnn.to_torch(out).float()
            r["pcc"] = round(pcc(got, exp), 5)
            r["rel_l2"] = round(((got - exp).norm() / exp.norm()).item(), 4)
            r["ok"] = True
        except Exception as e:
            r["ok"] = False
            r["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()[:1500]; r["tb"] = traceback.format_exc()[-1800:]
        res["results"].append(r)
except Exception as e:
    res["fatal"] = "".join(traceback.format_exception_only(type(e), e)).strip()[:1500]
finally:
    if dev is not None:
        try: ttnn.close_device(dev)
        except Exception as e: res["close_error"] = repr(e)
print("RESULT " + json.dumps(res), flush=True)
