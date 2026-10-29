"""Run each given ssd_readout kernel.py on the chip (shared tt-lang 1.1.6) at [64,128] and at the model's [3072,128]; print PCC or the first error line."""
import importlib.util
import sys

import torch
import ttnn

dev = ttnn.open_device(device_id=0)
try:
    for path in sys.argv[1:]:
        try:
            spec = importlib.util.spec_from_file_location("k", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = mod.ssd_readout
            for rows in (64, 3072):
                g = torch.Generator().manual_seed(0)
                c = torch.randn(rows, 128, generator=g).to(torch.bfloat16)
                s = torch.randn(rows, 128, generator=g).to(torch.bfloat16)
                up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                out = up(torch.zeros(rows, 128, dtype=torch.bfloat16))
                fn(up(c), up(s), out)
                got = ttnn.to_torch(out).float()
                exp = (c.float() * s.float()).sum(-1, keepdim=True).expand(rows, 128)
                pcc = torch.corrcoef(torch.stack([got.flatten(), exp.flatten()]))[0, 1].item()
                print(path.split("kernels/")[-1], rows, "pcc", round(pcc, 5), "rel", round(((got - exp).norm() / exp.norm()).item(), 4), flush=True)
        except Exception as e:
            print(path.split("kernels/")[-1], "ERR", str(e).splitlines()[0][:160], flush=True)
finally:
    ttnn.close_device(dev)
