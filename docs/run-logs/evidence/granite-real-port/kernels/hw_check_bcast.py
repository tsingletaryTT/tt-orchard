"""Run each given ssd_state_update_bcast kernel.py on the chip (shared tt-lang 1.1.6) at [64,128] and [3072,128]; print PCC or the first error line."""
import importlib.util
import sys

import torch
import ttnn

dev = ttnn.open_device(device_id=0)
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
try:
    for path in sys.argv[1:]:
        name = path.split("kernels/")[-1]
        try:
            spec = importlib.util.spec_from_file_location("k", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = mod.ssd_state_update_bcast
            for rows in (64, 3072):
                g = torch.Generator().manual_seed(0)
                s = torch.randn(rows, 128, generator=g).to(torch.bfloat16)
                dec = torch.sigmoid(torch.randn(rows, 32, generator=g)).to(torch.bfloat16)
                dtx = torch.randn(rows, 32, generator=g).to(torch.bfloat16)
                b = torch.randn(32, 128, generator=g).to(torch.bfloat16)
                out = up(torch.zeros(rows, 128, dtype=torch.bfloat16))
                fn(up(s), up(dec), up(dtx), up(b), out)
                got = ttnn.to_torch(out).float()
                exp = dec[:, :1].float() * s.float() + dtx[:, :1].float() * b[:1, :].float()
                pcc = torch.corrcoef(torch.stack([got.flatten(), exp.flatten()]))[0, 1].item()
                print(name, rows, "pcc", round(pcc, 5), "rel", round(((got - exp).norm() / exp.norm()).item(), 4), flush=True)
        except Exception as e:
            print(name, "ERR", str(e).splitlines()[0][:160], flush=True)
finally:
    ttnn.close_device(dev)
