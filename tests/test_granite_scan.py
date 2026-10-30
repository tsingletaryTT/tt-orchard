"""The chunked Mamba-2 scan (pkgsrc/runtime/granite_scan.py) against the one-token recurrence the decode step uses.
Pure torch on the CPU: no chip."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("granite_scan", ROOT / "pkgsrc" / "runtime" / "granite_scan.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)

H, P, N = 6, 8, 16


def inputs(T, seed=0, a_scale=1.0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)  # noqa: E731
    x, B, C = r(T, H * P), r(T, N), r(T, N)
    dt = torch.nn.functional.softplus(r(T, H) + 0.5)
    A = -torch.exp(r(H) * 0.5) * a_scale
    D = r(H)
    S0 = r(H * P, N)
    return x, dt, B, C, A, D, S0


def sequential(x, dt, B, C, A, D, S0):
    """The decode step, one token at a time (what GraniteChip._mamba does)."""
    S, ys = S0.clone().view(H, P, N), []
    for t in range(x.shape[0]):
        decay = torch.exp(dt[t] * A).view(H, 1, 1)
        xt = x[t].view(H, P, 1)
        S = S * decay + (dt[t].view(H, 1, 1) * xt) * B[t].view(1, 1, N)
        y = S @ C[t].view(1, N, 1) + D.view(H, 1, 1) * xt
        ys.append(y.reshape(-1))
    return torch.stack(ys), S.reshape(H * P, N)


@pytest.mark.parametrize("T", [1, 5, 32])
def test_one_chunk_matches_the_recurrence(T):
    args = inputs(T)
    y, S = scan.ssd_chunk(*args, H=H, P=P, N=N)
    ry, rS = sequential(*args)
    assert torch.allclose(y, ry, atol=1e-9, rtol=1e-9)
    assert torch.allclose(S, rS, atol=1e-9, rtol=1e-9)


def test_strong_decay_is_stable():
    """A large |A| makes exp(cs_i) tiny and exp(-cs_j) huge when written as a product; the difference form must not overflow."""
    args = inputs(64, a_scale=40.0)
    y, S = scan.ssd_chunk(*args, H=H, P=P, N=N)
    ry, rS = sequential(*args)
    assert torch.isfinite(y).all() and torch.isfinite(S).all()
    assert torch.allclose(y, ry, atol=1e-9, rtol=1e-9)


def test_chunks_carry_the_state():
    """The state handed from one chunk to the next gives the same result as one pass over every token."""
    args = inputs(100, seed=3)
    x, dt, B, C, A, D, S0 = args
    ys, S = [], S0
    for a, b in ((0, 32), (32, 64), (64, 100)):                      # the last chunk is short
        y, S = scan.ssd_chunk(x[a:b], dt[a:b], B[a:b], C[a:b], A, D, S, H=H, P=P, N=N)
        ys.append(y)
    ry, rS = sequential(*args)
    assert torch.allclose(torch.cat(ys), ry, atol=1e-9, rtol=1e-9)
    assert torch.allclose(S, rS, atol=1e-9, rtol=1e-9)


def test_causal_conv_chunk_matches_stepwise():
    """The block conv with a three-row state equals the one-token conv, and leaves the last three inputs as the state."""
    g = torch.Generator().manual_seed(1)
    C_, T = 10, 37
    w, b = torch.randn(4, C_, generator=g), torch.randn(C_, generator=g)
    xbc, st = torch.randn(T, C_, generator=g), torch.randn(3, C_, generator=g)
    out, new = scan.causal_conv_chunk(xbc, st, w, b)
    hist, ref = [r for r in st], []
    for t in range(T):
        win = torch.stack([hist[-3], hist[-2], hist[-1], xbc[t]])
        ref.append(torch.nn.functional.silu((win * w).sum(0) + b))
        hist.append(xbc[t])
    assert torch.allclose(out, torch.stack(ref), atol=1e-5)
    assert torch.equal(new, torch.stack(hist[-3:]))
