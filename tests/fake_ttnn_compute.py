# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A stand-in `ttnn` that computes with torch on the CPU: the calls a generated port (orchard/port_template.py) makes, with the
argument names of the verified cheat sheet, so a test can run the generated port end to end and compare its logits with a plain torch
forward. Nothing here touches a chip. Install it with `install(monkeypatch)` before the generated module is imported; the real ttnn
package must never be imported in a test (importing it opens the device).

Shapes follow the reference ports: activations are [1, 1, S, features], heads are [1, heads, S, head_dim]. `COUNTS` tallies the calls
made, so a test can show that every projection ran through `linear`."""
from __future__ import annotations

import collections
import sys
import types

import torch

COUNTS: collections.Counter = collections.Counter()


class Tensor:
    """A device tensor: a torch tensor plus whether it was replicated on a mesh."""

    def __init__(self, t: torch.Tensor, replicated: bool = False, layout: str = "TILE"):
        self.t = t
        self.replicated = replicated
        self.layout = layout
        self.shape = tuple(t.shape)


class MeshDevice:
    def __init__(self, n: int):
        self.n = n

    def get_num_devices(self):
        return self.n


class ReplicateTensorToMesh:
    def __init__(self, mesh_device):
        self.mesh = mesh_device


class ConcatMeshToTensor:
    def __init__(self, mesh_device, dim):
        self.mesh, self.dim = mesh_device, dim


class WormholeComputeKernelConfig:
    def __init__(self, math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc):
        self.args = (math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc)


class MathFidelity:
    HiFi4 = "HiFi4"


def _count(name):
    COUNTS[name] += 1


def _rms(x, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _probe_common():
    """The probes' torch copies of the reference gated delta rule (scripts/qwen4_ops_probe/common.py imports only torch)."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "qwen4_ops_probe" / "common.py"
    spec = importlib.util.spec_from_file_location("qwen4_probe_common_for_fake", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


builtins_slice = slice          # the fake's own `slice` op below shadows the builtin inside make()


def make() -> types.ModuleType:
    """The fake module, fresh (the counts are shared)."""
    m = types.ModuleType("ttnn")
    m.bfloat16, m.uint32, m.int32, m.bfloat8_b = "bfloat16", "uint32", "int32", "bfloat8_b"
    m.TILE_LAYOUT, m.ROW_MAJOR_LAYOUT, m.DRAM_MEMORY_CONFIG = "TILE", "ROW_MAJOR", "DRAM"
    m.Tensor, m.MeshDevice = Tensor, MeshDevice
    m.ReplicateTensorToMesh, m.ConcatMeshToTensor = ReplicateTensorToMesh, ConcatMeshToTensor
    m.WormholeComputeKernelConfig, m.MathFidelity = WormholeComputeKernelConfig, MathFidelity

    def from_torch(tensor, dtype=None, layout=None, device=None, memory_config=None, mesh_mapper=None):
        _count("from_torch")
        if dtype == "uint32":
            if not (tensor.dtype in (torch.int32, torch.int64) and layout == "ROW_MAJOR"):
                raise TypeError("token ids must be an integer tensor in ROW_MAJOR layout")
            return Tensor(tensor.clone(), mesh_mapper is not None, "ROW_MAJOR")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"a float tensor is uploaded as bfloat16 from a bfloat16 host tensor, got {tensor.dtype}")
        if layout not in ("TILE", "ROW_MAJOR"):
            raise TypeError("a layout is required")
        return Tensor(tensor.float(), mesh_mapper is not None, layout)

    def to_torch(tensor, dtype=None, device=None, mesh_composer=None):
        _count("to_torch")
        if tensor.replicated:
            if mesh_composer is None:
                raise RuntimeError("Can't convert a tensor distributed on a mesh without a mesh composer")
            return torch.cat([tensor.t] * mesh_composer.mesh.get_num_devices(), dim=mesh_composer.dim)
        return tensor.t.clone()

    def linear(input_tensor_a, input_tensor_b, *, bias=None, transpose_a=False, transpose_b=False, memory_config=None, dtype=None,
               program_config=None, activation=None, compute_kernel_config=None, core_grid=None):
        _count("linear")
        if input_tensor_a.layout != "TILE" or input_tensor_b.layout != "TILE":
            raise RuntimeError("matmul inputs must be TILE_LAYOUT")
        a, b = input_tensor_a.t, input_tensor_b.t
        if transpose_b:
            b = b.transpose(-2, -1)
        if a.shape[-1] != b.shape[-2]:
            raise RuntimeError(f"The width of the first tensor must be equal to the height of the second tensor. Mismatch: width={a.shape[-1]} height={b.shape[-2]}")
        y = a @ b
        if bias is not None:
            y = y + bias.t
        return Tensor(y)

    def matmul(input_tensor_a, input_tensor_b, *, transpose_a=False, transpose_b=False, memory_config=None, dtype=None, program_config=None,
               activation=None, compute_kernel_config=None, core_grid=None):
        _count("matmul")
        return linear(input_tensor_a, input_tensor_b, transpose_b=transpose_b)

    def embedding(input_tensor, weight, *, padding_idx=None, layout=None, embeddings_type=None, dtype=None, memory_config=None):
        _count("embedding")
        if weight.layout != "ROW_MAJOR" or input_tensor.layout != "ROW_MAJOR":
            raise RuntimeError("embedding wants ROW_MAJOR ids and a ROW_MAJOR weight table")
        out = torch.nn.functional.embedding(input_tensor.t.long(), weight.t)
        return Tensor(out, layout="TILE" if layout == "TILE" else "ROW_MAJOR")

    def rms_norm(input_tensor, *, epsilon=1e-12, weight=None, bias=None, residual_input_tensor=None, memory_config=None, program_config=None,
                 compute_kernel_config=None):
        _count("rms_norm")
        if input_tensor.layout != "TILE" or (weight is not None and weight.layout != "TILE"):
            raise RuntimeError("rms_norm wants TILE_LAYOUT inputs")
        x = input_tensor.t
        if residual_input_tensor is not None:
            x = x + residual_input_tensor.t
        y = _rms(x, epsilon)
        if weight is not None:
            if weight.t.shape[-1] != x.shape[-1]:
                raise RuntimeError("a TILE weight needs the input's last dim")
            y = y * weight.t
        return Tensor(y)

    def silu(x):
        _count("silu")
        return Tensor(torch.nn.functional.silu(x.t))

    def sigmoid(x):
        _count("sigmoid")
        return Tensor(torch.sigmoid(x.t))

    def slice(input_tensor, slice_start, slice_end, slice_step=None, *, memory_config=None):
        """ttnn.slice on the padded shape: every start inside its dimension and every end at most the dimension (a slice outside fails)."""
        _count("slice")
        t = input_tensor.t
        for d, (lo, hi) in enumerate(zip(slice_start, slice_end)):
            if not (0 <= lo < t.shape[d] and 0 < hi <= t.shape[d] and lo < hi):
                raise RuntimeError(f"TT_FATAL: slice [{lo}, {hi}) is outside dimension {d} of size {t.shape[d]}")
        return Tensor(t[tuple(builtins_slice(lo, hi) for lo, hi in zip(slice_start, slice_end))], input_tensor.replicated, input_tensor.layout)

    def _same_layout(a, b):
        if isinstance(b, Tensor) and a.layout != b.layout:
            raise RuntimeError(f"a binary op needs both inputs in one layout, got {a.layout} and {b.layout}")

    def multiply(a, b):
        _count("multiply")
        _same_layout(a, b)
        return Tensor(a.t * (b.t if isinstance(b, Tensor) else b), layout=a.layout)

    def add(a, b):
        _count("add")
        _same_layout(a, b)
        return Tensor(a.t + (b.t if isinstance(b, Tensor) else b), layout=a.layout)

    def reshape(input_tensor, shape, memory_config=None):
        _count("reshape")
        return Tensor(input_tensor.t.reshape(tuple(shape)), input_tensor.replicated, input_tensor.layout)

    def permute(input_tensor, dims, memory_config=None):
        """ttnn.permute: the dimensions in the order given (all of them)."""
        _count("permute")
        if sorted(dims) != list(range(input_tensor.t.dim())):
            raise RuntimeError("permute needs every dimension exactly once")
        return Tensor(input_tensor.t.permute(*dims).contiguous(), input_tensor.replicated, input_tensor.layout)

    def relu(x, **kwargs):
        _count("relu")
        return Tensor(torch.relu(x.t), x.replicated, x.layout)

    def subtract(a, b):
        _count("subtract")
        _same_layout(a, b)
        return Tensor(a.t - (b.t if isinstance(b, Tensor) else b), layout=a.layout)

    def topk(input_tensor, k, dim=-1, largest=True, sorted=True, *, stable=False, memory_config=None, **kwargs):
        """ttnn.topk on the last dimension of a TILE tensor: (values, indices). With stable=True a tie goes to the lower index (the rule probe 7 measured on a
        board: `index_list_equals_lowest_index_rule` on every row); with stable=False the fake uses torch.topk, whose tie order is not that rule (the probe measured
        the device's own order as well, and it is deterministic but not documented)."""
        _count("topk")
        if input_tensor.layout != "TILE" or dim not in (-1, input_tensor.t.dim() - 1):
            raise RuntimeError("the fake covers topk on the last dimension of a TILE tensor")
        x = input_tensor.t
        if stable:
            order = torch.sort(x, dim=-1, descending=largest, stable=True)
            idx = order.indices[..., :k]
            val = order.values[..., :k]
        else:
            val, idx = torch.topk(x, k, dim=-1, largest=largest, sorted=sorted)
        return Tensor(val, input_tensor.replicated, "TILE"), Tensor(idx.to(torch.int64), input_tensor.replicated, "TILE")

    def concat(tensors, dim=0, memory_config=None):
        _count("concat")
        return Tensor(torch.cat([t.t for t in tensors], dim=dim), layout=tensors[0].layout)

    def deallocate(tensor):
        _count("deallocate")

    def to_layout(tensor, layout, dtype=None, memory_config=None):
        """The same values in the requested layout; a dtype of bfloat16 rounds them (what the device does when the dtype argument is given)."""
        _count("to_layout")
        t = tensor.t.to(torch.bfloat16).float() if dtype == "bfloat16" else tensor.t
        return Tensor(t, tensor.replicated, layout)

    def typecast(tensor, dtype, memory_config=None):
        _count("typecast")
        return Tensor(tensor.t.to(torch.bfloat16).float() if dtype == "bfloat16" else tensor.t, tensor.replicated, tensor.layout)

    def softplus(x, *, beta=1.0, threshold=20.0, memory_config=None, output_tensor=None, sub_core_grids=None):
        _count("softplus")
        return Tensor(torch.nn.functional.softplus(x.t, beta=beta, threshold=threshold), x.replicated, x.layout)

    def _unary(name, fn):
        def op(x, **kwargs):
            _count(name)
            return Tensor(fn(x.t), x.replicated, x.layout)
        return op

    def clamp(x, min=None, max=None, *, memory_config=None, output_tensor=None):
        """ttnn.clamp with float bounds (the tensor-bound overload is not faked)."""
        _count("clamp")
        return Tensor(torch.clamp(x.t, min=min, max=max), x.replicated, x.layout)

    def sum_(x, dim=None, keepdim=False, *, memory_config=None, compute_kernel_config=None, scalar=1.0, correction=None, sub_core_grids=None,
             fast_and_approximate_mode=False, output_layout=None):
        """ttnn.sum over one dimension (torch.sum): with keepdim the reduced dimension stays with size 1."""
        _count("sum")
        t = x.t.sum() if dim is None else x.t.sum(dim=dim, keepdim=keepdim)
        return Tensor(t * scalar, x.replicated, x.layout)

    m.abs, m.sign, m.sqrt = _unary("abs", torch.abs), _unary("sign", torch.sign), _unary("sqrt", torch.sqrt)
    m.clamp, m.sum = clamp, sum_
    m.from_torch, m.to_torch, m.linear, m.matmul, m.embedding, m.rms_norm = from_torch, to_torch, linear, matmul, embedding, rms_norm
    m.silu, m.multiply, m.add, m.reshape, m.concat, m.deallocate, m.to_layout = silu, multiply, add, reshape, concat, deallocate, to_layout
    m.sigmoid, m.slice, m.typecast, m.softplus = sigmoid, slice, typecast, softplus
    m.permute, m.relu, m.subtract, m.topk = permute, relu, subtract, topk

    ex = types.ModuleType("ttnn.experimental")

    def nlp_create_qkv_heads(input, input_kv=None, *, num_heads, num_kv_heads=None, transpose_k_heads=True, memory_config=None):
        _count("nlp_create_qkv_heads")
        x = input.t                                   # [1, 1, S, (nh + 2 nkv) hd]
        nkv = num_kv_heads or num_heads
        hd = x.shape[-1] // (num_heads + 2 * nkv)
        s = x.shape[-2]
        q = x[..., : num_heads * hd].reshape(1, s, num_heads, hd).transpose(1, 2)
        k = x[..., num_heads * hd: (num_heads + nkv) * hd].reshape(1, s, nkv, hd).transpose(1, 2)
        v = x[..., (num_heads + nkv) * hd:].reshape(1, s, nkv, hd).transpose(1, 2)
        if transpose_k_heads:
            k = k.transpose(-2, -1)
        return Tensor(q), Tensor(k), Tensor(v)

    def rotary_embedding_hf(input_tensor, cos_cache, sin_cache, is_decode_mode=False, memory_config=None, compute_kernel_config=None):
        _count("rotary_embedding_hf")
        x, cos, sin = input_tensor.t, cos_cache.t, sin_cache.t      # x [1, h, S, hd]; cos and sin [1, 1, S, hd]
        return Tensor(x * cos + _rotate_half(x) * sin)

    def nlp_concat_heads(input_tensor, memory_config=None):
        _count("nlp_concat_heads")
        x = input_tensor.t                            # [1, h, S, hd]
        return Tensor(x.transpose(1, 2).reshape(1, 1, x.shape[2], -1))

    ex.nlp_create_qkv_heads, ex.rotary_embedding_hf, ex.nlp_concat_heads = nlp_create_qkv_heads, rotary_embedding_hf, nlp_concat_heads
    m.experimental = ex

    tr = types.ModuleType("ttnn.transformer")

    def scaled_dot_product_attention(input_tensor_q, input_tensor_k, input_tensor_v, *, attn_mask=None, is_causal=True, scale=None,
                                     sliding_window_size=None, memory_config=None, program_config=None, compute_kernel_config=None):
        _count("scaled_dot_product_attention")
        q, k, v = input_tensor_q.t, input_tensor_k.t, input_tensor_v.t
        if q.shape[-2] % 32:
            raise RuntimeError("the sequence must be padded to a multiple of 32")
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        scores = (q @ k.transpose(-2, -1)) * (scale if scale is not None else q.shape[-1] ** -0.5)
        if attn_mask is not None:
            if is_causal:
                raise RuntimeError("an explicit attn_mask goes with is_causal=False")
            if attn_mask.layout != "TILE":
                raise RuntimeError("attn_mask must be a TILE tensor")
            scores = scores + attn_mask.t               # additive mask [1, 1 or heads, S, S]
        if is_causal:
            s = q.shape[-2]
            scores = scores.masked_fill(torch.ones(s, s, dtype=torch.bool).triu(1), float("-inf"))
        return Tensor(torch.softmax(scores, dim=-1) @ v)

    def chunk_gated_delta_rule(q, k, v, g, beta, *, scale=None, initial_state=None, output_final_state=False, chunk_size=64, use_qk_l2norm=False,
                               output_head_major=False, program_config=None, wy_inverse=None, memory_config=None, compute_kernel_config=None,
                               eye=None, tril=None, ones=None, masks=None, num_key_heads=None):
        """The device op's rank-3 (flat, token-major) form on torch: the reference chunked gated delta rule of the probes (a copy of the
        transformers code) with the q/k L2 norm and the K ** -0.5 scale the kernel applies. A torch stand-in: it shows that the generated code
        passes the layouts and head counts the op's documentation asks for (TILE inputs, rank 3, chunk 32, T a multiple of 32, num_key_heads),
        and that the reference algorithm gives the transformers result. It does not show that the device kernel computes the same numbers."""
        _count("chunk_gated_delta_rule")
        for name, x in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
            if x.layout != "TILE":
                raise RuntimeError(f"chunk_gated_delta_rule: {name} must be a TILE_LAYOUT device tensor, got {x.layout}")
            if len(x.t.shape) != 3:
                raise RuntimeError(f"chunk_gated_delta_rule: {name} must be rank 3 (flat, token-major), got rank {len(x.t.shape)}")
        if chunk_size != 32 or output_final_state or initial_state is not None or scale is not None:
            raise RuntimeError("the fake covers the prefill call: chunk_size 32, no initial state, the default scale, no final state")
        batch, seq, _ = q.t.shape
        if seq % chunk_size:
            raise RuntimeError("T must be a multiple of chunk_size on the flat path")
        hv = beta.t.shape[-1]
        if num_key_heads is None:
            raise RuntimeError("the generated code passes num_key_heads (K may differ from V)")
        hk = num_key_heads
        dk, dv = q.t.shape[-1] // hk, v.t.shape[-1] // hv
        if q.t.shape[-1] != hk * dk or k.t.shape != q.t.shape or v.t.shape[-1] != hv * dv or g.t.shape != beta.t.shape or hv % hk:
            raise RuntimeError("chunk_gated_delta_rule: the head counts do not divide the widths")
        rep = hv // hk
        qh = q.t.float().reshape(batch, seq, hk, dk).repeat_interleave(rep, dim=2)
        kh = k.t.float().reshape(batch, seq, hk, dk).repeat_interleave(rep, dim=2)
        vh = v.t.float().reshape(batch, seq, hv, dv)
        out, _ = _probe_common().torch_chunk_gated_delta_rule(qh, kh, vh, g.t.float(), beta.t.float(), chunk_size=64, use_qk_l2norm_in_kernel=True)
        return Tensor(out.float(), layout="ROW_MAJOR"), None

    tr.scaled_dot_product_attention = scaled_dot_product_attention
    tr.chunk_gated_delta_rule = chunk_gated_delta_rule
    m.transformer = tr
    return m


def install(monkeypatch) -> types.ModuleType:
    """Put the fake in sys.modules as `ttnn` (and its two namespaces) for the test, and reset the counts."""
    m = make()
    monkeypatch.setitem(sys.modules, "ttnn", m)
    monkeypatch.setitem(sys.modules, "ttnn.experimental", m.experimental)
    monkeypatch.setitem(sys.modules, "ttnn.transformer", m.transformer)
    COUNTS.clear()
    return m
