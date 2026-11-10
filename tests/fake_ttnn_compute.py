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

    def multiply(a, b):
        _count("multiply")
        return Tensor(a.t * (b.t if isinstance(b, Tensor) else b))

    def add(a, b):
        _count("add")
        return Tensor(a.t + (b.t if isinstance(b, Tensor) else b))

    def reshape(input_tensor, shape, memory_config=None):
        _count("reshape")
        return Tensor(input_tensor.t.reshape(tuple(shape)), input_tensor.replicated)

    def concat(tensors, dim=0, memory_config=None):
        _count("concat")
        return Tensor(torch.cat([t.t for t in tensors], dim=dim))

    def deallocate(tensor):
        _count("deallocate")

    def to_layout(tensor, layout, dtype=None, memory_config=None):
        return tensor

    m.from_torch, m.to_torch, m.linear, m.matmul, m.embedding, m.rms_norm = from_torch, to_torch, linear, matmul, embedding, rms_norm
    m.silu, m.multiply, m.add, m.reshape, m.concat, m.deallocate, m.to_layout = silu, multiply, add, reshape, concat, deallocate, to_layout

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
        if is_causal:
            s = q.shape[-2]
            scores = scores.masked_fill(torch.ones(s, s, dtype=torch.bool).triu(1), float("-inf"))
        return Tensor(torch.softmax(scores, dim=-1) @ v)

    tr.scaled_dot_product_attention = scaled_dot_product_attention
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
