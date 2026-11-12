# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The fake ttnn's sharded tensors and all_reduce (tests/fake_ttnn_compute.py): the ops the expert-parallel MoE block needs (issue 723). A tensor uploaded with
ShardTensorToMesh(dim=0) holds one slice per chip and reports the shape of one chip's slice; the ops work on every slice at once; all_reduce adds the slices."""
import pytest
import torch

import fake_ttnn_compute as fake


@pytest.fixture
def tt(monkeypatch):
    return fake.install(monkeypatch)


def sharded(tt, mesh, t, layout="TILE"):
    return tt.from_torch(t.bfloat16(), dtype=tt.bfloat16, layout=tt.TILE_LAYOUT if layout == "TILE" else tt.ROW_MAJOR_LAYOUT, device=mesh,
                         mesh_mapper=tt.ShardTensorToMesh(mesh, dim=0))


def test_a_sharded_tensor_reports_one_chips_shape_and_keeps_every_slice(tt):
    mesh = tt.MeshDevice(4)
    w = sharded(tt, mesh, torch.arange(4 * 32 * 32, dtype=torch.float32).reshape(4, 1, 32, 32))
    assert w.shape == (1, 1, 32, 32) and w.t.shape == (4, 1, 32, 32)
    back = tt.to_torch(w, mesh_composer=tt.ConcatMeshToTensor(mesh, dim=0))
    assert back.shape == (4, 1, 32, 32)


def test_a_replicated_input_times_a_sharded_weight_gives_a_sharded_result_with_each_chips_own_product(tt):
    mesh = tt.MeshDevice(2)
    x = torch.randn(1, 1, 32, 32).bfloat16()
    w = torch.randn(2, 1, 32, 32).bfloat16()
    xr = tt.from_torch(x, dtype=tt.bfloat16, layout=tt.TILE_LAYOUT, device=mesh, mesh_mapper=tt.ReplicateTensorToMesh(mesh))
    y = tt.matmul(xr, sharded(tt, mesh, w))
    assert y.shape == (1, 1, 32, 32)
    for c in range(2):
        assert torch.allclose(y.t[c].float(), (x.float() @ w[c].float()), atol=1e-2)


def test_all_reduce_adds_the_slices_and_leaves_the_same_sum_on_every_chip(tt):
    mesh = tt.MeshDevice(4)
    parts = torch.randn(4, 1, 32, 32).bfloat16()
    y = tt.all_reduce(sharded(tt, mesh, parts), cluster_axis=1, num_links=1, topology=tt.Topology.Linear)
    want = parts.float().sum(0)
    for c in range(4):
        assert torch.allclose(y.t[c].float(), want, atol=5e-2)
    assert y.shape == (1, 1, 32, 32)


def test_embedding_with_sharded_ids_gathers_each_chips_own_rows_and_reshape_keeps_the_slices(tt):
    mesh = tt.MeshDevice(2)
    table = torch.randn(64, 32).bfloat16()
    tab = tt.from_torch(table, dtype=tt.bfloat16, layout=tt.TILE_LAYOUT, device=mesh, mesh_mapper=tt.ReplicateTensorToMesh(mesh))
    ids = torch.tensor([[3, 4, 5, 6] + [0] * 28, [10, 11, 12, 13] + [0] * 28], dtype=torch.int32)
    got = tt.embedding(tt.from_torch(ids, dtype=tt.uint32, layout=tt.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=tt.ShardTensorToMesh(mesh, dim=0)), tab, layout=tt.TILE_LAYOUT)
    got = tt.reshape(got, (1, 1, 32, 32))
    assert got.shape == (1, 1, 32, 32) and got.t.shape == (2, 1, 32, 32)
    assert torch.equal(got.t[1, 0, 0].bfloat16(), table[10]) and torch.equal(got.t[0, 0, 3].bfloat16(), table[6])


def test_a_sharded_tensor_added_to_a_replicated_one_stays_sharded(tt):
    mesh = tt.MeshDevice(2)
    a = sharded(tt, mesh, torch.ones(2, 1, 32, 32))
    b = tt.from_torch(torch.ones(1, 1, 32, 32).bfloat16(), dtype=tt.bfloat16, layout=tt.TILE_LAYOUT, device=mesh, mesh_mapper=tt.ReplicateTensorToMesh(mesh))
    s = tt.add(a, b)
    assert s.t.shape == (2, 1, 32, 32) and s.shape == (1, 1, 32, 32)


def test_sharding_needs_a_leading_dimension_equal_to_the_chip_count(tt):
    mesh = tt.MeshDevice(4)
    with pytest.raises(RuntimeError, match="leading dimension"):
        sharded(tt, mesh, torch.zeros(3, 1, 32, 32))
