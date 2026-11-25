# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The torch stand-in of ttnn (orchard/skills/full-port-templates/cpu_ttnn.py) and the device calls of the stage 2 board path.

In the Hy4-flash trial of stage 2 (2026-11-25) a board test run against the stand-in failed at `init` with "module 'ttnn' has no attribute
'set_fabric_config'": the template's own `open_device`, the trace calls and `synchronize_device` need names the stand-in did not define. The template's CPU
path (`port_check.py --cpu`) hands the port a ready device and never calls them, so only a rehearsal of the board path hit the gap. The stand-in now has
them, as calls that open nothing.

The upload rule is held here too. The runtime's `ttnn.from_torch` (ttnn/operations/core.py in the tt-metal checkout) passes any torch tensor on to the C++
`ttnn.Tensor` with the dtype asked for, so a float32 host tensor with an explicit dtype is an accepted call; the stand-in refused it. A float tensor with no
dtype keeps its own dtype in the runtime, which would put float32 on the device, so the stand-in keeps refusing that one. Each guard has a mutation in the PR
body that turns its test red."""
import importlib.util
import shutil
from pathlib import Path

import pytest
import torch

TEMPLATES = Path(__file__).resolve().parents[1] / "orchard" / "skills" / "full-port-templates"


def load(name):
    spec = importlib.util.spec_from_file_location("t_" + name.replace(".", "_"), TEMPLATES / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def tt():
    return load("cpu_ttnn.py").make()


@pytest.fixture()
def check(tmp_path):
    """port_check.py finds its run directory from its own location (the nearest `stages` above it), so it is loaded from a copy in a stage directory."""
    stage = tmp_path / "run" / "stages" / "2"
    stage.mkdir(parents=True)
    shutil.copy2(TEMPLATES / "port_check.py", stage / "port_check.py")
    spec = importlib.util.spec_from_file_location("t_port_check", stage / "port_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DEVICE_NAMES = ("set_fabric_config", "FabricConfig", "MeshShape", "open_mesh_device", "close_mesh_device", "open_device", "close_device", "MeshTraceId",
                "begin_trace_capture", "end_trace_capture", "execute_trace", "release_trace", "synchronize_device")


@pytest.mark.parametrize("name", DEVICE_NAMES)
def test_the_stand_in_defines_every_device_call_the_board_path_makes(tt, name):
    assert hasattr(tt, name), name


def test_the_templates_own_open_device_gives_one_device_for_one_chip_and_a_mesh_for_several(tt, check):
    dev, closer = check.open_device(tt, 1)
    assert dev.get_num_devices() == 1
    closer(dev)
    dev, closer = check.open_device(tt, 4)
    assert dev.get_num_devices() == 4
    closer(dev)


def test_the_mesh_shape_gives_the_chip_count_of_the_mesh(tt):
    assert tt.open_mesh_device(tt.MeshShape(1, 4)).get_num_devices() == 4
    assert tt.open_mesh_device(tt.MeshShape(2, 2)).get_num_devices() == 4


def test_a_trace_capture_and_replay_sequence_runs_and_each_capture_has_its_own_id(tt, check):
    dev = tt.open_mesh_device(tt.MeshShape(1, 2))
    a = tt.begin_trace_capture(dev, cq_id=0)
    tt.end_trace_capture(dev, a, cq_id=0)
    b = tt.begin_trace_capture(dev, cq_id=0)
    tt.end_trace_capture(dev, b, cq_id=0)
    assert a != b
    tt.execute_trace(dev, a, cq_id=0, blocking=False)
    tt.synchronize_device(dev)
    tt.release_trace(dev, a)
    assert check.trace_key(a) is not None and check.trace_key(b) is not None and check.trace_key(a) != check.trace_key(b)


def test_a_trace_id_has_no_hash_as_the_runtimes_has_none(tt):
    tid = tt.begin_trace_capture(tt.open_device(device_id=0), cq_id=0)
    with pytest.raises(TypeError):
        hash(tid)


def test_a_trace_id_is_a_trace_id_and_not_a_plain_int(tt):
    tid = tt.begin_trace_capture(tt.open_device(device_id=0), cq_id=0)
    assert isinstance(tid, tt.MeshTraceId) and not isinstance(tid, int)


# ---- the upload rule -----------------------------------------------------------------------------------------------

def upload(tt, tensor, **kw):
    return tt.from_torch(tensor, layout=tt.TILE_LAYOUT, device=tt.open_device(device_id=0), **kw)


def test_a_float32_host_tensor_with_an_explicit_bfloat16_dtype_is_uploaded(tt):
    out = upload(tt, torch.ones(32, 32, dtype=torch.float32), dtype=tt.bfloat16)
    assert out.t.shape == (32, 32)


def test_a_float16_host_tensor_with_an_explicit_block_float_dtype_is_uploaded(tt):
    out = upload(tt, torch.ones(32, 32, dtype=torch.float16), dtype=tt.bfloat8_b)
    assert out.dtype == "bfloat8_b"


def test_the_uploaded_values_are_the_bfloat16_rounding_of_the_float32_ones(tt):
    # 1.0040283203125 rounds to 1.0078125 straight to bfloat16 and to 1.0 by way of float16; 70000 is finite in bfloat16 and infinite in float16
    src = torch.tensor([[1.0040283203125, 3.14159265, 70000.0]], dtype=torch.float32)
    out = upload(tt, src, dtype=tt.bfloat16)
    assert torch.equal(out.t, src.to(torch.bfloat16).float())


def test_a_float_host_tensor_with_no_dtype_is_still_refused_when_it_is_not_bfloat16(tt):
    with pytest.raises(TypeError, match="dtype"):
        upload(tt, torch.ones(32, 32, dtype=torch.float32))


def test_a_bfloat16_host_tensor_with_no_dtype_is_uploaded_as_before(tt):
    assert upload(tt, torch.ones(32, 32, dtype=torch.bfloat16)).t.shape == (32, 32)


def test_token_ids_still_need_an_integer_tensor_in_row_major_layout(tt):
    with pytest.raises(TypeError):
        tt.from_torch(torch.ones(1, 4, dtype=torch.float32), dtype=tt.uint32, layout=tt.ROW_MAJOR_LAYOUT, device=tt.open_device(device_id=0))
