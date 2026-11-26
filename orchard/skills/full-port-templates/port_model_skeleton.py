# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Skeleton of the port module that stage 2 of a full port calls (orchard/skills/full-port-check.md).

The supervisor's check imports this file as `port_model` and calls the four functions below. Replace each body. Until you do, each one
raises NotImplementedError, so a port left as it is fails the check at its first step with a message that says so.

`stages/2/port_facts.json` lists the config fields, the layer schedule and every tensor's dtype and shape. `read_tensors` returns the
checkpoint's tensors by name whether it is one `model.safetensors` or several shards. The check measures `load` with a timer, so read
and place the weights inside it."""
from __future__ import annotations

import glob
import os
from collections.abc import Mapping

import torch  # noqa: F401  (a port builds torch tensors from the checkpoint before it places them on the device)
import ttnn  # noqa: F401  (the gate refuses a port whose sources do not mention ttnn)
from safetensors import safe_open


class _Tensors(Mapping):
    """The checkpoint as a read-only mapping {name: torch tensor}. A tensor is read from disk only when it is asked for, and it is copied out of the
    memory-mapped file: `get_tensor` returns a view of the mapping, and on the PyPI ttnn 0.80.0 wheel `ttnn.from_torch` of such a view never returns
    (issue 813). Reading one tensor at a time keeps a large checkpoint from being held in host memory twice."""

    def __init__(self, model_dir):
        self._where = {}
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(path, framework="pt") as f:
                for name in f.keys():
                    self._where[name] = path

    def __getitem__(self, name):
        with safe_open(self._where[name], framework="pt") as f:
            return f.get_tensor(name).clone()

    def __contains__(self, name):
        return name in self._where          # the mapping base class would read the tensor to answer this

    def __iter__(self):
        return iter(self._where)

    def __len__(self):
        return len(self._where)


def read_tensors(model_dir):
    """Every tensor of the checkpoint by name, from one file or from shards: `read_tensors(model_dir)[name]` or `.items()`. Each tensor is read and
    copied when it is asked for (see `_Tensors`); keep a tensor in one dict only, as the notes say."""
    return _Tensors(model_dir)


def load(model_dir, mesh_device):
    """Load the weights from the snapshot directory onto the open device and return the model object the other functions take."""
    raise NotImplementedError("port_model.load is not written yet: read stages/2/port_facts.json, then write it")


def prefill(model, ids):
    """Reset the cache, run the prompt (a list of ints) and return the logits for the next token. It must start the request afresh."""
    raise NotImplementedError("port_model.prefill is not written yet")


def decode_step(model, token_id, position):
    """Append `token_id` at `position` (its index in the whole sequence, prompt included) and return the logits for the token after it."""
    raise NotImplementedError("port_model.decode_step is not written yet")


def close(model):
    """Release what `load` took. The check closes the device itself."""
    raise NotImplementedError("port_model.close is not written yet")
