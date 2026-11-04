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

import torch  # noqa: F401  (a port builds torch tensors from the checkpoint before it places them on the device)
import ttnn  # noqa: F401  (the gate refuses a port whose sources do not mention ttnn)
from safetensors import safe_open


def read_tensors(model_dir):
    """Every tensor of the checkpoint as {name: torch tensor}, from one file or from shards."""
    out = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                out[name] = f.get_tensor(name)
    return out


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
