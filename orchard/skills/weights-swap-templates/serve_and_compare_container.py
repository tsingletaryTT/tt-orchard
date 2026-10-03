#!/usr/bin/env python3
"""Serve the swapped weights with a container package and compare the chip's tokens with the CPU
reference (stage 4 on the weights-only path).

The weights-swap-configs skill copies this file, serve_and_compare.py and prepare_swap.py into one
configuration directory, stages/4/configs/<chips>/, writes swap_config.json there and runs
prepare_swap.py, which builds model-dir/ (for a container package it builds no run.sh). The
supervisor runs this file as one of the stage's hardware tests, on leased chips. It names the
chips twice: TT_VISIBLE_DEVICES (PCI addresses) and ORCHARD_DEVICE_IDS (the /dev/tenstorrent
indices of the same chips, which docker needs). It also sets ORCHARD_TEST_LABEL, a docker label
for every container a test of this run starts, so the supervisor can find a container this script
failed to stop.

`tt-model serve` takes no extra docker arguments or environment. So this script asks it for the
command it would run (`tt-model serve <package> ... --print`, which prints and starts nothing),
edits that argv, and runs the edited `docker run` itself. Editing the argv with exact counts is
safer than a sed over the printed text: each edit names the token it changes, and a printed
command whose shape this script does not know exits 2 before anything starts.

The edits (each must apply exactly once unless it says otherwise):
- `--detach` is added (the printed command has none), so `docker run` returns the container id.
- `--name` becomes `orchard-<chips>chip-<port>`. The test container then never takes the name a
  `tt-model serve` of the same package would use, and `tt-model stop` never finds it.
- Every `--label` is dropped and ORCHARD_TEST_LABEL is added. The tt-model labels would make the
  container look like a tt-model server of that package to `tt-model stop` and to tt-model's
  free-chip scan.
- Each `--device` must be one `/dev/tenstorrent/<N>` node, and the N must be exactly
  ORCHARD_DEVICE_IDS. A mapping of the whole `/dev/tenstorrent` directory exits 2: a container
  that opens a device it was not leased can crash another tenant's chips.
- `--env TT_CACHE_PATH` must be `/tensor-cache`, and the `/tensor-cache` volume's source becomes
  `tt_cache`, a new directory for this model and this configuration. The package's own cache
  (~/.cache/tt-model/<package>/tensors) is keyed only by layer name and mesh, so it would serve
  the base model's tensors without any error.
- The `/hf` volume's source becomes `hf-isolated/`, an empty directory next to this script. The
  printed command mounts the operator's whole Hugging Face cache there, and that cache holds the
  base model. Without it nothing in the container can load the base weights by their hub id, so a
  runtime that ignores MODEL_WEIGHTS_DIR fails with an error. A `/hf-hub` volume (HF_HUB_CACHE
  outside HF_HOME) exits 2.
- `model-dir/` is mounted read-only at its own absolute path, and so is each directory its links
  point into (the new model's blobs). The links then resolve inside the container as they do on
  the host, and the served model name is the model-dir path, as in serve_and_compare.py.
- `--env HF_MODEL=...` becomes the model-dir path, and `--env MODEL_WEIGHTS_DIR=<model-dir>` is
  set (replaced if present, added if not). The TT runtime takes its weights directory from
  MODEL_WEIGHTS_DIR, then HF_MODEL, then the config path.
- `--env HF_TOKEN` is dropped if present; a test container gets no token.
- In the server's arguments after the image, the nearest model's id becomes the model-dir path,
  and `--revision <sha>` and `--tokenizer-revision <sha>` are deleted (at most one of each),
  because a local directory has no revision. The server's `--port` must equal the config's port.

Any option this script does not know exits 2. A new option in tt-model's output must be read by a
person before a test trusts it.
"""
from __future__ import annotations

import os
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
TT_DEVICE = "/dev/tenstorrent"
VALUE_OPTIONS = {"--name", "--user", "--label", "--device", "--ipc", "--mount", "--volume", "--env",
                 "--publish"}
FLAG_OPTIONS = {"--detach", "--rm"}


class EditError(Exception):
    """The printed command is not the shape this script edits. Nothing has started."""


def blob_dirs(model_dir: Path) -> list[str]:
    """The directories model-dir's links point into, sorted. prepare_swap.py writes absolute,
    fully resolved link targets, so these are the new model's blob directories."""
    return sorted({os.path.dirname(os.path.realpath(p)) for p in model_dir.iterdir() if p.is_symlink()})


def _once(count: int, what: str) -> None:
    if count != 1:
        raise EditError(f"expected exactly one {what} in the printed command, found {count}")


def edit_docker_argv(argv: list[str], *, image: str, nearest: str, model_dir, tt_cache, hf_dir,
                     name: str, label: str, device_ids: list[int], port: int,
                     blobs: list[str]) -> list[str]:
    """The edited `docker run` argv (see the module docstring). Raises EditError."""
    if argv[:2] != ["docker", "run"]:
        raise EditError(f"the printed command does not start with 'docker run': {argv[:2]}")
    hits = [i for i, a in enumerate(argv) if a == image]
    _once(len(hits), f"image {image!r}")
    opts, server = argv[2:hits[0]], list(argv[hits[0] + 1:])
    md = str(model_dir)
    out = ["docker", "run", "--detach"]
    seen = {"--name": 0, "/tensor-cache": 0, "/hf": 0, "HF_MODEL": 0, "MODEL_WEIGHTS_DIR": 0,
            "TT_CACHE_PATH": 0}
    devices = []
    i = 0
    while i < len(opts):
        opt = opts[i]
        if opt in FLAG_OPTIONS:
            if opt == "--rm":
                out.append(opt)
            i += 1                                    # --detach is already in `out`
            continue
        if opt not in VALUE_OPTIONS or i + 1 >= len(opts):
            raise EditError(f"unknown option {opt!r} in the printed command; read it before "
                            "trusting this script with it")
        val = opts[i + 1]
        i += 2
        if opt == "--name":
            seen["--name"] += 1
            out += [opt, name]
        elif opt == "--label":
            continue
        elif opt == "--device":
            src = val.split(":")[0]
            if not src.startswith(TT_DEVICE + "/"):
                raise EditError(f"--device {val} is not one /dev/tenstorrent/<N> node; refusing a "
                                "container that can open chips it was not leased")
            devices.append(src[len(TT_DEVICE) + 1:])
            out += [opt, val]
        elif opt == "--volume":
            rest = val.partition(":")[2]
            dst = rest.split(":")[0]
            if dst == "/hf-hub":
                raise EditError("the printed command mounts /hf-hub (HF_HUB_CACHE is outside "
                                "HF_HOME); unset HF_HUB_CACHE for this test")
            if dst in ("/tensor-cache", "/hf"):
                seen[dst] += 1
                out += [opt, f"{tt_cache if dst == '/tensor-cache' else hf_dir}:{rest}"]
            else:
                out += [opt, val]
        elif opt == "--env":
            key = val.split("=", 1)[0]
            if key in ("HF_MODEL", "MODEL_WEIGHTS_DIR"):
                seen[key] += 1
                out += [opt, f"{key}={md}"]
            elif key == "TT_CACHE_PATH":
                if val != "TT_CACHE_PATH=/tensor-cache":
                    raise EditError(f"--env {val}: the tensor cache must be the /tensor-cache volume")
                seen[key] += 1
                out += [opt, val]
            elif key != "HF_TOKEN":
                out += [opt, val]
        else:
            out += [opt, val]
    _once(seen["--name"], "--name")
    _once(seen["/tensor-cache"], "--volume <dir>:/tensor-cache")
    _once(seen["/hf"], "--volume <dir>:/hf")
    _once(seen["TT_CACHE_PATH"], "--env TT_CACHE_PATH=/tensor-cache")
    _once(seen["HF_MODEL"], "--env HF_MODEL=<id>")
    if seen["MODEL_WEIGHTS_DIR"] > 1:
        raise EditError(f"expected at most one --env MODEL_WEIGHTS_DIR, found {seen['MODEL_WEIGHTS_DIR']}")
    want = sorted(str(d) for d in device_ids)
    if sorted(devices) != want:
        raise EditError(f"the printed command maps devices {sorted(devices)}; the lease gave {want}")
    out += ["--label", label, "--volume", f"{md}:{md}:ro"]
    for d in blobs:
        out += ["--volume", f"{d}:{d}:ro"]
    if seen["MODEL_WEIGHTS_DIR"] == 0:
        out += ["--env", f"MODEL_WEIGHTS_DIR={md}"]
    out.append(image)
    model_hits = [j for j, a in enumerate(server) if a == nearest]
    _once(len(model_hits), f"server argument {nearest!r}")
    server[model_hits[0]] = md
    for flag in ("--revision", "--tokenizer-revision"):
        at = [j for j, a in enumerate(server) if a == flag]
        if len(at) > 1:
            raise EditError(f"expected at most one {flag} in the server arguments, found {len(at)}")
        if at:
            del server[at[0]:at[0] + 2]
    ports = [server[j + 1] for j, a in enumerate(server[:-1]) if a == "--port"]
    if ports != [str(port)]:
        raise EditError(f"the server's --port is {ports}; the config's port is {port}")
    return out + server
