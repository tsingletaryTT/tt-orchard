"""The docker command `tt-model serve <container package> ... --print` prints, for the container
template tests (tests/test_container_template.py, tests/fake_tt_model.py).

`printed_argv` follows tt-model's compose_run (tt_kernel/container.py, read on 2026-10-03) for the
4-chip plain package changh95/qwen3.8-27b-p300x2, profile batch32: name, user and labels; one
--device per leased chip; ipc, hugepages; the /hf, /cache, /weight-cache and /tensor-cache
volumes, each beside its variable; the published port; the serve env sorted by name; the image;
then the vllm-plugin launcher's server argv (`vllm serve <weights id> --revision <sha> ...`).
--print composes with detach=False, so the printed command has no --detach.
"""
from __future__ import annotations

PACKAGE = "changh95/qwen3.8-27b-p300x2"
IMAGE = "tt-model/qwen3.8-27b-p300x2:0becf4834925"
NEAREST = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ADDITIONAL = '{"tt": {"fabric_config": "FABRIC_1D", "trace_region_size": 1073741824}}'


def printed_argv(*, hf, pkg_cache, port, device_ids, whole_dir=False, extra_options=(),
                 extra_env=()) -> list[str]:
    devices = (["--device", "/dev/tenstorrent"] if whole_dir else
               [x for d in device_ids for x in ("--device", f"/dev/tenstorrent/{d}:/dev/tenstorrent/{d}")])
    env = {"ARCH_NAME": "blackhole", "HF_HUB_OFFLINE": "1", "HF_MODEL": NEAREST,
           "MESH_DEVICE": "P150x4", "QWEN36_DRAFTER": "mtp", "VLLM_RPC_TIMEOUT": "900000",
           **dict(extra_env)}
    return (["docker", "run", "--name", "tt-model-qwen3.8-27b-p300x2-batch32", "--user", "1000:1000",
             "--label", "org.tenstorrent.tt-model=qwen3.8-27b-p300x2",
             "--label", "org.tenstorrent.tt-model.profile=batch32",
             "--label", "org.tenstorrent.tt-model.devices=" + ",".join(str(d) for d in device_ids)]
            + devices
            + ["--ipc", "host", "--mount", "type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G",
               "--volume", f"{hf}:/hf", "--env", "HF_HOME=/hf",
               "--volume", f"{pkg_cache}/cache:/cache", "--env", "TT_METAL_CACHE=/cache",
               "--volume", f"{pkg_cache}/weights:/weight-cache", "--env", "TT_DIT_CACHE_DIR=/weight-cache",
               "--volume", f"{pkg_cache}/tensors:/tensor-cache", "--env", "TT_CACHE_PATH=/tensor-cache",
               "--publish", f"{port}:{port}"]
            + list(extra_options)
            + [x for k, v in sorted(env.items()) for x in ("--env", f"{k}={v}")]
            + [IMAGE, "vllm", "serve", NEAREST, "--revision", REVISION, "--max-model-len", "262144",
               "--max-num-seqs", "32", "--block-size", "64", "--additional-config", ADDITIONAL,
               "--max-num-batched-tokens", "262144", "--port", str(port)])
