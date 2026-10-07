#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
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

Steps when run: read swap_config.json (keys below) and the supervisor's two variables; refuse a
tt_cache inside ~/.cache/tt-model and apply serve_and_compare.py's cache guard (exit 3); look up
the package's image in ~/.cache/tt-model/installed.json; ask `tt-model serve ... --print` with
HOME and HF_HOME set to the operator's (the shell this runs in has its own HOME); edit the argv and
save it as evidence/docker-argv.json; create every volume source as this user (docker would
create a missing one as root); `docker run`; then measure exactly as serve_and_compare.py does
(its `measure`, with the container standing in for the process) and write evidence/swap-check.json
through its `write_report`. The `try` starts on the line after `docker run` returns, and its
`finally` runs `docker stop` (SIGTERM, then SIGKILL after STOP_GRACE_S), saves `docker logs` to
evidence/server.log, runs `docker rm --force` and checks that docker no longer lists the
container.

Config keys: run_dir, nearest_model_id, new_model_id, tt_cache, hf_home, operator_home, port,
package, profile, chips (and health_timeout_s, test_raise_after_ready for tests, as in
serve_and_compare.py).

Exit codes: 0 when the measurements completed, whatever they say. 2 the config, the supervisor's
variables or the printed command could not be used; nothing was started. 3 the tensor cache was
refused. 4 the container did not start, exited, or never became healthy. 5 the server answered a
request with an HTTP error.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import struct
import subprocess
import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE_DIR))       # serve_and_compare.py is copied next to this file
DOCKER_TIMEOUT_S = 120.0
STOP_GRACE_S = 60
REQUIRED = ("run_dir", "nearest_model_id", "new_model_id", "tt_cache", "hf_home", "operator_home",
            "port", "package", "profile", "chips")
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


# The same check as prepare_swap.py (each template is copied into the stage directory on its own, and a test
# keeps the two sources identical).
MTP_NAME = re.compile(r"(^|\.)mtp\.")


def shard_tensor_names(path: Path) -> list[str] | None:
    """Tensor names from a safetensors header, or None when the file is not readable as one."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return None
            (n,) = struct.unpack("<Q", raw)
            if not 0 < n <= 100 * 1024 * 1024:
                return None
            header = json.loads(f.read(n).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return [k for k in header if k != "__metadata__"] if isinstance(header, dict) else None


def has_mtp_tensors(new: Path) -> bool | None:
    """Does the new model hold `mtp.*` tensors? Read from its weight index, else from the shard headers.
    None when neither can be read, so the caller can leave things alone."""
    try:
        weight_map = json.loads((new / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        if isinstance(weight_map, dict):
            return any(MTP_NAME.search(k) for k in weight_map)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    shards = sorted(new.glob("*.safetensors"))
    if not shards:
        return None
    names = []
    for shard in shards:
        got = shard_tensor_names(shard)
        if got is None:
            return None
        names += got
    return any(MTP_NAME.search(k) for k in names)


def serving_from_argv(argv: list[str]) -> dict:
    """The same record as serve_and_compare.serving_state, read from the edited `docker run` argv: the
    `--env QWEN36_DRAFTER=` option and the server's `--additional-config`."""
    drafter = "not set by the package"
    for i, a in enumerate(argv[:-1]):
        if a == "--env" and argv[i + 1].startswith("QWEN36_DRAFTER="):
            value = argv[i + 1].split("=", 1)[1]
            drafter = "off" if value == "" else f"on ({value})"
    config = next((argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--additional-config"), None)
    if config is None:
        sampling = "not set by the package"
    else:
        try:
            tt = json.loads(config).get("tt")
        except (ValueError, AttributeError):
            tt = "unknown"
        mode = tt.get("sample_on_device_mode") if isinstance(tt, dict) else None
        sampling = "unknown" if tt == "unknown" else (f"on device ({mode})" if mode else "host")
    return {"drafter": drafter, "sampling": sampling}


def edit_docker_argv(argv: list[str], *, image: str, nearest: str, model_dir, tt_cache, hf_dir,
                     name: str, label: str, device_ids: list[int], port: int,
                     blobs: list[str], drafter_off: bool = False) -> list[str]:
    """The edited `docker run` argv (see the module docstring). Raises EditError. With `drafter_off`,
    `--env QWEN36_DRAFTER=<x>` becomes `--env QWEN36_DRAFTER=` (at most one such option): the speculative
    drafter needs the model's mtp.* tensors, and the engine dies at start without them."""
    if argv[:2] != ["docker", "run"]:
        raise EditError(f"the printed command does not start with 'docker run': {argv[:2]}")
    hits = [i for i, a in enumerate(argv) if a == image]
    _once(len(hits), f"image {image!r}")
    opts, server = argv[2:hits[0]], list(argv[hits[0] + 1:])
    md = str(model_dir)
    out = ["docker", "run", "--detach"]
    seen = {"--name": 0, "/tensor-cache": 0, "/hf": 0, "HF_MODEL": 0, "MODEL_WEIGHTS_DIR": 0,
            "TT_CACHE_PATH": 0, "QWEN36_DRAFTER": 0}
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
            elif key == "QWEN36_DRAFTER" and drafter_off:
                seen[key] += 1
                out += [opt, "QWEN36_DRAFTER="]
            elif key != "HF_TOKEN":
                out += [opt, val]
        else:
            out += [opt, val]
    _once(seen["--name"], "--name")
    _once(seen["/tensor-cache"], "--volume <dir>:/tensor-cache")
    _once(seen["/hf"], "--volume <dir>:/hf")
    _once(seen["TT_CACHE_PATH"], "--env TT_CACHE_PATH=/tensor-cache")
    _once(seen["HF_MODEL"], "--env HF_MODEL=<id>")
    if seen["QWEN36_DRAFTER"] > 1:
        raise EditError(f"expected at most one --env QWEN36_DRAFTER, found {seen['QWEN36_DRAFTER']}")
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


# ---- running the test ----------------------------------------------------------------------------

def fail(message: str, code: int = 2) -> None:
    print(f"serve_and_compare_container: {message}", file=sys.stderr)
    sys.exit(code)


def docker(args: list[str], timeout: float = DOCKER_TIMEOUT_S) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


class Container:
    """The started container, with what serve_and_compare.measure asks of a Popen: poll() and
    returncode."""

    def __init__(self, cid: str, log_path: Path):
        self.cid, self.log_path, self.returncode = cid, log_path, None

    def save_log(self) -> None:
        r = docker(["logs", self.cid])
        self.log_path.write_text(r.stdout + r.stderr, encoding="utf-8")

    def poll(self):
        r = docker(["inspect", "--format", "{{.State.Running}} {{.State.ExitCode}}", self.cid])
        parts = r.stdout.split()
        if r.returncode == 0 and parts[:1] == ["true"]:
            return None
        self.save_log()                 # measure prints the log's tail when the server is gone
        self.returncode = int(parts[1]) if r.returncode == 0 and len(parts) == 2 else -1
        return self.returncode

    def stop(self) -> bool:
        """Stop, save the log, remove. True when docker no longer lists the container."""
        docker(["stop", "-t", str(STOP_GRACE_S), self.cid], timeout=STOP_GRACE_S + DOCKER_TIMEOUT_S)
        self.save_log()
        docker(["rm", "--force", self.cid])
        left = docker(["ps", "--all", "--quiet", "--filter", f"id={self.cid}"])
        return left.returncode == 0 and not left.stdout.strip()


def inside(path: Path, root: Path) -> bool:
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def package_image(cfg: dict) -> str:
    path = Path(cfg["operator_home"]) / ".cache" / "tt-model" / "installed.json"
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))[cfg["package"]]
    except (OSError, ValueError, KeyError) as exc:
        fail(f"{cfg['package']} is not installed according to {path}: {exc!r}")
    if not entry.get("container") or not entry.get("image"):
        fail(f"{cfg['package']} is not a container package in {path}")
    return entry["image"]


def printed_command(cfg: dict, ids: list[int]) -> list[str]:
    env = dict(os.environ, HOME=str(cfg["operator_home"]), HF_HOME=str(cfg["hf_home"]))
    for key in ("HF_HUB_CACHE", "HF_TOKEN"):
        env.pop(key, None)
    argv = ["tt-model", "serve", cfg["package"], "--local-only", "--no-update-check", "--port",
            str(cfg["port"]), "--profile", cfg["profile"], "--device-id",
            ",".join(str(i) for i in ids), "--print"]
    r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=DOCKER_TIMEOUT_S)
    if r.returncode != 0:
        fail(f"{shlex.join(argv)} exited {r.returncode}: {(r.stderr or r.stdout)[-2000:]}")
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("docker run ")]
    if len(lines) != 1:
        fail(f"expected one 'docker run' line from {shlex.join(argv)}, found {len(lines)}:\n"
             f"{r.stdout[-2000:]}")
    return shlex.split(lines[0])


def main() -> int:
    from tokenizers import Tokenizer      # imported here so a missing package fails before docker
    from serve_and_compare import (ServerHTTPError, guard_cache, load_reference, measure,
                                   write_report)

    try:
        cfg = json.loads((STAGE_DIR / "swap_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"cannot read swap_config.json: {exc}")
    missing = [k for k in REQUIRED if cfg.get(k) in (None, "")]
    if missing:
        fail(f"swap_config.json is missing {missing}")
    label = os.environ.get("ORCHARD_TEST_LABEL", "")
    try:
        ids = [int(i) for i in os.environ.get("ORCHARD_DEVICE_IDS", "").split(",")]
    except ValueError:
        ids = []
    if not label or len(ids) != int(cfg["chips"]):
        fail(f"the supervisor sets ORCHARD_TEST_LABEL and ORCHARD_DEVICE_IDS ({cfg['chips']} ids) "
             f"for a hardware test; got {label!r} and {os.environ.get('ORCHARD_DEVICE_IDS')!r}")
    run_dir = Path(cfg["run_dir"]).resolve()
    reference = load_reference(run_dir)
    model_dir = STAGE_DIR / "model-dir"
    if not (model_dir / "tokenizer.json").exists():
        fail(f"{model_dir} has no tokenizer.json; run prepare_swap.py first")
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    cache = Path(cfg["tt_cache"])
    shared = Path(cfg["operator_home"]) / ".cache" / "tt-model"
    if inside(cache, shared):
        fail(f"tt_cache {cache} is inside {shared}, where the packages keep their own tensor "
             "caches. Use a new directory for this model and this configuration.", 3)
    guard_cache(cache, cfg["new_model_id"])
    image = package_image(cfg)
    hf_dir = STAGE_DIR / "hf-isolated"
    hf_dir.mkdir(exist_ok=True)
    try:
        argv = edit_docker_argv(printed_command(cfg, ids), image=image,
                                nearest=cfg["nearest_model_id"], model_dir=model_dir,
                                tt_cache=cache, hf_dir=hf_dir,
                                name=f"orchard-{cfg['chips']}chip-{cfg['port']}", label=label,
                                device_ids=ids, port=int(cfg["port"]), blobs=blob_dirs(model_dir),
                                drafter_off=(has_mtp_tensors(model_dir) is False))
    except EditError as exc:
        fail(str(exc))
    evidence = STAGE_DIR / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "docker-argv.json").write_text(json.dumps(argv, indent=1), encoding="utf-8")
    for flag, value in zip(argv, argv[1:]):
        if flag == "--volume":
            Path(value.split(":")[0]).mkdir(parents=True, exist_ok=True)
    log_path = evidence / "server.log"
    r = docker(argv[1:])
    if r.returncode != 0:
        print(f"serve_and_compare_container: docker run exited {r.returncode}: {r.stderr[-2000:]}")
        return 4
    container = Container(r.stdout.split()[-1], log_path)
    stopped = False
    try:
        m = measure(cfg, container, log_path, tokenizer, reference[0], reference[1])
    except ServerHTTPError as exc:
        print(f"serve_and_compare_container: the server returned an error: {exc}")
        sys.exit(5)
    finally:
        stopped = container.stop()
        if not stopped:
            print(f"serve_and_compare_container: docker still lists container {container.cid} "
                  "after stop and rm; the supervisor looks for it by its label", file=sys.stderr)
    weights_env = {"MODEL_WEIGHTS_DIR": str(model_dir), "HF_MODEL": str(model_dir)}
    write_report(cfg, run_dir, cache, weights_env, m, tokenizer, reference, log_path,
                 extra={"kind": "container", "package": cfg["package"], "profile": cfg["profile"],
                        "chips": int(cfg["chips"]), "device_ids": ids, "docker_argv": argv,
                        "hf_isolated": str(hf_dir), "container_stopped": stopped,
                        "serving": serving_from_argv(argv)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
