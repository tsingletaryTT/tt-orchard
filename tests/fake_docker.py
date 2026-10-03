"""A stand-in for `docker`, for tests/test_container_template.py. It opens no device.

State lives in the directory $FAKE_DOCKER_STATE: config.json ({"image", "fake_server",
"server_config"}), one containers/<id>.json per container docker would still list, a log per
container and runs.jsonl (every `docker run` argv). Supported:

- `run --detach ... <image> <server args>`: refuses (exit 125) a volume whose source does not
  exist, and a link in an identically mounted directory whose target lies outside every mounted
  source (it would not resolve inside a real container). Otherwise it starts
  tests/fake_swap_server.py in its own session with only the `--env` variables (plus PATH), the
  published port and the model the server arguments name, and prints the container id.
- `inspect --format ... <id>`: "true 0" while the server runs, "false 1" after; exit 1 if unknown.
- `logs <id>`, `stop -t N <id>` (SIGTERM to the group, then SIGKILL), `rm --force <id>`.
- `ps --all --quiet --filter id=<id>` and `ps --quiet --filter label=<label>`.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

STATE = Path(os.environ.get("FAKE_DOCKER_STATE", "/nonexistent"))
FLAGS = {"--detach", "--rm"}


def records() -> dict:
    d = STATE / "containers"
    d.mkdir(parents=True, exist_ok=True)
    return {p.stem: json.loads(p.read_text()) for p in d.glob("*.json")}


def alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run(args: list[str]) -> int:
    cfg = json.loads((STATE / "config.json").read_text())
    with open(STATE / "runs.jsonl", "a") as f:
        f.write(json.dumps(args) + "\n")
    k = args.index(cfg["image"])
    opts, server = args[:k], args[k + 1:]
    env, volumes, labels, port, i = {}, [], [], None, 0
    while i < len(opts):
        if opts[i] in FLAGS:
            i += 1
            continue
        flag, val = opts[i], opts[i + 1]
        i += 2
        if flag == "--env":
            key, _, value = val.partition("=")
            env[key] = value
        elif flag == "--volume":
            volumes.append(val.split(":"))
        elif flag == "--label":
            labels.append(val)
        elif flag == "--publish":
            port = val.split(":")[0]
    for src, *_ in volumes:
        if not os.path.isdir(src):
            print(f"docker: bind source path does not exist: {src}", file=sys.stderr)
            return 125
    sources = [os.path.realpath(v[0]) for v in volumes]
    for src, dst, *_ in volumes:
        if src != dst:
            continue
        for p in Path(src).iterdir():
            target = os.path.realpath(p)
            if p.is_symlink() and not any(os.path.commonpath([target, s]) == s for s in sources):
                print(f"docker: {p} links to {target}, which no volume mounts", file=sys.stderr)
                return 125
    cid = f"fakecid{len((STATE / 'runs.jsonl').read_text().splitlines()):04d}"
    log = open(STATE / f"{cid}.log", "wb")
    model = server[server.index("serve") + 1]
    proc = subprocess.Popen([sys.executable, cfg["fake_server"], "--config", cfg["server_config"],
                             "--port", port, "--model", model],
                            env={**env, "PATH": os.environ.get("PATH", "")}, stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    (STATE / "containers" / f"{cid}.json").write_text(json.dumps(
        {"pid": proc.pid, "labels": labels, "argv": args}))
    print(cid)
    return 0


def stop_group(pid: int, wait_s: float = 5.0) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.monotonic() + wait_s
    while time.monotonic() < end and alive(pid):
        time.sleep(0.05)
    if alive(pid):
        os.killpg(pid, signal.SIGKILL)


def main() -> int:
    args = sys.argv[1:]
    cmd, rest = args[0], args[1:]
    recs = records()
    if cmd == "run":
        return run(rest)
    if cmd == "ps":
        filt = rest[rest.index("--filter") + 1]
        key, _, value = filt.partition("=")
        for cid, rec in recs.items():
            if (key == "id" and cid == value) or (key == "label" and value in rec["labels"]):
                print(cid)
        return 0
    cid = rest[-1]
    if cid not in recs:
        print(f"Error: No such container: {cid}", file=sys.stderr)
        return 1
    pid = recs[cid]["pid"]
    if cmd == "inspect":
        print("true 0" if alive(pid) else "false 1")
    elif cmd == "logs":
        sys.stdout.write((STATE / f"{cid}.log").read_text(errors="replace"))
    elif cmd == "stop":
        stop_group(pid)
        print(cid)
    elif cmd == "rm":
        stop_group(pid, wait_s=0.0)
        (STATE / "containers" / f"{cid}.json").unlink()
        print(cid)
    else:
        print(f"fake docker: {cmd} is not supported", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
