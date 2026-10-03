#!/usr/bin/env python3
"""Boot an installed copy of the staged package and compare its tokens with the CPU reference.

Stage 7 copies this file and serve_and_compare.py (the weights-swap-check template, whose server
and request helpers it reuses) into stages/7/verify/, next to `verify_config.json`. The supervisor
runs it as the stage's hardware test on a leased board, with that board's chips in
TT_VISIBLE_DEVICES.

It tests the package as a consumer gets it. The server is the copy's own `bash run.sh --port P`.
MODEL_WEIGHTS_DIR, HF_MODEL, TT_CACHE_PATH, TT_CACHE_HOME and HF_HUB_CACHE are removed from the
environment, so the bundle must set the weights directory itself and uses its own empty tensor
cache. HF_HOME is a directory that links only the new model and the auxiliary repos the manifest
names, and HF_HUB_OFFLINE=1, so a bundle that tried to load the nearest model's weights would fail.

Before any token is compared it checks that the answers will come from this server:
- exit 6 if something already answers on the port (the check would read another server);
- exit 7 if /v1/models does not list the copy's model-dir;
- exit 8 if the server process's own environment does not set MODEL_WEIGHTS_DIR and HF_MODEL to
  the copy's model-dir (read from /proc/<pid>/environ; run.sh execs the server, so it keeps the pid);
- exit 9 if model-dir/.weights does not name the manifest's weights and revision.
Then: 32 greedy tokens from the stage 1 prompt, and for each of the 32 reference positions one
token with the reference prefix, re-tokenized and compared with the reference id. Exit 4: the
server exited or never became healthy, or exited during the check. Exit 5: an HTTP error. Exit 0:
measured, whatever the numbers say; evidence/verify.json holds them. The server's process group
is stopped on every exit path.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serve_and_compare as sac  # noqa: E402

SCRUBBED_ENV = ("MODEL_WEIGHTS_DIR", "HF_MODEL", "TT_CACHE_PATH", "TT_CACHE_HOME", "HF_HUB_CACHE")


def port_answers(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def served_models(port: int) -> list[str]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as resp:
        return [m.get("id") for m in json.loads(resp.read().decode("utf-8")).get("data", [])]


def process_env(pid: int) -> dict:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    return dict(kv.split("=", 1) for kv in raw.decode("utf-8", "replace").split("\0") if "=" in kv)


def stop(code: int, message: str) -> None:
    print(f"verify_bundle: {message}")
    sys.exit(code)


def main() -> int:
    from tokenizers import Tokenizer      # imported here so a missing package fails before Popen

    cfg = json.loads((HERE / "verify_config.json").read_text(encoding="utf-8"))
    run_dir, bundle, port = Path(cfg["run_dir"]), Path(cfg["bundle"]), int(cfg["port"])
    model_dir = bundle / "model-dir"
    ref = run_dir / "stages" / "1" / "evidence" / "reference"
    prompt_ids = json.loads((ref / "prompt-ids.json").read_text(encoding="utf-8"))["prompt_ids"]
    generated_ids = json.loads((ref / "generated-ids.json").read_text(encoding="utf-8"))["generated_ids"]
    if len(generated_ids) < sac.N_TOKENS:
        stop(2, f"the reference has {len(generated_ids)} generated ids; {sac.N_TOKENS} are needed")
    if port_answers(port):
        stop(6, f"something already answers on port {port}; refusing, so the check cannot read "
                "another server")
    evidence = HERE / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    log_path = evidence / "server.log"
    env = {k: v for k, v in os.environ.items() if k not in SCRUBBED_ENV}
    env.update(HF_HOME=str(cfg["hf_home"]), HF_HUB_OFFLINE="1")
    log = open(log_path, "wb")
    proc = subprocess.Popen(["bash", str(bundle / "run.sh"), "--port", str(port)], env=env,
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    try:
        ready_s = sac.wait_healthy(proc, port, float(cfg["health_timeout_s"]), log_path)
        models = served_models(port)
        if str(model_dir) not in models:
            stop(7, f"the server on port {port} serves {models}, not {model_dir}")
        seen = process_env(proc.pid)
        weights_env = {k: seen.get(k) for k in ("MODEL_WEIGHTS_DIR", "HF_MODEL")}
        if set(weights_env.values()) != {str(model_dir)}:
            stop(8, f"the server process has {weights_env}; both must be {model_dir}")
        want = f"{cfg['model_id']}@{cfg['revision']}"
        mf = model_dir / ".weights"
        marker = mf.read_text(encoding="utf-8") if mf.is_file() else None
        if marker != want:
            stop(9, f"model-dir/.weights is {marker!r}; expected {want!r}")
        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        free_text = sac.complete(port, str(model_dir), prompt_ids, sac.N_TOKENS)
        matches, mismatches = 0, []
        for k in range(sac.N_TOKENS):
            got = sac.complete(port, str(model_dir), prompt_ids + generated_ids[:k], 1)
            ids = tokenizer.encode(got, add_special_tokens=False).ids
            if ids and ids[0] == generated_ids[k]:
                matches += 1
            else:
                mismatches.append({"position": k, "expected_id": generated_ids[k],
                                   "got_id": ids[0] if ids else None, "got_text": got[:300]})
        if proc.poll() is not None:
            stop(4, f"the server exited with code {proc.returncode} during the check")
    except sac.ServerHTTPError as exc:
        stop(5, f"the server returned an error: {exc}")
    finally:
        sac.stop_server(proc)
        log.close()
    coh = sac.coherence(free_text)
    report = {"label": "measured", "model_id": cfg["model_id"], "revision": cfg["revision"],
              "served_model": str(model_dir), "server_weights_env": weights_env,
              "model_dir_weights": marker, "server_ready_s": round(ready_s, 1),
              "n_tokens": sac.N_TOKENS, "matches": matches,
              "top1_agreement": matches / sac.N_TOKENS, **coh, "free_run_text": free_text,
              "mismatches": mismatches,
              "evidence": [os.path.relpath(evidence / "verify.json", run_dir),
                           os.path.relpath(log_path, run_dir)]}
    (evidence / "verify.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("top1_agreement", "coherent", "server_ready_s")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
