# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A stand-in for the `tt-model` CLI in the stage 7 tests. It opens no device and uploads nothing.

Every call appends its arguments as one JSON line to $FAKE_TT_MODEL_LOG. Only
`package-thin ... --out DIR` does anything: it writes a minimal v6 bundle shaped like the real one
(tt_kernel_manifest.json with the build host's name in producer.hostname, run.sh with the generated
command layout, an install.sh that is not executable, the wheels it was given, model.py,
requirements.txt and vllm_models/<name>/vllm_metadata.json). Any other subcommand exits 1.

install.sh makes venv/bin/python a small wrapper, using three variables it reads when it runs (so
the staged install.sh names no path on this machine): $FAKE_TT_MODEL_PYTHON, $FAKE_SWAP_SERVER and
$FAKE_SWAP_CONFIG. Run as `python -m vllm.entrypoints.openai.api_server`, the wrapper execs the fake
server with that config, so the server keeps run.sh's pid and environment; run any other way, it
execs the real interpreter.
$FAKE_TT_MODEL_FAIL makes package-thin print an error and exit 3.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
from pathlib import Path

FAKE_SERVER = Path(__file__).resolve().with_name("fake_swap_server.py")   # for FAKE_SWAP_SERVER


def options(argv):
    opts, positional, i = {}, [], 0
    while i < len(argv):
        if argv[i].startswith("--"):
            opts.setdefault(argv[i], []).append(argv[i + 1])
            i += 2
        else:
            positional.append(argv[i])
            i += 1
    return opts, positional


RUN_SH = """#!/usr/bin/env bash
# Serve this model on TT hardware. Assumes ./install.sh has been run.
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
VENV="${{VENV:-$HERE/venv}}"
PYBIN="$VENV/bin/python"
export PYTHONPATH="$HERE:${{PYTHONPATH:-}}"
export HF_HOME="${{HF_HOME:-$HERE/.hf}}"
export TT_CACHE_PATH="${{TT_CACHE_PATH:-$HERE/.tt_cache}}"
export TT_CACHE_HOME="${{TT_CACHE_HOME:-$HERE/.tt_cache}}"
export HF_MODEL="${{HF_MODEL:-{weights}}}"
export TT_MODEL_WEIGHTS_REVISION="${{TT_MODEL_WEIGHTS_REVISION:-{rev}}}"
{exports}
CMD=("$PYBIN" -m vllm.entrypoints.openai.api_server --model "{weights}" --max_num_seqs {seqs} --block_size {block} --revision {rev} --tokenizer-revision {rev} --max_model_len {ctx} "$@")
if [ "${{TT_MODEL_PRINT:-0}}" = "1" ]; then
  printf 'HF_MODEL=%s\\n  %s\\n' "${{HF_MODEL:-}}" "${{CMD[*]}}"
  exit 0
fi
exec "${{CMD[@]}}"
"""

INSTALL_SH = """#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/venv/bin"
cat > "$HERE/venv/bin/python" <<WRAP
#!/bin/sh
if [ "\\$1" = "-m" ] && [ "\\$2" = "vllm.entrypoints.openai.api_server" ]; then
  shift 2
  exec "${FAKE_TT_MODEL_PYTHON:?}" "${FAKE_SWAP_SERVER:?}" --config "${FAKE_SWAP_CONFIG:?}" "\\$@"
fi
exec "${FAKE_TT_MODEL_PYTHON:?}" "\\$@"
WRAP
chmod +x "$HERE/venv/bin/python"
echo "installed into $HERE/venv"
"""


def package_thin(opts, positional) -> int:
    if os.environ.get("FAKE_TT_MODEL_FAIL"):
        print("fake tt-model: SFPI 7.83.0 does not match the wheel's 7.73.0", file=sys.stderr)
        return 3
    out = Path(opts["--out"][0])
    out.mkdir(parents=True)
    name, weights, rev = opts["--name"][0], opts["--weights"][0], opts["--weights-revision"][0]
    env = dict(e.split("=", 1) for e in opts.get("--env", []))
    (out / "wheels").mkdir()
    plugin = [shutil.copy(w, out / "wheels") for w in opts.get("--plugin-wheel", [])]
    ops = [shutil.copy(w, out / "wheels") for w in opts.get("--ops-wheel", [])]
    models = [shutil.copy(w, out / "wheels") for w in opts.get("--models-wheel", [])]
    shutil.copy(opts["--model-py"][0], out / "model.py")
    shutil.copy(opts["--requirements"][0], out / "requirements.txt")
    meta = json.loads(Path(opts["--metadata"][0]).read_text())
    (out / "vllm_models" / name).mkdir(parents=True)
    (out / "vllm_models" / name / "vllm_metadata.json").write_text(json.dumps(meta, indent=2))
    rel = lambda paths: [f"wheels/{Path(p).name}" for p in paths]        # noqa: E731
    manifest = {
        "schema_version": "6", "name": name, "arch": opts["--arch"][0],
        "device_count": int(opts["--device-count"][0]),
        "producer": {"tt_kernel_version": "0.1.0", "created_at": "2026-10-03T00:00:00+00:00",
                     "hostname": socket.gethostname()},
        "weights": {"repo_id": weights, "revision": rev, "allow_patterns": None,
                    "ignore_patterns": None, "repo_type": "model"},
        "mesh": {"devices": int(opts["--device-count"][0]), "topology": opts["--mesh"][0], "fabric": None},
        "entrypoint": {"cls": meta["main_class"], "arch_name": meta["arch"]},
        "resources": {"max_model_len": int(opts["--max-model-len"][0]),
                      "max_num_seqs": int(opts["--max-num-seqs"][0]),
                      "block_size": int(opts["--block-size"][0]), "extra_args": []},
        "env": env,
        "deps": {"python": opts.get("--python", ["3.12"])[0], "requirements": "requirements.txt",
                 "wheels": rel(plugin + ops), "wheels_dir": "wheels", "models_wheels": rel(models),
                 "vllm": {"version": opts["--vllm-version"][0], "target_device": "empty"},
                 "model_dir": ".", "kind": "vllm"},
    }
    (out / "tt_kernel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    exports = "\n".join(f'export {k}="{v}"' for k, v in env.items())
    (out / "run.sh").write_text(RUN_SH.format(
        weights=weights, rev=rev, exports=exports, seqs=opts["--max-num-seqs"][0],
        block=opts["--block-size"][0], ctx=opts["--max-model-len"][0]))
    (out / "install.sh").write_text(INSTALL_SH)
    for f in ("run.sh", "install.sh"):
        (out / f).chmod(0o600)              # the real package-thin leaves both without +x
    print(f"staged v6 thin bundle {name} at {out}")
    return 0


def main(argv) -> int:
    log = os.environ.get("FAKE_TT_MODEL_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(argv) + "\n")
    if argv[:1] == ["package-thin"]:
        return package_thin(*options(argv[1:]))
    print(f"fake tt-model: {argv[:1]} is not faked", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
