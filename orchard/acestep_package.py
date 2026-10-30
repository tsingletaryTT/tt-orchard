# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage and check the ACE-Step 1.5 turbo package: a local, weights-free package that runs the model on one Blackhole chip behind a small HTTP server.

    python -m orchard.acestep_package stage --evidence DIR --bundle-code CODE_DIR --snapshot WEIGHTS_DIR --out PKG_DIR
    python -m orchard.acestep_package summary --sweep SWEEP.json --bars BARS.json [--bars BARS2.json] --agreement AGREE.json --boot BOOT.json --out summary.json
    python -m orchard.acestep_package gate PKG_DIR

It follows `orchard/granite_package.py`: `stage` copies the runtime and the tools, copies the evidence, and writes the card, `LICENSE`, `NOTICE`, `SECURITY.md`,
`tt-model.yaml`, `run.sh`, `requirements.txt`, `pins.json`, `package-record.json` and `SHA256SUMS`; `gate` reads the directory as it is on disk and returns every problem.
The card's number rows are produced from `evidence/summary.json` by this module, and the gate looks for each row in the card.

What the package does not contain, and why:
  * the model weights (MIT; the user downloads revision `REVISION` of `ACE-Step/Ace-Step1.5`);
  * the code of the community bundle `sunandbean/acestep-1.5-turbo-p100a` at `BUNDLE_REVISION` (Apache 2.0): ACE-Step's own Python, tt-metal's `models/` and the
    chip port `tt_acestep`. The package runs it from the user's copy. `pins.json` holds a hash over every file of that copy and `tools/check_bundle.py` compares it, so
    a changed bundle is refused before anything starts. The hash is computed at staging from the copy the package was tested with.
`pins.json` also holds the size and sha256 of each weight file.

`gate` checks: no weights, links, caches or oversized files (the audio package's checks), no home path, host name, token or key file name in any text file, the required
files, Python files compile and `run.sh` passes `bash -n`, licence texts, the card's front matter and sections, every card number row against the summary, the manifest,
the recorded file hashes and `SHA256SUMS`. It does not judge whether prose is true, and it does not start the server. Nothing here publishes: `publish_commands` returns text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from orchard import audio_package as ap
from orchard import resident

MODEL_ID = "ACE-Step/Ace-Step1.5"
REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
BUNDLE_REPO = "sunandbean/acestep-1.5-turbo-p100a"
BUNDLE_REVISION = "dabe15cd44f9fb14e948e111fb671d7ab89fda17"
NAMESPACE = "episod"
PACKAGE_NAME = "ace-step-1.5-turbo-blackhole"
PORT = 20120
VERSION = "0.1.0"
TT_METAL_TAG = "v0.80.0"
ROOT = Path(__file__).resolve().parent.parent
SRC_FILES = (("scripts/acestep/acestep_run.py", "runtime/acestep_run.py"), ("scripts/acestep/acestep_serve.py", "runtime/acestep_serve.py"),
             ("orchard/audio_reference.py", "runtime/orchard/audio_reference.py"), ("orchard/pinned_files.py", "runtime/orchard/pinned_files.py"),
             ("pkgsrc/acestep/tools/check_bundle.py", "tools/check_bundle.py"), ("pkgsrc/acestep/tools/smoke.py", "tools/smoke.py"))
RUNTIME_INIT = '"""A stand-in package so the copied runtime files import `orchard.audio_reference` from this directory."""\n'
WEIGHT_FILES = ("acestep-v15-turbo/model.safetensors", "vae/diffusion_pytorch_model.safetensors", "Qwen3-Embedding-0.6B/model.safetensors")
EVIDENCE_FILES = ("sweep.json", "bars-lofi.json", "bars-folk.json", "agreement.json", "boot.json")
REQUIRED = ("README.md", "LICENSE", "NOTICE", "SECURITY.md", "CHANGELOG.md", "SHA256SUMS", "tt-model.yaml", "run.sh", "requirements.txt", "pins.json", "package-record.json",
            "evidence/summary.json", "runtime/orchard/__init__.py") + tuple(d for _, d in SRC_FILES) + tuple(f"evidence/{n}" for n in EVIDENCE_FILES)
RECORDED = ("LICENSE", "NOTICE", "evidence/summary.json", "pins.json")
FORBIDDEN_TEXT = (".tt_ai_key", "litellm", "HF_TOKEN=", "Authorization: Bearer", "TT_AI_KEY")
CARD_SECTIONS = ("## Weights and bundle code are not included", "## Try it", "## API", "## Numbers", "## Agreement with the CPU reference", "## Limits", "## Not tested",
                 "## Security", "## License", "## Provenance")
OVERCLAIMS = (r"\bproduction[- ]ready\b", r"\bstate[- ]of[- ]the[- ]art\b", r"\b(?:official|officially|endorsed|certified|verified)\b",
              r"\bfaster than (?:a )?gpu\b", r"\bbest\b", r"\bcommercial[- ]grade\b")
STEP_MARGIN = 2.0


# ---- the summary and the numbers on the card ---------------------------------------------------------------------------------
def _f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _read(path):
    return json.loads(Path(path).read_text())


def make_summary(sweep, bars, agreement, boot) -> dict:
    """The one file the card's numbers come from, built from the raw result files. `sweep` is a sweep JSON of `acestep_run.py --sweep`; `bars` a list of
    `orchard.step_bars` results (one per prompt); `agreement` the output of `acestep_report.py agree`; `boot` the boot-check record (`tools/smoke.py` output
    plus the ready time and the comparison with a recorded clip)."""
    rows = []
    for r in sweep["rows"]:
        runs = r["runs"]
        tot = resident.spread([x["generate_and_decode_s"] for x in runs])
        rows.append({"duration_s": r["duration_s"], "n": len(runs), "warmup": r["warmup"], "min": tot["min"], "median": tot["median"], "max": tot["max"],
                     "dit_median": resident.spread([x["dit_s"] for x in runs])["median"], "vae_median": resident.spread([x["vae_s"] for x in runs])["median"],
                     "label": resident.label_for(len(runs), r["warmup"])})
    prompts = []
    for b in bars:
        prompts.append({"margin": b["margin"], "usable": b["usable"], "chip": [c["chip"] for c in b["chip"]], "floor": [c["floor"] for c in b["chip"]],
                        "bar": list(b["detection_limit_rel_l2"]), "vae": b["vae"], "wrong_step": b["controls"]["wrong_step"]["rel_l2"]})
    return {"sweep": {"rows": rows, "load": sweep.get("load", {}), "seed": sweep.get("seed")}, "step_bars": prompts, "agreement": agreement, "boot": boot,
            "limits": ["The planner language model is not loaded, so there is no thinking mode and no audio-code conditioning; the clip follows the caption and lyrics only.",
                       "Only the diffusion transformer and the VAE decoder run on the chip. The text and lyric encoders, the sampler and the post-processing run on the CPU, and they are most of each request.",
                       "One request runs at a time. The first request at a new duration compiles kernels and takes several seconds longer.",
                       "Seeds do not reproduce across backends: a chip clip and a CPU clip of the same seed are different music of the same kind.",
                       "The longest clip measured is 480 s."],
            "not_tested": ["Whether the music is good, or whether sung words are intelligible. The checks measure length and level only.",
                           "Languages other than English, lyrics longer than two sections, and styles other than the prompts listed in the evidence.",
                           "Any machine other than one Blackhole p300c board, and more than one chip.",
                           "The XL transformer and the 1.7B planner."],
            "changelog": ["First version: the server, the runtime copied from tt-orchard, the bundle and weight pins, the evidence and the card."]}


def speed_rows(s: dict) -> list[str]:
    return [f"| {r['duration_s']:g} s clip | {_f(r['min'])} / {_f(r['median'])} / {_f(r['max'])} s | {_f(r['dit_median'])} s | {_f(r['vae_median'])} s | "
            f"{_f(r['duration_s'] / r['median'], 1)} | {r['label']} (n={r['n']}, warm-up {r['warmup']}) | `evidence/sweep.json` |" for r in s["sweep"]["rows"]]


def load_rows(s: dict) -> list[str]:
    ld = s["sweep"]["load"]
    return [f"| Load, chip open to ready | {_f(ld.get('chip_open_upload_vae_warm_s'), 1)} s | single sample | `evidence/sweep.json` |",
            f"| Server start to ready line, this package (polled every 2 s) | {_f(s['boot'].get('ready_s'), 1)} s | single sample | `evidence/boot.json` |"]


def bar_rows(s: dict) -> list[str]:
    out = []
    for i, p in enumerate(s["step_bars"]):
        name = ("lo-fi, instrumental", "folk, with lyrics")[i] if i < 2 else f"prompt {i + 1}"
        out.append(f"| {name}: chip | " + " | ".join(f"{x:.3f}" for x in p["chip"]) + f" | {p['vae']['chip']:.4f} |")
        out.append(f"| {name}: bfloat16 CPU floor | " + " | ".join(f"{x:.3f}" for x in p["floor"]) + f" | {p['vae']['floor']:.4f} |")
        out.append(f"| {name}: bar ({p['margin']:g} x floor) | " + " | ".join(f"{x:.3f}" for x in p["bar"]) + f" | {p['vae']['bar']:.4f} |")
        out.append(f"| {name}: wrong-step control | " + " | ".join(f"{x:.3f}" for x in p["wrong_step"]) + " | |")
    return out


def agreement_line(s: dict) -> str:
    a = s["agreement"]
    return (f"End to end against the float32 CPU run (every step feeds the next): {a['passed']} of {a['arrays']} recorded arrays pass the provisional bars; "
            f"failing: {', '.join(a['failed']) or 'none'}. Final latent PCC {a['latent_pcc']:.3f}, waveform PCC {a['waveform_pcc']:.3f}.")


def boot_line(s: dict) -> str:
    b = s["boot"]
    return (f"Boot check of this package on one chip: ready in {_f(b['ready_s'], 1)} s; a {b['duration_s']:g} s request returned in {_f(b['request_s'], 2)} s; "
            f"the WAV is {b['sample_rate']} Hz, {b['channels']} channels, {b['clip_duration_s']:.2f} s, and passes the sound checks; "
            f"it {'has the same samples as' if b['identical_to_recorded_clip'] else 'differs from'} the clip recorded in the bring-up run with the same prompt and seed.")


# ---- the card -------------------------------------------------------------------------------------------------------------
def render_card(s: dict) -> str:
    lines = ["---", "license: apache-2.0", f"base_model: {MODEL_ID}", "pipeline_tag: text-to-audio", "tags:", "- blackhole", "- tenstorrent", "- music", "- ace-step", "---", "",
             f"# {PACKAGE_NAME}", "", f"Version {VERSION}. See `CHANGELOG.md`.", "",
             f"> Runs [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) (revision `{REVISION}`), the ACE-Step 1.5 turbo text-to-music model, on one Tenstorrent Blackhole chip. "
             "The model weights are not in this repository. This is a community package and not a release of ACE-Step or Tenstorrent.", "",
             "A local HTTP server that keeps the diffusion transformer (24 layers, 8 steps) and the Oobleck VAE decoder on one chip and makes a 48 kHz stereo clip from a "
             "caption and optional lyrics. The chip port is the community bundle named below; this package adds the server, the checks, the evidence and the pins.", "",
             "## Weights and bundle code are not included", "",
             f"- Weights: [{MODEL_ID}](https://huggingface.co/{MODEL_ID}), MIT, not gated. Revision `{REVISION}`.",
             f"- Chip port and ACE-Step code: the `code/` directory of [{BUNDLE_REPO}](https://huggingface.co/{BUNDLE_REPO}) (Apache 2.0), revision `{BUNDLE_REVISION}`.",
             "- `bash run.sh fetch` downloads the weights into your Hugging Face cache (about 6.5 GB) and the bundle code into `fetched/bundle` as real files. "
             "The bundle code must not be a cache snapshot of links: ACE-Step finds its own model code through its resolved path, and with links it silently runs the model "
             "repository's copy and makes different music (measured: waveform correlation 0.65 against the same prompt and seed with real files). `tools/check_bundle.py` and the server refuse links.",
             "- `bash run.sh check` compares both with `pins.json`: a hash over every bundle file, and the size of each weight file.", "",
             "## Try it", "",
             f"Needs a Tenstorrent Blackhole chip, a tt-metal build with `ttnn` (tested with tt-metal {TT_METAL_TAG}), Python 3.12, torch, numpy and the packages in `requirements.txt`.", "",
             "```bash", "pip install -r requirements.txt", "export TT_METAL_HOME=<your tt-metal checkout>   # used when `import ttnn` does not work yet",
             "bash run.sh fetch                                # once", "bash run.sh check                                # compares the bundle and the weights with pins.json; opens no device",
             "bash run.sh                                      # starts the server; on a shared machine take a chip lease first",
             "python tools/smoke.py --duration 30 --out clip.wav", "```", "",
             f"The server listens on http://127.0.0.1:{PORT} (set `PORT`) and holds the chip while it runs.", "",
             "## API", "",
             "`GET /health` and `POST /v1/music` with a JSON body.", "",
             "- `prompt` (required, at most 2,000 characters), `lyrics` (optional, at most 8,000), `duration` (5 to 480 seconds, default 30), `seed` (integer, default 42). Other fields get a `400`.",
             "- The reply is a WAV (48 kHz, stereo, peak-normalised by ACE-Step) with `X-Timings` (JSON) and `X-Sample-Rate` headers.",
             "- A second request while one is running gets `503` with `Retry-After: 5`. A body over 64 KiB, malformed JSON or a bad field gets a `400`. A failed generation gets a `500` with the error type only.", "",
             "```bash", f"curl -s http://127.0.0.1:{PORT}/v1/music -H 'Content-Type: application/json' \\",
             "  -d '{\"prompt\":\"warm lo-fi hip hop, mellow rhodes piano\",\"duration\":30,\"seed\":42}' -o clip.wav", "```", "",
             "## Numbers", "",
             "Measured on one chip of a Blackhole p300c board, with the weights above, on a host shared with other jobs (load average 2 to 6). "
             "Total is the time of one generate call (text and lyric encoders, 8 transformer steps, decode, post-processing). A label states how many runs are behind a number.", "",
             "| Length | Total, min / median / max | Transformer (chip) | VAE (chip) | Audio seconds per second | Label | Evidence |", "| --- | --- | --- | --- | --- | --- | --- |", *speed_rows(s), "",
             "| Number | Value | Label | Evidence |", "| --- | --- | --- | --- |", *load_rows(s), "",
             "The rest of each total is on the CPU: encoders, sampler, copies between host and chip, and post-processing.", "",
             "## Agreement with the CPU reference", "",
             agreement_line(s), "",
             "That comparison lets each step feed its error into the next. A CPU run with the same two stages in bfloat16 also fails it, by more than the chip does, so the failures come from precision "
             "feeding back through the sampler. They do not point to a chip fault. The per-step check holds the input of every step equal: the chip's transformer receives the float32 CPU run's recorded input of "
             "step k and its velocity is compared with the recorded one (relative L2 error); the decoder receives the CPU run's final latent. The bar of each step is "
             f"{STEP_MARGIN:g} times the error a bfloat16 CPU run makes on the same input. The wrong-step control is the neighbouring step's velocity; it fails the bar at every step.", "",
             "| Run | " + " | ".join(f"step {k}" for k in range(8)) + " | decoder |", "| --- | " + " | ".join("---" for _ in range(9)) + " |", *bar_rows(s), "",
             "A bar is also the largest error that can hide at that step: noise of 25 percent of the velocity gets under the bar at steps 0 and 1. The margin of 2 was chosen after the chip's "
             "decoder came out at 1.6 to 1.9 times the CPU floor, from two prompts and one run each. `evidence/` holds the raw files.", "",
             boot_line(s), "",
             "## Limits", "", *[f"- {x}" for x in s["limits"]], "",
             "## Not tested", "", *[f"- {x}" for x in s["not_tested"]], "",
             "## Security", "",
             "The server has no sign-in, no TLS and no rate limit, and binds 127.0.0.1 by default. `HOST=0.0.0.0` listens on every interface: use it only on a network you trust. "
             "The model loads its own Python files from the weights directory (`trust_remote_code`), so only use weights from the pinned revision. `SECURITY.md` has the rest.", "",
             "## License", "",
             "The model is licensed under the MIT License by the ACE-Step authors (the model repository's metadata says `license: mit`). The bundle code is Apache 2.0 by its publisher. "
             "The code in this package is Tenstorrent's under the Apache License 2.0. `LICENSE` is the Apache License 2.0 text and `NOTICE` names the parties. "
             "The model card says generated music may be used commercially; read it before you rely on that.", "",
             "## Provenance", "",
             f"- Model: {MODEL_ID}, revision `{REVISION}`. Bundle: {BUNDLE_REPO}, revision `{BUNDLE_REVISION}`. Neither is part of this package.",
             f"- Tested with tt-metal {TT_METAL_TAG}. The bundle was built against another tt-metal commit and ran on {TT_METAL_TAG} without a change. `SHA256SUMS` lists the hash of every file.",
             "- `runtime/` and `tools/` come from the tt-orchard repository (tsingletaryTT/tt-orchard); `runtime/orchard/` holds the two modules the runtime imports.",
             "- `tt-model.yaml` is a draft v5.1 manifest of kind `http-server`. It has not been built into an image and this package is not in the tt-model catalog.",
             "- Every number on this card comes from `evidence/summary.json`, which is built from the raw result files beside it.", ""]
    return "\n".join(lines)


NOTICE = f"""{PACKAGE_NAME}

Model: {MODEL_ID} (revision {REVISION}), licensed under the MIT License by its authors. The model weights are not part of this package.
Bundle code: {BUNDLE_REPO} (revision {BUNDLE_REVISION}), licensed under the Apache License, Version 2.0 by its publisher. It is not part of this package.

Runtime, server and tools in this package: Copyright 2026 Tenstorrent USA, Inc., licensed under the Apache License, Version 2.0 (see LICENSE).

Licensed under the Apache License, Version 2.0; you may not use this package except in compliance with the License. Unless required by applicable
law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""

SECURITY = """# Security note

This package is for use on your own machine.

- The server binds 127.0.0.1 by default. It has no sign-in, no TLS and no rate limit. Anyone who can reach its port can use the chip.
- `HOST=0.0.0.0` listens on every network interface. Do that only on a network you trust, or put an authenticating proxy in front.
- One request runs at a time. A second request gets `503` with `Retry-After`. A request body over 64 KiB is refused.
- ACE-Step's loader runs Python files that sit in the weights directory (`trust_remote_code`) and copies its own model files into a private checkpoint directory
  under `ACESTEP_WORK`. Use the pinned revision of the weights and the pinned bundle; `bash run.sh check` compares them with `pins.json`.
- The package holds no weights and no credentials. `run.sh fetch` downloads the pinned revisions with your own Hugging Face environment.
- Report a problem in the tt-orchard repository (tsingletaryTT/tt-orchard) issue tracker.
"""

REQUIREMENTS = """# Python packages the bundle's ACE-Step code needs beyond torch, numpy, soundfile and ttnn. torch, numpy, soundfile and ttnn come from your tt-metal
# environment (tested with torch 2.7.0 on the CPU, numpy 2.5.3 and ttnn 0.80.0). The pins below are the ones in the bundle's own requirements.lock for the
# packages that environment did not already provide; they were installed with --no-deps.
transformers==4.57.6
tokenizers==0.22.2
einx==0.4.3
vector-quantize-pytorch==1.31.6
torch-einops-utils==0.1.36
pytorch-wavelets==1.3.0
modelscope==1.40.2
frozendict==2.4.7
"""

RUN_SH = f"""#!/usr/bin/env bash
# run.sh [serve] | fetch | check | --self-test
#   serve (default)  start the HTTP server on one chip. On a shared machine, take a chip lease first; this script resets nothing.
#   fetch            download the pinned weights into your Hugging Face cache and the bundle code into ./fetched/bundle as real files (needs `hf` and a network).
#   check            compare the bundle code and the weights with pins.json (opens no device and no network connection).
#   --self-test      check, then import the runtime files without opening a device.
# Environment: ACESTEP_PYTHON (python to use), ACESTEP_BUNDLE_CODE (default ./fetched/bundle/code) and ACESTEP_SNAPSHOT (default: the pinned revision in the Hugging Face cache),
# ACESTEP_WORK (a writable directory, default ./work), PORT, HOST, TT_METAL_HOME (only when `import ttnn` does not work yet).
set -euo pipefail
cd "$(dirname "$0")"
PY="${{ACESTEP_PYTHON:-python}}"
HUB="${{HF_HUB_CACHE:-${{HF_HOME:-$HOME/.cache/huggingface}}/hub}}"
MODEL_REV={REVISION}
BUNDLE_REV={BUNDLE_REVISION}
CODE="${{ACESTEP_BUNDLE_CODE:-$PWD/fetched/bundle/code}}"
SNAP="${{ACESTEP_SNAPSHOT:-$HUB/models--ACE-Step--Ace-Step1.5/snapshots/$MODEL_REV}}"
WORK="${{ACESTEP_WORK:-$PWD/work}}"
cmd="${{1:-serve}}"
case "$cmd" in
  fetch)
    hf download {MODEL_ID} --revision "$MODEL_REV" --include "acestep-v15-turbo/*" "vae/*" "Qwen3-Embedding-0.6B/*" "config.json"
    # The bundle code goes to a real directory. A link into the cache breaks ACE-Step's lookup of its model code, which uses its own resolved path.
    hf download {BUNDLE_REPO} --revision "$BUNDLE_REV" --include "code/*" --local-dir "$PWD/fetched/bundle"
    exit 0 ;;
  check)
    exec "$PY" tools/check_bundle.py --code "$CODE" --snapshot "$SNAP" ;;
  --self-test)
    "$PY" tools/check_bundle.py --code "$CODE" --snapshot "$SNAP"
    PYTHONPATH="runtime${{PYTHONPATH:+:$PYTHONPATH}}" "$PY" -c "import acestep_run, acestep_serve; print('runtime imports: pass')"
    exit 0 ;;
  serve) ;;
  *) echo "unknown command: $cmd" >&2; exit 2 ;;
esac
if ! "$PY" -c "import ttnn" 2>/dev/null; then
  : "${{TT_METAL_HOME:?ttnn is not importable: set TT_METAL_HOME to a tt-metal checkout with a built ttnn}}"
  export TT_METAL_RUNTIME_ROOT="${{TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}}"
  export PYTHONPATH="$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools${{PYTHONPATH:+:$PYTHONPATH}}"
  export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:$TT_METAL_HOME/build/tt_metal:$TT_METAL_HOME/build/ttnn:$TT_METAL_HOME/build/tt_stl:$TT_METAL_HOME/build/_deps/fmt-build:$TT_METAL_HOME/build/tt_metal/third_party/umd/device${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
fi
"$PY" tools/check_bundle.py --code "$CODE" --snapshot "$SNAP"
mkdir -p "$WORK"
exec "$PY" runtime/acestep_serve.py --bundle-code "$CODE" --snapshot "$SNAP" --root "$WORK" --host "${{HOST:-127.0.0.1}}" --port "${{PORT:-{PORT}}}"
"""


def manifest_text() -> str:
    return f"""# DRAFT. NOT BUILT. A v5.1 container manifest of kind `http-server` for runtime/acestep_serve.py. It cannot be built from this directory alone:
# `source.tt_metal` must be a tt-metal checkout, `source.extra_code` must name this directory, and the bundle code and weights are fetched by the user.
# `bash run.sh` runs the package without tt-model.
schema: "5.1"
repo: {NAMESPACE}/{PACKAGE_NAME}
name: {PACKAGE_NAME}
weights:
  repo: {MODEL_ID}
  revision: "{REVISION}"
kind: http-server
arch: blackhole
image:
  registry: hf
source:
  tt_metal: /path/to/tt-metal
  code:
    - models/common
  extra_code:
    - root: /path/to/this/package
      paths: [runtime, tools]
  ubuntu: "22.04"
  python: "3.12"
runtime:
  command: [python, runtime/acestep_serve.py, --host, "{{host}}", --port, "{{port}}"]
  ready_line: '"event": "ready"'
  packages: [numpy, torch, soundfile, transformers, tokenizers]
serve:
  port: {PORT}
serve_profiles:
  - name: p150
    description: One Blackhole chip (a p150, or one chip of a p300 or QuietBox 2).
    hardware: p150
    mesh_device: P150
default_profile: p150
"""


def publish_commands() -> str:
    """Text only. Nothing in this module runs a publish command."""
    return "\n".join([
        f"# 1. private upload:  hf upload --repo-type model --private {NAMESPACE}/{PACKAGE_NAME} <package dir> .",
        f"# 2. after a pull and a boot check of the pulled copy, a separate decision:  hf repos settings {NAMESPACE}/{PACKAGE_NAME} --no-private", ""])


# ---- staging ----------------------------------------------------------------------------------------------------------------
def _write(pkg: Path, rel: str, data, mode: int = 0o644) -> None:
    p = pkg / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data.encode() if isinstance(data, str) else data)
    p.chmod(mode)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_pins(bundle_code, snapshot) -> dict:
    """What the package was tested with, measured from the two directories given: the hash over the bundle code and the size and sha256 of each weight file."""
    sys.path.insert(0, str(ROOT / "pkgsrc" / "acestep" / "tools"))
    import check_bundle
    tree, n = check_bundle.tree_hash(bundle_code)
    snap = Path(snapshot)
    if snap.resolve().name != REVISION and snap.name != REVISION:
        raise SystemExit(f"the weights snapshot directory is named {snap.name}; the pinned revision is {REVISION}")
    weights = {rel: {"size": (snap / rel).stat().st_size, "sha256": _sha(snap / rel)} for rel in WEIGHT_FILES}
    return {"model": MODEL_ID, "model_revision": REVISION, "bundle_repo": BUNDLE_REPO, "bundle_revision": BUNDLE_REVISION, "bundle_tree_sha256": tree,
            "bundle_files": n, "weights": weights}


def stage(evidence_dir, bundle_code, snapshot, out) -> Path:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    ev = Path(evidence_dir)
    summary = _read(ev / "summary.json")
    license_text = (ROOT / "LICENSE").read_text()
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise SystemExit("the repository LICENSE is not the Apache License 2.0")
    for src, dst in SRC_FILES:
        _write(out, dst, (ROOT / src).read_bytes(), 0o755 if dst.startswith("tools/") else 0o644)
    _write(out, "runtime/orchard/__init__.py", RUNTIME_INIT)
    for name in EVIDENCE_FILES:
        _write(out, f"evidence/{name}", (ev / name).read_bytes())
    _write(out, "evidence/summary.json", json.dumps(summary, indent=1) + "\n")
    _write(out, "pins.json", json.dumps(make_pins(bundle_code, snapshot), indent=1) + "\n")
    _write(out, "LICENSE", license_text)
    _write(out, "NOTICE", NOTICE)
    _write(out, "README.md", render_card(summary))
    _write(out, "requirements.txt", REQUIREMENTS)
    _write(out, "run.sh", RUN_SH, 0o755)
    _write(out, "tt-model.yaml", manifest_text())
    _write(out, "SECURITY.md", SECURITY)
    _write(out, "CHANGELOG.md", "\n".join(["# Changelog", "", f"## {VERSION}", "", *[f"- {x}" for x in summary["changelog"]], ""]))
    files = {f.relative_to(out).as_posix(): _sha(f) for f in ap._tree(out)}
    _write(out, "package-record.json", json.dumps({"package": PACKAGE_NAME, "version": VERSION, "model": MODEL_ID, "revision": REVISION, "bundle_revision": BUNDLE_REVISION,
                                                    "tt_metal_tag": TT_METAL_TAG, "files": files}, indent=1) + "\n")
    _write(out, "SHA256SUMS", "".join(f"{_sha(f)}  {f.relative_to(out).as_posix()}\n" for f in ap._tree(out)))
    return out


# ---- the gate -----------------------------------------------------------------------------------------------------------------
def _front(card: str) -> dict:
    m = re.match(r"---\n(.*?)\n---\n", card, re.S)
    out = {}
    for line in (m.group(1).splitlines() if m else []):
        if ":" in line and not line.startswith("-"):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def package_problems(pkg, *, hostname: str | None = None, home: str | None = None) -> list[str]:
    pkg = Path(pkg)
    if not pkg.is_dir():
        return [f"{pkg} is not a directory"]
    odd = ap.unreadable_entries(pkg)
    if odd:
        return sorted(set(ap.weights_problems(pkg) + odd))
    problems = ap.weights_problems(pkg)
    problems += [f"scrub: {h}" for h in ap.scrub_files(pkg, hostname=hostname, home=home)]
    for rel in REQUIRED:
        if not (pkg / rel).is_file():
            problems.append(f"missing file: {rel}")
    for f in ap._tree(pkg):
        rel = f.relative_to(pkg).as_posix()
        text = ap._text(f) if f.suffix in (".py", ".sh", ".md", ".yaml", ".txt", ".json", "") or f.name in ("NOTICE", "LICENSE") else None
        if text is None:
            continue
        for bad in FORBIDDEN_TEXT:
            if bad.lower() in text.lower():
                problems.append(f"{rel}: contains {bad!r}")
        if f.suffix == ".py":
            try:
                compile(text, rel, "exec")
            except SyntaxError as exc:
                problems.append(f"{rel}: does not compile ({exc.msg}, line {exc.lineno})")
    if (pkg / "run.sh").is_file() and subprocess.run(["bash", "-n", str(pkg / "run.sh")], capture_output=True).returncode != 0:
        problems.append("run.sh: bash -n reports a syntax error")
    problems += _record_problems(pkg)
    problems += _release_problems(pkg)
    problems += _pins_problems(pkg)
    problems += _license_problems(pkg)
    problems += _card_problems(pkg)
    problems += _manifest_problems(pkg)
    return sorted(set(problems))


def _record_problems(pkg: Path) -> list[str]:
    try:
        rec = json.loads(ap._text(pkg / "package-record.json"))
        listed = rec["files"]
        if not isinstance(listed, dict):
            raise TypeError
    except (TypeError, ValueError, KeyError):
        return ["package-record.json is missing or has no file map"]
    out = []
    for rel in RECORDED + tuple(d for _, d in SRC_FILES):
        f = pkg / rel
        if rel not in listed:
            out.append(f"package-record.json does not list {rel}")
        elif f.is_file() and _sha(f) != listed[rel]:
            out.append(f"{rel} changed after it was staged")
    return out


def _release_problems(pkg: Path) -> list[str]:
    out = []
    sums = ap._text(pkg / "SHA256SUMS")
    if sums is not None:
        listed = {}
        for ln in sums.splitlines():
            h, _, p = ln.partition("  ")
            listed[p] = h
        actual = {f.relative_to(pkg).as_posix() for f in ap._tree(pkg)} - {"SHA256SUMS"}
        for p in sorted(actual - set(listed)):
            out.append(f"SHA256SUMS does not list {p}")
        for p in sorted(set(listed) - actual):
            out.append(f"SHA256SUMS lists {p}, which is not in the package")
        for p in sorted(actual & set(listed)):
            if _sha(pkg / p) != listed[p]:
                out.append(f"SHA256SUMS: {p} does not match its hash")
    try:
        rec = json.loads(ap._text(pkg / "package-record.json"))
        if rec.get("version") != VERSION:
            out.append(f"package-record.json: version must be {VERSION}")
        if rec.get("tt_metal_tag") != TT_METAL_TAG:
            out.append("package-record.json: tt_metal_tag is missing or wrong")
    except (TypeError, ValueError):
        pass
    if f"Version {VERSION}" not in (ap._text(pkg / "README.md") or ""):
        out.append(f"card: does not state Version {VERSION}")
    if f"## {VERSION}" not in (ap._text(pkg / "CHANGELOG.md") or ""):
        out.append(f"CHANGELOG.md: has no section for {VERSION}")
    if "Security note" not in (ap._text(pkg / "SECURITY.md") or ""):
        out.append("SECURITY.md: missing its security note")
    return out


def _pins_problems(pkg: Path) -> list[str]:
    try:
        pins = json.loads(ap._text(pkg / "pins.json"))
    except (TypeError, ValueError):
        return ["pins.json is missing or is not JSON"]
    out = []
    for key, want in (("model", MODEL_ID), ("model_revision", REVISION), ("bundle_repo", BUNDLE_REPO), ("bundle_revision", BUNDLE_REVISION)):
        if pins.get(key) != want:
            out.append(f"pins.json: {key} must be {want}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(pins.get("bundle_tree_sha256", ""))):
        out.append("pins.json: bundle_tree_sha256 is not a sha256")
    if not isinstance(pins.get("bundle_files"), int) or pins["bundle_files"] < 1:
        out.append("pins.json: bundle_files is not a positive number")
    w = pins.get("weights")
    if not isinstance(w, dict) or set(w) != set(WEIGHT_FILES):
        out.append(f"pins.json: weights must pin exactly {sorted(WEIGHT_FILES)}")
    else:
        for rel, v in w.items():
            if not (isinstance(v, dict) and isinstance(v.get("size"), int) and v["size"] > 0 and re.fullmatch(r"[0-9a-f]{64}", str(v.get("sha256", "")))):
                out.append(f"pins.json: weights[{rel}] needs a size and a sha256")
    run_sh = ap._text(pkg / "run.sh") or ""
    for rev in (REVISION, BUNDLE_REVISION):
        if rev not in run_sh:
            out.append(f"run.sh: does not name the pinned revision {rev[:12]}")
    return out


def _license_problems(pkg: Path) -> list[str]:
    text = ap._text(pkg / "LICENSE")
    if text is None:
        return []
    out = []
    if "Apache License" not in text or "Version 2.0, January 2004" not in text or "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" not in text:
        out.append("LICENSE is not the Apache License 2.0 text")
    notice = ap._text(pkg / "NOTICE") or ""
    if MODEL_ID not in notice or "MIT License" not in notice or BUNDLE_REPO not in notice:
        out.append("NOTICE does not name the model with its MIT License and the bundle")
    return out


def _card_problems(pkg: Path) -> list[str]:
    card = ap._text(pkg / "README.md")
    if card is None:
        return ["the package has no README.md card"]
    out = []
    fm = _front(card)
    for k, v in (("license", "apache-2.0"), ("base_model", MODEL_ID), ("pipeline_tag", "text-to-audio")):
        if fm.get(k) != v:
            out.append(f"card front matter: {k} must be {v}")
    for sec in CARD_SECTIONS:
        if sec not in card:
            out.append(f"card: missing section {sec!r}")
    for rev in (REVISION, BUNDLE_REVISION):
        if f"`{rev}`" not in card:
            out.append(f"card: does not name the pinned revision {rev[:12]}")
    for pat in OVERCLAIMS:
        m = re.search(pat, card, re.I)
        if m:
            out.append(f"card: overclaim {m.group(0)!r}")
    try:
        summary = json.loads(ap._text(pkg / "evidence" / "summary.json"))
        for row in speed_rows(summary) + load_rows(summary) + bar_rows(summary) + [agreement_line(summary), boot_line(summary)]:
            if row not in card:
                out.append(f"card: a number differs from evidence/summary.json: {row[:80]}")
        for key in ("limits", "not_tested"):
            for x in summary[key]:
                if x not in card:
                    out.append(f"card: {key} line missing: {x[:50]}")
        if not summary["boot"].get("ok"):
            out.append("evidence/summary.json: the boot check did not pass")
        for i, p in enumerate(summary["step_bars"]):
            if not p.get("usable"):
                out.append(f"evidence/summary.json: step bars of prompt {i + 1} were refused by orchard.step_bars")
    except (TypeError, ValueError, KeyError) as exc:
        out.append(f"evidence/summary.json cannot be used to check the card ({type(exc).__name__})")
    return out


def _manifest_problems(pkg: Path) -> list[str]:
    import yaml
    text = ap._text(pkg / "tt-model.yaml")
    if text is None:
        return ["tt-model.yaml is missing"]
    try:
        m = yaml.safe_load(text)
    except yaml.YAMLError:
        return ["tt-model.yaml is not valid YAML"]
    if not isinstance(m, dict):
        return [f"tt-model.yaml: the document root must be a mapping, not {type(m).__name__}"]
    out = []
    expect = {"schema": "5.1", "kind": "http-server", "arch": "blackhole", "name": PACKAGE_NAME, "repo": f"{NAMESPACE}/{PACKAGE_NAME}"}
    for k, v in expect.items():
        if m.get(k) != v:
            out.append(f"tt-model.yaml: {k} must be {v!r}")
    w = m.get("weights") or {}
    if w.get("repo") != MODEL_ID or w.get("revision") != REVISION:
        out.append("tt-model.yaml: weights must pin the model id and the revision")
    rt = m.get("runtime") or {}
    cmd = rt.get("command") or []
    if "{port}" not in " ".join(map(str, cmd)):
        out.append("tt-model.yaml: runtime.command needs a {port} placeholder")
    for sc in [c for c in cmd if str(c).endswith(".py")]:
        if not (pkg / sc).is_file():
            out.append(f"tt-model.yaml: runtime.command names {sc}, which is not in the package")
    marker = str(rt.get("ready_line") or "")
    if not marker:
        out.append("tt-model.yaml: runtime.ready_line is missing")
    elif marker.strip("'") not in (ap._text(pkg / "runtime" / "acestep_serve.py") or ""):
        out.append("tt-model.yaml: the ready line marker is not in runtime/acestep_serve.py, so the server never prints it")
    for extra in (m.get("source") or {}).get("extra_code", []):
        for p in extra.get("paths", []):
            if not (pkg / p).is_dir():
                out.append(f"tt-model.yaml: extra_code path {p} is not in the package")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stage")
    st.add_argument("--evidence", required=True, help="a directory with summary.json and the raw files of EVIDENCE_FILES")
    st.add_argument("--bundle-code", required=True)
    st.add_argument("--snapshot", required=True)
    st.add_argument("--out", required=True)
    sm = sub.add_parser("summary")
    sm.add_argument("--sweep", required=True)
    sm.add_argument("--bars", action="append", required=True)
    sm.add_argument("--agreement", required=True)
    sm.add_argument("--boot", required=True)
    sm.add_argument("--out", required=True)
    g = sub.add_parser("gate")
    g.add_argument("pkg")
    a = p.parse_args(argv)
    if a.cmd == "summary":
        s = make_summary(_read(a.sweep), [_read(b) for b in a.bars], _read(a.agreement), _read(a.boot))
        Path(a.out).write_text(json.dumps(s, indent=1) + "\n")
        print(f"wrote {a.out}")
        return 0
    if a.cmd == "stage":
        out = stage(a.evidence, a.bundle_code, a.snapshot, a.out)
        print(f"staged {out}")
        print(publish_commands())
        return 0
    problems = package_problems(a.pkg)
    for x in problems:
        print("PROBLEM", x)
    print("gate:", "FAIL" if problems else "pass", f"({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
