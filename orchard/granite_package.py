# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage and check the Granite-4.0-H-1B package (a text model served by an OpenAI-compatible http-server on one Blackhole chip).

    python -m orchard.granite_package stage --summary SUMMARY.json --evidence DIR [--evidence-extra FILE ...] --out PKG_DIR
    python -m orchard.granite_package gate PKG_DIR

`stage` copies the runtime, the server and the tools from `pkgsrc/`, the three coder kernels, the evidence files, and writes the card
(`README.md`), `LICENSE`, `NOTICE`, `tt-model.yaml`, `run.sh`, `requirements.txt` and `package-record.json`. No weights are copied.
The card's numbers come from `evidence/summary.json`, which `stage` copies into the package.

`gate` reads the directory as it is on disk and returns every problem found (an empty list is a pass):
  * no weights, no cache or venv directory, no link, FIFO or oversized file (the audio package's checks, reused);
  * no home path, host name, token, `.tt_ai_key` or `litellm` in any text file;
  * the required files exist, every Python file compiles and `run.sh` passes `bash -n`;
  * `LICENSE` is the Apache License 2.0 and has the sha256 recorded at staging, and the card's front matter names the license, the base
    model and the pipeline;
  * every number row of the card is the row that `summary.json` produces, and the card has its required sections;
  * `tt-model.yaml` is a v5.1 manifest of kind `http-server` whose command, ready line and source paths exist in the package.
It does not judge whether prose is true and does not run the server. Nothing here publishes: `publish_commands` returns text.
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
from orchard import safeio

MODEL_ID = "ibm-granite/granite-4.0-h-1b"
REVISION = "d18cca4c121edb87d022116d281ce212c9136f57"
NAMESPACE = "episod"
PACKAGE_NAME = "granite-4.0-h-1b-blackhole"
PORT = 20110
VERSION = "0.3.0"
TT_METAL_TAG = "v0.80.0"
ROOT = Path(__file__).resolve().parent.parent
SRC_FILES = (("pkgsrc/runtime/granite_chip.py", "runtime/granite_chip.py"), ("pkgsrc/runtime/granite_trace.py", "runtime/granite_trace.py"),
             ("pkgsrc/server/serve.py", "server/serve.py"),
             ("pkgsrc/tools/evaluate.py", "tools/evaluate.py"), ("pkgsrc/tools/smoke.py", "tools/smoke.py"), ("pkgsrc/tools/selftest.py", "tools/selftest.py"))
KERNEL_FILES = (("docs/run-logs/evidence/granite-readout-pair/kernels/state.py", "runtime/kernels/state.py"),
                ("docs/run-logs/evidence/granite-readout-pair/kernels/readout.py", "runtime/kernels/readout.py"),
                ("docs/run-logs/evidence/granite-repeat-trace/port2/bcast_kernel_attempt8.py", "runtime/kernels/bcast.py"))
REQUIRED = ("README.md", "LICENSE", "NOTICE", "SECURITY.md", "CHANGELOG.md", "SHA256SUMS", "tt-model.yaml", "run.sh", "requirements.txt", "package-record.json", "evidence/summary.json",
            "runtime/granite_chip.py", "runtime/granite_trace.py", "server/serve.py", "tools/evaluate.py", "tools/smoke.py", "tools/selftest.py") + tuple(d for _, d in KERNEL_FILES)
RECORDED = ("LICENSE", "NOTICE", "evidence/summary.json")
FORBIDDEN_TEXT = (".tt_ai_key", "litellm", "HF_TOKEN=", "Authorization: Bearer", "TT_AI_KEY")
CARD_SECTIONS = ("## Weights are not included", "## Try it", "## API", "## Numbers", "## Agreement with the CPU reference", "## Context and prefill",
                 "## Limits", "## Not tested", "## Optional kernel modes", "## Security", "## License", "## Provenance")
OVERCLAIMS = (r"\bproduction[- ]ready\b", r"\bstate[- ]of[- ]the[- ]art\b", r"\b(?:official|officially|endorsed|certified|verified)\b",
              r"\bfaster than (?:a )?gpu\b", r"\bbest\b")


# ---- the numbers on the card ----------------------------------------------------------------------------------------------
def _f(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def headline(s: dict) -> dict:
    """The session whose numbers go in the machine-readable rows: the session named in the summary's `headline`, else the session recorded
    with the lowest host load. A summary that names a session the list does not hold is an error."""
    if s.get("headline"):
        return next(x for x in s["sessions"] if x["name"] == s["headline"])
    return min(s["sessions"], key=lambda x: x["load"])


def number_rows(s: dict) -> list[str]:
    """Markdown table rows [number | value | label | evidence], derived from summary.json. The gate compares the card with these.
    The label `measured` is what card readers (the tt-model ranker) accept; the other labels say how many runs are behind a number."""
    rows = []
    h = headline(s)
    n = s["sessions"][0]["prompt_tokens"]
    ev = "`evidence/summary.json`"
    rows.append((f"Decode speed, `chat` prompt ({n} tokens, traced), tok/s", f"{_f(h['decode']['mean'])} tok/s", "measured", f"{ev}, {h['evidence']}"))
    hu = h if h["decode_untraced"] else next(x for x in sorted(s["sessions"], key=lambda x: x["load"]) if x["decode_untraced"])
    rows.append((f"Decode speed, `chat-untraced` prompt ({n} tokens, untraced), tok/s", f"{_f(hu['decode_untraced']['mean'])} tok/s", "measured", f"{ev}, {hu['evidence']}"))
    rows.append((f"Time to first token, `chat` prompt ({n} tokens)", f"{_f(h['ttft_s'], 2)} s", "measured", f"{ev}, {h['evidence']}"))
    ag = s["agreement_short"]["chip"]["top1"]
    rows.append(("top1 agreement with the fp32 CPU reference, mean of four prompts", f"{sum(ag) / len(ag):.3f} fraction", "measured", "`evidence/agree-prose-2048.json`"))
    for sess in s["sessions"]:
        lab = f"{sess['name']}: {sess['when']}"
        u = sess["decode_untraced"]
        rows += [(f"Decode speed, traced, {sess['name']} (tok/s)", f"{_f(sess['decode']['mean'])} ({_f(sess['decode']['min'])} to {_f(sess['decode']['max'])})", lab, sess["evidence"]),
                 (f"Decode speed, untraced, {sess['name']} (tok/s)", f"{_f(u['mean'])} ({_f(u['min'])} to {_f(u['max'])})" if u else "not measured", lab, sess["evidence"]),
                 (f"Share of the one-die bf16 decode floor (175.2 tok/s), traced, {sess['name']}", f"{sess['decode']['mean'] / s['floor_tok_s'] * 100:.1f}%", lab, sess["evidence"]),
                 (f"First token after a {n}-token prompt, {sess['name']} (s)", _f(sess["ttft_s"], 2), lab, sess["evidence"])]
        for npr, v in sess["prefill"]:
            rows.append((f"Prefill, {npr}-token prompt, {sess['name']} (tok/s)", _f(v), lab, sess["evidence"]))
        if sess.get("server"):
            sv = sess["server"]
            rows.append((f"Through the server: first streamed content, {sess['name']} (s)", _f(sv["first_token_s"], 2), lab, sess["server_evidence"]))
            rows.append((f"Through the server: decode of a 64-token completion, {sess['name']} (tok/s)", _f(sv["decode_tok_s"]), lab, sess["server_evidence"]))
            rows.append((f"Through the server: {sv['long_prompt_tokens']:,}-token prompt to first token, {sess['name']} (s)", _f(sv["long_prompt_first_token_s"]), lab, sess["server_evidence"]))
    rows.append(("Context limit (prompt plus generated tokens)", f"{s['context']}", "setting", ev))
    for c in s["context_sweep"]:
        rows.append((f"Decode speed with the limit set to {c['max_len']} (tok/s)", _f(c["mean"]), "5 repeats after 1 warm-up, session 1", "`evidence/bench-L%d.json`" % c["max_len"]))
    return [f"| {a} | {b_} | {c_} | {e} |" for a, b_, c_, e in rows]


def agreement_rows(s: dict) -> list[str]:
    a = s["agreement_short"]
    out = []
    for name, key in (("bf16 CPU run (the floor)", "floor"), ("this package", "chip")):
        r = a[key]
        out.append(f"| {name} | {', '.join(f'{x:.4f}' for x in r['rel_l2'])} | {', '.join(f'{x:.3f}' for x in r['top1'])} | {', '.join(str(x) for x in r['prefix'])} |")
    return out


def long_rows(s: dict) -> list[str]:
    out = []
    for t in s["agreement_long"]:
        for label, key in (("this package", "chip"), ("bf16 CPU run", "floor")):
            r = t[key]
            out.append(f"| {t['kind']}, {t['tokens']:,} tokens | {label} | {r['rel_l2_mean']:.4f} | {r['rel_l2_max']:.3f} | {r['top1']:.3f} |")
        if t.get("first_1024"):
            for label, key in (("this package", "chip_vs_fp32"), ("bf16 CPU run", "bf16_cpu_vs_fp32")):
                r = t["first_1024"][key]
                out.append(f"| {t['kind']}, first 1,024 tokens | {label} | {r['rel_l2_mean']:.4f} | {r['rel_l2_max']:.3f} | {r['top1']:.3f} |")
    return out


# ---- the card and the other files ---------------------------------------------------------------------------------------------
def render_changelog(s: dict) -> str:
    items = s.get("changelog", [])
    older = s.get("changelog_older", [])                 # [{"version": "0.2.0", "items": [...]}], newest first
    prev = older[0]["version"] if older else "0.1.0"
    parts = []
    for o in older:
        parts += [f"## {o['version']}", "", *[f"- {x}" for x in o["items"]], ""]
    return "\n".join(["# Changelog", "", f"## {VERSION}", "", f"Changes from {prev}:", "", *[f"- {x}" for x in items], "", *parts,
                      "## 0.1.0", "", "First upload: the traced composed Granite-4.0-H-1B runtime with a float32 Mamba-2 state, a 2,048-token default context, an OpenAI-compatible chat server, "
                      "the card and the gate.", ""])


def render_card(s: dict, license_text: str) -> str:
    kern = s["kernel_modes"]
    lines = ["---", "license: apache-2.0", f"base_model: {MODEL_ID}", "pipeline_tag: text-generation", "tags:", "- blackhole", "- tenstorrent", "- mamba2",
             "- granite", "---", "", f"# {PACKAGE_NAME}", "", f"Version {VERSION}. See `CHANGELOG.md`.", "",
             f"> Runs [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) (revision `{REVISION}`) by IBM on one Tenstorrent Blackhole chip. "
             "The model weights are not in this repository. This is a community package and not a release of IBM or Tenstorrent.", "",
             "A ttnn port of Granite-4.0-H-1B (a 1.5B-parameter hybrid of 36 Mamba-2 layers and 4 attention layers, 40 layers in all), with an OpenAI-compatible "
             "chat server. Weights are bf16 on one chip. The decode step is captured as a ttnn trace and replayed for each token. "
             "Prompts are processed one token per step (a loop of decode steps). Sampling on the host is available, and greedy decoding is what was measured. "
             "The Mamba-2 state, the step size and the decay are kept in float32 on the chip by default (`--ssd-precision bf16` keeps them in bf16).", "",
             "## Weights are not included", "",
             f"Download them yourself: [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) is Apache-2.0 and is not gated. `run.sh` fetches the pinned revision "
             f"`{REVISION}` into your Hugging Face cache on the first start (about 3 GB).", "",
             "## Try it", "",
             f"Needs a Tenstorrent Blackhole chip, a tt-metal build with `ttnn` (tested with tt-metal {TT_METAL_TAG}), and Python 3.12.", "",
             "```bash", "python -m venv venv && . venv/bin/activate", "pip install -r requirements.txt",
             "export TT_METAL_HOME=<your tt-metal checkout>     # run.sh adds its ttnn to PYTHONPATH when `import ttnn` does not work yet",
             "bash run.sh --self-test                           # one line per check; opens no device (add --open-device to test that too)",
             "bash run.sh                                       # on a shared machine, take a chip lease first", "```", "",
             f"The server listens on http://127.0.0.1:{PORT} (set `PORT`). It binds the local machine only and has no sign-in; see Security. "
             "It holds the chip while it runs, and one request runs at a time.", "",
             "## API", "",
             "`GET /v1/models`, `GET /health` and `POST /v1/chat/completions` (`messages`, `max_tokens` or `max_completion_tokens`, `stream`, `stop`, `temperature`, `top_p`, `seed`).", "",
             "- The prompt is the model's own chat template, applied to `messages` with the generation prompt added.",
             "- Generation ends at the end-of-text token, at a `stop` string, at `max_tokens` (default 256, at most 4096), or when prompt plus output reach the context limit.",
             "- `finish_reason` is `stop` when the loop ended on the end-of-text token or a stop string, and `length` when it ended on `max_tokens` or the context limit.",
             "- Without `temperature`, or with 0, decoding is greedy. A temperature above 0 samples on the host; the same `seed` repeats the output.",
             "- A response carries a `timings` object (prefill seconds, first-token seconds, decode tokens per second) next to `usage`.",
             "- Streamed responses are written as the tokens are produced. A multi-byte character is sent whole.",
             "- A second request while one is running gets `503` with `Retry-After: 5` at once. A client that disconnects mid-stream frees the slot at the next token.",
             "- A request body over 1 MiB, malformed JSON, bad fields, a prompt longer than the context, tool calls and document inputs get a `400`. An idle connection is closed after 30 seconds.", "",
             "```bash", f"curl -s http://127.0.0.1:{PORT}/v1/chat/completions -H 'Content-Type: application/json' \\",
             "  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":32}'", "```", "",
             "## Numbers", "",
             f"Measured on one chip of a Blackhole p300c board with the pinned weights, in {len(s['sessions'])} sessions on a host shared with other jobs. "
             f"{s['numbers_note']} A label states how many runs are behind a number. The rows labelled `measured` are the headline numbers (session with the lowest host load).", "",
             "| Number | Value | Label | Evidence |", "| --- | --- | --- | --- |", *number_rows(s), "",
             f"{s['sessions_note']}", "",
             "## Agreement with the CPU reference", "",
             "Teacher forced over four short prompts (5, 16, 7 and 12 prompt tokens; 24 reference tokens each, fp32 transformers on the CPU is the reference). "
             "The bf16 CPU row is the same model run in bf16 on the CPU, which is the floor any bf16 implementation is read against. "
             "Relative L2 of the logits, mean over positions, then top-1 agreement, then the number of greedy tokens before the first difference (prompt lengths 3, 24, 24, 24 tokens). "
             "Columns list prompts 0 to 3.", "",
             "| Run | Relative L2, mean | Top-1 | First difference at token |", "| --- | --- | --- | --- |", *agreement_rows(s), "",
             f"{s['agreement_note']}", "",
             "Teacher forced against fp32 on the CPU over longer sequences of three kinds of text (one sequence of each kind):", "",
             "| Text | Run | Relative L2, mean | Relative L2, max | Top-1 |", "| --- | --- | --- | --- | --- |", *long_rows(s), "",
             f"{s['long_note']}", "",
             "## Context and prefill", "",
             f"The context limit is {s['context']} tokens (prompt plus generated). The key/value cache has that fixed length, and every decode step reads all of it, "
             "so the limit sets the step time: the table above lists decode speed at several limits. The model itself supports 131,072 tokens; "
             "this package does not.", "",
             f"{s['prefill_note']}", "",
             "## Limits", "",
             *[f"- {x}" for x in s["limits"]], "",
             "## Not tested", "",
             *[f"- {x}" for x in s["not_tested"]], "",
             "## Optional kernel modes", "",
             "The kernel modes run with a bf16 state whatever `--ssd-precision` says.", kern["text"], "",
             "## Security", "",
             "The server has no sign-in, no TLS and no rate limit, and binds 127.0.0.1 by default. `HOST=0.0.0.0` listens on every interface: use it only on a network you trust. "
             "`SECURITY.md` has the rest.", "",
             "## License", "",
             "The model is licensed under the Apache License 2.0 by IBM (the model repository's metadata says `license: apache-2.0`; it ships no separate license file). "
             "The code in this package is Tenstorrent's, also under the Apache License 2.0. `LICENSE` is the Apache License 2.0 text and `NOTICE` names the parties.", "",
             "## Provenance", "",
             f"- Model: {MODEL_ID}, revision `{REVISION}`; weights are not part of this package.",
             f"- Tested with tt-metal {TT_METAL_TAG}, Python 3.12 and the package versions in `requirements.txt`. `SHA256SUMS` lists the hash of every file.",
             "- `runtime/granite_chip.py`, `runtime/granite_trace.py`, `server/serve.py`, `tools/evaluate.py`, `tools/smoke.py` and `tools/selftest.py` come from the tt-orchard repository (tsingletaryTT/tt-orchard).",
             "- `runtime/kernels/` holds three tt-lang kernels written by a local coder model and used unchanged. Their hashes are in `package-record.json`.",
             "- `tt-model.yaml` is a draft v5.1 manifest of kind `http-server`. It has not been built into an image, and this package has not been listed in the tt-model catalog.",
             "- Every number on this card comes from `evidence/summary.json`, which is built from the raw result files beside it.", ""]
    return "\n".join(lines)


NOTICE = f"""granite-4.0-h-1b-blackhole

Model: {MODEL_ID} (revision {REVISION}), Copyright IBM Corp., licensed under the Apache License, Version 2.0.
The model weights are not part of this package.

Runtime, server and tools in this package: Copyright 2026 Tenstorrent USA, Inc., licensed under the Apache License, Version 2.0 (see LICENSE).
The three kernels in runtime/kernels/ are covered by the SPDX header in each file (Apache-2.0).

Licensed under the Apache License, Version 2.0; you may not use this package except in compliance with the License. Unless required by applicable
law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""

SECURITY = """# Security note

This package is for use on your own machine.

- The server binds 127.0.0.1 by default. It has no sign-in, no TLS and no rate limit. Anyone who can reach its port can use the chip.
- `HOST=0.0.0.0` (or `--host 0.0.0.0`) listens on every network interface. Do that only on a network you trust, or put an authenticating proxy in front.
- One request runs at a time. A second request gets `503` with `Retry-After`. A request body over 1 MiB is refused.
- The server runs the model's chat template in a sandboxed Jinja environment and loads no code from the weights directory.
- The package holds no weights and no credentials. `run.sh` downloads the pinned revision of the weights from the Hugging Face Hub with your own environment.
- Report a problem in the tt-orchard repository (tsingletaryTT/tt-orchard) issue tracker.
"""

REQUIREMENTS = """# Python packages for the server and the runtime. Each range starts at the version this package was tested with and stops before the next
# minor release. ttnn itself comes from your tt-metal build (see README.md); the lines under "ttnn's own dependencies" are the ones
# tt-metal's pyproject.toml lists, which `import ttnn` needs in this environment.
numpy>=2.5.4,<2.6
torch>=2.15.0,<2.16
safetensors>=0.8.0,<0.9
tokenizers>=0.23.3,<0.24
jinja2>=3.1.6,<3.2
huggingface_hub>=2.4.0,<2.5
# ttnn's own dependencies
loguru>=0.7.3,<0.8
networkx>=3.7,<3.8
graphviz>=0.21,<0.22
pyyaml>=6.0.3,<7
click>=8.5.0,<8.6
pandas>=3.1.0,<3.2
seaborn>=0.13.2,<0.14
ml_dtypes>=0.6.0,<0.7
pydantic>=2.14.2,<2.15
"""

RUN_SH = f"""#!/usr/bin/env bash
# Starts the OpenAI-compatible chat server for {MODEL_ID} on one Blackhole chip.
# Weights are not in this package: the first start downloads revision {REVISION} into your Hugging Face cache.
# Set GRANITE_MODEL_DIR to use a directory you already have (config.json, model.safetensors, tokenizer.json, chat_template.jinja).
# On a shared machine, take a chip lease first. This script does not reset or touch other chips.
set -euo pipefail
cd "$(dirname "$0")"
if ! python -c "import ttnn" 2>/dev/null; then
  : "${{TT_METAL_HOME:?ttnn is not importable: set TT_METAL_HOME to a tt-metal checkout with a built ttnn}}"
  export TT_METAL_RUNTIME_ROOT="${{TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}}"
  export PYTHONPATH="$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools${{PYTHONPATH:+:$PYTHONPATH}}"
  export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:$TT_METAL_HOME/build/tt_metal:$TT_METAL_HOME/build/ttnn:$TT_METAL_HOME/build/tt_stl:$TT_METAL_HOME/build/_deps/fmt-build:$TT_METAL_HOME/build/tt_metal/third_party/umd/device${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
fi
if [ "${{1:-}}" = "--self-test" ]; then shift; exec python tools/selftest.py "$@"; fi
exec python server/serve.py --port "${{PORT:-{PORT}}}" "$@"
"""


def manifest_text() -> str:
    return f"""# DRAFT. NOT BUILT. A v5.1 container manifest of kind `http-server` for server/serve.py. It cannot be built from this directory alone:
# `source.tt_metal` must be a tt-metal checkout and `source.extra_code` must name this directory. `bash run.sh` runs the package without tt-model.
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
      paths: [runtime, server]
  ubuntu: "22.04"
  python: "3.12"
runtime:
  command: [python, server/serve.py, --host, "{{host}}", --port, "{{port}}"]
  ready_line: '"event": "ready"'
  packages: [numpy, torch, safetensors, tokenizers, jinja2, huggingface_hub]
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
def _write(pkg: Path, rel: str, data: bytes | str, mode: int = 0o644) -> None:
    p = pkg / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data.encode() if isinstance(data, str) else data)
    p.chmod(mode)


def stage(summary_path, evidence_dir, out, *, extra_evidence=()) -> Path:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    summary = json.loads(Path(summary_path).read_text())
    license_text = (ROOT / "LICENSE").read_text()
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise SystemExit("the repository LICENSE is not the Apache License 2.0")
    for src, dst in SRC_FILES + KERNEL_FILES:
        _write(out, dst, (ROOT / src).read_bytes())
    for f in sorted(Path(evidence_dir).glob("*.json")) + [Path(x) for x in extra_evidence]:
        _write(out, f"evidence/{f.name}", f.read_bytes())
    _write(out, "evidence/summary.json", json.dumps(summary, indent=1) + "\n")
    _write(out, "LICENSE", license_text)
    _write(out, "NOTICE", NOTICE)
    _write(out, "README.md", render_card(summary, license_text))
    _write(out, "requirements.txt", REQUIREMENTS)
    _write(out, "run.sh", RUN_SH, 0o755)
    _write(out, "tt-model.yaml", manifest_text())
    _write(out, "SECURITY.md", SECURITY)
    _write(out, "CHANGELOG.md", render_changelog(summary))
    files = [{"path": f.relative_to(out).as_posix(), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f in ap._tree(out)]
    _write(out, "package-record.json", json.dumps({"package": PACKAGE_NAME, "version": VERSION, "model": MODEL_ID, "revision": REVISION,
                                                    "tt_metal_tag": TT_METAL_TAG, "files": files}, indent=1) + "\n")
    sums = "".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out).as_posix()}\n" for f in ap._tree(out))
    _write(out, "SHA256SUMS", sums)
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
    problems += _license_problems(pkg)
    problems += _card_problems(pkg)
    problems += _manifest_problems(pkg)
    return sorted(set(problems))


def _record_problems(pkg: Path) -> list[str]:
    text = ap._text(pkg / "package-record.json")
    try:
        rec = json.loads(text)
        listed = {r["path"]: r["sha256"] for r in rec["files"]}
    except (TypeError, ValueError, KeyError):
        return ["package-record.json is missing or has no file list"]
    out = []
    for rel in RECORDED:
        data = ap._text(pkg / rel)
        if rel not in listed:
            out.append(f"package-record.json does not list {rel}")
        elif data is not None and hashlib.sha256(data.encode()).hexdigest() != listed[rel]:
            out.append(f"{rel} changed after it was staged")
    for _, dst in KERNEL_FILES:
        f = pkg / dst
        if f.is_file() and listed.get(dst) != hashlib.sha256(f.read_bytes()).hexdigest():
            out.append(f"{dst} changed after it was staged")
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
            if hashlib.sha256((pkg / p).read_bytes()).hexdigest() != listed[p]:
                out.append(f"SHA256SUMS: {p} does not match its hash")
    try:
        rec = json.loads(ap._text(pkg / "package-record.json"))
        if rec.get("version") != VERSION:
            out.append(f"package-record.json: version must be {VERSION}")
        if rec.get("tt_metal_tag") != TT_METAL_TAG:
            out.append("package-record.json: tt_metal_tag is missing or wrong")
    except (TypeError, ValueError):
        pass
    card = ap._text(pkg / "README.md") or ""
    if f"Version {VERSION}" not in card:
        out.append(f"card: does not state Version {VERSION}")
    if f"## {VERSION}" not in (ap._text(pkg / "CHANGELOG.md") or ""):
        out.append(f"CHANGELOG.md: has no section for {VERSION}")
    if "Security note" not in (ap._text(pkg / "SECURITY.md") or ""):
        out.append("SECURITY.md: missing its security note")
    return out


def _license_problems(pkg: Path) -> list[str]:
    text = ap._text(pkg / "LICENSE")
    if text is None:
        return []
    out = []
    if "Apache License" not in text or "Version 2.0, January 2004" not in text or "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" not in text:
        out.append("LICENSE is not the Apache License 2.0 text")
    notice = ap._text(pkg / "NOTICE") or ""
    if MODEL_ID not in notice or "Apache License, Version 2.0" not in notice:
        out.append("NOTICE does not name the model and its license")
    return out


def _card_problems(pkg: Path) -> list[str]:
    card = ap._text(pkg / "README.md")
    if card is None:
        return ["the package has no README.md card"]
    out = []
    fm = _front(card)
    if fm.get("license") != "apache-2.0":
        out.append("card front matter: license must be apache-2.0")
    if fm.get("base_model") != MODEL_ID:
        out.append(f"card front matter: base_model must be {MODEL_ID}")
    if fm.get("pipeline_tag") != "text-generation":
        out.append("card front matter: pipeline_tag must be text-generation")
    for sec in CARD_SECTIONS:
        if sec not in card:
            out.append(f"card: missing section {sec!r}")
    if f"`{REVISION}`" not in card:
        out.append("card: does not name the pinned revision")
    for pat in OVERCLAIMS:
        m = re.search(pat, card, re.I)
        if m:
            out.append(f"card: overclaim {m.group(0)!r}")
    try:
        summary = json.loads(ap._text(pkg / "evidence" / "summary.json"))
        for row in number_rows(summary) + agreement_rows(summary) + long_rows(summary):
            if row not in card:
                out.append(f"card: a number row differs from evidence/summary.json: {row[:80]}")
        for key in ("limits", "not_tested"):
            for x in summary[key]:
                if x not in card:
                    out.append(f"card: {key} line missing: {x[:50]}")
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
    scripts = [c for c in cmd if str(c).endswith(".py")]
    for sc in scripts:
        if not (pkg / sc).is_file():
            out.append(f"tt-model.yaml: runtime.command names {sc}, which is not in the package")
    if not rt.get("ready_line"):
        out.append("tt-model.yaml: runtime.ready_line is missing")
    serve = (ap._text(pkg / "server" / "serve.py") or "")
    marker = str(rt.get("ready_line") or "")
    if marker and marker not in serve:             # the server's ready event is built from a dict literal with this exact text, json.dumps prints it unchanged
        out.append("tt-model.yaml: the ready line marker is not in server/serve.py, so the server never prints it")
    for extra in (m.get("source") or {}).get("extra_code", []):
        for p in extra.get("paths", []):
            if not (pkg / p).is_dir():
                out.append(f"tt-model.yaml: extra_code path {p} is not in the package")
    return out


def main(argv=None) -> int:
    ap_ = argparse.ArgumentParser()
    sub = ap_.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stage")
    st.add_argument("--summary", required=True)
    st.add_argument("--evidence", required=True)
    st.add_argument("--evidence-extra", action="append", default=[])
    st.add_argument("--out", required=True)
    g = sub.add_parser("gate")
    g.add_argument("pkg")
    a = ap_.parse_args(argv)
    if a.cmd == "stage":
        out = stage(a.summary, a.evidence, a.out, extra_evidence=a.evidence_extra)
        print(f"staged {out}")
        print(publish_commands())
        return 0
    problems = package_problems(a.pkg)
    for p in problems:
        print("PROBLEM", p)
    print("gate:", "FAIL" if problems else "pass", f"({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
