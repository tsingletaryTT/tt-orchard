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
ROOT = Path(__file__).resolve().parent.parent
SRC_FILES = (("pkgsrc/runtime/granite_chip.py", "runtime/granite_chip.py"), ("pkgsrc/server/serve.py", "server/serve.py"),
             ("pkgsrc/tools/evaluate.py", "tools/evaluate.py"), ("pkgsrc/tools/smoke.py", "tools/smoke.py"))
KERNEL_FILES = (("docs/run-logs/evidence/granite-readout-pair/kernels/state.py", "runtime/kernels/state.py"),
                ("docs/run-logs/evidence/granite-readout-pair/kernels/readout.py", "runtime/kernels/readout.py"),
                ("docs/run-logs/evidence/granite-repeat-trace/port2/bcast_kernel_attempt8.py", "runtime/kernels/bcast.py"))
REQUIRED = ("README.md", "LICENSE", "NOTICE", "tt-model.yaml", "run.sh", "requirements.txt", "package-record.json", "evidence/summary.json",
            "runtime/granite_chip.py", "server/serve.py", "tools/evaluate.py", "tools/smoke.py") + tuple(d for _, d in KERNEL_FILES)
RECORDED = ("LICENSE", "NOTICE", "evidence/summary.json")
FORBIDDEN_TEXT = (".tt_ai_key", "litellm", "HF_TOKEN=", "Authorization: Bearer", "TT_AI_KEY")
CARD_SECTIONS = ("## Weights are not included", "## Try it", "## API", "## Numbers", "## Agreement with the CPU reference", "## Context and prefill",
                 "## Limitations", "## Not tested", "## Optional kernel modes", "## License", "## Provenance")
OVERCLAIMS = (r"\bproduction[- ]ready\b", r"\bstate[- ]of[- ]the[- ]art\b", r"\b(?:official|officially|endorsed|certified|verified)\b",
              r"\bfaster than (?:a )?gpu\b", r"\bbest\b")


# ---- the numbers on the card ----------------------------------------------------------------------------------------------
def _f(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def number_rows(s: dict) -> list[str]:
    """Markdown table rows [number | value | label | evidence], derived from summary.json. The gate compares the card with these."""
    d, u = s["decode"], s["decode_untraced"]
    rows = [("Decode speed, traced, batch 1 (tok/s)", f"{_f(d['mean'])} ({_f(d['min'])} to {_f(d['max'])})", s["decode_label"], "evidence/summary.json"),
            ("Decode speed, untraced, batch 1 (tok/s)", f"{_f(u['mean'])} ({_f(u['min'])} to {_f(u['max'])})", s["decode_untraced_label"], "evidence/summary.json"),
            ("Share of the one-die bf16 decode floor (175.2 tok/s), traced", f"{d['mean'] / s['floor_tok_s'] * 100:.1f}%", s["decode_label"], "evidence/summary.json"),
            ("Context limit (prompt plus generated tokens)", f"{s['context']}", "setting", "evidence/summary.json"),
            ("First token after a 12-token prompt (s, median of 5)", _f(s["first_token_s_12"], 2), "5 repeats", "evidence/summary.json")]
    for p in s["prefill"]:
        rows.append((f"Prefill, {p['n']}-token prompt (tok/s)", _f(p["tok_s"]), p["label"], "evidence/summary.json"))
    for c in s["context_sweep"]:
        rows.append((f"Decode speed with the limit set to {c['max_len']} (tok/s)", _f(c["mean"]), c["label"], "evidence/summary.json"))
    b = s["boot"]
    rows.append(("Through the server: first token for a short chat (s)", _f(b["first_token_s"], 2), b["label"], "evidence/summary.json"))
    rows.append(("Through the server: decode speed (tok/s)", _f(b["decode_tok_s"]), b["label"], "evidence/summary.json"))
    return [f"| {a} | {b_} | {c_} | {e} |" for a, b_, c_, e in rows]


def agreement_rows(s: dict) -> list[str]:
    a = s["agreement"]
    out = []
    for name, key in (("bf16 CPU run (the floor)", "floor"), ("this package", "chip")):
        r = a[key]
        out.append(f"| {name} | {', '.join(f'{x:.4f}' for x in r['rel_l2'])} | {', '.join(f'{x:.3f}' for x in r['top1'])} | {', '.join(str(x) for x in r['prefix'])} |")
    return out


def long_rows(s: dict) -> list[str]:
    lg = s["long"]
    return [f"| this package against fp32 | {lg['chip']['rel_l2_mean']:.4f} | {lg['chip']['rel_l2_max']:.3f} | {lg['chip']['top1']:.3f} |",
            f"| bf16 CPU run against fp32 | {lg['floor']['rel_l2_mean']:.4f} | {lg['floor']['rel_l2_max']:.3f} | {lg['floor']['top1']:.3f} |"]


# ---- the card and the other files ---------------------------------------------------------------------------------------------
def render_card(s: dict, license_text: str) -> str:
    kern = s["kernel_modes"]
    lines = ["---", "license: apache-2.0", f"base_model: {MODEL_ID}", "pipeline_tag: text-generation", "tags:", "- blackhole", "- tenstorrent", "- mamba2",
             "- granite", "---", "", f"# {PACKAGE_NAME}", "",
             f"> Runs [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) (revision `{REVISION}`) by IBM on one Tenstorrent Blackhole chip. "
             "The model weights are not in this repository. This is a community package and not a release of IBM or Tenstorrent.", "",
             "A ttnn port of Granite-4.0-H-1B (a 1.5B-parameter hybrid of 36 Mamba-2 layers and 4 attention layers, 40 layers in all), with an OpenAI-compatible "
             "chat server. Weights are bf16 on one chip. The decode step is captured as a ttnn trace and replayed for each token. "
             "Prompts are processed one token per step (a loop of decode steps). Sampling on the host is available, and greedy decoding is what was measured.", "",
             "## Weights are not included", "",
             f"Download them yourself: [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) is Apache-2.0 and is not gated. `run.sh` fetches the pinned revision "
             f"`{REVISION}` into your Hugging Face cache on the first start (about 3 GB).", "",
             "## Try it", "",
             "Needs a Tenstorrent Blackhole chip, a tt-metal build with `ttnn` (tested with tt-metal v0.80.0), and Python 3.12.", "",
             "```bash", "python -m venv venv && . venv/bin/activate", "pip install -r requirements.txt",
             "export TT_METAL_HOME=<your tt-metal checkout>     # run.sh adds its ttnn to PYTHONPATH when `import ttnn` does not work yet",
             "./run.sh                                          # on a shared machine, take a chip lease first", "```", "",
             f"The server listens on http://127.0.0.1:{PORT} (set `PORT`). It binds the local machine only and has no sign-in. "
             "It holds the chip while it runs, and one request runs at a time.", "",
             "## API", "",
             "`GET /v1/models`, `GET /health` and `POST /v1/chat/completions` (`messages`, `max_tokens` or `max_completion_tokens`, `stream`, `stop`, `temperature`, `top_p`, `seed`).", "",
             "- The prompt is the model's own chat template, applied to `messages` with the generation prompt added.",
             "- Generation ends at the end-of-text token, at a `stop` string, at `max_tokens` (default 256, at most 4096), or when prompt plus output reach the context limit.",
             "- `finish_reason` is `stop` when the loop ended on the end-of-text token or a stop string, and `length` when it ended on `max_tokens` or the context limit.",
             "- Without `temperature`, or with 0, decoding is greedy. A temperature above 0 samples on the host.",
             "- A response carries a `timings` object (prefill seconds, first-token seconds, decode tokens per second) next to `usage`.",
             "- Tool calls and document inputs are refused with a 400.", "",
             "```bash", f"curl -s http://127.0.0.1:{PORT}/v1/chat/completions -H 'Content-Type: application/json' \\",
             "  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":32}'", "```", "",
             "## Numbers", "",
             f"Measured on one chip of a Blackhole p300c board with the pinned weights, on a host shared with other jobs. {s['numbers_note']} "
             "A label states how many runs are behind a number.", "",
             "| Number | Value | Label | Evidence |", "| --- | --- | --- | --- |", *number_rows(s), "",
             "## Agreement with the CPU reference", "",
             "Teacher forced over four short prompts (5, 16, 7 and 12 prompt tokens; 24 reference tokens each, fp32 transformers on the CPU is the reference). "
             "The bf16 CPU row is the same model run in bf16 on the CPU, which is the floor any bf16 implementation is read against. "
             "Relative L2 of the logits, mean over positions, then top-1 agreement, then the number of greedy tokens before the first difference (prompt lengths 3, 24, 24, 24 tokens). "
             "Columns list prompts 0 to 3.", "",
             "| Run | Relative L2, mean | Top-1 | First difference at token |", "| --- | --- | --- | --- |", *agreement_rows(s), "",
             f"{s['agreement_note']}", "",
             f"One {s['long']['tokens']}-token natural-language sequence, teacher forced, against fp32 on the CPU:", "",
             "| Run | Relative L2, mean | Relative L2, max | Top-1 |", "| --- | --- | --- | --- |", *long_rows(s), "",
             f"{s['long_note']}", "",
             "## Context and prefill", "",
             f"The context limit is {s['context']} tokens (prompt plus generated). The key/value cache has that fixed length, and every decode step reads all of it, "
             "so the limit sets the step time: the table above lists decode speed at several limits. The model itself supports 131,072 tokens; "
             "this package does not.", "",
             f"{s['prefill_note']}", "",
             "## Limitations", "",
             *[f"- {x}" for x in s["limitations"]], "",
             "## Not tested", "",
             *[f"- {x}" for x in s["not_tested"]], "",
             "## Optional kernel modes", "",
             kern["text"], "",
             "## License", "",
             "The model is licensed under the Apache License 2.0 by IBM (the model repository's metadata says `license: apache-2.0`; it ships no separate license file). "
             "The code in this package is Tenstorrent's, also under the Apache License 2.0. `LICENSE` is the Apache License 2.0 text and `NOTICE` names the parties.", "",
             "## Provenance", "",
             f"- Model: {MODEL_ID}, revision `{REVISION}`; weights are not part of this package.",
             "- `runtime/granite_chip.py`, `server/serve.py`, `tools/evaluate.py` and `tools/smoke.py` come from the tt-orchard repository (tsingletaryTT/tt-orchard).",
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

REQUIREMENTS = """# Python packages for the server and the runtime. ttnn comes from your tt-metal build (see README.md), not from this file.
numpy
torch
safetensors
tokenizers
jinja2
huggingface_hub
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
exec python server/serve.py --port "${{PORT:-{PORT}}}" "$@"
"""


def manifest_text() -> str:
    return f"""# DRAFT. NOT BUILT. A v5.1 container manifest of kind `http-server` for server/serve.py. It cannot be built from this directory alone:
# `source.tt_metal` must be a tt-metal checkout and `source.extra_code` must name this directory. `run.sh` runs the package without tt-model.
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
    files = [{"path": f.relative_to(out).as_posix(), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f in ap._tree(out)]
    _write(out, "package-record.json", json.dumps({"package": PACKAGE_NAME, "model": MODEL_ID, "revision": REVISION, "files": files}, indent=1) + "\n")
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
        for key in ("limitations", "not_tested"):
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
    if rt.get("ready_line") and rt["ready_line"].strip("'") not in serve.replace("\\", "").replace("'", '"') and '"event": "ready"' not in serve.replace("\\", ""):
        out.append("tt-model.yaml: the ready line is not printed by server/serve.py")
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
