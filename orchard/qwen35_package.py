# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage and check the Qwen3.5-35B-A3B package for one Blackhole p300 board (two dies).

The package is a recipe, not a built image. It reproduces the setup measured in docs/run-logs/2026-10-29-fusion-port.md and
docs/run-logs/2026-10-29-qwen35-package.md: the image of the fused-decode bundle `raahemnabeel/qwen3-coder-next` (pulled by the user),
a 41-line patch that is mounted over that image's weight loader so it reads this checkpoint's layout, a script that fetches the text
weights from the Hub at a pinned revision, and a launcher. It is not a v5.1 container package that `tt-model package` can build: that
command needs a tt-metal tree and copies code into a new image, and this package changes one file of someone else's published bundle.
`tt-model.yaml` is therefore a draft that records the serve profile in the v5.1 vocabulary.

`stage_package(src, evidence, out)` copies the source files, renders the card from the measured facts and writes a record with the sha256
of every file. `package_problems(pkg)` is the gate: it reads the directory as it is on disk and returns one sentence per problem. It checks
files and numbers. It does not judge whether prose is true, and it does not run anything. `publish_text()` returns the upload commands as
text; nothing in this module runs them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

from orchard.audio_package import scrub_files
from orchard.decode_floor import ASSUMPTIONS as FLOOR_ASSUMPTIONS

PACKAGE_NAME = "qwen3.5-35b-a3b-blackhole-p300"
MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
IMAGE_TAG = "tt-model/qwen3-coder-next:0c170421ca1b"
IMAGE_ID = "sha256:0c170421ca1b5b95f17ebf28d0c8256ccb8c5d320957bc8c4e72e7a07c43acfb"
LICENSE_SHA256 = None            # set by the staged copy's record; the gate compares the file with the record and with the first line below
LICENSE_FIRST = "Apache License"
SOURCE_FILES = ("fetch_weights.py", "run.sh", "weights_qwen35_moe.patch", "LICENSE", "prompts.json",
                "check/ref_cpu.py", "check/device_quality.py", "check/compare.py", "check/bench.py", "check/prompt.txt")
EXECUTABLE = ("run.sh", "fetch_weights.py")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".tensorbin")
MAX_FILE_BYTES = 5 * 1024 * 1024
REVISION_RE = re.compile(r"\b[0-9a-f]{40}\b")
BANNED_FILES = (".tt_ai_key",)
BANNED_WORDS = ("litellm", ".tt_ai_key")
# phrases the card has to carry: each is a limit or a statement the reader needs
REQUIRED_PHRASES = ("Apache License", "two dies", "vision", "not tested", "revision", "estimate")


class PackageError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _f(x, nd=1):
    return f"{x:.{nd}f}"


def render_card(facts: dict) -> str:
    """The card, from the measured facts. Every number in it comes from `facts`; the gate checks each one again."""
    b, fl, q = facts["bench"], facts["floor"], facts["quality"]
    ab = facts["ablations"]
    cold = facts.get("cold_start_s")
    cold_text = (f"{cold} seconds from container start to the first answer on a first boot with empty kernel and tensor caches"
                 if cold else "not measured")
    rows = "\n".join(
        f"| {r['prompt']!r} | {r['leading_agree']} of {r['tokens']} | {r['same_position_share']} | "
        + (f"token {r['first_diff']['index']}: device token {'inside' if r['first_diff']['device_token_in_ref_top5'] else 'outside'} the reference top 5"
           + (f", reference gap {r['first_diff']['ref_gap_nats']} nats" if r['first_diff']['ref_gap_nats'] is not None else "")
           if "first_diff" in r else "none") + " |"
        for r in q["rows"])
    return f"""---
license: apache-2.0
base_model: {MODEL_ID}
pipeline_tag: text-generation
library_name: vllm
tags: [tenstorrent, blackhole, p300, moe, qwen3.5]
---

# {PACKAGE_NAME}

A launch recipe for [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) (35B parameters, about 3B active per token) on one Tenstorrent Blackhole
p300 board (two dies), using the fused decode ops of the `raahemnabeel/qwen3-coder-next` bundle. This package holds no weights and no image.
It holds a launcher, a 41-line patch for the bundle's weight loader, a script that fetches the text weights, and the checks and results behind
the numbers below. Text only: the vision tower of the checkpoint is not included.

## What you need

- One p300 board with the Tenstorrent driver, 1 GB hugepages mounted at `/dev/hugepages-1G`, and Docker.
- The bundle's image, pulled with `tt-model pull raahemnabeel/qwen3-coder-next` (image id `{IMAGE_ID}`).
- About 70 GB of disk for the weights, and `pip install huggingface_hub`.

## Run it

```
python3 fetch_weights.py ./model            # 69.3 GB, text weights at the pinned revision
MODEL_DIR=$PWD/model DEVICES="0 1" PORT=8000 ./run.sh
```

`run.sh` extracts the image's weight loader, checks its sha256, applies `weights_qwen35_moe.patch`, mounts the result over the image's file
and starts `vllm serve` for one sequence and a 4096-token context. The served model name is `m`. See `docs/running.md`.

## Measured numbers

One p300 board (chips 2 and 3 of a four-chip machine; the other board served a different model), two dies, mesh P300, precision policy
`coder_next_routed_bfp4_mixers_bfp8` (bfp4 routed experts and head, bfp8 attention and shared experts), one sequence.

| Quantity | Value | Label |
| --- | --- | --- |
| Decode speed, median of {b['runs']} runs ({b['tokens']} greedy tokens each, after {b['warmup']} discarded warm-up run) | {_f(b['median'], 2)} tok/s (min {_f(b['min'], 2)}, max {_f(b['max'], 2)}) | benchmark |
| Time to first token for a short prompt | {facts['ttft_ms']} ms | benchmark |
| Estimated floor on decode speed | {_f(fl['floor_tok_s'], 0)} tok/s ({fl['floor_ms']} ms, {_f(fl['mb_per_token'], 0)} MB read per token) | estimate |
| Share of the estimated floor | {_f(100 * b['median'] / fl['floor_tok_s'], 1)} percent | estimate |
| Start, kernel and tensor caches warm | {facts['warm_start_s'][0]} to {facts['warm_start_s'][1]} seconds | single sample per run, three runs |
| Start, empty caches | {cold_text} | single sample |

The floor is `orchard/decode_floor.py` with these inputs: 2 dies at 512 GB/s each (datasheet peak), {fl['policy']}.
Its assumptions: {FLOOR_ASSUMPTIONS} A share of 100 percent cannot be reached.

## Agreement with the Hugging Face reference

The reference is the same checkpoint in bfloat16 on the CPU (Hugging Face transformers, greedy). Each of {len(q['rows'])} fixed prompts
(`prompts.json`) was run for {q['rows'][0]['tokens']} tokens on both. "Leading tokens" is how many tokens agree before the first difference.

| Prompt | Leading tokens that agree | Same token at the same position | First difference |
| --- | --- | --- | --- |
{rows}

The device runs bfp4 routed experts, so near ties between two words can flip. The 16-token check on the earlier run (a prompt about the printing
press) matched for 8 tokens, then flipped a tie of 0.125 nats ("developed" against "invented") and matched again from token 11.

A short chat request through `/v1/chat/completions` answered: {facts['chat']['default']!r} (finish reason {facts['chat']['finish']}).
With thinking off, a request for the primary colors answered: {facts['chat']['thinking_off']!r}.
A prompt of {facts['long_prompt']['tokens']} tokens with a code word at the start and the question at the end answered {facts['long_prompt']['text']!r}.

## Two changes that were tried and not kept

| Change | Median decode speed | Result |
| --- | --- | --- |
| Baseline | {_f(b['median'], 2)} tok/s | |
| Sampler scan grid 10 x 5 instead of 10 x 10 | {_f(ab['grid_median'], 2)} tok/s | inside the baseline spread, under the 3 percent bar, not kept |
| Router top-k constant 72 instead of 80 | {_f(ab['rk_median'], 2)} tok/s | inside the baseline spread, under the 3 percent bar, not kept |

A scan grid wider than 10 columns fails with a `TT_FATAL` (the merge core sits at column 10).

## Limits and what was not tested

- Two dies of one p300 board only. The bundle's fabric layer boundary runs on four dies and is not used here.
- Text only. The vision tower and the multi-token-prediction head are not fetched or served.
- One sequence at a time (`--max-num-seqs 1`) and a 4096-token context. The longest prompt tested was {facts['long_prompt']['tokens']} tokens.
  Longer contexts, concurrent requests, tool calling and sampling at a temperature above zero were not tested.
- The numbers come from one machine and one day, with other jobs on the host (one-minute load average {facts['load']}).
- Cold compile: {cold_text}.
- The patch was made for one image (id above). `run.sh` refuses a loader whose sha256 differs.
- The quality check covers {len(q['rows'])} prompts of {q['rows'][0]['tokens']} tokens. It is not an evaluation benchmark. No accuracy benchmark was run.
- The scores above are for the bfp4 routed-expert policy; a bfloat16 or bfp8 expert policy was not tried.

## License

The weights are {MODEL_ID}, released under the Apache License 2.0 by the Qwen team (Alibaba Cloud). The license file is in this package
(`LICENSE`, copied from the model repository at revision `{REVISION}`). The image and the loader that the patch changes belong to the bundle
`raahemnabeel/qwen3-coder-next`, whose loader file carries an Apache-2.0 header. This package's own files are Apache-2.0. See `NOTICE`.
The weights revision pinned here is `{REVISION}`.
"""


def notice_text() -> str:
    return f"""Qwen3.5-35B-A3B recipe package for Tenstorrent Blackhole

This package contains no model weights and no container image.

The model, {MODEL_ID}, is by the Qwen team (Alibaba Cloud) and is licensed under the Apache License, Version 2.0 (see LICENSE, copied
from the model repository at revision {REVISION}). The weights are fetched from the Hugging Face Hub by fetch_weights.py.

weights_qwen35_moe.patch changes one file (tt/weights.py) of the autoport code in the image {IMAGE_TAG}, built by Tenstorrent from
tt-metal (Apache License 2.0). The change adds a reader for this checkpoint's tensor layout. The change is described in the patch and
in docs/running.md.
"""


def manifest_text() -> str:
    return f"""# DRAFT v5.1 container manifest. It records the serve profile in the tt-model vocabulary. It cannot be built with
# `tt-model package --container`: that command needs a tt-metal tree and copies model code into a new image, and this package
# reuses the published image of raahemnabeel/qwen3-coder-next and replaces one file at run time (run.sh). Use run.sh.
schema: "5.1"
repo: NAMESPACE/{PACKAGE_NAME}
name: {PACKAGE_NAME}
weights: {{repo: {MODEL_ID}, revision: {REVISION}}}     # text tensors only, fetched by fetch_weights.py
kind: vllm-plugin
arch: blackhole
image: {{reuse: {IMAGE_TAG}, id: "{IMAGE_ID}"}}        # not built here
serve:
  mesh_device: P300
  hardware: p300
  max_num_seqs: 1
  block_size: 64
  max_model_len: 4096
  additional_config: {{tt: {{sample_on_device_mode: all, trace_region_size: 805306368, l1_small_size: 24576, fabric_config: FABRIC_1D}}}}
  env:
    ARCH_NAME: blackhole
    QWEN35_ATTN_DECODE_OPS: "1"
    QWEN35_CHUNKED_PREFILL: "1"
    QWEN35_DECODE_REPLICATED: "1"
    QWEN35_GDN_DECODE_OPS: "1"
    QWEN35_MOE_FUSED_DECODE: "1"
    QWEN35_PROGRAM_GUARD: "1"
    QWEN35_ROWS_TOPK: "1"
    QWEN35_TRACE_REGION_MIB: "768"
  args: [--enable-prefix-caching, --enable-chunked-prefill, --max-num-batched-tokens, "2048"]
"""


def running_doc() -> str:
    return f"""# Running {PACKAGE_NAME}

1. Pull the image: `tt-model pull raahemnabeel/qwen3-coder-next` (or load image id {IMAGE_ID} some other way).
2. Fetch the text weights: `python3 fetch_weights.py ./model`. It reads byte ranges of the tensors the text model needs from
   {MODEL_ID} at revision {REVISION} and writes one `model.safetensors`, an index, the tokenizer files and an edited `config.json`.
   The edits: `architectures` is `["Qwen3NextForCausalLM"]` (the class the bundle registers), `mtp_num_hidden_layers` is 0, and the
   `mrope_section` and `mrope_interleaved` keys are removed (with them the vLLM plugin expects a rotary delta from prefill that the
   bundle's generator does not return).
3. Start the server: `MODEL_DIR=$PWD/model DEVICES="0 1" PORT=8000 ./run.sh`. `DEVICES` lists the two dies of one board. On a machine where
   other users hold chips, take a lease first (for example `gozer run`).
4. Ask it: `curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{{"model":"m","messages":[{{"role":"user","content":"Hello"}}],"max_tokens":64}}'`.

What the patch does: the loader of the Qwen3-Coder-Next bundle expects that checkpoint's layout (`model.` prefix, fused GDN projections, one tensor
per expert). Qwen3.5-35B-A3B has a `model.language_model.` prefix, separate GDN projections and fused expert tensors, so the patch adds a branch that maps
those keys to the names the bundle's model code uses, and skips the vision and MTP keys.

Checks, in `check/`: `device_quality.py` (fixed prompts, a chat request, a long prompt), `ref_cpu.py` (the CPU reference), `compare.py`,
`bench.py` (decode speed with a warm-up and measured runs). The first cold start builds kernel and tensor caches in `work/`; later starts reuse them.
"""


def publish_text(namespace: str = "NAMESPACE") -> str:
    """The commands that would publish the staged package, as text. No function in this module runs them."""
    return "\n".join([
        "# The operator has not approved publishing this package. Nothing here was run.",
        f"#    hf upload --repo-type model --private {namespace}/{PACKAGE_NAME} ./{PACKAGE_NAME} .",
        f"#    hf repos settings {namespace}/{PACKAGE_NAME} --no-private      # a separate decision",
        ""])


def stage_package(src, facts: dict, evidence, out) -> dict:
    """Write the package to the new directory `out` (it must not exist) and return its record."""
    src, evidence, out = Path(src), Path(evidence), Path(out)
    if out.exists():
        raise PackageError("the output directory already exists; stage into a new one")
    files: dict[str, bytes | str] = {}
    for rel in SOURCE_FILES:
        files[rel] = (src / rel).read_bytes()
    files["README.md"] = render_card(facts)
    files["NOTICE"] = notice_text()
    files["tt-model.yaml"] = manifest_text()
    files["docs/running.md"] = running_doc()
    files["publish-commands.txt"] = publish_text()
    for p in sorted(evidence.glob("*")):
        if p.is_file():
            files[f"evidence/{p.name}"] = p.read_bytes()
    record = {"package": PACKAGE_NAME, "revision": REVISION, "image_id": IMAGE_ID, "files": {}}
    for rel, data in files.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = data.encode() if isinstance(data, str) else data
        path.write_bytes(raw)
        if rel in EXECUTABLE:
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        record["files"][rel] = hashlib.sha256(raw).hexdigest()
    (out / "package-record.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    return record


def _numbers_in(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def package_problems(pkg, *, hostname: str | None = None, home: str | None = None) -> list[str]:
    """One sentence per problem found in the directory `pkg`; empty when it passes."""
    pkg = Path(pkg)
    problems: list[str] = []
    if not pkg.is_dir():
        return ["the package directory does not exist"]
    problems += [f"scrub: {h}" for h in scrub_files(pkg, hostname=hostname, home=home)]
    all_files = [p for p in pkg.rglob("*") if p.is_file() or p.is_symlink()]
    for p in all_files:
        rel = p.relative_to(pkg).as_posix()
        if p.is_symlink():
            problems.append(f"{rel} is a link")
            continue
        if p.suffix.lower() in WEIGHT_SUFFIXES or p.stat().st_size > MAX_FILE_BYTES:
            problems.append(f"{rel} looks like weights or is larger than {MAX_FILE_BYTES} bytes")
        if p.name in BANNED_FILES:
            problems.append(f"{rel} is a credential file")
        try:
            text = p.read_text(encoding="utf-8").lower()
        except (UnicodeDecodeError, OSError):
            continue
        for w in BANNED_WORDS:
            if w in text and not rel.startswith("evidence/ledger"):
                problems.append(f"{rel} mentions {w!r}")
    need = ["README.md", "NOTICE", "LICENSE", "run.sh", "fetch_weights.py", "weights_qwen35_moe.patch", "tt-model.yaml", "package-record.json"]
    for n in need:
        if not (pkg / n).is_file():
            problems.append(f"{n} is missing")
    if problems and any("is missing" in x for x in problems):
        return problems
    lic = (pkg / "LICENSE").read_text(encoding="utf-8")
    if LICENSE_FIRST not in lic[:200] or "Version 2.0" not in lic[:300]:
        problems.append("LICENSE is not the Apache License 2.0 text")
    card = (pkg / "README.md").read_text(encoding="utf-8")
    if not re.search(r"^license:\s*apache-2\.0\s*$", card, re.M):
        problems.append("the card front matter does not say license: apache-2.0")
    for ph in REQUIRED_PHRASES:
        if ph not in card:
            problems.append(f"the card lacks the phrase {ph!r}")
    revs = {n: set(REVISION_RE.findall((pkg / n).read_text(encoding="utf-8"))) for n in ("README.md", "tt-model.yaml", "fetch_weights.py", "NOTICE")}
    for n, r in revs.items():
        if r != {REVISION}:
            problems.append(f"{n} does not pin exactly the revision {REVISION} (found {sorted(r)})")
    if IMAGE_ID.split(":")[1] not in (pkg / "run.sh").read_text(encoding="utf-8"):
        problems.append("run.sh does not name the image id the numbers were measured on")
    # the numbers in the card's tables must be in the evidence the package carries
    ev = "".join(p.read_text(encoding="utf-8", errors="replace") for p in (pkg / "evidence").glob("*") if p.is_file()) if (pkg / "evidence").is_dir() else ""
    facts_file = pkg / "evidence" / "facts.json"
    if not facts_file.is_file():
        problems.append("evidence/facts.json is missing")
    else:
        facts = json.loads(facts_file.read_text(encoding="utf-8"))
        want = [facts["bench"]["median"], facts["bench"]["min"], facts["bench"]["max"], facts["ablations"]["grid_median"], facts["ablations"]["rk_median"]]
        for v in want:
            if f"{v:.2f}" not in card:
                problems.append(f"the card does not show the measured value {v:.2f}")
        share = f"{100 * facts['bench']['median'] / facts['floor']['floor_tok_s']:.1f} percent"
        if share not in card:
            problems.append(f"the card does not show the share of the floor, {share}")
        if "Start, empty caches | not measured" in card and facts.get("cold_start_s"):
            problems.append("the card says the cold start was not measured but the facts have it")
    rec = json.loads((pkg / "package-record.json").read_text(encoding="utf-8"))
    for rel, sha in rec.get("files", {}).items():
        f = pkg / rel
        if not f.is_file():
            problems.append(f"{rel} is in the record and missing on disk")
        elif _sha(f) != sha:
            problems.append(f"{rel} was changed after staging (sha256 differs from the record)")
    on_disk = {p.relative_to(pkg).as_posix() for p in all_files if p.name != "package-record.json"}
    for extra in sorted(on_disk - set(rec.get("files", {}))):
        problems.append(f"{extra} is not in the record")
    for rel in EXECUTABLE:
        if (pkg / rel).is_file() and not os.access(pkg / rel, os.X_OK):
            problems.append(f"{rel} is not executable")
    return problems
