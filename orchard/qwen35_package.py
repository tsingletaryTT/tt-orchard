# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage and check the Qwen3.5-35B-A3B package for one Blackhole p300 board (two dies).

The package is a recipe and holds no built image. It reproduces the setup measured in docs/run-logs/2026-10-29-fusion-port.md and
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
from orchard.decode_floor import ASSUMPTIONS as FLOOR_ASSUMPTIONS, floor_for_board
from orchard import model_config as mc

PACKAGE_NAME = "qwen3.5-35b-a3b-blackhole-p300"
MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
IMAGE_TAG = "tt-model/qwen3-coder-next:0c170421ca1b"
IMAGE_ID = "sha256:0c170421ca1b5b95f17ebf28d0c8256ccb8c5d320957bc8c4e72e7a07c43acfb"
LICENSE_SHA256 = None            # set by the staged copy's record; the gate compares the file with the record and with the first line below
LICENSE_FIRST = "Apache License"
SOURCE_FILES = ("fetch_weights.py", "run.sh", "pins.env", "weights_qwen35_moe.patch", "LICENSE", "prompts.json",
                "check/ref_cpu.py", "check/device_quality.py", "check/compare.py", "check/bench.py", "check/prompt.txt")
EXECUTABLE = ("run.sh", "fetch_weights.py")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".tensorbin")
MAX_FILE_BYTES = 5 * 1024 * 1024
REVISION_RE = re.compile(r"\b[0-9a-f]{40}\b")
BANNED_FILES = (".tt_ai_key",)
BANNED_WORDS = ("litellm", ".tt_ai_key")
# phrases the card has to carry: each is a limit or a statement the reader needs
LOADER_PATH = "/opt/tt-metal/models/autoports/qwen_qwen3_coder_next/tt/weights.py"
ORIG_SHA = "a8be9a77d9190b6d13fa64a57bde649b6b7885b0711da6b2825cf65e924c8245"
PATCHED_SHA = "1b94978020b09501cdc404d55041f143b67258dbb609c485e1ac68bae54e8a7b"
POLICY = {"attention": "bfp8", "dense_mlp": "bfp8", "router": "bf16", "routed_experts": "bfp4", "shared_experts": "bfp8", "lm_head": "bfp4"}
POLICY_TEXT = "bfp8 attention and shared experts, bfp4 routed experts and head, bf16 router"
EARLIER_WARM = "267 to 272 seconds in the earlier fusion-port runs (the kernel cache was seeded from the coder's)"
# every file the evidence directory must hold; the facts are derived from these and nothing else
EVIDENCE_FILES = ("bench_package.log", "bench_40layer_grid105.log", "bench_40layer_rk72.log", "boot.txt", "cpu_reference.json",
                  "device_quality.json", "config.json")
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
    diffs = [r['first_diff'] for r in q['rows'] if 'first_diff' in r]
    if not diffs:
        agree_text = "All tokens of every prompt matched the reference."
    elif all(d['device_token_in_ref_top5'] for d in diffs):
        gaps = [d['ref_gap_nats'] for d in diffs if d['ref_gap_nats'] is not None]
        agree_text = (f"In every prompt that differs, the first difference is a position where the device's token is inside the reference's top 5 "
                      f"(largest gap {max(gaps) if gaps else 0} nats).")
    else:
        agree_text = "In at least one prompt the device's token at the first difference is outside the reference's top 5; see the table."
    n_same = sum(1 for r in q['rows'] if 'first_diff' not in r)
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

One p300 board (chips 2 and 3 of a four-chip machine), two dies, mesh P300, precision policy
`coder_next_routed_bfp4_mixers_bfp8` (bfp4 routed experts and head, bfp8 attention and shared experts), one sequence.

| Quantity | Value | Label |
| --- | --- | --- |
| Decode speed, median of {b['runs']} runs ({b['tokens']} greedy tokens each, after {b['warmup']} discarded warm-up run) | {_f(b['median'], 2)} tok/s (min {_f(b['min'], 2)}, max {_f(b['max'], 2)}) | benchmark |
| Time to first token for a short prompt | {facts['ttft_ms']} ms | benchmark |
| Estimated floor on decode speed | {_f(fl['floor_tok_s'], 0)} tok/s ({fl['floor_ms']} ms, {_f(fl['mb_per_token'], 0)} MB read per token) | estimate |
| Share of the estimated floor | {_f(100 * b['median'] / fl['floor_tok_s'], 1)} percent | estimate |
| Start, kernel and tensor caches warm | {facts['warm_start_s']} seconds on the boot check; {EARLIER_WARM} | single sample (one boot) |
| Start, empty caches | {cold_text} | single sample |

The floor is `orchard/decode_floor.py` with these inputs: 2 dies at 512 GB/s each (datasheet peak), {POLICY_TEXT}.
Its assumptions: {FLOOR_ASSUMPTIONS} A share of 100 percent cannot be reached.

## Agreement with the Hugging Face reference

The reference is the same checkpoint in bfloat16 on the CPU (Hugging Face transformers, greedy). Each of {len(q['rows'])} fixed prompts
(`prompts.json`) was run for {q['rows'][0]['tokens']} tokens on both. "Leading tokens" is how many tokens agree before the first difference.

| Prompt | Leading tokens that agree | Same token at the same position | First difference |
| --- | --- | --- | --- |
{rows}

{agree_text} The device runs bfp4 routed experts, so a near tie between two words can flip. After a flip the two free-running texts diverge, so the
leading-token counts are short even when the device text is fluent. {n_same} of {len(q['rows'])} prompts matched in full. The counts are what was measured; no claim is made
that a longer prefix would agree. The earlier 16-token check in `2026-10-29-fusion-port.md` (a different prompt) matched for 8 tokens, then flipped a tie of 0.125 nats and matched again from token 11.

A short chat request through `/v1/chat/completions` answered: {facts['chat']['default']!r} (finish reason {facts['chat']['finish']}).
With thinking off, a request for the primary colors answered: {facts['chat']['thinking_off']!r}.
A prompt of {facts['long_prompt']['tokens']} tokens with a code word at the start and the question at the end answered {facts['long_prompt']['text']!r}.

## Two changes that were tried and not kept

| Change | Median decode speed | Result |
| --- | --- | --- |
| Baseline | {_f(b['median'], 2)} tok/s | |
| Sampler scan grid 10 x 5 (default 10 x 10) | {_f(ab['grid_median'], 2)} tok/s | inside the baseline spread, under the 3 percent bar, dropped |
| Router top-k constant 72 (default 80) | {_f(ab['rk_median'], 2)} tok/s | inside the baseline spread, under the 3 percent bar, dropped |

A scan grid wider than 10 columns fails with a `TT_FATAL` (the merge core sits at column 10).

## Limits and what was not tested

- Two dies of one p300 board only. The bundle's fabric layer boundary runs on four dies and is not used here.
- Text only. The vision tower and the multi-token-prediction head are not fetched or served.
- One sequence at a time (`--max-num-seqs 1`) and a 4096-token context. The longest prompt tested was {facts['long_prompt']['tokens']} tokens.
  Longer contexts, concurrent requests, tool calling and sampling at a temperature above zero were not tested.
- The numbers come from one machine and one day, with other jobs on the host (one-minute load average {facts['load']}).
- Cold compile: {cold_text}.
- The warm start was measured once. The board's lease ended during that boot check, so another job may have used the board at the same time; the figure was not re-measured.
- The patch was made for one image (id above). `run.sh` refuses a loader whose sha256 differs.
- The quality check covers {len(q['rows'])} prompts of {q['rows'][0]['tokens']} tokens. It is not an evaluation benchmark. No accuracy benchmark was run.
- The default chat template starts a thinking section; a 96-token reply can end inside it (finish reason `length`). Pass `chat_template_kwargs {{"enable_thinking": false}}` to skip it.
- Raw completion prompts are not chat-formatted: one test prompt (Spanish) continued into a thinking block after its first line.
- The scores above are for the bfp4 routed-expert policy; a bfloat16 or bfp8 expert policy was not tried.

## License

The weights are {MODEL_ID}, released under the Apache License 2.0 by the Qwen team (Alibaba Cloud). The license file is in this package
(`LICENSE`, copied from the model repository at revision `{REVISION}`). The image and the loader that the patch changes belong to the bundle
`raahemnabeel/qwen3-coder-next`, whose loader file carries an Apache-2.0 header. This package's own files are Apache-2.0. See `NOTICE`.
The weights revision pinned here is `{REVISION}`.
"""


def _bench_line(path: Path) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("BENCH "):
            d = json.loads(line[6:])
            if not d.get("runs"):
                raise PackageError(f"{path.name} has no measured runs")
            return d
    raise PackageError(f"{path.name} has no BENCH line")


def _compare_module(base: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("qwen35_compare", base / "check" / "compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def facts_from_evidence(evidence, compare_dir) -> dict:
    """Every measured number on the card, recomputed from the raw files in `evidence`. `compare_dir` holds `check/compare.py`.
    Raises PackageError when a file is missing or cannot be read."""
    ev = Path(evidence)
    for n in EVIDENCE_FILES:
        if not (ev / n).is_file():
            raise PackageError(f"evidence/{n} is missing")
    try:
        b = _bench_line(ev / "bench_package.log")
        runs = b["runs"]
        boots = {}
        for line in (ev / "boot.txt").read_text(encoding="utf-8").splitlines():
            m = re.match(r"BOOT (cold|warm) api_up_s=(\d+) first_answer_s=(\d+)$", line.strip())
            if m:
                boots[m.group(1)] = int(m.group(3))
        if set(boots) != {"cold", "warm"}:
            raise PackageError("evidence/boot.txt needs one cold and one warm BOOT line")
        ref = json.loads((ev / "cpu_reference.json").read_text(encoding="utf-8"))
        dev = json.loads((ev / "device_quality.json").read_text(encoding="utf-8"))
        rows = _compare_module(Path(compare_dir)).compare_rows(ref, dev["completions"])
        cfg = mc.read_model_config(ev / "config.json")
        floor = floor_for_board(cfg, POLICY, "p300")
        ttft = sorted(r["ttft_s"] for r in runs)
        chat = dev["chat"]
        return {
            "bench": {"median": b["median"], "min": b["min"], "max": b["max"], "runs": len(runs), "warmup": 1, "tokens": runs[0]["tokens"]},
            "floor": {"floor_tok_s": round(floor.floor_tok_s, 1), "floor_ms": round(floor.floor_ms, 3), "mb_per_token": round(floor.bytes_per_token / 1e6, 1)},
            "ttft_ms": round(1000 * ttft[len(ttft) // 2]), "warm_start_s": boots["warm"], "cold_start_s": boots["cold"],
            "load": "%.1f to %.1f" % (min(r["load1"] for r in runs), max(r["load1"] for r in runs)),
            "quality": {"rows": rows},
            "chat": {"default": chat["default"]["text"][:160] + "...", "finish": chat["default"]["finish"], "thinking_off": chat["thinking_off"]["text"]},
            "long_prompt": {"tokens": dev["long_prompt"]["prompt_tokens"], "text": dev["long_prompt"]["text"].split("\n")[0]},
            "ablations": {"grid_median": _bench_line(ev / "bench_40layer_grid105.log")["median"],
                          "rk_median": _bench_line(ev / "bench_40layer_rk72.log")["median"]},
        }
    except (KeyError, ValueError, TypeError, OSError, mc.ModelConfigError) as exc:
        if isinstance(exc, PackageError):
            raise
        raise PackageError(f"the evidence could not be used: {type(exc).__name__}: {exc}") from exc


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


def stage_package(src, evidence, out) -> dict:
    """Write the package to the new directory `out` (it must not exist) and return its record. The card is rendered from facts recomputed
    from the raw files in `evidence`; those files and the facts are copied into the package."""
    src, evidence, out = Path(src), Path(evidence), Path(out)
    if out.exists():
        raise PackageError("the output directory already exists; stage into a new one")
    facts = facts_from_evidence(evidence, src)
    files: dict[str, bytes | str] = {}
    for rel in SOURCE_FILES:
        files[rel] = (src / rel).read_bytes()
    files["README.md"] = render_card(facts)
    files["NOTICE"] = notice_text()
    files["tt-model.yaml"] = manifest_text()
    files["docs/running.md"] = running_doc()
    files["publish-commands.txt"] = publish_text()
    for n in EVIDENCE_FILES:
        files[f"evidence/{n}"] = (evidence / n).read_bytes()
    files["evidence/facts.json"] = json.dumps(facts, indent=1, sort_keys=True)
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


def _pins(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def package_problems(pkg, *, hostname: str | None = None, home: str | None = None) -> list[str]:
    """One sentence per problem found in the directory `pkg`; empty when it passes."""
    pkg = Path(pkg)
    if not pkg.is_dir():
        return ["the package directory does not exist"]
    # Stage 1: facts about the files that need no reading. A link, a weights file, a credential file or an oversized file is reported
    # and the gate returns at once, so nothing below reads, hashes or parses what was rejected.
    early: list[str] = []
    for p in sorted(pkg.rglob("*")):
        rel = p.relative_to(pkg).as_posix()
        if p.is_symlink():
            early.append(f"{rel} is a link")
        elif p.is_file():
            if p.name in BANNED_FILES:
                early.append(f"{rel} is a credential file")
            if p.suffix.lower() in WEIGHT_SUFFIXES or p.stat().st_size > MAX_FILE_BYTES:
                early.append(f"{rel} looks like weights or is larger than {MAX_FILE_BYTES} bytes")
    if early:
        return early
    problems: list[str] = [f"scrub: {h}" for h in scrub_files(pkg, hostname=hostname, home=home)]
    all_files = [p for p in pkg.rglob("*") if p.is_file()]
    for p in all_files:
        rel = p.relative_to(pkg).as_posix()
        try:
            text = p.read_text(encoding="utf-8").lower()
        except (UnicodeDecodeError, OSError):
            continue
        for w in BANNED_WORDS:
            if w in text:
                problems.append(f"{rel} mentions {w!r}")
    need = ["README.md", "NOTICE", "LICENSE", "run.sh", "pins.env", "fetch_weights.py", "weights_qwen35_moe.patch", "tt-model.yaml",
            "package-record.json", "check/compare.py"] + [f"evidence/{n}" for n in EVIDENCE_FILES] + ["evidence/facts.json"]
    missing = [n for n in need if not (pkg / n).is_file()]
    if missing:
        return problems + [f"{n} is missing" for n in missing]
    lic = (pkg / "LICENSE").read_text(encoding="utf-8")
    if LICENSE_FIRST not in lic[:200] or "Version 2.0" not in lic[:300]:
        problems.append("LICENSE is not the Apache License 2.0 text")
    card = (pkg / "README.md").read_text(encoding="utf-8")
    if not re.search(r"^license:\s*apache-2\.0\s*$", card, re.M):
        problems.append("the card front matter does not say license: apache-2.0")
    for ph in REQUIRED_PHRASES:
        if ph not in card:
            problems.append(f"the card lacks the phrase {ph!r}")
    for n in ("README.md", "tt-model.yaml", "fetch_weights.py", "NOTICE"):
        found = set(REVISION_RE.findall((pkg / n).read_text(encoding="utf-8")))
        if found != {REVISION}:
            problems.append(f"{n} does not pin exactly the revision {REVISION} (found {sorted(found)})")
    pins = _pins(pkg / "pins.env")
    for k, v in (("IMAGE_ID", IMAGE_ID), ("LOADER", LOADER_PATH), ("ORIG_SHA", ORIG_SHA), ("PATCHED_SHA", PATCHED_SHA)):
        if pins.get(k) != v:
            problems.append(f"pins.env {k} is {pins.get(k)!r}, expected {v!r}")
    run = (pkg / "run.sh").read_text(encoding="utf-8")
    if "pins.env" not in run or "PATCHED_SHA" not in run:
        problems.append("run.sh does not read pins.env and check PATCHED_SHA")
    # The card must equal a rendering of facts recomputed from the raw evidence, and facts.json must equal those facts. A number edited in
    # the card, in facts.json or in a raw file fails here.
    try:
        facts = facts_from_evidence(pkg / "evidence", pkg)
    except PackageError as exc:
        problems.append(str(exc))
        facts = None
    if facts is not None:
        stored = json.loads((pkg / "evidence" / "facts.json").read_text(encoding="utf-8"))
        if stored != json.loads(json.dumps(facts)):
            diff = sorted(k for k in set(stored) | set(facts) if stored.get(k) != facts.get(k))
            problems.append(f"evidence/facts.json differs from the facts recomputed from the raw evidence in: {', '.join(diff)}")
        if card != render_card(facts):
            problems.append("README.md is not the card rendered from the evidence (a measured value or sentence was edited)")
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
