# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The package builder for a finished text-runtime run (the text counterpart of orchard/audio_package.py).

It stages a local, weights-free package from the run's results under `<out>/<package name>/` and checks it. It never uploads or publishes: `publish_commands`
returns text.

    README.md              the card, rendered from the run's result files (`render_card`); every number is a table row that the gate recomputes
    LICENSE                the license text the model's repository states (a copy of the file given with --license-file, checked for its sha256)
    NOTICE                 what the model's card says about the weights' terms, quoted, and what this package adds
    tt-model.yaml          a v5.1 container manifest of kind `http-server` (a draft: it names a tt-metal checkout it cannot supply)
    run.sh                 a direct launcher for the OpenAI-compatible server; it checks the model's config.json against the pinned sha256 first
    requirements.txt       the host Python packages the scripts import besides ttnn
    runtime/scripts/<dir>/ the adapter's scripts, copied unmodified; runtime/orchard/ holds the orchard modules they import, copied unmodified
    docs/running.md        how to start it and how to call it
    evidence/              the run's three result files and the earlier context-length measurement the card cites
    package-record.json    the sha256 of every file and what each came from

No weights are staged. `package_problems` is the gate; it reads the directory as it is on disk, so a card edited after it was rendered is checked as edited.

What the gate does not do: judge whether prose is true, catch a claim worded in a way its patterns do not list, or decide whether Tenstorrent may publish a
model under its license (the card says what the model's repository states and what could not be verified).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path

from orchard import audio_package as ap
from orchard import safeio
from orchard import text_runtime as rt
from orchard.card_use import PERF_IN_PROSE

SCHEMA = 1
RECORD_FILE = "package-record.json"
EVIDENCE = ("evidence/text_reference.json", "evidence/text_result.json", "evidence/text_bench.json")
CONTEXT_EVIDENCE = "evidence/context-bench.json"
FORBIDDEN_WORDS = (".tt_ai_key", "litellm", "HF_TOKEN=", "/home/")
COMMERCIAL_CLAIMS = (r"(?<!non-)(?<!non )commercial use is (?:allowed|permitted|free|unrestricted|fine)", r"free for commercial", r"free to use commercially")
REVISION = re.compile(r"[0-9a-f]{40}")
HOST_REQUIREMENTS = ("numpy", "safetensors", "torch", "transformers")
NOT_TESTED_HEAD = "## Not tested"
NUMBERS_HEAD = "## Numbers"
LICENSE_HEAD = "## License"
PORT = 20110


class TextPackageError(Exception):
    """The run or the inputs cannot back a package."""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path, what) -> dict:
    try:
        return safeio.read_json(path, max_bytes=8 << 20, expect=dict)
    except (OSError, ValueError, safeio.SafeReadError) as exc:
        raise TextPackageError(f"{what} cannot be read: {exc}") from exc


# ---- the numbers ------------------------------------------------------------------------------------------------------------------

def fmt(x, digits=4) -> str:
    return f"{x:.{digits}g}" if isinstance(x, float) else str(x)


def number_rows(res: dict, bench: dict, context: dict | None) -> list[tuple[str, str, str, str]]:
    """(name, value, label, evidence) for every number on the card, from the result files. The gate calls this on the package's own evidence and compares."""
    rows = []
    for name, v in sorted((bench.get("variants") or {}).items()):
        s = v["decode_tok_s"]
        load = [x for x in (v.get("summary") or {}).get("host_loadavg_1m", []) if isinstance(x, (int, float))]
        load_text = f"; host load average {min(load):.1f} to {max(load):.1f} while it ran" if load else "; host load not recorded"
        rows.append((f"decode speed, {name}", f"{fmt(s['median'], 4)} tok/s (min {fmt(s['min'], 4)}, max {fmt(s['max'], 4)}){load_text}", v["label"], "evidence/text_bench.json"))
        rows.append((f"share of the decode floor, {name}", f"{v['share_of_floor_median'] * 100:.1f} percent of {v['floor']['floor_tok_s']:.1f} tok/s", "estimate",
                     "evidence/text_bench.json"))
    for name, v in sorted((res.get("variants") or {}).items()):
        a = v["numeric"]["agreement"]
        label = "measured (provisional bar met)" if v.get("ok") else "measured (the stage 2 bar was not met)"
        rows.append((f"teacher-forced next-token agreement, {name}", f"{a['matches']} of {a['of']}", label, "evidence/text_result.json"))
        p = v["perplexity"]
        rows.append((f"perplexity on {p['tokens']} tokens, {name}", f"{fmt(p['chip_ppl'], 5)} on the chip, {fmt(p['reference_ppl'], 5)} on the CPU in float32",
                     label, "evidence/text_result.json"))
        w = v["numeric"]["worst_ratio_to_floor"]
        if w:
            rows.append((f"worst block error over the CPU bfloat16 error, {name}", f"{fmt(w['ratio'], 3)} at {w['name']}", label, "evidence/text_result.json"))
    if context:
        rows.append(("decode speed at the deepest position measured", f"{fmt(context['tok_s'], 4)} tok/s for {context['steps']} steps from position {context['start_position']}",
                     "measured in an earlier run, not repeated by the stages", CONTEXT_EVIDENCE))
    return rows


def read_context(path) -> dict | None:
    """The earlier context-length record (PR 210's ctx_bench.py): decode speed in windows at growing positions. Returns the last window's start position,
    steps and speed, or None when no file is given."""
    if not path:
        return None
    data = _json(path, "the context-length record")
    if data.get("tp") != 1:
        raise TextPackageError(f"the context-length record is for tp={data.get('tp')!r}; this package is for one chip, so only a one-chip (tp 1) record can be cited")
    wins = [w for w in data.get("windows") or [] if isinstance(w, dict) and all(isinstance(w.get(k), (int, float)) for k in ("start_position", "steps", "tok_s"))]
    if not wins:
        raise TextPackageError("the context-length record has no window with a start position, steps and speed")
    last = max(wins, key=lambda w: w["start_position"])
    return {"start_position": int(last["start_position"]), "steps": int(last["steps"]), "tok_s": last["tok_s"], "windows": len(wins)}


# ---- rendering ----------------------------------------------------------------------------------------------------------------------

def _row(r) -> str:
    return f"| {r[0]} | {r[1]} | {r[2]} | `{r[3]}` |"


def render_card(rtm, res: dict, bench: dict, context: dict | None, *, revision: str, namespace: str, license_name: str, weights_terms: str) -> str:
    rows = number_rows(res, bench, context)
    variants = {v.name: v for v in rtm.variants}
    lines = ["---", f"license: {rtm.license_id}", f"base_model: {rtm.model_ids[0]}", "library_name: ttnn", "pipeline_tag: text-generation", "tags:", "- tenstorrent",
             "- blackhole", "- text-generation", "---", "", f"# {rtm.package_name}", "",
             f"A runtime for `{rtm.model_ids[0]}` (revision `{revision}`) on one Tenstorrent Blackhole chip, with an OpenAI-compatible chat server. The package holds the "
             "scripts and no weights. Download the weights from the model's own repository.", "",
             "Variants:", ""]
    lines += [f"- `{n}`: {v.description}" for n, v in variants.items()]
    lines += ["", NUMBERS_HEAD, "", "Every number comes from the run's result files under `evidence/`. The bars the stages used are provisional choices, set from the CPU "
              "bfloat16 error; they are not a measured chip noise level.", "", "| Number | Value | Label | Evidence |", "|---|---|---|---|"]
    lines += [_row(r) for r in rows]
    for name, v in sorted((res.get("variants") or {}).items()):
        if not v.get("ok"):
            lines += ["", f"The `{name}` variant did not meet the numeric bar of stage 2: at least one block's error against the float32 reference is above the bar's multiple of the CPU "
                      "bfloat16 error. Its agreement and perplexity rows are listed above as measured. `evidence/text_result.json` names the first block that left."]
    lines += ["", "## Running it", "", "See `docs/running.md`. In short: set `MINICPM3_DIR` and `MINICPM3_WEIGHTS` (the repository's files, and the weights as one safetensors file made by "
              "`convert_weights.py`), take a chip lease, and run `./run.sh`. The server listens on 127.0.0.1 and answers `POST /v1/chat/completions`.", "",
              LICENSE_HEAD, "", f"The model's repository card states the license `{rtm.license_id}` ({license_name}). `LICENSE` is that license's text. {weights_terms}", "",
              "The scripts in this package are covered by the Apache License 2.0 (tt-orchard).", "", NOT_TESTED_HEAD, ""]
    lines += [f"- {x}" for x in (*rtm.limits, "Nobody has judged the quality of the generated text beyond the agreement and perplexity numbers above.",
                                f"The server's default context is {rtm.context_limit} tokens (`--max-len` raises it). Speed was measured out to the deepest position in the table. "
                                "Agreement and perplexity were measured over the first few hundred positions only; nothing here measures accuracy at longer contexts.",
                                "The package was boot-checked once on one chip (see `docs/running.md`); no load test or concurrent use was run.",
                                "The tt-model manifest is a draft and was not built or validated by a registry.")]
    return "\n".join(lines) + "\n"


def requirements_text() -> str:
    return "\n".join(["# the host Python packages the scripts import; ttnn comes from the Tenstorrent environment", "numpy", "safetensors", "torch", "transformers>=5.0",
                      "pandas", "pyarrow", ""])


def manifest_text(rtm, revision: str, namespace: str) -> str:
    return f"""# DRAFT. NOT BUILT. A v5.1 container manifest of kind `http-server` for runtime/scripts/{rtm.scripts_dir.name}/serve.py. It cannot be built from this directory
# alone: `source.tt_metal` must be a tt-metal checkout and `source.extra_code` must name this directory. Use `run.sh` to run the package without tt-model.
schema: "5.1"
repo: {namespace}/{rtm.package_name}
name: {rtm.package_name}
weights:
  repo: {rtm.model_ids[0]}
  revision: "{revision}"
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
      paths: [runtime]
  ubuntu: "22.04"
  python: "3.12"
runtime:
  command: [python, runtime/scripts/{rtm.scripts_dir.name}/serve.py, --port, "{{port}}"]
  ready_line: "serving"
  packages: ["numpy", "safetensors", "torch", "transformers"]
serve:
  port: {PORT}
serve_profiles:
  - name: p150
    description: One Blackhole chip (a P150, or one chip of a P300 or QuietBox 2).
    hardware: p150
    mesh_device: P150
default_profile: p150
"""


def run_script(rtm, config_sha256: str, revision: str) -> str:
    d = rtm.scripts_dir.name
    return f"""#!/usr/bin/env bash
# Starts the OpenAI-compatible server for {rtm.model_ids[0]} on one Blackhole chip. This package does not include the weights.
# Take a chip lease first (gozer run --chips 1 ... -- ./run.sh). The server binds 127.0.0.1 only.
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
: "${{{rtm.model_dir_env}:?set {rtm.model_dir_env} to a download of {rtm.model_ids[0]} at revision {revision} (config.json and the tokenizer files)}}"
: "${{{rtm.weights_env}:?set {rtm.weights_env} to the weights as one safetensors file (runtime/scripts/{d}/convert_weights.py makes it)}}"
PYTHON="${{ORCHARD_TEXT_PYTHON:-python3}}"
PINNED_CONFIG_SHA256="{config_sha256}"
GOT="$(sha256sum "${rtm.model_dir_env}/config.json" | cut -d' ' -f1)"
if [ "$GOT" != "$PINNED_CONFIG_SHA256" ]; then
  echo "run.sh: ${rtm.model_dir_env}/config.json is not the pinned configuration (revision {revision}); stopping" >&2
  exit 3
fi
PORT="${{PORT:-8210}}"
# The checked directory and weights come last, so an argument forwarded in "$@" cannot point the server at files that were not checked.
exec "$PYTHON" "$HERE/runtime/scripts/{d}/serve.py" --port "$PORT" "$@" {rtm.model_dir_flag} "${rtm.model_dir_env}" {rtm.weights_flag} "${rtm.weights_env}"
"""


def running_doc(rtm, revision: str, config_sha256: str) -> str:
    d = rtm.scripts_dir.name
    return f"""# Running {rtm.package_name}

1. Get the model files: `hf download {rtm.model_ids[0]} --revision {revision} --local-dir MODEL_DIR` (this package holds no weights).
2. Convert the weights once: `python runtime/scripts/{d}/convert_weights.py --model-dir MODEL_DIR --out MODEL.safetensors`. The pickle is read with `weights_only=True`.
3. `export MINICPM3_DIR=MODEL_DIR MINICPM3_WEIGHTS=MODEL.safetensors`. `run.sh` stops unless `MODEL_DIR/config.json` has the sha256 `{config_sha256}`.
4. Take a chip lease for one chip and run `./run.sh` (add `--mlp-dtype bfp8 --mlp-fidelity lofi` for the 8-bit MLP variant).
5. Ask it: `curl -s 127.0.0.1:8210/v1/chat/completions -d '{{"messages":[{{"role":"user","content":"Hello"}}],"max_tokens":64}}'`.

The server runs one request at a time and decodes greedily; `temperature` and `top_p` are accepted and ignored. Each request restarts at position 0 and the
prompt is fed one token per step.

## Boot check

`boot-check.json` in the staging directory (not part of the package) records the one start of `run.sh` on a chip before this package was handed over.
"""


# ---- the runtime files ----------------------------------------------------------------------------------------------------------------

def orchard_imports(source: str) -> set[str]:
    """The `orchard` modules a Python source imports: `from orchard import a, b`, `from orchard.a import x`, `import orchard.a`."""
    out = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module == "orchard":
                out |= {a.name for a in node.names}
            elif node.module and node.module.startswith("orchard."):
                out.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            out |= {a.name.split(".")[1] for a in node.names if a.name.startswith("orchard.")}
    return out


def orchard_closure(sources: dict[str, str], root: Path) -> list[str]:
    """Names of the orchard modules (files under `root/orchard`) that the sources import, directly or through each other."""
    need, todo = set(), set().union(*(orchard_imports(s) for s in sources.values())) if sources else set()
    while todo:
        m = todo.pop()
        f = root / "orchard" / f"{m}.py"
        if m in need or not f.is_file():
            continue
        need.add(m)
        todo |= orchard_imports(f.read_text(encoding="utf-8")) - need
    return sorted(need)


# ---- staging ------------------------------------------------------------------------------------------------------------------------

def _recorded_scripts(run: Path) -> dict:
    """The script hashes the run's `text runtime selected` decision recorded at stage 0."""
    try:
        for line in (run / "ledger.jsonl").read_text(encoding="utf-8").splitlines():
            if rt.TEXT_RUNTIME_SELECTED in line:
                e = json.loads(line)
                if e.get("data", {}).get("decision") == rt.TEXT_RUNTIME_SELECTED and isinstance(e["data"].get("scripts"), dict):
                    return e["data"]["scripts"]
    except (OSError, ValueError):
        pass
    return {}


def ledger_stage_ends(run: Path) -> tuple[dict, dict]:
    """(result by stage, evidence list of the last passing stage_end by stage) from the run's ledger."""
    ends, evidence = {}, {}
    try:
        for line in (run / "ledger.jsonl").read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e.get("event") == "stage_end":
                ends[e["stage"]] = e["data"].get("result")
                if e["data"].get("result") == "pass":
                    evidence[e["stage"]] = e["data"].get("evidence") or []
    except (OSError, ValueError) as exc:
        raise TextPackageError(f"the run's ledger cannot be read: {exc}") from exc
    return ends, evidence


def verify_revision(run: Path, revision: str, run_config_sha256: str, hub_config_sha256: str | None) -> str:
    """How the supplied revision is tied to the weights the run measured. The run's snapshot directory name (stage 0 records it) is a Hugging Face revision when it is 40 hex
    digits, and then it must be the supplied one. A run from another directory has no revision of its own: the supplied revision is accepted only with the sha256 of the model
    repository's config.json at that revision (`hub_config_sha256`), which must be the configuration the run used."""
    try:
        snapshot = _json(run / "stages" / "0" / "evidence" / "triage.json", "stage 0's triage evidence").get("snapshot")
    except TextPackageError:
        snapshot = None
    if isinstance(snapshot, str) and REVISION.fullmatch(snapshot):
        if snapshot != revision:
            raise TextPackageError(f"--revision {revision[:12]} is not the revision of the snapshot the run used ({snapshot[:12]}); the card and the download instructions would name "
                                   "other weights than the measured ones")
        return "the run's snapshot directory name"
    if not hub_config_sha256:
        raise TextPackageError("the run's snapshot directory is not named for a revision, so --revision cannot be verified; give --config-sha256-at-revision (the sha256 of the model "
                               "repository's config.json at that revision)")
    if hub_config_sha256 != run_config_sha256:
        raise TextPackageError("the configuration at the supplied revision is not the configuration the run used (the sha256 values differ)")
    return "the hash of the configuration at that revision"


def stage_package(run_dir, *, rtm, namespace: str, revision: str, license_file, context_file=None, out, repo_root=None, weights_terms: str,
                  config_sha256_at_revision: str | None = None) -> dict:
    """Stage the package in `out/<package name>/`. Order of refusals: the directory exists and is not empty; the revision is not a commit hash; the run has not passed
    stages 1, 2 and 6; the license file is not readable. Nothing is written before every input has been read."""
    run = Path(run_dir)
    root = Path(repo_root) if repo_root else rt.REPO
    pkg = Path(out) / rtm.package_name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", namespace or ""):
        raise TextPackageError("--namespace is the Hugging Face user or organisation the package would be published under, for example my-user")
    if not REVISION.fullmatch(revision or ""):
        raise TextPackageError("--revision is the 40-character commit hash of the model repository the run used")
    if pkg.exists() or pkg.is_symlink():
        if not pkg.is_dir() or pkg.is_symlink() or any(pkg.iterdir()):
            raise TextPackageError(f"{pkg} exists and is not an empty directory; a package is never overwritten in place")
    ref, res, bench = (_json(run / "stages" / str(n) / name, name) for n, name in ((1, rt.REFERENCE_FILE), (2, rt.RESULT_FILE), (6, rt.BENCH_FILE)))
    ends, evidence = ledger_stage_ends(run)
    for n, name in ((1, rt.REFERENCE_FILE), (2, rt.RESULT_FILE), (6, rt.BENCH_FILE)):
        if ends.get(n) != "pass":
            raise TextPackageError(f"stage {n} of the run did not pass (its ledger result is {ends.get(n)!r}); nothing is packaged from a run that has not passed")
        recorded = {e.get("path"): e.get("sha256") for e in evidence.get(n, []) if isinstance(e, dict)}
        rel = f"stages/{n}/{name}"
        if recorded.get(rel) != sha((run / rel).read_bytes()):
            raise TextPackageError(f"{rel} is not the file stage {n}'s gate accepted (its sha256 differs from the one in the ledger's passing stage_end); it was changed after the gate passed")
    recorded_scripts = _recorded_scripts(run)
    for name in rtm.staged_scripts():
        if recorded_scripts.get(name) != sha((rtm.scripts_dir / name).read_bytes()):
            raise TextPackageError(f"{name} is not the script the run used (its sha256 differs from the one in the run's text runtime selection); a package must carry the code that was measured")
    verified_by = verify_revision(run, revision, ref["config_sha256"], config_sha256_at_revision)
    if res.get("runtime") != rtm.name or bench.get("runtime") != rtm.name or ref.get("runtime") != rtm.name:
        raise TextPackageError("the run's result files are for another runtime")
    default = res["variants"][res["default_variant"]]
    if not default.get("ok"):
        raise TextPackageError("the default variant did not meet the stage 2 bar")
    license_bytes = Path(license_file).read_bytes() if license_file else b""
    if b"Apache License" not in license_bytes or b"Version 2.0" not in license_bytes:
        raise TextPackageError("--license-file must be the Apache License 2.0 text the model repository's card names")
    context = read_context(context_file)
    sources = {n: (rtm.scripts_dir / n).read_bytes() for n in rtm.shipped_files}
    mods = orchard_closure({n: b.decode("utf-8") for n, b in sources.items()}, root)
    own_license = (root / "LICENSE").read_bytes() if (root / "LICENSE").is_file() else b""
    files = []
    w = lambda rel, data, mode=0o644: files.append(ap._write(pkg, rel, data, mode))     # noqa: E731
    pkg.mkdir(parents=True, exist_ok=True)
    w("LICENSE", license_bytes)
    w("NOTICE", f"{rtm.model_ids[0]} is released under the license named in its repository card ({rtm.license_id}). {weights_terms}\n"
                "This package adds scripts (Apache License 2.0, tt-orchard) and no weights.\n")
    w("LICENSE-runtime", own_license or b"Apache License 2.0 (tt-orchard)\n")
    w("tt-model.yaml", manifest_text(rtm, revision, namespace))
    w("run.sh", run_script(rtm, ref["config_sha256"], revision), 0o755)
    w("requirements.txt", requirements_text())
    w("docs/running.md", running_doc(rtm, revision, ref["config_sha256"]))
    for n, b in sources.items():
        w(f"runtime/scripts/{rtm.scripts_dir.name}/{n}", b)
    w("runtime/orchard/__init__.py", '"""A stand-in package so the copied scripts import their orchard modules from this directory."""\n')
    for m in mods:
        w(f"runtime/orchard/{m}.py", (root / "orchard" / f"{m}.py").read_bytes())
    for rel, name, n in zip(EVIDENCE, (rt.REFERENCE_FILE, rt.RESULT_FILE, rt.BENCH_FILE), (1, 2, 6)):
        w(rel, (run / "stages" / str(n) / name).read_bytes())
    if context:
        w(CONTEXT_EVIDENCE, Path(context_file).read_bytes())
    w("README.md", render_card(rtm, res, bench, context, revision=revision, namespace=namespace, license_name="Apache License 2.0", weights_terms=weights_terms))
    record = {"schema": SCHEMA, "model": rtm.model_ids[0], "revision": revision, "revision_verified_by": verified_by, "runtime": rtm.name, "namespace": namespace, "config_sha256": ref["config_sha256"],
              "license": {"id": rtm.license_id, "sha256": sha(license_bytes)}, "weights_terms": weights_terms, "files": sorted(files, key=lambda r: r["path"])}
    ap._write(pkg, RECORD_FILE, json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record


def publish_commands(rtm, namespace: str, pkg_path: str) -> str:
    """The commands that would publish the staged package, as text. Nothing here runs them."""
    return "\n".join([
        "# Commands that would publish the staged package. The builder never ran them. Each needs the operator's explicit go-ahead.",
        "# Read the card's License section first: the weights' terms are the model authors' to state, and the link in their card was not reachable when this was staged.",
        "# 1. Create a private repository and upload the package directory (the weights are not in it):",
        f"hf upload --repo-type model --private {namespace}/{rtm.package_name} {pkg_path} .",
        "# 2. After a pull from the private repository and a second boot check, make it public (a separate decision):",
        f"#    hf repos settings {namespace}/{rtm.package_name} --no-private", ""])


# ---- the gate ----------------------------------------------------------------------------------------------------------------------

def _table_rows(card: str) -> list[str]:
    out, on = [], False
    for ln in card.splitlines():
        if ln.startswith("## "):
            on = ln.strip() == NUMBERS_HEAD
        elif on and ln.startswith("|") and not ln.startswith("|---") and not ln.startswith("| Number |"):
            out.append(ln.strip())
    return out


def package_problems(pkg, *, rtm, hostname: str | None = None, home: str | None = None, validate_manifest=None) -> list[str]:
    """The gate, from the files on disk. Refuses weights, a link or a non-regular file, a host name, token or home path, a word the package must not carry, a card
    whose numbers differ from its evidence, a license that is not the recorded one, a commercial-use claim, a missing Not tested section, a launcher that does not
    pin the configuration, a file the record does not list or lists with another hash, and a runtime file that is not a copy of its source."""
    pkg = Path(pkg)
    problems = ap.unreadable_entries(pkg)
    if problems:
        return problems
    problems = ap.weights_problems(pkg) + [f"scrub: {h}" for h in ap.scrub_files(pkg, hostname=hostname, home=home)]
    for f in ap._tree(pkg):
        text = ap._text(f)
        if text is None:
            continue
        rel = f.relative_to(pkg).as_posix()
        for word in FORBIDDEN_WORDS:
            if word in text and rel != "package-record.json":
                problems.append(f"{rel}: contains {word!r}, which a package must not carry")
    card = ap._text(pkg / "README.md")
    if card is None:
        return problems + ["the package has no README.md card"]
    for name in ("LICENSE", "NOTICE", "LICENSE-runtime", "tt-model.yaml", "run.sh", "requirements.txt", "docs/running.md", *EVIDENCE):
        if not (pkg / name).is_file():
            problems.append(f"the package has no {name}")
    if problems and any("has no" in p for p in problems):
        return problems
    front = ap._front(card)
    if front.get("license") != rtm.license_id:
        problems.append(f"the card's license is {front.get('license')!r}; the adapter expects {rtm.license_id!r}")
    for head in (NUMBERS_HEAD, LICENSE_HEAD, NOT_TESTED_HEAD):
        if head not in card:
            problems.append(f"the card needs a {head!r} section")
    for pat in COMMERCIAL_CLAIMS:
        if re.search(pat, card, re.I):
            problems.append("the card makes a commercial-use claim the model's license statement does not support")
    prose = "\n".join(ln for ln in card.splitlines() if not ln.startswith("|"))
    if PERF_IN_PROSE.search(prose):
        problems.append("the card states a speed or a percentage outside its Numbers table")
    try:
        res, bench = (_json(pkg / EVIDENCE[i], EVIDENCE[i]) for i in (1, 2))
        ctx = read_context(pkg / CONTEXT_EVIDENCE) if (pkg / CONTEXT_EVIDENCE).is_file() else None
        want = [_row(r) for r in number_rows(res, bench, ctx)]
    except (TextPackageError, KeyError, TypeError) as exc:
        return problems + [f"the evidence cannot back the card: {exc}"]
    have = _table_rows(card)
    if have != want:
        problems.append("the card's Numbers table is not what its evidence files give: " + ("; ".join(
            f"missing or changed row {r[:60]!r}" for r in want if r not in have)[:300] or "extra rows"))
    run = ap._text(pkg / "run.sh") or ""
    rec = _json(pkg / RECORD_FILE, RECORD_FILE) if (pkg / RECORD_FILE).is_file() else None
    if rec is None:
        return problems + [f"the package has no {RECORD_FILE}"]
    if f'PINNED_CONFIG_SHA256="{rec.get("config_sha256")}"' not in run or "sha256sum" not in run:
        problems.append("run.sh must carry the recorded configuration sha256 and check it before it starts the server")
    if f'{rtm.model_dir_flag} "${rtm.model_dir_env}" {rtm.weights_flag} "${rtm.weights_env}"' not in run.split('"$@"')[-1] or '"$@"' not in run:
        problems.append("run.sh must pass the checked model directory and weights after the forwarded arguments, so they cannot be overridden")
    if f'revision: "{rec.get("revision")}"' not in (ap._text(pkg / "tt-model.yaml") or ""):
        problems.append("tt-model.yaml must pin the recorded model revision")
    if f"`{rec.get('revision')}`" not in card:
        problems.append("the card must name the recorded model revision")
    names = {re.split(r"[<>=!~ ;\[]", ln.strip(), maxsplit=1)[0].lower() for ln in (ap._text(pkg / "requirements.txt") or "").splitlines() if ln.strip() and not ln.startswith("#")}
    problems += [f"requirements.txt must name {d}" for d in HOST_REQUIREMENTS if d not in names]
    listed = {f["path"]: f["sha256"] for f in rec.get("files", [])}
    on_disk = {f.relative_to(pkg).as_posix() for f in ap._tree(pkg)} - {RECORD_FILE}
    for rel in sorted(on_disk - set(listed)):
        problems.append(f"{rel}: is in the package and not in {RECORD_FILE}")
    for rel, digest in listed.items():
        if rel not in on_disk:
            problems.append(f"{rel}: is in {RECORD_FILE} and not in the package")
        elif sha((pkg / rel).read_bytes()) != digest:
            problems.append(f"{rel}: differs from the hash in {RECORD_FILE}")
    if sha((pkg / "LICENSE").read_bytes()) != (rec.get("license") or {}).get("sha256"):
        problems.append("LICENSE is not the license text the record names")
    if validate_manifest is not None:
        problems += [f"tt-model.yaml: {p}" for p in validate_manifest(pkg / "tt-model.yaml")]
    return problems


# ---- the command ----------------------------------------------------------------------------------------------------------------------

def main(args, *, out=None, hostname=None) -> int:
    """`tt-orchard text-package RUN_DIR --namespace ORG --revision SHA --license-file FILE --out DIR [--context FILE] [--check]`. Exit 0 when the package passes
    the gate, 1 when it does not, 2 when the run cannot be packaged."""
    import sys
    out = out or sys.stdout
    say = lambda text="": print(text, file=out, flush=True)      # noqa: E731
    run = Path(args.run_dir)
    sel = rt.selection_from_run_dir(run)
    rtm = rt.get(sel[0]) if sel else None
    if rtm is None:
        say("refused: the run holds no text runtime selection")
        return 2
    pkg = Path(args.out) / rtm.package_name
    try:
        if not args.check:
            record = stage_package(run, rtm=rtm, namespace=args.namespace, revision=args.revision, license_file=args.license_file, context_file=args.context,
                                   out=args.out, weights_terms=args.weights_terms,
                                   config_sha256_at_revision=getattr(args, "config_sha256_at_revision", None))
            say(f"staged {len(record['files'])} files in {pkg}")
        problems = package_problems(pkg, rtm=rtm, hostname=hostname, validate_manifest=ap.kernel_manifest_problems)
    except (TextPackageError, KeyError, OSError) as exc:
        say(f"refused: {exc}")
        return 2
    text = publish_commands(rtm, args.namespace, str(pkg))
    safeio.atomic_write(Path(args.out) / "TEXT_PUBLISH_COMMANDS.txt", text, mode=0o644)
    if problems:
        say("the package does not pass the gate:")
        for p in problems:
            say(f"  - {p}")
        return 1
    say(f"the package passes the gate. Nothing was uploaded or published; the commands that would publish it are in {Path(args.out) / 'TEXT_PUBLISH_COMMANDS.txt'} (text only).")
    return 0
