#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Serve a model through tt-transformers as it is, and measure it against the stage 1 CPU reference (stage 2 of the stock-runtime path).

Two commands, both reading `stock_config.json` in this directory (the supervisor writes it, orchard/stock_runtime.py, draft_config):

    python3 stock_check.py prepare   no chip. Copies the host tt-metal tree's `models/` into the scratch directory (once), builds a model
                                     directory named after the tt-transformers model, makes a reference file when tt-transformers has none,
                                     and writes hw_test.json and handoff.json.
    python3 stock_check.py run       on the leased board. Runs tt-transformers' own token-accuracy test and a short free run from the scratch
                                     copy, then writes evidence/stock-check.json, the log of each run and evidence/scratch.patch.

The host tt-metal tree is never written. The scratch copy is a plain directory copy of `models/`, `conftest.py` and `pytest.ini`; the
runtime is the host build's `ttnn`, found on the interpreter's path. The numbers in stock-check.json are parsed from tt-transformers' own log
lines; the gate (orchard/stages.py, gate_stock_runtime) judges them again against the bars in orchard/defaults.py.

stock_config.json keys: run_dir, model_id, model_name, model_snapshot, tt_metal_dir, scratch_dir, mesh_device, chips, optimizations,
tt_cache, deadline_s; optional python_path_extra (directories put after the scratch copy on PYTHONPATH) and import_check (modules `prepare`
must be able to import, default numpy, cv2 and torch). Exit codes: 0 done, 2 the config or an input does not fit (the message names it), 3 a run step failed (the log says
which), 6 a model was fetched from the Hub."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS = ("run_dir", "model_id", "model_name", "model_snapshot", "tt_metal_dir", "scratch_dir", "mesh_device", "chips", "optimizations",
        "tt_cache", "deadline_s")
# The root conftest.py imports `tests.scripts.common`, so the package that holds it is copied too (about 700 KB).
COPIED = ("models", "conftest.py", "pytest.ini", "tests/__init__.py", "tests/scripts")
OPTIONAL_KEYS = {"python_path_extra": [], "import_check": ["numpy", "cv2", "torch"]}
MARKER = ".orchard-scratch.json"
DEMO = "models/tt_transformers/demo/simple_text_demo.py"
REFERENCE_DIR = "models/tt_transformers/tests/reference_outputs"
ACCURACY_TEST = "accuracy and ci-token-matching"
FREE_TEST = "accuracy and batch-1"
ACCURACY = re.compile(r"Top1 Accuracy:\s*([0-9.]+)%,\s*Top5 Accuracy:\s*([0-9.]+)%")
ITERATION = re.compile(r"Iteration (\d+):")
SPEED = re.compile(r"Average speed: [0-9.]+ms @ ([0-9.]+) tok/s/user")
TTFT = re.compile(r"Average Time to First Token \(TTFT\): ([0-9.]+)ms")
USER_PROMPT = re.compile(r"==USER 0 - PROMPT\n(.*?)\s*\n==USER 0 - OUTPUT\n", re.S)
# The output ends at the next marker, at the next log record (loguru starts each with a date) or at the end of the log.
USER_OUTPUT = re.compile(r"==USER 0 - OUTPUT\n(.*?)(?:\n==|\n\d{4}-\d\d-\d\d[ T]|\Z)", re.S)
PATCH_LIMIT = 2_000_000


def fail(code: int, message: str) -> None:
    print(f"stock_check: {message}", file=sys.stderr)
    sys.exit(code)


def say(message: str) -> None:
    print(f"stock_check: {message}", flush=True)


def template_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_config() -> dict:
    try:
        cfg = json.loads((HERE / "stock_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(2, f"stock_config.json is not readable: {exc}")
    for key in KEYS:
        if key not in cfg:
            fail(2, f"stock_config.json needs {key}")
    for key in ("run_dir", "model_snapshot", "tt_metal_dir", "scratch_dir", "tt_cache"):
        if not Path(cfg[key]).is_absolute():
            fail(2, f"{key} must be an absolute path")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", str(cfg["model_name"])):
        fail(2, "model_name must be one path part")
    for key, default in OPTIONAL_KEYS.items():
        cfg.setdefault(key, default)
        if not isinstance(cfg[key], list) or not all(isinstance(v, str) for v in cfg[key]):
            fail(2, f"{key} must be a list of strings")
    if any(not Path(d).is_absolute() for d in cfg["python_path_extra"]):
        fail(2, "python_path_extra must hold absolute paths")
    if any(not re.fullmatch(r"[A-Za-z0-9_]+", m) for m in cfg["import_check"]):
        fail(2, "import_check must hold module names")
    return cfg


def python_path(cfg: dict, scratch: Path) -> str:
    """PYTHONPATH for every child: the scratch copy first (so `models` is the copy), then any directories the config names."""
    return os.pathsep.join([str(scratch), *cfg["python_path_extra"]])


def check_imports(cfg: dict, scratch: Path) -> None:
    """The interpreter has to import what the root conftest and the demo import. On 2026-10-26 the shared environment had numpy 2.5.3
    while ttnn needs numpy below 2 and cv2 was built for it, so the demo died at its first import. This is checked in `prepare`, where
    it costs seconds and needs no chip. `torch` and `numpy` import no device code; ttnn is never imported here."""
    for module in cfg["import_check"]:
        done = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": python_path(cfg, scratch)})
        if done.returncode != 0:
            last = (done.stderr.strip().splitlines() or ["no message"])[-1]
            fail(2, f"the interpreter cannot import {module}: {last}. Fix the environment (or name a directory with a working copy in "
                    "python_path_extra); the model was not tried")


def tree_facts(tt_metal_dir: str) -> dict:
    """The host tree's tag and whether it has uncommitted changes (read only). None when git cannot say."""
    def git(*args):
        try:
            done = subprocess.run(["git", "-C", tt_metal_dir, *args], capture_output=True, text=True, timeout=60,
                                  env={"PATH": "/usr/bin:/bin", "GIT_OPTIONAL_LOCKS": "0"})
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"tag": git("describe", "--tags", "--always"), "dirty": None if status is None else bool(status)}


# ---- prepare ----------------------------------------------------------------------------------------------------

def make_scratch(cfg: dict) -> Path:
    scratch, host = Path(cfg["scratch_dir"]), Path(cfg["tt_metal_dir"])
    marker = scratch / MARKER
    if marker.is_file():
        say(f"the scratch copy exists ({scratch.name}); keeping it, with any change a stage made")
        return scratch
    if scratch.exists() and any(scratch.iterdir()):
        fail(2, f"{scratch} exists and has no {MARKER}; it is not a scratch copy this script made")
    scratch.mkdir(parents=True, exist_ok=True)
    for name in COPIED:
        src = host / name
        if not src.exists():
            fail(2, f"{src} does not exist: tt_metal_dir is not a tt-metal tree")
        dst = scratch / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)
    marker.write_text(json.dumps({"copied": list(COPIED), "tt_metal": tree_facts(cfg["tt_metal_dir"])}, indent=2) + "\n")
    say(f"copied {', '.join(COPIED)} from the host tree into {scratch.name}")
    return scratch


def make_model_dir(cfg: dict, scratch: Path) -> Path:
    """A directory named after the tt-transformers model that holds the snapshot's files. tt-transformers takes the model's name from
    the last part of HF_MODEL, so the name has to be the name its tables use; `_name_or_path` in config.json would override it, so the
    config is a copy without that key. Every other file is a link to the snapshot."""
    snap, target = Path(cfg["model_snapshot"]), scratch / "hf-models" / cfg["model_name"]
    if not (snap / "config.json").is_file():
        fail(2, f"{snap} has no config.json")
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(snap.iterdir()):
        link = target / item.name
        if item.name == "config.json":
            data = json.loads(item.read_text(encoding="utf-8"))
            data.pop("_name_or_path", None)
            link.write_text(json.dumps(data, indent=2) + "\n")
        elif not link.exists() and not link.is_symlink() and item.is_file():
            link.symlink_to(item.resolve())
    return target


def snapshot_id(cfg: dict) -> str:
    """The name of the snapshot directory, which is the revision in a Hugging Face cache."""
    return Path(cfg["model_snapshot"]).resolve().name


def reference_identity(cfg: dict) -> dict:
    return {"model_id": cfg["model_id"].split("@", 1)[0], "snapshot": snapshot_id(cfg)}


def reference_files(cfg: dict, scratch: Path) -> tuple[Path, Path]:
    ref = scratch / REFERENCE_DIR / f"{cfg['model_name']}.refpt"
    return ref, ref.with_name(ref.name + ".orchard.json")


def ensure_reference(cfg: dict, scratch: Path, model_dir: Path) -> str:
    """tt-transformers' accuracy test reads `<reference_outputs>/<model name>.refpt`: reference tokens and the top-5 predictions of the
    Hugging Face model on the same text. The file name is the tt-transformers model's name, which any organisation's copy shares, so a
    file the host tree ships for that name is the reference of the model the tree lists and not of this run's snapshot (a fine-tune matched
    through its base would be compared with its base's predictions). The reference is therefore always made here, on the CPU, from the run's
    own snapshot with tt-transformers' generator script, and a record beside it (`<file>.orchard.json`: model id and snapshot name) lets a
    second `prepare` reuse it. The host tree's file is never touched; the scratch copy of it is replaced."""
    ref, record = reference_files(cfg, scratch)
    want = reference_identity(cfg)
    try:
        have = json.loads(record.read_text(encoding="utf-8")) if ref.is_file() and record.is_file() else None
    except (OSError, ValueError):
        have = None
    if have == want:
        return f"reference file {ref.name} already made from this snapshot"
    say("making the reference from this model's snapshot on the CPU (this takes a while)")
    ref.unlink(missing_ok=True)
    record.unlink(missing_ok=True)
    script = scratch / "models/tt_transformers/tests/generate_reference_hf.py"
    env = {**os.environ, "PYTHONPATH": python_path(cfg, scratch), "HF_HUB_OFFLINE": "1"}
    done = subprocess.run([sys.executable, str(script), "--total_length", "1024", "--output_file", str(ref), "--model", str(model_dir)],
                          cwd=scratch, env=env, capture_output=True, text=True)
    (HERE / "log").mkdir(exist_ok=True)
    (HERE / "log" / "generate-reference.log").write_text(done.stdout + done.stderr)
    if done.returncode != 0 or not ref.is_file():
        fail(2, "the reference file could not be generated; see log/generate-reference.log")
    record.write_text(json.dumps(want) + "\n")
    return f"reference file {ref.name} generated"


def reference_evidence(cfg: dict, scratch: Path) -> dict:
    """The reference the run is about to use, read back from the record and the file (exit 2 when it is not this snapshot's)."""
    ref, record = reference_files(cfg, scratch)
    try:
        have = json.loads(record.read_text(encoding="utf-8"))
        digest = hashlib.sha256(ref.read_bytes()).hexdigest()
    except (OSError, ValueError):
        fail(2, "the reference file or its record is missing; run `prepare` first")
    if have != reference_identity(cfg):
        fail(2, f"the reference record {have!r} does not match this run's model and snapshot; run `prepare` again")
    return {"source": "generated", **have, "sha256": digest}


def prepare() -> None:
    cfg = load_config()
    check_imports(cfg, Path(cfg["scratch_dir"]))
    scratch = make_scratch(cfg)
    model_dir = make_model_dir(cfg, scratch)
    note = ensure_reference(cfg, scratch, model_dir)
    say(note)
    stage = HERE.name
    (HERE / "hw_test.json").write_text(json.dumps({"command": f"python3 stages/{stage}/stock_check.py run", "deadline_s": cfg["deadline_s"]},
                                                  indent=2) + "\n")
    (HERE / "handoff.json").write_text(json.dumps({
        "goal": f"serve {cfg['model_name']} through tt-transformers on {cfg['mesh_device']} and compare it with the CPU reference",
        "stage": int(stage) if stage.isdigit() else stage, "evidence": [f"stages/{stage}/stock_config.json"],
        "next_action": "read evidence/stock-check.json", "check_on_return": "the hardware test's exit code"}, indent=2) + "\n")
    say("wrote hw_test.json and handoff.json")


# ---- run --------------------------------------------------------------------------------------------------------

def pytest_run(cfg: dict, scratch: Path, model_dir: Path, test: str, log: Path, deadline: float) -> int:
    env = {k: v for k, v in os.environ.items()}
    env.update({"HF_MODEL": str(model_dir), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MESH_DEVICE": cfg["mesh_device"],
                "PYTHONPATH": python_path(cfg, scratch), "TT_CACHE_PATH": cfg["tt_cache"]})
    argv = [sys.executable, "-m", "pytest", DEMO, "-k", test, "-s", "-x", "-p", "no:cacheprovider"]
    say(f"running: pytest {DEMO} -k {test!r} (mesh {cfg['mesh_device']})")
    with open(log, "wb") as out:
        try:
            done = subprocess.run(argv, cwd=scratch, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=deadline)
        except subprocess.TimeoutExpired:
            return 124
    return done.returncode


def parse_accuracy(text: str):
    found = ACCURACY.findall(text)
    return (round(float(found[-1][0]) / 100, 6), round(float(found[-1][1]) / 100, 6)) if found else None


def parse_iterations(text: str):
    """Decode steps the run made: the highest `Iteration N:` line plus one, or None when the log has none."""
    found = ITERATION.findall(text)
    return max(int(n) for n in found) + 1 if found else None


def parse_free_run(text: str) -> dict:
    """The first user's prompt and output, the decode speed and the first-token time from the batch-1 demo's log."""
    prompt, out = USER_PROMPT.search(text), USER_OUTPUT.search(text)
    speed, ttft = SPEED.findall(text), TTFT.findall(text)
    return {"prompt": prompt.group(1).strip() if prompt else None, "text": out.group(1).strip() if out else "",
            "new_tokens": parse_iterations(text) or 0, "finish": "length",
            "decode_tokens_per_s": float(speed[-1]) if speed else None, "ttft_s": float(ttft[-1]) / 1000 if ttft else None}


def hub_fetches(model_dir: Path, log_text: str) -> list[str]:
    """Anything the run fetched from the Hub: the runs switch it off, so any fetch is a failure."""
    return sorted({m for m in re.findall(r"(?:Fetching \d+ files|snapshot_download)[^\n]*", log_text)})


def scratch_patch(cfg: dict, evidence: Path) -> str:
    """The change between the host tree's models/ and the scratch copy (what a stage changed), as a unified diff."""
    host, scratch = Path(cfg["tt_metal_dir"]) / "models", Path(cfg["scratch_dir"]) / "models"
    done = subprocess.run(["diff", "-ruN", "-x", "__pycache__", "-x", "*.pyc", "-x", "*.refpt", "-x", "*.refpt.orchard.json", str(host), str(scratch)],
                          capture_output=True, text=True)
    lines = [ln.split("\t", 1)[0] if ln.startswith(("--- ", "+++ ")) else ln for ln in done.stdout.splitlines()
             if not ln.startswith("diff ")]                         # no command line, no timestamps, no machine path
    text = "\n".join(lines).replace(str(host), "a/models").replace(str(scratch), "b/models")
    text = text + "\n" if text else ""
    if len(text) > PATCH_LIMIT:
        text = text[:PATCH_LIMIT] + "\n[cut: the patch is longer than the evidence limit]\n"
    (evidence / "scratch.patch").write_text(text)
    return "evidence/scratch.patch"


def run() -> None:
    cfg = load_config()
    scratch = Path(cfg["scratch_dir"])
    if not (scratch / MARKER).is_file():
        fail(2, "the scratch copy does not exist; run `prepare` first")
    model_dir = scratch / "hf-models" / cfg["model_name"]
    reference = reference_evidence(cfg, scratch)
    (HERE / "log").mkdir(exist_ok=True)
    evidence = HERE / "evidence"
    evidence.mkdir(exist_ok=True)
    started = time.time()
    deadline = float(cfg["deadline_s"])
    acc_log = HERE / "log" / "token-accuracy.log"
    rc = pytest_run(cfg, scratch, model_dir, ACCURACY_TEST, acc_log, deadline)
    acc_text = acc_log.read_text(errors="replace")
    accuracy = parse_accuracy(acc_text)
    if rc != 0 and accuracy is None:
        fail(3, f"the token-accuracy run exited {rc} and printed no accuracy; see log/token-accuracy.log")
    free_log = HERE / "log" / "free-run.log"
    rc2 = pytest_run(cfg, scratch, model_dir, FREE_TEST, free_log, max(60.0, deadline - (time.time() - started)))
    free_text = free_log.read_text(errors="replace")
    free = parse_free_run(free_text)
    fetched = hub_fetches(model_dir, acc_text + free_text)
    result = {
        "template_sha256": template_sha256(), "model_id": cfg["model_id"], "chips": cfg["chips"], "device_name": cfg["mesh_device"],
        "optimizations": cfg["optimizations"], "tt_metal": json.loads((scratch / MARKER).read_text())["tt_metal"],
        "teacher_forced": {"tokens": parse_iterations(acc_text), "top1": accuracy[0] if accuracy else None,
                           "top5": accuracy[1] if accuracy else None},
        "free_run": {k: free[k] for k in ("prompt", "text", "new_tokens", "finish")},
        "speed": {"decode_tokens_per_s": free["decode_tokens_per_s"], "ttft_s": free["ttft_s"]}, "hub_fetch": fetched,
        "reference": reference,
        "scratch_patch": scratch_patch(cfg, evidence), "exit_codes": {"token_accuracy": rc, "free_run": rc2},
        "seconds": round(time.time() - started, 1)}
    (evidence / "stock-check.json").write_text(json.dumps(result, indent=2) + "\n")
    say(f"wrote evidence/stock-check.json (top1 {result['teacher_forced']['top1']}, top5 {result['teacher_forced']['top5']})")
    if fetched:
        fail(6, "the run fetched from the Hub: " + "; ".join(fetched))
    if rc != 0 or rc2 != 0:
        fail(3, f"a run step exited non-zero (token accuracy {rc}, free run {rc2}); see log/")


def main(argv: list[str]) -> None:
    if len(argv) != 2 or argv[1] not in ("prepare", "run"):
        fail(2, "usage: stock_check.py prepare | run")
    {"prepare": prepare, "run": run}[argv[1]]()


if __name__ == "__main__":
    main(sys.argv)
