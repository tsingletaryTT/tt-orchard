# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard inventory verify-candidates`: run each candidate ttnn op against a torch reference on the ttsim simulator.

The op inventory (orchard/opinventory.py) proposes ttnn names for aten ops that have no table row (`candidate`) and
recognises groups of ops that one ttnn op does at once (`pattern`). Both are guesses. This command tests them without a
chip: it starts the operator's simulator image, runs the script orchard/skills/ops-inventory-templates/verify_in_sim.py
there, and reads back one result per candidate.

What a result says, and what it does not:

    verified          the ttnn op ran on the simulator for ONE small shape (rows and columns up to 64, tile aligned,
                      from the traced shape), one seed, and matched a torch reference within the bars below. It is not a
                      run on a chip, it used the image's ttnn wheel (named in the file) and not the host's ttnn, and it
                      says nothing about the shapes, dtypes or layouts the model uses. It does not make the op `mapped`.
    candidate-failed  the op ran and the numbers were outside the bars, or the call raised. The reason is kept.
    not-tested        this repository has no torch reference for the pair; the candidate stays a candidate.
    suspect           a control misbehaved (the right op did not pass, or a wrong op did), so no result of this run is
                      believed. The controls run in every call.

Bars (the style of orchard/skills/kernel-loop-templates/kernel_check.py, hardware column): PCC at least 0.99, relative
L2 error at most 0.05, slope of the output on the reference between 0.95 and 1.05 (PCC alone passes an output that is the
wrong size). Integer and mask ops must be equal.

Output: `verify-candidates.json` (every result), `verify-candidates.md` (the report and the table rows a person can paste
into orchard/data/aten_to_ttnn.json) and `verified-mappings.json` (the entries `opinventory.status_of` reads as the
separate source `verified`). `--record` also merges the entries into orchard/data/verified_mappings.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from orchard import opinventory

SCRIPT = Path(__file__).resolve().parent / "skills" / "ops-inventory-templates" / "verify_in_sim.py"
DEFAULT_IMAGE = "ttsim-blackhole-space:local"
SIM_SO = "/opt/sim/bh/libttsim_bh.so"
MARK = "ORCHARD_VERIFY "
SHAPE_CAP = 64
TILE = 32
DEFAULT_MAX_OPS = 40
DEFAULT_TIMEOUT_S = 900
CONTAINER_ENV = {"TT_METAL_SIMULATOR": SIM_SO, "TT_METAL_ARCH_NAME": "blackhole", "TT_METAL_SLOW_DISPATCH_MODE": "1",
                 "TT_METAL_DISABLE_SFPLOADMACRO": "1"}
LABEL = ("verified on the ttsim simulator with one small shape and one seed, with the image's ttnn wheel; "
         "not run on hardware and not a claim about this model's shapes, dtypes or layouts")
BARS = {"pcc": 0.99, "rel_l2": 0.05, "slope": [0.95, 1.05]}

# Two ops that must be told apart in every run. If the first does not pass, or either of the others does, the run is
# not believed: an instrument that cannot fail says nothing when it passes.
CONTROLS = [
    {"id": "control:right-op", "kind": "op", "aten": "aten.exp2.default", "ttnn": "exp2", "rows": 32, "cols": 64,
     "expect": "pass"},
    {"id": "control:wrong-op", "kind": "op", "aten": "aten.exp2.default", "ttnn": "exp", "as": ["exp2", "exp2"],
     "rows": 32, "cols": 64, "expect": "fail"},
    {"id": "control:wrong-integer-op", "kind": "op", "aten": "aten.__and__.Tensor", "ttnn": "bitwise_or",
     "as": ["__and__", "bitwise_and"], "dtype": "torch.int32", "rows": 32, "cols": 64, "expect": "fail"},
]


def small_shape(shape) -> tuple:
    """(rows, cols) for a test tensor: the last two traced dimensions, each capped at 64 and rounded up to a multiple of
    32. A missing or odd shape gives 32 by 64."""
    dims = [d for d in (shape or []) if isinstance(d, int) and not isinstance(d, bool) and d > 0]

    def fit(d):
        d = min(d, SHAPE_CAP)
        return max(TILE, -(-d // TILE) * TILE)

    if len(dims) >= 2:
        return fit(dims[-2]), fit(dims[-1])
    if len(dims) == 1:
        return TILE, fit(dims[0])
    return TILE, 2 * TILE


def select_cases(inv: dict, max_ops: int = DEFAULT_MAX_OPS):
    """(cases, left_out): one case per candidate pair of a `candidate` op and one per recognised group, capped at
    `max_ops`. `left_out` counts the pairs over the cap."""
    cases = []
    for r in inv.get("ops") or []:
        if not isinstance(r, dict) or r.get("status") != "candidate" or not isinstance(r.get("op"), str):
            continue
        sample = r.get("sample") if isinstance(r.get("sample"), dict) else {}
        shapes = sample.get("output_shapes")
        rows, cols = small_shape(shapes[0] if isinstance(shapes, list) and shapes else None)
        cands = r.get("candidates") if isinstance(r.get("candidates"), list) else []
        for c in cands:
            if not isinstance(c, dict) or not isinstance(c.get("ttnn"), str):
                continue
            cases.append({"id": f"{r['op']}->{c['ttnn']}", "kind": "op", "aten": r["op"], "ttnn": c["ttnn"],
                          "why": c.get("why"), "rows": rows, "cols": cols, "dtype": sample.get("dtype")})
    pats = inv.get("patterns") if isinstance(inv.get("patterns"), list) else []
    for f in pats:
        if (not isinstance(f, dict) or f.get("status") != "pattern" or not isinstance(f.get("pattern"), str)
                or not isinstance(f.get("ttnn"), str)):
            continue
        rows, cols = small_shape(f.get("shape") if isinstance(f.get("shape"), list) else None)
        cases.append({"id": f"pattern:{f['pattern']}", "kind": "pattern", "pattern": f["pattern"], "ttnn": f["ttnn"],
                      "rows": rows, "cols": cols, "const": [f["eps"]] if isinstance(f.get("eps"), float) and 0 < f["eps"] <= 1e-2 else []})
    return cases[:max_ops], max(0, len(cases) - max_ops)


def container_argv(image, workdir, name):
    """The `docker run` that starts the script. No network, the work directory read-only, no chip device."""
    argv = ["docker", "run", "--rm", "--name", name, "--network", "none", "--entrypoint", "/opt/venv/bin/python3"]
    for k, v in CONTAINER_ENV.items():
        argv += ["-e", f"{k}={v}"]
    return argv + ["-v", f"{workdir}:/work:ro", image, "/work/verify_in_sim.py", "/work/cases.json"]


def parse(stdout: str) -> dict:
    """The result object on the last `ORCHARD_VERIFY ` line."""
    found = None
    for line in stdout.splitlines():
        i = line.find(MARK)
        if i >= 0:
            found = line[i + len(MARK):]
    if found is None:
        raise ValueError("the simulator run printed no ORCHARD_VERIFY line")
    d = json.loads(found)
    if not isinstance(d, dict) or not isinstance(d.get("results"), list):
        raise ValueError("the ORCHARD_VERIFY line has no results list")
    return d


def check_controls(results) -> list:
    """The reasons the controls were not met; [] when the right op passed and both wrong ops failed."""
    by_id = {r.get("id"): r for r in results if isinstance(r, dict)}
    bad = []
    for c in CONTROLS:
        got = (by_id.get(c["id"]) or {}).get("outcome")
        if c["expect"] == "pass" and got != "pass":
            bad.append(f"{c['id']}: expected pass, got {got}")
        if c["expect"] == "fail" and got != "fail":
            bad.append(f"{c['id']}: expected fail, got {got}")
    return bad


def classify(result: dict) -> str:
    return {"pass": "verified", "fail": "candidate-failed", "error": "candidate-failed",
            "no_reference": "not-tested"}.get(result.get("outcome"), "candidate-failed")


def entries_from(cases, run: dict, *, image, date=None) -> dict:
    """{"aten op" or "pattern:<kind>": entry} for each op or group with at least one passing candidate. The first passing
    candidate in the order they were proposed is `ttnn`; the others that passed are `also_passed`."""
    by_id = {r["id"]: r for r in run["results"] if isinstance(r, dict) and "id" in r}
    out: dict = {}
    for case in cases:
        r = by_id.get(case["id"])
        if not r or r.get("outcome") != "pass":
            continue
        key = case["aten"] if case["kind"] == "op" else "pattern:" + case["pattern"]
        if key in out:
            out[key]["also_passed"].append(case["ttnn"])
            continue
        out[key] = {"ttnn": case["ttnn"], "also_passed": [], "compared": r.get("compared"), "pcc": r.get("pcc"),
                    "rel_l2": r.get("rel_l2"), "slope": r.get("slope"), "shape": [case["rows"], case["cols"]],
                    "dtype": case.get("dtype"), "why": case.get("why"), "image": image, "ttnn_wheel": run.get("ttnn_version"),
                    "torch": run.get("torch_version"), "date": date, "label": LABEL}
    return out


def summarize(cases, run, controls_bad, left_out) -> dict:
    """Rows for the report: [{id, status, reason, ...numbers}] and counts."""
    by_id = {r["id"]: r for r in run["results"] if isinstance(r, dict) and "id" in r}
    rows = []
    for case in cases:
        r = by_id.get(case["id"], {"outcome": "error", "reason": "no result came back for this case"})
        status = "suspect" if controls_bad else classify(r)
        rows.append({"id": case["id"], "kind": case["kind"], "aten": case.get("aten"), "ttnn": case["ttnn"],
                     "pattern": case.get("pattern"), "status": status, "outcome": r.get("outcome"),
                     "reason": r.get("reason", ""), "pcc": r.get("pcc"), "rel_l2": r.get("rel_l2"),
                     "slope": r.get("slope"), "compared": r.get("compared"), "seconds": r.get("seconds"),
                     "shape": [case["rows"], case["cols"]]})
    counts: dict = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"rows": rows, "counts": counts, "controls_bad": controls_bad, "left_out": left_out}


def render_md(summary: dict, entries: dict, run: dict, image: str) -> str:
    lines = ["# Candidate verification (ttsim simulator, no chip)", "",
             f"Image `{image}`: ttnn wheel `{run.get('ttnn_version')}`, torch `{run.get('torch_version')}`.", "",
             LABEL.capitalize() + ".", "",
             f"Bars: PCC at least {BARS['pcc']}, relative L2 at most {BARS['rel_l2']}, slope between {BARS['slope'][0]} "
             f"and {BARS['slope'][1]}; integer and mask ops must be equal.", ""]
    if summary["controls_bad"]:
        lines += ["**No result of this run is believed.** A control behaved wrongly: " + "; ".join(summary["controls_bad"]),
                  ""]
    else:
        lines += ["Controls: the right op passed and both wrong ops failed.", ""]
    lines += ["| status | " + " | ".join(["case", "PCC", "rel L2", "slope", "shape", "note"]) + " |",
              "| --- | --- | --- | --- | --- | --- | --- |"]

    def num(x):
        return "" if x is None else f"{x:.6g}"

    for r in summary["rows"]:
        lines.append(f"| {r['status']} | `{r['id']}` | {num(r['pcc'])} | {num(r['rel_l2'])} | {num(r['slope'])} | "
                     f"{r['shape'][0]}x{r['shape'][1]} | {str(r['reason'])[:120].replace('|', '/')} |")
    if summary["left_out"]:
        lines += ["", f"{summary['left_out']} candidate pairs were over the cap and were not run."]
    ops = {k: v for k, v in entries.items() if not k.startswith("pattern:")}
    if ops and not summary["controls_bad"]:
        lines += ["", "Rows a person can paste into orchard/data/aten_to_ttnn.json once they accept the simulator check "
                      "as enough (each is `mapped` only after that edit):", "", "```"]
        lines += [f'  "{k}": {{"ttnn": "{v["ttnn"]}"}},' for k, v in sorted(ops.items())]
        lines += ["```"]
    pats = {k: v for k, v in entries.items() if k.startswith("pattern:")}
    if pats and not summary["controls_bad"]:
        lines += ["", "Groups that matched: " + ", ".join(f"{k[8:]} -> {v['ttnn']}" for k, v in sorted(pats.items())) + ".",
                  "A group is a recognition of a composite; its member ops keep their own statuses."]
    return "\n".join(lines) + "\n"


def write_outputs(out_dir, cases, run, summary, entries, image) -> list:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify-candidates.json").write_text(json.dumps({"label": LABEL, "image": image, "run": run,
                                                            "summary": summary}, indent=2, sort_keys=True) + "\n")
    (out / "verify-candidates.md").write_text(render_md(summary, entries, run, image))
    keep = {} if summary["controls_bad"] else entries
    (out / "verified-mappings.json").write_text(json.dumps({"schema": 1, "label": LABEL, "entries": keep},
                                                           indent=2, sort_keys=True) + "\n")
    return [out / "verify-candidates.json", out / "verify-candidates.md", out / "verified-mappings.json"]


def merge_record(entries: dict, path=None) -> Path:
    """Merge `entries` into the package's verified_mappings.json (newer entries replace older ones of the same key)."""
    p = Path(path) if path else opinventory.DATA_DIR / "verified_mappings.json"
    try:
        cur = json.loads(p.read_text(encoding="utf-8"))
        have = cur.get("entries") if isinstance(cur, dict) and isinstance(cur.get("entries"), dict) else {}
    except (OSError, ValueError):
        have = {}
    have.update(entries)
    tmp = p.with_name(p.name + ".tmp")
    try:
        tmp.write_text(json.dumps({"schema": 1, "label": LABEL, "entries": have}, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)
    return p


def _real_run(argv, timeout, name=None):
    """Run in its own session. On a timeout the group is stopped and the named container removed. Returns (rc, out)."""
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    except OSError as exc:
        return 127, f"{type(exc).__name__}: {exc}"
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except OSError:
                pass
            try:
                out, _ = proc.communicate(timeout=10)
                break
            except subprocess.TimeoutExpired:
                continue
        else:
            out = ""
        if name:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
        return 124, f"timed out after {timeout} s and was stopped\n{(out or '')[-500:]}"
    except BaseException:
        if name:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
        raise
    return proc.returncode, out


def run_simulator(cases, *, image=DEFAULT_IMAGE, timeout=DEFAULT_TIMEOUT_S, run=_real_run, workroot=None) -> dict:
    """Write the cases and the script to a scratch directory, run the container, return the parsed result.
    Raises RuntimeError with the reason when the container could not run or printed no result."""
    work = Path(tempfile.mkdtemp(prefix="orchard-verify-", dir=workroot))
    name = f"orchard-verify-{os.getpid()}-{int(time.time())}"
    try:
        shutil.copy(SCRIPT, work / "verify_in_sim.py")
        (work / "cases.json").write_text(json.dumps({"cases": cases}))
        os.chmod(work, 0o755)
        rc, out = run(container_argv(image, work, name), timeout, name)
        if rc != 0:
            raise RuntimeError(f"the simulator run failed (exit {rc}): {out.strip()[-400:]}")
        return parse(out)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def load_inventory(path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "ops-inventory.json"
    inv = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(inv, dict) or not isinstance(inv.get("ops"), list):
        raise ValueError(f"{p}: not an ops-inventory.json")
    return inv


def main(inventory, *, out_dir, image=DEFAULT_IMAGE, max_ops=DEFAULT_MAX_OPS, timeout=DEFAULT_TIMEOUT_S, record=False,
         run=_real_run, say=print, env=None, record_path=None) -> int:
    """Exit 0 when the run finished and its controls held (whatever the candidates did), 2 when it could not run or
    the controls failed, 3 when there was nothing to test."""
    env = os.environ if env is None else env
    try:
        inv = load_inventory(inventory)
    except (OSError, ValueError) as exc:
        say(f"cannot read the inventory {inventory}: {exc}")
        return 2
    cases, left_out = select_cases(inv, max_ops)
    if not cases:
        say("no candidate or pattern rows in the inventory: nothing to verify")
        return 3
    say(f"verifying {len(cases)} candidate pairs on the ttsim simulator (image {image}); no chip is used")
    try:
        got = run_simulator(cases + CONTROLS, image=image, timeout=timeout, run=run)
    except (RuntimeError, ValueError, OSError) as exc:
        say(str(exc))
        return 2
    controls_bad = check_controls(got["results"])
    entries = entries_from(cases, got, image=image, date=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    summary = summarize(cases, got, controls_bad, left_out)
    paths = write_outputs(out_dir, cases, got, summary, entries, image)
    say("wrote " + ", ".join(str(p) for p in paths))
    say("results: " + ", ".join(f"{k} {v}" for k, v in sorted(summary["counts"].items())))
    if controls_bad:
        say("the controls failed, so no result is believed: " + "; ".join(controls_bad))
        return 2
    if record:
        say(f"recorded {len(entries)} verified entries in {merge_record(entries, record_path)}")
    return 0


def cli(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m orchard.opverify", description=__doc__.split("\n\n")[0])
    ap.add_argument("--inventory", required=True, help="an ops-inventory.json or the directory that holds it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--max-ops", type=int, default=DEFAULT_MAX_OPS)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args(argv)
    return main(a.inventory, out_dir=a.out, image=a.image, max_ops=a.max_ops, timeout=a.timeout, record=a.record)


if __name__ == "__main__":
    sys.exit(cli())
