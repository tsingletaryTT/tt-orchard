#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The quality readout of the OpenCode v2 against orchard comparison (run log addendum 1).

    python3 scripts/oc_quality.py PHASE [--root DIR] [--no-heldout]

Reads ROOT/PHASE/timeline.jsonl (written by scripts/opencode_ab.py) and writes ROOT/PHASE/quality.jsonl (one record per attempt) and
ROOT/PHASE/quality.md (the tables). Per attempt: the unedited check's measures and their distance to the bars, the failure class and whether it
is a near miss, agent wall seconds and the load flag, how the work got done (writes of kernel.py, runs of kernel_check.py, tool calls by tool,
how it ended), and for passing kernels a static check against the cheat sheet in the attempt directory, lines of code, whether the kernel has the
structure of the reference example, and a held-out run.

The held-out run uses the probe's own `authoritative_check` with a temporary copy of the shipped templates. The check, the harness and
references.py in the copy are byte-identical to the shipped ones (their sha256 are compared and recorded); only `ops/<op>.json`, which is data,
carries different seeds. A run that needs any other change is not made.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from orchard import kernelprobe  # noqa: E402

HELD_OUT_SEEDS = [101, 102, 103]
BARS = {"pcc": 0.999, "rel_l2": 0.01, "slope": (0.98, 1.02)}
NEAR = {"pcc": 0.99, "rel_l2": 0.1, "slope": (0.8, 1.2)}                 # within 10 times of the bar (run log addendum 1)
STRUCTURAL = {"deadlock", "dataflow_step", "shape_mismatch", "import_error", "tt_lang_error", "forbidden_code", "no_kernel", "timeout",
              "harness_error"}
REFERENCE_STRUCTURE = ("ttl.operation", "ttl.compute", "ttl.datamovement", "ttl.make_dataflow_buffer_like", "ttl.copy", "wait", "reserve", "store")
METHOD_NAMES = {"wait": "DataflowBuffer.wait", "reserve": "DataflowBuffer.reserve", "store": "TensorBlock.store",
                "if_src": "PipeNet.if_src", "if_dst": "PipeNet.if_dst"}


def suite_windows() -> list:
    """The load windows (name, start, end) in UTC, from `suite-window.json` next to the results (written from the user journal, since systemd
    removes a finished unit). An end of None means the unit was still running when the file was written."""
    f = Path.home() / "orchard-runs" / "ocprobe" / "suite-window.json"
    if not f.is_file():
        return []
    return [(w["name"], w.get("start"), w.get("end")) for w in json.loads(f.read_text()).get("windows", [])]


def overlaps(rec: dict, windows: list) -> list:
    """The names of the windows the attempt overlapped."""
    out = []
    for name, start, end in windows:
        if start and rec["started_utc"] <= (end or "9999") and rec["ended_utc"] >= start:
            out.append(name)
    return out


def dotted(node) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def sheet_entries(sheet_path: Path) -> dict:
    d = json.loads(sheet_path.read_text())
    out = {}
    for sec in d["sections"]:
        for e in sec["entries"]:
            sig = e.get("signature") or ""
            m = re.search(r"\((.*)\)", sig, re.S)
            params = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", m.group(1))) if m else set()
            out[e["name"]] = {"status": e.get("status"), "params": params, "signature": sig}
    return out


def reference_method_names(adir: Path) -> set:
    names = set()
    ref = adir / "reference"
    for f in ref.glob("*.py") if ref.is_dir() else []:
        try:
            for node in ast.walk(ast.parse(f.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    names.add(node.func.attr)
        except SyntaxError:
            pass
    return names


def static_check(kernel: str, entries: dict, ref_methods: set) -> dict:
    tree = ast.parse(kernel)
    used = {}
    unlisted, bad_kwargs, in_reference_only = [], [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted(node.func)
        kw = [k.arg for k in node.keywords if k.arg]
        key = None
        if name and name.split(".")[0] in ("ttl", "sim"):
            key = name
        elif isinstance(node.func, ast.Attribute) and node.func.attr in METHOD_NAMES:
            key = METHOD_NAMES[node.func.attr]
        elif isinstance(node.func, ast.Attribute) and name and name.startswith("ttl"):
            key = name
        if key is None:
            if isinstance(node.func, ast.Attribute) and node.func.attr in ref_methods and node.func.attr not in ("wait", "reserve", "store"):
                in_reference_only.append(node.func.attr)
            continue
        used[key] = used.get(key, 0) + 1
        if key not in entries:
            unlisted.append(key)
        else:
            params = entries[key]["params"]
            extra = [k for k in kw if params and k not in params]
            if extra:
                bad_kwargs.append({"call": key, "keywords_not_in_signature": extra, "signature": entries[key]["signature"]})
    by_status = {}
    for k, n in used.items():
        st = entries[k]["status"] if k in entries else "not in the cheat sheet"
        by_status.setdefault(st, []).append(k)
    return {"calls": used, "by_status": {k: sorted(v) for k, v in by_status.items()}, "unlisted": sorted(set(unlisted)),
            "keyword_not_in_hardware_signature": bad_kwargs, "block_methods_seen_only_in_reference": sorted(set(in_reference_only))}


def structure(kernel: str) -> dict:
    tree = ast.parse(kernel)
    decorators, calls, methods = [], set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for d in node.decorator_list:
                n = dotted(d.func if isinstance(d, ast.Call) else d)
                if n:
                    decorators.append(n)
        if isinstance(node, ast.Call):
            n = dotted(node.func)
            if n:
                calls.add(n)
            if isinstance(node.func, ast.Attribute):
                methods.add(node.func.attr)
    have = {"ttl.operation": "ttl.operation" in decorators, "ttl.compute": "ttl.compute" in decorators,
            "ttl.datamovement": decorators.count("ttl.datamovement"), "ttl.make_dataflow_buffer_like": "ttl.make_dataflow_buffer_like" in calls,
            "ttl.copy": "ttl.copy" in calls, "wait": "wait" in methods, "reserve": "reserve" in methods, "store": "store" in methods}
    loc = len([l for l in kernel.splitlines() if l.strip() and not l.strip().startswith("#")])
    follows = (have["ttl.operation"] and have["ttl.compute"] and have["ttl.datamovement"] >= 2 and have["ttl.make_dataflow_buffer_like"]
               and have["ttl.copy"] and have["wait"] and have["reserve"] and have["store"])
    return {"lines_of_code": loc, "elements": have, "follows_reference_structure": bool(follows)}


def held_out(adir: Path, op: str, env, facts, scratch: Path) -> dict:
    scratch.mkdir(parents=True, exist_ok=True)
    tdir = kernelprobe.TEMPLATES
    tmp = scratch / "templates"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(tdir, tmp)
    spec_path = tmp / "ops" / f"{op}.json"
    spec = json.loads(spec_path.read_text())
    orig_seeds = spec.get("seeds")
    spec["seeds"] = HELD_OUT_SEEDS
    spec_path.write_text(json.dumps(spec, indent=2))
    same = {}
    for name in ("kernel_check.py", "kernel_harness.py", "references.py"):
        same[name] = hashlib.sha256((tdir / name).read_bytes()).hexdigest() == hashlib.sha256((tmp / name).read_bytes()).hexdigest()
    if not all(same.values()):
        return {"ran": False, "reason": "a shipped check file differs in the copy", "files_identical": same}
    vdir = scratch / "heldout-verdict"
    shutil.rmtree(vdir, ignore_errors=True)
    res = kernelprobe.authoritative_check(adir, op=op, env=env, tt_lang=facts, vdir=vdir, templates=tmp)
    sets = res.get("sets") if isinstance(res.get("sets"), list) else []
    return {"ran": True, "passed": res.get("passed") is True, "category": res.get("category"), "first_line": res.get("first_line"),
            "seeds": HELD_OUT_SEEDS, "shipped_seeds": orig_seeds, "files_identical": same, "shape_varied": False, "sets": len(sets),
            **kernelprobe.set_measures(sets)}


def orchard_process(adir: Path) -> dict:
    logs = sorted((adir / "log").glob("probe-*.jsonl"))
    tools, writes, own_checks, auto = {}, 0, 0, 0
    steps = 0
    for f in logs:
        for line in f.read_text().splitlines():
            r = json.loads(line)
            steps += 1
            for tc in (r.get("received") or {}).get("tool_calls") or []:
                name = tc["function"]["name"]
                tools[name] = tools.get(name, 0) + 1
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except ValueError:
                    args = {}
                if name == "write_file" and str(args.get("path", "")).endswith("kernel.py"):
                    writes += 1
                if name == "shell" and "kernel_check.py" in str(args.get("command", "")):
                    own_checks += 1
    auto = writes                                                     # orchard runs the shipped check after every write of kernel.py
    prompt = completion = 0
    for f in logs:
        for line in f.read_text().splitlines():
            u = json.loads(line).get("usage") or {}
            prompt += int(u.get("prompt_tokens") or 0)
            completion += int(u.get("completion_tokens") or 0)
    return {"turns_or_steps": steps, "tool_calls": tools, "kernel_writes": writes, "agent_ran_check": own_checks,
            "automatic_checks_after_writes": auto, "tokens": {"input": prompt, "output": completion}}


def opencode_process(out: Path, op: str) -> dict:
    meta = out / "oc-meta" / f"{op}-attempt-1"
    ev = []
    try:
        ev = [json.loads(l) for l in (meta / "events.jsonl").read_text().splitlines() if l.strip().startswith("{")]
    except OSError:
        pass
    tools, writes, own_checks, steps, tokens = {}, 0, 0, 0, {}
    for e in ev:
        p = e.get("part") or {}
        if e.get("type") == "step_start":
            steps += 1
        if e.get("type") == "step_finish":
            for k, v in (p.get("tokens") or {}).items():
                if isinstance(v, (int, float)):
                    tokens[k] = tokens.get(k, 0) + v
        if e.get("type") == "tool_use":
            name = p.get("tool")
            tools[name] = tools.get(name, 0) + 1
            inp = (p.get("state") or {}).get("input") or {}
            if name in ("write", "edit") and str(inp.get("path", inp.get("filePath", ""))).endswith("kernel.py"):
                writes += 1
            if name == "shell" and "kernel_check.py" in str(inp.get("command", "")):
                own_checks += 1
    return {"turns_or_steps": steps, "tool_calls": tools, "kernel_writes": writes, "agent_ran_check": own_checks,
            "automatic_checks_after_writes": 0, "tokens": tokens}


def server_row(rec: dict, process: dict) -> dict | None:
    """The model server's own counters for the attempt (change across the attempt, taken by the driver). None when the driver did not take them
    (the pilot). `requests_match` compares the server's request count with the harness's own count of model calls: equal means no other client."""
    d = rec.get("server_delta")
    if not isinstance(d, dict) or "error" in d:
        return None
    wall = rec["attempt"].get("agent_wall_s") if rec.get("attempt") else None
    prompt = d["prompt_tokens"]
    row = dict(d)
    row["cache_hit_share_arithmetic"] = (d["prompt_tokens_cache_hit"] / prompt) if prompt else None
    row["host_and_tool_s_arithmetic"] = (wall - d["prefill_s"] - d["decode_s"] - d["queue_s"]) if wall is not None else None
    row["requests_match"] = (int(d["requests"]) == int((process or {}).get("turns_or_steps") or -1))
    return row


def find_adir(out: Path) -> Path | None:
    hits = sorted(out.glob("*/attempt-1"))
    return hits[0] if hits else None


def margin(attempt: dict) -> dict:
    sets = attempt.get("sets") or []
    pcc = [s["pcc"] for s in sets if isinstance(s.get("pcc"), (int, float))]
    l2 = [s["rel_l2"] for s in sets if isinstance(s.get("rel_l2"), (int, float))]
    sl = [s["slope"] for s in sets if isinstance(s.get("slope"), (int, float))]
    names = [(s.get("name"), s.get("pcc")) for s in sets]
    tail = [s for s in sets if s.get("heavy_tail")]
    plain = [s for s in sets if not s.get("heavy_tail")]

    def worst(ss):
        v = [s["pcc"] for s in ss if isinstance(s.get("pcc"), (int, float))]
        return min(v) if v else None
    return {"pcc_min": min(pcc) if pcc else None, "rel_l2_max": max(l2) if l2 else None, "slope_min": min(sl) if sl else None,
            "slope_max": max(sl) if sl else None, "pcc_min_plain_sets": worst(plain), "pcc_min_outlier_sets": worst(tail), "sets": len(sets),
            "pcc_margin_to_bar": (min(pcc) - BARS["pcc"]) if pcc else None, "rel_l2_margin_to_bar": (BARS["rel_l2"] - max(l2)) if l2 else None,
            "slope_margin_to_band": (min(min(sl) - BARS["slope"][0], BARS["slope"][1] - max(sl)) if sl else None), "_names": names}


def near_miss(attempt: dict) -> str:
    cat = attempt.get("category")
    if cat in STRUCTURAL:
        return "structural"
    if cat == "numeric" or (attempt.get("sets") and not attempt.get("passed_sim")):
        m = margin(attempt)
        ok = ((m["pcc_min"] is None or m["pcc_min"] >= NEAR["pcc"]) and (m["rel_l2_max"] is None or m["rel_l2_max"] <= NEAR["rel_l2"])
              and (m["slope_min"] is None or m["slope_min"] >= NEAR["slope"][0]) and (m["slope_max"] is None or m["slope_max"] <= NEAR["slope"][1]))
        return "near miss" if ok else "numeric, far from the bar"
    return "other"


def main(argv: list) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--root", default=str(Path.home() / "orchard-runs" / "ocprobe"))
    ap.add_argument("--no-heldout", action="store_true")
    a = ap.parse_args(argv)
    tdir = Path(a.root) / a.phase
    recs = [json.loads(l) for l in (tdir / "timeline.jsonl").read_text().splitlines()]
    window = suite_windows()
    env = kernelprobe.default_env(argparse.Namespace())
    facts = kernelprobe.tt_lang_facts(env)
    scratch = Path(a.root) / a.phase / "quality-scratch"
    scratch.mkdir(exist_ok=True)
    out_rows = []
    for rec in recs:
        att = rec.get("attempt") or {}
        out = Path(rec["out"])
        adir = find_adir(out)
        harness = "opencode" if rec["harness"] == "O" else "orchard"
        row = {"harness": harness, "op": rec["op"], "n": rec["n"], "pair": rec["pair"], "seed": rec["seed"], "started_utc": rec["started_utc"],
               "ended_utc": rec["ended_utc"], "suite_overlap": overlaps(rec, window), "passed_sim": att.get("passed_sim"),
               "category": att.get("category"), "first_line": (att.get("first_line") or "")[:200], "status": att.get("status"),
               "agent_wall_s": att.get("agent_wall_s"), "turns": att.get("turns"), "status_detail": att.get("status_detail"),
               "margin": {k: v for k, v in margin(att).items() if k != "_names"},
               "closeness": None if att.get("passed_sim") else near_miss(att)}
        if adir is not None:
            row["process"] = orchard_process(adir) if harness == "orchard" else opencode_process(out, rec["op"])
            row["server"] = server_row(rec, row["process"])
            row["loadavg_start"], row["loadavg_end"] = rec.get("loadavg_start"), rec.get("loadavg_end")
            row["ended_on"] = "its own final answer" if att.get("status") == "done" else f"a limit or stop: {att.get('status')}"
            kpath = adir / "kernel.py"
            if att.get("passed_sim") and kpath.is_file():
                kernel = kpath.read_text()
                sheet = adir / "ttlang_cheat_sheet.md"
                entries = sheet_entries(kernelprobe.TEMPLATES / "ttlang_cheat_sheet.json")
                row["static"] = static_check(kernel, entries, reference_method_names(adir))
                row["structure"] = structure(kernel)
                row["cheat_sheet_in_attempt_dir"] = sheet.is_file()
                if not a.no_heldout:
                    row["held_out"] = held_out(adir, rec["op"], env, facts, scratch / f"{harness}-{rec['op']}-{rec['n']}")
        out_rows.append(row)
    with open(tdir / "quality.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    (tdir / "quality.md").write_text(render(out_rows, window), encoding="utf-8")
    print(f"wrote {tdir / 'quality.jsonl'} and {tdir / 'quality.md'}", flush=True)
    return 0


def fmt(x, nd=4):
    return "-" if x is None else (f"{x:.{nd}g}" if isinstance(x, float) else str(x))


def render(rows: list, window: tuple) -> str:
    L = ["Load windows (UTC): " + "; ".join(f"{n} {a} to {b or 'not ended when this was written'}" for n, a, b in window) +
         ". Attempts whose `load` column names a window overlapped it; wall-time differences are not read from them.", ""]
    L += ["### Per attempt", "", "| harness | op | n | result | class / closeness | agent wall s | load | turns or steps | kernel writes | agent ran check | auto checks | tool calls | ended on |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p = r.get("process") or {}
        res = "pass" if r["passed_sim"] else "fail"
        cls = "-" if r["passed_sim"] else f"{r['category']} / {r['closeness']}"
        L.append(f"| {r['harness']} | {r['op']} | {r['n']} | {res} | {cls} | {fmt(r['agent_wall_s'])} | {', '.join(r['suite_overlap']) or 'no'} | "
                 f"{fmt(p.get('turns_or_steps'))} / {fmt(r['turns'])} | {fmt(p.get('kernel_writes'))} | {fmt(p.get('agent_ran_check'))} | "
                 f"{fmt(p.get('automatic_checks_after_writes'))} | {json.dumps(p.get('tool_calls', {}))} | {r.get('ended_on', '-')} |")
    L += ["", "### Per op and harness", "", "| op | harness | attempts | passes | agent wall s of passes | all attempts' agent wall s | passes per agent hour (arithmetic) | passes within the other harness's longest pass time |",
          "|---|---|---|---|---|---|---|---|"]
    ops = sorted({r["op"] for r in rows}, key=lambda o: ["add", "center", "row_sum"].index(o) if o in ("add", "center", "row_sum") else 9)
    for op in ops:
        by = {h: [r for r in rows if r["op"] == op and r["harness"] == h] for h in ("opencode", "orchard")}
        for h in ("opencode", "orchard"):
            other = "orchard" if h == "opencode" else "opencode"
            mine = by[h]
            passes = [r for r in mine if r["passed_sim"]]
            total = sum(r["agent_wall_s"] or 0 for r in mine)
            op_pass_times = [r["agent_wall_s"] for r in by[other] if r["passed_sim"] and r["agent_wall_s"] is not None]
            T = max(op_pass_times) if op_pass_times else None
            within = "undefined" if T is None else sum(1 for r in passes if r["agent_wall_s"] is not None and r["agent_wall_s"] <= T)
            per_hour = fmt(len(passes) / total * 3600, 3) if total else "-"
            L.append(f"| {op} | {h} | {len(mine)} | {len(passes)} | {[fmt(r['agent_wall_s']) for r in passes]} | {fmt(total)} | {per_hour} | "
                     f"{within} (T = {fmt(T)}) |")
    L += ["", "### Tokens and server time per attempt (server counters are the change across the attempt; one client at a time)", "",
          "| harness | op | n | harness-side prompt tokens | completion tokens | server prompt tokens | from cache | computed | cache share (arithmetic) | generated | requests (server / harness) | replies cut at 4096 | prefill s | decode s | queue s | host and tools s (arithmetic) | agent wall s | load average at start |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p, sv = r.get("process") or {}, r.get("server")
        t = p.get("tokens") or {}
        if sv:
            L.append(f"| {r['harness']} | {r['op']} | {r['n']} | {fmt(t.get('input'))} | {fmt(t.get('output'))} | {fmt(sv['prompt_tokens'])} | {fmt(sv['prompt_tokens_cache_hit'])} | "
                     f"{fmt(sv['prompt_tokens_computed'])} | {fmt(sv['cache_hit_share_arithmetic'], 3)} | {fmt(sv['generation_tokens'])} | "
                     f"{fmt(sv['requests'])} / {fmt(p.get('turns_or_steps'))}{'' if sv['requests_match'] else ' (differ)'} | {fmt(sv['requests_hit_reply_cap'])} | {fmt(sv['prefill_s'])} | "
                     f"{fmt(sv['decode_s'])} | {fmt(sv['queue_s'])} | {fmt(sv['host_and_tool_s_arithmetic'])} | {fmt(r['agent_wall_s'])} | {' '.join(r.get('loadavg_start') or [])} |")
        else:
            L.append(f"| {r['harness']} | {r['op']} | {r['n']} | {fmt(t.get('input'))} | {fmt(t.get('output'))} | not taken | - | - | - | - | - | - | - | - | - | - | {fmt(r['agent_wall_s'])} | - |")
    L += ["", "### Margins of passing kernels (unedited check; bars: PCC at least 0.999, relative L2 at most 0.01, slope 0.98 to 1.02)", "",
          "| op | harness | passes | worst PCC | median PCC | worst outlier-set PCC | worst rel L2 | median rel L2 | slope range over passes |", "|---|---|---|---|---|---|---|---|---|"]
    for op in ops:
        for h in ("opencode", "orchard"):
            ps = [r for r in rows if r["op"] == op and r["harness"] == h and r["passed_sim"]]
            if not ps:
                L.append(f"| {op} | {h} | 0 | - | - | - | - | - | - |")
                continue
            pcc = [r["margin"]["pcc_min"] for r in ps if r["margin"]["pcc_min"] is not None]
            l2 = [r["margin"]["rel_l2_max"] for r in ps if r["margin"]["rel_l2_max"] is not None]
            tail = [r["margin"]["pcc_min_outlier_sets"] for r in ps if r["margin"]["pcc_min_outlier_sets"] is not None]
            sl = [(r["margin"]["slope_min"], r["margin"]["slope_max"]) for r in ps if r["margin"]["slope_min"] is not None]
            L.append(f"| {op} | {h} | {len(ps)} | {fmt(min(pcc) if pcc else None, 6)} | {fmt(statistics.median(pcc) if pcc else None, 6)} | "
                     f"{fmt(min(tail) if tail else None, 6)} | {fmt(max(l2) if l2 else None)} | {fmt(statistics.median(l2) if l2 else None)} | "
                     f"{fmt(min(s[0] for s in sl) if sl else None)} to {fmt(max(s[1] for s in sl) if sl else None)} |")
    L += ["", "### Failures: closeness", "", "| harness | op | n | class | closeness | first line |", "|---|---|---|---|---|---|"]
    for r in rows:
        if not r["passed_sim"]:
            L.append(f"| {r['harness']} | {r['op']} | {r['n']} | {r['category']} | {r['closeness']} | {r['first_line'][:110]} |")
    L += ["", "### Passing kernels: what a simulator pass does not show", "",
          "| harness | op | n | lines | follows the reference structure | calls with status `compiles on hardware` | `hardware signature only` | `simulator only` | not in the cheat sheet | keyword not in the hardware signature | held-out (seeds 101, 102, 103) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["passed_sim"] or "static" not in r:
            continue
        s, st, h = r["static"], r["structure"], r.get("held_out") or {}
        bs = s["by_status"]
        ho = "not run" if not h else ("pass" if h.get("passed") else f"fail ({h.get('category')})") if h.get("ran") else f"not possible: {h.get('reason')}"
        kw = "; ".join(f"{b['call']}({', '.join(b['keywords_not_in_signature'])})" for b in s["keyword_not_in_hardware_signature"]) or "none"
        L.append(f"| {r['harness']} | {r['op']} | {r['n']} | {st['lines_of_code']} | {'yes' if st['follows_reference_structure'] else 'no'} | "
                 f"{', '.join(bs.get('compiles on hardware', [])) or '-'} | {', '.join(bs.get('hardware signature only', [])) or '-'} | "
                 f"{', '.join(bs.get('simulator only', [])) or '-'} | {', '.join(s['unlisted']) or '-'} | {kw} | {ho} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
