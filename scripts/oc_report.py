#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Markdown tables of the OpenCode v2 against orchard comparison, from the files the batch wrote.

    python3 scripts/oc_report.py [--root DIR]        (DIR holds batch/timeline.jsonl, batch/quality.jsonl, batch/hwcompile.json)

Prints the tables of the run log's Batch and Quality sections. Every figure comes from `quality.jsonl` (scripts/oc_quality.py), `hwcompile.json`
(scripts/hwcompile_probe_passes.py of PR 802, which runs the compile step of each chip package on the CPU with no device) and the timeline. A figure
labelled arithmetic is arithmetic on those numbers. Wall times of attempts that overlapped a load window or another client on the server are
marked and left out of the wall-time comparisons.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from math import comb
from pathlib import Path

OPS = ["add", "center", "row_sum"]
H = ["opencode", "orchard"]
PAIR_ORDER = None


def fisher(a, n1, b, n2):
    """Two-sided Fisher exact p for a of n1 against b of n2."""
    N, K = n1 + n2, a + b

    def p(x):
        return comb(n1, x) * comb(n2, K - x) / comb(N, K)
    p0 = p(a)
    return sum(p(x) for x in range(max(0, K - n2), min(n1, K) + 1) if p(x) <= p0 + 1e-12)


def load(root: Path):
    rows = [json.loads(l) for l in (root / "batch" / "quality.jsonl").read_text().splitlines()]
    comp = json.loads((root / "batch" / "hwcompile.json").read_text())
    versions = [t["version"] for t in comp["targets"]]
    crow = comp["rows"]
    # the probe was given the OpenCode directories first and the orchard ones second, 14 each, in sorted order
    for i, r in enumerate(crow):
        r["harness"] = "opencode" if i < 14 else "orchard"
    comp_by = {}
    for r in crow:
        n = int(r["run"].rsplit("-", 1)[1])
        for k in r["kernels"]:
            comp_by[(r["harness"], r["op"], n)] = {c["version"]: (c["category"], c["first_line"]) for c in k["compile"]}
    return rows, comp_by, versions


def clean(r):
    return not r["suite_overlap"] and not (r.get("server") or {}).get("other_client_suspected")


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "orchard-runs" / "ocprobe"))
    a = ap.parse_args(argv)
    root = Path(a.root)
    rows, comp_by, versions = load(root)
    pair_v, old_v = versions[0], versions[1]
    out = []
    by = {(r["harness"], r["op"], r["n"]): r for r in rows}
    out += ["#### Results by pair (O is OpenCode, R is orchard; `*` load window, `+` another client on the server)", "",
            "| pair | O | R |", "|---|---|---|"]

    def cell(r):
        flags = ("*" if r["suite_overlap"] else "") + ("+" if (r.get("server") or {}).get("other_client_suspected") else "")
        res = "PASS" if r["passed_sim"] else f"fail {r['category']}"
        return f"{res}{flags}, {r['process']['turns_or_steps']} {'steps' if r['harness'] == 'opencode' else 'turns'}, {r['agent_wall_s']:.0f} s, ended {r['status']}"
    for op in OPS:
        for n in sorted({r['n'] for r in rows if r['op'] == op}):
            out.append(f"| {op}{n} | {cell(by[('opencode', op, n)])} | {cell(by[('orchard', op, n)])} |")
    out += ["", "#### Counts per op and harness", "",
            f"| op | harness | attempts | simulator passes | passes the pair's compiler accepts ({pair_v}) | passes tt-lang {old_v} accepts | failure classes | ended on |",
            "|---|---|---|---|---|---|---|---|"]
    totals = {h: [0, 0, 0, 0] for h in H}
    for op in OPS:
        for h in H:
            rs = [r for r in rows if r["op"] == op and r["harness"] == h]
            ps = [r for r in rs if r["passed_sim"]]
            ok_pair = sum(1 for r in ps if comp_by.get((h, op, r["n"]), {}).get(pair_v, ("",))[0] == "compiled")
            ok_old = sum(1 for r in ps if comp_by.get((h, op, r["n"]), {}).get(old_v, ("",))[0] == "compiled")
            fails, ended = {}, {}
            for r in rs:
                if not r["passed_sim"]:
                    fails[r["category"]] = fails.get(r["category"], 0) + 1
                ended[r["status"]] = ended.get(r["status"], 0) + 1
            out.append(f"| {op} | {h} | {len(rs)} | {len(ps)} | {ok_pair} | {ok_old} | {', '.join(f'{k} {v}' for k, v in sorted(fails.items())) or '-'} | "
                       f"{', '.join(f'{k} {v}' for k, v in sorted(ended.items()))} |")
            for i, v in enumerate((len(rs), len(ps), ok_pair, ok_old)):
                totals[h][i] += v
    out.append(f"| all | opencode | {totals['opencode'][0]} | {totals['opencode'][1]} | {totals['opencode'][2]} | {totals['opencode'][3]} | | |")
    out.append(f"| all | orchard | {totals['orchard'][0]} | {totals['orchard'][1]} | {totals['orchard'][2]} | {totals['orchard'][3]} | | |")
    o, r_ = totals["opencode"], totals["orchard"]
    out += ["", "Fisher exact two-sided p for the pass counts (arithmetic on the counts above): "
            f"all simulator passes {o[1]} of {o[0]} against {r_[1]} of {r_[0]}, p = {fisher(o[1], o[0], r_[1], r_[0]):.2f}; "
            f"passes the pair's compiler accepts {o[2]} of {o[0]} against {r_[2]} of {r_[0]}, p = {fisher(o[2], o[0], r_[2], r_[0]):.2f}; "
            f"`row_sum` simulator passes 3 of 6 against 1 of 6, p = {fisher(3, 6, 1, 6):.2f}; `center` 0 of 6 against 1 of 6, p = {fisher(0, 6, 1, 6):.2f}.", ""]

    out += ["#### Time to a pass (agent wall seconds of passing attempts; `*` or `+` attempts are listed but not compared)", "",
            "| op | harness | pass walls, s | all attempts' agent wall s | passes per agent hour (arithmetic, all attempts) | passes within the other harness's longest clean pass time |",
            "|---|---|---|---|---|---|"]
    for op in OPS:
        for h in H:
            other = "orchard" if h == "opencode" else "opencode"
            rs = [r for r in rows if r["op"] == op and r["harness"] == h]
            ps = [r for r in rs if r["passed_sim"]]
            walls = ", ".join(f"{r['agent_wall_s']:.0f}{'' if clean(r) else ('*' if r['suite_overlap'] else '+')}" for r in ps) or "-"
            total = sum(r["agent_wall_s"] for r in rs)
            ot = [r["agent_wall_s"] for r in rows if r["op"] == op and r["harness"] == other and r["passed_sim"] and clean(r)]
            T = max(ot) if ot else None
            within = ("undefined (no clean pass by the other harness)" if T is None else
                      f"{sum(1 for r in ps if clean(r) and r['agent_wall_s'] <= T)} of {sum(1 for r in ps if clean(r))} clean passes (T = {T:.0f} s)"
                      + ("" if all(clean(r) for r in ps) else "; a flagged pass is not counted"))
            out.append(f"| {op} | {h} | {walls} | {total:.0f} | {fmt(len(ps) / total * 3600, 2)} | {within} |")

    out += ["", "#### Margins of the passing kernels (the unedited check; bars: PCC at least 0.999, relative L2 at most 0.01, slope 0.98 to 1.02)", "",
            "| harness | op | n | PCC, worst set | PCC, outlier sets | relative L2, worst | slope range | held-out seeds 101 to 103 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["passed_sim"]:
            m, ho = r["margin"], r.get("held_out") or {}
            out.append(f"| {r['harness']} | {r['op']} | {r['n']} | {m['pcc_min']:.7g} | {m['pcc_min_outlier_sets']:.7g} | {m['rel_l2_max']:.3g} | "
                       f"{m['slope_min']:.4g} to {m['slope_max']:.4g} | {'pass' if ho.get('passed') else 'not run or fail'} |")

    out += ["", "#### Failed attempts: how near the bar", "", "| harness | op | n | class | near the bar | first line of the check's output |", "|---|---|---|---|---|---|"]
    for r in rows:
        if not r["passed_sim"]:
            out.append(f"| {r['harness']} | {r['op']} | {r['n']} | {r['category']} | {r['closeness']} | {r['first_line'][:100].replace('|', '/')} |")

    out += ["", f"#### Passing kernels against the chip compilers ({pair_v} is the pair, the package the probe's hardware phase compiles with)", "",
            f"| harness | op | n | lines | {pair_v} | {old_v} | calls the old cheat sheet flags |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["passed_sim"]:
            c = comp_by.get((r["harness"], r["op"], r["n"]), {})
            f = lambda v: ("compiles" if c.get(v, ("",))[0] == "compiled" else "refused: " + c.get(v, ("", "-"))[1][:70].replace("|", "/"))
            s = r["static"]
            flags = [x["call"] + "(" + ", ".join(x["keywords_not_in_signature"]) + ")" for x in s["keyword_not_in_hardware_signature"]] + s["unlisted"]
            out.append(f"| {r['harness']} | {r['op']} | {r['n']} | {r['structure']['lines_of_code']} | {f(pair_v)} | {f(old_v)} | {', '.join(flags) or 'none'} |")

    out += ["", "#### How the work got done (all attempts of an op and harness)", "",
            "| op | harness | steps or turns (min-median-max) | writes of kernel.py | checks the agent ran itself | automatic checks after writes (orchard only) | attempts with no kernel | tool calls |", "|---|---|---|---|---|---|---|---|"]
    for op in OPS:
        for h in H:
            rs = [r for r in rows if r["op"] == op and r["harness"] == h]
            steps = [r["process"]["turns_or_steps"] for r in rs]
            tc = {}
            for r in rs:
                for k, v in (r["process"]["tool_calls"] or {}).items():
                    tc[k] = tc.get(k, 0) + v
            out.append(f"| {op} | {h} | {min(steps)}-{int(st.median(steps))}-{max(steps)} | {sum(r['process']['kernel_writes'] for r in rs)} | "
                       f"{sum(r['process']['agent_ran_check'] for r in rs)} | {sum(r['process']['automatic_checks_after_writes'] for r in rs)} | "
                       f"{sum(1 for r in rs if r['category'] == 'no_kernel')} | {', '.join(f'{k} {v}' for k, v in sorted(tc.items()))} |")

    out += ["", "#### Tokens and server time (server counters are the change across each attempt; attempts with another client on the server are left out of the server columns)", "",
            "| op | harness | prompt tokens, harness count | completion tokens, harness count | attempts with clean server counters | prompt tokens from the prefix cache | prefill s | decode s | generated tokens | decode tokens per s (arithmetic) | replies cut at 4096 (server count minus the probe's one speed-check request per attempt) |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for op in OPS:
        for h in H:
            rs = [r for r in rows if r["op"] == op and r["harness"] == h]
            good = [r for r in rs if r.get("server") and not r["server"]["other_client_suspected"]]
            sv = lambda k: sum(r["server"][k] for r in good)
            pt = sv("prompt_tokens")
            out.append(f"| {op} | {h} | {sum(r['process']['tokens']['input'] for r in rs):,} | {sum(r['process']['tokens']['output'] for r in rs):,} | {len(good)} of {len(rs)} | "
                       f"{sv('prompt_tokens_cache_hit') / pt * 100:.1f}% of {int(pt):,} | {sv('prefill_s'):.0f} | {sv('decode_s'):.0f} | {int(sv('generation_tokens')):,} | "
                       f"{sv('generation_tokens') / sv('decode_s'):.1f} | {int(sv('requests_hit_reply_cap')) - len(good)} |")
    allgood = [r for r in rows if r.get("server") and not r["server"]["other_client_suspected"]]
    shares = [r["server"]["decode_s"] / r["agent_wall_s"] for r in allgood]
    out += ["", f"Over the {len(allgood)} attempts with clean server counters, decode seconds are {min(shares) * 100:.0f}% to {max(shares) * 100:.0f}% of agent wall time "
            f"(median {st.median(shares) * 100:.0f}%) and prefill seconds are {min(r['server']['prefill_s'] / r['agent_wall_s'] for r in allgood) * 100:.0f}% to "
            f"{max(r['server']['prefill_s'] / r['agent_wall_s'] for r in allgood) * 100:.0f}%.", ""]
    print("\n".join(out))


if __name__ == "__main__":
    main()
