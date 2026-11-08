#!/usr/bin/env python3
"""Check the CadQuery in each saved reply. Run it with an interpreter that has cadquery:  check_cq.py REPLIES.json OUT.json

Static checks (no code is run for these), per reply:
  header_json       the first line parses as JSON
  one_python_block  the reply has exactly one ```python block
  parses            the block parses
  imports_ok        every alias of every import names cadquery or math (`import math, os` fails)
  ends_with_result  the last top-level statement assigns `result`
A reply is executed only when all four of the last checks hold. It runs in cq_sandbox.py (scrubbed environment, resource limits, a timeout,
Landlock file and TCP limits when the kernel has them; the layers that applied are recorded). The generated code's own output is never read
for the verdict. After it runs, a trusted epilogue exports `result` to a STEP file; this script loads that file and checks that it holds at
least one CadQuery solid, every solid valid and with positive volume (valid_solid).
"""
import ast
import json
import os
import re
import sys
import tempfile

ALLOWED_MODULES = {"cadquery", "math"}
EPILOGUE = '''
import cadquery as _cq
_r = result
if not isinstance(_r, (_cq.Workplane, _cq.Shape)):
    raise TypeError("result is neither a Workplane nor a Shape: " + type(_r).__name__)
_cq.exporters.export(_r, "result.step")
'''


def static_checks(text: str) -> dict:
    row = {}
    lines = text.strip().splitlines()
    first = lines[0] if lines else ""
    try:
        json.loads(first)
        row["header_json"] = True
    except ValueError:
        row["header_json"] = False
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    row["one_python_block"] = len(blocks) == 1
    row["python_blocks"] = len(blocks)
    if len(blocks) != 1:
        return row
    code = blocks[0]
    row["code"] = code
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        row["parses"] = False
        row["error"] = str(exc)[:120]
        return row
    row["parses"] = True
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules.append("." if node.level or not node.module else node.module.split(".")[0])
    row["imports_ok"] = set(modules) <= ALLOWED_MODULES
    last = tree.body[-1] if tree.body else None
    row["ends_with_result"] = isinstance(last, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "result" for t in last.targets)
    return row


def verify_step(step_bytes: bytes) -> dict:
    """Load the exported STEP and judge it. Runs outside the sandbox and never runs generated code."""
    import cadquery as cq
    with tempfile.TemporaryDirectory(prefix="cqverify-") as d:
        path = os.path.join(d, "result.step")
        with open(path, "wb") as f:
            f.write(step_bytes)
        try:
            solids = cq.importers.importStep(path).solids().vals()
        except Exception as exc:  # noqa: BLE001 - a file that does not load is a failed verdict
            return {"valid_solid": False, "verdict": f"STEP did not load: {type(exc).__name__}"}
    if not solids:
        return {"valid_solid": False, "verdict": "no solid in the exported result"}
    vols = [s.Volume() for s in solids]
    good = all(isinstance(s, cq.Solid) and s.isValid() and v > 0 for s, v in zip(solids, vols))
    return {"valid_solid": bool(good), "solids": len(solids), "volume": round(sum(vols), 3), "verdict": "ok" if good else "a solid is invalid or has no volume"}


def check_reply(text: str, run_code=None) -> dict:
    row = static_checks(text)
    code = row.pop("code", None)
    eligible = all(row.get(k) for k in ("one_python_block", "parses", "imports_ok", "ends_with_result"))
    row["executed"] = False
    if not eligible:
        row["not_executed_because"] = [k for k in ("one_python_block", "parses", "imports_ok", "ends_with_result") if not row.get(k)]
        row["executes"] = row["valid_solid"] = False
        return row
    if run_code is None:
        import cq_sandbox
        run_code = cq_sandbox.run_code
    res = run_code(code, EPILOGUE)
    row["executed"] = True
    row["sandbox"] = res["layers"]
    row["executes"] = res["returncode"] == 0 and not res["timed_out"] and bool(res["step_bytes"])
    if res["timed_out"]:
        row["output"] = "timeout"
    elif res["returncode"] != 0:
        row["output"] = res["stderr"][-160:]
    if row["executes"]:
        row.update(verify_step(next(iter(res["step_bytes"].values()))))
    else:
        row["valid_solid"] = False
    return row


def main(argv) -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    res = json.load(open(argv[1]))
    rows = []
    for r in res["decode"]:
        row = {"prompt_index": r["prompt_index"], "repeat": r["repeat"]}
        row.update(check_reply(r["text"]))
        rows.append(row)
    keys = ("header_json", "one_python_block", "parses", "imports_ok", "ends_with_result", "executes", "valid_solid")
    summary = {k: sum(1 for r in rows if r.get(k)) for k in keys}
    layers = next((r["sandbox"] for r in rows if r.get("sandbox")), {})
    out = {"n": len(rows), "distinct_prompts": len({r["prompt_index"] for r in rows}), "summary": summary, "executed": sum(1 for r in rows if r["executed"]),
           "sandbox": layers, "network": ("TCP connect and bind refused by Landlock; UDP and Unix sockets not covered" if layers.get("tcp") else "not blocked"), "rows": rows}
    json.dump(out, open(argv[2], "w"), indent=1)
    print(out["n"], out["distinct_prompts"], summary, out["sandbox"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
