#!/usr/bin/env python3
"""Check the CadQuery in each saved reply (run with the cadquery venv): the JSON header parses, one python block exists, only cadquery and
math are imported, the script ends with `result = ...`, it executes (inside a child process with a time limit), and the result is a valid solid."""
import ast, json, re, subprocess, sys, tempfile, textwrap, os
res = json.load(open(sys.argv[1]))
rows = []
for r in res["decode"]:
    t = r["text"]; row = {"prompt_index": r["prompt_index"], "repeat": r["repeat"]}
    first = t.strip().splitlines()[0] if t.strip() else ""
    try: json.loads(first); row["header_json"] = True
    except Exception: row["header_json"] = False
    m = re.search(r"```python\n(.*?)```", t, re.S)
    row["code_block"] = bool(m)
    if m:
        code = m.group(1)
        try:
            tree = ast.parse(code); row["parses"] = True
            mods = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)} | {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            row["imports_ok"] = mods <= {"cadquery", "math"}
            row["ends_with_result"] = bool(re.search(r"^result\s*=", code, re.M))
        except SyntaxError as e:
            row["parses"] = False; row["error"] = str(e)[:120]
        if row.get("parses"):
            prog = code + "\nv = result.val() if hasattr(result,'val') else result\nprint('VALID' if v.isValid() else 'INVALID', round(v.Volume(),3))\n"
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(prog)
            try:
                p = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=60)
                row["executes"] = p.returncode == 0; row["output"] = p.stdout.strip()[-60:] if p.returncode == 0 else p.stderr.strip()[-160:]
                row["valid_solid"] = p.returncode == 0 and p.stdout.strip().startswith("VALID")
            except subprocess.TimeoutExpired:
                row["executes"] = False; row["output"] = "timeout"
            os.unlink(f.name)
    rows.append(row)
n = len(rows)
summary = {k: sum(1 for r in rows if r.get(k)) for k in ("header_json", "code_block", "parses", "imports_ok", "ends_with_result", "executes", "valid_solid")}
json.dump({"n": n, "summary": summary, "rows": rows}, open(sys.argv[2], "w"), indent=1)
print(n, summary)
