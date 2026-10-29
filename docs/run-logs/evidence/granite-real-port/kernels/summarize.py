"""Print one line per attempt of a kernel-probe output directory: op, attempt, phase, passed, category, first line."""
import json
import sys
from pathlib import Path

for root in sys.argv[1:]:
    for check in sorted(Path(root).glob("*/attempt-*/check.json"), key=lambda p: (p.parts[-3], int(p.parts[-2].split("-")[1]))):
        d = json.load(open(check))
        print(f"{check.parts[-4]}/{check.parts[-3]}/{check.parts[-2]}: passed={d['passed']} category={d['category']} :: {d['first_line'][:100]}")
