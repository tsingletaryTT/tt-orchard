"""After the hand-written card edit: copy the evidence it cites, scrub again, update package.json, run gate_package.
(package.finish rewrites the card from its template, so it is not run again after the edit.)"""
import os, socket
from pathlib import Path
from orchard import package
from orchard.stages import gate_package

RUN = Path(os.path.expanduser("~/orchard-runs/bytkim--qwen3.8-27b-pi"))
STAGE = RUN / "stages" / "7"
facts = package.read_run(RUN)
pkg = package._json(STAGE / "package.json", "package.json")
host = socket.gethostname()
for rec in pkg["profiles"]:
    out = facts.run_dir / rec["dir"]
    rec["evidence_files"] = package.copy_evidence(facts, out, (out / "README.md").read_text(encoding="utf-8"), hostname=host)
    rec["scrub"] = package.scrub_package(out, hostname=host, namespace=pkg["namespace"])
    print(rec["name"], rec["evidence_files"], rec["scrub"])
package._write_json(STAGE / "package.json", pkg)
print("gate_package", gate_package(STAGE, RUN))
