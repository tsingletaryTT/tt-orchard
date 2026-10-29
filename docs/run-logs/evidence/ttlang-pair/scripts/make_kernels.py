"""Collect the kernels for the pair check into k/ (one file each) and write k/manifest.json."""
import json, re, shutil
from pathlib import Path
R = Path.home() / "orchard-runs"
K = R / "ttlang-pair" / "k"
G = R / "ibm-granite--granite-4.0-h-1b" / "kernels"
SRC = Path.home() / "orchard-src" / "tt-lang"
LIB = Path.home() / "code/tt-orchard/orchard/skills/kernel-loop-templates/library"
man = []
# smoke: the kernel kernelprobe.smoke_test runs (examples/eltwise_add.py in the pinned checkout)
shutil.copy(SRC / "examples" / "eltwise_add.py", K / "smoke_eltwise_add.py")
man.append(dict(name="smoke_eltwise_add", file="smoke_eltwise_add.py", fn="eltwise_add", kind="add", shapes=[[256, 256]]))
# the simulator-passing ssd_readout kernels
for run, att in [("readout-r1", n) for n in (6, 7, 9, 10)] + [("readout-hw1", n) for n in (1, 2, 3)] + [("readout-hw2", n) for n in (1, 2, 3, 4)]:
    name = f"readout_{run[8:]}_a{att}"
    shutil.copy(G / run / "ssd_readout" / f"attempt-{att}" / "kernel.py", K / f"{name}.py")
    man.append(dict(name=name, file=f"{name}.py", fn="ssd_readout", kind="readout", shapes=[[64, 128], [3072, 128]]))
# library rmsnorm
t = (LIB / "rmsnorm.py").read_text().replace("OPERATION", "rmsnorm")
(K / "library_rmsnorm.py").write_text(t)
man.append(dict(name="library_rmsnorm", file="library_rmsnorm.py", fn="rmsnorm", kind="rmsnorm", shapes=[[64, 128], [3072, 128]]))
# grid_size placement variants, built from the smoke kernel
base = (SRC / "examples" / "eltwise_add.py").read_text()
base = base[: base.index("def main()")]
CALL = "grid_cols, grid_rows = ttl.grid_size(dims=2)"
ROWS = "    rows_per_node = -(-row_tiles // grid_rows)\n    cols_per_node = -(-col_tiles // grid_cols)\n"
def shadow(src, threads):
    # re-call grid_size inside the named threads (right after the node() line)
    for th in threads:
        pat = re.compile(r"(    def %s\(\):\n)" % th)
        src = pat.sub(r"\1        grid_cols, grid_rows = ttl.grid_size(dims=2)\n        rows_per_node = -(-row_tiles // grid_rows)\n        cols_per_node = -(-col_tiles // grid_cols)\n", src)
    return src
assert CALL in base
variants = {
    "gs_compute": shadow(base, ["compute"]),
    "gs_datamovement": shadow(base, ["read", "write"]),
}
# all threads, none in the operation body: remove the body call and the two lines that use it
allt = base.replace("    " + CALL + "\n", "").replace(ROWS, "")
variants["gs_threads_only"] = shadow(allt, ["compute", "read", "write"])
for n, s in variants.items():
    s = s.replace("def eltwise_add(", f"def {n}(")
    (K / f"{n}.py").write_text(s)
    man.append(dict(name=n, file=f"{n}.py", fn=n, kind="add", shapes=[[256, 256]]))
(K / "manifest.json").write_text(json.dumps(man, indent=1))
print(len(man), "kernels")
