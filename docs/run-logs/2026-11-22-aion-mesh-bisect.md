# Aion bundle: mesh-open bisect (2026-11-22, host rebooted 15:25 PT, load about 0 for every run below)

Each row was one run under its own `gozer run --chips 4` lease (no `--fresh`), `timeout -k 20 300` (or 120/150) on the exact child, no tt-smi.
Logs: `bisect/<name>.log`. Time is seconds from process start to the stated step.

| id | what changed (one variable against the row above it) | result |
| --- | --- | --- |
| T3 | system python (`~/.tenstorrent-venv`, ttnn from the source tree ~/tt-metal, host SFPI 7.71.0), 1-D fabric, 1x4 mesh, one add | opened 2.1 s, add ok, closed |
| T1 | bundle venv python (PyPI ttnn 0.80.0 wheel), no fabric, none of run.sh's variables | opened 3.6 s, add ok |
| T2 | same, with the 1-D fabric (as the server does) | opened 6.8 s, add ok |
| G1 | + empty kernel cache (TT_METAL_CACHE) | opened 7.4 s |
| G2 | + TT_METAL_HOME and TT_METAL_RUNTIME_ROOT = ttnn dir | opened 1.7 s |
| G3 | + LD_PRELOAD = _ttnncpp.so | opened 1.7 s |
| G4 | + TT_METAL_VISIBLE_DEVICES=0,1,2,3, MESH_DEVICE=1x4 | opened 1.7 s |
| ALL | the whole run.sh environment (run-with.sh), empty cache | opened 7.9 s, add ok |
| S | the server's own `startup()` (no uvicorn), run.sh environment | mesh opened; stuck in `port.load` for 300 s (killed). py-spy `--nonblocking`: MainThread in `ttnn.from_torch` (core.py:396) <- `_upload` aion_port.py:125 <- `load` line 310 = the embedding table |
| P1/P2 | `from_torch` of a random 200,029 x 2,048 bf16 ROW_MAJOR table, replicated, source build / wheel | 0.18 s / 0.14 s |
| R1 | the real embedding table from the checkpoint (a memory-mapped safetensors tensor), source build | upload 0.26 s |
| R2 | same, PyPI wheel | upload did not return in 120 s (killed) |
| R3 | same, wheel, tensor copied with `.clone()` first | upload 0.20 s |
| S2 | the server's `startup()` with the clone in the port's loader | ready, load 7.2 s, startup 9.2 s |

Cause: with the PyPI `ttnn` 0.80.0 wheel, `ttnn.from_torch` of a tensor that is backed by the memory-mapped checkpoint file does not return (100 percent CPU),
while the source build the bring-up run used handles it. The mesh open was never the problem; both earlier server "hangs" were this upload
(the log is silent between the fabric line and the ready line). Fix: the port's tensor reader returns `get_tensor(name).clone()`; this is the only difference between the
shipped port and the stage 3 port.

Not caused by: chips, firmware, the 1-D fabric, the empty kernel cache, any run.sh variable, the hand-extracted SFPI.

SFPI: `install.sh` of a ttnn 0.80.0 thin bundle stops on a host whose SFPI is older than 7.85.0 (this host: 7.71.0). The bundle does not carry SFPI.
On this host the runs above used SFPI 7.85.0 unpacked into the venv's `ttnn/runtime/sfpi` (hash checked against the value in the wheel's `sfpi-version`).
Whether the bundle works with host SFPI 7.71.0 alone was not tested. The card says a host SFPI of 7.85.0 or newer is required.
