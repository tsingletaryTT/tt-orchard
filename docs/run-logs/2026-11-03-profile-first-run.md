# Profile first run: a matmul loop under `tt-orchard profile` (issue 113, step 1)

Date: 2026-11-02 (the file name carries the date of the PR). Issue 113. Code: `orchard/profile_summary.py`, command `tt-orchard profile`. Section 5.9ea of the README
describes the command. The files below were cut from this run and are test fixtures: `tests/fixtures/profile_ops_perf_results_sample.csv` (34 consecutive
rows from the middle of the run, signposts included) and `tests/fixtures/profile_perf_report_sample.csv` (the first 8 rows of `tt-perf-report --csv`).

## What ran

One Blackhole chip on a free board, taken with `gozer run --exact` through `tt-orchard profile`. The workload is a loop of three fake layers. Each
layer does `matmul(1024 x 1024 x 1024, bf16)`, `add`, `matmul`, `multiply`, with a `layer<N>/mixer` signpost before the first pair and a
`layer<N>/mlp` signpost before the second. One warm-up pass, then 20 timed passes, then one `add` after an `outside_layers` signpost: 253 device ops.
The shared virtual environment's Python and the `~/tt-metal` build were used. The command line was
`tt-orchard profile --tt-metal-home <tt-metal checkout> --exclude <the other agent's board> -- <venv python> <workload>.py`.

The profiled command took 59.5 s from the lease to the end of `tt-perf-report`. That figure includes the Tracy capture, the post-processing and the
chip reset on release, so it says nothing about the profiler's cost on the run itself (see the overhead section).

## What the issue listed as unknown

**Is the profiler enabled in the host build, and what does the exported per-op CSV hold on this version?**
Yes. The tt-metal build has `ENABLE_TRACY=ON` and the Tracy tools (`tracy-capture`, `tracy-csvexport`) under `build/tools/profiler/bin`.
`python -m tracy -r` produced `ops_perf_results_<timestamp>.csv`, `profile_log_device.csv` and a `.tracy` host trace under
`<out>/tracy/reports/<timestamp>/`. No rebuild was needed.

The ops CSV has 86 columns and one row per device op plus one row per signpost (253 op rows and 121 signpost rows here; a signpost row has
`OP TYPE` = `signpost` and the signpost text in `OP CODE`). In this run these columns were filled on every op row: op code, device id and arch,
attributes, math fidelity, core count, host start, end and duration, device FW and kernel durations, `OP TO OP LATENCY [ns]`, the five RISC kernel
durations, input and output shape, layout, datatype and memory, the kernel source and hash, program cache hit, kernel sizes, and the
performance-model columns (`PM IDEAL`, `PM COMPUTE`, `PM BANDWIDTH`, `PM REQ I BW`, `PM REQ O BW`, `PM FPU UTIL (%)`). These columns were empty on
every op row: the per-core min, max and average kernel durations, the ERISC duration, the compute CB wait and reserve times, the dispatch times,
the metal trace ids, and all of `NOC UTIL (%)`, `MULTICAST NOC UTIL (%)`, `DRAM BW UTIL (%)`, `DRAM BW UTIL PER CTRL (%)`, `ETH BW UTIL (%)` and
`NPE CONG IMPACT (%)`.

The empty NoC columns matter for the issue's list (per-bank DRAM use, network-row contention). `tools/tracy/__main__.py` sets
`TT_METAL_DEVICE_PROFILER_NOC_EVENTS` only when `python -m tracy` gets `--collect-noc-traces`, and `process_ops_logs.py` fills the per-controller
DRAM and congestion columns from those events. `tt-orchard profile --noc-traces` passes the flag. It was not run on hardware here (the task allowed
one profiled run), so the format of `DRAM BW UTIL PER CTRL (%)` is not known from a real cell. The reader accepts a list of percentages and, for a
cell that is not one, the summary says how many rows it could not read and does not claim that the counters were off.
Per-bank GB/s cannot come from the CSV in any case: the cell is a percent of peak and the file holds no peak.

`tt-perf-report --csv` adds per-op `Device Time` (microseconds), `Op-to-Op Gap`, `Cores`, `DRAM` (GB/s, derived from the performance model),
`DRAM %`, `FLOPs`, `FLOPs %`, `Bound`, and an `Advice` text for matmul rows. With signposts present it keeps only the data after the last signpost
unless it gets `--ignore-signposts`, which the command passes. It appends the operation's shape to the op code ("MatmulDeviceOperation 1024 x
1024 x 1024"). The first row of a report has no gap value.

**What overhead does the profiler add?**
The timed loop (240 ops, host dispatch plus one synchronize, signposts included) took 0.0111 s and 0.0128 s in two unprofiled runs in one lease, and
0.0526 s in the one profiled run: 4.1 to 4.7 times as long. That is two samples against one, on a loop that is limited by how fast the host enqueues
small ops, which is the case Tracy slows most. It is not a measure of the profiler's cost on a long model run where the device is the limit, and it
does not say whether device kernel times change: the unprofiled runs produce no device times to compare. A comparison of two configurations is
safe only when both were profiled. Run-to-run noise was not measured beyond the two unprofiled samples (they differ by 15 percent).

**How much of the memory report is usable as text?** Not tried. It needs `--device-memory-profiler`, which this run did not pass. Step 5 of the
issue stays open.

**Does a profile run under a lease need a rebuild or an environment setting?** No rebuild. One environment setting: `python -m tracy` finds its
tools under `$TT_METAL_HOME/build/tools/profiler/bin` and falls back to the current directory when the variable is unset. The first attempt had no
`TT_METAL_HOME`, took the lease, printed "Tracy tools were not found" and exited. The command now checks `TT_METAL_HOME` (or `--tt-metal-home`) for
both tools before it takes a lease, and passes it to the profiled command. Two other facts from the two attempts: the workload's script directory
must not hold a file named like a standard library module (a stray `inspect.py` next to the script shadowed the standard one and broke the import),
and the first attempt was cut short by gozer itself ("unreadable tt_serial", while another agent's board was being reset), before the profiler
started. That attempt left no lease behind and the second one worked.

## What the summary said (re-run with the final code on the same files)

The summary is 7,014 bytes with the top 5 ops, against 253 op rows and a 589 KB ops CSV.

| figure | value |
|---|---|
| ops, device time (sum of kernel durations) | 253 ops, 7,113.8 us |
| gaps up to 1 ms | 7,220.4 us, which is 28.5 us per op |
| gaps over 1 ms | 15 ops, 2,392,860 us (host idle: the first op after the device opened and the syncs between timed loops) |
| rule-of-thumb floor (3 to 5 us per op including its gap) | 759 to 1,265 us |
| group `mixer` | 120 ops, 3,484.0 us in ops, 3,125.9 us in gaps |
| group `moe_or_mlp` | 120 ops, 3,279.1 us in ops, 3,111.2 us in gaps |
| group `outside_layers` | 13 ops (12 warm-up, 1 final), 350.7 us in ops |
| group `layer_boundary` | none (the workload has no collectives) |
| largest op | the 1024-cube bf16 matmul: 43 us on 88 cores, 28 percent of DRAM peak by the model, 20 percent of FLOPs peak, advice "place input 0 in L1", "in0_block_w=1 is small", "use HiFi4 with BF16 activations if not FLOP-bound" |
| per-bank DRAM, network congestion | not measured (NoC event capture was off); both are named in `could_not_measure` |

The first version of the summary added every gap into `gap_us` and reported 2.4 seconds of gap for 253 ops. One multi-second stall at the start
of the run dominated it. Gaps over 1 ms are now counted separately (`long_gap_ops`, `long_gap_us`), so `gap_us` is the part the 3 to 5
microsecond rule applies to. The measured 28.5 us per op in gaps is host-bound dispatch with tracing off, about six to ten times the rule of thumb.
That comparison says this loop is dispatch-bound. It does not say anything about a traced decode.

A layer number that repeats (one signpost per pass here) is summed: layer 0's mixer shows 40 ops for 20 passes of 2 ops.

## What is left (issue 113 steps 2 to 6 stay open)

- Run once with `--noc-traces` and read a real `DRAM BW UTIL PER CTRL (%)` and `NPE CONG IMPACT (%)` cell.
- Profile a traced decode loop, where the gaps between ops are the question. The signposts for that must be emitted by the model code.
- The memory report (`--device-memory-profiler`) and the other counter groups (`--profiler-capture-perf-counters`).
- Steps 2 to 6: the kernel probe, full-port stages 5 and 6, configuration choices, memory failures and the web UI panel.
