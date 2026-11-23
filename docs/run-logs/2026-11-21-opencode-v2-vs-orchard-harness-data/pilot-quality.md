Suite window (systemd, UTC): start 2026-11-21T19:25:35Z, end 2026-11-21T19:51:00Z. Attempts with `load` = yes overlapped it; wall-time differences are not read from them.

### Per attempt

| harness | op | n | result | class / closeness | agent wall s | load | turns or steps | kernel writes | agent ran check | auto checks | tool calls | ended on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opencode | add | 1 | pass | - | 103.8 | yes | 11 / 11 | 1 | 1 | 0 | {"read": 9, "glob": 3, "write": 1, "shell": 2} | its own final answer |
| orchard | add | 1 | pass | - | 71.86 | yes | 9 / 9 | 2 | 1 | 2 | {"read_file": 6, "write_file": 2, "shell": 1} | its own final answer |
| orchard | center | 1 | fail | tt_lang_error / structural | 307.1 | yes | 40 / 40 | 2 | 1 | 2 | {"read_file": 15, "shell": 23, "write_file": 2} | a limit or stop: turns |
| opencode | center | 1 | fail | tt_lang_error / structural | 238.8 | yes | 40 / 40 | 1 | 1 | 0 | {"read": 13, "glob": 4, "grep": 2, "write": 1, "shell": 22} | a limit or stop: turns |

### Per op and harness

| op | harness | attempts | passes | agent wall s of passes | all attempts' agent wall s | passes per agent hour (arithmetic) | passes within the other harness's longest pass time |
|---|---|---|---|---|---|---|---|
| add | opencode | 1 | 1 | ['103.8'] | 103.8 | 34.7 | 0 (T = 71.86) |
| add | orchard | 1 | 1 | ['71.86'] | 71.86 | 50.1 | 1 (T = 103.8) |
| center | opencode | 1 | 0 | [] | 238.8 | 0 | undefined (T = -) |
| center | orchard | 1 | 0 | [] | 307.1 | 0 | undefined (T = -) |

### Tokens and server time per attempt (server counters are the change across the attempt; one client at a time)

| harness | op | n | harness-side prompt tokens | completion tokens | server prompt tokens | from cache | computed | cache share (arithmetic) | generated | requests (server / harness) | replies cut at 4096 | prefill s | decode s | queue s | host and tools s (arithmetic) | agent wall s | load average at start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opencode | add | 1 | 539514 | 2112 | not taken | - | - | - | - | - | - | - | - | - | - | 103.8 | - |
| orchard | add | 1 | 79402 | 2110 | not taken | - | - | - | - | - | - | - | - | - | - | 71.86 | - |
| orchard | center | 1 | 1002839 | 5546 | not taken | - | - | - | - | - | - | - | - | - | - | 307.1 | - |
| opencode | center | 1 | 2225212 | 5008 | not taken | - | - | - | - | - | - | - | - | - | - | 238.8 | - |

### Margins of passing kernels (unedited check; bars: PCC at least 0.999, relative L2 at most 0.01, slope 0.98 to 1.02)

| op | harness | passes | worst PCC | median PCC | worst outlier-set PCC | worst rel L2 | median rel L2 | slope range over passes |
|---|---|---|---|---|---|---|---|---|
| add | opencode | 1 | 1 | 1 | 1 | 0 | 0 | 1 to 1 |
| add | orchard | 1 | 1 | 1 | 1 | 0 | 0 | 1 to 1 |
| center | opencode | 0 | - | - | - | - | - | - |
| center | orchard | 0 | - | - | - | - | - | - |

### Failures: closeness

| harness | op | n | class | closeness | first line |
|---|---|---|---|---|---|
| orchard | center | 1 | tt_lang_error | structural | error: TypeError: reduce_sum() missing 1 required positional argument: 'shape' |
| opencode | center | 1 | tt_lang_error | structural | error: TypeError: reduce_sum() missing 1 required positional argument: 'shape' |

### Passing kernels: what a simulator pass does not show

| harness | op | n | lines | follows the reference structure | calls with status `compiles on hardware` | `hardware signature only` | `simulator only` | not in the cheat sheet | keyword not in the hardware signature | held-out (seeds 101, 102, 103) |
|---|---|---|---|---|---|---|---|---|---|---|
| opencode | add | 1 | 57 | yes | DataflowBuffer.reserve, DataflowBuffer.wait, TensorBlock.store, ttl.compute, ttl.copy, ttl.datamovement, ttl.grid_size, ttl.make_dataflow_buffer_like, ttl.node, ttl.operation | - | - | - | none | pass |
| orchard | add | 1 | 54 | yes | DataflowBuffer.reserve, DataflowBuffer.wait, TensorBlock.store, ttl.compute, ttl.copy, ttl.datamovement, ttl.grid_size, ttl.make_dataflow_buffer_like, ttl.node, ttl.operation | - | - | - | none | pass |
