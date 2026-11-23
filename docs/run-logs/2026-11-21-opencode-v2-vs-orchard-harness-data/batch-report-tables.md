#### Results by pair (O is OpenCode, R is orchard; `*` load window, `+` another client on the server)

| pair | O | R |
|---|---|---|
| add1 | PASS, 12 steps, 75 s, ended done | PASS, 9 turns, 51 s, ended done |
| add2 | PASS, 18 steps, 126 s, ended done | PASS, 13 turns, 107 s, ended done |
| center1 | fail tt_lang_error, 40 steps, 176 s, ended turns | fail tt_lang_error, 37 turns, 198 s, ended escalate |
| center2 | fail shape_mismatch, 40 steps, 298 s, ended turns | fail tt_lang_error, 39 turns, 524 s, ended escalate |
| center3 | fail tt_lang_error, 42 steps, 1184 s, ended turns | PASS, 35 turns, 561 s, ended done |
| center4 | fail tt_lang_error, 40 steps, 324 s, ended turns | fail tt_lang_error, 40 turns, 720 s, ended turns |
| center5 | fail tt_lang_error, 40 steps, 184 s, ended turns | fail tt_lang_error*, 40 turns, 372 s, ended turns |
| center6 | fail shape_mismatch*, 40 steps, 510 s, ended turns | fail tt_lang_error*, 11 turns, 143 s, ended escalate |
| row_sum1 | fail numeric_mismatch*, 44 steps, 1690 s, ended error | fail shape_mismatch*, 13 turns, 127 s, ended escalate |
| row_sum2 | fail no_kernel*, 40 steps, 145 s, ended turns | fail numeric_mismatch*, 38 turns, 416 s, ended pause |
| row_sum3 | fail numeric_mismatch*+, 43 steps, 1351 s, ended turns | PASS*+, 40 turns, 492 s, ended turns |
| row_sum4 | PASS, 40 steps, 657 s, ended turns | fail tt_lang_error*+, 34 turns, 986 s, ended error |
| row_sum5 | PASS, 28 steps, 223 s, ended done | fail numeric_mismatch, 40 turns, 853 s, ended turns |
| row_sum6 | PASS, 41 steps, 532 s, ended turns | fail tt_lang_error, 40 turns, 396 s, ended turns |

#### Counts per op and harness

| op | harness | attempts | simulator passes | passes the pair's compiler accepts (1.1.10.dev20261026+light) | passes tt-lang 1.1.6 accepts | failure classes | ended on |
|---|---|---|---|---|---|---|---|
| add | opencode | 2 | 2 | 2 | 2 | - | done 2 |
| add | orchard | 2 | 2 | 2 | 2 | - | done 2 |
| center | opencode | 6 | 0 | 0 | 0 | shape_mismatch 2, tt_lang_error 4 | turns 6 |
| center | orchard | 6 | 1 | 0 | 0 | tt_lang_error 5 | done 1, escalate 3, turns 2 |
| row_sum | opencode | 6 | 3 | 2 | 0 | no_kernel 1, numeric_mismatch 2 | done 1, error 1, turns 4 |
| row_sum | orchard | 6 | 1 | 0 | 0 | numeric_mismatch 2, shape_mismatch 1, tt_lang_error 2 | error 1, escalate 1, pause 1, turns 3 |
| all | opencode | 14 | 5 | 4 | 2 | | |
| all | orchard | 14 | 4 | 2 | 2 | | |

Fisher exact two-sided p for the pass counts (arithmetic on the counts above): all simulator passes 5 of 14 against 4 of 14, p = 1.00; passes the pair's compiler accepts 4 of 14 against 2 of 14, p = 0.65; `row_sum` simulator passes 3 of 6 against 1 of 6, p = 0.55; `center` 0 of 6 against 1 of 6, p = 1.00.

#### Time to a pass (agent wall seconds of passing attempts; `*` or `+` attempts are listed but not compared)

| op | harness | pass walls, s | all attempts' agent wall s | passes per agent hour (arithmetic, all attempts) | passes within the other harness's longest clean pass time |
|---|---|---|---|---|---|
| add | opencode | 75, 126 | 201 | 35.82 | 1 of 2 clean passes (T = 107 s) |
| add | orchard | 51, 107 | 158 | 45.63 | 2 of 2 clean passes (T = 126 s) |
| center | opencode | - | 2676 | 0.00 | 0 of 0 clean passes (T = 561 s) |
| center | orchard | 561 | 2518 | 1.43 | undefined (no clean pass by the other harness) |
| row_sum | opencode | 657, 223, 532 | 4598 | 2.35 | undefined (no clean pass by the other harness) |
| row_sum | orchard | 492* | 3270 | 1.10 | 0 of 0 clean passes (T = 657 s); a flagged pass is not counted |

#### Margins of the passing kernels (the unedited check; bars: PCC at least 0.999, relative L2 at most 0.01, slope 0.98 to 1.02)

| harness | op | n | PCC, worst set | PCC, outlier sets | relative L2, worst | slope range | held-out seeds 101 to 103 |
|---|---|---|---|---|---|---|---|
| opencode | add | 1 | 0.9999999 | 0.9999999 | 0 | 1 to 1 | pass |
| orchard | add | 1 | 0.9999999 | 0.9999999 | 0 | 1 to 1 | pass |
| orchard | add | 2 | 0.9999999 | 0.9999999 | 0 | 1 to 1 | pass |
| opencode | add | 2 | 0.9999999 | 0.9999999 | 0 | 1 to 1 | pass |
| orchard | center | 3 | 1 | 1 | 0 | 1 to 1 | pass |
| orchard | row_sum | 3 | 0.9999999 | 0.9999999 | 1.7e-07 | 1 to 1 | pass |
| opencode | row_sum | 4 | 0.9999999 | 0.9999999 | 0 | 1 to 1 | pass |
| opencode | row_sum | 5 | 0.9999999 | 0.9999999 | 0 | 1 to 1 | pass |
| opencode | row_sum | 6 | 0.9999999 | 0.9999999 | 0 | 1 to 1 | pass |

#### Failed attempts: how near the bar

| harness | op | n | class | near the bar | first line of the check's output |
|---|---|---|---|---|---|
| opencode | center | 1 | tt_lang_error | structural | error: TypeError: reduce_sum() missing 1 required positional argument: 'shape' |
| orchard | center | 1 | tt_lang_error | structural | error: TypeError: reduce_sum() missing 1 required positional argument: 'shape' |
| orchard | center | 2 | tt_lang_error | structural | Deadlock detected: all generators blocked |
| opencode | center | 2 | shape_mismatch | structural | error: ValueError: Tensor shape (32, 32) does not match Block shape (1, 8) (tile counts: 1 vs 8) |
| opencode | center | 3 | tt_lang_error | structural | error: ValidationError: 1 validation error for Block.__getitem__ |
| orchard | center | 4 | tt_lang_error | structural | error: ValueError: Shape mismatch in binary operation: left shape (1, 1) does not match right shape  |
| opencode | center | 4 | tt_lang_error | structural | error: ValueError: reduce shape (1, 8) does not match expected result shape (1, 1) (block shape (1,  |
| opencode | center | 5 | tt_lang_error | structural | error: TypeError: reduce_sum() missing 1 required positional argument: 'shape' |
| orchard | center | 5 | tt_lang_error | structural | error: ValueError: Shape mismatch in binary operation: left shape (1, 8) does not match right shape  |
| orchard | center | 6 | tt_lang_error | structural | error: IndexError: row slice 0:32 is outside the tensor, which has 2 tile(s) along it |
| opencode | center | 6 | shape_mismatch | structural | error: ValueError: Tensor shape (32, 32) does not match Block shape (2, 8) (tile counts: 1 vs 16) |
| opencode | row_sum | 1 | numeric_mismatch | numeric, far from the bar | set seed0: pcc 0.1757563203573227, finite True, bar 0.999 |
| orchard | row_sum | 1 | shape_mismatch | structural | error: ValueError: Tensor shape (64, 32) does not match Block shape (2, 8) (tile counts: 2 vs 16) |
| orchard | row_sum | 2 | numeric_mismatch | numeric, far from the bar | set seed0: pcc -0.0016424593050032854, finite True, bar 0.999 |
| opencode | row_sum | 2 | no_kernel | structural | kernel.py does not exist |
| opencode | row_sum | 3 | numeric_mismatch | numeric, far from the bar | set seed0: pcc 0.03266708925366402, finite True, bar 0.999 |
| orchard | row_sum | 4 | tt_lang_error | structural | error: RuntimeError: Cannot perform pop(): not a valid next dataflow step for this block. |
| orchard | row_sum | 5 | numeric_mismatch | numeric, far from the bar | set seed0: pcc 0.04401608183979988, finite True, bar 0.999 |
| orchard | row_sum | 6 | tt_lang_error | structural | error: ValidationError: 1 validation error for Block.__getitem__ |

#### Passing kernels against the chip compilers (1.1.10.dev20261026+light is the pair, the package the probe's hardware phase compiles with)

| harness | op | n | lines | 1.1.10.dev20261026+light | 1.1.6 | calls the old cheat sheet flags |
|---|---|---|---|---|---|---|
| opencode | add | 1 | 57 | compiles | compiles | none |
| orchard | add | 1 | 57 | compiles | compiles | none |
| orchard | add | 2 | 54 | compiles | compiles | none |
| opencode | add | 2 | 54 | compiles | compiles | none |
| orchard | center | 3 | 68 | refused: ttl.diagnostics.TTLangCompileError: error: <ast.Div object at 0x...> n | refused: ttl.diagnostics.TTLangCompileError: error: reduce_sum() got an unexpec | ttl.math.reduce_sum(shape), ttl.block.broadcast, ttl.block.fill |
| orchard | row_sum | 3 | 47 | refused: ttl.diagnostics.TTLangCompileError: error: constant type NoneType not  | refused: ttl.diagnostics.TTLangCompileError: error: constant type NoneType not  | ttl.math.reduce_sum(shape) |
| opencode | row_sum | 4 | 46 | compiles | refused: ttl.diagnostics.TTLangCompileError: error: reduce_sum() got an unexpec | ttl.math.reduce_sum(shape) |
| opencode | row_sum | 5 | 39 | compiles | refused: ttl.diagnostics.TTLangCompileError: error: reduce_sum() got an unexpec | ttl.math.reduce_sum(shape) |
| opencode | row_sum | 6 | 56 | refused: AttributeError: 'UnaryMinusOp' object has no attribute 'type' | refused: ttl.diagnostics.TTLangCompileError: error: Unknown function: ttl.grid_ | ttl.math.reduce_sum(shape) |

#### How the work got done (all attempts of an op and harness)

| op | harness | steps or turns (min-median-max) | writes of kernel.py | checks the agent ran itself | automatic checks after writes (orchard only) | attempts with no kernel | tool calls |
|---|---|---|---|---|---|---|---|
| add | opencode | 12-15-18 | 3 | 9 | 0 | 0 | glob 3, list 1, read 18, shell 10, write 3 |
| add | orchard | 9-11-13 | 6 | 6 | 6 | 0 | read_file 9, shell 6, write_file 6 |
| center | opencode | 40-40-42 | 25 | 20 | 0 | 0 | edit 2, glob 21, grep 7, list 4, read 105, shell 98, write 24 |
| center | orchard | 11-38-40 | 49 | 0 | 49 | 0 | read_file 93, run_shell_command 1, shell 68, write_file 55 |
| row_sum | opencode | 28-40-44 | 29 | 39 | 0 | 1 | glob 18, grep 10, list 1, read 105, shell 74, write 30 |
| row_sum | orchard | 13-39-40 | 47 | 3 | 47 | 0 | list_directory 1, read_file 139, shell 31, write_file 52 |

#### Tokens and server time (server counters are the change across each attempt; attempts with another client on the server are left out of the server columns)

| op | harness | prompt tokens, harness count | completion tokens, harness count | attempts with clean server counters | prompt tokens from the prefix cache | prefill s | decode s | generated tokens | decode tokens per s (arithmetic) | replies cut at 4096 (server count minus the probe's one speed-check request per attempt) |
|---|---|---|---|---|---|---|---|---|---|---|
| add | opencode | 1,269,006 | 5,204 | 2 of 2 | 90.2% of 1,269,058 | 30 | 162 | 5,804 | 35.7 | 0 |
| add | orchard | 198,120 | 5,824 | 2 of 2 | 79.6% of 198,172 | 10 | 143 | 6,424 | 45.1 | 0 |
| center | opencode | 11,301,921 | 69,935 | 6 of 6 | 94.1% of 11,302,077 | 182 | 2456 | 71,735 | 29.2 | 2 |
| center | orchard | 5,205,036 | 82,734 | 6 of 6 | 91.7% of 5,205,192 | 112 | 2350 | 84,534 | 36.0 | 3 |
| row_sum | opencode | 11,090,670 | 121,597 | 5 of 6 | 93.7% of 8,909,511 | 154 | 3064 | 86,633 | 28.3 | 10 |
| row_sum | orchard | 6,206,444 | 93,722 | 4 of 6 | 92.8% of 4,131,217 | 80 | 1688 | 55,335 | 32.8 | 4 |

Over the 25 attempts with clean server counters, decode seconds are 78% to 97% of agent wall time (median 92%) and prefill seconds are 3% to 17%.

