# `ttl.raw_element_read` and `ttl.raw_element_write` on hardware (2026-10-27)

The pinned tt-lang simulator (`fec38c6`) does not implement these two functions, which the tt-lang 0.20 specification
documents for data movement kernels. Two helper models wrote nine `linear_scan` kernels, seven of them with these
functions. The simulator stopped each at `'_TTLNamespace' object has no attribute 'raw_element_read'`, so none was run
anywhere. This note records what the hardware path does with them.

Setup: the probe's hardware harness (`kernel_harness.py --phase hw`, shipped spec and reference for `linear_scan`),
one Blackhole chip leased through gozer, host tt-metal tree v0.80.0, tt-lang pin `fec38c6`, shared TT environment.
Each kernel ran once. Nothing was reset by hand.

## The nine saved kernels

| kernels | result on hardware |
| --- | --- |
| 6 of the 7 that use `raw_element_*` | compile error: `list elements must be constants`, at a line such as `carry = [0.0] * TILE_SIZE` |
| 1 of the 7 (Sonnet run 3) | compile error: `raw_element_write() got an unexpected keyword argument 'value'` (the value is positional) |
| the 2 that do not use them (Sonnet runs 2 and 4) | ran; PCC 0.84 and 0.79 against the scan reference. They compute only the first step of the recurrence |

The errors are tt-lang compile errors. None of the nine is a working `linear_scan`.

## Probe kernels written for this question

The kernels are in `docs/run-logs/evidence/raw-element/`. They use one data movement thread that copies `a` and `b`
into blocks, waits on them, reads and writes elements into a scan block, one compute thread that passes that block on, and
a second data movement thread that writes the result out. The check used a reference for each (the copy of `b`, a constant
1.5, `b` shifted down one row), with the probe's inputs (three seeds, plain and heavy-tailed).

| kernel | compiled | numeric result |
| --- | --- | --- |
| copy `b` element by element (`raw_element_write(s, i, j, raw_element_read(b, i, j))`) | yes | PCC 1.0, max abs error 0.0, all 6 input sets |
| write the Python constant 1.5 | yes | max abs error 0.0, all 6 sets |
| shift `b` down one row (row 0 gets 0.0; other rows read row `i - 1`) | yes | PCC 1.0, max abs error 0.0, all 6 sets |
| `read(a) + read(b)` | no | `operand #0 must be signless-non-zero-bitwidth-integer-like, but got 'bf16'` |
| `read(a) * 2.0` | no | `'F32Type' object has no attribute 'element_type'` (an internal error) |
| `a if read(a) > read(b) else read(b)` | no | `conditional expressions are not supported in TT-Lang kernels` |
| `read(a)` from a block taken with `reserve()` | no | `block must be acquired from ttl.cb_wait, but traces to ttl.cb_reserve` |

## Answers

- The compiler accepts `ttl.raw_element_read` on this tree, on a block taken with `wait()` in a data movement thread. A
  block taken with `reserve()` is refused.
- A data movement kernel can copy values, write Python constants and permute elements (the row shift is exact). It cannot
  compute `a * h + b` from read values: `+` and `*` on read values fail to compile (the message says the operand must be an
  integer type, which suggests the compiler treats the operator as integer arithmetic; I did not check the compiler source), conditional expressions are refused, and a Python list of values is refused. The
  specification's rule that written values come from a read or a constant is enforced in practice by those refusals.
  The specification's table lists only `>` and `<` for read values; `>` inside a conditional expression did not work either.
- A `linear_scan` therefore cannot be written with these functions alone. A mixed design (data movement reorders elements,
  compute does the arithmetic on tiles, for example a shift-and-combine scan) was not tried.
- The operation needs exactly one compute and two data movement threads ("TTNN interop requires exactly 3 kernels").

## Not determined

- A first attempt put an empty compute thread (`pass`) beside the two data movement threads and copied from a block taken
  with `reserve()` in the writer. Its first run hung on the device for the 600 s limit and I stopped the second one by hand
  at about 14 minutes. The cause (the empty compute thread, or copying out of a block that was never pushed) was not
  separated. gozer reset the chip on release; no later run was affected.
- One run of each kernel, one chip, one tree. The `bf16` refusal may differ with another tt-metal or tt-lang version.
- Timing: a compile failure or a run took about 3 s each on a leased chip. The lease's release reset takes about 40 s.
