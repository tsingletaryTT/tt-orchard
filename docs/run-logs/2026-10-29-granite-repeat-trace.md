# Granite-4.0-H-1B: remove the expanded input copies and trace the decode step (2026-10-29)

Operator request after `2026-10-29-granite-readout-on-pair.md` (PR 224): "let's figure out the ttnn.repeat copies". That run found the
coder's readout kernel slower than composed ttnn, and named two causes that it did not test: the untraced step is dispatch bound, and
the port makes five `ttnn.repeat` copies before each kernel call. This run tests both. Status: IN PROGRESS (first update, results
tables follow when the last lease session has run).

Same environment as the earlier run: the pair venv (tt-lang-light 1.1.10.dev20261026, tt-lang `fec38c6de`, tt-metal `0e9d200`, host
tt-metal v0.80.0), chips 2 and 3 only, one process at a time under `timeout`. The code, the scripts and the result files are in
`evidence/granite-repeat-trace/`.
