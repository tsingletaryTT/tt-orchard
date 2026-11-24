# Cheat-sheet recipes on a chip: all eight kernels fail in tt-metal's JIT build (2026-11-24, issue 800)

Steps 1 and 2 of issue 800: each of the four recipe skeletons (`ttlang_recipes.json`) and each of the four control skeletons (`ttlang_recipes_control.json`) went through `kernel_harness.py --phase hw`, one at a time, on one leased board, with the managed pair venv (tt-lang 1.1.10.dev20261026, light wheel) and its recorded exports. The harness came from a scratch copy of the templates with the recipes' specs and references added, the same overlay `scripts/verify_ttlang_recipes.py` builds for the simulator check. Steps 3 to 5 (the three single-kernel claims, the A/B softmax kernel and the byte comparison with the device-less check) were not reached.

## Result

All eight kernels, the four elementwise controls included, failed in the same way, about 1.3 seconds after the harness started the run phase. The kernel text compiled to C++ in tt-lang, and tt-metal's JIT build of the reader and writer kernels (`ncrisc` and `brisc`) then stopped on:

```
fatal error: tools/profiler/kernel_profiler.hpp: No such file or directory
    9 | #include "tools/profiler/kernel_profiler.hpp"
```

| Kernel | Kind | Outcome | Seconds |
| --- | --- | --- | --- |
| row-reduce-spread | recipe | JIT build failed, missing include | 2.7 |
| column-reduce-accumulate | recipe | same | 2.7 |
| elementwise-with-operands | recipe | same | 2.6 |
| running-total-down-rows | recipe | same | 2.6 |
| ctl-add-rows | control | same | 2.6 |
| ctl-sub-two-rows | control | same | 2.6 |
| ctl-mul-tiles | control | same | 2.6 |
| ctl-double | control | same | 2.6 |

The controls are elementwise kernels that the 2026-10-29 chip runs showed working. That they fail identically says the failure is in the host build of tt-metal against this compiler package, and no recipe is at fault. No recipe can be judged on a chip until this is resolved, so no `chip` entry was written to `ttlang_recipes.verified.json`.

## What was checked

- The generated sources include `tools/profiler/kernel_profiler.hpp`. The host tt-metal checkout is at commit `64bc494efdf` (2026-10-23). In that tree the header is at `tt_metal/hw/inc/api/debug/kernel_profiler.hpp` only; `git ls-tree` on that commit lists no other copy. The build directory's installed layout (`build_Release/libexec/tt-metalium/tt_metal/tools/profiler/kernel_profiler.hpp`) has the old path.
- The pin records the compiler's own build tree as tt-metal `0e9d200db976` and notes that the host runtime is newer (`orchard/ttlang_pin.py`, `TT_METAL_NOTE`). This run shows a cost of that gap: the compiler writes a header path the newer host tree does not have on its include path.
- One attempt to work around it: running the control kernel `ctl-double` with `TT_METAL_HOME` and `TT_METAL_RUNTIME_ROOT` set to the installed layout (`build_Release/libexec/tt-metalium`). The firmware paths changed to that tree and the same include error stayed.
- The exports file also puts `build/lib` (not `build_Release`) first on the library path, so the compiled libraries and the source tree may come from different builds. Not checked.

No change was made to the host tt-metal tree, the pair venv, the recipes or the harness.

## What this changes for issue 800

- The device-less check cannot see this class of failure. It matches the issue's remark that a kernel can compile in tt-lang and then fail in tt-metal's build.
- Next step for whoever schedules the chip run: make the pair's tt-metal match the compiler's build tree (`TT_METAL_REF`), or add the old include path to the host tree, then repeat steps 1 and 2 first with the controls. Each control kernel takes under three seconds, so the controls are a quick check that the stack works before a recipe is blamed.
- `scripts/verify_ttlang_recipes.py` has no `--chip` mode; this run used a separate driver. The issue's request for that mode stands.
